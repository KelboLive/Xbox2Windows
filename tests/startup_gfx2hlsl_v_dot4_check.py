"""VOP3P 4-byte integer dot products (v_dot4_i32_i8, v_dot4_u32_u8) in tools/startup/gfx2hlsl.py.

  (no arguments)                       VOP3P decode, on-demand helpers, value evaluation (wrapping and clamp),
                                       EXEC masking, refusal and mutation controls, pure Python
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile the translated pair with FXC ps_5_1 when given and with DXC ps_6_0,
                                       draw one pixel per four input triples and compare the four result words

RDNA 2 ISA, VOP3P (0x16 / 0x17):
  V_DOT4_I32_I8 vD, S0, S1, S2 [clamp]: D = sum_{k=0..3} int8(S0_k)*int8(S1_k) + int32(S2) (sat int32 if clamp)
  V_DOT4_U32_U8 vD, S0, S1, S2 [clamp]: D = sum_{k=0..3} uint8(S0_k)*uint8(S1_k) + uint32(S2) (sat uint32 if clamp)

The pixel fixture decodes synthetic gfx1013 VOP3P bytes via G.decode() ONLY (no game data). It reads three
flat parameters (v0..v3, v4..v7, v8..v11), computes v_dot4_* across all four lanes into v12..v15, and exports mrt0.
"""
import struct
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
from startup_flat_param_check import compile_hlsl, draw, root_blob  # noqa: E402
from startup_gfx2hlsl_bit_alu_check import (  # noqa: E402
    MASK,
    POST_WORDS,
    PRE_WORDS,
    Probe,
    check,
    vs_words,
    words_text,
)

OPCODES = {
    'v_dot4_i32_i8': 0x16,
    'v_dot4_u32_u8': 0x17,
}


def vop3p(op, vdst, src0, src1, src2, *, clamp=0, op_sel=0, op_sel_hi=0b111, neg_lo=0, neg_hi=0, lit=None):
    """Encode one synthetic gfx1013 VOP3P 64-bit (or 96-bit with literal) instruction."""
    w0 = (0xcc000000
          | (op << 16)
          | (clamp << 15)
          | (((op_sel_hi >> 2) & 1) << 14)
          | ((op_sel & 0x7) << 11)
          | ((neg_hi & 0x7) << 8)
          | (vdst & 0xff))
    w1 = (((neg_lo & 0x7) << 29)
          | ((op_sel_hi & 0x3) << 27)
          | ((src2 & 0x1ff) << 18)
          | ((src1 & 0x1ff) << 9)
          | (src0 & 0x1ff))
    out = struct.pack('<2I', w0, w1)
    if lit is not None:
        out += struct.pack('<I', lit & MASK)
    return out


def ps_words(op, clamp=0):
    code = OPCODES[op]
    body = b''.join(
        vop3p(code, 12 + lane, 256 + lane, 256 + 4 + lane, 256 + 8 + lane, clamp=clamp)
        for lane in range(4)
    )
    return PRE_WORDS + body + POST_WORDS


def translate(op, clamp=0):
    zero = [(0, 0, 0)] * 4
    pipe = G.Pipeline(vs_words(zero), ps_words(op, clamp), root_blob())
    return pipe.hlsl_vs(), pipe.hlsl_ps()


def sext8(b):
    b &= 0xff
    return b - 256 if b >= 128 else b


def sext32(v):
    v &= MASK
    return v - (1 << 32) if v >= (1 << 31) else v


def ref_dot4_u32_u8(a, b, c, clamp=False):
    s = sum(((a >> (8 * k)) & 0xff) * ((b >> (8 * k)) & 0xff) for k in range(4))
    total = (c & MASK) + s
    if clamp and total > MASK:
        return MASK
    return total & MASK


def ref_dot4_i32_i8(a, b, c, clamp=False):
    s = sum(sext8(a >> (8 * k)) * sext8(b >> (8 * k)) for k in range(4))
    total = sext32(c) + s
    if clamp:
        if total > 0x7fffffff:
            return 0x7fffffff
        if total < -0x80000000:
            return 0x80000000
    return total & MASK


def eval_dot4_u32_u8_hlsl(a, b, c, sat):
    a, b, c = a & MASK, b & MASK, c & MASK
    s = (((a & 0xff) * (b & 0xff)
          + ((a >> 8) & 0xff) * ((b >> 8) & 0xff)
          + ((a >> 16) & 0xff) * ((b >> 16) & 0xff)
          + ((a >> 24) & 0xff) * ((b >> 24) & 0xff)) & MASK)
    r = (c + s) & MASK
    return 0xffffffff if (sat and r < c) else r


def eval_dot4_i32_i8_hlsl(a, b, c, sat):
    a, b, c = a & MASK, b & MASK, c & MASK
    s = (sext8(a) * sext8(b)
         + sext8(a >> 8) * sext8(b >> 8)
         + sext8(a >> 16) * sext8(b >> 16)
         + sext8(a >> 24) * sext8(b >> 24))
    r = (c + (s & MASK)) & MASK
    if sat and sext32(c) > 0 and s > 0 and sext32(r) < 0:
        return 0x7fffffff
    if sat and sext32(c) < 0 and s < 0 and sext32(r) >= 0:
        return 0x80000000
    return r


def stage_refused(stage, op, line, needle=''):
    try:
        stage.one(op, line)
    except G.Unsupported as exc:
        return (not needle or needle in str(exc)) and not stage.lines
    return False


WORDS_A = (0x00000000, 0xffffffff, 0x80808080, 0x7f7f7f7f, 0x01020304, 0xfefdfcfb, 0x807f01ff)
WORDS_B = (0x00000000, 0xffffffff, 0x80808080, 0x7f7f7f7f, 0x05060708, 0x01ff807f)
ACCUMS = (0x00000000, 0x00000001, 0x7fffffff, 0x7fff0000, 0x80000000, 0x8000ffff, 0xffff0000, 0xffffffff)


def run_controls():
    # 1. Decode verification from synthetic gfx1013 VOP3P bytes via G.decode().
    check('decode v_dot4_i32_i8 plain',
          G.decode(vop3p(0x16, 12, 256, 260, 264))[0][3] == 'v_dot4_i32_i8 v12, v0, v4, v8')
    check('decode v_dot4_i32_i8 clamp',
          G.decode(vop3p(0x16, 12, 256, 260, 264, clamp=1))[0][3] == 'v_dot4_i32_i8 v12, v0, v4, v8 clamp')
    check('decode v_dot4_u32_u8 plain',
          G.decode(vop3p(0x17, 12, 256, 260, 264))[0][3] == 'v_dot4_u32_u8 v12, v0, v4, v8')
    check('decode v_dot4_u32_u8 clamp',
          G.decode(vop3p(0x17, 12, 256, 260, 264, clamp=1))[0][3] == 'v_dot4_u32_u8 v12, v0, v4, v8 clamp')

    # 2. Exhaustive grid for both wrapping and saturating modes.
    u_ok = True
    i_ok = True
    for a in WORDS_A:
        for b in WORDS_B:
            for c in ACCUMS:
                for sat in (False, True):
                    if eval_dot4_u32_u8_hlsl(a, b, c, sat) != ref_dot4_u32_u8(a, b, c, sat):
                        u_ok = False
                    if eval_dot4_i32_i8_hlsl(a, b, c, sat) != ref_dot4_i32_i8(a, b, c, sat):
                        i_ok = False
    n_cases = len(WORDS_A) * len(WORDS_B) * len(ACCUMS) * 2
    check('v_dot4_u32_u8 grid (%d cases, wrap + clamp)' % n_cases, u_ok)
    check('v_dot4_i32_i8 grid (%d cases, wrap + clamp)' % n_cases, i_ok)

    # 3. On-demand helper emission: each helper emitted only when its instruction is used.
    _, ps_u = translate('v_dot4_u32_u8', 0)
    _, ps_i = translate('v_dot4_i32_i8', 0)
    check('dot4_u32_u8 emitted on demand and dot4_i32_i8 omitted',
          'uint dot4_u32_u8(' in ps_u and 'dot4_i32_i8' not in ps_u)
    check('dot4_i32_i8 emitted on demand and dot4_u32_u8 omitted',
          'uint dot4_i32_i8(' in ps_i and 'dot4_u32_u8' not in ps_i)
    check('dot4_i32_i8 helper keeps both signed overflow sign guards',
          'asint(c) > 0 && s > 0 && asint(r) < 0' in ps_i and
          'asint(c) < 0 && s < 0 && asint(r) >= 0' in ps_i)
    zero = [(0, 0, 0)] * 4
    unused_ps = G.Pipeline(vs_words(zero), PRE_WORDS + POST_WORDS, root_blob()).hlsl_ps()
    check('both dot4 helpers omitted when unused', 'dot4_' not in unused_ps)

    # 4. EXEC masking & clamp flag propagation.
    for op, helper in (('v_dot4_u32_u8', 'dot4_u32_u8'), ('v_dot4_i32_i8', 'dot4_i32_i8')):
        p = Probe()
        p.exec_known = False
        p.one(op, '%s v12, v0, v4, v8 clamp' % op)
        check('%s exec-masked + clamp' % op,
              p.lines == ['if (exec & 1) v12 = %s(v0, v4, v8, true);' % helper])

    # 5. Refusals: arity, non-clamp VOP3P modifiers (op_sel, op_sel_hi, neg_lo, neg_hi), bad dst/src, source mods.
    for op, code in OPCODES.items():
        check('%s 3 operands refused' % op, stage_refused(Probe(), op, '%s v1, v2, v3' % op, 'expected 4'))
        check('%s 5 operands refused' % op, stage_refused(Probe(), op, '%s v1, v2, v3, v4, v5' % op, 'expected 4'))
        for label, bad_word in (
            ('op_sel', vop3p(code, 12, 256, 260, 264, op_sel=1)),
            ('op_sel_hi', vop3p(code, 12, 256, 260, 264, op_sel_hi=0)),
            ('neg_lo', vop3p(code, 12, 256, 260, 264, neg_lo=1)),
            ('neg_hi', vop3p(code, 12, 256, 260, 264, neg_hi=1)),
        ):
            asm = G.decode(bad_word)[0][3]
            check('%s VOP3P %s refused' % (op, label),
                  stage_refused(Probe(), op, asm, 'unsupported modifiers'))
        for bad_dst in ('s4', 'v[0:1]', 'v256'):
            check('%s bad dst %s refused' % (op, bad_dst),
                  stage_refused(Probe(), op, '%s %s, v0, v4, v8' % (op, bad_dst), 'destination'))
        for bad_src in ('|v0|', '-v0', 'neg(1)', 'v[0:1]', 's[0:1]', 'v256', 's106'):
            check('%s bad src %s refused' % (op, bad_src),
                  stage_refused(Probe(), op, '%s v12, %s, v4, v8' % (op, bad_src)))

    # 6. Unlowered control: removing handlers from Stage restores refusals.
    saved = {name: getattr(G.Stage, name) for name in ('op_v_dot4_i32_i8', 'op_v_dot4_u32_u8')}
    try:
        for name in saved:
            delattr(G.Stage, name)
        for op in OPCODES:
            check('unlowered %s refused' % op, stage_refused(Probe(), op, '%s v12, v0, v4, v8' % op))
            try:
                translate(op)
            except G.Unsupported:
                check('unlowered %s pipeline refused' % op, True)
            else:
                check('unlowered %s pipeline refused' % op, False)
    finally:
        for name, fn in saved.items():
            setattr(G.Stage, name, fn)


# Four-lane draw cases covering wrapping, positive/negative signed sums, and saturation.
DRAW_CASES = (
    ('v_dot4_u32_u8', 0, [
        (0x01020304, 0x05060708, 0x3f800000),  # 0x3f800000 + 70
        (0xff00ff00, 0x01000200, 0x3f800000),  # 0x3f800000 + 765
        (0xffffffff, 0xffffffff, 0x3f800000),  # 0x3f800000 + 260100 (max unsigned dot4)
        (0x00000001, 0x00000001, 0xffffffff),  # wraps cleanly to 0x00000000 (+0.0)
    ]),
    ('v_dot4_u32_u8', 1, [
        (0x01020304, 0x05060708, 0x3f800000),  # no saturation
        (0x00000001, 0x00000001, 0xfffffffe),  # saturates to 0xffffffff
        (0x10203040, 0x01010101, 0x40000000),  # no saturation
        (0x00000000, 0xffffffff, 0x3f800000),  # zero sum
    ]),
    ('v_dot4_i32_i8', 0, [
        (0x01020304, 0x05060708, 0x3f800000),  # positive sum (+70)
        (0xfefdfcfb, 0x01020304, 0x3f800000),  # negative sum (-40)
        (0x80808080, 0x80808080, 0x3f800000),  # (-128)*(-128)*4 = +65536
        (0x80808080, 0x7f7f7f7f, 0x3f800000),  # (-128)*(+127)*4 = -65024
    ]),
    ('v_dot4_i32_i8', 1, [
        (0x80808080, 0x80808080, 0x7fff0000),  # positive overflow -> saturates to 0x7fffffff
        (0x80808080, 0x7f7f7f7f, 0x80000000),  # negative overflow -> saturates to 0x80000000
        (0x80808080, 0x7f7f7f7f, 0x3f800000),  # no overflow
        (0x01020304, 0x05060708, 0x3f800000),  # no overflow
    ]),
    ('v_dot4_i32_i8', 1, [
        (0xfefdfcfb, 0x01020304, 0x00000001),  # c > 0, s < 0, r < 0: no overflow (-39)
        (0x01020304, 0x05060708, 0xffffffff),  # c < 0, s > 0, r >= 0: no overflow (+69)
        (0x7f7f7f7f, 0x7f7f7f7f, 0x7fff0000),  # close to INT32_MAX, no overflow
        (0x80808080, 0x7f7f7f7f, 0xffff0000),  # negative + negative, no overflow
    ]),
)


def run_draw(dxc, renderer, adapter, fxc=None):
    run_controls()
    compilers = ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else []) + [('dxc', dxc, 'vs_6_0', 'ps_6_0')]
    with tempfile.TemporaryDirectory(prefix='x2w_v_dot4_') as tmp:
        td = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            first_vs = None
            first_ps = None
            first_exp = None
            for op, clamp, triples in DRAW_CASES:
                ref_fn = ref_dot4_i32_i8 if op == 'v_dot4_i32_i8' else ref_dot4_u32_u8
                _, ps_hlsl = translate(op, clamp)
                ps_bin = compile_hlsl(compiler, kind, ps_hlsl, 'ps_%s_%d_%s' % (op, clamp, kind), ps_profile, td)
                vs_hlsl = G.Pipeline(vs_words(triples), ps_words(op, clamp), root_blob()).hlsl_vs()
                vs_bin = compile_hlsl(compiler, kind, vs_hlsl, 'vs_%s_%d_%s' % (op, clamp, kind), vs_profile, td)
                raw_exp = tuple(ref_fn(a, b, c, bool(clamp)) for a, b, c in triples)
                expected = words_text(raw_exp)
                res = draw(renderer, adapter, vs_bin, ps_bin, expected)
                if res.returncode == 77:
                    print('SKIP (%s): %s' % (adapter, res.stdout.strip() or res.stderr.strip()))
                    sys.exit(77)
                if res.returncode != 0:
                    raise AssertionError('%s %s %s clamp=%d expected=%s\n%s\n%s' % (
                        kind, adapter, op, clamp, expected, res.stdout, res.stderr))
                print('%s %s %s clamp=%d: %s' % (kind, adapter, op, clamp, expected), flush=True)
                if first_vs is None:
                    first_vs, first_ps, first_exp = vs_bin, ps_bin, raw_exp

            bad = words_text((first_exp[0] ^ 1, first_exp[1], first_exp[2], first_exp[3]))
            neg = draw(renderer, adapter, first_vs, first_ps, bad)
            check('draw %s %s negative control rejects wrong expectation' % (kind, adapter),
                  neg.returncode != 0)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        run_controls()
        print('v_dot4 controls: ok')
    elif len(sys.argv) in (5, 6) and sys.argv[1] == 'draw':
        run_draw(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5] if len(sys.argv) == 6 else None)
        print('v_dot4 draw (%s): ok' % sys.argv[4])
    else:
        print('usage: startup_gfx2hlsl_v_dot4_check.py [draw <dxc> <offscreen_check> <warp|hardware> [fxc]]',
              file=sys.stderr)
        sys.exit(2)
