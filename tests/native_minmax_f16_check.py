"""Direct compute v_min_f16 and v_max_f16 (VOP2 and VOP3A, exact integer ordering, high-half preserving)."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_dpp_quad_check import request

MASK = 0xFFFFFFFF
SENTINEL = 0x69696969
HIGH_SEED = 0xBEEF0000

CASES = (
    (0xCAFE8000, 0x12340000),  # -0.0h vs +0.0h (with high junk)
    (0x00000000, 0x00008000),  # +0.0h vs -0.0h (reversed order)
    (0x00003C00, 0x0000BC00),  # +1.0h vs -1.0h
    (0x00000001, 0x00008001),  # +min_sub vs -min_sub
    (0x00007C00, 0x0000FC00),  # +Inf vs -Inf
    (0x00007E00, 0x0000C000),  # qNaN vs -2.0h -> -2.0h
    (0x00004000, 0x00007E01),  # +2.0h vs qNaN -> +2.0h
    (0x00007E01, 0x00007E02),  # qNaN vs qNaN -> 0x7E00 (clamped to 0)
)


def v_mov_lit(vdst, lit):
    return [0x7E0002FF | ((vdst & 0xFF) << 17), lit & MASK]


def vop2(op, vdst, src0, vsrc1):
    return ((op & 0x3F) << 25) | ((vdst & 0xFF) << 17) | ((vsrc1 & 0xFF) << 9) | (src0 & 0x1FF)


def vop3_binary(op, vdst, src0, src1, *, abs_m=0, neg_m=0, clamp=0, omod=0, op_sel=0):
    w0 = (0xD4000000 | ((op & 0x3FF) << 16) | ((clamp & 1) << 15) |
          ((op_sel & 0xF) << 11) | ((abs_m & 7) << 8) | (vdst & 0xFF))
    w1 = ((neg_m & 7) << 29) | ((omod & 3) << 27) | ((src1 & 0x1FF) << 9) | (src0 & 0x1FF)
    return [w0, w1]


def program():
    code = [0xBE940304, 0xBE950380, 0xF408080A, 0xFA000020,
            0x8F078C05, 0x34100087, 0x4A101007, 0xBF8CC07F]
    for ci, (a, b) in enumerate(CASES):
        code += v_mov_lit(2, a) + v_mov_lit(3, b)
        # 0: VOP2 v_min_f16 v4, v2, v3
        code += v_mov_lit(4, HIGH_SEED | 0x1111) + [vop2(0x3A, 4, 258, 3), 0xE0701000 | (4 * (ci * 4 + 0)), 0x80080408]
        # 1: VOP2 v_max_f16 v4, v2, v3
        code += v_mov_lit(4, HIGH_SEED | 0x2222) + [vop2(0x39, 4, 258, 3), 0xE0701000 | (4 * (ci * 4 + 1)), 0x80080408]
        # 2: VOP3A v_min_f16 v4, -v2, |v3|
        code += v_mov_lit(4, HIGH_SEED | 0x3333) + vop3_binary(0x13A, 4, 258, 259, neg_m=1, abs_m=2) + [
            0xE0701000 | (4 * (ci * 4 + 2)), 0x80080408]
        # 3: VOP3A v_max_f16 v4, v2, v3 clamp
        code += v_mov_lit(4, HIGH_SEED | 0x4444) + vop3_binary(0x139, 4, 258, 259, clamp=1) + [
            0xE0701000 | (4 * (ci * 4 + 3)), 0x80080408]
    return code + [0xBF810000]


def f16_key(u, ignore_signed_zero=False):
    u &= 0xFFFF
    if ignore_signed_zero and (u & 0x7FFF) == 0:
        return 0
    return (-1 - (u & 0x7FFF)) if (u & 0x8000) else u


def minmax_f16(a, b, is_min, *, clamp=False, ignore_signed_zero=False, propagate_nan=False):
    a &= 0xFFFF
    b &= 0xFFFF
    a_nan = (a & 0x7FFF) > 0x7C00
    b_nan = (b & 0x7FFF) > 0x7C00
    if a_nan and b_nan:
        res = 0x7E00
    elif a_nan:
        res = 0x7E00 if propagate_nan else b
    elif b_nan:
        res = 0x7E00 if propagate_nan else a
    else:
        ka = f16_key(a, ignore_signed_zero=ignore_signed_zero)
        kb = f16_key(b, ignore_signed_zero=ignore_signed_zero)
        res = a if ((ka < kb) if is_min else (ka > kb)) else b
    if clamp:
        if (res & 0x7FFF) > 0x7C00 or (res & 0x8000):
            res = 0
        elif res > 0x3BFF:
            res = 0x3C00
    return res & 0xFFFF


def reference(*, ignore_signed_zero=False, propagate_nan=False, no_clamp=False, zero_high=False):
    hi = 0 if zero_high else HIGH_SEED
    words = [SENTINEL] * 8
    for _lane in range(64):
        for a, b in CASES:
            r_min = minmax_f16(a, b, True, ignore_signed_zero=ignore_signed_zero, propagate_nan=propagate_nan)
            r_max = minmax_f16(a, b, False, ignore_signed_zero=ignore_signed_zero, propagate_nan=propagate_nan)
            r_mod = minmax_f16((a & 0xFFFF) ^ 0x8000, b & 0x7FFF, True,
                               ignore_signed_zero=ignore_signed_zero, propagate_nan=propagate_nan)
            r_clamp = minmax_f16(a, b, False, clamp=not no_clamp,
                                 ignore_signed_zero=ignore_signed_zero, propagate_nan=propagate_nan)
            words.extend((hi | r_min, hi | r_max, hi | r_mod, hi | r_clamp))
    return words + [SENTINEL] * 8


def controls(cli, sdk, work, root):
    marker = request(work, root, program())
    output = marker.with_suffix('.cso')
    result = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', marker, output],
                            capture_output=True, text=True)
    assert result.returncode == 0 and output.is_file(), (result.stdout, result.stderr)
    dump = subprocess.run([Path(sdk) / 'dxc.exe', '-dumpbin', output], capture_output=True,
                          text=True, check=True).stdout
    assert 'icmp slt i32 ' in dump and 'icmp sgt i32 ' in dump, dump
    for tag, bad, msg in (
        ('omod', vop3_binary(0x13A, 4, 258, 259, omod=1) + [0xBF810000], 'unsupported v_min_f16 operand modifiers'),
        ('op_sel', vop3_binary(0x139, 4, 258, 259, op_sel=1) + [0xBF810000], 'unsupported v_max_f16 operand modifiers'),
    ):
        sub = work / tag
        sub.mkdir(exist_ok=True)
        bad_marker = request(sub, root, bad)
        bad_res = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', bad_marker,
                                  bad_marker.with_suffix('.cso')], capture_output=True, text=True)
        assert bad_res.returncode != 0 and msg in (bad_res.stdout + bad_res.stderr), (tag, bad_res.stdout, bad_res.stderr)
    assert reference() != reference(ignore_signed_zero=True)
    assert reference() != reference(propagate_nan=True)
    assert reference() != reference(no_clamp=True)
    assert reference() != reference(zero_high=True)
    print('v_min_f16/v_max_f16 DXIL verified; invalid modifiers refused; signed-zero, NaN, clamp and high-half controls differ')
    return 0


def hardware(replay, sdk, work, root):
    marker = request(work, root, program())
    output = work / 'out.bin'
    result = subprocess.run([replay, sdk, marker, work / 'cache', '0', output, '--direct'])
    if result.returncode == 77:
        return 77
    result.check_returncode()
    expected = reference()
    actual = struct.unpack(f'<{len(expected)}I', output.read_bytes())
    mismatch = next((i for i, (a, e) in enumerate(zip(actual, expected)) if a != e), None)
    if mismatch is not None:
        raise AssertionError(f'word {mismatch}: got {actual[mismatch]:08x}, expected {expected[mismatch]:08x}')
    print('v_min_f16/v_max_f16: 2048 result words (high halves kept) and 16 guards matched')
    return 0


def main(mode, executable, sdk):
    with tempfile.TemporaryDirectory(prefix='x2w_native_minmax_f16_') as temp:
        work = Path(temp)
        shader = work / 'root.hlsl'
        shader.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
        root = work / 'root.bin'
        subprocess.run([Path(sdk) / 'dxc.exe', '-T', 'rootsig_1_1', '-E', 'SIG', '-Fo', root, shader], check=True)
        return (controls if mode == 'controls' else hardware)(executable, sdk, work, root.read_bytes())


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1], sys.argv[2], sys.argv[3]))
