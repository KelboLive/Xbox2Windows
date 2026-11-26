"""gfx1013 machine code -> HLSL for a vertex+pixel pipeline (startup tools; see docs/startup-runtime.md).

Model: one lane. Registers are uint variables; exec/vcc/mask registers hold the lane's bit in bit 0.
Scalar registers are also tracked symbolically at translation time so that pointer and descriptor
values (tables, vertex-buffer table, root buffers) turn into HLSL resources instead of arithmetic.
The output is HLSL for both stages plus a manifest that tells the runtime what to bind where.
"""
import bisect
from contextlib import suppress
import functools
import os
import re
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import disasm_rdna2 as D  # noqa: E402


class Unsupported(Exception):
    pass


PREFETCH = bytes.fromhex('0300a0bf')

# SDWA (sub-dword addressing) selections, with the same meaning as tools/pixel_aot.py: a source
# selection is zero-extended, or sign-extended under sext(); a destination selection writes the low
# bits of the result into that field, UNUSED_PAD zeroes the others and UNUSED_PRESERVE keeps them.
SDWA_SEL = {'BYTE_0': (0, 8), 'BYTE_1': (8, 8), 'BYTE_2': (16, 8), 'BYTE_3': (24, 8),
            'WORD_0': (0, 16), 'WORD_1': (16, 16), 'DWORD': (0, 32)}


def sdwa_source(word, sel, sext):
    """The uint expression a source selection reads out of a 32-bit register."""
    offset, width = SDWA_SEL[sel]
    if width == 32:
        return word
    if sext:
        return 'asuint(asint(%s << %du) >> %du)' % (word, 32 - offset - width, 32 - width)
    return '((%s >> %du) & %#xu)' % (word, offset, (1 << width) - 1)


def sdwa_dest(result, sel, unused, old):
    """The uint expression a destination selection writes back into a 32-bit register."""
    offset, width = SDWA_SEL[sel]
    if width == 32:
        return result
    if unused == 'UNUSED_SEXT':
        raise Unsupported('SDWA UNUSED_SEXT')
    field = '((%s & %#xu) << %du)' % (result, (1 << width) - 1, offset)
    if unused == 'UNUSED_PAD':
        return field
    if unused != 'UNUSED_PRESERVE':
        raise Unsupported('SDWA dst_unused %s' % unused)
    keep = ~(((1 << width) - 1) << offset) & 0xffffffff
    return '((asuint(%s) & %#xu) | %s)' % (old, keep, field)


def strip_header(code):
    at = code.find(PREFETCH, 0, 0x80)
    # A call (s_swappc_b64) just before the prefetch is the program's first instruction, not header: keep it.
    if at >= 4 and at % 4 == 0 and struct.unpack_from('<I', code, at - 4)[0] & 0xff80ff00 == 0xbe802100:
        at -= 4
    return code[at:] if at >= 0 and at % 4 == 0 else code


# A stream vertex shader whose inputs come from a fetch routine the driver builds from the input layout starts
# with this call; the routine's address arrives in s[4:5]. Its FSSM header lists, from +0x138, one 5-byte record
# per fetched input: kind (6), input-signature register, component mask, first VGPR, component count - 1.
# Checked offline against the input signatures of every readable vertex program of one title's shader archive
# that starts with the call: each record's register and mask match a signature input (the mask is the read
# mask; the count reaches its highest component), the first VGPR is v4 and each next record is 4 VGPRs on.
# Programs without the call have no records. The header's u32 at +0x1d8 holds the record count.
# Any other header size, record kind or count is refused.
FETCH_CALL = 0xbe842104  # s_swappc_b64 s[4:5], s[4:5]
FETCH_HEADER_BASE, FETCH_RECORDS, FETCH_KIND, FETCH_COUNT = 0x2f8, 0x138, 6, 0x1d8
FETCH_SLOTS = 8  # vertex-buffer slots the stand-in binds (D3D12_VERTEX_BUFFER_VIEW vb[8])
# DXGI format -> (BUF_FMT name in FETCH_FORMATS, bytes per element).
FETCH_DXGI = {2: ('BUF_FMT_32_32_32_32_FLOAT', 16), 6: ('BUF_FMT_32_32_32_FLOAT', 12), 16: ('BUF_FMT_32_32_FLOAT', 8),
              41: ('BUF_FMT_32_FLOAT', 4), 28: ('BUF_FMT_8_8_8_8_UNORM', 4), 49: ('BUF_FMT_8_8_UNORM', 2),
              61: ('BUF_FMT_8_UNORM', 1), 34: ('BUF_FMT_16_16_FLOAT', 4), 35: ('BUF_FMT_16_16_UNORM', 4)}
# Typed buffer loads: (word load or None, the four component expressions). Components a format lacks read as
# 0, 0, 0, 1 here, as in the existing tbuffer translation; that default is an assumption, not a measurement.
FETCH_FORMATS = {
    'BUF_FMT_8_8_8_8_UNORM': ('ldu({0}, a)', ['U((w & 255) / 255.0)', 'U(((w >> 8) & 255) / 255.0)', 'U(((w >> 16) & 255) / 255.0)', 'U((w >> 24) / 255.0)']),
    'BUF_FMT_8_8_UNORM': ('ldu({0}, a)', ['U((w & 255) / 255.0)', 'U(((w >> 8) & 255) / 255.0)', '0u', 'U(1.0)']),
    'BUF_FMT_8_UNORM': ('ldu({0}, a)', ['U((w & 255) / 255.0)', '0u', '0u', 'U(1.0)']),
    'BUF_FMT_8_UINT': ('{0}.Load(a & ~3u) >> ((a & 3u) * 8u)', ['w & 255u', '0u', '0u', '1u']),
    'BUF_FMT_16_16_UNORM': ('ldu({0}, a)', ['U((w & 65535) / 65535.0)', 'U((w >> 16) / 65535.0)', '0u', 'U(1.0)']),
    'BUF_FMT_16_16_FLOAT': ('ldu({0}, a)', ['U(f16tof32(w))', 'U(f16tof32(w >> 16))', '0u', 'U(1.0)']),
    # Four half channels in one 8-byte element: x/y in the first dword, z/w in the second.
    'BUF_FMT_16_16_16_16_FLOAT': (None, ['U(f16tof32(ldu({0}, a)))', 'U(f16tof32(ldu({0}, a) >> 16))',
                                         'U(f16tof32(ldu({0}, a + 4)))', 'U(f16tof32(ldu({0}, a + 4) >> 16))']),
    'BUF_FMT_32_32_32_32_FLOAT': (None, ['{0}.Load(a)', '{0}.Load(a + 4)', '{0}.Load(a + 8)', '{0}.Load(a + 12)']),
    'BUF_FMT_32_32_32_FLOAT': (None, ['{0}.Load(a)', '{0}.Load(a + 4)', '{0}.Load(a + 8)', 'U(1.0)']),
    'BUF_FMT_32_32_FLOAT': (None, ['{0}.Load(a)', '{0}.Load(a + 4)', '0u', 'U(1.0)']),
    'BUF_FMT_32_FLOAT': (None, ['{0}.Load(a)', '0u', '0u', 'U(1.0)'])}


def starts_with_fetch_call(code):
    return len(code) >= 4 and struct.unpack_from('<I', code)[0] == FETCH_CALL


def parse_input_layout(text):
    """'ia1 <n>' then one '<semantic> <index> <format> <slot> <offset> <class> <step>' line per element."""
    lines = text.split('\n')
    if not re.fullmatch(r'ia1 ([1-9]|[12][0-9]|3[0-2])', lines[0]) or lines[-1] != '' or len(lines) != int(lines[0][4:]) + 2:
        raise Unsupported('invalid input layout request')
    elements, ends = [], {}
    for line in lines[1:-1]:
        m = re.fullmatch(r'([A-Za-z_][A-Za-z0-9_]{0,62}) (\d{1,10}) (\d{1,10}) (\d{1,2}) (\d{1,10}) (\d) (\d{1,10})', line)
        if not m:
            raise Unsupported('invalid input layout element')
        name, index, fmt, slot, offset, cls, step = m.group(1).upper(), *map(int, m.groups()[1:])
        if offset == 0xffffffff:  # D3D12_APPEND_ALIGNED_ELEMENT: right after the previous element of the slot
            # shortcut: unknown when an earlier size is unknown or ends off a dword (packing not measured).
            offset = ends.get(slot, 0)
            if offset is not None and offset % 4:
                offset = None
        size = FETCH_DXGI[fmt][1] if fmt in FETCH_DXGI else None
        ends[slot] = None if offset is None or size is None else offset + size
        # Format, class and slot are checked in stream_fetch, only for the elements a record fetches.
        elements.append(dict(name=name, index=index, format=fmt, slot=slot, offset=offset, cls=cls, step=step))
    return elements


def signature_inputs(isg1):
    """ISG1 part -> {register: (semantic name, semantic index, system value)}."""
    count, first = struct.unpack_from('<II', isg1, 0)
    out = {}
    for i in range(count):
        _, name_at, index, system, _, register, _ = struct.unpack_from('<6IB', isg1, first + 32 * i)
        out[register] = (isg1[name_at:isg1.index(b'\0', name_at)].decode('ascii').upper(), index, system)
    return out


def stream_fetch(vs_container, layout_text):
    """The fetch a stream vertex shader expects before its first instruction: [dict(vgpr, count, ...layout)]."""
    from pixel_aot import container_image
    from resource_inventory import parts
    if layout_text is None:
        raise Unsupported('vertex stage calls a fetch routine and the pipeline declares no input layout')
    sections = parts(vs_container)
    image = container_image(sections[b'DXIL'], stage=1)
    if len(image) < FETCH_HEADER_BASE or struct.unpack_from('<Q', image, 192)[0] != FETCH_HEADER_BASE:
        raise Unsupported('unverified FSSM header for a vertex fetch table')
    inputs, layout = signature_inputs(sections[b'ISG1']), parse_input_layout(layout_text)
    fetch, at = [], FETCH_RECORDS
    while image[at] != 0:
        kind, register, mask, vgpr, last = image[at:at + 5]
        count = last + 1
        # The count reaches the highest component the shader reads (mask 0x4 comes with count 3): load all of them.
        if (kind != FETCH_KIND or mask >> last != 1 or not 1 <= count <= 4 or vgpr < 1 or vgpr + count > 256
                or len(fetch) == 16):
            raise Unsupported('unverified vertex fetch record %s' % image[at:at + 5].hex())
        if register not in inputs:
            raise Unsupported('vertex fetch record for input register %d outside the signature' % register)
        name, index, _ = inputs[register]
        match = [e for e in layout if (e['name'], e['index']) == (name, index)]
        if len(match) != 1:
            raise Unsupported('input layout has %d elements for vertex input register %d' % (len(match), register))
        e = match[0]
        if e['format'] not in FETCH_DXGI:
            raise Unsupported('input layout format %d' % e['format'])
        if e['cls'] != 0 or e['step'] != 0:
            raise Unsupported('per-instance input layout element')
        if e['slot'] >= FETCH_SLOTS:
            raise Unsupported('input layout slot %d: the stand-in binds vertex-buffer slots 0 to %d' % (e['slot'], FETCH_SLOTS - 1))
        if e['offset'] is None:
            raise Unsupported('appended input layout element after an element of unknown or unaligned size')
        fetch.append(dict(e, vgpr=vgpr, count=count))
        at += 5
    if not fetch:
        raise Unsupported('vertex stage calls a fetch routine and its container lists no fetched inputs')
    stored = struct.unpack_from('<I', image, FETCH_COUNT)[0]
    if stored != len(fetch):
        raise Unsupported('vertex fetch header counts %d records, the table holds %d' % (stored, len(fetch)))
    taken = [r for e in fetch for r in range(e['vgpr'], e['vgpr'] + e['count'])]
    if len(taken) != len(set(taken)):
        raise Unsupported('vertex fetch records overlap')
    return fetch


def decode(code)
    attributes = {}
    count = struct.unpack_from('<I', ps_image, IFACE_PS_COUNT)[0]
    if count > (IFACE_PS_COUNT - IFACE_PS_RECORDS) // 8:
        raise Unsupported('pixel input table exceeds its header range')
    for i in range(count):
        kind, register, mask, attr, b4, b5, again, interp = ps_image[IFACE_PS_RECORDS + 8 * i:IFACE_PS_RECORDS + 8 * i + 8]
        if (kind != 6 or attr != i or b4 or b5 or again != register or interp not in (1, 2) or not mask
                or mask & ~0xf or register in attributes):
            raise Unsupported('unverified pixel input record %s' % ps_image[IFACE_PS_RECORDS + 8 * i:IFACE_PS_RECORDS + 8 * i + 8].hex())
        declared = 0
        for entry in inputs:
            if entry[0] == register:
                declared |= entry[4]
        if mask != declared:
            raise Unsupported('pixel input record mask 0x%x differs from the signature mask 0x%x of register %d'
                              % (mask, declared, register))
        attributes[register] = mask
    if not attributes:
        return {}
    params = {}
    count = struct.unpack_from('<I', vs_image, IFACE_VS_COUNT)[0]
    if count > (IFACE_VS_COUNT - IFACE_VS_RECORDS) // 6:
        raise Unsupported('vertex output table exceeds its header range')
    for i in range(count):
        kind, register, param, *rest = vs_image[IFACE_VS_RECORDS + 6 * i:IFACE_VS_RECORDS + 6 * i + 6]
        if kind != 6 or param != i or any(rest) or register in params:
            raise Unsupported('unverified vertex output record %s' % vs_image[IFACE_VS_RECORDS + 6 * i:IFACE_VS_RECORDS + 6 * i + 6].hex())
        params[register] = param
    outputs = {}
    for register, name, index, system, mask in signature_entries(vs[b'OSG1']):
        if system:
            continue
        if (name, index) in outputs:
            raise Unsupported('duplicate vertex output semantic %s%d' % (name, index))
        outputs[name, index] = register, mask
    mapping = {}
    for attr, (register, mask) in enumerate(attributes.items()):
        sources = set()
        for _, name, index, _, semantic_mask in (e for e in inputs if e[0] == register):
            output = outputs.get((name, index))
            if not output or output[1] != semantic_mask:
                raise Unsupported('pixel input %s%d has no vertex output with the same components' % (name, index))
            sources.add(output[0])
        if len(sources) != 1 or next(iter(sources)) not in params:
            raise Unsupported('pixel attribute %d does not come from exactly one exported vertex register' % attr)
        mapping[attr] = (params[next(iter(sources))], mask)
    return mapping


# Binary32 words for inline source encodings 240 through 248, in encoding order.
CNDMASK_INLINE_FLOAT_BITS = (0x3f000000, 0xbf000000, 0x3f800000, 0xbf800000,
                            0x40000000, 0xc0000000, 0x40800000, 0xc0800000,
                            0x3e22f983)  # 1/(2*pi)


def check_encoding(code, off, n, name, asm):
    # These SOP1 forms have an unused field that the disassembler does not print.
    # Only the ordinary four-byte encodings (no literal or hidden operand) are used
    # by the calculated-branch proof.
    if name in ('s_getpc_b64', 's_setpc_b64'):
        word, = struct.unpack_from('<I', code, off)
        unused = word & (0xff if name == 's_getpc_b64' else 0x7f0000)
        if n != 4 or unused:
            raise Unsupported('s_setpc_b64: unsupported %s encoding at +0x%x' % (name, off))
    if name in ('v_interp_p1ll_f16', 'v_interp_p2_f16'):
        # The shared disassembler ignores op_sel (w0[14:11]) here, as LLVM 15 does. Mesa ACO's
        # v_interp_p2_hi_f16 is p2 with op_sel bit 3 (w0[14]): it writes the high half of the destination. The
        # attribute half `high` is a different bit (w1[8]) and is printed, so it is not refused here.
        if struct.unpack_from('<I', code, off)[0] & 0x7800:
            raise Unsupported('%s with op_sel: only the plain form and high are lowered' % name)
    if name == 'image_load_mip':
        # The shared disassembler does not expose these gfx10 MIMG flags.
        word0, word1 = struct.unpack_from('<2I', code, off)
        if word0 & (1 << 16):
            raise Unsupported('image_load_mip tfe')
        if word1 & (1 << 30):
            raise Unsupported('image_load_mip a16')
    if re.fullmatch(r'v_cmpx?_(lt|eq|le|gt|ne|ge)_[iu]16_e64', name):
        # The decoder prints NEG on an inline integer as a sign (inline 1 with NEG reads `-1`, the same text as
        # inline -1), and LLVM 15 rejects `-v1` and `|v1|` on these sources: refuse either bit (VOP3 w0[10:8] ABS,
        # w1[31:29] NEG) so a printed -N is always a genuine inline integer.
        word0, word1 = struct.unpack_from('<2I', code, off)
        if word0 & 0x700 or word1 & 0xe0000000:
            raise Unsupported('%s with neg/abs: not a 16-bit integer source modifier' % name)
    if name in ('v_sad_u8', 'v_sad_u16', 'v_sad_u32'):
        # VOP3A integer SAD forms take no source abs/neg; refuse those bits here so neg on an inline
        # integer does not print as -N and get accepted as a negative constant.
        word0, word1 = struct.unpack_from('<2I', code, off)
        if word0 & 0x700 or word1 & 0xe0000000:
            raise Unsupported('%s with neg/abs' % name)
    if name == 'v_cvt_off_f32_i4_e64':
        # Integer-nibble conversion: the instruction has one source, so only src0's VOP3A ABS (w0[8]) and NEG
        # (w1[29]) can carry a modifier. Masking all three bit positions of each field (w0[10:8], w1[31:29])
        # also refuses the unused src1/src2 bits, which keeps an inline constant from masquerading as a
        # negative integer and keeps a hidden modifier out of the printed operand.
        word0, word1 = struct.unpack_from('<2I', code, off)
        if word0 & 0x700 or word1 & 0xe0000000:
            raise Unsupported('%s with neg/abs: not a 4-bit integer source modifier' % name)
    if name == 'v_cndmask_b32_e64':
        # The printed -1 is ambiguous: it can be plain inline -1 or inline 1 with NEG.
        word0, word1 = struct.unpack_from('<2I', code, off)
        for i in range(2):
            source = (word1 >> (9 * i)) & 0x1ff
            if not (128 <= source <= 208 or 240 <= source <= 248 or source == 255):
                continue
            if word0 & (1 << (8 + i)):
                raise Unsupported('v_cndmask_b32_e64 ABS on an immediate')
            if not word1 & (1 << (29 + i)):
                continue
            if source <= 192:
                bits = source - 128
            elif source <= 208:
                bits = (192 - source) & 0xffffffff
            elif source <= 248:
                bits = CNDMASK_INLINE_FLOAT_BITS[source - 240]
            else:
                if n != 12 or off + n > len(code):
                    raise Unsupported('v_cndmask_b32_e64 missing literal dword')
                bits, = struct.unpack_from('<I', code, off + 8)
            # Preserve the other operand spellings and any modifiers on the final operand.
            fields = asm.split(',', 3)
            if len(fields) != 4:
                raise Unsupported('malformed v_cndmask_b32_e64 operands')
            operand = fields[i + 1]
            fields[i + 1] = operand[:len(operand) - len(operand.lstrip())] + '0x%08x' % (bits ^ 0x80000000)
            asm = ','.join(fields)
    if n >= 8 and struct.unpack_from('<I', code, off)[0] >> 26 == 0b110101:
        # VOP3 NEG (w1[31:29], bit 29 + i for source i) on a non-negative inline integer (128..192) or a literal (255).
        # The decoder prints inline 1 with NEG as `-1`, the text of the plain inline -1 (193), so both would translate to
        # 0xffffffff, and a literal 5 would become 0xfffffffb. Only v_cndmask_b32_e64 models NEG on its two data
        # sources (above, as the sign bit flipped). Inline floats, registers and negative inlines are not affected.
        # ABS is not tested: on the VOP3b forms w0[10:8] is SDST, not a modifier.
        word1, = struct.unpack_from('<I', code, off + 4)
        for i in range(3):
            if name == 'v_cndmask_b32_e64' and i < 2 or not word1 & (1 << (29 + i)):
                continue
            source = (word1 >> (9 * i)) & 0x1ff
            if 128 <= source <= 192 or source == 255:
                raise Unsupported('%s with NEG on an integer or literal source %d: the printed sign would read as inline -N'
                                  % (name, i))
    return asm


def decode(code, refusals=None):
    out, off = [], 0
    while off + 4 <= len(code):
        try:
            n, name, asm, _ = D.decode_instruction(code, off)
        except Exception as error:
            raise Unsupported('undecodable word at +0x%x: %s' % (off, error))
        if name == 'image_load_mip':
            # The shared disassembler does not expose these gfx10 MIMG flags.
            word0, word1 = struct.unpack_from('<2I', code, off)
            if word0 & (1 << 16):
                raise Unsupported('image_load_mip tfe')
            if word1 & (1 << 30):
                raise Unsupported('image_load_mip a16')
        if name == 'v_cndmask_b32_e64':
            # The printed -1 is ambiguous: it can be plain inline -1 or inline 1 with NEG.
            # Preserve plain immediates and refuse NEG on immediate data operands by encoding.
            word1, = struct.unpack_from('<I', code, off + 4)
            for i in range(2):
                if not word1 & (1 << (29 + i)):
                    continue
                source = (word1 >> (9 * i)) & 0x1ff
                if 128 <= source <= 208 or 240 <= source <= 248 or source == 255:
                    raise Unsupported('v_cndmask_b32_e64 NEG on an immediate')
        out.append((off, n, name, asm))
        off += n
    last = max((i for i, x in enumerate(out) if x[2] == 's_endpgm'), default=len(out) - 1)
    return out[:last + 1]


def split_top(text, sep=','):
    parts, depth, cur = [], 0, ''
    for ch in text:
        if ch in '[(':
            depth += 1
        elif ch in '])':
            depth -= 1
        if ch == sep and depth == 0:
            parts.append(cur.strip())
            cur = ''
        else:
            cur += ch
    if cur.strip():
        parts.append(cur.strip())
    return parts


def parse(asm):
    """-> (mnemonic, [operands], [modifier tokens])"""
    name, _, rest = asm.partition(' ')
    ops = split_top(rest)
    mods = []
    if ops:
        tail = split_top(ops[-1], ' ')
        ops[-1], mods = tail[0], tail[1:]
    return name, ops, mods


FLOATS = {'0.5': 0.5, '-0.5': -0.5, '1.0': 1.0, '-1.0': -1.0, '2.0': 2.0, '-2.0': -2.0, '4.0': 4.0, '-4.0': -4.0,
          '0.15915494': 0.15915494}


def fbits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def reg_range(tok):
    """'s[8:11]' -> ('s', 8, 11); 's8' -> ('s', 8, 8); 'v[1:3]' -> ('v', 1, 3); 'exec'/'vcc' pairs too."""
    m = re.fullmatch(r'([sv])\[(\d+):(\d+)\]', tok)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3))
    m = re.fullmatch(r'([sv])(\d+)', tok)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(2))
    return None



            out.update({'_lo': {mask}, '_hi': {mask + '_hi'}}.get(half, {mask, mask + '_hi'}))
        else:
            out.add('s' + single if single else m0 or scc)
    return out


def is_exec(regs):
    return regs in ({'exec'}, EXEC_KEYS)


def mask_words(tok, count):
    """The scalar state keys holding a mask operand's words, low word first ('0' for a zero literal), or None when
    the operand is not `count` whole words of scalar state."""
    if tok == '0':
        return ['0'] * count
    m = re.fullmatch(r's\[(\d+):(\d+)\]|s(\d+)|(vcc|exec)(_lo|_hi)?', tok)
    if not m:
        return None
    if m.group(4):
        words = {'_lo': [m.group(4)], '_hi': [m.group(4) + '_hi']}.get(m.group(5), [m.group(4), m.group(4) + '_hi'])
    else:
        words = ['s%d' % n for n in range(int(m.group(1) or m.group(3)), int(m.group(2) or m.group(3)) + 1)]
    return words if len(words) == count else None


@functools.lru_cache(maxsize=None)
def clear_shape(name, asm):
    """What clear_after needs from one instruction's text, which the fixed point would otherwise parse again on every
    visit: 'cmpx', None for a form that carries no fact, or (operation, saving suffix, word count, operand words)."""
    base = re.sub(r'_e(32|64)$', '', name)
    if base.startswith('v_cmpx'):
        return 'cmpx'
    m = re.fullmatch(r's_(mov|and|or|xor|andn2)(_saveexec|_wrexec)?_b(32|64)', base)
    if not m or (m.group(2) and m.group(1) == 'mov'):
        return None
    count = 2 if m.group(3) == '64' else 1
    words = [mask_words(op, count) for op in parse(asm)[1]]
    if None in words or len(words) != (2 if m.group(2) or m.group(1) == 'mov' else 3):
        return None
    return m.group(1), m.group(2), count, tuple(tuple(word) for word in words)


def clear_after(name, asm, may, clear):
    """Forward step of the proof Stage.execz_refusals runs over a block that s_cbranch_execz skips. A fact (key, h)
    says: for every lane of wave half h (0: lanes 0 to 31, 1: lanes 32 to 63) whose exec bit was clear at the branch,
    that lane's bit of the scalar state word `key` is clear. Returns the facts after the instruction, given the
    facts before it.

    Only bitwise forms carry a fact, per word: a zero literal holds it for both halves; D = A & B holds what either
    source holds; A | B and A ^ B what both hold; A & ~B and a move what A holds. A saveexec or wrexec form is its
    operation on exec plus the copy it saves, and v_cmpx ANDs its result into exec. Any other write, a compare
    result included, drops the facts of what it may write."""
    out = {fact for fact in clear if fact[0] not in may - EXEC_KEYS}
    shape = clear_shape(name, asm)
    if shape == 'cmpx':
        return out
    if shape is None:
        return {fact for fact in out if fact[0] not in may}
    op, saving, count, words = shape
    halves = lambda word: {0, 1} if word == '0' else {h for key, h in clear if key == word}
    dest, srcs = words[0], list(words[1:])
    if saving:  # RDNA 2 ISA SOP1: the operation is S0 op EXEC, so andn2 is S0 & ~EXEC
        srcs.append(('exec', 'exec_hi')[:count])
    combine = {'and': set.union, 'or': set.intersection, 'xor': set.intersection}.get(op, lambda a, *b: a)
    value = [combine(*(halves(src[j]) for src in srcs)) for j in range(count)]
    writes = list(zip(dest, value))
    if saving:  # saveexec keeps the old exec in D; wrexec writes the result to both
        writes = list(zip(('exec', 'exec_hi'), value)) + (writes if saving == '_wrexec' else
                                                          [(word, halves(old)) for word, old in zip(dest, srcs[-1])])
    out = {fact for fact in out if fact[0] not in dict(writes)}
    return out | {(word, h) for word, held in writes for h in held}


def scalar_effects(name, asm):
    """(reads, must-writes, may-writes) of scalar state for one decoded instruction."""
    base = re.sub(r'_e(32|64)$', '', name)
    _, ops, _ = parse(asm)
    if base.startswith(NO_SCALAR_EFFECT) or base == 's_endpgm':
        reads = {'scc'} if base.startswith('s_cbranch_scc') else {'vcc'} if base.startswith('s_cbranch_vcc') else \
            {'exec'} if base.startswith('s_cbranch_exec') else set()
        return reads, set(), set()
    if base.startswith('v_cmpx'):
        return scalar_tokens(' '.join(ops)) | {'exec'}, {'exec'}, set(EXEC_KEYS)
    if '_store' in base or base.startswith(('ds_write', 'exp') + NO_DESTINATION) or not ops:
        return scalar_tokens(' '.join(ops)) | ({'m0'} if 'movrel' in base else set()), set(), set()
    if base.startswith(('s_cmp', 's_bitcmp')):
        return scalar_tokens(' '.join(ops)), {'scc'}, {'scc'}
    two = base.startswith(TWO_DESTINATIONS)
    dests, srcs = ops[:2] if two else ops[:1], ops[2:] if two else ops[1:]
    writes = scalar_tokens(' '.join(dests))
    reads = scalar_tokens(' '.join(srcs))
    if base.startswith(READ_MODIFY_WRITE):
        reads |= writes
    if base.startswith(SCC_READERS) or (base in SCA and 'scc' in SCA[base][0]):
        reads.add('scc')
    if base.startswith(IMPLICIT_VCC_READ):
        reads.add('vcc')
    may = set()
    if 'saveexec' in base or 'wrexec' in base:
        reads.add('exec')
        writes.add('exec')
        may |= EXEC_KEYS
    if (base in SCA and SCA[base][1]) or 'saveexec' in base or 'wrexec' in base:
        writes.add('scc')
    return reads, writes, writes | may | ({'scc'} if base.startswith('s_') else set())

# Forms with no scalar destination: their printed operands are all sources (v_cmpx writes only EXEC, which the
# symbolic tracking does not hold).
NO_SCALAR_DESTINATION = ('s_cbranch', 's_branch', 's_waitcnt', 's_nop', 's_endpgm', 's_cmp', 's_bitcmp', 's_setpc',
                         's_barrier', 's_sendmsg', 's_inst_prefetch', 's_clause', 's_setprio', 's_sleep', 's_setreg',
                         's_code_end', 's_dcache', 's_icache', 's_ttrace', 'exp', 'ds_write', 'v_cmpx')
TWO_SCALAR_DESTINATIONS = ('v_add_co', 'v_sub_co', 'v_subrev_co', 'v_div_scale', 'v_mad_u64_u32', 'v_mad_i64_i32')
# Symbols whose register also holds the value at run time: constants are assigned when set, entry root constants are
# assigned at entry when a join needs it, and the others are assigned when tracked.
RUN_TIME_SYMBOLS = ('const', 'ud', 'execcopy', 'fixed_buffer_word')


def run_time_symbol(sym):
    """True when the register also holds the symbol's value at run time (a constant marked unassigned does not)."""
    return sym[0] in RUN_TIME_SYMBOLS and not (sym[0] == 'const' and len(sym) > 2)


def names_one_value(sym):
    """True when two equal symbols stand for the same value, so a join where every path brings it can keep it.
    A modified descriptor word, descnum, a fixed buffer word, a sampler-heap word and an unplaced entry register
    compare equal for different values (what was written, or where it came from, is not in the tuple). An exec copy
    is taken only while exec holds the entry mask, so equal copies hold the same mask."""
    if sym is None:
        return False
    if sym[0] == 'desc':
        return len(sym) == 4  # ('desc', source, load offset, dword); the 'modified' mark drops the written value
    # An add on a pointer keeps its symbol and marks it 'added' (the offset is not in the tuple), so only an
    # untouched pointer, exactly (kind, source), names one value.
    return sym[0] in ('const', 'ud', 'execcopy') or (sym[0] in ('ptr', 'ptrhi') and len(sym) == 2)


def scalar_destinations(name, asm):
    """The SGPRs (and 'm0') an instruction may write; 'any' for s_movrel*, whose register comes from m0."""
    if name.startswith(NO_SCALAR_DESTINATION) or '_store' in name:
        return set()
    if 'movrel' in name:
        return {'any'}
    ops = parse(asm)[1]
    out = set()
    for tok in ops[:2] if name.startswith(TWO_SCALAR_DESTINATIONS) else ops[:1]:
        if tok == 'm0':
            out.add('m0')
        r = reg_range(tok)
        if r and r[0] == 's':
            out.update(range(r[1], r[2] + 1))
    return out


def desc_key(sym):
    """Resource key of the descriptor a ('desc', source, load offset, dword) word belongs to: the key descriptor()
    gives its first word, so a write into a later descriptor of a wide load marks that descriptor."""
    return sym[1], None if sym[2] is None else sym[2] + 4 * (sym[3] & ~3)


def smem_offset(tok):
    """The SMEM offset operand as a token: 'off' (no offset operand) is the literal 0, anything else unchanged."""
    return '0' if tok == 'off' else tok


def smem_mods(mods):
    """Non-negative 21-bit signed immediate byte offset from SMEM modifiers, refusing unknown or negative forms."""
    imm = None
    for m in mods:
        if m in ('glc', 'dlc'):
            continue
        if m.startswith('offset:') and imm is None and re.fullmatch(r'0[xX][0-9a-fA-F]+|\d+', m[7:]):
            # SMEM OFFSET[20:0] is signed two's complement on RDNA 2; disasm_rdna2 prints w1 & 0x1fffff
            # unsigned, so 0x100000..0x1fffff are negative byte offsets and must not read at +1..2 MiB.
            value = int(m[7:], 0)
            if value > 0xfffff:
                raise Unsupported('SMEM immediate offset %s' % m)
            imm = value
        else:
            raise Unsupported('SMEM modifier %s' % m)
    return imm


class Stage:
def destinations(name, ops):
    """The operands an instruction writes: the first, and the SGPR destination of the VOP3b forms. Compares write none."""
    if not ops or name.startswith(('s_cmp', 's_bitcmp')):
        return []
    return [ops[0]] + ([ops[1]] if len(ops) > 1 and VOP3B_SDST.match(name) else [])


def writes_exec(name, ops):
    """Conservatively: whether the instruction can write EXEC, explicitly or implicitly."""
    for tok in destinations(name, ops):
        r = reg_range(tok)
        if tok in ('exec', 'exec_lo', 'exec_hi') or r and r[0] == 's' and r[1] <= 127 and r[2] >= 126:
            return True
    return 'saveexec' in name or 'wrexec' in name or name.startswith(('v_cmpx', 's_movreld'))


def pc_pair(token):
    """Only aligned pairs of ordinary SGPRs can carry a symbolic code address."""
    r = reg_range(token)
    if r and r[0] == 's' and r[2] == r[1] + 1 <= 105 and r[1] % 2 == 0:
        return r[1]
    return None


def pc_word(token):
    """A displacement word, not a register or an inline floating-point constant."""
    if re.fullmatch(r'-?(?:0x[0-9a-f]+|\d+)', token):
        value = int(token, 16 if '0x' in token else 10)
        if -0x80000000 <= value <= 0xffffffff:
            return ((value & 0xffffffff) ^ 0x80000000) - 0x80000000
    return None


def scalar_registers(text):
    """All explicit SGPR words in an operand, including modifiers and ranges."""
    out = set()
    for low, high, single in re.findall(r'\bs(?:\[(\d+):(\d+)\]|(\d+)\b)', text):
        if single:
            out.add(int(single))
        else:
            out.update(range(int(low), int(high) + 1))
    return out


def reads_scc(name, asm):
    # LLVM AMDGPU SOPInstructions.td lists implicit SCC uses. Cover each supported family here
    # (s_cmov includes s_cmovk_i32) so both the calculated-PC and descriptor checks see new readers.
    return 'src_scc' in asm or name.startswith(('s_cbranch_scc', 's_cmov', 's_cselect', 's_addc', 's_subb'))


def calculated_pc_proof(ins, kind):
    """Return offset -> target for proven SETPCs, and offset -> None for their address arithmetic.

    RDNA 2 GETPC returns the address of its successor. A low-word ADD_U32 and a
    high-word ADDC_U32 with its matching sign extension add one signed byte
    displacement, including any carry across the (unknown) loaded PC's low word.
    A completed pair is therefore the shader's load address plus a known byte
    offset, not an integer value that can be emitted to HLSL.

    At a CFG join retain only pair facts that agree on every incoming edge;
    retain *all* potentially address-dependent words and SCCs. A recognized
    address instruction can be elided only if it is recognized on every path.
    No other operation can observe an elided word or SCC. This also checks loops:
    a newly discovered SETPC successor participates in the same worklist.
    Ordinary reads of those SGPRs also need a single reaching non-PC definition
    matching the definition seen in the linear lowering order: a path around an
    elided write may still hold a symbolic root binding, not its zero-initialized
    HLSL local.
    """
    if not any(name == 's_setpc_b64' for _, _, name, _ in ins):
        return {}
    by_off = {off: (n, name, asm) for off, n, name, asm in ins}
    end = ins[-1][0] + ins[-1][1]
    tracked = {reg for _, _, name, asm in ins if name == 's_getpc_b64' for reg in scalar_registers(asm)}
    entry_defs = dict.fromkeys(tracked, -1)

    def refuse(off, reason):
        raise Unsupported('s_setpc_b64: %s at +0x%x' % (reason, off))

    def boundary(off, target):
        if target % 4:
            refuse(off, 'unaligned calculated branch target %#x' % target)
        if not 0 <= target < end:
            refuse(off, 'target %#x outside decoded program' % target)
        if target not in by_off:
            refuse(off, 'target %#x in the interior of an instruction or undecoded bytes' % target)

    if 0 not in by_off:
        refuse(0, 'missing entry instruction')
    if not any(name == 's_endpgm' for _, _, name, _ in ins):
        refuse(0, 'decoded program has no s_endpgm')

    def ordinary_accesses(name, ops, mods, refs):
        # These scalar operations and loads have explicit unconditional destinations.
        # Otherwise conservatively treat a mentioned PC word as a read, and any
        # possible SGPR destination as an invalidation of its reaching definition.
        written = (scalar_registers(ops[0]) if ops and
                   (name in SCA or name.startswith(('s_load_dword', 's_buffer_load_dword'))
                    or name in ('s_cselect_b32', 's_cselect_b64')) else set())
        reads = scalar_registers(', '.join(ops[1:]) + ' ' + ' '.join(mods)) if written else refs
        maybe_written = {reg for dst in destinations(name, ops) for reg in scalar_registers(dst)} - written
        return written, reads, maybe_written

    # Each state is (possibly elided SGPR words, definitely proven pairs, SCC,
    # definitely reaching ordinary definitions of SGPRs the lowering can erase).
    # pairs[low] = (target, None) for a complete pair, or (target, high immediate)
    # for a partial low-word sum. SCC is None for ordinary state, the pair's low
    # register for the corresponding carry, or -1 for an elided/unusable flag.
    incoming = {0: (set(), {}, None, entry_defs)}
    pending = [0]
    lowered = {}

    def propagate(off, words, pairs, scc, definitions):
        state = incoming.get(off)
        if state is None:
            merged = (set(words), dict(pairs), scc, dict(definitions))
        else:
            old_words, old_pairs, old_scc, old_defs = state
            merged = (old_words | words, {p: v for p, v in old_pairs.items() if pairs.get(p) == v},
                      old_scc if old_scc == scc else -1,
                      {reg: d for reg, d in old_defs.items() if definitions.get(reg) == d})
        if state != merged:
            incoming[off] = merged
            pending.append(off)

    while pending:
        off = pending.pop()
        n, name, asm = by_off[off]
        _, ops, mods = parse(asm)
        words, pairs, scc, definitions = incoming[off]
        words, pairs, definitions = set(words), dict(pairs), dict(definitions)
        refs = scalar_registers(asm)
        successors = [off + n]

        def clobber(written):
            nonlocal scc
            words.difference_update(written)
            for reg in written:
                definitions.pop(reg, None)
            for p in list(pairs):
                if p in written or p + 1 in written:
                    del pairs[p]
                    if scc == p:
                        scc = -1

        if name == 's_getpc_b64':
            p = pc_pair(ops[0]) if len(ops) == 1 else None
            if p is None or mods or n != 4:
                refuse(off, 'unsupported GETPC destination/form')
            clobber({p, p + 1})
            words.update((p, p + 1))
            pairs[p] = (off + n, None)
            lowered[off] = None
        elif name in ('s_add_u32', 's_addc_u32') and words & refs:
            row = reg_range(ops[0]) if ops else None
            p = row[1] if row and row[0] == 's' and row[1] == row[2] else None
            imm = pc_word(ops[2]) if len(ops) == 3 else None
            if p is None or imm is None or mods or n not in (4, 8) or ops[0] != ops[1]:
                refuse(off, 'unsupported arithmetic on calculated PC words')
            if name == 's_add_u32':
                fact = pairs.get(p)
                if fact is None or fact[1] is not None:
                    refuse(off, 'unknown or ambiguous input pair for calculated PC low-word addition')
                pairs[p] = (fact[0] + imm, -1 if imm < 0 else 0)
                scc = p
            else:
                fact = pairs.get(p - 1)
                if fact is None or fact[1] is None or scc != p - 1:
                    refuse(off, 'missing matching low-word carry for calculated PC high-word addition')
                if imm != fact[1]:
                    refuse(off, 'calculated PC high word is not the signed displacement extension')
                pairs[p - 1] = (fact[0], None)
                scc = -1
            lowered[off] = None
        elif name == 's_setpc_b64':
            p = pc_pair(ops[0]) if len(ops) == 1 else None
            if p is None or mods or n != 4:
                refuse(off, 'unsupported SETPC source/form')
            fact = pairs.get(p)
            if fact is None:
                refuse(off, 'unknown or ambiguous calculated target from %s' % ops[0])
            target, high = fact
            if high is not None:
                refuse(off, 'incomplete calculated target (missing high-word addition with carry)')
            boundary(off, target)
            lowered[off] = target
            successors = [target]
        else:
            written, reads, maybe_written = ordinary_accesses(name, ops, mods, refs)
            if words & reads or words and name.startswith(('s_movrels', 's_movreld')):
                refuse(off, 'calculated PC word escapes into %s' % name)
            clobber(written)
            definitions.update((reg, off) for reg in written & tracked)
            for reg in maybe_written:
                definitions.pop(reg, None)
            if scc is not None:
                if reads_scc(name, asm):
                    refuse(off, 'calculated PC SCC escapes into %s' % name)
                # These comparisons always emit a replacement SCC (or refuse).
                # Other scalar writers may not model SCC, so they cannot make
                # a tainted flag usable; they also invalidate any pending carry.
                if re.fullmatch(r's_cmpk?_(eq|lg|gt|ge|lt|le)_[iu]32|s_bitcmp[01]_b32', name):
                    scc = None
                elif not name.startswith(('v_', 's_branch', 's_cbranch_', 's_mov_', 's_movk_', 's_load_',
                                          's_buffer_load_', 's_nop', 's_waitcnt', 's_inst_prefetch', 's_clause',
                                          'buffer_', 'tbuffer_', 'image_', 'exp', 'ds_')):
                    scc = -1

            if name in ('s_swappc_b64', 's_call_b64', 's_rfe_b64'):
                # The only pre-existing exception is the vertex entry fetch call.
                if not (name == 's_swappc_b64' and kind == 'vs' and off == 0
                        and ops == ['s[4:5]', 's[4:5]'] and not mods):
                    refuse(off, 'unsupported control transfer %s' % name)
            if name == 's_endpgm':
                successors = []
            elif name == 's_branch' or name.startswith('s_cbranch_'):
                try:
                    target = int(ops[0].split('_')[1], 16)
                except (IndexError, ValueError):
                    refuse(off, 'unsupported direct branch operand')
                boundary(off, target)
                successors = [target] if name == 's_branch' else [target, off + n]

        for successor in successors:
            if successor not in by_off:
                refuse(off, 'missing decoded successor %#x' % successor)
            propagate(successor, words, pairs, scc, definitions)

    for off, _, name, _ in ins:
        if name == 's_setpc_b64' and off not in lowered:
            refuse(off, 'unreachable SETPC has no proven target')

    # Stage's existing resource/constant symbol table is intentionally linear.
    # Do not substitute a symbolic address into it or pretend an erased symbol
    # was a runtime word. Every later ordinary read must see the same *ordinary*
    # definition in both the listing and on all reachable paths. Include writes
    # in unreachable code when tracking the listing because Stage lowers those too.
    linear_defs = dict(entry_defs)
    for off, _, name, asm in ins:
        _, ops, mods = parse(asm)
        if off in lowered:
            if lowered[off] is None:
                for reg in scalar_registers(ops[0]):
                    linear_defs.pop(reg, None)
            continue
        written, reads, maybe_written = ordinary_accesses(name, ops, mods, scalar_registers(asm))
        if off in incoming:
            reaching = incoming[off][3]
            for reg in sorted(reads & tracked):
                if reg not in linear_defs or reaching.get(reg) != linear_defs[reg]:
                    refuse(off, 'elided PC register s%d has no matching ordinary definition on every incoming path' % reg)
        for reg in maybe_written:
            linear_defs.pop(reg, None)
        linear_defs.update((reg, off) for reg in written & tracked)
    return lowered


def _lds_single(tok):
    """Single hardware VGPR vN (v0..v255) for bounded LDS exchange validation."""
    m = re.fullmatch(r'v(\d+)', tok)
    return int(m.group(1)) if m and int(m.group(1)) < 256 else None


def _lds_pair(tok):
    """Exact contiguous 2-VGPR pair v[n:n+1] (v0..v255) for bounded 64-bit LDS exchanges."""
    m = re.fullmatch(r'v\[(\d+):(\d+)\]', tok)
    if not m:
        return None
    lo, hi = int(m.group(1)), int(m.group(2))
    return lo if hi == lo + 1 and hi < 256 else None


def lds_byte_offsets(name, mods):
    if name != 'ds_read2_b32':
        if not mods:
            return (0,)
        if len(mods) == 1 and re.fullmatch(r'offset:\d+', mods[0]):
            return (int(mods[0][7:]),)
        raise Unsupported('LDS exchange needs two VGPRs and an optional byte offset')
    # DS_READ2 encodes each offset independently in dwords. An omitted field is zero.
    offsets = {}
    for mod in mods:
        field = re.fullmatch(r'offset([01]):(\d+)', mod)
        if not field or field[1] in offsets or int(field[2]) > 255:
            raise Unsupported('LDS read2 needs ordinary dword offsets')
        offsets[field[1]] = 4 * int(field[2])
    first, second = offsets.get('0', 0), offsets.get('1', 0)
    if first == second:
        raise Unsupported('LDS read2 needs two distinct offsets')
    return first, second


def check_lds_exchange(ins, group, words):
    """Prove one race-free 32-bit or 64-bit fill of a contiguous LDS span, one uniform barrier, then reads of initialized words.

    RDNA 2 ISA (Ch. 9 DS): ds_write_b32 / ds_read_b32 / ds_read2_b32 access 4-byte dwords and
    ds_write_b64 / ds_read_b64 access two contiguous little-endian dwords (v[n] at addr+offset, v[n+1] at addr+offset+4).
    LLVM 15 AMDGPUUsage defines COMPUTE_PGM_RSRC2.GRANULATED_LDS_SIZE in 128-dword (512-byte) units on gfx10.
    """
    if not 0 < words <= 8192 or group[1:] != (1, 1):
        raise Unsupported('LDS exchange needs a 1D group and a nonzero allocation of at most 32 KiB')
    first = None
    elem_bytes = None
    published = False
    read = False
    prev = (None, [], [])
    for i, (_, _, _, asm) in enumerate(ins):
        name, ops, mods = parse(asm)
        base = name.removesuffix('_e32').removesuffix('_e64')
        if (base.startswith(('s_branch', 's_cbranch', 's_setpc', 's_swappc')) or writes_exec(base, ops)
                or base == 's_endpgm' and i + 1 < len(ins)):
            raise Unsupported('LDS exchange needs straight-line code and entry EXEC')
        if base == 's_barrier':
            if (first is None or published or prev != ('s_waitcnt', ['lgkmcnt(0)'], [])
                    or mods or ops not in ([], ['0'], ['0x0'])):
                raise Unsupported('LDS exchange needs one barrier after a completed fill')
            published = True
        elif base in ('ds_write_b32', 'ds_read_b32', 'ds_read2_b32', 'ds_write_b64', 'ds_read_b64'):
            write = base.startswith('ds_write')
            width = 8 if base.endswith('_b64') else 4
            addr_op = ops[0] if (write and len(ops) == 2) else (ops[1] if len(ops) == 2 else '')
            val_op = ops[1] if (write and len(ops) == 2) else (ops[0] if len(ops) == 2 else '')
            if base == 'ds_read2_b32':
                if len(ops) != 2 or _lds_pair(val_op) is None or _lds_single(addr_op) is None:
                    raise Unsupported('LDS read2 needs a VGPR pair and a byte address VGPR')
            elif (len(ops) != 2 or _lds_single(addr_op) is None
                  or (_lds_pair(val_op) is None if width == 8 else _lds_single(val_op) is None)):
                raise Unsupported('LDS exchange needs %s, %s and an optional byte offset'
                                  % ('two VGPRs' if width == 4 else 'a VGPR address',
                                     'an aligned 2-VGPR pair' if width == 8 else 'a matching data VGPR'))
            offsets = lds_byte_offsets(base, mods)
            producer, args, modifiers = prev
            if write:
                offset, = offsets
                shift = '3' if width == 8 else '2'
                # The prefix below preserves the entry X ID in v0. Each lane writes exactly one distinct element.
                # 8-byte alignment is not required for LDS dword-addressed writes.
                if (first is not None or producer != 'v_lshlrev_b32' or args != [addr_op, shift, 'v0']
                        or modifiers or offset % 4 or offset + group[0] * width > words * 4):
                    raise Unsupported('LDS exchange needs one bounded %s per local X ID'
                                      % ('qword' if width == 8 else 'dword'))
                first = offset
                elem_bytes = width
            else:
                if not published or modifiers or not args or args[0] != addr_op:
                    raise Unsupported('LDS read needs a bounded address after the fill barrier')
                if elem_bytes != width:
                    raise Unsupported('LDS read width (%dB) does not match initialized fill (%dB)'
                                      % (width, elem_bytes or 0))
                if producer == 'v_mov_b32' and len(args) == 2 and re.fullmatch(r'0x[0-9a-f]+|\d+', args[1]):
                    low = high = int(args[1], 0)
                elif (producer == 'v_and_b32' and len(args) == 3 and re.fullmatch(r'0x[0-9a-f]+|\d+', args[1])
                      and _lds_single(args[2]) is not None):
                    low, high = 0, int(args[1], 0)
                else:
                    raise Unsupported('LDS read needs a literal or literal-masked byte address')
                # Every byte offset in [a + offset, a + offset + width) must be 4-byte aligned and inside the filled span.
                if (low % 4 or high % 4
                        or any(offset % 4 or low + offset < first or high + offset + (width - 4) >= first + group[0] * width
                               for offset in offsets)):
                    raise Unsupported('LDS read outside the aligned, initialized span')
                if i + 1 >= len(ins) or ins[i + 1][3] != 's_waitcnt lgkmcnt(0)':
                    raise Unsupported('LDS read needs a completed LGKM wait')
                read = True
        elif base.startswith('ds_') or 'lds' in mods:
            raise Unsupported('LDS exchange does not cover ' + asm)
        elif first is None and not base.startswith('s_'):
            # Simple VALU instructions only write their single destination. Other forms, e.g. v_swap_b32,
            # can also change a source. After the fill, each read proves its own address without the entry X ID.
            if (base not in VEC_U and base not in VEC_F and base != 'v_mov_b32'
                    or not ops or not re.fullmatch(r'v[0-9]+', ops[0]) or ops[0] == 'v0'):
                raise Unsupported('LDS fill needs the unchanged entry local X ID')
        prev = base, ops, mods
    if not read:
        raise Unsupported('LDS exchange needs a fill, a barrier and a completed read')


# ---- slot 57 mesh stage: the group-memory count exchange (#2413) --------------------------------
# A group is two wave64 waves and thread = wave * 64 + lane. 128 is also D3D12's limit for the numthreads of a mesh
# shader, and the declared size of both output arrays.
MESH_THREADS = 128
MESH_VREG = re.compile(r'\bv(?:\[(\d+):(\d+)\]|(\d+))')
MESH_EXCHANGE = [
    r's_mov_b64 exec, 1',
    r's_mov_b64 (?P<lane>s\[\d+:\d+\]), exec',
    r's_and_b32 (?P<a_bits>s\d+), s3, 0xf000000',
    r's_cbranch_scc1 label_(?P<a_other>\w+)',
    r'v_lshrrev_b32_e32 v5, 7, v5',
    r'v_mov_b32_e32 (?P<a_zero>v\d+), 0',
    r'ds_write_b32 {a_zero}, v5 offset:8',
    r's_waitcnt_depctr 0xffe3',
    r's_barrier 0x0',
    r's_branch label_(?P<a_join>\w+)',
    r's_barrier 0x0',
    r'v_mov_b32_e32 (?P<a_zero2>v\d+), 0',
    r'ds_read_b32 v5, {a_zero2} offset:8',
]
MESH_COUNTS = [
    r's_barrier 0x0',
    r's_cmp_eq_i32 (?P<wave>s\d+), 0',
    r's_cbranch_scc0 label_(?P<b_reader>\w+)',
    r's_mov_b64 exec, {lane}',
    r's_cbranch_execz label_(?P<b_wskip>\w+)',
    r'v_mov_b32_e32 (?P<b_p>v\d+), (?P<prims>s\d+)',
    r'v_mov_b32_e32 (?P<b_zero>v\d+), 0',
    r'v_mov_b32_e32 (?P<b_v>v\d+), (?P<verts>s\d+)',
    r'ds_write_b32 {b_zero}, {b_p}',
    r'ds_write_b32 {b_zero}, {b_v} offset:4',
    r's_mov_b64 exec, -1',
    r's_waitcnt_depctr 0xffe3',
    r's_barrier 0x0',
    r's_lshl_b32 (?P<b_tmp>s\d+), {prims}, 12',
    r's_or_b32 m0, {b_tmp}, {verts}',
    r's_sendmsg 0x9',
    r's_branch label_(?P<b_join>\w+)',
    r's_barrier 0x0',
    r's_mov_b64 exec, {lane}',
    r's_cbranch_execz label_(?P<b_rskip>\w+)',
    r'v_mov_b32_e32 (?P<b_zero2>v\d+), 0',
    r'ds_read_b32 (?P<b_r0>v\d+), {b_zero2}',
    r'ds_read_b32 (?P<b_r1>v\d+), {b_zero2} offset:4',
    r's_waitcnt lgkmcnt\(1\)',
    r'v_readfirstlane_b32 {prims}, {b_r0}',
    r's_waitcnt lgkmcnt\(0\)',
    r'v_readfirstlane_b32 {verts}, {b_r1}',
    r's_mov_b64 exec, -1',
]


def mesh_vregs(tokens):
    out = set()
    for m in MESH_VREG.finditer(' '.join(tokens)):
        out.update(range(int(m[1] or m[3]), int(m[2] or m[3]) + 1))
    return out


def mesh_run(ins, at, what, patterns, env):
    """Match consecutive instructions; {name} in a pattern is the text an earlier (?P<name>...) bound."""
    for k, pattern in enumerate(patterns):
        text = ins[at + k][3] if at + k < len(ins) else ''
        m = re.fullmatch(re.sub(r'\{(\w+)\}', lambda ref: re.escape(env[ref[1]]), pattern), text)
        if not m:
            raise Unsupported('slot 57 mesh %s: instruction %d of the accepted shape (%s) is absent'
                              % (what, k + 1, pattern.split()[0]))
        env.update(m.groupdict())
    return at + len(patterns)


def mesh_skeleton(ins):
    """Prove the one accepted group-memory shape of a slot 57 mesh program and replace it by per-thread instructions.

    Accepted shape, in program order (anything else is refused with its own message):
    - Exchange: exec = 1 (lane 0 of each wave), the mask saved in an SGPR pair; wave 0 (s3 bits 24..27 zero) stores
      entry v5 >> 7 at byte 8, waits and passes a barrier; every other wave passes a barrier and reads byte 8 to v5.
    - Straight-line code up to the next barrier. Its only read of v5 may be one v_readfirstlane_b32 under exec = -1.
    - Counts: one barrier; the wave whose index (s_bfe_u32 of s3, 4 bits at 24) is zero stores SGPR P at byte 0 and
      SGPR V at byte 4 under the saved lane-0 mask, waits, passes a barrier and sends one allocation request with
      m0 = P << 12 | V; every other wave passes a barrier and reads both words back into the same P and V.
    - Straight-line code to s_endpgm. Exports only here: pos0 and param under exec = -1 narrowed by one
      v_cmpx_gt_i32 V, thread; the primitive export likewise with P. thread is the v_mbcnt pair over wave << 6.
    Each path therefore passes three barriers, each store has one writer (one lane of one wave), and a wait and a
    barrier separate it from every read. The lowering keeps one writer and one barrier: thread 0 stores, all read.

    Registers the two arms leave different, that the lowering does not compute, or that hold an entry value the
    model does not define, are refused when read: VGPRs by the scan below, SGPRs by a symbol (setup_entry and the
    op_x2w_mesh_* handlers), SCC and m0 after the counts. Of s3 only the wave index is modelled.
    """
    env = {}
    start = 1 if ins and ins[0][2] == 's_inst_prefetch' else 0
    a_end = mesh_run(ins, start, 'group-index exchange', MESH_EXCHANGE, env)
    count_at = next((i for i in range(a_end, len(ins)) if ins[i][2] == 's_barrier'), len(ins))
    b_end = mesh_run(ins, count_at, 'count exchange', MESH_COUNTS, env)

    def lands(key, index, what):
        if index >= len(ins) or int(env[key], 16) != ins[index][0]:
            raise Unsupported('slot 57 mesh %s: a branch leaves the accepted shape' % what)
    lands('a_other', start + 10, 'group-index exchange')
    lands('a_join', a_end, 'group-index exchange')
    lands('b_reader', count_at + 17, 'count exchange')
    lands('b_wskip', count_at + 10, 'count exchange')
    lands('b_join', b_end, 'count exchange')
    lands('b_rskip', count_at + 27, 'count exchange')
    lane = set(range(*(lambda r: (r[1], r[2] + 1))(reg_range(env['lane']))))
    sreg = {key: int(env[key][1:]) for key in ('wave', 'prims', 'verts', 'b_tmp')}
    vreg = {key: int(env[key][1:]) for key in ('a_zero', 'a_zero2', 'b_p', 'b_zero', 'b_v', 'b_zero2', 'b_r0', 'b_r1')}
    if (len({vreg['b_p'], vreg['b_zero'], vreg['b_v']}) != 3 or len({vreg['b_r0'], vreg['b_r1']}) != 2
            or vreg['b_r0'] == vreg['b_zero2'] or 5 in (vreg['a_zero'], vreg['a_zero2'])
            or len({sreg['prims'], sreg['verts'], sreg['b_tmp']}) != 3 or {sreg['prims'], sreg['verts']} & lane
            or sreg['wave'] in lane | {3, sreg['prims'], sreg['verts']} or int(env['a_bits'][1:]) in lane):
        raise Unsupported('slot 57 mesh count exchange: registers of the accepted shape overlap')
    if not ins or ins[-1][2] != 's_endpgm':
        raise Unsupported('slot 57 mesh program does not end with s_endpgm')

    out = list(ins[:start])
    span = lambda first, last: (ins[first][0], ins[last][0] + ins[last][1] - ins[first][0])
    out.append(span(start, start + 1) + ('x2w_mesh_lane0', 'x2w_mesh_lane0 ' + env['lane']))
    out.append(ins[start + 2])
    out.append(span(start + 3, a_end - 1) + ('s_nop', 's_nop 0'))
    # State of the walk over the code outside the two exchanges.
    stale = set(range(256))   # VGPRs the model does not hold for every lane; no entry VGPR is modelled
    pending = set()           # of those, the ones written only for the lanes of a narrowed exec
    mask = 'lane0'         # 'lane0', None (all lanes), 'verts' or 'prims' (all lanes narrowed by one compare)
    wave_known = False     # sreg['wave'] holds s_bfe_u32(s3, 4 bits at 24)
    thread = base = None   # VGPR of the thread index, SGPR holding wave << 6
    scc_stale = False
    exports = []
    for index in list(range(a_end, count_at)) + [None] + list(range(b_end, len(ins) - 1)):
        if index is None:
            if not wave_known:
                raise Unsupported('slot 57 mesh count exchange: the compared register is not the wave index')
            out.append(span(count_at, b_end - 2) + (
                'x2w_mesh_counts', 'x2w_mesh_counts %s, %s, %s' % (env['prims'], env['verts'], env['b_tmp'])))
            out.append(ins[b_end - 1])
            stale |= pending | {vreg[k] for k in ('b_p', 'b_zero', 'b_v', 'b_zero2', 'b_r0', 'b_r1')}
            pending.clear()
            mask, scc_stale = None, True
            continue
        off, size, name, asm = ins[index]
        after = index >= b_end
        _, ops, _ = parse(asm)
        base_name = re.sub(r'_e(32|64)$', '', name)
        if name.startswith(('s_cbranch', 's_branch', 's_setpc', 's_call', 's_swappc')):
            raise Unsupported('slot 57 mesh: a branch outside the group-memory exchanges is not covered')
        if name.startswith('ds_') or name in ('s_barrier', 's_sendmsg', 's_sendmsghalt'):
            raise Unsupported('slot 57 mesh: %s outside the group-memory exchanges' % name)
        if name.startswith(('v_readlane', 'v_writelane')):
            raise Unsupported('slot 57 mesh: lane spills are not covered')
        keep = ins[index]
        if base_name == 'v_readfirstlane_b32':
            if after or asm != 'v_readfirstlane_b32 %s, v5' % ops[0] or mask is not None or 5 not in stale:
                raise Unsupported('slot 57 mesh: v_readfirstlane_b32 outside the group-index read is not covered')
            keep = (off, size, 'x2w_mesh_group', 'x2w_mesh_group ' + ops[0])
            ops = ops[:1]
        writes_s = set()
        if not base_name.startswith(('v_cmpx', 'exp', 'ds_write', 'buffer_store')):
            for tok in destinations(name, ops):
                r = reg_range(tok)
                if r and r[0] == 's':
                    writes_s.update(range(r[1], r[2] + 1))
        if 3 in writes_s or writes_s & lane:
            raise Unsupported('slot 57 mesh: the wave word s3 or the saved lane-0 mask is written')
        if writes_s and name.startswith('v_') and keep[2] != 'x2w_mesh_group':
            raise Unsupported('slot 57 mesh: %s into an SGPR is not covered' % base_name)
        # Only the wave index of s3 is modelled.
        if not re.fullmatch(r's_bfe_u32 s\d+, s3, 0x40018', asm) and any(
                r and r[0] == 's' and r[1] <= 3 <= r[2] for r in map(reg_range, ops)):
            raise Unsupported('slot 57 mesh: s3 is read other than as the wave index')
        if after and writes_s & {sreg['prims'], sreg['verts']}:
            raise Unsupported('slot 57 mesh: an output count is written after the count exchange')
        if after and re.search(r'\bm0\b', asm):
            raise Unsupported('slot 57 mesh: m0 is used after the allocation request')
        if scc_stale and reads_scc(name, asm):
            raise Unsupported('slot 57 mesh: SCC of the count exchange is read')
        if name.startswith(('s_cmp', 's_bitcmp')):
            scc_stale = False
        # EXEC and VCC as wave-wide masks are outside the per-thread model here: only the two forms below touch EXEC.
        if asm == 's_mov_b64 exec, -1':
            mask = None
            stale |= pending
            pending.clear()
        elif base_name.startswith('v_cmpx'):
            count = {env['verts']: 'verts', env['prims']: 'prims'}.get(ops[0] if ops else None)
            if (base_name != 'v_cmpx_gt_i32' or not after or mask is not None or count is None
                    or thread is None or ops[1:] != ['v%d' % thread]):
                raise Unsupported('slot 57 mesh: an EXEC compare other than output count > thread index under '
                                  'exec = -1 is not covered')
            mask = count
        elif writes_exec(name, ops) or re.search(r'\bexec(_lo|_hi)?\b', asm) or (
                name.startswith('s_') and re.search(r'\bvcc(_lo|_hi)?\b', asm)):
            raise Unsupported('slot 57 mesh: %s on a wave mask is not covered' % base_name)
        # The thread index: v_mbcnt_lo(-1, wave << 6) then v_mbcnt_hi(-1, that) under exec = -1.
        if asm == 's_bfe_u32 %s, s3, 0x40018' % env['wave']:
            wave_known = True
        elif sreg['wave'] in writes_s:
            wave_known = False
        if wave_known and re.fullmatch(r's_lshl_b32 s\d+, %s, 6' % env['wave'], asm) and ops[0] != env['wave']:
            base = int(ops[0][1:])
        elif base in writes_s:
            base = None
        reads_v = mesh_vregs(ops if base_name.startswith(('v_cmp', 'exp', 'ds_write', 'buffer_store')) else ops[1:])
        dest = reg_range(ops[0]) if ops and not base_name.startswith(('v_cmp', 'exp', 'buffer_store')) else None
        writes_v = set(range(dest[1], dest[2] + 1)) if dest and dest[0] == 'v' else set()
        if base_name == 'v_mbcnt_lo_u32_b32' and base is not None and ops[1:] == ['-1', 's%d' % base] and mask is None \
                and index + 1 < len(ins) and ins[index + 1][3] == 'v_mbcnt_hi_u32_b32 %s, -1, %s' % (ops[0], ops[0]):
            thread = dest[1]
        elif thread in writes_v and not (base_name == 'v_mbcnt_hi_u32_b32' and ins[index - 1][2].startswith('v_mbcnt_lo')):
            raise Unsupported('slot 57 mesh: the thread index register is rewritten')
        if keep[2] != 'x2w_mesh_group' and reads_v & stale:
            raise Unsupported('slot 57 mesh: v%d is read where it is not defined for every lane (an entry value, or '
                              'left different per wave by an exchange)' % min(reads_v & stale))
        if mask is not None:
            pending |= writes_v & stale
        else:
            pending -= writes_v
        stale -= writes_v
        if name == 'exp':
            target = asm.split()[1].rstrip(',')
            want = 'prims' if target == 'target_20' else 'verts' if re.fullmatch(r'pos0|param\d+', target) else None
            if want is None:
                raise Unsupported('slot 57 mesh: export target %s is not covered' % target)
            if not after or mask != want:
                raise Unsupported('slot 57 mesh: export %s is not under exec = -1 narrowed by its output count > '
                                  'thread index' % target)
            exports.append(target)
        out.append(keep)
    if exports.count('pos0') != 1 or exports.count('target_20') != 1:
        raise Unsupported('slot 57 mesh: one pos0 and one primitive export are required')
    out.append(ins[-1])
    return out


class Stage:
    """Translates one stage. Shares the resource and entry-constant registries of its pipeline."""

    # Immutable fallbacks for unit probes that bypass __init__; __init__ and translate() assign fresh
    # per-instance sets, so nothing is shared across Stage instances.
    needs_sm6 = False         # the lowering uses a shader model 6.0 intrinsic, so the stage cannot compile as 5.1
    exec_known = False        # unit probes without __init__ customize this explicitly when needed
    low16_only_v = ()         # VGPRs whose upper 16 bits are undefined after odd-dmask d16 image_load
    pinned_low16_v = ()       # low16_only_v live across basic-block boundaries
    whole_read_v = ()         # VGPRs read outside a proven low-16-bit consumer
    used_atomic = ()          # atomic result types ('uint', 'int') needing declarations in main()
    # pixel: VGPR -> (attr, channel, high) of the v_interp_p1ll_f16 that wrote it. Created on first use, because
    # unit probes build a Stage without __init__.
    interp_p1 = None
    # The whole-program proof of interp_f16_scan depends only on the decoded program, so it runs once per Stage
    # (a class default keeps probes built without __init__ working).
    interp_scanned = False
    defined_uniform_s = None
    bad_sgprs = None
    _uniform_written = None

    def __init__(self, pipe, kind, code, refusals=None, decoded=None):
        self.pipe, self.kind = pipe, kind
        self.ins = decode(strip_header(code))
        self.lines = []           # body statements (per block)
        self.sym = {}             # scalar register -> symbolic value
        self.used_s, self.used_v = set(), set()
        self.exec_known = True    # exec still equals the entry mask (the lane is active)
        self.exports = {}         # 'pos0' / 'paramN' / 'mrtN' -> [4 expressions]
        self.attrs = {}           # pixel: attr index -> 'linear' | 'flat'
        self.uses_pos = False
        self.masks = set()        # scalar registers currently holding lane masks
        self.fetched = False      # vertex: the entry call into the fetch routine was translated as loads
        self.setup_entry()
        self.translate()

    # ---- entry state ---------------------------------------------------------------------
    def setup_entry(self):
        root = self.pipe.root
        reg = 8
        for index, p in enumerate(root['params']):
            if p['type'] == 0:
                self.sym[reg] = ('ptr', ('table', index))
                reg += 1
            elif p['type'] == 1:
                for i in range(p['count']):
                    self.sym[reg] = ('ud', self.pipe.ud(('rootconst', index, i)))
                    reg += 1
            else:
                self.sym[reg] = ('ptr', ('root', index))
                self.sym[reg + 1] = ('ptrhi', ('root', index))
                reg += 2
        if root['flags'] & 0x400:
            self.sym[reg] = ('ptr', ('resheap',))
            reg += 1
        if root['flags'] & 0x800:
            self.sym[reg] = ('ptr', ('smpheap',))
            reg += 1
        if self.kind == 'vs':
            self.sym[0] = ('ptr', ('vbtab',))
            self.sym[1] = ('ptrhi', ('vbtab',))
            self.sym[2] = ('ud', self.pipe.ud(('basevertex',)))
            self.sym[3] = ('ud', self.pipe.ud(('startinstance',)))
        else:
            self.sym[reg] = ('const', 0)  # primitive mask: only ever moved to m0

    # ---- operand helpers ---------------------------------------------------------------------
    def S(self, n):
        """Runtime expression of scalar register n (uint)."""
        s = self.sym.get(n)
        if s:
            if s[0] == 'const':
                return '0x%08xu' % (s[1] & 0xffffffff)
            if s[0] == 'ud':
                return 'ud%d' % s[1]
            if s[0] in ('execcopy', 'fixed_buffer_word'):
                self.used_s.add(n)
                return 's%d' % n
            raise Unsupported('numeric use of s%d holding %s' % (n, s[0]))
        self.used_s.add(n)
        return 's%d' % n

    def u(self, tok):
        """uint expression for a 32-bit operand."""
        if tok in ('vcc_lo', 'vcc'):
            return 'vcc'
        if tok in ('exec_lo', 'exec'):
            return 'exec'
        if tok == 'm0':
            m = self.sym.get('m0')
            if m:
                raise Unsupported('numeric use of m0 holding %s' % m[0])
            return 'm0'
        r = reg_range(tok)
        if r:
            if r[0] == 'v':
                self.used_v.add(r[1])
                return 'v%d' % r[1]
            return self.S(r[1])
        if tok in FLOATS:
            return '0x%08xu' % fbits(FLOATS[tok])
        try:
            return '0x%08xu' % (int(tok, 0) & 0xffffffff)
        except ValueError:
            raise Unsupported('operand %r' % tok)

    def u_src(self, tok):
        """uint expression for an integer-context source that the disassembler may print negated."""
        # In an integer instruction the `-` modifier is the source's sign bit flipped: the same bit
        # pattern a float negation produces. Only handlers that model it ask for this; other integer
        # sources keep using u(), so an unexpected negation stays a refusal rather than a silent flip.
        if tok.startswith('-') and not re.match(r'-[\d.]', tok):
            return '(%s ^ 0x80000000u)' % self.u(tok[1:])
        return self.u(tok)

    def f(self, tok):
        """float expression, with the inline neg/abs operand modifiers."""
        neg = False
        if tok.startswith('-') and not re.match(r'-[\d.]', tok):
            neg, tok = True, tok[1:]
        absolute = tok.startswith('|') and tok.endswith('|')
        if absolute:
            tok = tok[1:-1]
        if tok in FLOATS:
            text = repr(FLOATS[tok])
        elif tok == '-0':
            text = '0.0'
        else:
            text = 'F(%s)' % self.u(tok)
        if absolute:
            text = 'abs(%s)' % text
        return '(-%s)' % text if neg else text

    def is_const(self, n):
        s = self.sym.get(n)
        return s[1] if s and s[0] == 'const' else None

    def emit(self, text):
        self.lines.append(text)

    def clear_low16(self, n):
        if isinstance(self.low16_only_v, set) and self.exec_known and n not in self.pinned_low16_v:
            self.low16_only_v.discard(n)

    def set_v(self, n, expr, *, low16_only=False):
        if getattr(self, 'interp_p1', None):
            self.interp_p1.pop(n, None)
        if hasattr(self, 'uniform_v'):
            self.uniform_v.discard(n)
        self.last_v = (n, expr)
        if low16_only:
            if not isinstance(self.low16_only_v, set):
                self.low16_only_v = set()
            self.low16_only_v.add(n)
        else:
            self.clear_low16(n)
        self.used_v.add(n)
        self.emit('v%d = %s;' % (n, expr) if self.exec_known else 'if (exec & 1) v%d = %s;' % (n, expr))

    def set_vf(self, n, expr, mods=()):
        # The hardware applies the output modifier before the clamp, but the disassembler prints `clamp`
        # first, so the order of `mods` is not the order of the operations: scale first, saturate last.
        scale = 1.0
        clamp = False
        for m in mods:
            if m == 'clamp':
                clamp = True
            elif m in ('mul:2', 'mul:4'):
                scale *= float(m[-1])
            elif m == 'div:2':
                scale *= 0.5
            elif m.startswith('op_sel'):
                pass
            else:
                raise Unsupported('modifier %s' % m)
        if scale != 1.0:
            factor = ('%d.0' % int(scale)) if scale == int(scale) else ('%g' % scale)
            expr = '(%s) * %s' % (expr, factor)
        if clamp:
            expr = 'saturate(%s)' % expr
        self.set_v(n, 'U(%s)' % expr)

    def set_s(self, n, expr, sym=None, scc=None):
        """Assign scalar register n at run time and/or symbolically."""
        if sym is not None:
            self.sym[n] = sym
            if sym[0] != 'const':
                return
        else:
            self.sym.pop(n, None)
        self.used_s.add(n)
        self.emit('s%d = %s;' % (n, expr))
        if scc:
            self.emit('scc = %s;' % scc.replace('$', 's%d' % n))

    def descriptor(self, tok):
        """Resource key for a descriptor operand like s[8:11]: (source, byte offset or None)."""
        r = reg_range(tok)
        s = self.sym.get(r[1]) if r else None
        if s and s[0] == 'desc':
            # A descriptor can start inside a wider load (s_load_dwordx16 fetches two image
            # descriptors): its offset is the load's offset plus the first register's dword index.
            return s[1], (None if s[2] is None else s[2] + 4 * s[3])
        if s and s[0] == 'ptr':
            return s[1], None
        raise Unsupported('descriptor operand %s holds %s' % (tok, s))

    def heap_sampler(self, tok):
        """'qh[...]' when the four sampler words of tok came from one run-time sampler heap load."""
        r = reg_range(tok)
        if not r or r[2] - r[1] != 3:
            return None
        first = self.sym.get(r[1])
        if not first or first[0] != 'smpheap':
            return None
        if any(self.sym.get(r[1] + i) != ('smpheap', first[1], i) for i in range(4)):
            raise Unsupported('sampler operand %s mixes sampler heap words' % tok)
        self.pipe.uses_sampler_heap = True
        # Sampler descriptors are 16 bytes; x2w_sampler_entry() maps the byte offset to its entry and
        # sends unaligned or out-of-range offsets to the fallback entry.
        return 'qh[x2w_sampler_entry(s%d)]' % first[1]

    def const_words(self, tok):
        r = reg_range(tok)
        words = [self.is_const(n) for n in range(r[1], r[2] + 1)]
        return None if any(w is None for w in words) else words

    # ---- instruction translation -------------------------------------------------------------
    def translate(self):
        targets = set()
        for off, n, name, asm in self.ins:
            if name.startswith('s_cbranch') or name == 's_branch':
                targets.add(int(parse(asm)[1][0].split('_')[1], 16))
        self.blocks = []          # (offset, [lines], terminator)
        self.uniform_v = set()
        self.has_branches = bool(targets)
        start = 0
        for off, n, name, asm in self.ins:
            if off in targets and off != start:
                self.blocks.append((start, self.lines, None))
                self.lines, start = [], off
            self.pc = off
            term = self.one(name, asm)
            if term:
                self.blocks.append((start, self.lines, term))
                self.lines, start = [], off + n
        if self.lines:
            self.blocks.append((start, self.lines, None))
        self.materialized = materialize  # declarations() assigns these before the body (and before any loop)
        if set(scc_effects.values()) - {'emitted'}:
            for kind, name in self.descriptor_scc_reads(dict(branches), scc_effects):
                reason = 'scc from %s in %s' % ('descriptor data' if kind == 'descriptor' else kind, name)
                if refusals is None:
                    raise Unsupported(reason)
                refusals.append((reason, self.kind))
        self.pinned_low16_v.update(self.low16_only_v)
        leak = self.pinned_low16_v & self.whole_read_v
        if leak:
            reason = 'whole-word read of v%d after odd d16 image_load' % min(leak)
            if refusals is None:
                raise Unsupported(reason)
            refusals.append((reason, self.kind))

    def descriptor_scc_reads
    def one(self, name, asm):
        _, ops, mods = parse(asm)
        base = re.sub(r'_e(32|64)$', '', name)
        if getattr(self, 'fetched', False):  # unit probes build a Stage without __init__
            # Behind a fetch call the definite-assignment guard relies on exec_known: every exec write must clear it.
            # s_wqm of exec into itself only adds lanes, so it stays a no-op; other s_wqm forms (a no-op here) and
            # exec writes outside the handlers that clear exec_known are refused. Other programs keep their handling.
            if base in ('s_wqm_b32', 's_wqm_b64') and not (ops[0] in ('exec', 'exec_lo') and ops[1:] == ops[:1]):
                raise Unsupported('%s %s after a fetch call' % (base, ', '.join(ops)))
            if ops and ops[0] in ('exec', 'exec_lo', 'exec_hi') and not base.startswith('s_wqm') and (
                    ops[0] == 'exec_hi' or not (base in SCA or base.startswith('v_cmp') or
                                                (base in ('s_cselect_b32', 's_cselect_b64') and ops[0] == 'exec'))):
                raise Unsupported('%s into %s after a fetch call' % (base, ops[0]))
        if base in ('s_wqm_b32', 's_wqm_b64'):
            if ops[0] in ('exec', 'exec_lo') and ops[1:] == ops[:1] and getattr(self, 'exec_known', False):
                self.wqm_exec = True   # every lane of a touched quad is now active, helper lanes included
            if hasattr(self, 'uniform_v'):
                self.uniform_v.clear()
            if hasattr(self, 'scc_uniform'):
                self.scc_uniform = False  # the hardware SCC result is not modelled
            return None
        if base == 's_setreg_imm32_b32':
            # The write itself changes nothing the lowering models, except MODE[1:0] (the binary32 rounding mode):
            # v_cvt_pk_u8_f32 and the other conversions are lowered for round-to-nearest-even, and the hardware
            # follows MODE.round (round toward zero truncates). s_round_mode and s_setreg_b32 are refused already.
            field = re.fullmatch(r'hwreg\(HW_REG_MODE\|1\)(?:,\s*(\d+),\s*(\d+))?\)', ops[0].strip()) if ops else None
            if field and int(field.group(2) or 0) < 2:   # shortcut: an offset >= 2 cannot reach MODE[1:0]
                # The immediate is known: the write is harmless when it leaves MODE[1:0] at 0 (nearest even).
                offset, width = int(field.group(2) or 0), int(field.group(3) or 32)
                try:
                    value = int(ops[1], 0) & 0xffffffff
                except IndexError:
                    value = None
                except ValueError:
                    # The disassembler prints some inline constants as floats (1.0, 0.5): take their binary32 bits.
                    try:
                        value = struct.unpack('<I', struct.pack('<f', float(ops[1])))[0]
                    except (ValueError, OverflowError):
                        raise Unsupported('%s %s: immediate not decoded' % (base, ', '.join(ops)))
                if value is None:
                    raise Unsupported('%s %s: immediate not decoded' % (base, ', '.join(ops)))
                if ((value & ((1 << width) - 1)) << offset) & 3:
                    raise Unsupported('%s %s writes the binary32 rounding mode MODE[1:0]' % (base, ', '.join(ops)))
            if not ops or not ops[0].strip().startswith('hwreg('):
                raise Unsupported('%s %s: hardware register not decoded' % (base, ', '.join(ops)))
            return None
        # s_waitcnt_depctr only delays the wave until the named dependency counters reach their thresholds. LLVM
        # defines it with one immediate and no register definition, use or side effect (SOPInstructions.td,
        # S_WAITCNT_DEPCTR) and inserts it only as a hazard delay (GCNHazardRecognizer.cpp,
        # fixVMEMtoScalarWriteHazards). This model has no outstanding operation a counter could count.
        if base in ('s_inst_prefetch', 's_waitcnt', 's_waitcnt_depctr', 's_nop', 'v_nop', 's_clause', 's_sopp_0f',
                    's_code_end', 's_setprio'):
            return None
        if base == 's_endpgm':
            return ('end',)
        if base == 's_branch':
            return ('jump', int(ops[0].split('_')[1], 16))
        if base.startswith('s_cbranch_'):
            cond = {'scc0': '!scc', 'scc1': 'scc', 'vccz': '(vcc & 1) == 0', 'vccnz': '(vcc & 1) != 0',
                    'execz': '(exec & 1) == 0', 'execnz': '(exec & 1) != 0'}[base[10:]]
            return ('branch', cond, int(ops[0].split('_')[1], 16))
        if base == 'exp':
            return self.export(asm)
        handler = getattr(self, 'op_' + base, None)
        if handler:
            return handler(ops, mods)
        if base.startswith('v_cmp'):
            # Before the VOP1/VOP2 SDWA route below: VOPC SDWA names also end in _sdwa.
            return self.compare(base, ops, mods)
        if base.endswith('_sdwa'):
            return self.sdwa(base[:-len('_sdwa')], ops, mods)
        if base in VEC_F:
            arity, text = VEC_F[base]
            # A missing source used to raise IndexError and an extra one was dropped without a word.
            if len(ops) != arity + 1:
                raise Unsupported('%s with %d operands, expected %d' % (base, len(ops), arity + 1))
            args = [self.f(o) for o in ops[1:1 + arity]]
            if base in ('v_fmac_f32', 'v_mac_f32'):
                args.append(self.f(ops[0]))
            return self.set_vf(reg_range(ops[0])[1], text.format(*args), mods)
        if base in VEC_U_TO_F:
            arity, text_f = VEC_U_TO_F[base]
            if len(ops) != arity + 1:
                raise Unsupported('%s with %d operands, expected %d' % (base, len(ops), arity + 1))
            # Only the output modifiers a conversion can carry; op_sel and the source modifiers are not
            # accepted here, because ignoring them would silently change the operand.
            if any(m not in ('clamp', 'mul:2', 'mul:4', 'div:2') for m in mods):
                raise Unsupported('%s with %s' % (base, mods))
            dst = reg_range(ops[0])[1]
            low16 = base in ('v_cvt_f32_f16', 'v_cvt_f32_ubyte0', 'v_cvt_f32_ubyte1', 'v_cvt_off_f32_i4')
            self.set_vf(dst, text_f.format(*[self.u(o, low16_only=low16) for o in ops[1:1 + arity]]), mods)
            return None
        if base in VEC_U:
            arity, text = VEC_U[base]
            if mods:
                raise Unsupported('%s with %s' % (base, mods))
            if len(ops) != arity + 1:
                raise Unsupported('%s with %d operands, expected %d' % (base, len(ops), arity + 1))
            return self.set_v(reg_range(ops[0])[1], text.format(*[self.u(o) for o in ops[1:1 + arity]]))
        if base in SCA:
            return self.scalar(base, ops)
        raise Unsupported(base)

    # VOP1/VOP2 with sub-dword addressing: select the source fields, run the same operation, then place
    # the low bits of the result in the destination field. The integer forms reuse VEC_U unchanged;
    # float forms are not lowered yet, and dst_unused UNUSED_SEXT has no lowering.
    def sdwa(self, base, ops, mods):
        if base == 'v_cvt_off_f32_i4':
            raise Unsupported('SDWA form of %s' % base)
        fields = {}
        for m in mods:
            key, _, value = m.partition(':')
            if key in ('src0_sel', 'src1_sel', 'src2_sel', 'dst_sel', 'dst_unused') and value:
                fields[key] = value
            else:
                raise Unsupported('%s with SDWA modifier %s' % (base, m))
        if base in VEC_U_TO_F:
            arity, text = VEC_U_TO_F[base]
            float_result = True
        elif base in VEC_U:
            arity, text = VEC_U[base]
            float_result = False
        else:
            raise Unsupported('SDWA form of %s' % base)
        sources = []
        for index in range(arity):
            token = ops[1 + index]
            sext = token.startswith('sext(') and token.endswith(')')
            if sext:
                token = token[len('sext('):-1]
            selection = fields.get('src%d_sel' % index)
            if not selection:
                raise Unsupported('%s SDWA without src%d_sel' % (base, index))
            sources.append(sdwa_source(self.u(token), selection, sext))
        result = text.format(*sources)
        if float_result:
            # The table holds the float value; the destination selection writes its bit pattern.
            result = 'U(%s)' % result
        select = fields.get('dst_sel')
        if not select:
            raise Unsupported('%s SDWA without dst_sel' % base)
        # The operation's HLSL carries its own operators, so it has to be grouped before the
        # destination field masks it: "v7 | v8 & 0xffff" would otherwise bind as "v7 | (v8 & 0xffff)".
        if SDWA_SEL[select][1] < 32:
            result = '(%s)' % result
        destination = reg_range(ops[0])[1]
        self.set_v(destination, sdwa_dest(result, select, fields.get('dst_unused', 'UNUSED_PAD'),
                                          'v%d' % destination))

    # scalar ALU with descriptor-word bookkeeping
    def scalar(self, base, ops):
        text, scc = SCA[base]
        dst = reg_range(ops[0])
        pair = base.endswith('_b64')
        srcs = ops[1:]
        # 64-bit operands name the low register; exec/vcc name themselves.
        def low(tok):
            r = reg_range(tok)
            return ('s%d' % r[1]) if r and r[0] == 's' and r[1] != r[2] else tok
        srcs = [low(t) for t in srcs]
        dname = low(ops[0])
        if pair and ('m0' in srcs or dname == 'm0'):
            raise Unsupported('64-bit scalar operand m0')
        def src_sym(t):
            if t == 'm0':
                return self.sym.get('m0')
            r = reg_range(t)
            return self.sym.get(r[1]) if r and r[0] == 's' else None
        syms = [src_sym(t) for t in srcs]
        if any(s and s[0] in ('desc', 'descnum', 'ptr', 'ptrhi') for s in syms):
            if dname == 'm0':
                if scc:
                    raise Unsupported('scc from descriptor data')
                self.sym['m0'] = syms[0] if base == 's_mov_b32' else ('descnum',)
                return None
            if dname in ('exec', 'vcc', 'exec_lo', 'vcc_lo'):
                raise Unsupported('mask from descriptor data')
            d = reg_range(dname)[1]
            own = self.sym.get(d)
            if base in ('s_mov_b32', 's_mov_b64'):
                self.sym[d] = syms[0]
                if pair:
                    hi = self.sym.get(reg_range(ops[1])[2])
                    if hi:
                        self.sym[d + 1] = hi
                    else:
                        self.sym.pop(d + 1, None)
            elif own and own[0] in ('desc', 'ptr', 'ptrhi') and any(s == own for s in syms):
                if own[0] == 'desc':
                    self.pipe.modified.add((own[1], own[2]))
            else:
                self.sym[d] = ('descnum',)
            return None
        values = [self.u(t) for t in srcs]
        expr = text.format(*values)
        if dname == 'm0':
            self.sym.pop('m0', None)
            self.emit('m0 = %s;' % expr)
            if scc:
                self.emit('scc = %s;' % scc.replace('$', 'm0'))
            return None
        if dname in ('exec', 'exec_lo', 'vcc', 'vcc_lo'):
            var = dname[:4].rstrip('_') if dname.startswith('exec') else 'vcc'
            if var == 'exec':
                copy = base in ('s_mov_b64', 's_mov_b32') and self.exec_known and (syms[0] == ('execcopy',))
                if copy:
                    return None
                self.exec_known = False
            self.emit('%s = (%s) & 1;' % (var, expr))
            if scc:
                self.emit('scc = %s;' % scc.replace('$', var))
            return None
        d = reg_range(dname)[1]
        is_mask = pair or any(t in ('exec', 'exec_lo', 'vcc', 'vcc_lo') or (reg_range(t) and reg_range(t)[0] == 's' and reg_range(t)[1] in self.masks)
                              for t in srcs) and base.split('_')[1] in ('mov', 'and', 'or', 'xor', 'andn2', 'orn2', 'nand', 'nor', 'xnor', 'not', 'cselect')
        if base in ('s_mov_b32', 's_mov_b64', 's_movk_i32') and re.fullmatch(r'-?(0x[0-9a-f]+|\d+)', srcs[0]) and srcs[0] not in FLOATS:
            value = int(srcs[0], 0)
            self.set_s(d, values[0], ('const', value & 0xffffffff))
            if pair:
                self.set_s(d + 1, '0x%08xu' % ((value >> 32) & 0xffffffff), ('const', (value >> 32) & 0xffffffff))
            self.masks.discard(d)
            return None
        if base in ('s_mov_b64', 's_mov_b32') and srcs[0] in ('exec', 'exec_lo') and self.exec_known:
            self.set_s(d, 'exec')
            self.sym[d] = ('execcopy',)
            self.masks.add(d)
            if pair:
                self.sym.pop(d + 1, None)
            return None
        if is_mask:
            expr = '(%s) & 1' % expr
            self.masks.add(d)
        else:
            self.masks.discard(d)
        self.set_s(d, expr, scc=scc)
        if pair and base != 's_mov_b64':
            self.set_s(d + 1, '0u')
        elif pair:
            r = reg_range(ops[1])
            self.set_s(d + 1, self.S(r[2]) if r and r[1] != r[2] else '0u')
        return None

    def op_s_and_saveexec_b64(self, ops, mods, text='exec & {0}'):
        dst = reg_range(ops[0])
        if not dst or dst[0] != 's':
            raise Unsupported('saveexec destination %s' % ops[0])
        d = dst[1]
        src = ops[1] if not reg_range(ops[1]) else 's%d' % reg_range(ops[1])[1]
        self.set_s(d, 'exec')
        self.masks.add(d)
        self.exec_known = False
        self.emit('exec = (%s) & 1; scc = exec != 0;' % text.format(self.u(src)))

    op_s_and_saveexec_b32 = op_s_and_saveexec_b64

    def op_s_or_saveexec_b64(self, ops, mods):
        self.op_s_and_saveexec_b64(ops, mods, 'exec | {0}')

    op_s_or_saveexec_b32 = op_s_or_saveexec_b64

    def op_s_andn2_saveexec_b64(self, ops, mods):
        self.op_s_and_saveexec_b64(ops, mods, '{0} & ~exec')

    def op_s_andn2_wrexec_b64(self, ops, mods):
        dst = reg_range(ops[0])
        if not dst or dst[0] != 's':
            raise Unsupported('wrexec destination %s' % ops[0])
        d = dst[1]
        src = ops[1] if not reg_range(ops[1]) else 's%d' % reg_range(ops[1])[1]
        self.exec_known = False
        self.emit('exec = (%s & ~exec) & 1; scc = exec != 0;' % self.u(src))
        self.set_s(d, 'exec')
        self.masks.add(d)

    op_s_andn2_wrexec_b32 = op_s_andn2_wrexec_b64

    def op_s_cselect_b32(self, ops, mods):
        self.scalar_select(ops, False)

    def op_s_cselect_b64(self, ops, mods):
        self.scalar_select(ops, True)

    def scalar_select(self, ops, pair):
        low = [('s%d' % reg_range(t)[1]) if reg_range(t) and reg_range(t)[0] == 's' else t for t in ops]
        if pair and 'm0' in low:
            raise Unsupported('64-bit scalar operand m0')
        expr = 'scc ? %s : %s' % (self.u(low[1]), self.u(low[2]))
        if low[0] == 'm0':
            self.sym.pop('m0', None)
            self.emit('m0 = %s;' % expr)
            return
        if low[0] in ('exec', 'vcc'):
            if low[0] == 'exec':
                self.exec_known = False
            self.emit('%s = (%s) & 1;' % (low[0], expr))
            return
        d = reg_range(low[0])[1]
        self.set_s(d, '(%s) & 1' % expr if pair else expr)
        if pair:
            self.masks.add(d)
            self.set_s(d + 1, '0u')

    def smem_soffset(self, tok, record_use=True):
        """Runtime expression for an SMEM scalar offset operand (s0..s105 or m0)."""
        if tok == 'm0':
            return self.u(tok)
        r = reg_range(tok)
        if r and r[0] == 's' and r[1] == r[2] and r[1] <= 105:
            if record_use:
                return self.S(r[1])
            saved = set(self.used_s)
            try:
                return self.S(r[1])
            finally:
                self.used_s = saved
        raise Unsupported('operand %r' % tok)

    def const_soffset(self, tok):
        """The constant held by an SMEM offset register s0..s105, or None (m0 and unknown values)."""
        r = reg_range(tok)
        # Symbols are not merged at control-flow joins, so a constant proves the runtime value only without branches.
        if getattr(self, 'has_branches', True):
            return None
        if r and r[0] == 's' and r[1] == r[2] and r[1] <= 105:
            # A tagged constant (the ('const', 0, 'unassigned') default-sampler placeholder) is not a run-time value.
            sym = self.sym.get(r[1])
            value = self.is_const(r[1]) if sym and len(sym) == 2 else None
            if value is not None and 0 <= value <= 0xfffff:
                return value
        return None

    def op_s_load_dwordx4(self, ops, mods, count=4):
        imm = smem_mods(mods)
        if len(ops) != 3:
            raise Unsupported('s_load with %d operands' % len(ops))
        dst, base = reg_range(ops[0]), reg_range(ops[1])
        if not dst or dst[0] != 's' or dst[2] - dst[1] + 1 != count or dst[2] > 105:
            raise Unsupported('s_load destination %s for %d dwords' % (ops[0], count))
        if not base or base[0] != 's' or base[2] - base[1] != 1 or base[2] > 105:
            raise Unsupported('s_load base %s' % ops[1])
        p = self.sym.get(base[1])
        if not p or p[0] != 'ptr':
            raise Unsupported('s_load through s%d holding %s' % (base[1], p))
        # 'off' names no offset operand at all, so the byte offset is zero, as for the literal 0.
        token = smem_offset(ops[2])
        known = self.const_soffset(token) if p[1] != ('smpheap',) else None
        if known is not None:
            # A register offset that holds a known constant (s_movk before the load) is that literal byte
            # offset; the modifier adds to it. This reaches descriptors in a bindless heap (#1723).
            total = known + (imm or 0)
            if total > 0xfffff:
                raise Unsupported('SMEM register offset %#x' % total)
            token, imm = str(total), None
        if re.fullmatch(r'-?(?:0[xX][0-9a-fA-F]+|\d+)', token):
            off = int(token, 0)
            if off < 0 or off > 0xfffff:
                raise Unsupported('SMEM immediate offset %s' % ops[2])
            if imm is not None:
                raise Unsupported('SMEM literal offset %s with offset:%#x' % (ops[2], imm))
            if p[1][0] == 'root':
                buf = self.pipe.resource((p[1], None), 'buf')
                for i in range(count):
                    self.set_s(dst[1] + i, '%s.Load(%d)' % (buf, off + 4 * i))
                return
            for i in range(count):
                self.sym[dst[1] + i] = ('desc', p[1], off, i)
            return
        if p[1] == ('smpheap',):
            if self.pipe.sampler_heap and count == 4:
                # A sampler picked at run time from the sampler heap: keep the byte offset in the
                # first destination register; image() indexes the heap table with it.
                d = dst[1]
                self.used_s.add(d)
                expr = self.smem_soffset(token)
                if imm:
                    expr = '%s + %d' % (expr, imm)
                self.emit('s%d = %s;' % (d, expr))
                for i in range(count):
                    self.sym[d + i] = ('smpheap', d, i)
                return
            self.smem_soffset(token, record_use=False)
            if count != 4:
                raise Unsupported('s_load of %d dwords from %s' % (count, p[1]))
            # Shortcut (default unless X2W_SAMPLER_HEAP=1): a fixed default sampler.
            for i in range(count):
                self.sym[dst[1] + i] = ('const', 0)
            self.pipe.notes.add('dynamic sampler index replaced by a default sampler')
            return
        raise Unsupported('s_load with a register offset through %s' % (p[1],))

    def op_s_bfm_b32(self, ops, mods):
        """Fold constant bitfield masks: titles build inline sampler words with them."""
        vals = []
        for o in ops[1:3]:
            r = reg_range(o)
            if r and r[0] == 's':
                vals.append(self.is_const(r[1]))
            else:
                try:
                    vals.append(int(o, 0))
                except ValueError:
                    vals.append(None)
        if ops[0] == 'm0':
            if None not in vals:
                value = (((1 << (vals[0] & 31)) - 1) << (vals[1] & 31)) & 0xffffffff
                expr = '0x%08xu' % value
            else:
                expr = '(((1u << (%s & 31)) - 1) << (%s & 31))' % (self.u(ops[1]), self.u(ops[2]))
            self.sym.pop('m0', None)
            self.emit('m0 = %s;' % expr)
            return None
        d = reg_range(ops[0])[1]
        if None not in vals:
            value = (((1 << (vals[0] & 31)) - 1) << (vals[1] & 31)) & 0xffffffff
            self.set_s(d, '0x%08xu' % value, ('const', value))
            return None
        self.set_s(d, '(((1u << (%s & 31)) - 1) << (%s & 31))' % (self.u(ops[1]), self.u(ops[2])))
        return None

    def op_s_load_dwordx8(self, ops, mods):
        self.op_s_load_dwordx4(ops, mods, 8)

    def op_s_load_dwordx16(self, ops, mods):
        self.op_s_load_dwordx4(ops, mods, 16)

    def op_s_load_dwordx2(self, ops, mods):
        self.op_s_load_dwordx4(ops, mods, 2)

    def op_s_load_dword(self, ops, mods):
        self.op_s_load_dwordx4(ops, mods, 1)

    def op_s_buffer_load_dword(self, ops, mods, count=1):
        imm = smem_mods(mods)
        if len(ops) != 3:
            raise Unsupported('s_buffer_load with %d operands' % len(ops))
        dst = reg_range(ops[0])
        if not dst or dst[0] != 's' or dst[2] - dst[1] + 1 != count or dst[2] > 105:
            raise Unsupported('s_buffer_load destination %s for %d dwords' % (ops[0], count))
        buf = self.pipe.resource(self.descriptor(ops[1]), 'buf')
        token = smem_offset(ops[2])
        is_lit = bool(re.fullmatch(r'-?(?:0[xX][0-9a-fA-F]+|\d+)', token))
        if is_lit:
            val = int(token, 0)
            if val < 0 or val > 0xfffff:
                raise Unsupported('SMEM immediate offset %s' % ops[2])
            if imm is not None:
                raise Unsupported('SMEM literal offset %s with offset:%#x' % (ops[2], imm))
            offset = self.u(token)
            extra = 0
        else:
            offset = self.smem_soffset(token)
            extra = imm or 0
            sreg = reg_range(token)
            if sreg and offset == ('s%d' % sreg[1]) and dst[1] <= sreg[1] < dst[2]:
                self.emit('a = %s;' % offset)
                offset = 'a'
        for i in range(count):
            self.set_s(dst[1] + i, '%s.Load(%s + %d)' % (buf, offset, extra + 4 * i))
            if is_lit:
                self.sym[dst[1] + i] = ('fixed_buffer_word',)

    def op_s_buffer_load_dwordx2(self, ops, mods):
        self.op_s_buffer_load_dword(ops, mods, 2)

    def op_s_buffer_load_dwordx4(self, ops, mods):
        self.op_s_buffer_load_dword(ops, mods, 4)

    def op_s_buffer_load_dwordx8(self, ops, mods):
        self.op_s_buffer_load_dword(ops, mods, 8)

    def op_s_buffer_load_dwordx16(self, ops, mods):
        self.op_s_buffer_load_dword(ops, mods, 16)

    def buffer_address(self, ops, mods):
        # MUBUF and MTBUF share the same address calculation: optional index * stride (idxen),
        # optional byte VGPR offset (offen), scalar byte offset (soffset) and 12-bit immediate
        # offset. Cache-policy flags (glc, slc, dlc) do not change the loaded bytes; lds, tfe
        # and any other modifier change the destination or operand layout and are refused first.
        offset = 0
        for m in mods:
            if m.startswith('offset:') and re.fullmatch(r'0x[0-9a-f]+|\d+', m[7:]):
                offset += int(m[7:], 0)
            elif m not in ('idxen', 'offen', 'glc', 'slc', 'dlc'):
                raise Unsupported('buffer modifier %s' % m)
        if len(ops) != 4:
            raise Unsupported('buffer load with %d operands' % len(ops))
        key = self.descriptor(ops[2])
        buf = self.pipe.resource(key, 'buf')
        soffset = None
        if re.fullmatch(r'0x[0-9a-f]+|\d+', ops[3]):
            offset += int(ops[3], 0)
        else:
            sreg = reg_range(ops[3])
            if not sreg or sreg[0] != 's' or sreg[1] != sreg[2] or sreg[1] > 105:
                raise Unsupported('buffer soffset %s' % ops[3])
            soffset = self.S(sreg[1])
        idxen, offen = 'idxen' in mods, 'offen' in mods
        terms = []
        if idxen and offen:
            if key in self.pipe.modified:
                raise Unsupported('indexed load through a modified descriptor')
            r = reg_range(ops[1])
            if not r or r[0] != 'v' or r[2] - r[1] != 1 or r[2] > 255:
                raise Unsupported('buffer vaddr %s for idxen offen' % ops[1])
            stride = self.pipe.ud(('stride', self.pipe.resources[key]['index']))
            terms.append('%s * ud%d' % (self.u('v%d' % r[1]), stride))
            terms.append(self.u('v%d' % r[2]))
        elif idxen:
            if key in self.pipe.modified:
                raise Unsupported('indexed load through a modified descriptor')
            r = reg_range(ops[1])
            if not r or r[0] != 'v' or r[1] != r[2] or r[1] > 255:
                raise Unsupported('buffer vaddr %s for idxen' % ops[1])
            stride = self.pipe.ud(('stride', self.pipe.resources[key]['index']))
            terms.append('%s * ud%d' % (self.u(ops[1]), stride))
        elif offen:
            r = reg_range(ops[1])
            if not r or r[0] != 'v' or r[1] != r[2] or r[1] > 255:
                raise Unsupported('buffer vaddr %s for offen' % ops[1])
            terms.append(self.u(ops[1]))
        elif ops[1] != 'off':
            raise Unsupported('buffer vaddr %s without idxen or offen' % ops[1])
        if soffset is not None:
            terms.append(soffset)
        terms.append('%d' % offset)
        return buf, ' + '.join(terms)

    def op_buffer_load_dword(self, ops, mods, count=1):
        dst = reg_range(ops[0])
        buf, address = self.buffer_address(ops, mods)
        self.emit('a = %s;' % address)
        for i in range(count):
            self.set_v(dst[1] + i, '%s.Load(a + %d)' % (buf, 4 * i))

    def op_buffer_load_dwordx2(self, ops, mods):
        self.op_buffer_load_dword(ops, mods, 2)

    def op_buffer_load_dwordx3(self, ops, mods):
        self.op_buffer_load_dword(ops, mods, 3)

    def op_buffer_load_dwordx4(self, ops, mods):
        self.op_buffer_load_dword(ops, mods, 4)

    def tbuffer(self, ops, mods, count):
        dst = reg_range(ops[0])
        formats = [m for m in mods if m.startswith('format:')]
        fmt = formats[0][8:-1] if formats else 'BUF_FMT_8_UNORM'  # the disassembler omits the default format
        buf, address = self.buffer_address(ops, [m for m in mods if not m.startswith('format:')])
        self.emit('a = %s;' % address)
        self.load_format(fmt, buf, dst[1], count)

    def load_format(self, fmt, buf, first, count):
        if fmt not in FETCH_FORMATS:
            raise Unsupported('buffer format %s' % fmt)
        load, comps = FETCH_FORMATS[fmt]
        if load:
            self.emit('w = %s;' % load.format(buf))
        for i in range(count):
            self.set_v(first + i, comps[i].format(buf))

    def op_s_swappc_b64(self, ops, mods):
        # Only the entry call into the driver's fetch routine is translated: as the loads that routine makes.
        if self.kind != 'vs' or self.pc != 0 or ops != ['s[4:5]', 's[4:5]']:
            raise Unsupported('s_swappc_b64 %s at +0x%x' % (', '.join(ops), self.pc))
        if not self.pipe.vs_fetch:
            raise Unsupported('vertex stage calls a fetch routine and the pipeline declares no input layout')
        base = self.pipe.ud(('basevertex',))
        for e in self.pipe.vs_fetch:
            key = (('vbtab',), 16 * e['slot'])
            buf = self.pipe.resource(key, 'buf')
            stride = self.pipe.ud(('stride', self.pipe.resources[key]['index']))
            self.emit('a = (vid + ud%d) * ud%d + %d;' % (base, stride, e['offset']))
            self.load_format(FETCH_DXGI[e['format']][0], buf, e['vgpr'], e['count'])
        self.fetched = True

    def op_tbuffer_load_format_x(self, ops, mods):
        self.tbuffer(ops, mods, 1)

    def op_tbuffer_load_format_xy(self, ops, mods):
        self.tbuffer(ops, mods, 2)

    def op_tbuffer_load_format_xyz(self, ops, mods):
        self.tbuffer(ops, mods, 3)

    def op_tbuffer_load_format_xyzw(self, ops, mods):
        self.tbuffer(ops, mods, 4)

    def image(self, ops, mods, how):
        dst = reg_range(ops[0])
        dim = [m for m in mods if m.startswith('dim:')][0][4:]
        dmask = int([m for m in mods if m.startswith('dmask:')][0][6:], 0)
        kinds = {'SQ_RSRC_IMG_1D': 'tex1d', 'SQ_RSRC_IMG_2D': 'tex2d', 'SQ_RSRC_IMG_3D': 'tex3d', 'SQ_RSRC_IMG_2D_ARRAY': 'tex2da'}
        if dim not in kinds:
            raise Unsupported('image dimension %s' % dim)
        # A 2D array takes the slice as its third address component, where a volume takes z. Sample and
        # Load on a Texture2DArray take it in the same place, so only the declared type differs.
        one = dim == 'SQ_RSRC_IMG_1D'
        three = dim in ('SQ_RSRC_IMG_3D', 'SQ_RSRC_IMG_2D_ARRAY')
        coords = ops[1]
        if coords.startswith('['):
            cs = [c.strip() for c in coords[1:-1].split(',')]
        else:
            r = reg_range(coords)
            cs = ['v%d' % n for n in range(r[1], r[2] + 1)]
        if one and (how != 'sample' or len(cs) != 1 or not dmask or any(not m.startswith(('dim:', 'dmask:')) for m in mods)):
            raise Unsupported('unverified 1D image sample address or modifiers')
        lod = None
        if how == 'l':
            # Address dwords are the coordinates and then the explicit level; offsets, bias, clamp and derivatives
            # would add dwords or modifiers this lowering does not read, so they are refused. Only the cache-policy
            # bits (glc, slc, dlc), which never change the sampled value, are accepted next to dim and dmask;
            # unorm, r128, tfe, lwe, a16, d16 and reserved bits would each change the address, resource or result
            # layout and are refused by name.
            for m in mods:
                if not (m.startswith(('dim:', 'dmask:')) or m in ('dlc', 'glc', 'slc')):
                    raise Unsupported('image_sample_l modifier %s' % m)
            if dmask == 0:
                raise Unsupported('image_sample_l with an empty dmask')
            if len(cs) != (4 if three else 3):
                raise Unsupported('image_sample_l with %d address registers' % len(cs))
            lod, cs = cs[-1], cs[:-1]
        tex = self.pipe.resource(self.descriptor(ops[2]), kinds[dim])
        if how in ('load', 'load_mip'):
            # image_load_mip addresses x, y, [slice or z] and then the mip, so a 2D array is four registers
            # and Texture2DArray.Load takes int4(x, y, slice, mip) like a volume's int4(x, y, z, mip).
            axes = 3 if three else 2
            packed_2d = how == 'load' and dim == 'SQ_RSRC_IMG_2D' and 'a16' in mods
            allowed = ('unorm', 'glc', 'slc', 'dlc')
            if packed_2d:
                allowed += ('a16',)
            unsupported = any(not m.startswith(('dim:', 'dmask:')) and m not in allowed for m in mods)
            wrong_address = (packed_2d and len(cs) != 1) or (how == 'load_mip' and len(cs) != axes + 1)
            if unsupported or wrong_address or not dmask:
                raise Unsupported('image_%s address or modifiers: %s' % (how, ' '.join(mods)))
            if packed_2d:
                packed = self.u(cs[0])
                address = ['%s & 0xffffu' % packed, '%s >> 16' % packed]
            else:
                address = [self.u(c) for c in cs[:axes]]
            mip = self.u(cs[axes]) if how == 'load_mip' else '0'
            call = '%s.Load(int%d(%s))' % (tex, axes + 1, ', '.join(address + [mip]))
        else:
            heap = self.heap_sampler(ops[3])
            words = None if heap else self.const_words(ops[3])
            smp = heap or self.pipe.sampler(tuple(words) if words else self.descriptor(ops[3]))
            uv = (self.f(cs[0]) if one else 'float3(%s, %s, %s)' % (self.f(cs[0]), self.f(cs[1]), self.f(cs[2])) if three
                  else 'float2(%s, %s)' % (self.f(cs[0]), self.f(cs[1])))
            if how == 'l':
                call = '%s.SampleLevel(%s, %s, %s)' % (tex, smp, uv, self.f(lod))
            elif how == 'lz' or not self.exec_known or self.kind != 'ps' or len(self.blocks) > 0:
                call = '%s.SampleLevel(%s, %s, 0)' % (tex, smp, uv)
            else:
                call = '%s.Sample(%s, %s)' % (tex, smp, uv)
        self.emit('t = %s;' % call)
        n = dst[1]
        for bit, comp in enumerate('xyzw'):
            if dmask & (1 << bit):
                self.set_v(n, 'U(t.%s)' % comp)
                n += 1

    def image_compare(self, ops, mods, how):
        """image_sample_c and image_sample_c_lz: a 2D or 2D-array depth comparison returning one float.

        The address dwords are the reference, then u and v (and the slice of a 2D array), which is what
        SampleCmp* takes as its third argument and its location. The comparison function and the filter live in the sampler descriptor,
        so they are read from its constant words by the decoder the pixel AOT path already checks against
        D3D12 (pixel_aot.comparison_sampler_state) and bound as a SamplerComparisonState. HLSL has no
        SampleCmpLevel before shader model 6.7 (the stages here compile as 5.1 and 6.0), so
        image_sample_c_l is refused instead of being sampled at level zero.
        """
        base = 'image_sample_' + how
        if how == 'c_l':
            raise Unsupported('image_sample_c_l: SampleCmpLevel needs shader model 6.7; only c and c_lz are lowered')
        for m in mods:
            if not (m.startswith(('dim:', 'dmask:')) or m in ('dlc', 'glc', 'slc')):
                raise Unsupported('%s modifier %s' % (base, m))
        dims = [m[4:] for m in mods if m.startswith('dim:')]
        masks = [m[6:] for m in mods if m.startswith('dmask:')]
        if len(dims) != 1 or len(masks) != 1:
            raise Unsupported('%s without exactly one dim and dmask' % base)
        dim, dmask = dims[0], int(masks[0], 0)
        if dim not in ('SQ_RSRC_IMG_2D', 'SQ_RSRC_IMG_2D_ARRAY'):
            raise Unsupported('%s on %s' % (base, dim))
        array = dim == 'SQ_RSRC_IMG_2D_ARRAY'
        if dmask != 1:
            raise Unsupported('%s with dmask %#x: a comparison returns one channel' % (base, dmask))
        if len(ops) != 4:
            raise Unsupported('%s with %d operands' % (base, len(ops)))
        coords = ops[1]
        if coords.startswith('['):
            cs = [c.strip() for c in coords[1:-1].split(',')]
        else:
            r = reg_range(coords)
            cs = ['v%d' % n for n in range(r[1], r[2] + 1)]
        # The reference, u and v; a 2D array adds the slice as the last address register.
        if len(cs) != (4 if array else 3):
            raise Unsupported('%s with %d address registers' % (base, len(cs)))
        if self.heap_sampler(ops[3]):
            raise Unsupported('%s through a sampler heap' % base)
        words = self.const_words(ops[3])
        if words is None:
            raise Unsupported('%s with a sampler that is not constant' % base)
        import pixel_aot
        try:
            pixel_aot.comparison_sampler_state(tuple(words))
        except ValueError as error:
            raise Unsupported('%s sampler: %s' % (base, error))
        tex = self.pipe.resource(self.descriptor(ops[2]), 'tex2da' if array else 'tex2d')
        smp = self.pipe.sampler(tuple(words), comparison=True)
        # Texture2DArray takes the slice as the third location component, like Sample and Load on the same type.
        location = ('float3(%s, %s, %s)' % (self.f(cs[1]), self.f(cs[2]), self.f(cs[3])) if array
                    else 'float2(%s, %s)' % (self.f(cs[1]), self.f(cs[2])))
        args = '%s, %s, %s' % (smp, location, self.f(cs[0]))
        # MAX_LOD (word 1 bits 23:12) is not carried into the static sampler, and the comparison-state mapping ignores
        # the mip filter and uses level zero. Only a source limited to level zero is therefore exact for the implicit
        # form; any other limit would need mip selection and blending that is not lowered in this slice, so it is refused.
        max_lod = (words[1] >> 12) & 0xfff
        if how == 'c' and max_lod:
            raise Unsupported('%s with implicit mip selection (sampler MAX_LOD %#x)' % (base, max_lod))
        call = '%s.SampleCmpLevelZero(%s)' % (tex, args)
        self.set_v(reg_range(ops[0])[1], 'U(%s)' % call)

    def op_image_sample_c(self, ops, mods):
        self.image_compare(ops, mods, 'c')

    def op_image_sample_c_lz(self, ops, mods):
        self.image_compare(ops, mods, 'c_lz')

    def op_image_sample_c_l(self, ops, mods):
        self.image_compare(ops, mods, 'c_l')

    def op_image_sample(self, ops, mods):
        self.image(ops, mods, 'sample')

    def op_image_sample_lz(self, ops, mods):
        self.image(ops, mods, 'lz')

    def op_image_sample_l(self, ops, mods):
        self.image(ops, mods, 'l')

    def op_image_load(self, ops, mods):
        self.image(ops, mods, 'load')

    def op_image_load_mip(self, ops, mods):
        self.image(ops, mods, 'load_mip')

    def op_v_cvt_i32_f32(self, ops, mods):
        if mods:
            raise Unsupported('v_cvt_i32_f32 with modifiers')
        dst = reg_range(ops[0])[1]
        self.set_v(dst, 'uint(int(F(%s)))' % self.u(ops[1]))

    def op_v_cvt_pk_u8_f32(self, ops, mods):
        # VOP3A opcode 350 of the RDNA 1.0 and RDNA 2 ISA guides:
        #   D = (S2 & ~(0xff << 8 * S1[1:0])) | ((flt32_to_uint8(S0) & 0xff) << 8 * S1[1:0])
        # The guides leave flt32_to_uint8 undefined. Measured on RDNA 2 hardware (RX 6900 XT, gfx1030, 5432 rows,
        # see docs/experiments.md): round to nearest even, saturated to [0, 255], NaN and negatives (-0.0, -inf
        # included) give 0, +inf gives 255. MODE[1:0] = 3 (round toward zero) truncates instead, which is why a write
        # of those bits is refused in s_setreg_imm32_b32. s_denorm_mode 0 changed nothing.
        # Lowering: (bits - 1u) < 0x7f800000u holds exactly from the smallest positive subnormal to +inf and fails
        # for both zeros, every negative and every NaN, so those take the 0u arm. It is an integer test on the raw
        # word: a float compare may be folded under fast-math (DXC emits `fcmp fast ogt`, so NaN would depend on
        # the driver), and min/max treat a NaN operand differently between back ends.
        # Above zero, min(S0, 255.0) saturates (+inf included) and HLSL round() is round_ne (DXIL Round_ne), so the
        # conversion is exact and in range.
        if mods:
            raise Unsupported('v_cvt_pk_u8_f32 with modifiers')
        if len(ops) != 4:
            raise Unsupported('v_cvt_pk_u8_f32 with %d operands, expected 4' % len(ops))
        dst = reg_range(ops[0])
        if not dst or dst[0] != 'v' or dst[1] != dst[2]:
            raise Unsupported('v_cvt_pk_u8_f32 destination %s' % ops[0])
        if reg_range(ops[1]):
            src = reg_range(ops[1])
            if src[1] != src[2]:
                raise Unsupported('v_cvt_pk_u8_f32 source %s' % ops[1])
        bits, select, rest = self.u(ops[1]), self.u(ops[2]), self.u(ops[3])
        value = 'F(%s)' % bits
        shift = '((%s & 3u) << 3u)' % select
        self.set_v(dst[1], '((%s & ~(0xffu << %s)) | (((((%s - 1u) < 0x7f800000u) ? uint(round(min(%s, 255.0))) : 0u) & 0xffu) << %s))'
                   % (rest, shift, bits, value, shift))

    def op_s_ff1_i32_b32(self, ops, mods):
        if self.kind != 'cs' or mods or len(ops) != 2 or ops[1] != 'exec_lo':
            raise Unsupported('s_ff1_i32_b32 requires compute exec_lo')
        dst = reg_range(ops[0])
        if not dst or dst[0] != 's' or dst[1] != dst[2] or dst[1] > 105:
            raise Unsupported('s_ff1_i32_b32 requires one SGPR destination')
        if self.past_entry or not self.exec_known:
            raise Unsupported('s_ff1_i32_b32 requires the compute entry EXEC mask')
        # At dispatch entry even the last partial wave starts at lane zero.
        # FF1 reports its index and leaves SCC unchanged.
        self.set_s(dst[1], '0u', ('const', 0))

    # ---- slot 57 mesh stage: instructions mesh_skeleton() puts in place of the group-memory exchanges ----
    def op_x2w_mesh_lane0(self, ops, mods):
        # `s_mov_b64 exec, 1` leaves lane 0 of each wave running. The saved copy of that mask is not modelled.
        r = reg_range(ops[0])
        for n in range(r[1], r[2] + 1):
            self.sym[n] = ('the saved mesh lane-0 mask',)
        self.emit('exec = (tid & 63u) == 0u ? 1u : 0u;')
        self.exec_known = False

    def op_x2w_mesh_group(self, ops, mods):
        # The word is entry v5 >> 7 of wave 0, lane 0. What v5 holds at entry is not verified, so a use is refused.
        self.sym[reg_range(ops[0])[1]] = ('the mesh group index',)

    def op_x2w_mesh_counts(self, ops, mods):
        prims, verts, temporary = (reg_range(tok)[1] for tok in ops)
        self.emit('if (tid == 0u) { lds[0] = %s; lds[1] = %s; }' % (self.S(prims), self.S(verts)))
        self.emit('GroupMemoryBarrierWithGroupSync();')
        # Thread t writes vertex t and primitive t, so a count above the thread count names outputs that no
        # thread writes. The clamp keeps the declared arrays in bounds and changes nothing a thread writes.
        for n, word in ((prims, 0), (verts, 1)):
            self.set_s(n, 'min(lds[%d], %du)' % (word, MESH_THREADS), uniform=True)
        # Without a vertex no index is valid, so no primitive is.
        self.emit('if (%s == 0u) %s = 0u;' % (self.S(verts), self.S(prims)))
        self.emit('SetMeshOutputCounts(%s, %s);' % (self.S(verts), self.S(prims)))
        self.mesh_verts = self.S(verts)
        self.sym[temporary] = ('a mesh allocation temporary',)

    def mesh_export(self, asm, target, vals, flags):
        if target == 'target_20':
            # First word: three 9-bit vertex indices at bits 0, 10 and 20, an edge flag above each (not modelled:
            # triangles are filled) and the null bit 31 (Mesa ac_nir_pack_ngg_prim_exp_arg).
            if vals[1:] != ['off'] * 3 or [f for f in flags if f != 'done']:
                raise Unsupported('mesh primitive export form %r' % asm)
            # An index at or past the vertex count has no defined vertex; the clamp keeps it inside the array.
            # A null primitive names one vertex three times, which covers no pixel.
            self.emit('if (exec & 1) { w = %s; tris[tid] = w >> 31 != 0u ? (uint3)0u : '
                      'min(uint3(w, w >> 10, w >> 20) & 511u, max(%s, 1u) - 1u); }'
                      % (self.u(vals[0]), self.mesh_verts))
            return None
        first = len(self.lines)
        self.in_mesh_export = True
        try:
            self.export(asm)
        finally:
            self.in_mesh_export = False
        self.lines[first:] = ['if (exec & 1) { %s mesh_vertex = true; }' % ' '.join(self.lines[first:])]
        return None

    def op_v_mbcnt_lo_u32_b32(self, ops, mods, high=False):
        name = 'v_mbcnt_%s_u32_b32' % ('hi' if high else 'lo')
        if self.kind == 'ms':
            # V_MBCNT counts the mask bits strictly below the lane, in the low or high half of a wave64 mask.
            dst = reg_range(ops[0]) if len(ops) == 3 else None
            if mods or not dst or dst[0] != 'v' or dst[1] != dst[2] or ops[1] != '-1':
                raise Unsupported('%s in a mesh stage requires a VGPR, the mask -1 and a base' % name)
            below = 'max(tid & 63u, 32u) - 32u' if high else 'min(tid & 63u, 32u)'
            self.set_v(dst[1], '(%s + %s)' % (self.u(ops[2]), below))
            return
        if self.kind != 'cs':
            raise Unsupported(name)
        if mods or len(ops) != 3:
            raise Unsupported('%s requires three unmodified operands' % name)
        x, y, z = self.pipe.group
        if x * y * z > 32:
            raise Unsupported('%s requires at most 32 threads per group' % name)
        dst = reg_range(ops[0])
        if not dst or dst[0] != 'v' or dst[1] != dst[2] or dst[1] > 255:
            raise Unsupported('%s destination %s' % (name, ops[0]))
        # Both halves use the tested constant-mask profile, even though HI only copies the base.
        mask = self.const_u32(ops[1])
        if mask is None:
            raise Unsupported('%s requires a known mask' % name)
        base = reg_range(ops[2])
        base_is_vgpr = base and base[0] == 'v' and base[1] == base[2] and base[2] <= 255
        if not base_is_vgpr and self.const_u32(ops[2]) is None:
            raise Unsupported('%s requires a VGPR or known base' % name)
        result = self.u(ops[2])
        # RDNA 2 V_MBCNT counts mask bits strictly below the lane. All threads of
        # this group fit in the low half of either guest wave size, including lane 31.
        if not high:
            lane = '(tid.x + %du * (tid.y + %du * tid.z))' % (x, y)
            result = '(countbits(0x%08xu & ((1u << %s) - 1u)) + %s)' % (mask, lane, result)
        self.set_v(dst[1], result)

    def op_v_mbcnt_hi_u32_b32(self, ops, mods):
        self.op_v_mbcnt_lo_u32_b32(ops, mods, high=True)

    def op_v_readfirstlane_b32(self, ops, mods):
        src, dst = reg_range(ops[1]), reg_range(ops[0])
        if (mods or self.has_branches or not self.exec_known or not dst or dst[0] != 's' or
                not src or src[0] != 'v' or src[1] not in self.uniform_v):
            raise Unsupported('v_readfirstlane_b32 requires a proven uniform source')
        self.set_s(reg_range(ops[0])[1], self.u(ops[1]))

    def op_v_mov_b32(self, ops, mods):
        self.set_v(reg_range(ops[0])[1], self.u(ops[1]))

    def op_v_cndmask_b32(self, ops, mods):
        # The select takes no output modifier: a clamp or mul on it would change the value, so a
        # modifier that reaches here is refused rather than dropped from the emitted text.
        if mods:
            raise Unsupported('v_cndmask_b32 with %s' % mods)
        mask = self.u(ops[3]) if len(ops) > 3 else 'vcc'
        self.set_v(reg_range(ops[0])[1],
                   '(%s & 1) ? %s : %s' % (mask, self.u_src(ops[2]), self.u_src(ops[1])))

    def op_v_interp_p1_f32(self, ops, mods):
        self.note_attr(ops[2], 'linear')

    def op_v_interp_p2_f32(self, ops, mods):
        attr, comp = self.note_attr(ops[2], 'linear')
        self.set_v(reg_range(ops[0])[1], 'U(i.param%d.%s)' % (attr, comp))

    def op_v_interp_mov_f32(self, ops, mods):
        attr, comp = self.note_attr(ops[2], 'flat')
        self.set_v(reg_range(ops[0])[1], 'i.fparam%d.%s' % (attr, comp))

    def note_attr(self, tok, how):
        m = re.fullmatch(r'attr(\d+)\.([xyzw])', tok)
        attr = int(m.group(1))
        self.attrs.setdefault(attr, set()).add(how)
        return attr, m.group(2)

    def mix(self, ops, mods, high):
        """v_fma_mix{lo,hi}_f16: fma of operands that are f32 or f16 halves; the f16 result goes to one half."""
        sel = [0, 0, 0]
        for m in mods:
            if m.startswith('op_sel_hi:'):
                sel = [int(x) for x in m[11:-1].split(',')]
            elif m.startswith('op_sel:'):
                raise Unsupported('op_sel on fma_mix')
        def operand(tok, is_half):
            if not is_half or tok in ('-0', '0'):
                return self.f(tok)
            return 'f16tof32(%s)' % self.u(tok)
        expr = '%s * %s + %s' % tuple(operand(t, s) for t, s in zip(ops[1:4], sel))
        if 'clamp' in mods:
            expr = 'saturate(%s)' % expr
        d = reg_range(ops[0])[1]
        self.used_v.add(d)
        if high:
            self.set_v(d, '(v%d & 0xffffu) | (f32tof16(%s) << 16)' % (d, expr))
        else:
            self.set_v(d, '(v%d & 0xffff0000u) | f32tof16(%s)' % (d, expr))

    def op_v_fma_mixlo_f16(self, ops, mods):
        self.mix(ops, mods, False)

    def op_v_fma_mixhi_f16(self, ops, mods):
        self.mix(ops, mods, True)

    def cube(self, kind, ops, mods):
        """v_cube{id,sc,tc,ma}_f32: LLVM's ConstantFoldAMDGCNCubeIntrinsic.

        The major axis is z when |z| is at least |x| and |y|, otherwise y when |y| is at
        least |x|, otherwise x. A face is negative only for a nonzero, non-NaN negative
        coordinate. cubema returns twice the signed major coordinate.
        """
        if len(ops) != 4:
            raise Unsupported('v_cube%s_f32 with %d operands, expected 4' % (kind, len(ops)))
        if any(m != 'clamp' for m in mods):
            raise Unsupported('v_cube%s_f32 with %s' % (kind, ', '.join(mods)))
        dst = reg_range(ops[0])
        if not dst or dst[0] != 'v' or dst[2] != dst[1] or dst[2] > 255:
            raise Unsupported('v_cube%s_f32 destination %s' % (kind, ops[0]))
        x, y, z = [self.f(o) for o in ops[1:4]]
        temp = 'cube%d' % getattr(self, 'cube_temps', 0)
        self.cube_temps = getattr(self, 'cube_temps', 0) + 1
        self.emit('float3 %s = float3(%s, %s, %s);' % (temp, x, y, z))
        c = temp
        a = 'abs(%s)' % c
        zmajor = '(%s.z >= %s.x && %s.z >= %s.y)' % (a, a, a, a)
        ymajor = '(%s.y >= %s.x)' % (a, a)
        expression = {
            'id': '(%s ? (%s.z < 0.0f ? 5.0f : 4.0f) : %s ? (%s.y < 0.0f ? 3.0f : 2.0f) '
                  ': (%s.x < 0.0f ? 1.0f : 0.0f))' % (zmajor, c, ymajor, c, c),
            'sc': '(%s ? (%s.z < 0.0f ? -%s.x : %s.x) : %s ? %s.x '
                  ': (%s.x < 0.0f ? %s.z : -%s.z))' % (zmajor, c, c, c, ymajor, c, c, c, c),
            'tc': '(%s ? -%s.y : %s ? (%s.y < 0.0f ? -%s.z : %s.z) : -%s.y)' % (
                zmajor, c, ymajor, c, c, c, c),
            'ma': '((%s ? %s.z : %s ? %s.y : %s.x) * 2.0f)' % (zmajor, c, ymajor, c, c),
        }[kind]
        self.set_vf(dst[1], expression, mods)

    def op_v_cubeid_f32(self, ops, mods):
        self.cube('id', ops, mods)

    def op_v_cubesc_f32(self, ops, mods):
        self.cube('sc', ops, mods)

    def op_v_cubetc_f32(self, ops, mods):
        self.cube('tc', ops, mods)

    def op_v_cubema_f32(self, ops, mods):
        self.cube('ma', ops, mods)

    def i64_words(self, tok):
        """(low, high) uint expressions of a 64-bit integer source: a VGPR/SGPR pair or an inline int."""
        pair = reg_range(tok)
        if pair and pair[2] == pair[1] + 1:
            if pair[0] == 'v' and pair[2] <= 255:
                return self.u('v%d' % pair[1]), self.u('v%d' % pair[2])
            if pair[0] == 's' and pair[1] % 2 == 0 and pair[2] <= 105:
                return self.u('s%d' % pair[1]), self.u('s%d' % pair[2])
            raise Unsupported('64-bit source %s' % tok)
        if re.fullmatch(r'0|-?[1-9]\d*', tok) and -16 <= int(tok) <= 64:
            value = int(tok) & 0xffffffffffffffff
            return '%du' % (value & 0xffffffff), '%du' % (value >> 32)
        raise Unsupported('64-bit source %s' % tok)

    def shift64(self, ops, mods, helper):
        if mods:
            raise Unsupported('64-bit shift with %s' % mods)
        if len(ops) != 3:
            raise Unsupported('64-bit shift with %d operands, expected 3' % len(ops))
        dst = reg_range(ops[0])
        if not dst or dst[0] != 'v' or dst[2] != dst[1] + 1 or dst[2] > 255:
            raise Unsupported('64-bit shift destination %s' % ops[0])
        sr = reg_range(ops[1])
        if (sr and (sr[1] != sr[2] or (sr[0] == 'v' and sr[1] > 255) or (sr[0] == 's' and sr[1] > 105))
                or ops[1] in ('vcc', 'exec') or ops[1] in FLOATS):
            raise Unsupported('64-bit shift amount %s' % ops[1])
        amount = self.u(ops[1])
        low, high = self.i64_words(ops[2])
        body = '{ uint2 t = %s(%s, %s, %s); v%d = t.x; v%d = t.y; }' % (
            helper, low, high, amount, dst[1], dst[2])
        self.uniform_v.discard(dst[1])
        self.uniform_v.discard(dst[2])
        self.used_v.update((dst[1], dst[2]))
        self.emit(body if self.exec_known else 'if (exec & 1) ' + body)

    def op_v_lshlrev_b64(self, ops, mods):
        self.shift64(ops, mods, 'shl64')

    def op_v_lshrrev_b64(self, ops, mods):
        self.shift64(ops, mods, 'shr64u')

    def op_v_ashrrev_i64(self, ops, mods):
        self.shift64(ops, mods, 'shr64i')
    def op_v_cvt_pkrtz_f16_f32(self, ops, mods):
        self.set_v(reg_range(ops[0])[1], 'f32tof16(%s) | (f32tof16(%s) << 16)' % (self.f(ops[1]), self.f(ops[2])))

    # VOPC with sub-dword addressing (tools/disasm_rdna2.py _decode_vopc_sdwa prints it): each source
    # reads its src0_sel/src1_sel field, zero-extended or, under sext(), sign-extended, and the
    # compare then reads that 32-bit value as its type. Integer compares only (see compare()).
    def compare_sdwa_sources(self, base, typ, ops, mods):
        fields = {}
        for m in mods:
            key, _, value = m.partition(':')
            if key not in ('src0_sel', 'src1_sel') or value not in SDWA_SEL or key in fields:
                raise Unsupported('%s with SDWA modifier %s' % (base, m))
            fields[key] = value
        sources = []
        for index, token in enumerate(ops):
            selection = fields.get('src%d_sel' % index)
            if not selection:
                raise Unsupported('%s SDWA without src%d_sel' % (base, index))
            sext = token.startswith('sext(') and token.endswith(')')
            if sext:
                token = token[len('sext('):-1]
            word = sdwa_source(self.u(token), selection, sext)
            sources.append('int(%s)' % word if typ == 'i32' else word)
        return sources

    def compare(self, base, ops, mods=()):
        # v_cmp_class_f32 / v_cmpx_class_f32 test the binary32 class of the value in SRC0 against the
        # class-bit mask in SRC1 (the ISA order: the float operand first, the integer mask second).
        if base in ('v_cmp_class_f32', 'v_cmpx_class_f32', 'v_cmp_class_f16', 'v_cmpx_class_f16'):
            f16 = base.endswith('_f16')
            if len(ops) == 3:
                dst, value, mask = ops
            elif base.startswith('v_cmpx_') and len(ops) == 2:
                # The _e32 spelling writes EXEC implicitly and prints no destination.
                dst, (value, mask) = 'exec', ops
            else:
                raise Unsupported('%s with %d operands' % (base, len(ops)))
            if f16:
                # The helper classifies the raw binary16 bits in the low half of the word, so the value is
                # read as an integer and an abs/neg modifier flips bit 15 (not 31). A float-printed
                # operand would be lowered as its binary32 bits, so it is refused instead.
                operand = value
                if operand.startswith('-') and not re.match(r'-[\d.]', operand):
                    operand = operand[1:]
                elif operand.startswith('|') and operand.endswith('|'):
                    operand = operand[1:-1]
                if operand in FLOATS or re.fullmatch(r'-?\d+\.\d+', operand):
                    raise Unsupported('binary16 class operand %s' % value)
                bits = self.u(operand)
                if value.startswith('-') and not re.match(r'-[\d.]', value):
                    bits = '(%s ^ 0x8000u)' % bits
                elif value.startswith('|') and value.endswith('|'):
                    bits = '(%s & 0x7fffu)' % bits
                cond = '(class16(%s) & %s) != 0u' % (bits, self.u(mask))
            else:
                cond = '(class32(%s) & %s) != 0u' % (self.f(value), self.u(mask))
            predicate = '((%s) ? 1u : 0u)' % cond
            # A narrowed EXEC must stay narrowed: intersect with it, as the generic compare path does.
            if not self.exec_known:
                predicate = '(exec & %s)' % predicate
            if base.startswith('v_cmpx_') or dst in ('exec', 'exec_lo'):
                self.exec_known = False
                self.emit('exec = %s;' % predicate)
            elif dst in ('vcc', 'vcc_lo'):
                self.emit('vcc = %s;' % predicate)
            else:
                d = reg_range(dst)[1]
                self.set_s(d, predicate)
                self.masks.add(d)
            return None

        m = re.fullmatch(r'v_cmp(x?)_(\w+?)_(f32|u32|i32)(_sdwa)?', base)
        if not m:
            raise Unsupported(base)
        writes_exec, cond, typ, sdwa = m.group(1) == 'x', m.group(2), m.group(3), bool(m.group(4))
        table = {'lt': '{0} < {1}', 'eq': '{0} == {1}', 'le': '{0} <= {1}', 'gt': '{0} > {1}', 'lg': '{0} != {1}', 'ne': '{0} != {1}',
                 'ge': '{0} >= {1}', 'neq': '!({0} == {1})', 'nlt': '!({0} < {1})', 'nle': '!({0} <= {1})', 'ngt': '!({0} > {1})',
                 'nge': '!({0} >= {1})', 'nlg': '!({0} != {1})', 'o': '({0} == {0} && {1} == {1})', 'u': '({0} != {0} || {1} != {1})',
                 't': 'true', 'f': 'false'}
        # Conditions outside the table, v_cmp_class_* among them, stay refused in both forms.
        if cond not in table:
            raise Unsupported(base)
        if sdwa and typ == 'f32':
            # The f32 predicates are shared with the non-SDWA forms, whose lg/nlg do not yet follow the
            # ordered RDNA 2 definition for NaN (#1920). SDWA keeps them refused until that is verified.
            raise Unsupported('%s: f32 SDWA compares are not lowered' % base)
        if sdwa:
            # v_cmpx_*_sdwa prints its two sources and writes EXEC (gfx10 forbids sdst_en there);
            # v_cmp_*_sdwa first prints vcc_lo or the SGPR it writes.
            if len(ops) != (2 if writes_exec else 3):
                raise Unsupported('%s with %d operands' % (base, len(ops)))
            if not writes_exec and ops[0] not in ('vcc', 'vcc_lo', 'exec', 'exec_lo') and not (
                    reg_range(ops[0]) and reg_range(ops[0])[0] == 's'):
                raise Unsupported('%s into %s' % (base, ops[0]))
        if len(ops) == 3:
            dst, a, b = ops
        else:
            dst, (a, b) = ('exec' if writes_exec else 'vcc'), ops
        if sdwa:
            x, y = self.compare_sdwa_sources(base, typ, [a, b], mods)
        elif typ == 'f32':
            x, y = self.f(a), self.f(b)
        elif typ == 'i32':
            x, y = 'int(%s)' % self.u(a), 'int(%s)' % self.u(b)
        else:
            x, y = self.u(a), self.u(b)
        expr = '((%s) ? 1u : 0u)' % table[cond].format(x, y)
        if not self.exec_known:
            expr = '(exec & %s)' % expr
        if writes_exec or dst in ('exec', 'exec_lo'):
            self.exec_known = False
            self.emit('exec = %s;' % expr)
        elif dst in ('vcc', 'vcc_lo'):
            self.emit('vcc = %s;' % expr)
        else:
            d = reg_range(dst)[1]
            self.set_s(d, expr)
            self.masks.add(d)

    def export(self, asm):
        m = re.fullmatch(r'exp (\w+) (.*?)((?: done| compr| vm)*)', asm)
        target, vals, flags = m.group(1), [v.strip() for v in m.group(2).split(',')], m.group(3).split()
        if 'vm' in flags and not self.exec_known:
            self.emit('if (!(exec & 1)) discard;')
        if target == 'mrtz':
            if (self.kind != 'ps' or self.has_branches or self.blocks or not self.exec_known or 'depth' in self.exports
                    or 'compr' in flags or len(vals) != 4 or not re.fullmatch(r'v\d+', vals[0])
                    or vals[1:] != ['off', 'off', 'off']):
                raise Unsupported('MRTZ requires one uncompressed depth export in a straight-line pixel shader with entry EXEC')
            self.exports['depth'] = True
            self.emit('o.depth = %s;' % self.f(vals[0]))
            return None
        name = {'pos0': 'pos'}.get(target, target)
        if not re.fullmatch(r'pos|param\d+|mrt[0-7]', name):
            raise Unsupported('export target %s' % target)
        kind = self.pipe.mrt_kind(int(name[3:])) if name.startswith('mrt') else 'float'
        if 'compr' in flags:
            if kind in ('uint', 'int'):
                raise Unsupported('packed integer export %s' % name)
            a, b = self.u(vals[0]), self.u(vals[2])
            exprs = ['f16tof32(%s)' % a, 'f16tof32(%s >> 16)' % a, 'f16tof32(%s)' % b, 'f16tof32(%s >> 16)' % b]
            words = ['U(%s)' % e for e in exprs]
        elif kind in ('uint', 'int'):
            exprs = ['0u' if v == 'off' else self.u(v) for v in vals]
            words = exprs
        elif kind == 'bits':
            ud = self.pipe.ud(('mrtbits',))
            exprs = ['0.0' if v == 'off' else 'F((%s) ^ ud%d)' % (self.u(v), ud) for v in vals]
            words = None
        else:
            exprs = ['0.0' if v == 'off' else 'F(%s)' % self.u(v) for v in vals]
            words = ['0u' if v == 'off' else self.u(v) for v in vals]
        self.exports[name] = True
        expr = '%s(%s)' % ('uint4' if kind in ('uint', 'int') else 'float4', ', '.join(exprs))
        self.emit('o.%s = %s;' % (name, 'asint(%s)' % expr if kind == 'int' else expr))
        if name.startswith('param'):
            # The flat copy read by v_interp_mov_f32 carries the register words. A float copy would let
            # FXC fold asfloat() of a constant and flush integer bit patterns that are binary32
            # denormals (#1259). hlsl_vs() drops the line for attributes the pixel stage reads only
            # interpolated.
            self.emit('o.f%s = uint4(%s);' % (name, ', '.join(words)))
        return None

    # ---- output ------------------------------------------------------------------------------
    def body(self):
        """Statements of the whole program as structured HLSL."""
        out = []
        offsets = [b[0] for b in self.blocks]
        index = {off: i for i, off in enumerate(offsets)}
        loops = any(t and t[0] in ('jump', 'branch') and t[-1] <= b[0] for b in self.blocks for t in [b[2]])
        if len(self.blocks) == 1:
            return ['    ' + line for line in self.blocks[0][1]]

        def goto(off):
            if off not in index:
                raise Unsupported('branch into the middle of a block (+0x%x)' % off)
            return index[off]
        out.append('    uint pc = 0;')
        if loops:
            out.append('    [loop] for (uint guard = 0; guard < 65536 && pc != 0xffffu; ++guard) {')
        for i, (off, lines, term) in enumerate(self.blocks):
            out.append('    %sif (pc == %d) {' % ('else ' if loops and i else '', i))
            out += ['        ' + line for line in lines]
            nxt = i + 1 if i + 1 < len(self.blocks) else 0xffff
            if not term:
                out.append('        pc = %d;' % nxt)
            elif term[0] == 'end':
                out.append('        pc = 0xffffu;')
            elif term[0] == 'jump':
                out.append('        pc = %d;' % goto(term[1]))
            else:
                out.append('        pc = (%s) ? %d : %d;' % (term[1], goto(term[2]), nxt))
            out.append('    }')
        if loops:
            out.append('    }')
        return out

    def declarations(self):
        lines = []
        if self.used_s:
            lines.append('    uint %s;' % ', '.join('s%d = 0' % n for n in sorted(self.used_s)))
        if self.used_v:
            lines.append('    uint %s;' % ', '.join('v%d = 0' % n for n in sorted(self.used_v)))
        packed = self.pipe.packed_position_vgpr
        if self.kind == 'ps' and packed in self.used_v:
            lines.append('    v%d = (uint(i.pos.x) & 0xffffu) | ((uint(i.pos.y) & 0xffffu) << 16);' % packed)
        lines.append('    uint vcc = 0, exec = 1, m0 = 0, a = 0, w = 0; bool scc = false; float4 t = 0;')
        return lines


# name -> (source operand count, expression); operands are floats
VEC_F = {
    'v_add_f32': (2, '{0} + {1}'), 'v_sub_f32': (2, '{0} - {1}'), 'v_subrev_f32': (2, '{1} - {0}'),
    'v_mul_f32': (2, '{0} * {1}'), 'v_mul_legacy_f32': (2, 'mul_legacy({0}, {1})'),
    'v_min_f32': (2, 'min({0}, {1})'), 'v_max_f32': (2, 'max({0}, {1})'),
    'v_fmac_f32': (2, '{0} * {1} + {2}'), 'v_mac_f32': (2, 'mad32({0}, {1}, {2}, false)'),
    'v_fma_f32': (3, '{0} * {1} + {2}'), 'v_mad_f32': (3, 'mad32({0}, {1}, {2}, false)'),
    'v_mad_legacy_f32': (3, 'mad32({0}, {1}, {2}, true)'),
    'v_fmamk_f32': (3, '{0} * {1} + {2}'), 'v_fmaak_f32': (3, '{0} * {1} + {2}'),
    'v_madmk_f32': (3, 'mad32({0}, {1}, {2}, false)'), 'v_madak_f32': (3, 'mad32({0}, {1}, {2}, false)'),
    'v_rcp_f32': (1, '1.0 / {0}'), 'v_rcp_iflag_f32': (1, '1.0 / {0}'), 'v_rsq_f32': (1, 'rsqrt({0})'), 'v_sqrt_f32': (1, 'sqrt({0})'),
    'v_exp_f32': (1, 'exp2({0})'), 'v_log_f32': (1, 'log2({0})'), 'v_floor_f32': (1, 'floor({0})'), 'v_ceil_f32': (1, 'ceil({0})'),
    'v_trunc_f32': (1, 'trunc({0})'), 'v_fract_f32': (1, 'frac({0})'), 'v_rndne_f32': (1, 'round({0})'),
    'v_sin_f32': (1, 'sin({0} * 6.283185307)'), 'v_cos_f32': (1, 'cos({0} * 6.283185307)'),
    'v_med3_f32': (3, 'med3({0}, {1}, {2})'), 'v_min3_f32': (3, 'min({0}, min({1}, {2}))'), 'v_max3_f32': (3, 'max({0}, max({1}, {2}))'),
}
# Conversions whose result is binary32. The expression is the float value, so an output modifier
# (clamp, mul:2, mul:4, div:2) can wrap it the way it wraps any other float result. The uint-result
# conversions stay in VEC_U.
VEC_U_TO_F = {
    'v_cvt_f32_u32': (1, 'float({0})'), 'v_cvt_f32_i32': (1, 'float(int({0}))'),
    'v_cvt_f32_ubyte0': (1, 'float({0} & 255)'), 'v_cvt_f32_ubyte1': (1, 'float(({0} >> 8) & 255)'),
    'v_cvt_f32_ubyte2': (1, 'float(({0} >> 16) & 255)'), 'v_cvt_f32_ubyte3': (1, 'float({0} >> 24)'),
    'v_cvt_f32_f16': (1, 'f16tof32({0})'),
    'v_cvt_off_f32_i4': (1, '(float(int({0} << 28) >> 28) * 0.0625)'),
}
# operands are uints; the result is a uint expression
VEC_U = {
    'v_add_nc_u32': (2, '{0} + {1}'), 'v_sub_nc_u32': (2, '{0} - {1}'), 'v_subrev_nc_u32': (2, '{1} - {0}'),
    'v_mul_lo_u32': (2, '{0} * {1}'), 'v_and_b32': (2, '{0} & {1}'), 'v_or_b32': (2, '{0} | {1}'), 'v_xor_b32': (2, '{0} ^ {1}'),
    'v_not_b32': (1, '~{0}'), 'v_lshlrev_b32': (2, '{1} << ({0} & 31)'), 'v_lshrrev_b32': (2, '{1} >> ({0} & 31)'),
    'v_ashrrev_i32': (2, 'uint(int({1}) >> ({0} & 31))'), 'v_min_u32': (2, 'min({0}, {1})'), 'v_max_u32': (2, 'max({0}, {1})'),
    'v_min_i32': (2, 'uint(min(int({0}), int({1})))'), 'v_max_i32': (2, 'uint(max(int({0}), int({1})))'),
    'v_cvt_u32_f32': (1, 'uint(max(F({0}), 0.0))'),
    'v_cvt_f16_f32': (1, 'f32tof16(F({0}))'),
    'v_bfe_u32': (3, '({0} >> ({1} & 31)) & ((1u << ({2} & 31)) - 1)'), 'v_mad_u32_u24': (3, '({0} & 0xffffffu) * ({1} & 0xffffffu) + {2}'),
    'v_lshl_add_u32': (3, '({0} << ({1} & 31)) + {2}'), 'v_add_lshl_u32': (3, '({0} + {1}) << ({2} & 31)'), 'v_add3_u32': (3, '{0} + {1} + {2}'),
    'v_or3_b32': (3, '{0} | {1} | {2}'), 'v_and_or_b32': (3, '({0} & {1}) | {2}'), 'v_lshl_or_b32': (3, '({0} << ({1} & 31)) | {2}'),
    'v_xad_u32': (3, '({0} ^ {1}) + {2}'), 'v_bfi_b32': (3, '({0} & {1}) | (~{0} & {2})'),
    'v_mul_u32_u24': (2, '({0} & 0xffffffu) * ({1} & 0xffffffu)'),
    # Per-lane bit operations (AMD RDNA 2 ISA reference). The bitscan results count from the most significant
    # bit for the "h" forms and read 0xffffffff when no bit qualifies; firstbitlow(0) is already 0xffffffff.
    # v_bfe_i32 extracts a field at S1[4:0] of width S2[4:0] (0 reads 0) and sign-extends it from its top bit,
    # taking the bits above bit 31 as copies of bit 31; v_bcnt_u32_b32 adds S1 to the population count of S0.
    'v_bfe_i32': (3, '(({2} & 31u) == 0u ? 0u : asuint(asint({0} << (32u - min(({1} & 31u) + ({2} & 31u), 32u))) '
                     '>> (32u - min(({1} & 31u) + ({2} & 31u), 32u) + ({1} & 31u))))'),
    'v_bcnt_u32_b32': (2, 'countbits({0}) + {1}'), 'v_ffbl_b32': (1, 'firstbitlow({0})'),
    'v_ffbh_u32': (1, '({0} == 0u ? 0xffffffffu : 31u - firstbithigh({0}))'),
    'v_ffbh_i32': (1, '({0} == 0u || {0} == 0xffffffffu ? 0xffffffffu : 31u - firstbithigh(asint({0}) < 0 ? ~{0} : {0}))'),
    'v_bfrev_b32': (1, 'reversebits({0})'),
}
# scalar ops: (expression over uint sources, SCC rule with $ for the result or None)
SCA = {
    's_mov_b32': ('{0}', None), 's_mov_b64': ('{0}', None), 's_movk_i32': ('{0}', None),
    's_and_b32': ('{0} & {1}', '$ != 0'), 's_or_b32': ('{0} | {1}', '$ != 0'), 's_xor_b32': ('{0} ^ {1}', '$ != 0'),
    's_andn2_b32': ('{0} & ~{1}', '$ != 0'), 's_orn2_b32': ('{0} | ~{1}', '$ != 0'), 's_not_b32': ('~{0}', '$ != 0'),
    's_and_b64': ('{0} & {1}', '$ != 0'), 's_or_b64': ('{0} | {1}', '$ != 0'), 's_xor_b64': ('{0} ^ {1}', '$ != 0'),
    's_andn2_b64': ('{0} & ~{1}', '$ != 0'), 's_orn2_b64': ('{0} | ~{1}', '$ != 0'), 's_not_b64': ('~{0}', '$ != 0'),
    's_nand_b64': ('~({0} & {1})', '$ != 0'), 's_nor_b64': ('~({0} | {1})', '$ != 0'), 's_xnor_b64': ('~({0} ^ {1})', '$ != 0'),
    's_lshl_b32': ('{0} << ({1} & 31)', '$ != 0'), 's_lshr_b32': ('{0} >> ({1} & 31)', '$ != 0'),
    's_ashr_i32': ('uint(int({0}) >> ({1} & 31))', '$ != 0'),
    's_add_i32': ('{0} + {1}', None), 's_add_u32': ('{0} + {1}', None), 's_sub_i32': ('{0} - {1}', None), 's_sub_u32': ('{0} - {1}', None),
    's_mul_i32': ('{0} * {1}', None), 's_min_u32': ('min({0}, {1})', None), 's_max_u32': ('max({0}, {1})', None),
    's_min_i32': ('uint(min(int({0}), int({1})))', None), 's_max_i32': ('uint(max(int({0}), int({1})))', None),
    's_bfe_u32': ('({0} >> ({1} & 31)) & ((1u << (({1} >> 16) & 127)) - 1)', '$ != 0'),
    's_brev_b32': ('reversebits({0})', None),
    # s_bfm_b32: a mask of S0[4:0] ones shifted left by S1[4:0]; SCC unchanged.
    's_bfm_b32': ('(((1u << ({0} & 31)) - 1) << ({1} & 31))', None),
}
for _name, _op in (('eq', '=='), ('lg', '!='), ('gt', '>'), ('ge', '>='), ('lt', '<'), ('le', '<=')):
    for _typ, _cast in (('i32', 'int({0}) %s int({1})'), ('u32', '{0} %s {1}')):
        def _make(op=_op, cast=_cast):
            def handler(self, ops, mods):
                self.emit('scc = %s;' % (cast % op).format(self.u(ops[0]), self.u(ops[1])))
            return handler
        setattr(Stage, 'op_s_cmp_%s_%s' % (_name, _typ), _make())
        setattr(Stage, 'op_s_cmpk_%s_%s' % (_name, _typ), _make())


def parse_root(blob):
    if not blob or blob[:4] != b'DXBC':
        raise Unsupported('root signature blob missing')
    n, = struct.unpack_from('<I', blob, 28)
    for o in struct.unpack_from('<%dI' % n, blob, 32):
        if blob[o:o + 4] == b'RTS0':
            d = blob[o + 8:]
            ver, nparam, poff, nsamp, soff, flags = struct.unpack_from('<6I', d, 0)
            params = []
            for i in range(nparam):
                ptype, vis, off = struct.unpack_from('<3I', d, poff + 12 * i)
                p = dict(type=ptype, vis=vis)
                if ptype == 1:
                    p['reg'], p['space'], p['count'] = struct.unpack_from('<3I', d, off)
                params.append(p)
            return dict(flags=flags, params=params)
    raise Unsupported('no RTS0 part')


# Run-time sampler heap indexing: an s_load from the sampler heap with a register offset becomes a
# read of a bounded table, SamplerState qh[SAMPLER_HEAP_SIZE] : register(s0, space1), the form a
# desktop SM 5.1 / 6.0 shader can index at run time. Opt-in with X2W_SAMPLER_HEAP=1 until the
# stand-in binds that table; otherwise such a load keeps the default-sampler shortcut.
#
# Contract at the manifest boundary ('smpheap <count> <space> fallback <entry>'):
# - entries 0 .. SAMPLER_HEAP_FALLBACK - 1 hold the title's sampler-heap entries, entry n at byte
#   offset 16 * n;
# - entry SAMPLER_HEAP_FALLBACK is reserved: the host fills it with a fallback sampler;
# - the shader computes the entry itself (x2w_sampler_entry): a byte offset that is not a multiple
#   of 16, or that lies at or past 16 * SAMPLER_HEAP_FALLBACK, reads the fallback entry. The host
#   cannot check the offset, which is computed in the shader, so the guard is in the shader.
SAMPLER_HEAP_SIZE = 2048  # D3D12's limit for a shader-visible sampler heap
SAMPLER_HEAP_SPACE = 1
SAMPLER_HEAP_FALLBACK = SAMPLER_HEAP_SIZE - 1


# DXGI_FORMAT values for integer RTVs and the binary32 float targets.
UINT_RTVS = frozenset((3, 7, 12, 17, 25, 30, 36, 42, 50, 57, 62))
SINT_RTVS = frozenset((4, 8, 14, 18, 32, 38, 43, 52, 59, 64))
FLOAT32_RTVS = frozenset((2, 6, 16, 41))


class Pipeline:
    vertex_kind = 'vs'
    uses_pack16_rne = False
    uses_gather_2d_lod = False
    uses_sad = ()
    srv_formats = ()
    uavs = None  # key -> dict(index, kind): images a pixel stage stores to (u registers); None until the first store

    def __init__(self, vs_code, ps_code, root_blob, sampler_heap=None, rtv_formats=(), ps_hw_inputs=None, vs_fetch=None,
                 refusals=None, decoded=None, ps_attr_map=None, srv_formats=()):
        self.root = parse_root(root_blob)
        self.vs_fetch = vs_fetch  # stream_fetch(): the loads the vertex stage's entry call stands for
        self.packed_position_vgpr = None
        if ps_hw_inputs is not None:
            ena, addr = ps_hw_inputs
            if ena != addr or not 0 <= ena <= 0xffff:
                raise Unsupported('unverified pixel input enable/address combination')
            if ena & (1 << 15):
                # SPI PS inputs 0..6 supply IJ pairs (the pull model supplies three);
                # line stipple through sample coverage each occupy one VGPR.
                widths = (2, 2, 2, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1)
                self.packed_position_vgpr = sum(w for bit, w in enumerate(widths) if addr & (1 << bit))
        self.rtv_formats = rtv_formats
        self.sampler_heap = os.environ.get('X2W_SAMPLER_HEAP') == '1' if sampler_heap is None else sampler_heap
        self.uses_sampler_heap = False
        self.resources = {}   # key -> dict(index, kind)
        self.samplers = {}    # key -> index
        self.uds = []         # entry-constant sources
        self.notes = set()    # shortcuts taken (written to the manifest as comments)
        self.modified = set()
        self.rect = self.ud(('rect',))
        if refusals is None:
            self.vs = Stage(self, self.vertex_kind, vs_code)
            self.ps = Stage(self, 'ps', ps_code) if ps_code is not None else None
        else:
            decoded = {} if decoded is None else decoded
            self.vs = Stage(self, self.vertex_kind, vs_code, refusals=refusals, decoded=decoded.get('vs'))
            self.ps = Stage(self, 'ps', ps_code, refusals=refusals, decoded=decoded.get('ps')) if ps_code is not None else None
        if self.ps is None:
            self.params, self.flat, self.f16_plain = [], [], []
            if 'pos' not in self.vs.exports:
                raise Unsupported('vertex stage exports no position')
            return
        self.params = sorted(int(k[5:]) for k in self.vs.exports if k.startswith('param'))
        self.flat = sorted(a for a, how in self.ps.attrs.items() if 'flat' in how)
        missing = [a for a in self.ps.attrs if a not in self.params]
        if missing:
            raise Unsupported('pixel stage reads attributes %s the vertex stage does not export' % missing)
        if 'pos' not in self.vs.exports:
            raise Unsupported('vertex stage exports no position')

    def mrt_kind(self, slot):
        if slot >= len(self.rtv_formats):
            return 'float'
        fmt = self.rtv_formats[slot]
        if fmt in UINT_RTVS:
            if fmt != 42:
                raise Unsupported('unverified UINT RTV format %d' % fmt)
            return 'uint'
        if fmt in SINT_RTVS:
            if fmt != 43:
                raise Unsupported('unverified SINT RTV format %d' % fmt)
            return 'int'
        return 'bits' if fmt in FLOAT32_RTVS else 'float'

    def ud(self, source):
        if source not in self.uds:
            self.uds.append(source)
        return self.uds.index(source)

    def resource(self, key, kind):
        r = self.resources.setdefault(key, dict(index=len(self.resources), kind=kind))
        if r['kind'] != kind:
            raise Unsupported('resource %s used as %s and %s' % (key, r['kind'], kind))
        return 'r%d' % r['index']

    def sampler(self, key, comparison=False):
        if not isinstance(key[0], int):
            raise Unsupported('sampler from a descriptor')
        # A comparison sampler is its own binding even when its words equal an ordinary sampler's.
        key = ('cmp',) + tuple(key) if comparison else key
        return 'q%d' % self.samplers.setdefault(key, len(self.samplers))

    def common(self):
        out = ['#define F(x) asfloat(x)', '#define U(x) asuint(x)',
               'float mul_legacy(float a, float b) { return (a == 0.0 || b == 0.0) ? 0.0 : a * b; }',
               # Match pixel_aot: round the product before adding and flush denormals to signed zero.
               'float ftz32(float value) {',
               '    uint bits = asuint(value);',
               # Select the mask as an integer so FXC preserves the sign of a flushed zero.
               '    uint mask = (bits & 0x7fffffffu) < 0x00800000u ? 0x80000000u : 0xffffffffu;',
               '    return asfloat(bits & mask);',
               '}',
               'float mad32(float a, float b, float c, bool legacy) {',
               '    a = ftz32(a);',
               '    b = ftz32(b);',
               '    precise float product = ftz32(legacy ? mul_legacy(a, b) : a * b);',
               '    precise float sum = product + ftz32(c);',
               '    return ftz32(sum);',
               '}',
               'float med3(float a, float b, float c) { return max(min(a, b), min(max(a, b), c)); }',
               // The 64-bit shifts (issue #1815): a pair of words, so the count never reaches 32 in a 32-bit
               // shift and an in-place destination keeps the original value (the caller passes both words in).
               'uint2 shl64(uint lo, uint hi, uint sh) { sh &= 63u; uint sb = sh & 31u, sc = (32u - sb) & 31u;'
               ' uint cy = sb == 0u ? 0u : (lo >> sc);'
               ' return uint2(sh < 32u ? (lo << sb) : 0u, sh < 32u ? ((hi << sb) | cy) : (lo << sb)); }',
               'uint2 shr64u(uint lo, uint hi, uint sh) { sh &= 63u; uint sb = sh & 31u, sc = (32u - sb) & 31u;'
               ' uint cy = sb == 0u ? 0u : (hi << sc); uint h = hi >> sb;'
               ' return uint2(sh < 32u ? ((lo >> sb) | cy) : h, sh < 32u ? h : 0u); }',
               'uint2 shr64i(uint lo, uint hi, uint sh) { sh &= 63u; uint sb = sh & 31u, sc = (32u - sb) & 31u;'
               ' uint cy = sb == 0u ? 0u : (hi << sc); uint h = asuint(asint(hi) >> sb);'
               ' return uint2(sh < 32u ? ((lo >> sb) | cy) : h, sh < 32u ? h : asuint(asint(hi) >> 31u)); }',
               // The ten binary32 classes v_cmp_class_f32 tests, from the raw encoding (#1795).
               'uint class32(float value) { uint bits = asuint(value), magnitude = bits & 0x7fffffffu;'
               ' bool negative = bits >= 0x80000000u;'
               ' if (magnitude > 0x7f800000u) return (bits & 0x00400000u) != 0u ? 0x2u : 0x1u;'
               ' if (magnitude == 0x7f800000u) return negative ? 0x4u : 0x200u;'
               ' if (magnitude >= 0x00800000u) return negative ? 0x8u : 0x100u;'
               ' if (magnitude != 0u) return negative ? 0x10u : 0x80u;'
               ' return negative ? 0x20u : 0x40u; }',
               // The ten binary16 classes for v_cmp_class_f16, from bits [15:0] of the operand word
               // (ported from tools/pixel_aot.py): sNaN qNaN -inf -normal -subnormal -0 +0 +subnormal +normal +inf.
               'uint class16(uint bits) { uint h = bits & 0xffffu, magnitude = h & 0x7fffu;'
               ' bool negative = h >= 0x8000u;'
               ' if (magnitude > 0x7c00u) return (h & 0x0200u) != 0u ? 0x2u : 0x1u;'
               ' if (magnitude == 0x7c00u) return negative ? 0x4u : 0x200u;'
               ' if (magnitude >= 0x0400u) return negative ? 0x8u : 0x100u;'
               ' if (magnitude != 0u) return negative ? 0x10u : 0x80u;'
               ' return negative ? 0x20u : 0x40u; }',
               'uint ldu(ByteAddressBuffer b, uint a) { uint k = (a & 3) * 8, lo = b.Load(a & ~3u); if (k == 0) return lo; '
               'return (lo >> k) | (b.Load((a & ~3u) + 4) << (32 - k)); }',
               'cbuffer entry : register(b0) { %s };' % ' '.join('uint ud%d;' % i for i in range(len(self.uds)))]
        for key, r in sorted(self.resources.items(), key=lambda kv: kv[1]['index']):
            out.append('%s r%d : register(t%d);' % ({'buf': 'ByteAddressBuffer', 'tex1d': 'Texture1D<float4>', 'tex2d': 'Texture2D<float4>', 'tex2da': 'Texture2DArray<float4>', 'tex3d': 'Texture3D<float4>'}[r['kind']], r['index'], r['index']))
        for key, i in self.samplers.items():
            out.append('%s q%d : register(s%d);' % ('SamplerComparisonState' if key[0] == 'cmp' else 'SamplerState', i, i))
        if self.uses_sampler_heap:
            out.append('SamplerState qh[%d] : register(s0, space%d);' % (SAMPLER_HEAP_SIZE, SAMPLER_HEAP_SPACE))
            out.append('uint x2w_sampler_entry(uint offset) { return (offset & 15u) != 0 || offset >= %du ? %du : offset >> 4; }'
                       % (16 * SAMPLER_HEAP_FALLBACK, SAMPLER_HEAP_FALLBACK))
        return out

    def interface(self, for_pixel):
        fields = []
        for p in range(max(self.params) + 1 if self.params else 0):
            fields.append('    float4 param%d : PARAM%d;' % (p, p))
        # Attributes read without interpolation get a second, flat copy of the vertex output.
        for p in self.flat:
            fields.append('    nointerpolation uint4 fparam%d : FPARAM%d;' % (p, p))
        fields.append('    float4 pos : SV_Position;')
        return fields

    def hlsl_vs(self):
        s = self.vs
        out = self.common() + ['struct VSOut {'] + self.interface(False) + ['};', 'VSOut body(uint vid) {', '    VSOut o = (VSOut)0;']
        s.used_v.add(0)
        flat = re.compile(r'\s*o\.fparam(\d+) = ')
        body = [line for line in s.body() if not (flat.match(line) and int(flat.match(line).group(1)) not in self.flat)]
        if s.fetched:
            # Behind a fetch call a register read as 0 by default would hide a missing input, so every read must be
            # of a register defined on all paths. Proven here only for the entry block (straight-line code from the
            # call to the first branch or branch target): v0, the fetched registers, and registers an unpredicated
            # write in that block assigned before the read. A write under an exec predicate or in any later block
            # does not define a register.
            written = {0}
            for index, (start, lines, term) in enumerate(s.blocks):
                for line in lines + ([term[1]] if term and term[0] == 'branch' else []):
                    target = re.match(r'v(\d+) = (.*)', line)
                    unwritten = {int(n) for n in re.findall(r'\bv(\d+)\b', target.group(2) if target else line)} - written
                    if unwritten:
                        raise Unsupported('vertex stage reads v%d, which neither the fetch nor an unconditional write '
                                          'before the first branch defines' % min(unwritten))
                    if target and index == 0 and start == 0:
                        written.add(int(target.group(1)))
        out += s.declarations() + ['    v0 = vid;'] + body + ['    return o;', '}',
            '// Rectangle lists arrive as six vertices per rectangle: corners 0 1 2, 0 2 and the fourth corner 0 + 2 - 1.',
            'VSOut main(uint vid : SV_VertexID) {',
            '    if (ud%d == 0) return body(vid);' % self.rect,
            '    uint first = (vid / 6) * 3, corner = vid % 6;',
            '    if (corner < 3) return body(first + corner);',
            '    if (corner == 3) return body(first);',
            '    if (corner == 4) return body(first + 2);',
            '    VSOut p = body(first), q = body(first + 1), r = body(first + 2), o;']
        for p in range(max(self.params) + 1 if self.params else 0):
            out.append('    o.param%d = p.param%d + r.param%d - q.param%d;' % (p, p, p, p))
        out += ['    o.fparam%d = p.fparam%d;' % (p, p) for p in self.flat]
        out += ['    o.pos = p.pos + r.pos - q.pos;', '    return o;', '}']
        return '\n'.join(out) + '\n'

    def hlsl_ps(self):
        s = self.ps
        targets = sorted(int(k[3:]) for k in s.exports if k.startswith('mrt'))
        depth = 'depth' in s.exports
        out = self.common() + ['struct PSIn {'] + self.interface(True) + ['};']
        if not targets and not depth:
            out += ['void main(PSIn i) {'] + s.declarations() + s.body() + ['}']
            return '\n'.join(out) + '\n'
        out.append('struct PSOut {')
        if depth:
            out.append('    float depth : SV_Depth;')
        for t in targets:
            kind = self.mrt_kind(t)
            out.append('    %s mrt%d : SV_Target%d;' % ({'int': 'int4', 'uint': 'uint4'}.get(kind, 'float4'), t, t))
        out += ['};', 'PSOut main(PSIn i) {', '    PSOut o = (PSOut)0;']
        out += s.declarations() + s.body() + ['    return o;', '}']
        return '\n'.join(out) + '\n'

    def manifest(self):
        lines = ['x2w-manifest 1', 'ud %d' % len(self.uds)]
        for i, source in enumerate(self.uds):
            lines.append('udsrc %d %s' % (i, ' '.join(str(x) for x in source)))
        for key, r in sorted(self.resources.items(), key=lambda kv: kv[1]['index']):
            source, off = key
            where = {'table': 'table %d %d' % (source[1] if len(source) > 1 else 0, off or 0), 'vbtab': 'vb %d' % ((off or 0) // 16),
                     'root': 'root %d' % (source[1] if len(source) > 1 else 0)}.get(source[0])
            if where is None:
                raise Unsupported('resource through %s' % (source,))
            lines.append('srv %d %s %s' % (r['index'], r['kind'], where))
        for key, i in self.samplers.items():
            comparison = key[0] == 'cmp'
            lines.append('%s %d %s' % ('smpc' if comparison else 'smp', i, ' '.join('0x%x' % w for w in key[comparison:])))
        if self.uses_sampler_heap:
            lines.append('smpheap %d %d fallback %d' % (SAMPLER_HEAP_SIZE, SAMPLER_HEAP_SPACE, SAMPLER_HEAP_FALLBACK))
        lines.append('params %d' % (max(self.params) + 1 if self.params else 0))
        if self.ps is None:
            lines.append('ps none')
            return '\n'.join(lines) + '\n'
        lines.append('targets %s' % ' '.join(k[3:] for k in sorted(self.ps.exports) if k.startswith('mrt')))
        return '\n'.join(lines) + '\n'

def fnv(data, h=1469598103934665603):
    for b in data:
        h = ((h ^ b) * 1099511628211) & 0xffffffffffffffff
    return h


def pipeline_key(vs_code, ps_code, root_blob, layout=b''):
    """Hash the startup translation revision and the three blobs in runtime order, then a stream's input layout."""
    key = fnv(root_blob, fnv(ps_code, fnv(vs_code, fnv(b'x2w-startup-shader-15'))))
    return fnv(layout, key) if layout else key


def compute_key(cs_code, data, root_blob, desc, state_a=b'', state_b=b''):
    """The runtime key of a device slot 67 compute request (req_<key>.c67, .data, .root, .desc, .sa, .sb): the
    graphics revision followed by "cs", then each part preceded by its length as a little-endian u64: the code,
    the words 7-8 blob, the root signature, the 12 description words with their pointers zeroed and the two
    state blocks."""
    key = fnv(b'x2w-startup-shader-15cs')
    for part in (cs_code, data, root_blob, desc, state_a, state_b):
        key = fnv(part, fnv(struct.pack('<Q', len(part)), key))
    return key
S_ENDPGM = 0xbf810000


def compute_entry(state_a):
    """(COMPUTE_PGM_RSRC2, group size) from state block A. A hypothesis from a local observation, not decoded
    by the stand-in (docs/startup-runtime.md#compute-pipeline-translation): dword 3 = X | Y << 16, dword 4
    bits 15:0 = Z, dword 6 = COMPUTE_PGM_RSRC2."""
    if len(state_a) != 28:
        raise Unsupported('compute state block A is %d bytes, not 28' % len(state_a))
    words = struct.unpack('<7I', state_a)
    return words[6], (words[3] & 0xffff, words[3] >> 16, words[4] & 0xffff)


def load_compute_request(out_dir, key, refusals=None):
    """The ComputePipeline of a slot 67 request, after checking the files against .desc and the key."""
    name = os.path.join(out_dir, 'req_%016x' % key)
    try:
        parts = {ext: open(name + ext, 'rb').read() for ext in ('.c67', '.data', '.root', '.desc', '.sa', '.sb')}
    except OSError:
        raise Unsupported('unreadable shader request') from None  # the error text would name the cache path
    desc = parts['.desc']
    if len(desc) != 96:
        raise Unsupported('compute description is %d bytes, not 96' % len(desc))
    words = struct.unpack('<12Q', desc)
    for ext, word in (('.c67', 6), ('.data', 8), ('.sa', 2), ('.sb', 4)):
        if len(parts[ext]) != words[word]:
            raise Unsupported('compute request %s is %d bytes, the description says %d' % (ext, len(parts[ext]), words[word]))
    if compute_key(parts['.c67'], parts['.data'], parts['.root'], desc, parts['.sa'], parts['.sb']) != key:
        raise Unsupported('compute request files do not match the key in their name')
    code = parts['.c67']
    if not code or len(code) % 4:
        raise Unsupported('compute code of %d bytes is not whole dwords' % len(code))
    if S_ENDPGM not in struct.unpack('<%dI' % (len(code) // 4), code):
        raise Unsupported('compute code without s_endpgm')
    rsrc2, group = compute_entry(parts['.sa'])
    return ComputePipeline(code, parts['.root'], rsrc2, group, refusals=refusals)


def compute_refusals(out_dir, key):
    """Refusal sites of one slot 67 compute request, without publishing a translation."""
    refusals, op = [], 'request loading'
    try:
        pipe = load_compute_request(out_dir, key, refusals)
        op = 'hlsl_cs'
        pipe.hlsl_cs()
        op = 'manifest'
        pipe.manifest()
    except Unsupported as reason:
        refusals.append((str(reason), 'pipeline'))
    except Exception as error:  # noqa: BLE001 - one request must not end the census
        refusals.append((handler_error(op, error, refusals), 'pipeline'))
    return refusals


def log_unvalidated_lod(key, pipe):
    """Once per translated module: the sample forms that read a heap sampler keep its LOD fields unvalidated (#2221)."""
    forms = getattr(pipe, 'lod_unvalidated', None)  # test stand-ins for Pipeline do not set it
    if forms:
        print('%016x: %s through a sampler heap: sampler MIN_LOD, MAX_LOD and LOD bias are not validated (#2221)'
              % (key, ', '.join(sorted(forms))), file=sys.stderr, flush=True)


def translate_compute(out_dir, key):
    """Writes <key>.cs.hlsl, then <key>.cs.cso (DXC cs_6_0) and <key>.txt last, or <key>.fail with the reason.
    X2W_FXC set to the empty string publishes the manifest alone, as for graphics."""
    name = os.path.join(out_dir, '%016x' % key)
    with suppress(FileNotFoundError):
        os.remove(name + '.txt')  # the result is unpublished until validation and compilation finish
    try:
        pipe = load_compute_request(out_dir, key)
        hlsl, manifest = pipe.hlsl_cs(), pipe.manifest()
    except Unsupported as reason:
        reason = fail_reason(reason)
        open(name + '.fail', 'w').write(reason + '\n')
        return 'unsupported: %s' % reason
    log_unvalidated_lod(key, pipe)
    open(name + '.cs.hlsl', 'w').write(hlsl)
    if os.environ.get('X2W_FXC') != '':
        compiler = find_dxc()
        if compiler is None:
            message = 'dxc not found (X2W_DXC=%s): a compute stage needs shader model 6.0' % os.environ.get('X2W_DXC', '<unset>')
            open(name + '.fail', 'w').write(message + '\n')
            return message
        import subprocess
        with tempfile.TemporaryDirectory(prefix='.dxc-', dir=out_dir) as staging:
            bytecode = os.path.join(staging, 'cs.cso')
            r = subprocess.run([compiler, '-Gis', '-T', 'cs_6_0', '-E', 'main', '-Fo', bytecode, name + '.cs.hlsl'],
                               capture_output=True, text=True)
            if r.returncode or not has_bytecode(bytecode):
                # DXC names the source path; keep only the diagnostic after it.
                lines = [re.sub(r'^.*?\.cs\.hlsl(:\d+:\d+)?:\s*', '', line) for line in r.stderr.strip().splitlines()]
                detail = lines[0] if lines else ('?' if r.returncode else 'no bytecode produced')
                open(name + '.fail', 'w').write('dxc cs: %s\n' % detail[:500])
                return 'dxc cs failed: %s' % detail[:500]
            os.replace(bytecode, name + '.cs.cso')
    temporary = '%s.txt.%d.tmp' % (name, os.getpid())
    with open(temporary, 'w') as result:
        result.write(manifest)
    os.replace(temporary, name + '.txt')
    with suppress(FileNotFoundError):
        os.remove(name + '.fail')
    return 'ok'

DEFAULT_FXC = r'C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\fxc.exe'
KITS_BIN = r'C:\Program Files (x86)\Windows Kits\10\bin'


def find_fxc():
    """The fxc.exe to compile with, '' when X2W_FXC is the empty string (manifest only), or None when none exists.

    Order: X2W_FXC, the 10.0.26100.0 SDK, the newest KITS_BIN/<version>/x64, then PATH."""
    configured = os.environ.get('X2W_FXC')
    if configured == '':
        return ''
    if configured and os.path.isfile(configured):
        return configured
    if os.path.isfile(DEFAULT_FXC):
        return DEFAULT_FXC
    versions = []
    try:
        for entry in os.listdir(KITS_BIN):
            if re.fullmatch(r'[0-9]+(?:\.[0-9]+)*', entry):
                versions.append((tuple(map(int, entry.split('.'))), entry))
    except OSError:
        pass
    for _, entry in sorted(versions, reverse=True):
        candidate = os.path.join(KITS_BIN, entry, 'x64', 'fxc.exe')
        if os.path.isfile(candidate):
            return candidate
    import shutil
    return shutil.which('fxc.exe')


def translate_to(out_dir, key, vs_code, ps_code, root_blob, check=True, rtv_formats=(), rtv_request='', vs_fetch=None):
    """Writes <key>.vs.hlsl, <key>.ps.hlsl and <key>.txt (last), or <key>.fail with the reason.

    With check, the pair is compiled to .vs.cso and .ps.cso first and a missing fxc is a failure.
    check=False, or X2W_FXC set to the empty string, publishes the manifest without bytecode."""
    name = os.path.join(out_dir, '%016x' % key)
    try:
        pipe = Pipeline(vs_code, ps_code, root_blob, rtv_formats=rtv_formats, vs_fetch=vs_fetch)
        vs, ps, manifest = pipe.hlsl_vs(), pipe.hlsl_ps(), pipe.manifest()
        if rtv_request:
            manifest += rtv_request + '\n'
    except Unsupported as reason:
        open(name + '.fail', 'w').write(str(reason) + '\n')
        return 'unsupported: %s' % reason
    open(name + '.vs.hlsl', 'w').write(vs)
    open(name + '.ps.hlsl', 'w').write(ps)
    fxc = find_fxc() if check else ''
    if fxc is None:
        message = 'fxc.exe not found (X2W_FXC=%s): set X2W_FXC to its path or install the Windows SDK' % os.environ.get('X2W_FXC', '<unset>')
        open(name + '.fail', 'w').write(message + '\n')
        return message
    if fxc:
        import subprocess
        for stage, profile in (('vs', 'vs_5_1'), ('ps', 'ps_5_1')):
            r = subprocess.run([fxc, '/nologo', '/T', profile, '/E', 'main', '/Fo', name + '.%s.cso' % stage, name + '.%s.hlsl' % stage],
                               capture_output=True, text=True)
            if r.returncode:
                open(name + '.fail', 'w').write('fxc %s: %s\n' % (stage, r.stderr[-2000:]))
                return 'fxc %s failed: %s' % (stage, r.stderr.strip().splitlines()[-1] if r.stderr.strip() else '?')
    temporary = '%s.txt.%d.tmp' % (name, os.getpid())
    with open(temporary, 'w') as result:
        result.write(manifest)
    os.replace(temporary, name + '.txt')
    return 'ok'


def classify(blob):
    return classify_decoded(decode(strip_header(blob)))


def load_request(out_dir, key, refusals=None, pair_info=None):
    """Read a base request or an RTV variant and classify the source stages."""
    decoded = {}

    def classify_stage(blob):
        if refusals is None:
            return classify(blob)
        ins = decode(strip_header(blob), refusals)
        kind = classify_decoded(ins)
        decoded[kind] = ins
        return kind

    name = os.path.join(out_dir, 'req_%016x' % key)
    request = ''
    formats = ()
    base = key
    if os.path.isfile(name + '.rtv'):
        with open(name + '.rtv', encoding='ascii') as file:
            request = file.read(256)
        if not re.fullmatch(r'rtv1 [0-9]{1,20}(?: [0-9]{1,10}){1,8}', request) or fnv(request.encode('ascii')) != key:
            raise Unsupported('invalid RTV variant request')
        base, *formats = map(int, request.split()[1:])
        if base > 0xffffffffffffffff or any(fmt > 0xffffffff for fmt in formats):
            raise Unsupported('invalid RTV variant values')
        name = os.path.join(out_dir, 'req_%016x' % base)
    a, b, root = (open(name + ext, 'rb').read() for ext in ('.a', '.b', '.root'))
    if pair_info is not None and not request:
        import hashlib
        pair_info.append((hashlib.sha256(struct.pack('<Q', len(a)) + a + b).digest(), root))
    # A stream with an input layout publishes it as .ia (before .a) and hashes it into the key.
    layout = open(name + '.ia', 'rb').read() if os.path.isfile(name + '.ia') else b''
    if request and pipeline_key(a, b, root, layout) != base:
        raise Unsupported('RTV variant source key mismatch')
    fetch = attr_map = None
    if a.startswith(b'DXBC') or b.startswith(b'DXBC'):
        # Slot 86 records complete containers in canonical (VS, PS) order.
        # Keep the container bytes in the cache key; translate only the bounded native code.
        import zlib
        from pixel_aot import container_code
        from resource_inventory import parts
        try:
            container, ps_container = a, b
            a, b = (container_code(parts(blob)[b'DXIL'], stage=stage)
                    for blob, stage in ((a, 1), (b, 0)))
            attr_map = stream_interface(container, ps_container)
            if starts_with_fetch_call(a):
                fetch = stream_fetch(container, layout.decode('ascii') if layout else None)
        except (KeyError, ValueError, IndexError, struct.error, zlib.error) as error:
            raise Unsupported('invalid stream shader: %s' % error) from error
        if classify_stage(a) != 'vs' or classify_stage(b) != 'ps':
            raise Unsupported('stream stage and shader export disagree')
        kinds = {'vs': a, 'ps': b}
    else:
        kinds = {classify_stage(a): a, classify_stage(b): b}
    return kinds, root, formats, request, fetch, decoded, attr_map


def mesh57_key(ms, ps, root):
    value = fnv(b'x2w-startup-mesh-57-1')
    for part in (ms, ps, root):
        value = fnv(part, fnv(struct.pack('<Q', len(part)), value))
    return value


def translate_mesh57(out_dir, key):
    """Translate a fixed-descriptor mesh request of the accepted shape (mesh_skeleton), or raise Unsupported.

    Writes <key>.ms57.hlsl and the compiled .ms57.cso, their .ps57 counterparts if present, and <key>.ms57.txt (last).
    No runtime path loads these names yet."""
    source = os.path.join(out_dir, 'req_%016x' % key)
    ms, ps, root = (open(source + ext, 'rb').read() for ext in ('.m57', '.ps57', '.root'))
    if mesh57_key(ms, ps, root) != key:
        raise Unsupported('slot 57 mesh request key mismatch')
    parse_root(root)
    mesh = decode(strip_header(ms))
    exports = [assembly.split()[1] for _, _, op, assembly in mesh if op == 'exp']
    if not mesh or mesh[-1][2] != 's_endpgm' or not any(
        op == 's_sendmsg' and assembly == 's_sendmsg 0x9' for _, _, op, assembly in mesh
    ) or not any(target.startswith('pos') for target in exports) or 'target_20' not in exports:
        raise Unsupported('slot 57 mesh stage has no verified allocation and primitive export')
    if ps:
        pixel = decode(strip_header(ps))
        # A pixel shader with no colour output ends in a null-target export (LLVM SILateBranchLowering, Mesa ACO create_fs_null_export): depth-only.
        null_only = {a.split()[1] for _, _, op, a in pixel if op == 'exp'} == {'null'}
        if not pixel or pixel[-1][2] != 's_endpgm' or classify_decoded(pixel) != 'ps' and not null_only:
            raise Unsupported('slot 57 pixel stage has an incompatible export or termination')
        if null_only:
            raise Unsupported('slot 57 mesh with a depth-only pixel stage is not covered')
    pipe = MeshPipeline(ms, ps or None, root)
    sources = [('ms', 'ms_6_5', pipe.hlsl_ms())]
    if ps:
        sources.append(('ps', 'ps_6_0', pipe.hlsl_ps()))
    manifest = pipe.manifest()
    name = os.path.join(out_dir, '%016x' % key)
    for stage, _, hlsl in sources:
        open('%s.%s57.hlsl' % (name, stage), 'w').write(hlsl)
    if os.environ.get('X2W_FXC') != '':
        compiler = find_dxc()
        if compiler is None:
            raise Unsupported('dxc not found (X2W_DXC=%s): a mesh stage needs shader model 6.5' % os.environ.get('X2W_DXC', '<unset>'))
        import subprocess
        # Publish bytecode only after all requested stages compile successfully.
        with tempfile.TemporaryDirectory(prefix='.dxc-', dir=out_dir) as staging:
            for stage, profile, _ in sources:
                bytecode = os.path.join(staging, stage + '.cso')
                r = subprocess.run([compiler, '-Gis', '-T', profile, '-E', 'main', '-Fo', bytecode, '%s.%s57.hlsl' % (name, stage)],
                                   capture_output=True, text=True)
                if r.returncode or not has_complete_mesh57_bytecode(bytecode):
                    # DXC names the source path; keep only the diagnostic after it.
                    lines = [re.sub(r'^.*?57\.hlsl(:\d+:\d+)?:\s*', '', line) for line in r.stderr.strip().splitlines()]
                    raise Unsupported('dxc %s: %s' % (stage, (lines[0] if lines else 'missing or incomplete bytecode')[:500]))
            for stage, _, _ in sources:
                os.replace(os.path.join(staging, stage + '.cso'), '%s.%s57.cso' % (name, stage))
    temporary = '%s.ms57.txt.%d.tmp' % (name, os.getpid())
    with open(temporary, 'w') as result:
        result.write(manifest)
    os.replace(temporary, name + '.ms57.txt')
    with suppress(FileNotFoundError):
        os.remove(name + '.fail')
    return 'ok'


def translate_request(out_dir, key):
    """Translate a base shader request or an RTV variant referencing its source blobs."""
    name = os.path.join(out_dir, 'req_%016x' % key)
    request = ''
    formats = ()
    try:
        base = key
        if os.path.isfile(name + '.rtv'):
            with open(name + '.rtv', encoding='ascii') as file:
                request = file.read(256)
            if not re.fullmatch(r'rtv1 [0-9]{1,20}(?: [0-9]{1,10}){1,8}', request) or fnv(request.encode('ascii')) != key:
                raise Unsupported('invalid RTV variant request')
            base, *formats = map(int, request.split()[1:])
            if base > 0xffffffffffffffff or any(fmt > 0xffffffff for fmt in formats):
                raise Unsupported('invalid RTV variant values')
            name = os.path.join(out_dir, 'req_%016x' % base)
        a, b, root = (open(name + ext, 'rb').read() for ext in ('.a', '.b', '.root'))
        # A stream with an input layout publishes it as .ia (before .a) and hashes it into the key.
        layout = open(name + '.ia', 'rb').read() if os.path.isfile(name + '.ia') else b''
        if request and pipeline_key(a, b, root, layout) != base:
            raise Unsupported('RTV variant source key mismatch')
        fetch = None
        if a.startswith(b'DXBC') or b.startswith(b'DXBC'):
            # Slot 86 records complete containers in canonical (VS, PS) order.
            # Keep the container bytes in the cache key; translate only the bounded native code.
            import zlib
            from pixel_aot import container_code
            from resource_inventory import parts
            try:
                container = a
                a, b = (container_code(parts(blob)[b'DXIL'], stage=stage)
                        for blob, stage in ((a, 1), (b, 0)))
                if starts_with_fetch_call(a):
                    fetch = stream_fetch(container, layout.decode('ascii') if layout else None)
            except (KeyError, ValueError, IndexError, struct.error, zlib.error) as error:
                raise Unsupported('invalid stream shader: %s' % error) from error
            if classify(a) != 'vs' or classify(b) != 'ps':
                raise Unsupported('stream stage and shader export disagree')
            kinds = {'vs': a, 'ps': b}
        else:
            kinds = {classify(a): a, classify(b): b}
    except (Unsupported, OSError, UnicodeError) as reason:
        open(os.path.join(out_dir, '%016x.fail' % key), 'w').write(str(reason) + '\n')
        return 'unsupported: %s' % reason
    if set(kinds) != {'vs', 'ps'}:
        open(os.path.join(out_dir, '%016x.fail' % key), 'w').write('stages %s\n' % sorted(kinds))
        return 'stages %s' % sorted(kinds)
    return translate_to(out_dir, key, kinds['vs'], kinds['ps'], root, rtv_formats=formats, rtv_request=request, vs_fetch=fetch)


def main():
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    if sys.argv[1] == '--capture':
        # --capture <dir> <capture.bin>: write request files for every pipeline the capture binds
        import capture
        cap = capture.Capture(sys.argv[3])
        done = set()
        for c in cap.calls:
            w = c['args'][0]
            if c['slot'] != 25 or w in done:
                continue
            done.add(w)
            p = cap.psos.get(w)
            seen = {w}
            while p and p['path'] == 55 and p['base']:
                if p['base'] in seen:
                    print('pso %x: wrapper chain loops back to %x' % (w & 0xffffff, p['base'] & 0xffffff))
                    p = None
                    break
                seen.add(p['base'])
                p = cap.psos.get(p['base'])
            if not p or p['path'] != 57:
                continue
            if len(p['desc']) < 96:
                print('pso %x: pipeline desc is %d bytes, need 96' % (w & 0xffffff, len(p['desc'])))
                continue
            q = struct.unpack_from('<12Q', p['desc'], 0)
            a, b = cap.memory.read(q[5], q[6]), cap.memory.read(q[9], q[10])
            root = cap.roots.get(q[0], b'')
            if not a or not b or len(a) != q[6] or len(b) != q[10]:
                print('pso %x: code not captured' % (w & 0xffffff))
                continue
            key = pipeline_key(a, b, root)
            for ext, blob in (('.a', a), ('.b', b), ('.root', root)):
                open(os.path.join(out_dir, 'req_%016x%s' % (key, ext)), 'wb').write(blob)
            print('pso %x -> request %016x' % (w & 0xffffff, key))
    # translate every request that has no result yet
    for name in sorted(os.listdir(out_dir)):
        m = re.fullmatch(r'req_([0-9a-f]{16})\.(?:a|rtv)', name)
        if not m:
            continue
        key = int(m.group(1), 16)
        result = os.path.join(out_dir, '%016x' % key)
        mesh = m.group(2) == 'm57'
        stages = ('.cs.cso',) if m.group(2) == 'c67' else ('.ms57.cso', '.ps57.cso') if mesh else ('.vs.cso', '.ps.cso')
        ready_bytecode = has_complete_mesh57_bytecode if mesh else has_bytecode
        manifest = result + ('.ms57.txt' if mesh else '.txt')
        # A published manifest with bytecode (or manifest-only mode) is done; otherwise resume (#2008).
        if os.path.isfile(manifest):
            if manifest_only or all(ready_bytecode(result + s) for s in stages):
                continue
            if mesh and has_complete_mesh57_bytecode(result + '.ms57.cso'):
                with suppress(OSError), open(manifest, 'rb') as published:
                    if b'ps none' in published.read().splitlines():
                        continue
        fail = os.path.join(out_dir, '%016x.fail' % key)
        # A repeated pass retries a failure only once its request is rewritten.
        if skip_failed and os.path.exists(fail) and os.path.getmtime(fail) >= os.path.getmtime(os.path.join(out_dir, name)):
            continue
        keys.append(key)
    return keys

if __name__ == '__main__':
    main()
