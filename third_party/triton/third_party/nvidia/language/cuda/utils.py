from triton.language import core


@core.extern
def globaltimer(_builder=None):
    return core.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l", [], dtype=core.int64, is_pure=False, pack=1,
                                       _builder=_builder)


@core.extern
def smid(_builder=None):
    return core.inline_asm_elementwise("mov.u32 $0, %smid;", "=r", [], dtype=core.int32, is_pure=True, pack=1,
                                       _builder=_builder)


@core.builtin
def num_threads(_builder=None):
    return core.constexpr(_builder.options.num_warps * 32)


@core.builtin
def num_warps(_builder=None):
    return core.constexpr(_builder.options.num_warps)


# ----- FP8E4M3B15 ------
# This data-type is a variant of the standard FP8E4M3 format.
# It was designed for fast software conversion to FP16 on
# nvidia GPUs that do not support it natively.
# This is the same format as FP8E4M3Nv, but:
#   - the exponent bias is 15 instead of 7
#   - 0xff and 0x7f are mapped to +-1.750 instead of +-nan
@core.builtin
def convert_fp8e4b15_to_float16(arg, _builder=None):
    return core.inline_asm_elementwise(
        "{                                      \n"
        ".reg .b32 a<2>, b<2>;                  \n"
        "prmt.b32 a0, 0, $2, 0x5746;            \n"
        "and.b32 b0, a0, 0x7f007f00;            \n"
        "and.b32 b1, a0, 0x00ff00ff;            \n"
        "and.b32 a1, a0, 0x00800080;            \n"
        "shr.b32  b0, b0, 1;                    \n"
        "add.u32 b1, b1, a1;                    \n"
        "lop3.b32 $0, b0, 0x80008000, a0, 0xf8; \n"
        "shl.b32 $1, b1, 7;                     \n"
        "}                                      \n", "=r,=r,r", [arg], dtype=core.float16, is_pure=True, pack=4,
        _builder=_builder)


@core.builtin
def convert_float16_to_fp8e4b15(arg, has_minx2, _builder=None):
    asm = """{
            .reg .pred p<4>;
            .reg .b32 a<2>, b<2>;
            .reg .b16 c<4>;
            .reg .b16 max_val_f16;
            .reg .b32 max_val_f16x2;
            mov.b16 max_val_f16,   0x3F00;
            mov.b32 max_val_f16x2, 0x3F003F00;
            and.b32 a0, $1, 0x7fff7fff;
            and.b32 a1, $2, 0x7fff7fff;"""
    if has_minx2:
        asm += """min.f16x2 a0, a0, max_val_f16x2;
                  min.f16x2 a1, a1, max_val_f16x2;"""
    else:
        asm += """setp.lt.f16x2  p0|p1, a0, max_val_f16x2;
                  setp.lt.f16x2  p2|p3, a1, max_val_f16x2;
                  mov.b32 {c0, c1}, a0;
                  mov.b32 {c2, c3}, a1;
                  selp.b16  c0, c0, max_val_f16, p0;
                  selp.b16  c1, c1, max_val_f16, p1;
                  selp.b16  c2, c2, max_val_f16, p2;
                  selp.b16  c3, c3, max_val_f16, p3;
                  mov.b32 a0, {c0, c1};
                  mov.b32 a1, {c2, c3};"""
    asm += """mad.lo.u32 a0, a0, 2, 0x00800080;
              mad.lo.u32 a1, a1, 2, 0x00800080;
              lop3.b32 b0, $1, 0x80008000, a0, 0xea;
              lop3.b32 b1, $2, 0x80008000, a1, 0xea;
              prmt.b32 $0, b0, b1, 0x7531;
              }"""
    return core.inline_asm_elementwise(asm, "=r,r,r", [arg], dtype=core.float8e4b15, is_pure=True, pack=4,
                                       _builder=_builder)


@core.builtin
def convert_custom_float8(arg, dst_ty, fp_downcast_rounding, has_minx2, _builder=None):
    if arg.type.scalar.is_fp8e4b15():
        upcast_val = convert_fp8e4b15_to_float16(arg, _builder=_builder)
        if dst_ty.scalar.is_fp32():
            upcast_val = upcast_val.to(core.float32, _builder=_builder)
        return upcast_val

    assert arg.type.scalar.is_fp16() or arg.type.scalar.is_fp32()
    downcast_val = arg
    if arg.type.scalar.is_fp32():
        downcast_val = downcast_val.to(core.float16, fp_downcast_rounding="rtz", _builder=_builder)
    downcast_val = convert_float16_to_fp8e4b15(downcast_val, has_minx2=has_minx2, _builder=_builder)
    return downcast_val


@core.builtin
def convert_custom_float8_sm80(arg, dst_ty, fp_downcast_rounding=None, _builder=None):
    return convert_custom_float8(arg, dst_ty, fp_downcast_rounding, has_minx2=True, _builder=_builder)


@core.builtin
def convert_custom_float8_sm70(arg, dst_ty, fp_downcast_rounding=None, _builder=None):
    return convert_custom_float8(arg, dst_ty, fp_downcast_rounding, has_minx2=False, _builder=_builder)
