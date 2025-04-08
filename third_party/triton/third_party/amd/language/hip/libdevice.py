from triton.language import core


@core.extern
def abs(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("int32"), ): ("__triton_hip_iabs", core.dtype("int32")),
            (core.dtype("int64"), ): ("__triton_hip_iabs", core.dtype("int64")),
            (core.dtype("fp32"), ): ("__triton_hip_fabs", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__triton_hip_fabs", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def floor(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_floor_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_floor_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def rsqrt(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_rsqrt_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_rsqrt_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ceil(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_ceil_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_ceil_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def trunc(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_trunc_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_trunc_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def exp2(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_exp2_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_exp2_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def exp(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_exp_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_exp_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def fast_expf(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__triton_hip_fast_expf", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def fast_dividef(arg0, arg1, _builder=None):
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("__triton_hip_fast_fdividef", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def sqrt(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_sqrt_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_sqrt_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def llrint(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__triton_hip_llrint", core.dtype("int64")),
            (core.dtype("fp64"), ): ("__triton_hip_llrint", core.dtype("int64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def nearbyint(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__ocml_nearbyint_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_nearbyint_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def isnan(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__ocml_isnan_f32", core.dtype("int32")),
            (core.dtype("fp64"), ): ("__ocml_isnan_f64", core.dtype("int32")),
        }, is_pure=True, _builder=_builder).to(core.int1, _builder=_builder)


@core.extern
def signbit(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__ocml_signbit_f32", core.dtype("int32")),
            (core.dtype("fp64"), ): ("__ocml_signbit_f64", core.dtype("int32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def copysign(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__ocml_copysign_f32", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__ocml_copysign_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def isinf(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_isinf_f32", core.dtype("int32")),
            (core.dtype("fp64"), ): ("__ocml_isinf_f64", core.dtype("int32")),
        }, is_pure=True, _builder=_builder).to(core.int1, _builder=_builder)


@core.extern
def nextafter(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__ocml_nextafter_f32", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__ocml_nextafter_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def sin(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_sin_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_sin_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def cos(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_cos_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_cos_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def tan(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_tan_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_tan_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def log2(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_log2_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_log2_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def cosh(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_cosh_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_cosh_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def sinh(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_sinh_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_sinh_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def tanh(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_tanh_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_tanh_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def atan2(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__ocml_atan2_f32", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__ocml_atan2_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def atan(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_atan_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_atan_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def asin(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_asin_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_asin_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def acos(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_acos_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_acos_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def log(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_log_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_log_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def log10(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_log10_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_log10_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def log1p(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_log1p_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_log1p_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def acosh(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_acosh_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_acosh_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def asinh(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_asinh_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_asinh_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def atanh(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_atanh_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_atanh_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def expm1(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_expm1_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_expm1_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def hypot(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__ocml_hypot_f32", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__ocml_hypot_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def j0(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_j0_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_j0_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def j1(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_j1_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_j1_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def y0(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_y0_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_y0_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def y1(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_y1_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_y1_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def cyl_bessel_i0(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_i0_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_i0_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def cyl_bessel_i1(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_i1_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_i1_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def erf(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_erf_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_erf_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def erfinv(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_erfinv_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_erfinv_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def erfc(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_erfc_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_erfc_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def erfcx(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_erfcx_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_erfcx_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def lgamma(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_lgamma_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_lgamma_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ldexp(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("int32")): ("__ocml_ldexp_f32", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("int32")): ("__ocml_ldexp_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def fmod(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__ocml_fmod_f32", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__ocml_fmod_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def fma(arg0, arg1, arg2, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")): ("__ocml_fma_f32", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64")): ("__ocml_fma_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def pow(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("int32")): ("__ocml_pown_f32", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("int32")): ("__ocml_pown_f64", core.dtype("fp64")),
            (core.dtype("fp32"), core.dtype("fp32")): ("__ocml_pow_f32", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__ocml_pow_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ilogb(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_ilogb_f32", core.dtype("int32")),
            (core.dtype("fp64"), ): ("__ocml_ilogb_f64", core.dtype("int32")),
        }, is_pure=True, _builder=_builder)
