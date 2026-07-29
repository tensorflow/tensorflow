/* Copyright 2026 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/backends/cpu/codegen/builtin_symbols.h"

#include <cstdlib>

#ifdef _MSC_VER
#include <math.h>
#endif  // _MSC_VER

#include <cmath>
#include <cstdint>  // IWYU pragma: keep
#include <cstring>

#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "xla/backends/cpu/codegen/builtin_atan2.h"
#include "xla/backends/cpu/codegen/builtin_fp16.h"
#include "xla/backends/cpu/codegen/builtin_pow.h"
#include "xla/backends/cpu/runtime/msan_emulated_tls.h"

namespace xla::cpu {

#if defined(PLATFORM_WINDOWS)
#ifdef __MINGW32__
#if defined(__i386__)
#undef _alloca
extern "C" void _alloca(void);
#elif defined(__x86_64__)
extern "C" void ___chkstk_ms(void);
#else
extern "C" void __chkstk(void);
#endif
#else
extern "C" void __chkstk(void);
#endif
#endif

extern "C" {
// Provided by compiler-rt and MLIR. On x86_64, Float16bits.cpp defines these
// with a float return type to match the SSE register return convention.
#if defined(__x86_64__) || defined(_M_X64)
using BF16ABIType = float;
#else
using BF16ABIType = uint16_t;
#endif
// Converts an F32 value to a BF16.
BF16ABIType __truncsfbf2(float);
// Converts an F64 value to a BF16.
BF16ABIType __truncdfbf2(double);

#if defined(__APPLE__) || defined(__s390x__)
// Converts an F32 value to a F16.
uint16_t __truncsfhf2(float);

float __extendhfsf2(uint16_t a);
#endif  // defined(__APPLE__) || defined(__s390x__)

}  // extern "C"

namespace {

using Registry = absl::flat_hash_map<absl::string_view, void*>;

// MSVC does not have sincos[f].
#ifdef _MSC_VER

void sincos(double x, double* sinv, double* cosv) {
  *sinv = sin(x);
  *cosv = cos(x);
}

void sincosf(float x, float* sinv, float* cosv) {
  *sinv = sinf(x);
  *cosv = cosf(x);
}

#endif  // _MSC_VER

template <typename R, typename... Args>
void* SymbolAddress(R (*func)(Args...)) {
  return reinterpret_cast<void*>(func);
}

// Register both the f32 (float) and f64 (double) versions of a libm symbol.
// Unfortunately the double versions are overloaded on some systems, e.g.
// Mac so we need an explicit cast. This requires passing the function signature
// for that case.
#define REGISTER_LIBM_SYMBOL(name, double_sig)  \
  registry[#name "f"] = SymbolAddress(name##f); \
  registry[#name] = SymbolAddress(static_cast<double_sig>(name));

Registry CreateRegistry() {
  Registry registry;

  // Some platforms have overloaded memcpy, memmove, and memset, so we need to
  // specify the signature type to get the address of the specific function.
  registry["memcpy"] =
      SymbolAddress(static_cast<void* (*)(void*, const void*, size_t)>(memcpy));
  registry["memmove"] = SymbolAddress(
      static_cast<void* (*)(void*, const void*, size_t)>(memmove));
  registry["memset"] =
      SymbolAddress(static_cast<void* (*)(void*, int, size_t)>(memset));
  registry["malloc"] = SymbolAddress(static_cast<void* (*)(size_t)>(malloc));
  registry["free"] = SymbolAddress(static_cast<void (*)(void*)>(free));

  registry["__gnu_f2h_ieee"] = SymbolAddress(__gnu_f2h_ieee);
  registry["__gnu_h2f_ieee"] = SymbolAddress(__gnu_h2f_ieee);

  registry["__truncdfhf2"] = SymbolAddress(__truncdfhf2);
  registry["__truncdfbf2"] = SymbolAddress(__truncdfbf2);
  registry["__truncsfbf2"] = SymbolAddress(__truncsfbf2);

  registry["__powisf2"] = SymbolAddress(__powisf2);
  registry["__powidf2"] = SymbolAddress(__powidf2);

  REGISTER_LIBM_SYMBOL(acos, double (*)(double));
  REGISTER_LIBM_SYMBOL(acosh, double (*)(double));
  REGISTER_LIBM_SYMBOL(asin, double (*)(double));
  REGISTER_LIBM_SYMBOL(asinh, double (*)(double));
  REGISTER_LIBM_SYMBOL(atan, double (*)(double));
  registry["atan2f"] = SymbolAddress(BuiltinAtan2f);
  registry["atan2"] = SymbolAddress(BuiltinAtan2);
  REGISTER_LIBM_SYMBOL(atanh, double (*)(double));
  REGISTER_LIBM_SYMBOL(cbrt, double (*)(double));
  REGISTER_LIBM_SYMBOL(ceil, double (*)(double));
  REGISTER_LIBM_SYMBOL(copysign, double (*)(double, double));
  REGISTER_LIBM_SYMBOL(cos, double (*)(double));
  REGISTER_LIBM_SYMBOL(cosh, double (*)(double));
  REGISTER_LIBM_SYMBOL(erf, double (*)(double));
  REGISTER_LIBM_SYMBOL(erfc, double (*)(double));
  REGISTER_LIBM_SYMBOL(exp, double (*)(double));
  REGISTER_LIBM_SYMBOL(exp2, double (*)(double));
  REGISTER_LIBM_SYMBOL(expm1, double (*)(double));
  REGISTER_LIBM_SYMBOL(fabs, double (*)(double));
  REGISTER_LIBM_SYMBOL(fdim, double (*)(double, double));
  REGISTER_LIBM_SYMBOL(floor, double (*)(double));
  REGISTER_LIBM_SYMBOL(fma, double (*)(double, double, double));
  REGISTER_LIBM_SYMBOL(fmax, double (*)(double, double));
  REGISTER_LIBM_SYMBOL(fmin, double (*)(double, double));
  REGISTER_LIBM_SYMBOL(fmod, double (*)(double, double));
  REGISTER_LIBM_SYMBOL(frexp, double (*)(double, int*));
  REGISTER_LIBM_SYMBOL(hypot, double (*)(double, double));
  REGISTER_LIBM_SYMBOL(ilogb, int (*)(double));
  REGISTER_LIBM_SYMBOL(ldexp, double (*)(double, int));
  REGISTER_LIBM_SYMBOL(lgamma, double (*)(double));
  REGISTER_LIBM_SYMBOL(llrint, long long (*)(double));   // NOLINT
  REGISTER_LIBM_SYMBOL(llround, long long (*)(double));  // NOLINT
  REGISTER_LIBM_SYMBOL(log, double (*)(double));
  REGISTER_LIBM_SYMBOL(log10, double (*)(double));
  REGISTER_LIBM_SYMBOL(log1p, double (*)(double));
  REGISTER_LIBM_SYMBOL(log2, double (*)(double));
  REGISTER_LIBM_SYMBOL(logb, double (*)(double));
  REGISTER_LIBM_SYMBOL(lrint, long (*)(double));   // NOLINT
  REGISTER_LIBM_SYMBOL(lround, long (*)(double));  // NOLINT
  REGISTER_LIBM_SYMBOL(modf, double (*)(double, double*));
  REGISTER_LIBM_SYMBOL(nan, double (*)(const char*));
  REGISTER_LIBM_SYMBOL(nearbyint, double (*)(double));
  REGISTER_LIBM_SYMBOL(nextafter, double (*)(double, double));
  REGISTER_LIBM_SYMBOL(nexttoward, double (*)(double, long double));
  REGISTER_LIBM_SYMBOL(pow, double (*)(double, double));
  REGISTER_LIBM_SYMBOL(remainder, double (*)(double, double));
  REGISTER_LIBM_SYMBOL(remquo, double (*)(double, double, int*));
  REGISTER_LIBM_SYMBOL(rint, double (*)(double));
  REGISTER_LIBM_SYMBOL(round, double (*)(double));
  REGISTER_LIBM_SYMBOL(scalbln, double (*)(double, long));  // NOLINT
  REGISTER_LIBM_SYMBOL(scalbn, double (*)(double, int));
  REGISTER_LIBM_SYMBOL(sin, double (*)(double));
  REGISTER_LIBM_SYMBOL(sinh, double (*)(double));
  REGISTER_LIBM_SYMBOL(sqrt, double (*)(double));
  REGISTER_LIBM_SYMBOL(tan, double (*)(double));
  REGISTER_LIBM_SYMBOL(tanh, double (*)(double));
  REGISTER_LIBM_SYMBOL(tgamma, double (*)(double));
  REGISTER_LIBM_SYMBOL(trunc, double (*)(double));

#ifdef __APPLE__
  REGISTER_LIBM_SYMBOL(__sincos, void (*)(double, double*, double*));
  registry["__sincosf_stret"] = SymbolAddress(__sincosf_stret);
  registry["__sincos_stret"] = SymbolAddress(__sincos_stret);
#else
  REGISTER_LIBM_SYMBOL(sincos, void (*)(double, double*, double*));
#endif

#undef REGISTER_LIBM_SYMBOL

#if defined(__APPLE__) || defined(__s390x__)
  registry["__truncsfhf2"] = SymbolAddress(__truncsfhf2);
  registry["__extendhfsf2"] = SymbolAddress(__extendhfsf2);
#elif defined(__powerpc__)
  // LLVM lowers F16 conversions on PowerPC to these libcalls, which use the
  // same uint16_t ABI as __gnu_f2h_ieee and __gnu_h2f_ieee.
  registry["__truncsfhf2"] = SymbolAddress(__gnu_f2h_ieee);
  registry["__extendhfsf2"] = SymbolAddress(__gnu_h2f_ieee);
#endif

#ifdef __APPLE__
  registry["__bzero"] = SymbolAddress(bzero);
  registry["bzero"] = SymbolAddress(bzero);
  registry["memset_pattern16"] = SymbolAddress(memset_pattern16);
#endif

#if defined(PLATFORM_WINDOWS)

#ifdef __MINGW32__
#if defined(__i386__)
  registry["__chkstk"] = SymbolAddress(_alloca);
#elif defined(__x86_64__)
  registry["__chkstk"] = SymbolAddress(___chkstk_ms);
#else
  registry["__chkstk"] = SymbolAddress(__chkstk);
#endif
#else
  registry["__chkstk"] = SymbolAddress(__chkstk);
#endif

#endif

  registry[kMsanEmutlsGetAddressBridgeSymbolName] =
      SymbolAddress(__xla_cpu_runtime_emutls_get_address);

  return registry;
}

}  // namespace

void* absl_nullable FindBuiltinSymbol(absl::string_view name) {
  static const absl::NoDestructor<Registry> registry(CreateRegistry());
  auto it = registry->find(name);
  return it == registry->end() ? nullptr : it->second;
}

}  // namespace xla::cpu
