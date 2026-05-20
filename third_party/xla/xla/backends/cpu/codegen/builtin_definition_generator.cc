/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/cpu/codegen/builtin_definition_generator.h"

#include <cstdlib>

#ifdef _MSC_VER
#include <math.h>
#endif  // _MSC_VER

#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <utility>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/AbsoluteSymbols.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/CoreContainers.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorSymbolDef.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Support/Error.h"
#include "xla/backends/cpu/codegen/builtin_fp16.h"
#include "xla/backends/cpu/codegen/builtin_pow.h"

namespace xla::cpu {

//===----------------------------------------------------------------------===//
// A global static registry of builtin symbols available to XLA:CPU executables.
//===----------------------------------------------------------------------===//

using Registry = absl::flat_hash_map<std::string, llvm::orc::ExecutorSymbolDef>;

// Create a new registry of builtin runtime symbols by looking up the addresses
// of the symbols in the current process. Defined below.
static Registry CreateRegistry();

// Returns a global static registry of builtin runtime symbols.
static const Registry& StaticRegistry() {
  static absl::NoDestructor<Registry> registry(CreateRegistry());
  return *registry;
}

static std::optional<llvm::orc::ExecutorSymbolDef> ResolveBuiltinSymbol(
    const llvm::DataLayout& data_layout, llvm::StringRef name) {
  const Registry& registry = StaticRegistry();

  if (name.size() > 1 && name.front() == data_layout.getGlobalPrefix()) {
    // On Mac OS X, 'name' may have a leading underscore prefix, even though the
    // registered name may not.
    std::string stripped_name(name.begin() + 1, name.end());
    if (registry.contains(stripped_name)) {
      return registry.at(stripped_name);
    }
  } else {
    if (registry.contains(name)) {
      return registry.at(name.str());
    }
  }

  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Create builtin runtime symbols registry for the current process.
//===----------------------------------------------------------------------===//

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
// Provided by compiler-rt and MLIR.
// Converts an F32 value to a BF16.
uint16_t __truncsfbf2(float);
// Converts an F64 value to a BF16.
uint16_t __truncdfbf2(double);

#ifdef __APPLE__
// Converts an F32 value to a F16.
uint16_t __truncsfhf2(float);

float __extendhfsf2(uint16_t a);
#endif  // __APPLE__

}  // extern "C"

// MSVC does not have sincos[f].
#ifdef _MSC_VER

static void sincos(double x, double* sinv, double* cosv) {
  *sinv = sin(x);
  *cosv = cos(x);
}

static void sincosf(float x, float* sinv, float* cosv) {
  *sinv = sinf(x);
  *cosv = cosf(x);
}

#endif  // _MSC_VER

template <typename R, typename... Args>
static llvm::orc::ExecutorSymbolDef SymbolDef(R (*func)(Args...)) {
  // We register runtime symbols as weak, because during concurrent compilation
  // different threads may race to register their symbols in the same dylib and
  // we get spurious "symbol already defined" errors.
  return llvm::orc::ExecutorSymbolDef{
      llvm::orc::ExecutorAddr(reinterpret_cast<uint64_t>(func)),
      llvm::JITSymbolFlags::Weak};
}

// Register both the f32 (float) and f64 (double) versions of a libm symbol.
// Unfortunately the double versions are overloaded on some systems, e.g.
// Mac so we need an explicit cast. This requires passing the function signature
// for that case.
#define REGISTER_LIBM_SYMBOL(name, double_sig) \
  registry[#name "f"] = SymbolDef(name##f);    \
  registry[#name] = SymbolDef(static_cast<double_sig>(name));

static Registry CreateRegistry() {
  Registry registry;

  // Some platforms have overloaded memcpy, memmove, and memset, so we need to
  // specify the signature type to get the address of the specific function.
  registry["memcpy"] =
      SymbolDef(static_cast<void* (*)(void*, const void*, size_t)>(memcpy));
  registry["memmove"] =
      SymbolDef(static_cast<void* (*)(void*, const void*, size_t)>(memmove));
  registry["memset"] =
      SymbolDef(static_cast<void* (*)(void*, int, size_t)>(memset));
  registry["malloc"] = SymbolDef(static_cast<void* (*)(size_t)>(malloc));
  registry["free"] = SymbolDef(static_cast<void (*)(void*)>(free));

  registry["__gnu_f2h_ieee"] = SymbolDef(__gnu_f2h_ieee);
  registry["__gnu_h2f_ieee"] = SymbolDef(__gnu_h2f_ieee);

  registry["__truncdfhf2"] = SymbolDef(__truncdfhf2);
  registry["__truncdfbf2"] = SymbolDef(__truncdfbf2);
  registry["__truncsfbf2"] = SymbolDef(__truncsfbf2);

  registry["__powisf2"] = SymbolDef(__powisf2);
  registry["__powidf2"] = SymbolDef(__powidf2);

  REGISTER_LIBM_SYMBOL(acos, double (*)(double));
  REGISTER_LIBM_SYMBOL(acosh, double (*)(double));
  REGISTER_LIBM_SYMBOL(asin, double (*)(double));
  REGISTER_LIBM_SYMBOL(asinh, double (*)(double));
  REGISTER_LIBM_SYMBOL(atan, double (*)(double));
  REGISTER_LIBM_SYMBOL(atan2, double (*)(double, double));
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
  registry["__sincosf_stret"] = SymbolDef(__sincosf_stret);
  registry["__sincos_stret"] = SymbolDef(__sincos_stret);
#else
  REGISTER_LIBM_SYMBOL(sincos, void (*)(double, double*, double*));
#endif

#undef REGISTER_LIBM_SYMBOL

#ifdef __APPLE__
  registry["__truncsfhf2"] = SymbolDef(__truncsfhf2);
  registry["__extendhfsf2"] = SymbolDef(__extendhfsf2);
  registry["__bzero"] = SymbolDef(bzero);
  registry["bzero"] = SymbolDef(bzero);
  registry["memset_pattern16"] = SymbolDef(memset_pattern16);
#endif

#if defined(PLATFORM_WINDOWS)

#ifdef __MINGW32__
#if defined(__i386__)
  registry["__chkstk"] = SymbolDef(_alloca);
#elif defined(__x86_64__)
  registry["__chkstk"] = SymbolDef(___chkstk_ms);
#else
  registry["__chkstk"] = SymbolDef(__chkstk);
#endif
#else
  registry["__chkstk"] = SymbolDef(__chkstk);
#endif

#endif

#ifdef MEMORY_SANITIZER
  registry["__msan_unpoison"] = SymbolDef(__msan_unpoison);
#endif

  return registry;
}

//===----------------------------------------------------------------------===//
// BuiltinDefinitionGenerator
//===----------------------------------------------------------------------===//

BuiltinDefinitionGenerator::BuiltinDefinitionGenerator(
    llvm::DataLayout data_layout)
    : data_layout_(std::move(data_layout)) {}

llvm::Error BuiltinDefinitionGenerator::tryToGenerate(
    llvm::orc::LookupState&, llvm::orc::LookupKind kind,
    llvm::orc::JITDylib& jit_dylib, llvm::orc::JITDylibLookupFlags,
    const llvm::orc::SymbolLookupSet& names) {
  llvm::orc::SymbolMap symbols;
  symbols.reserve(names.size());

  for (const auto& [name, flags] : names) {
    if (auto symbol = ResolveBuiltinSymbol(data_layout_, *name)) {
      symbols[name] = *symbol;
    }
  }

  cantFail(jit_dylib.define(llvm::orc::absoluteSymbols(std::move(symbols))));
  return llvm::Error::success();
}

}  // namespace xla::cpu
