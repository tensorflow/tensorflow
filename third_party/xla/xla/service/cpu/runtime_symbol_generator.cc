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

#include "xla/service/cpu/runtime_symbol_generator.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/AbsoluteSymbols.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/CoreContainers.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorSymbolDef.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Support/Error.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "xla/service/cpu/cpu_runtime.h"
#include "xla/service/cpu/runtime_conv2d.h"
#include "xla/service/cpu/runtime_conv2d_acl.h"
#include "xla/service/cpu/runtime_conv3d.h"
#include "xla/service/cpu/runtime_custom_call_status.h"
#include "xla/service/cpu/runtime_fork_join.h"
#include "xla/service/cpu/runtime_fp16.h"
#include "xla/service/cpu/runtime_handle_ffi_call.h"
#include "xla/service/cpu/runtime_key_value_sort.h"
#include "xla/service/cpu/runtime_matmul.h"
#include "xla/service/cpu/runtime_matmul_acl.h"
#include "xla/service/cpu/runtime_pow.h"
#include "xla/service/cpu/runtime_single_threaded_conv2d.h"
#include "xla/service/cpu/runtime_single_threaded_conv3d.h"
#include "xla/service/cpu/runtime_single_threaded_matmul.h"
#include "xla/service/cpu/runtime_topk.h"
#include "xla/service/cpu/windows_compatibility.h"
#include "xla/service/custom_call_target_registry.h"
#include "tsl/platform/logging.h"

#if defined(INTEL_MKL)
#include "xla/service/cpu/onednn_convolution.h"
#include "xla/service/cpu/onednn_layer_norm.h"
#include "xla/service/cpu/onednn_matmul.h"
#include "xla/service/cpu/onednn_softmax.h"
#endif

namespace xla::cpu {

RuntimeSymbolGenerator::RuntimeSymbolGenerator(llvm::DataLayout data_layout)
    : data_layout_(std::move(data_layout)) {}

llvm::Error RuntimeSymbolGenerator::tryToGenerate(
    llvm::orc::LookupState&, llvm::orc::LookupKind kind,
    llvm::orc::JITDylib& jit_dylib, llvm::orc::JITDylibLookupFlags,
    const llvm::orc::SymbolLookupSet& names) {
  llvm::orc::SymbolMap new_defs;

  for (const auto& kv : names) {
    const auto& name = kv.first;
    if (auto symbol = ResolveRuntimeSymbol(*name)) {
      new_defs[name] = *symbol;
    }
  }

  cantFail(jit_dylib.define(llvm::orc::absoluteSymbols(std::move(new_defs))));
  return llvm::Error::success();
}

std::optional<llvm::orc::ExecutorSymbolDef>
RuntimeSymbolGenerator::ResolveRuntimeSymbol(llvm::StringRef name) {
  void* fn_addr = nullptr;
  if (name.size() > 1 && name.front() == data_layout_.getGlobalPrefix()) {
    // On Mac OS X, 'name' may have a leading underscore prefix, even though the
    // registered name may not.
    std::string stripped_name(name.begin() + 1, name.end());
    fn_addr = CustomCallTargetRegistry::Global()->Lookup(stripped_name, "Host");
  } else {
    fn_addr = CustomCallTargetRegistry::Global()->Lookup(name.str(), "Host");
  }

  return llvm::orc::ExecutorSymbolDef{
      llvm::orc::ExecutorAddr(reinterpret_cast<uint64_t>(fn_addr)),
      llvm::JITSymbolFlags::None};
}

//===----------------------------------------------------------------------===//
// Register XLA:CPU runtime symbols with the CustomCallTargetRegistry.
//===----------------------------------------------------------------------===//

#if defined(PLATFORM_WINDOWS)
// This function is used by compiler-generated code on windows, but it's not
// declared anywhere. The signature does not matter, we just need the address.
extern "C" void __chkstk(size_t);
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

#define REGISTER_CPU_RUNTIME_SYMBOL(base_name)                               \
  do {                                                                       \
    auto* function_address =                                                 \
        reinterpret_cast<void*>(__xla_cpu_runtime_##base_name);              \
    registry->Register(xla::cpu::runtime::k##base_name##SymbolName,          \
                       function_address, "Host");                            \
    CHECK_EQ(absl::string_view(xla::cpu::runtime::k##base_name##SymbolName), \
             "__xla_cpu_runtime_" #base_name);                               \
  } while (false)

// Register both the f32 (float) and f64 (double) versions of a libm symbol.
// Unfortunately the double versions are overloaded on some systems, e.g.
// Mac so we need an explicit cast. This requires passing the function signature
// for that case.
#define REGISTER_LIBM_SYMBOL(name, double_sig)                                 \
  do {                                                                         \
    registry->Register(#name "f", reinterpret_cast<void*>(name##f), "Host");   \
    registry->Register(#name,                                                  \
                       reinterpret_cast<void*>(static_cast<double_sig>(name)), \
                       "Host");                                                \
  } while (false)

static bool RegisterKnownJITSymbols() {
  xla::CustomCallTargetRegistry* registry =
      xla::CustomCallTargetRegistry::Global();
  registry->Register("printf", reinterpret_cast<void*>(&printf), "Host");
  registry->Register("puts", reinterpret_cast<void*>(&puts), "Host");

  REGISTER_CPU_RUNTIME_SYMBOL(AcquireInfeedBufferForDequeue);
  REGISTER_CPU_RUNTIME_SYMBOL(AcquireOutfeedBufferForPopulation);
  REGISTER_CPU_RUNTIME_SYMBOL(AllReduce);
  REGISTER_CPU_RUNTIME_SYMBOL(CollectivePermute);
  REGISTER_CPU_RUNTIME_SYMBOL(AllToAll);
  REGISTER_CPU_RUNTIME_SYMBOL(AllGather);
  REGISTER_CPU_RUNTIME_SYMBOL(ReduceScatter);
  REGISTER_CPU_RUNTIME_SYMBOL(PartitionId);
  REGISTER_CPU_RUNTIME_SYMBOL(ReplicaId);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenConv2DF16);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenConv2DF32);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenConv3DF16);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenConv3DF32);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenMatMulF16);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenMatMulF32);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenMatMulF64);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenMatMulC64);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenMatMulC128);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenMatMulS32);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenBatchMatMulF32);
  REGISTER_CPU_RUNTIME_SYMBOL(ACLMatMulF32);
  REGISTER_CPU_RUNTIME_SYMBOL(ACLBatchMatMulF32);
  REGISTER_CPU_RUNTIME_SYMBOL(ACLConv2DF32);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenSingleThreadedConv2DF16);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenSingleThreadedConv2DF32);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenSingleThreadedConv3DF16);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenSingleThreadedConv3DF32);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenSingleThreadedMatMulF8E4M3FN);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenSingleThreadedMatMulF8E5M2);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenSingleThreadedMatMulF16);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenSingleThreadedMatMulF32);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenSingleThreadedMatMulF64);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenSingleThreadedMatMulC64);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenSingleThreadedMatMulC128);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenSingleThreadedMatMulS32);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenSingleThreadedMatMulU8);
  REGISTER_CPU_RUNTIME_SYMBOL(ParallelForkJoin);
  REGISTER_CPU_RUNTIME_SYMBOL(PrintfToStderr);
  REGISTER_CPU_RUNTIME_SYMBOL(ReleaseInfeedBufferAfterDequeue);
  REGISTER_CPU_RUNTIME_SYMBOL(ReleaseOutfeedBufferAfterPopulation);
  REGISTER_CPU_RUNTIME_SYMBOL(StatusIsSuccess);
  REGISTER_CPU_RUNTIME_SYMBOL(KeyValueSort);
  REGISTER_CPU_RUNTIME_SYMBOL(TopKF32);
  REGISTER_CPU_RUNTIME_SYMBOL(TracingStart);
  REGISTER_CPU_RUNTIME_SYMBOL(TracingEnd);
  REGISTER_CPU_RUNTIME_SYMBOL(HandleFfiCall);
#if defined(INTEL_MKL)
  REGISTER_CPU_RUNTIME_SYMBOL(OneDnnMatMul);
  REGISTER_CPU_RUNTIME_SYMBOL(OneDnnSoftmax);
  REGISTER_CPU_RUNTIME_SYMBOL(OneDnnLayerNorm);
  REGISTER_CPU_RUNTIME_SYMBOL(OneDnnConvolution);
  REGISTER_CPU_RUNTIME_SYMBOL(OneDnnMatMulReorder);
#endif  // INTEL_MKL

  registry->Register("__gnu_f2h_ieee", reinterpret_cast<void*>(__gnu_f2h_ieee),
                     "Host");
  registry->Register("__gnu_h2f_ieee", reinterpret_cast<void*>(__gnu_h2f_ieee),
                     "Host");
  registry->Register("__truncdfhf2", reinterpret_cast<void*>(__truncdfhf2),
                     "Host");
  registry->Register("__truncdfbf2", reinterpret_cast<void*>(__truncdfbf2),
                     "Host");
  registry->Register("__truncsfbf2", reinterpret_cast<void*>(__truncsfbf2),
                     "Host");

#ifdef __APPLE__
  registry->Register("__truncsfhf2", reinterpret_cast<void*>(__truncsfhf2),
                     "Host");
  registry->Register("__extendhfsf2", reinterpret_cast<void*>(__extendhfsf2),
                     "Host");
#endif  // __APPLE__
  registry->Register("__powisf2", reinterpret_cast<void*>(__powisf2), "Host");
  registry->Register("__powidf2", reinterpret_cast<void*>(__powidf2), "Host");

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
  REGISTER_LIBM_SYMBOL(llrint, long long (*)(double));   // NOLINT(runtime/int)
  REGISTER_LIBM_SYMBOL(llround, long long (*)(double));  // NOLINT(runtime/int)
  REGISTER_LIBM_SYMBOL(log, double (*)(double));
  REGISTER_LIBM_SYMBOL(log10, double (*)(double));
  REGISTER_LIBM_SYMBOL(log1p, double (*)(double));
  REGISTER_LIBM_SYMBOL(log2, double (*)(double));
  REGISTER_LIBM_SYMBOL(logb, double (*)(double));
  REGISTER_LIBM_SYMBOL(lrint, long (*)(double));   // NOLINT(runtime/int)
  REGISTER_LIBM_SYMBOL(lround, long (*)(double));  // NOLINT(runtime/int)
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
  REGISTER_LIBM_SYMBOL(scalbln,
                       double (*)(double, long));  // NOLINT(runtime/int)
  REGISTER_LIBM_SYMBOL(scalbn, double (*)(double, int));
  REGISTER_LIBM_SYMBOL(sin, double (*)(double));
#ifdef __APPLE__
  REGISTER_LIBM_SYMBOL(__sincos, void (*)(double, double*, double*));
  registry->Register("__sincosf_stret",
                     reinterpret_cast<void*>(__sincosf_stret), "Host");
  registry->Register("__sincos_stret", reinterpret_cast<void*>(__sincos_stret),
                     "Host");
#else
  REGISTER_LIBM_SYMBOL(sincos, void (*)(double, double*, double*));
#endif
  REGISTER_LIBM_SYMBOL(sinh, double (*)(double));
  REGISTER_LIBM_SYMBOL(sqrt, double (*)(double));
  REGISTER_LIBM_SYMBOL(tan, double (*)(double));
  REGISTER_LIBM_SYMBOL(tanh, double (*)(double));
  REGISTER_LIBM_SYMBOL(tgamma, double (*)(double));
  REGISTER_LIBM_SYMBOL(trunc, double (*)(double));

  registry->Register("memcpy", reinterpret_cast<void*>(memcpy), "Host");
  registry->Register("memmove", reinterpret_cast<void*>(memmove), "Host");
  registry->Register("memset", reinterpret_cast<void*>(memset), "Host");

  // Used by MLIR lowering.
  registry->Register("malloc", reinterpret_cast<void*>(malloc), "Host");
  registry->Register("calloc", reinterpret_cast<void*>(calloc), "Host");
  registry->Register("free", reinterpret_cast<void*>(free), "Host");
#ifndef _WIN32
  // TODO(b/246980307): fails to link on windows because it's marked dllimport.
  registry->Register("memrefCopy", reinterpret_cast<void*>(memrefCopy), "Host");
#endif

#ifdef __APPLE__
  registry->Register("__bzero", reinterpret_cast<void*>(bzero), "Host");
  registry->Register("bzero", reinterpret_cast<void*>(bzero), "Host");
  registry->Register("memset_pattern16",
                     reinterpret_cast<void*>(memset_pattern16), "Host");
#endif

#ifdef MEMORY_SANITIZER
  registry->Register("__msan_unpoison",
                     reinterpret_cast<void*>(__msan_unpoison), "Host");
#endif

#if defined(PLATFORM_WINDOWS)
  registry->Register("__chkstk", reinterpret_cast<void*>(__chkstk), "Host");
#endif

  return true;
}

#undef REGISTER_CPU_RUNTIME_SYMBOL
#undef REGISTER_LIBM_SYMBOL

static bool unused = RegisterKnownJITSymbols();

}  // namespace xla::cpu
