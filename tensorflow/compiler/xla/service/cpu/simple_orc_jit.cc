/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/cpu/simple_orc_jit.h"

#include <dlfcn.h>
#include <stdint.h>
#include <algorithm>
#include <list>
#include <utility>

#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/Mangler.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Host.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_runtime.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_runtime_avx.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_runtime_neon.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_runtime_sse4_1.h"
#include "tensorflow/compiler/xla/service/cpu/custom_call_target_registry.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_conv2d.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_fork_join.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_matmul.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_single_threaded_conv2d.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_single_threaded_matmul.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace cpu {
namespace {

// A simple SymbolResolver that delegates to the host dynamic linker.
class SimpleResolver : public llvm::JITSymbolResolver {
 public:
  explicit SimpleResolver(ExternalConstantPool* external_constant_pool)
      : external_constant_pool_(external_constant_pool) {}

  llvm::JITSymbol findSymbol(const std::string& name) override {
    if (const uint8* from_constant_pool =
            external_constant_pool_->Find(string(name))) {
      return llvm::JITEvaluatedSymbol(
          reinterpret_cast<uint64_t>(from_constant_pool),
          llvm::JITSymbolFlags::None);
    }

    void* func_addr = CustomCallTargetRegistry::Global()->Lookup(name);
    if (func_addr == nullptr) {
      return nullptr;
    }
    llvm::JITEvaluatedSymbol symbol_info(reinterpret_cast<uint64_t>(func_addr),
                                         llvm::JITSymbolFlags::None);
    return symbol_info;
  }
  llvm::JITSymbol findSymbolInLogicalDylib(const std::string& name) override {
    return nullptr;
  }

 private:
  ExternalConstantPool* external_constant_pool_;
};

llvm::SmallVector<std::string, 0> DetectMachineAttributes() {
  llvm::SmallVector<std::string, 0> result;
  llvm::StringMap<bool> host_features;
  if (llvm::sys::getHostCPUFeatures(host_features)) {
    for (auto& feature : host_features) {
      if (feature.second) {
        llvm::StringRef feature_name = feature.first();
        // Skip avx512 for now, it isn't quite ready in LLVM.
        if (feature_name.startswith("avx512")) {
          continue;
        }
        result.push_back(feature_name);
      }
    }
  }
  return result;
}

llvm::StringRef GetHostCpuName() {
  auto cpu_name = llvm::sys::getHostCPUName();
  // Skip avx512 for now, it isn't quite ready in LLVM.
  cpu_name.consume_back("-avx512");
  return cpu_name;
}

CompilerFunctor::VectorIntrinsics GetAvailableIntrinsics() {
  CompilerFunctor::VectorIntrinsics intrinsics;
  intrinsics.sse_intrinsics = (&__xla_cpu_runtime_ExpV4F32SSE != nullptr);
  intrinsics.avx_intrinsics = (&__xla_cpu_runtime_ExpV8F32AVX != nullptr);
  intrinsics.neon_intrinsics = (&__xla_cpu_runtime_ExpV4F32NEON != nullptr);
  return intrinsics;
}

}  // namespace

SimpleOrcJIT::SimpleOrcJIT(const llvm::TargetOptions& target_options,
                           llvm::CodeGenOpt::Level opt_level,
                           bool optimize_for_size, bool enable_fast_math,
                           bool disable_expensive_passes,
                           LLVMCompiler::ModuleHook pre_optimization_hook,
                           LLVMCompiler::ModuleHook post_optimization_hook)
    : target_machine_(
          CHECK_NOTNULL(llvm::EngineBuilder()
                            .setTargetOptions(target_options)
                            .setOptLevel(opt_level)
                            .selectTarget(
                                /*TargetTriple=*/llvm::Triple(), /*MArch=*/"",
                                /*MCPU=*/GetHostCpuName(),
                                /*MAttrs=*/DetectMachineAttributes()))),
      disassembler_(*target_machine_),
      data_layout_(target_machine_->createDataLayout()),
      object_layer_(
          [] { return std::make_shared<llvm::SectionMemoryManager>(); }),
      compile_layer_(
          object_layer_,
          CompilerFunctor(target_machine_.get(), &disassembler_, opt_level,
                          optimize_for_size, enable_fast_math,
                          disable_expensive_passes, GetAvailableIntrinsics(),
                          std::move(pre_optimization_hook),
                          std::move(post_optimization_hook))) {
  VLOG(1) << "CPU target: " << target_machine_->getTargetCPU().str()
          << " features: " << target_machine_->getTargetFeatureString().str();
}

SimpleOrcJIT::ModuleHandleT SimpleOrcJIT::AddModule(
    std::unique_ptr<llvm::Module> module) {
  auto handle = cantFail(compile_layer_.addModule(
      std::move(module), MakeUnique<SimpleResolver>(external_constant_pool())));
  module_handles_.push_back(handle);
  return handle;
}

void SimpleOrcJIT::RemoveModule(SimpleOrcJIT::ModuleHandleT handle) {
  module_handles_.erase(
      std::remove(module_handles_.begin(), module_handles_.end(), handle),
      module_handles_.end());
  cantFail(compile_layer_.removeModule(handle));
}

llvm::JITSymbol SimpleOrcJIT::FindSymbol(const std::string& name) {
  std::string mangled_name;
  {
    llvm::raw_string_ostream mangled_name_stream(mangled_name);
    llvm::Mangler::getNameWithPrefix(mangled_name_stream, name, data_layout_);
  }

  // Resolve symbol from last module to first, allowing later redefinitions of
  // symbols shadow earlier ones.
  for (auto& handle :
       llvm::make_range(module_handles_.rbegin(), module_handles_.rend())) {
    if (auto symbol =
            compile_layer_.findSymbolIn(handle, mangled_name,
                                        /*ExportedSymbolsOnly=*/true)) {
      return symbol;
    }
  }

  return nullptr;
}

namespace {
// Register some known symbols with the CustomCallTargetRegistry.
bool RegisterKnownJITSymbols() {
  CustomCallTargetRegistry* registry = CustomCallTargetRegistry::Global();

#define REGISTER_CPU_RUNTIME_SYMBOL(base_name)                                \
  do {                                                                        \
    auto* function_address =                                                  \
        reinterpret_cast<void*>(__xla_cpu_runtime_##base_name);               \
    registry->Register(xla::cpu::runtime::k##base_name##SymbolName,           \
                       function_address);                                     \
    CHECK_EQ(                                                                 \
        tensorflow::StringPiece(xla::cpu::runtime::k##base_name##SymbolName), \
        "__xla_cpu_runtime_" #base_name);                                     \
  } while (false)

  REGISTER_CPU_RUNTIME_SYMBOL(AcquireInfeedBufferForDequeue);
  REGISTER_CPU_RUNTIME_SYMBOL(AcquireOutfeedBufferForPopulation);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenConvF32);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenMatMulF32);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenMatMulF64);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenSingleThreadedConvF32);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenSingleThreadedMatMulF32);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenSingleThreadedMatMulF64);
  REGISTER_CPU_RUNTIME_SYMBOL(ExpV4F32NEON);
  REGISTER_CPU_RUNTIME_SYMBOL(ExpV4F32SSE);
  REGISTER_CPU_RUNTIME_SYMBOL(ExpV8F32AVX);
  REGISTER_CPU_RUNTIME_SYMBOL(LogV4F32NEON);
  REGISTER_CPU_RUNTIME_SYMBOL(LogV4F32SSE);
  REGISTER_CPU_RUNTIME_SYMBOL(LogV8F32AVX);
  REGISTER_CPU_RUNTIME_SYMBOL(ParallelForkJoin);
  REGISTER_CPU_RUNTIME_SYMBOL(ReleaseInfeedBufferAfterDequeue);
  REGISTER_CPU_RUNTIME_SYMBOL(ReleaseOutfeedBufferAfterPopulation);

#undef REGISTER_CPU_RUNTIME_SYMBOL

#define REGISTER_LIBM_SYMBOL(name)                                    \
  do {                                                                \
    /* Register both the F32 and F64 variants of the libm symbol.  */ \
    registry->Register(#name "f", reinterpret_cast<void*>(name##f));  \
    registry->Register(#name, reinterpret_cast<void*>(name));         \
  } while (false)

  REGISTER_LIBM_SYMBOL(acos);
  REGISTER_LIBM_SYMBOL(acosh);
  REGISTER_LIBM_SYMBOL(asin);
  REGISTER_LIBM_SYMBOL(asinh);
  REGISTER_LIBM_SYMBOL(atan);
  REGISTER_LIBM_SYMBOL(atan2);
  REGISTER_LIBM_SYMBOL(atanh);
  REGISTER_LIBM_SYMBOL(cbrt);
  REGISTER_LIBM_SYMBOL(ceil);
  REGISTER_LIBM_SYMBOL(copysign);
  REGISTER_LIBM_SYMBOL(cos);
  REGISTER_LIBM_SYMBOL(cosh);
  REGISTER_LIBM_SYMBOL(erf);
  REGISTER_LIBM_SYMBOL(erfc);
  REGISTER_LIBM_SYMBOL(exp);
  REGISTER_LIBM_SYMBOL(exp2);
  REGISTER_LIBM_SYMBOL(expm1);
  REGISTER_LIBM_SYMBOL(fabs);
  REGISTER_LIBM_SYMBOL(fdim);
  REGISTER_LIBM_SYMBOL(floor);
  REGISTER_LIBM_SYMBOL(fma);
  REGISTER_LIBM_SYMBOL(fmax);
  REGISTER_LIBM_SYMBOL(fmin);
  REGISTER_LIBM_SYMBOL(fmod);
  REGISTER_LIBM_SYMBOL(frexp);
  REGISTER_LIBM_SYMBOL(hypot);
  REGISTER_LIBM_SYMBOL(ilogb);
  REGISTER_LIBM_SYMBOL(ldexp);
  REGISTER_LIBM_SYMBOL(lgamma);
  REGISTER_LIBM_SYMBOL(llrint);
  REGISTER_LIBM_SYMBOL(llround);
  REGISTER_LIBM_SYMBOL(log);
  REGISTER_LIBM_SYMBOL(log10);
  REGISTER_LIBM_SYMBOL(log1p);
  REGISTER_LIBM_SYMBOL(log2);
  REGISTER_LIBM_SYMBOL(logb);
  REGISTER_LIBM_SYMBOL(lrint);
  REGISTER_LIBM_SYMBOL(lround);
  REGISTER_LIBM_SYMBOL(modf);
  REGISTER_LIBM_SYMBOL(nan);
  REGISTER_LIBM_SYMBOL(nearbyint);
  REGISTER_LIBM_SYMBOL(nextafter);
  REGISTER_LIBM_SYMBOL(nexttoward);
  REGISTER_LIBM_SYMBOL(pow);
  REGISTER_LIBM_SYMBOL(remainder);
  REGISTER_LIBM_SYMBOL(remquo);
  REGISTER_LIBM_SYMBOL(rint);
  REGISTER_LIBM_SYMBOL(round);
  REGISTER_LIBM_SYMBOL(scalbln);
  REGISTER_LIBM_SYMBOL(scalbn);
  REGISTER_LIBM_SYMBOL(sin);
  REGISTER_LIBM_SYMBOL(sincos);
  REGISTER_LIBM_SYMBOL(sinh);
  REGISTER_LIBM_SYMBOL(sqrt);
  REGISTER_LIBM_SYMBOL(tan);
  REGISTER_LIBM_SYMBOL(tanh);
  REGISTER_LIBM_SYMBOL(tgamma);
  REGISTER_LIBM_SYMBOL(trunc);

#undef REGISTER_LIBM_SYMBOL

  registry->Register("memcpy", reinterpret_cast<void*>(memcpy));
  registry->Register("memmove", reinterpret_cast<void*>(memmove));
  registry->Register("memset", reinterpret_cast<void*>(memset));
  return true;
}

bool unused = RegisterKnownJITSymbols();
}  // namespace

}  // namespace cpu
}  // namespace xla
