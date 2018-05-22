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

#include <stdint.h>
#include <algorithm>
#include <list>
#include <utility>

#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/Mangler.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Host.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_runtime.h"
#include "tensorflow/compiler/xla/service/cpu/custom_call_target_registry.h"
#include "tensorflow/compiler/xla/service/cpu/orc_jit_memory_mapper.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_conv2d.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_conv2d_mkl.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_fft.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_fork_join.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_fp16.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_matmul.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_matmul_mkl.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_single_threaded_conv2d.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_single_threaded_fft.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_single_threaded_matmul.h"
#include "tensorflow/compiler/xla/service/cpu/windows_compatibility.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace cpu {
namespace {

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
}  // namespace

/*static*/ std::unique_ptr<llvm::TargetMachine>
SimpleOrcJIT::InferTargetMachineForJIT(
    const llvm::TargetOptions& target_options,
    llvm::CodeGenOpt::Level opt_level) {
  std::unique_ptr<llvm::TargetMachine> target_machine(
      llvm::EngineBuilder()
          .setTargetOptions(target_options)
          .setOptLevel(opt_level)
          .selectTarget(
              /*TargetTriple=*/llvm::Triple(), /*MArch=*/"",
              /*MCPU=*/GetHostCpuName(),
              /*MAttrs=*/DetectMachineAttributes()));
  CHECK(target_machine != nullptr);
  return target_machine;
}

SimpleOrcJIT::SimpleOrcJIT(const llvm::TargetOptions& target_options,
                           llvm::CodeGenOpt::Level opt_level,
                           bool optimize_for_size, bool enable_fast_math,
                           bool disable_expensive_passes,
                           LLVMCompiler::ModuleHook pre_optimization_hook,
                           LLVMCompiler::ModuleHook post_optimization_hook)
    : target_machine_(InferTargetMachineForJIT(target_options, opt_level)),
      disassembler_(*target_machine_),
      data_layout_(target_machine_->createDataLayout()),
      symbol_resolver_(llvm::orc::createLegacyLookupResolver(
          execution_session_,
          [this](const std::string& name) -> llvm::JITSymbol {
            return this->ResolveRuntimeSymbol(name);
          },
          [](llvm::Error Err) {
            cantFail(std::move(Err), "lookupFlags failed");
          })),
      object_layer_(execution_session_,
                    [this](llvm::orc::VModuleKey) {
                      llvm::orc::RTDyldObjectLinkingLayer::Resources result;
                      result.MemMgr =
                          std::make_shared<llvm::SectionMemoryManager>(
                              orc_jit_memory_mapper::GetInstance());
                      result.Resolver = symbol_resolver_;
                      return result;
                    }),
      compile_layer_(object_layer_,
                     CompilerFunctor(target_machine_.get(), &disassembler_,
                                     opt_level, optimize_for_size,
                                     enable_fast_math, disable_expensive_passes,
                                     std::move(pre_optimization_hook),
                                     std::move(post_optimization_hook))) {
  VLOG(1) << "CPU target: " << target_machine_->getTargetCPU().str()
          << " features: " << target_machine_->getTargetFeatureString().str();
}

llvm::JITSymbol SimpleOrcJIT::ResolveRuntimeSymbol(const std::string& name) {
  if (const uint8* from_constant_pool =
          external_constant_pool_.Find(string(name))) {
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

SimpleOrcJIT::VModuleKeyT SimpleOrcJIT::AddModule(
    std::unique_ptr<llvm::Module> module) {
  auto key = execution_session_.allocateVModule();
  cantFail(compile_layer_.addModule(key, std::move(module)));
  module_keys_.push_back(key);
  return key;
}

void SimpleOrcJIT::RemoveModule(SimpleOrcJIT::VModuleKeyT key) {
  module_keys_.erase(std::remove(module_keys_.begin(), module_keys_.end(), key),
                     module_keys_.end());
  cantFail(compile_layer_.removeModule(key));
}

llvm::JITSymbol SimpleOrcJIT::FindCompiledSymbol(const std::string& name) {
  // Resolve symbol from last module to first, allowing later redefinitions of
  // symbols shadow earlier ones.
  for (auto& key :
       llvm::make_range(module_keys_.rbegin(), module_keys_.rend())) {
    if (auto symbol =
            compile_layer_.findSymbolIn(key, name,
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
  REGISTER_CPU_RUNTIME_SYMBOL(MKLConvF32);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenConvF16);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenConvF32);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenFft);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenMatMulF16);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenMatMulF32);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenMatMulF64);
  REGISTER_CPU_RUNTIME_SYMBOL(MKLMatMulF32);
  REGISTER_CPU_RUNTIME_SYMBOL(MKLMatMulF64);
  REGISTER_CPU_RUNTIME_SYMBOL(MKLSingleThreadedMatMulF32);
  REGISTER_CPU_RUNTIME_SYMBOL(MKLSingleThreadedMatMulF64);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenSingleThreadedConvF16);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenSingleThreadedConvF32);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenSingleThreadedFft);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenSingleThreadedMatMulF16);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenSingleThreadedMatMulF32);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenSingleThreadedMatMulF64);
  REGISTER_CPU_RUNTIME_SYMBOL(ParallelForkJoin);
  REGISTER_CPU_RUNTIME_SYMBOL(ReleaseInfeedBufferAfterDequeue);
  REGISTER_CPU_RUNTIME_SYMBOL(ReleaseOutfeedBufferAfterPopulation);

  registry->Register("__gnu_f2h_ieee", reinterpret_cast<void*>(__gnu_f2h_ieee));
  registry->Register("__gnu_h2f_ieee", reinterpret_cast<void*>(__gnu_h2f_ieee));

#undef REGISTER_CPU_RUNTIME_SYMBOL

// Register both the f32 (float) and f64 (double) versions of a libm symbol.
// Unfortunately the double versions are overloaded on some systems, e.g.
// Mac so we need an explicit cast. This requires passing the function signature
// for that case.
#define REGISTER_LIBM_SYMBOL(name, double_sig)                          \
  do {                                                                  \
    registry->Register(#name "f", reinterpret_cast<void*>(name##f));    \
    registry->Register(                                                 \
        #name, reinterpret_cast<void*>(static_cast<double_sig>(name))); \
  } while (false)

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
#else
  REGISTER_LIBM_SYMBOL(sincos, void (*)(double, double*, double*));
#endif
  REGISTER_LIBM_SYMBOL(sinh, double (*)(double));
  REGISTER_LIBM_SYMBOL(sqrt, double (*)(double));
  REGISTER_LIBM_SYMBOL(tan, double (*)(double));
  REGISTER_LIBM_SYMBOL(tanh, double (*)(double));
  REGISTER_LIBM_SYMBOL(tgamma, double (*)(double));
  REGISTER_LIBM_SYMBOL(trunc, double (*)(double));

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
