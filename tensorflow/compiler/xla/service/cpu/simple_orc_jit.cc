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
#include <cstdio>
#include <list>
#include <memory>
#include <utility>

#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/ExecutorProcessControl.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Host.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"  // from @llvm-project
#include "tensorflow/compiler/xla/service/cpu/cpu_runtime.h"
#include "tensorflow/compiler/xla/service/cpu/orc_jit_memory_mapper.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_conv2d.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_conv2d_mkl.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_conv3d.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_custom_call_status.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_fft.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_fork_join.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_fp16.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_key_value_sort.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_matmul.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_matmul_mkl.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_matmul_acl.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_pow.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_single_threaded_conv2d.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_single_threaded_conv3d.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_single_threaded_fft.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_single_threaded_matmul.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_topk.h"
#include "tensorflow/compiler/xla/service/cpu/windows_compatibility.h"
#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/logging.h"

// Provided by compiler-rt and MLIR.
// Converts an F32 value to a BF16.
extern "C" uint16_t __truncsfbf2(float);
// Converts an F64 value to a BF16.
extern "C" uint16_t __truncdfbf2(double);

namespace xla {
namespace cpu {
namespace {

llvm::SmallVector<std::string, 0> DetectMachineAttributes() {
  llvm::SmallVector<std::string, 0> result;
  llvm::StringMap<bool> host_features;
  if (llvm::sys::getHostCPUFeatures(host_features)) {
    for (auto& feature : host_features) {
      result.push_back((feature.second ? '+' : '-') +
                       std::string(feature.first()));
    }
  }
  return result;
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
              /*MCPU=*/llvm::sys::getHostCPUName(),
              /*MAttrs=*/DetectMachineAttributes()));
  CHECK(target_machine != nullptr);
  return target_machine;
}

SimpleOrcJIT::SimpleOrcJIT(
    std::unique_ptr<llvm::orc::ExecutorProcessControl> target_process_control,
    std::unique_ptr<llvm::orc::ExecutionSession> execution_session,
    const llvm::TargetOptions& target_options,
    llvm::CodeGenOpt::Level opt_level, bool optimize_for_size,
    bool disable_expensive_passes, llvm::FastMathFlags fast_math_flags,
    LLVMCompiler::ModuleHook pre_optimization_hook,
    LLVMCompiler::ModuleHook post_optimization_hook,
    std::function<void(const llvm::object::ObjectFile&)> post_codegen_hook)
    : target_machine_(InferTargetMachineForJIT(target_options, opt_level)),
      target_triple_(target_machine_->getTargetTriple()),
      data_layout_(target_machine_->createDataLayout()),
      target_process_control_(std::move(target_process_control)),
      execution_session_(std::move(execution_session)),
      object_layer_(*execution_session_,
                    []() {
                      return std::make_unique<llvm::SectionMemoryManager>(
                          orc_jit_memory_mapper::GetInstance());
                    }),
      compile_layer_(
          *execution_session_, object_layer_,
          std::make_unique<CompilerFunctor>(
              target_machine_.get(), opt_level, optimize_for_size,
              disable_expensive_passes, fast_math_flags,
              std::move(pre_optimization_hook),
              std::move(post_optimization_hook), std::move(post_codegen_hook))),
      main_jit_dylib_(&execution_session_->createBareJITDylib("<main>")),
      gdb_jit_event_listener_(
          llvm::JITEventListener::createGDBRegistrationListener()) {
  VLOG(1) << "CPU target: " << target_machine_->getTargetCPU().str()
          << " features: " << target_machine_->getTargetFeatureString().str();

  // Materialize unknown symbols from the runtime symbol table.
  class RuntimeSymbolGenerator : public llvm::orc::DefinitionGenerator {
    SimpleOrcJIT& jit_;

   public:
    explicit RuntimeSymbolGenerator(SimpleOrcJIT& jit) : jit_(jit) {}
    llvm::Error tryToGenerate(
        llvm::orc::LookupState&, llvm::orc::LookupKind,
        llvm::orc::JITDylib& jit_dylib, llvm::orc::JITDylibLookupFlags,
        const llvm::orc::SymbolLookupSet& names) override {
      llvm::orc::SymbolMap new_defs;

      for (const auto& kv : names) {
        const auto& name = kv.first;
        if (llvm::JITEvaluatedSymbol symbol =
                jit_.ResolveRuntimeSymbol(*name)) {
          new_defs[name] = symbol;
        }
      }

      cantFail(jit_dylib.define(absoluteSymbols(std::move(new_defs))));
      return llvm::Error::success();
    }
  };
  main_jit_dylib_->addGenerator(
      std::make_unique<RuntimeSymbolGenerator>(*this));
  object_layer_.registerJITEventListener(*this);

  // Copied from LLJIT, required to find symbols on Windows.
  if (target_triple_.isOSBinFormatCOFF()) {
    object_layer_.setOverrideObjectFlagsWithResponsibilityFlags(true);
    object_layer_.setAutoClaimResponsibilityForObjectSymbols(true);
  }
}

SimpleOrcJIT::~SimpleOrcJIT() {
  if (auto err = execution_session_->endSession()) {
    execution_session_->reportError(std::move(err));
  }
}

llvm::Expected<std::unique_ptr<SimpleOrcJIT>> SimpleOrcJIT::Create(
    const llvm::TargetOptions& target_options,
    llvm::CodeGenOpt::Level opt_level, bool optimize_for_size,
    bool disable_expensive_passes, llvm::FastMathFlags fast_math_flags,
    LLVMCompiler::ModuleHook pre_optimization_hook,
    LLVMCompiler::ModuleHook post_optimization_hook,
    std::function<void(const llvm::object::ObjectFile&)> post_codegen_hook) {
  auto SSP = std::make_shared<llvm::orc::SymbolStringPool>();
  auto target_process_control =
      llvm::orc::SelfExecutorProcessControl::Create(std::move(SSP));
  if (!target_process_control) {
    return target_process_control.takeError();
  }

  auto execution_session = std::make_unique<llvm::orc::ExecutionSession>(
      std::make_unique<llvm::orc::UnsupportedExecutorProcessControl>());
  return std::make_unique<SimpleOrcJIT>(
      std::move(*target_process_control), std::move(execution_session),
      target_options, opt_level, optimize_for_size, disable_expensive_passes,
      fast_math_flags, std::move(pre_optimization_hook),
      std::move(post_optimization_hook), std::move(post_codegen_hook));
}

llvm::JITEvaluatedSymbol SimpleOrcJIT::ResolveRuntimeSymbol(
    llvm::StringRef name) {
  void* func_addr = nullptr;
  if (name.size() > 1 && name.front() == data_layout_.getGlobalPrefix()) {
    // On Mac OS X, 'name' may have a leading underscore prefix, even though the
    // registered name may not.
    std::string stripped_name(name.begin() + 1, name.end());
    func_addr =
        xla::CustomCallTargetRegistry::Global()->Lookup(stripped_name, "Host");
  } else {
    func_addr =
        xla::CustomCallTargetRegistry::Global()->Lookup(name.str(), "Host");
  }

  if (func_addr == nullptr) {
    LOG(ERROR)
        << "Unable to resolve runtime symbol: `" << name.str()
        << "'.  Hint: if the symbol a custom call target, make sure you've "
           "registered it with the JIT using "
           "XLA_CPU_REGISTER_CUSTOM_CALL_TARGET.";
    return nullptr;
  }
  llvm::JITEvaluatedSymbol symbol_info(reinterpret_cast<uint64_t>(func_addr),
                                       llvm::JITSymbolFlags::None);
  return symbol_info;
}

void SimpleOrcJIT::notifyObjectLoaded(
    llvm::JITEventListener::ObjectKey key,
    const llvm::object::ObjectFile& object,
    const llvm::RuntimeDyld::LoadedObjectInfo& object_info) {
  gdb_jit_event_listener_->notifyObjectLoaded(key, object, object_info);
  size_of_generated_code_in_bytes_ += object.getData().size();
}

void SimpleOrcJIT::notifyFreeingObject(llvm::JITEventListener::ObjectKey key) {
  gdb_jit_event_listener_->notifyFreeingObject(key);
}

llvm::Error SimpleOrcJIT::AddModule(llvm::orc::ThreadSafeModule module) {
  return compile_layer_.add(*main_jit_dylib_, std::move(module));
}

void SimpleOrcJIT::DoneCompiling() {
  // The target machine takes a non-trivial amount of memory, so once we are
  // done compiling throw it away.
  target_machine_.reset();
}

llvm::Expected<llvm::JITEvaluatedSymbol> SimpleOrcJIT::FindCompiledSymbol(
    const std::string& name) {
  return execution_session_->lookup({main_jit_dylib_}, name);
}

#if defined(PLATFORM_WINDOWS)
// This function is used by compiler-generated code on windows, but it's not
// declared anywhere. The signature does not matter, we just need the address.
extern "C" void __chkstk(size_t);
#endif

namespace {
// Register some known symbols with the CustomCallTargetRegistry.
bool RegisterKnownJITSymbols() {
  xla::CustomCallTargetRegistry* registry =
      xla::CustomCallTargetRegistry::Global();
  registry->Register("printf", reinterpret_cast<void*>(&printf), "Host");
  registry->Register("puts", reinterpret_cast<void*>(&puts), "Host");

#define REGISTER_CPU_RUNTIME_SYMBOL(base_name)                               \
  do {                                                                       \
    auto* function_address =                                                 \
        reinterpret_cast<void*>(__xla_cpu_runtime_##base_name);              \
    registry->Register(xla::cpu::runtime::k##base_name##SymbolName,          \
                       function_address, "Host");                            \
    CHECK_EQ(absl::string_view(xla::cpu::runtime::k##base_name##SymbolName), \
             "__xla_cpu_runtime_" #base_name);                               \
  } while (false)

  REGISTER_CPU_RUNTIME_SYMBOL(AcquireInfeedBufferForDequeue);
  REGISTER_CPU_RUNTIME_SYMBOL(AcquireOutfeedBufferForPopulation);
  REGISTER_CPU_RUNTIME_SYMBOL(AllReduce);
  REGISTER_CPU_RUNTIME_SYMBOL(CollectivePermute);
  REGISTER_CPU_RUNTIME_SYMBOL(AllToAll);
  REGISTER_CPU_RUNTIME_SYMBOL(PartitionId);
  REGISTER_CPU_RUNTIME_SYMBOL(ReplicaId);
  REGISTER_CPU_RUNTIME_SYMBOL(MKLConv2DF32);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenConv2DF16);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenConv2DF32);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenConv3DF16);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenConv3DF32);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenFft);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenMatMulF16);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenMatMulF32);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenMatMulF64);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenMatMulC64);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenMatMulC128);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenMatMulS32);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenBatchMatMulF32);
  REGISTER_CPU_RUNTIME_SYMBOL(MKLMatMulF32);
  REGISTER_CPU_RUNTIME_SYMBOL(MKLMatMulF64);
  REGISTER_CPU_RUNTIME_SYMBOL(MKLSingleThreadedMatMulF32);
  REGISTER_CPU_RUNTIME_SYMBOL(MKLSingleThreadedMatMulF64);
  REGISTER_CPU_RUNTIME_SYMBOL(ACLMatMulF32);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenSingleThreadedConv2DF16);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenSingleThreadedConv2DF32);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenSingleThreadedConv3DF16);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenSingleThreadedConv3DF32);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenSingleThreadedFft);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenSingleThreadedMatMulF16);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenSingleThreadedMatMulF32);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenSingleThreadedMatMulF64);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenSingleThreadedMatMulC64);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenSingleThreadedMatMulC128);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenSingleThreadedMatMulS32);
  REGISTER_CPU_RUNTIME_SYMBOL(ParallelForkJoin);
  REGISTER_CPU_RUNTIME_SYMBOL(PrintfToStderr);
  REGISTER_CPU_RUNTIME_SYMBOL(ReleaseInfeedBufferAfterDequeue);
  REGISTER_CPU_RUNTIME_SYMBOL(ReleaseOutfeedBufferAfterPopulation);
  REGISTER_CPU_RUNTIME_SYMBOL(StatusIsSuccess);
  REGISTER_CPU_RUNTIME_SYMBOL(KeyValueSort);
  REGISTER_CPU_RUNTIME_SYMBOL(TopKF32);
  REGISTER_CPU_RUNTIME_SYMBOL(TracingStart);
  REGISTER_CPU_RUNTIME_SYMBOL(TracingEnd);

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
  registry->Register("__powisf2", reinterpret_cast<void*>(__powisf2), "Host");
  registry->Register("__powidf2", reinterpret_cast<void*>(__powidf2), "Host");

#undef REGISTER_CPU_RUNTIME_SYMBOL

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

#undef REGISTER_LIBM_SYMBOL

  registry->Register("memcpy", reinterpret_cast<void*>(memcpy), "Host");
  registry->Register("memmove", reinterpret_cast<void*>(memmove), "Host");
  registry->Register("memset", reinterpret_cast<void*>(memset), "Host");

  // Used by MLIR lowering.
  registry->Register("malloc", reinterpret_cast<void*>(malloc), "Host");
  registry->Register("calloc", reinterpret_cast<void*>(calloc), "Host");
  registry->Register("free", reinterpret_cast<void*>(free), "Host");
#ifndef _WIN32
  // TODO(kramerb): This fails to link on windows because it's marked dllimport.
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

bool unused = RegisterKnownJITSymbols();
}  // namespace

}  // namespace cpu
}  // namespace xla
