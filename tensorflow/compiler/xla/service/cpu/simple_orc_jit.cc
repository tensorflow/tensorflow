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

#include "external/llvm/include/llvm/ExecutionEngine/ExecutionEngine.h"
#include "external/llvm/include/llvm/ExecutionEngine/SectionMemoryManager.h"
#include "external/llvm/include/llvm/IR/Mangler.h"
#include "external/llvm/include/llvm/Support/CodeGen.h"
#include "external/llvm/include/llvm/Support/Host.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_runtime.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_runtime_avx.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_runtime_sse4_1.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_conv2d.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_matmul.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_single_threaded_conv2d.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_single_threaded_matmul.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace cpu {
namespace {

// Converts a symbol 'name' into the form expected by dlsym().
std::string CanonicalizeSymbol(const std::string &name) {
#if defined(__APPLE__)
  // On Mac OS X, dlsym() expects names not to be prefixed with a leading
  // underscore.
  if (!name.empty() && name.front() == '_') {
    return name.substr(1);
  }
#endif
  return name;
}

// A simple SymbolResolver that delegates to the host dynamic linker.
struct SimpleResolver : public llvm::JITSymbolResolver {
  llvm::JITSymbol findSymbol(const std::string &name) override {
    void *func_addr = nullptr;

    std::string canonical_name = CanonicalizeSymbol(name);
    if (canonical_name == runtime::kEigenMatmulF32SymbolName) {
      func_addr = reinterpret_cast<void *>(__xla_cpu_runtime_EigenMatMulF32);
    } else if (canonical_name ==
               runtime::kEigenSingleThreadedMatmulF32SymbolName) {
      func_addr = reinterpret_cast<void *>(
          __xla_cpu_runtime_EigenSingleThreadedMatMulF32);
    } else if (canonical_name == runtime::kEigenConvF32SymbolName) {
      func_addr = reinterpret_cast<void *>(__xla_cpu_runtime_EigenConvF32);
    } else if (canonical_name ==
               runtime::kEigenSingleThreadedConvF32SymbolName) {
      func_addr = reinterpret_cast<void *>(
          __xla_cpu_runtime_EigenSingleThreadedConvF32);
    } else if (canonical_name ==
               runtime::kAcquireInfeedBufferForDequeueSymbolName) {
      func_addr = reinterpret_cast<void *>(
          __xla_cpu_runtime_AcquireInfeedBufferForDequeue);
    } else if (canonical_name ==
               runtime::kReleaseInfeedBufferAfterDequeueSymbolName) {
      func_addr = reinterpret_cast<void *>(
          __xla_cpu_runtime_ReleaseInfeedBufferAfterDequeue);
    } else if (canonical_name == runtime::kExpV4F32) {
      func_addr = reinterpret_cast<void *>(runtime::ExpV4F32);
    } else if (canonical_name == runtime::kExpV8F32) {
      func_addr = reinterpret_cast<void *>(runtime::ExpV8F32);
    } else if (canonical_name == runtime::kLogV4F32) {
      func_addr = reinterpret_cast<void *>(runtime::LogV4F32);
    } else if (canonical_name == runtime::kLogV8F32) {
      func_addr = reinterpret_cast<void *>(runtime::LogV8F32);
    } else if (canonical_name == runtime::kTanhV4F32) {
      func_addr = reinterpret_cast<void *>(runtime::TanhV4F32);
    } else if (canonical_name == runtime::kTanhV8F32) {
      func_addr = reinterpret_cast<void *>(runtime::TanhV8F32);
    } else {
      func_addr = dlsym(RTLD_DEFAULT, canonical_name.c_str());
    }

    if (func_addr == nullptr) {
      return nullptr;
    }
    llvm::JITEvaluatedSymbol symbol_info(reinterpret_cast<uint64_t>(func_addr),
                                         llvm::JITSymbolFlags::None);
    return symbol_info;
  }
  llvm::JITSymbol findSymbolInLogicalDylib(const std::string &name) override {
    return nullptr;
  }
};

llvm::SmallVector<std::string, 0> DetectMachineAttributes() {
  llvm::SmallVector<std::string, 0> result;
  llvm::StringMap<bool> host_features;
  if (llvm::sys::getHostCPUFeatures(host_features)) {
    for (auto &feature : host_features) {
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
  intrinsics.sse_intrinsics = (&runtime::ExpV4F32 != nullptr);
  intrinsics.avx_intrinsics = (&runtime::ExpV8F32 != nullptr);
  return intrinsics;
}

}  // namespace

SimpleOrcJIT::SimpleOrcJIT(const llvm::TargetOptions &target_options,
                           llvm::CodeGenOpt::Level opt_level,
                           OptimizationCallback pre_optimization_callback,
                           OptimizationCallback post_optimization_callback)
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
      compile_layer_(object_layer_,
                     CompilerFunctor(target_machine_.get(), &disassembler_,
                                     opt_level, GetAvailableIntrinsics(),
                                     std::move(pre_optimization_callback),
                                     std::move(post_optimization_callback))) {
  VLOG(1) << "CPU target: " << target_machine_->getTargetCPU().str()
          << " features: " << target_machine_->getTargetFeatureString().str();
}

SimpleOrcJIT::ModuleHandleT SimpleOrcJIT::AddModule(
    std::unique_ptr<llvm::Module> module) {
  auto handle =
      compile_layer_.addModule(std::move(module), MakeUnique<SimpleResolver>());
  module_handles_.push_back(handle);
  return handle;
}

void SimpleOrcJIT::RemoveModule(SimpleOrcJIT::ModuleHandleT handle) {
  module_handles_.erase(
      std::remove(module_handles_.begin(), module_handles_.end(), handle),
      module_handles_.end());
  compile_layer_.removeModule(handle);
}

llvm::JITSymbol SimpleOrcJIT::FindSymbol(const std::string &name) {
  std::string mangled_name;
  {
    llvm::raw_string_ostream mangled_name_stream(mangled_name);
    llvm::Mangler::getNameWithPrefix(mangled_name_stream, name, data_layout_);
  }

  // Resolve symbol from last module to first, allowing later redefinitions of
  // symbols shadow earlier ones.
  for (auto &handle :
       llvm::make_range(module_handles_.rbegin(), module_handles_.rend())) {
    if (auto symbol =
            compile_layer_.findSymbolIn(handle, mangled_name,
                                        /*ExportedSymbolsOnly=*/true)) {
      return symbol;
    }
  }

  return nullptr;
}

}  // namespace cpu
}  // namespace xla
