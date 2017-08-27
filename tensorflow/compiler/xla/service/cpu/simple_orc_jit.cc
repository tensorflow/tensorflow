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
std::string CanonicalizeSymbol(const std::string& name) {
#if defined(__APPLE__)
  // On Mac OS X, dlsym() expects names not to be prefixed with a leading
  // underscore.
  if (!name.empty() && name.front() == '_') {
    return name.substr(1);
  }
#endif
  return name;
}

class JITSymbolTable {
 public:
  JITSymbolTable() { Populate(); }

  void* Lookup(llvm::StringRef jit_symbol_name) const {
    auto it = jit_symbol_table_.find(jit_symbol_name);
    return it == jit_symbol_table_.end() ? nullptr : it->getValue();
  }

  static bool MustBeInTable(llvm::StringRef name) {
    // In particular, names starting with
    // runtime::kXlaCpuRuntimeSymbolNamePrefix should not be dlsym'ed.
    return name.startswith(runtime::kXlaCpuRuntimeSymbolNamePrefix);
  }

 private:
  void AddJITSymbolToTable(llvm::StringRef jit_symbol_name,
                           llvm::StringRef cpp_symbol_name,
                           void* jit_symbol_value) {
    // The JIT symbol name and the C++ symbol name (with an extern "C" linkage)
    // need to match, otherwise AOT links will fail.
    CHECK(jit_symbol_name == cpp_symbol_name);
    CHECK(jit_symbol_table_.insert({jit_symbol_name, jit_symbol_value}).second);
  }

  void Populate() {
#define ADD_JIT_SYMBOL_TO_TABLE(base_name)                       \
  do {                                                           \
    AddJITSymbolToTable(                                         \
        xla::cpu::runtime::k##base_name##SymbolName,             \
        "__xla_cpu_runtime_" #base_name,                         \
        reinterpret_cast<void*>(__xla_cpu_runtime_##base_name)); \
  } while (false)

    ADD_JIT_SYMBOL_TO_TABLE(AcquireInfeedBufferForDequeue);
    ADD_JIT_SYMBOL_TO_TABLE(ReleaseInfeedBufferAfterDequeue);
    ADD_JIT_SYMBOL_TO_TABLE(AcquireOutfeedBufferForPopulation);
    ADD_JIT_SYMBOL_TO_TABLE(ReleaseOutfeedBufferAfterPopulation);
    ADD_JIT_SYMBOL_TO_TABLE(ExpV8F32AVX);
    ADD_JIT_SYMBOL_TO_TABLE(LogV8F32AVX);
    ADD_JIT_SYMBOL_TO_TABLE(ExpV4F32SSE);
    ADD_JIT_SYMBOL_TO_TABLE(LogV4F32SSE);
    ADD_JIT_SYMBOL_TO_TABLE(ExpV4F32NEON);
    ADD_JIT_SYMBOL_TO_TABLE(LogV4F32NEON);
    ADD_JIT_SYMBOL_TO_TABLE(EigenConvF32);
    ADD_JIT_SYMBOL_TO_TABLE(EigenMatMulF32);
    ADD_JIT_SYMBOL_TO_TABLE(EigenMatMulF64);
    ADD_JIT_SYMBOL_TO_TABLE(EigenSingleThreadedConvF32);
    ADD_JIT_SYMBOL_TO_TABLE(EigenSingleThreadedMatMulF32);
    ADD_JIT_SYMBOL_TO_TABLE(EigenSingleThreadedMatMulF64);

#undef ADD_JIT_SYMBOL_TO_TABLE
  }

  llvm::StringMap<void*> jit_symbol_table_;
};

const JITSymbolTable& GetJITSymbolTable() {
  static JITSymbolTable* symbol_table = new JITSymbolTable;
  return *symbol_table;
}

// A simple SymbolResolver that delegates to the host dynamic linker.
struct SimpleResolver : public llvm::JITSymbolResolver {
  llvm::JITSymbol findSymbol(const std::string& name) override {
    std::string canonical_name = CanonicalizeSymbol(name);
    const JITSymbolTable& jit_symbol_table = GetJITSymbolTable();

    void* func_addr = JITSymbolTable::MustBeInTable(canonical_name)
                          ? jit_symbol_table.Lookup(canonical_name)
                          : dlsym(RTLD_DEFAULT, canonical_name.c_str());

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
      compile_layer_(object_layer_,
                     CompilerFunctor(target_machine_.get(), &disassembler_,
                                     opt_level, optimize_for_size,
                                     enable_fast_math, GetAvailableIntrinsics(),
                                     std::move(pre_optimization_hook),
                                     std::move(post_optimization_hook))) {
  VLOG(1) << "CPU target: " << target_machine_->getTargetCPU().str()
          << " features: " << target_machine_->getTargetFeatureString().str();
}

SimpleOrcJIT::ModuleHandleT SimpleOrcJIT::AddModule(
    std::unique_ptr<llvm::Module> module) {
  auto handle = cantFail(compile_layer_.addModule(
      std::move(module), MakeUnique<SimpleResolver>()));
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

}  // namespace cpu
}  // namespace xla
