/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/runtime2/compiler.h"

#include <stddef.h>

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "iree-dialects/Dialect/Input/InputOps.h"
#include "absl/base/call_once.h"
#include "third_party/iree/compiler/bindings/c/iree/compiler/embedding_api.h"
#include "third_party/iree/compiler/bindings/c/iree/compiler/loader.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/xla/service/dump.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla.pb.h"

#if defined(PLATFORM_GOOGLE)
#include "tensorflow/compiler/xla/service/gpu/runtime2/google/compiler.h"
#else
#include "tensorflow/compiler/xla/service/gpu/runtime2/default/compiler.h"
#endif

namespace xla::gpu {

//===-----------------------------------------------------------------------===/
// RuntimeCompiler::Bytecode
//===-----------------------------------------------------------------------===/

RuntimeCompiler::Bytecode::Bytecode(iree_compiler_output_t* output, void* data,
                                    size_t length)
    : output_(output), data_(data), length_(length) {}

RuntimeCompiler::Bytecode::~Bytecode() { ireeCompilerOutputDestroy(output_); }

//===-----------------------------------------------------------------------===/
// RuntimeCompiler
//===-----------------------------------------------------------------------===/

RuntimeCompiler::RuntimeCompiler(iree_compiler_session_t* session,
                                 iree_compiler_invocation_t* inv)
    : session_(session), inv_(inv) {}

RuntimeCompiler::~RuntimeCompiler() {
  if (error_) {
    ireeCompilerErrorDestroy(error_);
  }

  ireeCompilerInvocationDestroy(inv_);
  ireeCompilerSessionDestroy(session_);

  if (output_) {
    ireeCompilerOutputDestroy(output_);
  }
}

bool RuntimeCompiler::ParseSourceBuffer(std::string_view buffer) {
  iree_compiler_source_t* source;
  auto* error = ireeCompilerSourceWrapBuffer(
      session_, "<jit>", buffer.data(), buffer.size(),
      /*isNullTerminated=*/false, &source);
  if (error) {
    SetError(error);
    return false;
  }

  return ireeCompilerInvocationParseSource(inv_, source);
}

bool RuntimeCompiler::SetFlag(const char* flag) {
  auto* error = ireeCompilerSessionSetFlags(session_, 1, &flag);
  if (error) {
    SetError(error);
    return false;
  }
  return true;
}

std::unique_ptr<RuntimeCompiler::Bytecode>
RuntimeCompiler::CompileStandardPipeline() {
  if (!ireeCompilerInvocationPipeline(inv_, IREE_COMPILER_PIPELINE_STD)) {
    return nullptr;
  }

  iree_compiler_error_t* error = ireeCompilerOutputOpenMembuffer(&output_);
  if (error) {
    SetError(error);
    return nullptr;
  }

  error = ireeCompilerInvocationOutputVMBytecode(inv_, output_);
  if (error) {
    SetError(error);
    return nullptr;
  }

  void* output_data = nullptr;
  uint64_t size;
  error = ireeCompilerOutputMapMemory(output_, &output_data, &size);
  if (error) {
    SetError(error);
    return nullptr;
  }

  // Transfer the output_ to Bytecode since the mapping is only
  // valid for the life of the output.
  iree_compiler_output_t* local_output = output_;
  output_ = nullptr;
  return std::make_unique<Bytecode>(local_output, output_data, size);
}

//===-----------------------------------------------------------------------===/
// Loading RuntimeCompiler from a library
//===-----------------------------------------------------------------------===/

static bool InitializeCompilerForProcess(const std::string& library_path) {
  if (!ireeCompilerLoadLibrary(library_path.c_str())) {
    return false;
  }

  ireeCompilerGlobalInitialize();
  return true;
}

static std::optional<std::string_view> LoadCompilerStubOnce(
    const std::string& library_path) {
  static std::string* loaded_path = nullptr;

  static absl::once_flag loaded;
  absl::call_once(loaded, [&] {
    if (InitializeCompilerForProcess(library_path)) {
      loaded_path = new std::string(library_path);
    }
  });

  if (loaded_path) return *loaded_path;
  return std::nullopt;
}

std::unique_ptr<RuntimeCompiler> CreateRuntimeCompiler() {
  LoadCompilerStubOnce(GetIREECompilerPath());

  auto* session = ireeCompilerSessionCreate();
  auto* inv = ireeCompilerInvocationCreate(session);

  ireeCompilerInvocationEnableConsoleDiagnostics(inv);

  return std::make_unique<RuntimeCompiler>(session, inv);
}

//===-----------------------------------------------------------------------===/
// Binding Xla device kernels to an input module.
//===-----------------------------------------------------------------------===/

using namespace mlir;                 // NOLINT
using namespace mlir::iree_compiler;  // NOLINT

// TODO(ezhulenev): Query compute capability from the XLA module and set it up
// at the module level.
static constexpr int kComputeCapability = 60;

static IREE::Input::ExecutableTargetAttr getExecutableTarget(MLIRContext* ctx) {
  Builder b(ctx);

  SmallVector<NamedAttribute> config{
      {b.getStringAttr("target_arch"),
       b.getStringAttr(llvm::formatv("sm_{0}", kComputeCapability))},
  };

  return IREE::Input::ExecutableTargetAttr::get(
      ctx, b.getStringAttr("cuda"), b.getStringAttr("cuda-nvptx-fb"),
      b.getDictionaryAttr(config));
}

static IREE::Input::ExecutableObjectAttr getExecutableObject(
    MLIRContext* ctx, const std::vector<uint8_t>& binary) {
  Builder b(ctx);

  // TODO(ezhulenev): Use dense i8 arrays to pass binary data.
  auto vec = VectorType::get(binary.size(), b.getI8Type());
  return IREE::Input::ExecutableObjectAttr::get(
      ctx, /*path=*/b.getStringAttr(""),
      DenseIntElementsAttr::get(vec, binary));
}

static IREE::Input::ExecutableObjectsAttr getExecutableObjects(
    IREE::Input::ExecutableTargetAttr target,
    IREE::Input::ExecutableObjectAttr executable) {
  Builder b(target.getContext());
  return IREE::Input::ExecutableObjectsAttr::get(
      b.getContext(), b.getArrayAttr(target),
      b.getArrayAttr(b.getArrayAttr(executable)));
}

StatusOr<std::string> BindXlaDeviceKernels(const DebugOptions& debug_options,
                                           mlir::ModuleOp module,
                                           std::string_view asm_text,
                                           const std::vector<uint8_t>& binary) {
  auto* ctx = module.getContext();
  SymbolTable sym_table(module);

  auto src =
      sym_table.lookup<IREE::Input::ExecutableSourceOp>("xla.module.ptx");

  // If we are running with StreamExecutor backend we might not have executable
  // source for kernels.
  if (!src) return OkStatus();

  // Bind XLA device kernels to an executable source.
  auto objects = getExecutableObjects(getExecutableTarget(ctx),
                                      getExecutableObject(ctx, binary));
  src.setObjectsAttr(objects);

  std::string module_str = llvm_ir::DumpToString(module);

  // Dump module with bound executable for debugging.
  std::optional<StringRef> module_name = module.getName();
  auto module_id = module->getAttrOfType<IntegerAttr>("hlo.unique_id");

  if (module_name && module_id) {
    DumpToFileInDirOrStdout(debug_options, module_id.getInt(), *module_name,
                            "gpu_rt_host_with_executable", "mlir", module_str);
  }

  return module_str;
}

}  // namespace xla::gpu
