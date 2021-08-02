/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tools/kernel_gen/tf_framework_c_interface.h"

#include <string>
#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"  // from @llvm-project
#include "mlir/ExecutionEngine/OptUtils.h"  // from @llvm-project
#include "mlir/Parser.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/kernel_creator.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/tf_jit_cache.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/resource_mgr.h"

#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
#include "tensorflow/compiler/mlir/tools/kernel_gen/tf_gpu_runtime_wrappers.h"
#endif

namespace mlir {
namespace kernel_gen {
namespace tf_framework {
namespace {

using tensorflow::Allocator;
using tensorflow::AllocatorAttributes;

Allocator* GetAllocator(void* op_kernel_ctx) {
  auto* ctx = static_cast<tensorflow::OpKernelContext*>(op_kernel_ctx);
  // TODO(pifon): Figure out how to set AllocatorAttributes correctly.
  AllocatorAttributes attrs;
  return ctx->get_allocator(attrs);
}

}  // namespace

extern "C" void* _mlir_ciface_tf_alloc(void* op_kernel_ctx, size_t num_elements,
                                       size_t element_size,
                                       int32_t output_index,
                                       int32_t num_candidates,
                                       int32_t* candidate_input_indices) {
  static constexpr int kAmbiguousOutputIndex = -1;
  auto* ctx = static_cast<tensorflow::OpKernelContext*>(op_kernel_ctx);
  if (output_index != kAmbiguousOutputIndex) {
    // Create a 1D shape, because the shapes don't have to match exactly for
    // input forwarding. Only the number of elements must be the same.
    tensorflow::TensorShape output_shape;
    output_shape.AddDim(num_elements);

    // Iterate over indices of all inputs that can potentially be used for
    // forwarding.
    for (int i = 0; i < num_candidates; ++i) {
      auto tensor = ctx->forward_input(candidate_input_indices[i], output_index,
                                       ctx->expected_output_dtype(output_index),
                                       output_shape,
                                       ctx->output_memory_type(output_index),
                                       ctx->output_alloc_attr(output_index));
      if (tensor != nullptr) {
        return tensor->data();
      }
    }

    CHECK(!ctx->output_expects_forwarding(output_index));
  }

  // If no forwarding happened, allocate a chunk of memory.
  return GetAllocator(op_kernel_ctx)
      ->AllocateRaw(Allocator::kAllocatorAlignment,
                    num_elements * element_size);
}

extern "C" void _mlir_ciface_tf_dealloc(void* op_kernel_ctx, void* ptr) {
  GetAllocator(op_kernel_ctx)->DeallocateRaw(ptr);
}

extern "C" void _mlir_ciface_tf_report_error(void* op_kernel_ctx,
                                             int32_t error_code, char* msg) {
  Optional<ErrorCode> symbol = symbolizeErrorCode(error_code);
  if (!symbol.hasValue()) {
    LOG(ERROR) << "No valid conversion from integer value = " << error_code
               << "to ErrorCode attribute";
    return;
  }
  auto* ctx = static_cast<tensorflow::OpKernelContext*>(op_kernel_ctx);
  ctx->CtxFailureWithWarning(
      tensorflow::Status{ConvertAttrToEnumValue(symbol.getValue()), msg});
}

static void ReportError(void* op_kernel_ctx, ErrorCode error_code,
                        const char* msg) {
  _mlir_ciface_tf_report_error(op_kernel_ctx, static_cast<uint32_t>(error_code),
                               const_cast<char*>(msg));
}

namespace {

// A callback to register all externally defined symbols needed by the kernel.
llvm::orc::SymbolMap TFFrameworkSymbolMap(llvm::orc::MangleAndInterner mangle) {
  llvm::orc::SymbolMap symbol_map;
  auto bind = [&](llvm::StringRef name, auto symbol_ptr) {
    symbol_map[mangle(name)] = llvm::JITEvaluatedSymbol(
        llvm::pointerToJITTargetAddress(symbol_ptr), llvm::JITSymbolFlags());
  };

  // Register all the symbols.
  bind("_mlir_ciface_tf_alloc", &_mlir_ciface_tf_alloc);
  bind("_mlir_ciface_tf_dealloc", &_mlir_ciface_tf_dealloc);
  bind("_mlir_ciface_tf_report_error", &_mlir_ciface_tf_report_error);
#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
  bind("_mlir_ciface_tf_launch_kernel", &_mlir_ciface_tf_launch_kernel);
#endif

  return symbol_map;
}

llvm::Expected<std::unique_ptr<ExecutionEngine>> Compile(
    const std::string code) {
  // For now, use some default parameters.
  // TODO(frgossen): Propagate these parameters through the JIT invocation.
  llvm::SmallVector<std::string, 4> architectures = {"sm_60", "sm_70", "sm_75"};
  llvm::SmallVector<int64_t, 1> tile_sizes = {1024};
  llvm::SmallVector<int64_t, 1> unroll_factors = {4};
  int64_t max_supported_rank = 5;
  bool enable_ftz = false;
  bool cpu_codegen = false;
  bool jit_compile = false;

  // Create the kernel.
  mlir::MLIRContext context;
  xla::StatusOr<mlir::OwningModuleRef> status_or_module =
      tensorflow::kernel_gen::GenerateKernelForTfCode(
          context, code, architectures, tile_sizes, unroll_factors,
          max_supported_rank, /*embed_memref_prints=*/false,
          /*print_ptx=*/false, enable_ftz, cpu_codegen, jit_compile);
  if (!status_or_module.ok()) return nullptr;
  mlir::OwningModuleRef module = std::move(status_or_module.ValueOrDie());

  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // Create execution engine with an inner optimization pipeline.
  auto opt_pipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/2, /*sizeLevel=*/0, /*targetMachine=*/nullptr);
  llvm::Expected<std::unique_ptr<ExecutionEngine>> engine =
      mlir::ExecutionEngine::create(module.get(), /*llvmModuleBuilder=*/nullptr,
                                    opt_pipeline);
  if (!engine) return nullptr;

  // Finally, register the missing symbols.
  engine.get()->registerSymbols(TFFrameworkSymbolMap);
  return engine;
}

}  // namespace

extern "C" void* _mlir_ciface_tf_jit_compile(void* op_kernel_ctx, char* code) {
  // Get the resource manager.
  auto* ctx = static_cast<tensorflow::OpKernelContext*>(op_kernel_ctx);
  tensorflow::ResourceMgr* rm = ctx->resource_manager();
  if (!rm) {
    ReportError(op_kernel_ctx, ErrorCode::UNKNOWN, "No resource manager.");
    return nullptr;
  }

  // Get the JIT cache.
  JITCache* jit_cache = nullptr;
  auto status = rm->LookupOrCreate<JITCache>(rm->default_container(),
                                             JITCache::kDefaultResourceName,
                                             &jit_cache, JITCache::Create);
  tensorflow::core::ScopedUnref jit_cache_ref(jit_cache);
  if (!status.ok()) {
    ReportError(op_kernel_ctx, ErrorCode::UNKNOWN,
                "Failed to find or create JIT cache.");
    return nullptr;
  }

  // Lookup or compile the execution module.
  ExecutionEngine* engine =
      jit_cache->LookupOrCompile(code, [&]() { return Compile(code); });
  if (engine == nullptr) {
    ReportError(op_kernel_ctx, ErrorCode::UNKNOWN, "JIT compilation failed.");
    return nullptr;
  }
  return engine;
}

extern "C" void _mlir_ciface_tf_jit_execute(void* op_kernel_ctx, void* callable,
                                            void* result, int64_t arg_rank,
                                            void* arg_descr) {
  ::UnrankedMemRefType<void> arg = {arg_rank, arg_descr};
  llvm::Error invocation_result =
      static_cast<ExecutionEngine*>(callable)->invoke("main", result,
                                                      op_kernel_ctx, &arg);
  if (invocation_result)
    ReportError(op_kernel_ctx, ErrorCode::UNKNOWN, "JIT invocation failed.");
}

}  // namespace tf_framework
}  // namespace kernel_gen
}  // namespace mlir
