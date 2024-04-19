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

#include <cstddef>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Mangling.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/MemRef/Transforms/AllocationOpInterfaceImpl.h"  // from @llvm-project
#include "mlir/ExecutionEngine/CRunnerUtils.h"  // from @llvm-project
#include "mlir/ExecutionEngine/ExecutionEngine.h"  // from @llvm-project
#include "mlir/ExecutionEngine/OptUtils.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tools/kernel_gen/compile_cache_item.pb.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/kernel_creator.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/tf_jit_cache.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/stream.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tsl/framework/allocator.h"

#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
#include <optional>

#include "tensorflow/compiler/mlir/tools/kernel_gen/tf_gpu_runtime_wrappers.h"
#endif

static constexpr absl::string_view kTFJitCacheDirEnvVar = "TF_JIT_CACHE_DIR";

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
  std::optional<ErrorCode> symbol = symbolizeErrorCode(error_code);
  if (!symbol.has_value()) {
    LOG(ERROR) << "No valid conversion from integer value = " << error_code
               << "to ErrorCode attribute";
    return;
  }
  auto* ctx = static_cast<tensorflow::OpKernelContext*>(op_kernel_ctx);
  ctx->CtxFailureWithWarning(
      tensorflow::Status{ConvertAttrToEnumValue(symbol.value()), msg});
}

static void ReportError(void* op_kernel_ctx, ErrorCode error_code,
                        const char* msg) {
  _mlir_ciface_tf_report_error(op_kernel_ctx, static_cast<uint32_t>(error_code),
                               const_cast<char*>(msg));
}

namespace {

std::string GetFileCachePath(const std::string cache_dir,
                             const std::string& code) {
  size_t hash = llvm::hash_value(code);
  return tensorflow::io::JoinPath(cache_dir, std::to_string(hash));
}

// A callback to register all externally defined symbols needed by the kernel.
llvm::orc::SymbolMap TFFrameworkSymbolMap(llvm::orc::MangleAndInterner mangle) {
  llvm::orc::SymbolMap symbol_map;
  auto bind = [&](llvm::StringRef name, auto symbol_ptr) {
    symbol_map[mangle(name)] = {llvm::orc::ExecutorAddr::fromPtr(symbol_ptr),
                                llvm::JITSymbolFlags()};
  };

  // Register TF framework symbols.
  bind("_mlir_ciface_tf_alloc", &_mlir_ciface_tf_alloc);
  bind("_mlir_ciface_tf_dealloc", &_mlir_ciface_tf_dealloc);
  bind("_mlir_ciface_tf_report_error", &_mlir_ciface_tf_report_error);
#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
  bind("_mlir_ciface_tf_launch_kernel", &_mlir_ciface_tf_launch_kernel);
#endif

  // Register malloc/free to avoid unexpected implementations from shared libs.
  bind("malloc", &malloc);
  bind("free", &free);

  return symbol_map;
}

void InitializeLlvmCompiler() {
  static const bool initialized = ([] {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    return true;
  })();
  (void)initialized;
}

llvm::Expected<std::unique_ptr<ExecutionEngine>> Compile(
    const std::string code, llvm::SmallVectorImpl<std::string>& architectures,
    llvm::SmallVectorImpl<int64_t>& tile_sizes,
    llvm::SmallVectorImpl<int64_t>& unroll_factors, bool enable_ftz,
    bool index_64bit) {
  std::string cache_dir;
  if (const char* dir = getenv(kTFJitCacheDirEnvVar.data())) {
    cache_dir = dir;
  }

  // Check if we already have a partially compiled module in the filesystem
  // based cache.
  CompilationCacheItem item;
  auto tenv = tensorflow::Env::Default();
  if (!cache_dir.empty() && tenv->RecursivelyCreateDir(cache_dir).ok()) {
    std::string data;
    if (tensorflow::ReadFileToString(tenv, GetFileCachePath(cache_dir, code),
                                     &data)
            .ok()) {
      item.ParseFromString(data);
      if (item.original_module() != code) {
        item.Clear();
      }
    }
  }

  // Create the kernel.
  mlir::DialectRegistry registry;
  mlir::memref::registerAllocationOpInterfaceExternalModels(registry);
  mlir::MLIRContext context(registry);

  mlir::OwningOpRef<mlir::ModuleOp> module;

  if (item.result_module().empty()) {
    // Otherwise, compile the module now.
    absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> status_or_module =
        tensorflow::kernel_gen::GenerateKernelForHloCode(
            context, code, architectures, tile_sizes, unroll_factors,
            /*print_ptx=*/false, /*print_llvmir=*/false, enable_ftz,
            index_64bit,
            /*jit_compile=*/false,
            /*jit_i64_indexed_for_large_tensors=*/false,
            /*apply_cl_options=*/false);
    if (!status_or_module.ok()) {
      LOG(ERROR) << status_or_module.status();
      return nullptr;
    }
    module = std::move(status_or_module.value());

    if (!cache_dir.empty() && tenv->RecursivelyCreateDir(cache_dir).ok()) {
      // Save the compilation result here for future processes to use.
      item.set_original_module(code);
      llvm::raw_string_ostream stream(*item.mutable_result_module());
      module.get().print(stream);
      stream.flush();

      tensorflow::WriteStringToFile(tenv, GetFileCachePath(cache_dir, code),
                                    item.SerializeAsString())
          .IgnoreError();
    }
  } else {
    module = tensorflow::kernel_gen::SetupContextAndParseModule(
                 context, item.result_module())
                 .value();
  }

  // Initialize LLVM targets.
  InitializeLlvmCompiler();

  // Create execution engine with an inner optimization pipeline.
  auto opt_pipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/2, /*sizeLevel=*/0, /*targetMachine=*/nullptr);
  mlir::ExecutionEngineOptions engine_options;
  engine_options.transformer = opt_pipeline;
  llvm::Expected<std::unique_ptr<ExecutionEngine>> engine =
      mlir::ExecutionEngine::create(module.get(), engine_options);
  if (!engine) {
    LOG(ERROR) << "Failed to create ExecutionEngine: "
               << toString(engine.takeError());
    return nullptr;
  }

  // Finally, register the missing symbols.
  engine.get()->registerSymbols(TFFrameworkSymbolMap);
  return engine;
}

template <typename T, typename U = T>
llvm::SmallVector<T, 8> SmallVectorFromCArray(int64_t num_elements,
                                              U* elements_ptr) {
  llvm::SmallVector<T, 8> result;
  result.reserve(num_elements);
  for (int i = 0; i < num_elements; ++i) result.push_back(elements_ptr[i]);
  return result;
}

}  // namespace

extern "C" void* _mlir_ciface_tf_jit_compile(
    void* op_kernel_ctx, char* code, int64_t num_tile_sizes,
    int64_t* tile_sizes_ptr, int64_t num_unroll_factors,
    int64_t* unroll_factors_ptr, bool enable_ftz, bool index_64bit) {
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

  // Determine the unique architecture for the current GPU, if any.
  SmallVector<std::string, 1> architectures;
#if defined(GOOGLE_CUDA)
  stream_executor::CudaComputeCapability cc =
      ctx->op_device_context()->stream()->GetCudaComputeCapability();
  architectures.push_back(absl::StrCat("sm_", cc.major, cc.minor));
#elif defined(TENSORFLOW_USE_ROCM)
  stream_executor::RocmComputeCapability cc =
      ctx->op_device_context()->stream()->GetRocmComputeCapability();
  architectures.push_back(cc.gcn_arch_name());
#endif

  // Construct `SmallVector`s from arguments.
  llvm::SmallVector<int64_t, 8> tile_sizes =
      SmallVectorFromCArray<int64_t>(num_tile_sizes, tile_sizes_ptr);
  llvm::SmallVector<int64_t, 8> unroll_factors =
      SmallVectorFromCArray<int64_t>(num_unroll_factors, unroll_factors_ptr);

  // Lookup or compile the execution module.
  ExecutionEngine* engine = jit_cache->LookupOrCompile(code, [&]() {
    return Compile(code, architectures, tile_sizes, unroll_factors, enable_ftz,
                   index_64bit);
  });
  if (engine == nullptr) {
    ReportError(op_kernel_ctx, ErrorCode::UNKNOWN, "JIT compilation failed.");
    return nullptr;
  }
  return engine;
}

extern "C" void _mlir_ciface_tf_jit_execute(void* op_kernel_ctx, void* callable,
                                            void* result, int64_t num_args,
                                            void* args_ptr) {
  // JIT compilation must have failed earlier if there is no callable ptr.
  // Return some empty memory descriptor to prevent a crash.
  if (callable == nullptr) {
    auto* desc = static_cast<::UnrankedMemRefType<void>*>(result);
    desc->rank = 0;
    auto* inner_desc = static_cast<StridedMemRefType<int8_t, 0>*>(
        malloc(sizeof(StridedMemRefType<int8_t, 0>)));
    inner_desc->basePtr = nullptr;
    inner_desc->data = nullptr;
    inner_desc->offset = 0;
    desc->descriptor = inner_desc;
    return;
  }

  // Build the argument array according to `ExecutionEngine`'s calling
  // convention.
  auto* typed_args_ptr = static_cast<::UnrankedMemRefType<void>*>(args_ptr);
  llvm::SmallVector<void*, 8> args_array = {&op_kernel_ctx};
  for (int i = 0; i < num_args; i++) {
    auto& desc = typed_args_ptr[i];
    args_array.push_back(&desc.rank);
    args_array.push_back(&desc.descriptor);
  }
  args_array.push_back(result);

  llvm::Error invocation_result =
      static_cast<ExecutionEngine*>(callable)->invokePacked("main", args_array);
  if (invocation_result)
    ReportError(op_kernel_ctx, ErrorCode::UNKNOWN, "JIT invocation failed.");
}

}  // namespace tf_framework
}  // namespace kernel_gen
}  // namespace mlir
