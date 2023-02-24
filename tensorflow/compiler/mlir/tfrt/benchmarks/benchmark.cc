/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tfrt/benchmarks/benchmark.h"

#include <string>

#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/runtime/transforms/compiler.h"
#include "tensorflow/core/platform/logging.h"
#include "tfrt/jitrt/jitrt_compiler.h"  // from @tf_runtime
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime
#include "tfrt/host_context/host_allocator.h"  // from @tf_runtime

namespace tensorflow {

using ::tfrt::HostContext;
using ::tfrt::jitrt::CompilationPipelineOptions;
using ::xla::runtime::MemrefType;

const bool kStaticDim = false;
const bool kDynamicDim = true;

mlir::LogicalResult FreeReturnedMemref(const ResultConversionCtx& ctx,
                                       RemainingResults results,
                                       unsigned result_index, const Type* type,
                                       const Type* runtime_type,
                                       void* result_ptr) {
  DCHECK(llvm::isa<MemrefType>(runtime_type)) << "expected memref result";
  // Cast result to the arbitrary chosen memref type and rank because we only
  // need to know the base pointer value.
  auto* memref = static_cast<StridedMemRefType<float, 0>*>(result_ptr);
  if (llvm::find(ctx.input_ptrs, memref->data) == ctx.input_ptrs.end()) {
    free(memref->basePtr);
  }
  return mlir::success();
}

JitExecutable& CreateJitExecutable(
    const HostContext& host, llvm::StringRef mlir_input,
    llvm::StringRef function_name, bool lower_from_tensorflow,
    const TfJitRtPipelineOptions& tf_jitrt_opts) {
  // Options for the default JitRt compilation pipeline (lowering to LLVM).
  CompilationPipelineOptions copts;
  copts.alignment = EIGEN_MAX_ALIGN_BYTES;
  copts.num_worker_threads = host.GetNumWorkerThreads();

  JitExecutable::Options opts;
  opts.compiler.register_dialects =
      [](xla::runtime::DialectRegistry& dialects) {
        mlir::RegisterAllTensorFlowDialects(*dialects);
        tfrt::jitrt::RegisterDefaultJitRtDialects(dialects);
      };
  opts.compiler.create_compilation_pipeline =
      [&, copts, lower_from_tensorflow](xla::runtime::PassManager& passes) {
        if (lower_from_tensorflow)
          tensorflow::CreateTfJitRtPipeline(*passes, tf_jitrt_opts);
        tfrt::jitrt::CreateDefaultJitRtCompilationPipeline(passes, copts);
      };
  opts.compiler.create_specialization_pipeline =
      CreateJitRtSpecializationPipeline;
  opts.compiler.calling_convention = xla::runtime::DefaultCallingConvention(
      mlir::bufferization::BufferizeTypeConverter());

  // Cache all jit executables, otherwise different benchmark runs will produce
  // different .so files and the same compiled function will have different
  // records in the perf profile.
  static auto* cache = new llvm::StringMap<std::unique_ptr<JitExecutable>>();

  std::string key =
      llvm::formatv("{0}/{1}/{2}", mlir_input.data(), copts.num_worker_threads,
                    hash_value(tf_jitrt_opts));

  // Compile and cache MLIR function.
  auto it = cache->find(key);
  if (it == cache->end()) {
    absl::StatusOr<JitExecutable> jit_executable =
        JitExecutable::Instantiate(mlir_input, function_name, opts);
    if (!jit_executable.ok())
      LOG(FATAL) << "Failed to instantiate JitExecutable from the function: "
                 << function_name.str()
                 << "; error: " << jit_executable.status().message();

    auto storage = std::make_unique<JitExecutable>(std::move(*jit_executable));
    it = cache->insert_or_assign(key, std::move(storage)).first;
  }

  return *(it->getValue());
}

MemrefDesc TensorToMemrefDesc(const Tensor& tensor) {
  llvm::SmallVector<ssize_t> dims(tensor.shape().dims());
  for (int d = 0; d < tensor.shape().dims(); ++d)
    dims[d] = tensor.shape().dim_size(d);

  xla::PrimitiveType dtype;
  if (tensor.dtype() == DT_FLOAT)
    dtype = xla::PrimitiveType::F32;
  else if (tensor.dtype() == DT_INT64)
    dtype = xla::PrimitiveType::S64;
  else
    LOG(FATAL) << "Unsupported tensor dtype: " << tensor.dtype();

  tfrt::TensorShape shape(dims);
  return MemrefDesc(
      shape.GetRank(), dtype, tensor.data(), 0, [&](auto sizes, auto strides) {
        MutableArrayRef<int64_t> sizes_ref(sizes.data(), sizes.size());
        MutableArrayRef<int64_t> strides_ref(strides.data(), strides.size());
        shape.GetDimensions(sizes_ref);
        shape.GetStrides(strides_ref);
      });
}

llvm::SmallVector<int64_t> GetTensorTypeShape(
    llvm::ArrayRef<int64_t> shape, llvm::ArrayRef<bool> dynamic_dims) {
  llvm::SmallVector<int64_t> type_shape;
  for (int64_t i = 0; i < shape.size(); ++i) {
    type_shape.push_back(
        dynamic_dims[i] == kDynamicDim ? mlir::ShapedType::kDynamic : shape[i]);
  }
  return type_shape;
}

std::string PrintTensorType(llvm::ArrayRef<int64_t> shape,
                            llvm::StringRef element_type) {
  std::string result{"tensor<"};
  llvm::raw_string_ostream ss(result);
  for (int64_t dim : shape) {
    if (mlir::ShapedType::isDynamic(dim)) {
      ss << '?';
    } else {
      ss << dim;
    }
    ss << 'x';
  }
  ss << element_type << '>';
  return result;
}

std::string PrintDenseArray(llvm::ArrayRef<int32_t> array) {
  std::string result{"dense<["};
  llvm::raw_string_ostream ss(result);
  for (auto elem : llvm::enumerate(array)) {
    if (elem.index() > 0) ss << ',';
    ss << elem.value();
  }
  ss << "]>";
  return result;
}

}  // namespace tensorflow
