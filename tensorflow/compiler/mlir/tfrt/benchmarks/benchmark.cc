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

#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "tensorflow/core/platform/logging.h"
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime
#include "tfrt/host_context/host_allocator.h"  // from @tf_runtime

namespace tensorflow {

using ::tfrt::HostContext;
using ::tfrt::cpu::jit::CompilationOptions;
using ::tfrt::cpu::jit::MemrefType;

const bool kStaticDim = false;
const bool kDynamicDim = true;

std::unique_ptr<HostContext> CreateSingleThreadedHostContext() {
  return std::make_unique<HostContext>(
      [](const tfrt::DecodedDiagnostic& diag) {
        LOG(FATAL) << "Runtime error: " << diag.message << "\n";
      },
      tfrt::CreateMallocAllocator(), tfrt::CreateSingleThreadedWorkQueue());
}

std::unique_ptr<HostContext> CreateMultiThreadedHostContext(int num_threads) {
  return std::make_unique<HostContext>(
      [](const tfrt::DecodedDiagnostic& diag) {
        LOG(FATAL) << "Runtime error: " << diag.message << "\n";
      },
      tfrt::CreateMallocAllocator(),
      tfrt::CreateMultiThreadedWorkQueue(num_threads,
                                         /*num_blocking_threads=*/1));
}

mlir::LogicalResult FreeReturnedMemref(const ResultConversionCtx&,
                                       RemainingResults results,
                                       unsigned result_index, const Type* type,
                                       const Type* runtime_type,
                                       void* result_ptr) {
  DCHECK(llvm::isa<MemrefType>(runtime_type)) << "expected memref result";
  // Cast result to the arbitrary chosen memref type and rank because we only
  // need to know the base pointer value.
  auto* memref = static_cast<StridedMemRefType<float, 0>*>(result_ptr);
  free(memref->basePtr);
  return mlir::success();
}

JitExecutable& CreateJitExecutable(
    const HostContext& host, llvm::StringRef mlir_input,
    llvm::StringRef function_name, bool lower_from_tensorflow,
    const TfCpuRtPipelineOptions& tf_cpurt_opts) {
  CompilationOptions opts;
  opts.num_worker_threads = host.GetNumWorkerThreads();
  opts.register_dialects = mlir::RegisterAllTensorFlowDialects;
  if (lower_from_tensorflow) {
    opts.register_pass_pipeline = [&](mlir::OpPassManager& pm) {
      tensorflow::CreateTfCpuRtPipeline(pm, tf_cpurt_opts);
    };
  }
  opts.type_converter = mlir::bufferization::BufferizeTypeConverter();

  // Cache all jit executables, otherwise different benchmark runs will produce
  // different .so files and the same compiled function will have different
  // records in the perf profile.
  static auto* cache = new llvm::StringMap<std::unique_ptr<JitExecutable>>();

  std::string key =
      llvm::formatv("{0}/{1}/{2}", mlir_input.data(), opts.num_worker_threads,
                    hash_value(tf_cpurt_opts));

  // Compile and cache MLIR function.
  auto it = cache->find(key);
  if (it == cache->end()) {
    llvm::Expected<JitExecutable> jit_executable =
        JitExecutable::Instantiate(mlir_input, function_name, opts);
    if (auto err = jit_executable.takeError())
      LOG(FATAL) << "Failed to instantiate JitExecutable from the function: "
                 << function_name.str() << "; error: " << tfrt::StrCat(err);

    auto storage = std::make_unique<JitExecutable>(std::move(*jit_executable));
    it = cache->insert_or_assign(key, std::move(storage)).first;
  }

  return *(it->getValue());
}

MemrefDesc TensorToMemrefDesc(const Tensor& tensor) {
  llvm::SmallVector<ssize_t> dims(tensor.shape().dims());
  for (int d = 0; d < tensor.shape().dims(); ++d)
    dims[d] = tensor.shape().dim_size(d);

  tfrt::DType dtype;
  if (tensor.dtype() == DT_FLOAT)
    dtype = tfrt::GetDType<float>();
  else if (tensor.dtype() == DT_INT64)
    dtype = tfrt::GetDType<int64_t>();
  else
    LOG(FATAL) << "Unsupported tensor dtype: " << tensor.dtype();

  tfrt::TensorShape shape(dims);
  MemrefDesc desc;
  desc.dtype = dtype;
  desc.data = tensor.data();
  desc.offset = 0;
  shape.GetDimensions(&desc.sizes);
  shape.GetStrides(&desc.strides);
  return desc;
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
