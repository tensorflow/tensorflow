/*
 * Copyright 2022 The TensorFlow Runtime Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorflow/compiler/mlir/tfrt/jit/tf_jitrt.h"

#include <array>
#include <memory>

#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "testing/base/public/benchmark.h"
#include "testing/base/public/gunit.h"

namespace tensorflow {

using ::tfrt::AsyncValue;
using ::tfrt::DType;
using ::tfrt::RCReference;
using ::tfrt::RemainingResults;

using ::tfrt::jitrt::MemrefType;
using ::tfrt::jitrt::ReturnStridedMemref;
using ::tfrt::jitrt::ReturnValueConversion;
using ::tfrt::jitrt::StaticReturnValueConverter;

using ReturnTensorflowTensor =
    ReturnValueConversion<TensorflowConversionContext,
                          ReturnStridedMemref<ConvertTensor>>;

using TensorflowReturnValueConverter =
    StaticReturnValueConverter<TensorflowConversionContext,
                               ReturnTensorflowTensor>;

static void BM_ReturnTensor(benchmark::State& state) {
  auto dims = std::array<int64_t, 4>({1, 2, 3, 4});
  auto type = std::make_unique<MemrefType>(dims, DType::F32);

  // Prepare a memref descriptor that will be returned as a tensor.
  StridedMemRefType<float, 4> memref{
      /*basePtr=*/reinterpret_cast<float*>(0xDEADBEEF),
      /*data=*/nullptr,
      /*offset=*/0,
      /*sizes=*/{1, 2, 3, 4},
      /*strides=*/{24, 12, 4, 1}};

  for (auto _ : state) {
    std::array<RCReference<AsyncValue>, 1> storage;
    RemainingResults results(storage);

    TensorflowConversionContext context(0, /*num_results=*/1);
    TensorflowReturnValueConverter converter(results, context);

    auto converted = converter.ReturnValue(0, type.get(), type.get(), &memref);
    CHECK(mlir::succeeded(converted)) << "Failed to convert memref";
  }
}

BENCHMARK(BM_ReturnTensor);

}  // namespace tensorflow
