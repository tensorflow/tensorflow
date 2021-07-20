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

#include "tensorflow/compiler/xla/service/gpu/tests/mlir_gpu_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace gpu {

using ::testing::ElementsAreArray;

class GemmTest : public MlirGpuTestBase {};

TEST_F(GemmTest, SimpleCase1) {
  const char* mlir_text = R"(
      module attributes {hlo.unique_id = 0 : i32} {
        func @main(%arg0: memref<2x2xf32> {lmhlo.params = 0 : index},
                   %arg1: memref<2x2xf32> {lmhlo.params = 1 : index},
                   %arg2: memref<2x2xf32> {lmhlo.output_index = dense<[0]> : tensor<1xindex>}) attributes {
                       result_xla_shape = "(f32[4]) "
                   } {
          "lmhlo_gpu.gemm"(%arg0, %arg1, %arg2) {alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, batch_size = 1 : i64, dot_dimension_numbers = {lhs_batching_dimensions = dense<> : tensor<0xi64>, lhs_contracting_dimensions = dense<1> : tensor<1xi64>, rhs_batching_dimensions = dense<> : tensor<0xi64>, rhs_contracting_dimensions = dense<0> : tensor<1xi64>}} : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
          "lmhlo.terminator"() : () -> ()
        }
      })";
  std::vector<float> arg0 = {2, 3, 4, 5};
  std::vector<float> arg1 = {1, 2, 3, 4};
  auto outputs = RunMlirTextWithHostBuffers(
                     mlir_text, {ToUint8Span(&arg0), ToUint8Span(&arg1)})
                     .ConsumeValueOrDie();
  ASSERT_EQ(1, outputs.size());
  EXPECT_THAT(FromUint8Span<float>(outputs[0]),
              ElementsAreArray<float>({11, 16, 19, 28}));
}

}  // namespace gpu
}  // namespace xla
