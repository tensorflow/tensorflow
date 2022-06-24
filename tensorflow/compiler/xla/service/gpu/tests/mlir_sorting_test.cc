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

class SortingTest : public MlirGpuTestBase {};

TEST_F(SortingTest, SimpleCase1) {
  const char* mlir_text = R"(
      func.func @main(%arg0: memref<4xf32> {lmhlo.params = 0 : index},
                 %arg1: memref<4xf32> {lmhlo.params = 1 : index},
                 %arg2: memref<4xf32> {lmhlo.output_index = dense<[0]> : tensor<1xindex>},
                 %arg3: memref<4xf32> {lmhlo.output_index = dense<[1]> : tensor<1xindex>},
                 %arg4: memref<4xf32> {lmhlo.output_index = dense<[2]> : tensor<1xindex>},
                 %arg5: memref<4xf32> {lmhlo.output_index = dense<[3]> : tensor<1xindex>}) attributes {
                     result_xla_shape = "(f32[4], f32[4], f32[4], f32[4]) "
                 } {
          "lmhlo.sort"(%arg0, %arg1, %arg2, %arg3) ({
          ^bb0(%a: tensor<f32>, %b: tensor<f32>, %c: tensor<f32>, %d: tensor<f32>):
            %7 = "mhlo.compare"(%a, %b) {comparison_direction = #mhlo<"comparison_direction GT">} : (tensor<f32>, tensor<f32>) -> tensor<i1>
            "mhlo.return"(%7) : (tensor<i1>) -> ()
          }) {dimension = 0 : i64, is_stable = true} : (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>) -> ()
          "lmhlo.sort"(%arg0, %arg1, %arg4, %arg5) ({
          ^bb0(%a: tensor<f32>, %b: tensor<f32>, %c: tensor<f32>, %d: tensor<f32>):
            %7 = "mhlo.compare"(%a, %b) {comparison_direction = #mhlo<"comparison_direction LT">} : (tensor<f32>, tensor<f32>) -> tensor<i1>
            "mhlo.return"(%7) : (tensor<i1>) -> ()
          }) {dimension = 0 : i64, is_stable = true} : (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>) -> ()
          "func.return" () : () -> ()
      })";
  std::vector<float> arg0 = {3, 1, 2, 4};
  std::vector<float> arg1 = {13, 12, 14, 11};
  auto outputs = RunMlirTextWithHostBuffers(
                     mlir_text, {ToUint8Span(&arg0), ToUint8Span(&arg1)})
                     .value();
  ASSERT_EQ(4, outputs.size());
  EXPECT_THAT(FromUint8Span<float>(outputs[0]),
              ElementsAreArray<float>({4, 3, 2, 1}));
  EXPECT_THAT(FromUint8Span<float>(outputs[1]),
              ElementsAreArray<float>({11, 13, 14, 12}));
  EXPECT_THAT(FromUint8Span<float>(outputs[2]),
              ElementsAreArray<float>({1, 2, 3, 4}));
  EXPECT_THAT(FromUint8Span<float>(outputs[3]),
              ElementsAreArray<float>({12, 14, 13, 11}));
}

}  // namespace gpu
}  // namespace xla
