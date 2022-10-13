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
#include "tensorflow/tsl/lib/core/status_test_util.h"

namespace xla {
namespace gpu {

using ::testing::ElementsAreArray;

class FftTest : public MlirGpuTestBase {};

TEST_F(FftTest, SimpleCase1) {
  const char* mlir_text = R"(
      module attributes {hlo.unique_id = 0 : i32} {
        func.func @main(%arg0: memref<4xf32> {
                       lmhlo.params = 0 : index
                   },
                   %arg1: memref<3xcomplex<f32>> {
                       lmhlo.output_index = dense<[0]> : tensor<1xindex>
                   }
        ) attributes {
            result_xla_shape = "(f32[6]) "
        } {
          "lmhlo.fft"(%arg0, %arg1) {
            fft_length = dense<4> : tensor<1xi64>,
            fft_type = #mhlo<fft_type RFFT>
          } : (memref<4xf32>, memref<3xcomplex<f32>>) -> ()
          "lmhlo.terminator"() : () -> ()
        }
      })";
  std::vector<float> arg0 = {1, 0, 1, 0};
  auto outputs =
      RunMlirTextWithHostBuffers(mlir_text, {ToUint8Span(&arg0)}).value();
  ASSERT_EQ(1, outputs.size());
  EXPECT_THAT(FromUint8Span<float>(outputs[0]),
              ElementsAreArray<float>({2, 0, 0, 0, 2, 0}));
}

}  // namespace gpu
}  // namespace xla
