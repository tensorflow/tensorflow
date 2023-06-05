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

#include <string>
#include <string_view>

#include "tensorflow/compiler/xla/service/gpu/tests/mlir_gpu_test_base.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"

namespace xla {
namespace gpu {

using ::testing::ElementsAreArray;

class GemmTest : public MlirGpuTestBase {
 public:
  std::vector<std::vector<uint8_t>> Run2x2Gemm(
      std::vector<float> arg0, std::vector<float> arg1,
      std::string matmul_options = "") {
    std::string mlir_text = absl::StrCat(
        R"(
      module attributes {hlo.unique_id = 0 : i32} {
        func.func @main(%raw_arg0: memref<16xi8> {lmhlo.params = 0 : index},
                   %raw_arg1: memref<16xi8> {lmhlo.params = 1 : index},
                   %raw_arg2: memref<16xi8> {lmhlo.output_index = dense<[0]> : tensor<1xindex>}) attributes {
                       result_xla_shape = "(f32[4]) "
                   } {
          %c0 = arith.constant 0 : index
          %arg0 = memref.view %raw_arg0[%c0][] : memref<16xi8> to memref<2x2xf32>
          %c1 = arith.constant 0 : index
          %arg1 = memref.view %raw_arg1[%c1][] : memref<16xi8> to memref<2x2xf32>
          %c2 = arith.constant 0 : index
          %arg2 = memref.view %raw_arg2[%c2][] : memref<16xi8> to memref<2x2xf32>
          "lmhlo_gpu.gemm"(%arg0, %arg1, %arg2) {alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, batch_size = 1 : i64, lhs_stride = 4 : i64, rhs_stride = 4 : i64, dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>)",
        matmul_options,
        R"(} : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
          "lmhlo.terminator"() : () -> ()
        }
      })");

    return RunMlirTextWithHostBuffers(mlir_text,
                                      {ToUint8Span(&arg0), ToUint8Span(&arg1)})
        .value();
  }

  std::vector<std::vector<uint8_t>> Run256x256Gemm(
      std::vector<float> arg0, std::vector<float> arg1,
      std::string matmul_options = "") {
    std::string mlir_text = absl::StrCat(
        R"(
      module attributes {hlo.unique_id = 0 : i32} {
        func.func @main(%raw_arg0: memref<262144xi8> {lmhlo.params = 0 : index},
                   %raw_arg1: memref<262144xi8> {lmhlo.params = 1 : index},
                   %raw_arg2: memref<262144xi8> {lmhlo.output_index = dense<[0]> : tensor<1xindex>}) attributes {
                       result_xla_shape = "(f32[4]) "
                   } {
          %c0 = arith.constant 0 : index
          %arg0 = memref.view %raw_arg0[%c0][] : memref<262144xi8> to memref<256x256xf32>
          %c1 = arith.constant 0 : index
          %arg1 = memref.view %raw_arg1[%c1][] : memref<262144xi8> to memref<256x256xf32>
          %c2 = arith.constant 0 : index
          %arg2 = memref.view %raw_arg2[%c2][] : memref<262144xi8> to memref<256x256xf32>
          "lmhlo_gpu.gemm"(%arg0, %arg1, %arg2) {alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, batch_size = 1 : i64, lhs_stride = 65536 : i64, rhs_stride = 65536 : i64, op_type = "", op_name = "", dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>)",
        matmul_options,
        R"(} : (memref<256x256xf32>, memref<256x256xf32>, memref<256x256xf32>) -> ()
          "lmhlo.terminator"() : () -> ()
        }
      })");

    return RunMlirTextWithHostBuffers(mlir_text,
                                      {ToUint8Span(&arg0), ToUint8Span(&arg1)})
        .value();
  }
};

TEST_F(GemmTest, SimpleCase1) {
  std::vector<float> arg0 = {2, 3, 4, 5};
  std::vector<float> arg1 = {1, 2, 3, 4};
  auto outputs = Run2x2Gemm(arg0, arg1);
  ASSERT_EQ(1, outputs.size());
  EXPECT_THAT(FromUint8Span<float>(outputs[0]),
              ElementsAreArray<float>({11, 16, 19, 28}));
}

static std::vector<float> diag_matrix(size_t dim, float diag_value) {
  std::vector<float> r(dim * dim, 0.0f);
  for (size_t i = 0; i < dim; ++i) {
    r[i * dim + i] = diag_value;
  }
  return r;
}

TEST_F(GemmTest, GemmPrecisionDefault) {
  auto outputs = Run256x256Gemm(
      diag_matrix(256, 0x1.fffffep+0), diag_matrix(256, 0x1.fffffep+0),
      R"(, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>])");
  ASSERT_EQ(1, outputs.size());
  auto stream = BorrowStream();
  if (stream->GetCudaComputeCapability().IsAtLeast(
          stream_executor::CudaComputeCapability::AMPERE)) {
    EXPECT_THAT(FromUint8Span<float>(outputs[0]), diag_matrix(256, 4.0f));
  } else {
    EXPECT_THAT(FromUint8Span<float>(outputs[0]),
                diag_matrix(256, 0x1.fffffcp+1));
  }
}

TEST_F(GemmTest, GemmPrecisionHighest) {
  auto outputs = Run256x256Gemm(
      diag_matrix(256, 0x1.fffffep+0), diag_matrix(256, 0x1.fffffep+0),
      R"(, precision_config = [#mhlo<precision HIGH>, #mhlo<precision HIGHEST>])");
  ASSERT_EQ(1, outputs.size());
  EXPECT_THAT(FromUint8Span<float>(outputs[0]),
              diag_matrix(256, 0x1.fffffcp+1));
}

TEST_F(GemmTest, GemmBatchedPrecisionHighest) {
  std::vector<float> arg0 = {0x1.fffffep+0, 0, 0, 0x1.fffffep+0,
                             0x1.fffffep+0, 0, 0, 0x1.fffffep+0};
  std::vector<float> arg1 = {0x1.fffffep+0, 0, 0, 0x1.fffffep+0};

  std::string_view mlir_text = R"(
    module attributes {hlo.unique_id = 0 : i32} {
      func.func @main(%raw_arg0: memref<32xi8> {lmhlo.params = 0 : index},
                 %raw_arg1: memref<16xi8> {lmhlo.params = 1 : index},
                 %raw_arg2: memref<32xi8> {lmhlo.output_index = dense<[0]> : tensor<1xindex>}) attributes {
                     result_xla_shape = "(f32[4]) "
                 } {
        %c0 = arith.constant 0 : index
        %arg0 = memref.view %raw_arg0[%c0][] : memref<32xi8> to memref<2x2x2xf32>
        %c1 = arith.constant 0 : index
        %arg1 = memref.view %raw_arg1[%c1][] : memref<16xi8> to memref<2x2xf32>
        %c2 = arith.constant 0 : index
        %arg2 = memref.view %raw_arg2[%c2][] : memref<32xi8> to memref<2x2x2xf32>
        "lmhlo_gpu.gemm"(%arg0, %arg1, %arg2) {alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, batch_size = 2 : i64, lhs_stride = 4 : i64, rhs_stride = 4 : i64, dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [0]>, precision_config = [#mhlo<precision HIGH>, #mhlo<precision HIGHEST>]} : (memref<2x2x2xf32>, memref<2x2xf32>, memref<2x2x2xf32>) -> ()
        "lmhlo.terminator"() : () -> ()
      }
    })";

  auto outputs = RunMlirTextWithHostBuffers(
                     mlir_text, {ToUint8Span(&arg0), ToUint8Span(&arg1)})
                     .value();

  ASSERT_EQ(1, outputs.size());
  EXPECT_THAT(FromUint8Span<float>(outputs[0]),
              ElementsAreArray<float>({0x1.fffffcp+1, 0, 0, 0x1.fffffcp+1,
                                       0x1.fffffcp+1, 0, 0, 0x1.fffffcp+1}));
}

}  // namespace gpu
}  // namespace xla
