/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/mlir_gpu/mlir_irgen_test_base.h"
#include "tensorflow/core/platform/path.h"

namespace xla {
namespace mlir_gpu {

class LhloGenTest : public MlirIrGenTestBase {};

TEST_F(LhloGenTest, Const) {
  CompileAndVerifyIr(
      /*hlo_text_filename=*/tensorflow::io::JoinPath(
          "tensorflow", "compiler", "xla", "service", "mlir_gpu", "tests",
          "const.hlo"),
      LoweringStage::LHLO);
}

TEST_F(LhloGenTest, BrokenAdd) {
  CompileAndVerifyErrors(
      /*hlo_text_filename=*/
      tensorflow::io::JoinPath("tensorflow", "compiler", "xla", "service",
                               "mlir_gpu", "tests", "broken_add.hlo"),
      LoweringStage::LHLO);
}

TEST_F(LhloGenTest, Add) {
  CompileAndVerifyIr(
      /*hlo_text_filename=*/tensorflow::io::JoinPath(
          "tensorflow", "compiler", "xla", "service", "mlir_gpu", "tests",
          "add.hlo"));
}

TEST_F(LhloGenTest, Compare) {
  CompileAndVerifyIr(
      /*hlo_text_filename=*/tensorflow::io::JoinPath(
          "tensorflow", "compiler", "xla", "service", "mlir_gpu", "tests",
          "compare.hlo"));
}

TEST_F(LhloGenTest, Copy) {
  CompileAndVerifyIr(
      /*hlo_text_filename=*/tensorflow::io::JoinPath(
          "tensorflow", "compiler", "xla", "service", "mlir_gpu", "tests",
          "copy.hlo"));
}

TEST_F(LhloGenTest, CopyTranspose) {
  CompileAndVerifyIr(
      /*hlo_text_filename=*/tensorflow::io::JoinPath(
          "tensorflow", "compiler", "xla", "service", "mlir_gpu", "tests",
          "copy_transpose.hlo"));
}

TEST_F(LhloGenTest, Select) {
  CompileAndVerifyIr(
      /*hlo_text_filename=*/tensorflow::io::JoinPath(
          "tensorflow", "compiler", "xla", "service", "mlir_gpu", "tests",
          "select.hlo"));
}

TEST_F(LhloGenTest, Exp) {
  CompileAndVerifyIr(
      /*hlo_text_filename=*/tensorflow::io::JoinPath(
          "tensorflow", "compiler", "xla", "service", "mlir_gpu", "tests",
          "exp.hlo"));
}

TEST_F(LhloGenTest, Log) {
  CompileAndVerifyIr(
      /*hlo_text_filename=*/tensorflow::io::JoinPath(
          "tensorflow", "compiler", "xla", "service", "mlir_gpu", "tests",
          "log.hlo"));
}

TEST_F(LhloGenTest, AddInGPUDialect) {
  CompileAndVerifyIr(
      /*hlo_text_filename=*/
      tensorflow::io::JoinPath("tensorflow", "compiler", "xla", "service",
                               "mlir_gpu", "tests", "add_in_gpu_dialect.hlo"),
      LoweringStage::GPU);
}

// This test verifies that the kernel signature is amended correctly. The actual
// body of the generated function does not matter, it is already checked at the
// GPU level above.
TEST_F(LhloGenTest, AddAsKernel) {
  CompileAndVerifyIr(
      tensorflow::io::JoinPath("tensorflow", "compiler", "xla", "service",
                               "mlir_gpu", "tests", "add_as_kernel.hlo"),
      LoweringStage::KERNEL);
}

// TODO(b/149302060) Reenable once fusion is fixed.
TEST_F(LhloGenTest, DISABLED_AddMultiply) {
  CompileAndVerifyIr(tensorflow::io::JoinPath("tensorflow", "compiler", "xla",
                                              "service", "mlir_gpu", "tests",
                                              "add_multiply.hlo"));
}

// TODO(b/149302060) Reenable once fusion is fixed.
TEST_F(LhloGenTest, DISABLED_IotaAddMultiply) {
  CompileAndVerifyIr(
      tensorflow::io::JoinPath("tensorflow", "compiler", "xla", "service",
                               "mlir_gpu", "tests", "iota_add_multiply.hlo"),
      LoweringStage::GPU);
}

TEST_F(LhloGenTest, AddMultiplyGPU) {
  CompileAndVerifyIr(
      tensorflow::io::JoinPath("tensorflow", "compiler", "xla", "service",
                               "mlir_gpu", "tests", "add_multiply_gpu.hlo"),
      LoweringStage::GPU);
}

// TODO(b/137624192): Reenable once we can fuse reductions.
TEST_F(LhloGenTest, DISABLED_FusedReduce) {
  CompileAndVerifyIr(tensorflow::io::JoinPath("tensorflow", "compiler", "xla",
                                              "service", "mlir_gpu", "tests",
                                              "fused_reduce.hlo"));
}

TEST_F(LhloGenTest, Broadcast) {
  CompileAndVerifyIr(tensorflow::io::JoinPath("tensorflow", "compiler", "xla",
                                              "service", "mlir_gpu", "tests",
                                              "broadcast.hlo"));
}

TEST_F(LhloGenTest, Iota) {
  CompileAndVerifyIr(tensorflow::io::JoinPath("tensorflow", "compiler", "xla",
                                              "service", "mlir_gpu", "tests",
                                              "iota.hlo"));
}

TEST_F(LhloGenTest, AddReduce) {
  CompileAndVerifyIr(tensorflow::io::JoinPath("tensorflow", "compiler", "xla",
                                              "service", "mlir_gpu", "tests",
                                              "add_reduce.hlo"));
}

TEST_F(LhloGenTest, Abs) {
  CompileAndVerifyIr(tensorflow::io::JoinPath("tensorflow", "compiler", "xla",
                                              "service", "mlir_gpu", "tests",
                                              "abs.hlo"));
}

TEST_F(LhloGenTest, Ceil) {
  CompileAndVerifyIr(tensorflow::io::JoinPath("tensorflow", "compiler", "xla",
                                              "service", "mlir_gpu", "tests",
                                              "ceil.hlo"));
}

TEST_F(LhloGenTest, Cos) {
  CompileAndVerifyIr(tensorflow::io::JoinPath("tensorflow", "compiler", "xla",
                                              "service", "mlir_gpu", "tests",
                                              "cos.hlo"));
}

TEST_F(LhloGenTest, Neg) {
  CompileAndVerifyIr(tensorflow::io::JoinPath("tensorflow", "compiler", "xla",
                                              "service", "mlir_gpu", "tests",
                                              "neg.hlo"));
}

TEST_F(LhloGenTest, ReduceWindow) {
  CompileAndVerifyIr(tensorflow::io::JoinPath("tensorflow", "compiler", "xla",
                                              "service", "mlir_gpu", "tests",
                                              "reduce_window.hlo"));
}

TEST_F(LhloGenTest, Rem) {
  CompileAndVerifyIr(tensorflow::io::JoinPath("tensorflow", "compiler", "xla",
                                              "service", "mlir_gpu", "tests",
                                              "rem.hlo"));
}

TEST_F(LhloGenTest, Rsqrt) {
  CompileAndVerifyIr(tensorflow::io::JoinPath("tensorflow", "compiler", "xla",
                                              "service", "mlir_gpu", "tests",
                                              "rsqrt.hlo"));
}

TEST_F(LhloGenTest, SelectAndScatter) {
  CompileAndVerifyIr(tensorflow::io::JoinPath("tensorflow", "compiler", "xla",
                                              "service", "mlir_gpu", "tests",
                                              "select_and_scatter.hlo"));
}

TEST_F(LhloGenTest, Sign) {
  CompileAndVerifyIr(tensorflow::io::JoinPath("tensorflow", "compiler", "xla",
                                              "service", "mlir_gpu", "tests",
                                              "rsqrt.hlo"));
}

TEST_F(LhloGenTest, Sqrt) {
  CompileAndVerifyIr(
      /*hlo_text_filename=*/tensorflow::io::JoinPath(
          "tensorflow", "compiler", "xla", "service", "mlir_gpu", "tests",
          "sqrt.hlo"));
}

TEST_F(LhloGenTest, Tanh) {
  CompileAndVerifyIr(tensorflow::io::JoinPath("tensorflow", "compiler", "xla",
                                              "service", "mlir_gpu", "tests",
                                              "tanh.hlo"));
}

}  // namespace mlir_gpu
}  // namespace xla
