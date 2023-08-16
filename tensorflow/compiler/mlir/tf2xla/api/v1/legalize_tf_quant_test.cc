/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/compiler/mlir/tf2xla/api/v0/compile_mlir_util.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"

namespace tensorflow {
namespace tf2xla {
namespace v1 {

TEST(LegalizeTFQuantTest, LegalizesModuleWithTFUniformQuantization) {
  constexpr char legalization[] = R"(
  module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
    func.func @main(%arg0 : tensor<1xf32>) -> tensor<1xf32> {
      %scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
      %zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

      %0 = "tf.UniformQuantize"(%arg0, %scales, %zps) {
        quantization_axis = -1 : i64, quantization_min_val = -128 : i64, quantization_max_val = 127 : i64
      } : (tensor<1xf32>, tensor<f32>, tensor<i32>) -> tensor<1x!tf_type.qint8>
      %1 = "tf.UniformDequantize"(%0, %scales, %zps) {
        quantization_axis = -1 : i64, quantization_min_val = -128 : i64, quantization_max_val = 127 : i64
      } : (tensor<1x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<1xf32>
      func.return %1 : tensor<1xf32>
    }
  })";

  std::vector<tensorflow::TensorShape> arg_shapes = {{1}};
  XlaCompilationResult compilation_result;

  TF_EXPECT_OK(CompileSerializedMlirToXlaHlo(
                   legalization, arg_shapes, /*device_type=*/"XLA_TPU_JIT",
                   /*use_tuple_args=*/true, /*enable_op_fallback=*/true,
                   /*shape_determination_fns=*/{}, &compilation_result)
                   .status());

  const xla::HloModuleProto& hlo_module =
      compilation_result.computation->proto();
  for (const xla::HloComputationProto computation : hlo_module.computations()) {
    for (const xla::HloInstructionProto instruction :
         computation.instructions()) {
      xla::StatusOr<xla::HloOpcode> opcode =
          xla::StringToHloOpcode(instruction.opcode());
      EXPECT_TRUE(opcode.ok());

      switch (opcode.value()) {
        case xla::HloOpcode::kConstant:
        case xla::HloOpcode::kDivide:
        case xla::HloOpcode::kAdd:
        case xla::HloOpcode::kFloor:
        case xla::HloOpcode::kConvert:
        case xla::HloOpcode::kMaximum:
        case xla::HloOpcode::kMinimum:
        case xla::HloOpcode::kSubtract:
        case xla::HloOpcode::kParameter:
        case xla::HloOpcode::kTuple:
        case xla::HloOpcode::kGetTupleElement:
          break;
        default:
          ADD_FAILURE() << "Failed to compile TF uniform quantized ops "
                        << "(unexpected opcode: " << opcode.value() << ")";
      }
    }
  }
}

}  // namespace v1
}  // namespace tf2xla
}  // namespace tensorflow
