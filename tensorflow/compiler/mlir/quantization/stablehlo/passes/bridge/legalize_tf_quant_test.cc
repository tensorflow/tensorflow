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
#include "tensorflow/compiler/mlir/tf2xla/api/v1/compile_mlir_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/hlo.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace tf2xla {
namespace v2 {

TEST(LegalizeTFQuantTest, LegalizesModuleWithTFUniformQuantization) {
  constexpr char legalization[] = R"mlir(
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
  })mlir";

  std::vector<tensorflow::TensorShape> arg_shapes = {{1}};
  XlaCompilationResult compilation_result;

  TF_ASSERT_OK(CompileSerializedMlirToXlaHlo(
                   legalization, arg_shapes, /*device_type=*/"XLA_TPU_JIT",
                   /*use_tuple_args=*/true, /*enable_op_fallback=*/true,
                   /*shape_determination_fns=*/{}, &compilation_result)
                   .status());

  const xla::HloModuleProto& hlo_module =
      compilation_result.computation->proto();
  for (const xla::HloComputationProto computation : hlo_module.computations()) {
    for (const xla::HloInstructionProto instruction :
         computation.instructions()) {
      TF_ASSERT_OK_AND_ASSIGN(xla::HloOpcode opcode,
                              xla::StringToHloOpcode(instruction.opcode()));
      switch (opcode) {
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
        case xla::HloOpcode::kBroadcast:
        case xla::HloOpcode::kClamp:
        case xla::HloOpcode::kRoundNearestEven:
          break;
        default:
          ADD_FAILURE() << "Failed to compile TF uniform quantized ops "
                        << "(unexpected opcode: " << opcode << ")";
      }
    }
  }
}

TEST(LegalizeTFQuantTest, LegalizesModuleWithDequantize) {
  constexpr char legalization[] = R"mlir(
  module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
    func.func @main(%arg0: tensor<1x!tf_type.qint8>) -> tensor<1xf32> {
      %min_range = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
      %max_range = "tf.Const"() { value = dense<5.0> : tensor<f32> } : () -> tensor<f32>
      %0 = "tf.Dequantize"(%arg0, %min_range, %max_range) : (tensor<1x!tf_type.qint8>, tensor<f32>, tensor<f32>) -> tensor<1xf32>
      func.return %0 : tensor<1xf32>
    }
  })mlir";

  std::vector<tensorflow::TensorShape> arg_shapes = {{1}};
  XlaCompilationResult compilation_result;

  TF_EXPECT_OK(CompileSerializedMlirToXlaHlo(
                   legalization, arg_shapes, /*device_type=*/"XLA_CPU_JIT",
                   /*use_tuple_args=*/true, /*enable_op_fallback=*/true,
                   /*shape_determination_fns=*/{}, &compilation_result)
                   .status());
}

TEST(LegalizeTFQuantTest, LegalizesModuleWithClipByValue) {
  constexpr char legalization[] = R"mlir(
  module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
    func.func @main(%arg0 : tensor<2x2xf32>) -> tensor<2x2xf32> {
      %max = "tf.Const"() { value = dense<12.0> : tensor<f32> } : () -> tensor<f32>
      %min = "tf.Const"() { value = dense<-25.0> : tensor<f32> } : () -> tensor<f32>
      %scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
      %zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

      %0 = "tf.UniformQuantize"(%arg0, %scales, %zps) {
        quantization_axis = -1 : i64, quantization_min_val = -2147483648 : i64, quantization_max_val = 2147483647 : i64
      } : (tensor<2x2xf32>, tensor<f32>, tensor<i32>) -> tensor<2x2x!tf_type.qint32>
      %qmax = "tf.UniformQuantize"(%max, %scales, %zps) {
        quantization_axis = -1 : i64, quantization_min_val = -2147483648 : i64, quantization_max_val = 2147483647 : i64
      } : (tensor<f32>, tensor<f32>, tensor<i32>) -> tensor<!tf_type.qint32>
      %qmin = "tf.UniformQuantize"(%min, %scales, %zps) {
        quantization_axis = -1 : i64, quantization_min_val = -2147483648 : i64, quantization_max_val = 2147483647 : i64
      } : (tensor<f32>, tensor<f32>, tensor<i32>) -> tensor<!tf_type.qint32>

      %1 = "tf.UniformQuantizedClipByValue"(%0, %qmin, %qmax, %scales, %zps) {
        quantization_axis = -1 : i64, quantization_min_val = -2147483648 : i64, quantization_max_val = 2147483647 : i64
      } : (tensor<2x2x!tf_type.qint32>, tensor<!tf_type.qint32>, tensor<!tf_type.qint32>, tensor<f32>, tensor<i32>) -> tensor<2x2x!tf_type.qint32>

      %2 = "tf.UniformDequantize"(%1, %scales, %zps) {
        quantization_axis = -1 : i64, quantization_min_val = -2147483648 : i64, quantization_max_val = 2147483647 : i64
      } : (tensor<2x2x!tf_type.qint32>, tensor<f32>, tensor<i32>) -> tensor<2x2xf32>
      func.return %2 : tensor<2x2xf32>
    }
  })mlir";

  std::vector<tensorflow::TensorShape> arg_shapes = {{2, 2}};
  XlaCompilationResult compilation_result;

  TF_EXPECT_OK(CompileSerializedMlirToXlaHlo(
                   legalization, arg_shapes, /*device_type=*/"XLA_TPU_JIT",
                   /*use_tuple_args=*/true, /*enable_op_fallback=*/true,
                   /*shape_determination_fns=*/{}, &compilation_result)
                   .status());
}

}  // namespace v2
}  // namespace tf2xla
}  // namespace tensorflow
