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

#include "tensorflow/compiler/mlir/tensorflow/utils/compile_mlir_util.h"

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace tensorflow {
namespace {

// A dummy shape representation function that simply converts given shape into
// an xla::Shape without assigning any layouts.
xla::StatusOr<xla::Shape> TestShapeRepresentation(const TensorShape& shape,
                                                  DataType type,
                                                  bool use_fast_memory) {
  xla::Shape xla_shape;
  TF_RETURN_IF_ERROR(TensorShapeToXLAShape(type, shape, &xla_shape));
  return xla_shape;
}

TEST(CompileSerializedMlirToXlaHloTest, InvalidSerliazedMlirModule) {
  string invalid_mlir_module = "totally @invalid MLIR module {here} <-";
  std::vector<TensorShape> arg_shapes;
  XlaCompiler::CompilationResult compilation_result;

  Status s = CompileSerializedMlirToXlaHlo(
      invalid_mlir_module, absl::Span<TensorShape>(arg_shapes),
      TestShapeRepresentation, &compilation_result);
  EXPECT_EQ(s.code(), tensorflow::errors::Code::INVALID_ARGUMENT);
}

TEST(CompileSerializedMlirToXlaHloTest, Success) {
  string mlir_module = R"(
    module attributes {tf.versions = {producer = 179 : i32}} {
      func @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
        %0 = "tf.AddV2"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", name = "add"} : (tensor<f32>, tensor<f32>) -> tensor<f32>
        return %0 : tensor<f32>
      }
    }
  )";

  std::vector<TensorShape> arg_shapes(2, TensorShape());
  XlaCompiler::CompilationResult compilation_result;

  Status s = CompileSerializedMlirToXlaHlo(
      mlir_module, absl::Span<TensorShape>(arg_shapes), TestShapeRepresentation,
      &compilation_result);
  ASSERT_TRUE(s.ok());

  const xla::HloModuleConfig module_config(
      compilation_result.computation->GetProgramShape().ValueOrDie());
  auto status_or_hlo_module = xla::HloModule::CreateFromProto(
      compilation_result.computation->proto(), module_config);
  ASSERT_TRUE(status_or_hlo_module.ok());
  string expected_hlo_module_string = R"(HloModule main.6

ENTRY %main.6 (arg_tuple.1: (f32[], f32[])) -> (f32[]) {
  %arg_tuple.1 = (f32[], f32[]) parameter(0)
  %get-tuple-element.2 = f32[] get-tuple-element((f32[], f32[]) %arg_tuple.1), index=0
  %get-tuple-element.3 = f32[] get-tuple-element((f32[], f32[]) %arg_tuple.1), index=1
  %add.4 = f32[] add(f32[] %get-tuple-element.2, f32[] %get-tuple-element.3)
  ROOT %tuple.5 = (f32[]) tuple(f32[] %add.4)
}

)";
  EXPECT_EQ(status_or_hlo_module.ValueOrDie()->ToString(),
            expected_hlo_module_string);

  // Expect an iota like input mapping.
  EXPECT_EQ(compilation_result.input_mapping, std::vector<int>({0, 1}));

  // Expect a single tuple-shape, containing two F32 scalars.
  EXPECT_EQ(compilation_result.xla_input_shapes.size(), 1);
  xla::Shape expected_input_shape =
      xla::ShapeUtil::MakeTupleShape({xla::ShapeUtil::MakeShape(xla::F32, {}),
                                      xla::ShapeUtil::MakeShape(xla::F32, {})});
  EXPECT_EQ(compilation_result.xla_input_shapes.front(), expected_input_shape);

  // Expect output shape is a tuple shape containing a single F32 Scalar type.
  const xla::Shape output_shape =
      xla::ShapeUtil::MakeShape(xla::PrimitiveType::F32, {});
  const xla::Shape tuple_output_shape =
      xla::ShapeUtil::MakeTupleShape({output_shape});
  EXPECT_EQ(compilation_result.xla_output_shape, tuple_output_shape);

  // Expect exactly 1 OutputDescrpition.
  EXPECT_EQ(compilation_result.outputs.size(), 1);
  const XlaCompiler::OutputDescription& output_desc =
      compilation_result.outputs.front();
  EXPECT_EQ(output_desc.type, DataType::DT_FLOAT);
  EXPECT_EQ(output_desc.shape, TensorShape());
  EXPECT_FALSE(output_desc.is_constant);
  EXPECT_FALSE(output_desc.is_tensor_list);

  // Expect no resource updates from computation.
  EXPECT_TRUE(compilation_result.resource_updates.empty());
}

TEST(CompileSerializedMlirToXlaHloTest, ShapeInference) {
  string mlir_module = R"(
    module attributes {tf.versions = {producer = 179 : i32}} {
      func @main(%arg0: tensor<*xf32>, %arg1: tensor<?x19xf32>) -> tensor<?x19xf32> {
        %0 = "tf.MatMul"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", transpose_a = false, transpose_b = false} : (tensor<*xf32>, tensor<?x19xf32>) -> tensor<?x19xf32>
        return %0 : tensor<?x19xf32>
      }
    }
  )";

  std::vector<TensorShape> arg_shapes{TensorShape({10, 17}),
                                      TensorShape({17, 19})};
  XlaCompiler::CompilationResult compilation_result;

  Status s = CompileSerializedMlirToXlaHlo(
      mlir_module, absl::Span<TensorShape>(arg_shapes), TestShapeRepresentation,
      &compilation_result);
  TF_ASSERT_OK(s);

  const xla::HloModuleConfig module_config(
      compilation_result.computation->GetProgramShape().ValueOrDie());
  auto status_or_hlo_module = xla::HloModule::CreateFromProto(
      compilation_result.computation->proto(), module_config);
  ASSERT_TRUE(status_or_hlo_module.ok());
  string expected_hlo_module_string = R"(HloModule main.6

ENTRY %main.6 (arg_tuple.1: (f32[10,17], f32[17,19])) -> (f32[10,19]) {
  %arg_tuple.1 = (f32[10,17]{1,0}, f32[17,19]{1,0}) parameter(0)
  %get-tuple-element.2 = f32[10,17]{1,0} get-tuple-element((f32[10,17]{1,0}, f32[17,19]{1,0}) %arg_tuple.1), index=0
  %get-tuple-element.3 = f32[17,19]{1,0} get-tuple-element((f32[10,17]{1,0}, f32[17,19]{1,0}) %arg_tuple.1), index=1
  %dot.4 = f32[10,19]{1,0} dot(f32[10,17]{1,0} %get-tuple-element.2, f32[17,19]{1,0} %get-tuple-element.3), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT %tuple.5 = (f32[10,19]{1,0}) tuple(f32[10,19]{1,0} %dot.4)
}

)";
  EXPECT_EQ(status_or_hlo_module.ValueOrDie()->ToString(),
            expected_hlo_module_string);
}

}  // namespace
}  // namespace tensorflow
