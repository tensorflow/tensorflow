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
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/testlib.h"
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

TEST(CompileSerializedMlirToXlaHloTest, InvalidSerializedMlirModule) {
  constexpr char invalid_mlir_module[] =
      "totally @invalid MLIR module {here} <-";
  std::vector<TensorShape> arg_shapes;
  XlaCompiler::CompilationResult compilation_result;

  Status s = CompileSerializedMlirToXlaHlo(
      invalid_mlir_module, arg_shapes, "XLA_CPU_JIT",
      /*use_tuple_args=*/true, TestShapeRepresentation, &compilation_result);
  EXPECT_EQ(s.code(), tensorflow::errors::Code::INVALID_ARGUMENT);
  EXPECT_EQ(s.ToString(),
            "Invalid argument: could not parse MLIR module-:1:1: error: "
            "custom op 'totally' is unknown\n");
}

constexpr llvm::StringRef kBinaryAddModule = R"(
  module attributes {tf.versions = {producer = 179 : i32}} {
    func @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
      %0 = "tf.AddV2"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", name = "add"} : (tensor<f32>, tensor<f32>) -> tensor<f32>
      return %0 : tensor<f32>
    }
  }
)";

TEST(CompileSerializedMlirToXlaHloTest, TupleArgs) {
  std::vector<TensorShape> arg_shapes(2, TensorShape());
  XlaCompiler::CompilationResult compilation_result;

  Status s = CompileSerializedMlirToXlaHlo(
      kBinaryAddModule, arg_shapes, "XLA_CPU_JIT",
      /*use_tuple_args=*/true, TestShapeRepresentation, &compilation_result);
  TF_ASSERT_OK(s);

  const xla::HloModuleConfig module_config(
      compilation_result.computation->GetProgramShape().ValueOrDie());
  auto status_or_hlo_module = xla::HloModule::CreateFromProto(
      compilation_result.computation->proto(), module_config);
  TF_ASSERT_OK(status_or_hlo_module.status());
  constexpr char expected_hlo_module_string[] = R"(HloModule main.6

ENTRY %main.6 (arg_tuple.1: (f32[], f32[])) -> (f32[]) {
  %arg_tuple.1 = (f32[], f32[]) parameter(0)
  %get-tuple-element.2 = f32[] get-tuple-element((f32[], f32[]) %arg_tuple.1), index=0
  %get-tuple-element.3 = f32[] get-tuple-element((f32[], f32[]) %arg_tuple.1), index=1
  %add.4 = f32[] add(f32[] %get-tuple-element.2, f32[] %get-tuple-element.3)
  ROOT %tuple.5 = (f32[]) tuple(f32[] %add.4)
}

)";
  EXPECT_EQ(expected_hlo_module_string,
            status_or_hlo_module.ValueOrDie()->ToString());

  // Expect an in order input mapping.
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

  // Expect exactly 1 OutputDescription.
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

TEST(CompileSerializedMlirToXlaHloTest, IndividualArgs) {
  std::vector<TensorShape> arg_shapes(2, TensorShape());
  XlaCompiler::CompilationResult compilation_result;

  Status s = CompileSerializedMlirToXlaHlo(
      kBinaryAddModule, arg_shapes, "XLA_CPU_JIT",
      /*use_tuple_args=*/false, TestShapeRepresentation, &compilation_result);
  TF_ASSERT_OK(s);

  const xla::HloModuleConfig module_config(
      compilation_result.computation->GetProgramShape().ValueOrDie());
  auto status_or_hlo_module = xla::HloModule::CreateFromProto(
      compilation_result.computation->proto(), module_config);
  TF_ASSERT_OK(status_or_hlo_module.status());
  constexpr char expected_hlo_module_string[] = R"(HloModule main.5

ENTRY %main.5 (Arg_0.1: f32[], Arg_1.2: f32[]) -> (f32[]) {
  %Arg_0.1 = f32[] parameter(0)
  %Arg_1.2 = f32[] parameter(1)
  %add.3 = f32[] add(f32[] %Arg_0.1, f32[] %Arg_1.2)
  ROOT %tuple.4 = (f32[]) tuple(f32[] %add.3)
}

)";
  EXPECT_EQ(expected_hlo_module_string,
            status_or_hlo_module.ValueOrDie()->ToString());

  // Expect an in order input mapping.
  EXPECT_EQ(compilation_result.input_mapping, std::vector<int>({0, 1}));

  // Expect two inputs, each containing a F32 scalar.
  EXPECT_EQ(compilation_result.xla_input_shapes.size(), 2);
  xla::Shape expected_input_shape = xla::ShapeUtil::MakeShape(xla::F32, {});
  EXPECT_EQ(compilation_result.xla_input_shapes[0], expected_input_shape);
  EXPECT_EQ(compilation_result.xla_input_shapes[1], expected_input_shape);

  // Expect output shape is a tuple shape containing a single F32 Scalar type.
  const xla::Shape output_shape =
      xla::ShapeUtil::MakeShape(xla::PrimitiveType::F32, {});
  const xla::Shape tuple_output_shape =
      xla::ShapeUtil::MakeTupleShape({output_shape});
  EXPECT_EQ(compilation_result.xla_output_shape, tuple_output_shape);

  // Expect exactly 1 OutputDescription.
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

// Tests that foldable ops are constant-folded to enable legalization of ops
// that require compile time constant operand.
TEST(CompileSerializedMlirToXlaHloTest, CompileTimeConstantFoldedSuccess) {
  // "tf.Shape" can only be folded away after shape inference. tf.Reshape can
  // only be lowered when tf.Shape is folded into a constant.
  constexpr char mlir_module[] = R"(
    module attributes {tf.versions = {producer = 179 : i32}} {
      func @main(%arg0: tensor<10x19xf32>, %arg1: tensor<19x10xf32> {xla_hlo.is_same_data_across_replicas}) -> tensor<10x19xf32> {
        %0 = "tf.Shape"(%arg0) : (tensor<10x19xf32>) -> tensor<2xi64>
        %1 = "tf.Reshape"(%arg1, %0) : (tensor<19x10xf32>, tensor<2xi64>) -> tensor<10x19xf32>
        return %1 : tensor<10x19xf32>
      }
    }
  )";

  std::vector<TensorShape> arg_shapes{TensorShape({10, 19}),
                                      TensorShape({19, 10})};
  XlaCompiler::CompilationResult compilation_result;

  Status s = CompileSerializedMlirToXlaHlo(
      mlir_module, arg_shapes, "XLA_CPU_JIT",
      /*use_tuple_args=*/true, TestShapeRepresentation, &compilation_result);
  TF_ASSERT_OK(s);

  const xla::HloModuleConfig module_config(
      compilation_result.computation->GetProgramShape().ValueOrDie());
  auto status_or_hlo_module = xla::HloModule::CreateFromProto(
      compilation_result.computation->proto(), module_config);
  TF_ASSERT_OK(status_or_hlo_module.status());
  constexpr char expected_hlo_module_string[] = R"(HloModule main.6

ENTRY %main.6 (arg_tuple.1: (f32[10,19], f32[19,10])) -> (f32[10,19]) {
  %arg_tuple.1 = (f32[10,19]{1,0}, f32[19,10]{1,0}) parameter(0), parameter_replication={false,true}
  %get-tuple-element.2 = f32[10,19]{1,0} get-tuple-element((f32[10,19]{1,0}, f32[19,10]{1,0}) %arg_tuple.1), index=0
  %get-tuple-element.3 = f32[19,10]{1,0} get-tuple-element((f32[10,19]{1,0}, f32[19,10]{1,0}) %arg_tuple.1), index=1
  %reshape.4 = f32[10,19]{1,0} reshape(f32[19,10]{1,0} %get-tuple-element.3)
  ROOT %tuple.5 = (f32[10,19]{1,0}) tuple(f32[10,19]{1,0} %reshape.4)
}

)";
  EXPECT_EQ(expected_hlo_module_string,
            status_or_hlo_module.ValueOrDie()->ToString());
}

TEST(CompileSerializedMlirToXlaHloTest, ShapeInference) {
  constexpr char mlir_module[] = R"(
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
      mlir_module, arg_shapes, "XLA_CPU_JIT",
      /*use_tuple_args=*/true, TestShapeRepresentation, &compilation_result);
  TF_ASSERT_OK(s);

  const xla::HloModuleConfig module_config(
      compilation_result.computation->GetProgramShape().ValueOrDie());
  auto status_or_hlo_module = xla::HloModule::CreateFromProto(
      compilation_result.computation->proto(), module_config);
  TF_ASSERT_OK(status_or_hlo_module.status());

  constexpr char expected_signature[] =
      R"((arg_tuple.1: (f32[10,17], f32[17,19])) -> (f32[10,19]))";
  EXPECT_THAT(status_or_hlo_module.ValueOrDie()->ToString(),
              ::testing::HasSubstr(expected_signature));
}

TEST(CompileSerializedMlirToXlaHloTest, ShapeInferenceAfterLegalization) {
  constexpr char mlir_module[] = R"(
    module attributes {tf.versions = {producer = 179 : i32}} {
      func @main(%arg0: tensor<8x16x16x64xbf16>, %arg1: tensor<64xf32>) -> (tensor<8x16x16x64xbf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<*xf32>) {
        %0:6 = "tf.FusedBatchNormV3"(%arg0, %arg1, %arg1, %arg1, %arg1) {data_format = "NHWC", device = "", epsilon = 9.99999974E-5 : f32, exponential_avg_factor = 1.000000e+00 : f32, is_training = false} : (tensor<8x16x16x64xbf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<8x16x16x64xbf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<*xf32>)
        return %0#0, %0#1, %0#2, %0#3, %0#4, %0#5 : tensor<8x16x16x64xbf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<*xf32>
      }
    }
  )";

  std::vector<TensorShape> arg_shapes{TensorShape({8, 16, 16, 64}),
                                      TensorShape({64})};
  XlaCompiler::CompilationResult compilation_result;

  Status s = CompileSerializedMlirToXlaHlo(
      mlir_module, arg_shapes, "XLA_CPU_JIT",
      /*use_tuple_args=*/true, TestShapeRepresentation, &compilation_result);
  TF_ASSERT_OK(s);

  const xla::HloModuleConfig module_config(
      compilation_result.computation->GetProgramShape().ValueOrDie());
  auto status_or_hlo_module = xla::HloModule::CreateFromProto(
      compilation_result.computation->proto(), module_config);
  TF_ASSERT_OK(status_or_hlo_module.status());

  constexpr char expected_signature[] =
      R"(-> (bf16[8,16,16,64], f32[64], f32[64], f32[64], f32[64], f32[0]))";
  EXPECT_THAT(status_or_hlo_module.ValueOrDie()->ToString(),
              ::testing::HasSubstr(expected_signature));
}

TEST(CompileSerializedMlirToXlaHloTest, ConstantFoldHook) {
  constexpr char mlir_module[] = R"(
module attributes {tf.versions = {producer = 179 : i32}} {
  func @main() -> (tensor<0xi32>, tensor<0xi32>) {
    %0 = "tf.Const"() {value = dense<[]> : tensor<0xi32>} : () -> tensor<0xi32>
    %r0, %r1 = "tf.BroadcastGradientArgs"(%0, %0) {T = i32} : (tensor<0xi32>, tensor<0xi32>) -> (tensor<0xi32>, tensor<0xi32>)
    return %r0, %r1 : tensor<0xi32>, tensor<0xi32>
  }
}
)";

  std::vector<TensorShape> arg_shapes(2, TensorShape());
  XlaCompiler::CompilationResult compilation_result;

  Status s = CompileSerializedMlirToXlaHlo(
      mlir_module, arg_shapes, "XLA_CPU_JIT",
      /*use_tuple_args=*/true, TestShapeRepresentation, &compilation_result);
  TF_ASSERT_OK(s);

  const xla::HloModuleConfig module_config(
      compilation_result.computation->GetProgramShape().ValueOrDie());
  auto status_or_hlo_module = xla::HloModule::CreateFromProto(
      compilation_result.computation->proto(), module_config);
  TF_ASSERT_OK(status_or_hlo_module.status());
  constexpr char expected_hlo_module_string[] = R"(HloModule main.4

ENTRY %main.4 (arg_tuple.1: ()) -> (s32[0], s32[0]) {
  %arg_tuple.1 = () parameter(0)
  %constant.2 = s32[0]{0} constant({})
  ROOT %tuple.3 = (s32[0]{0}, s32[0]{0}) tuple(s32[0]{0} %constant.2, s32[0]{0} %constant.2)
}

)";
  EXPECT_EQ(expected_hlo_module_string,
            status_or_hlo_module.ValueOrDie()->ToString());
}

// The following xla::OpSharding protos are used:
//  Serialized string:
//   "\08\03\1A\02\01\02\22\02\00\01"
//  Proto debug string:
//   type: OTHER
//   tile_assignment_dimensions: 1
//   tile_assignment_dimensions: 2
//   tile_assignment_devices: 0
//   tile_assignment_devices: 1
//
//  Serialized string:
//   "\08\01\1A\01\01\22\01\00"
//  Proto debug string:
//   type: MAXIMAL
//   tile_assignment_dimensions: 1
//   tile_assignment_devices: 0
//
//  Serialized string:
//   ""
//  Proto debug string (empty but would equivalent to):
//   type: REPLICATED
TEST(CompileSerializedMlirToXlaHloTest, ArgumentSharding) {
  constexpr char mlir_module[] = R"(
module attributes {tf.versions = {producer = 179 : i32}} {
  func @main(%arg0: tensor<128x10xf32> {xla_hlo.sharding = "\08\03\1A\02\01\02\22\02\00\01"}, %arg1: tensor<10x1024xf32> {xla_hlo.sharding = "\08\01\1A\01\01\22\01\00"}, %arg2: tensor<128x1024xf32> {xla_hlo.sharding = ""}) {
    return
  }
}
)";

  std::vector<TensorShape> arg_shapes{TensorShape({128, 10}),
                                      TensorShape({10, 1024}),
                                      TensorShape({128, 1024})};
  XlaCompiler::CompilationResult compilation_result;

  Status s = CompileSerializedMlirToXlaHlo(
      mlir_module, arg_shapes, "XLA_CPU_JIT",
      /*use_tuple_args=*/true, TestShapeRepresentation, &compilation_result);
  TF_ASSERT_OK(s);

  const xla::HloModuleConfig module_config(
      compilation_result.computation->GetProgramShape().ValueOrDie());
  auto status_or_hlo_module = xla::HloModule::CreateFromProto(
      compilation_result.computation->proto(), module_config);
  TF_ASSERT_OK(status_or_hlo_module.status());
  constexpr char expected_hlo_module_string[] = R"(HloModule main.6

ENTRY %main.6 (arg_tuple.1: (f32[128,10], f32[10,1024], f32[128,1024])) -> () {
  %arg_tuple.1 = (f32[128,10]{1,0}, f32[10,1024]{1,0}, f32[128,1024]{1,0}) parameter(0), sharding={{devices=[1,2]0,1}, {maximal device=0}, {replicated}}
  %get-tuple-element.2 = f32[128,10]{1,0} get-tuple-element((f32[128,10]{1,0}, f32[10,1024]{1,0}, f32[128,1024]{1,0}) %arg_tuple.1), index=0
  %get-tuple-element.3 = f32[10,1024]{1,0} get-tuple-element((f32[128,10]{1,0}, f32[10,1024]{1,0}, f32[128,1024]{1,0}) %arg_tuple.1), index=1
  %get-tuple-element.4 = f32[128,1024]{1,0} get-tuple-element((f32[128,10]{1,0}, f32[10,1024]{1,0}, f32[128,1024]{1,0}) %arg_tuple.1), index=2
  ROOT %tuple.5 = () tuple()
}

)";
  EXPECT_EQ(expected_hlo_module_string,
            status_or_hlo_module.ValueOrDie()->ToString());
}

TEST(CompileSerializedMlirToXlaHloTest, BadArgumentSharding) {
  constexpr char mlir_module[] = R"(
module attributes {tf.versions = {producer = 179 : i32}} {
  func @main(%arg0: tensor<128x10xf32> {xla_hlo.sharding = "bad_sharding"}) {
    return
  }
}
)";

  std::vector<TensorShape> arg_shapes{TensorShape({128, 10})};
  XlaCompiler::CompilationResult compilation_result;

  Status s = CompileSerializedMlirToXlaHlo(
      mlir_module, arg_shapes, "XLA_CPU_JIT",
      /*use_tuple_args=*/true, TestShapeRepresentation, &compilation_result);
  ASSERT_FALSE(s.ok());
  EXPECT_EQ(s.error_message(),
            "failed to parse argument sharding 0 'bad_sharding'");
}

TEST(CompileSerializedMlirToXlaHloTest, ResultSharding) {
  constexpr char mlir_module[] = R"(
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 351 : i32}} {
  func @main(%arg0: tensor<128x10xf32>, %arg1: tensor<10x1024xf32>, %arg2: tensor<128x1024xf32>) -> (tensor<128x10xf32> {xla_hlo.sharding = "\08\03\1A\02\01\02\22\02\00\01"}, tensor<10x1024xf32> {xla_hlo.sharding = "\08\01\1A\01\01\22\01\00"}, tensor<128x1024xf32> {xla_hlo.sharding = ""}) {
    return %arg0, %arg1, %arg2 : tensor<128x10xf32>, tensor<10x1024xf32>, tensor<128x1024xf32>
  }
}
)";

  std::vector<TensorShape> arg_shapes{TensorShape({128, 10}),
                                      TensorShape({10, 1024}),
                                      TensorShape({128, 1024})};
  XlaCompiler::CompilationResult compilation_result;

  Status s = CompileSerializedMlirToXlaHlo(
      mlir_module, arg_shapes, "XLA_CPU_JIT",
      /*use_tuple_args=*/true, TestShapeRepresentation, &compilation_result);
  TF_ASSERT_OK(s);

  const xla::HloModuleConfig module_config(
      compilation_result.computation->GetProgramShape().ValueOrDie());
  auto status_or_hlo_module = xla::HloModule::CreateFromProto(
      compilation_result.computation->proto(), module_config);
  TF_ASSERT_OK(status_or_hlo_module.status());
  constexpr char expected_hlo_module_string[] = R"(HloModule main.9

ENTRY %main.9 (arg_tuple.1: (f32[128,10], f32[10,1024], f32[128,1024])) -> (f32[128,10], f32[10,1024], f32[128,1024]) {
  %arg_tuple.1 = (f32[128,10]{1,0}, f32[10,1024]{1,0}, f32[128,1024]{1,0}) parameter(0)
  %get-tuple-element.2 = f32[128,10]{1,0} get-tuple-element((f32[128,10]{1,0}, f32[10,1024]{1,0}, f32[128,1024]{1,0}) %arg_tuple.1), index=0
  %reshape.5 = f32[128,10]{1,0} reshape(f32[128,10]{1,0} %get-tuple-element.2)
  %get-tuple-element.3 = f32[10,1024]{1,0} get-tuple-element((f32[128,10]{1,0}, f32[10,1024]{1,0}, f32[128,1024]{1,0}) %arg_tuple.1), index=1
  %reshape.6 = f32[10,1024]{1,0} reshape(f32[10,1024]{1,0} %get-tuple-element.3)
  %get-tuple-element.4 = f32[128,1024]{1,0} get-tuple-element((f32[128,10]{1,0}, f32[10,1024]{1,0}, f32[128,1024]{1,0}) %arg_tuple.1), index=2
  %reshape.7 = f32[128,1024]{1,0} reshape(f32[128,1024]{1,0} %get-tuple-element.4)
  ROOT %tuple.8 = (f32[128,10]{1,0}, f32[10,1024]{1,0}, f32[128,1024]{1,0}) tuple(f32[128,10]{1,0} %reshape.5, f32[10,1024]{1,0} %reshape.6, f32[128,1024]{1,0} %reshape.7), sharding={{devices=[1,2]0,1}, {maximal device=0}, {replicated}}
}

)";
  EXPECT_EQ(expected_hlo_module_string,
            status_or_hlo_module.ValueOrDie()->ToString());
}

// Verify that conversion from Graph to MLIR and empty shape representation
// function is successful.
TEST(CompileGraphToXlaHlo, Basic) {
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), {});
  Graph graph(OpRegistry::Global());

  Tensor dummy_tensor(DT_FLOAT, TensorShape({1}));
  test::FillValues<float>(&dummy_tensor, {-1.0});

  Node* arg = test::graph::Arg(&graph, 0, DT_FLOAT);
  test::graph::Retval(&graph, 0, arg);

  XlaCompiler::CompilationResult result;
  XlaCompiler::Argument compiler_arg;
  compiler_arg.kind = XlaCompiler::Argument::kParameter;
  compiler_arg.shape = TensorShape();

  TF_ASSERT_OK(
      CompileGraphToXlaHlo(graph, /*args=*/{compiler_arg}, "XLA_CPU_JIT",
                           /*use_tuple_args=*/false, flib_def, GraphDebugInfo(),
                           /*shape_representation_fn=*/nullptr, &result));

  const xla::HloModuleConfig module_config(
      result.computation->GetProgramShape().ValueOrDie());
  auto status_or_hlo_module = xla::HloModule::CreateFromProto(
      result.computation->proto(), module_config);
  ASSERT_TRUE(status_or_hlo_module.ok());

  constexpr char expected_hlo_module_string[] = R"(HloModule main.3

ENTRY %main.3 (Arg_0.1: f32[]) -> (f32[]) {
  %Arg_0.1 = f32[] parameter(0)
  ROOT %tuple.2 = (f32[]) tuple(f32[] %Arg_0.1)
}

)";

  EXPECT_EQ(expected_hlo_module_string,
            status_or_hlo_module.ValueOrDie()->ToString());
}

}  // namespace
}  // namespace tensorflow
