/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/tf2xla.h"

#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/tf2xla/tf2xla.pb.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/literal_util.h"
#include "xla/pjrt/cpu/cpu_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_client_options.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/platform/test.h"
#include "tsl/platform/tensor_float_32_utils.h"

namespace tensorflow {
namespace {

class ConvertGraphDefToXlaWithTF32Disabled : public ::testing::Test {
 public:
  ConvertGraphDefToXlaWithTF32Disabled() {
    tsl::enable_tensor_float_32_execution(false);
  }
  ~ConvertGraphDefToXlaWithTF32Disabled() override {
    tsl::enable_tensor_float_32_execution(true);
  }
};

AttrValue TypeAttrValue(DataType type) {
  AttrValue attr_value;
  SetAttrValue(type, &attr_value);
  return attr_value;
}

AttrValue StringAttrValue(absl::string_view str) {
  AttrValue attr_value;
  SetAttrValue(str, &attr_value);
  return attr_value;
}

AttrValue IntAttrValue(int i) {
  AttrValue attr_value;
  SetAttrValue(i, &attr_value);
  return attr_value;
}

AttrValue IntVectorAttrValue(const std::vector<int>& ints) {
  AttrValue attr_value;
  SetAttrValue(ints, &attr_value);
  return attr_value;
}

TensorShapeProto TensorShape(const std::vector<int>& dims) {
  TensorShapeProto shape;
  for (int i = 0; i < dims.size(); ++i) {
    shape.add_dim();
    shape.mutable_dim(i)->set_size(dims[i]);
  }
  return shape;
}

GraphDef SumGraph() {
  GraphDef graph_def;
  NodeDef* x = graph_def.add_node();
  x->set_name("x");
  x->set_op("Placeholder");
  (*x->mutable_attr())["dtype"] = TypeAttrValue(DT_INT32);
  NodeDef* y = graph_def.add_node();
  y->set_name("y");
  y->set_op("Placeholder");
  (*y->mutable_attr())["dtype"] = TypeAttrValue(DT_INT32);
  NodeDef* sum = graph_def.add_node();
  sum->set_name("sum");
  sum->set_op("Add");
  sum->add_input("x");
  sum->add_input("y");
  (*sum->mutable_attr())["T"] = TypeAttrValue(DT_INT32);
  return graph_def;
}

tf2xla::Config SumConfig() {
  tf2xla::Config config;
  config.add_feed()->mutable_id()->set_node_name("x");
  config.add_feed()->mutable_id()->set_node_name("y");
  config.add_fetch()->mutable_id()->set_node_name("sum");
  return config;
}

TEST(ConvertGraphDefToXla, Sum) {
  GraphDef graph_def = SumGraph();
  tf2xla::Config config = SumConfig();

  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          xla::GetPjRtCpuClient(xla::CpuClientOptions()));
  TF_ASSERT_OK_AND_ASSIGN(auto compiler,
                          xla::GetDefaultPjRtCompiler(xla::CpuName()));

  xla::XlaComputation computation;
  TF_EXPECT_OK(ConvertGraphDefToXla(graph_def, config, compiler, &computation,
                                    client.get()));

  // Set up arguments.
  auto x_literal = xla::LiteralUtil::CreateR0<int32_t>(10);
  auto y_literal = xla::LiteralUtil::CreateR0<int32_t>(32);
  TF_ASSERT_OK_AND_ASSIGN(
      auto x_buf,
      client->BufferFromHostLiteral(x_literal, client->memory_spaces()[0]));
  TF_ASSERT_OK_AND_ASSIGN(
      auto y_buf,
      client->BufferFromHostLiteral(y_literal, client->memory_spaces()[0]));

  // Execute and check result.
  TF_ASSERT_OK_AND_ASSIGN(
      auto loaded_exec,
      client->CompileAndLoad(computation, xla::CompileOptions{}));
  xla::ExecuteOptions execute_options;
  TF_ASSERT_OK_AND_ASSIGN(
      auto results,
      loaded_exec->Execute({{x_buf.get(), y_buf.get()}}, execute_options));
  TF_ASSERT_OK_AND_ASSIGN(auto result, results[0][0]->ToLiteralSync());
  EXPECT_EQ(42, result->Get<int32_t>({}));

  config.mutable_feed(0)->mutable_id()->set_output_index(
      123); /* invalid output_index */
  EXPECT_TRUE(absl::IsInvalidArgument(ConvertGraphDefToXla(
      graph_def, config, compiler, &computation, client.get())));
}

GraphDef EinsumGraph(DataType dtype = DT_FLOAT) {
  GraphDef graph_def;
  NodeDef* x = graph_def.add_node();
  x->set_name("x");
  x->set_op("Placeholder");
  (*x->mutable_attr())["dtype"] = TypeAttrValue(dtype);
  NodeDef* y = graph_def.add_node();
  y->set_name("y");
  y->set_op("Placeholder");
  (*y->mutable_attr())["dtype"] = TypeAttrValue(dtype);
  NodeDef* einsum = graph_def.add_node();
  einsum->set_name("einsum");
  einsum->set_op("Einsum");
  einsum->add_input("x");
  einsum->add_input("y");
  (*einsum->mutable_attr())["equation"] = StringAttrValue("ij,jk->ik");
  (*einsum->mutable_attr())["T"] = TypeAttrValue(dtype);
  (*einsum->mutable_attr())["N"] = IntAttrValue(2);
  return graph_def;
}

tf2xla::Config EinsumConfig() {
  tf2xla::Config config;

  tf2xla::Feed* x_feed = config.add_feed();
  x_feed->mutable_id()->set_node_name("x");
  *x_feed->mutable_shape() = TensorShape({2, 2});

  tf2xla::Feed* y_feed = config.add_feed();
  y_feed->mutable_id()->set_node_name("y");
  *y_feed->mutable_shape() = TensorShape({2, 2});

  config.add_fetch()->mutable_id()->set_node_name("einsum");
  return config;
}

TEST(ConvertGraphDefToXla, EinsumIsConvertedToDotWithDefaultPrecision) {
  GraphDef graph_def = EinsumGraph();
  tf2xla::Config config = EinsumConfig();

  TF_ASSERT_OK_AND_ASSIGN(auto compiler,
                          xla::GetDefaultPjRtCompiler(xla::CpuName()));
  xla::XlaComputation computation;
  TF_EXPECT_OK(ConvertGraphDefToXla(graph_def, config, compiler, &computation));

  int num_dots = 0;
  const xla::HloModuleProto& module_proto = computation.proto();
  for (const xla::HloComputationProto& computation_proto :
       module_proto.computations()) {
    for (const xla::HloInstructionProto& instruction_proto :
         computation_proto.instructions()) {
      if (instruction_proto.opcode() == "dot") {
        num_dots++;
        ASSERT_EQ(instruction_proto.precision_config().operand_precision_size(),
                  2);
        EXPECT_EQ(instruction_proto.precision_config().operand_precision(0),
                  xla::PrecisionConfig::DEFAULT);
        EXPECT_EQ(instruction_proto.precision_config().operand_precision(1),
                  xla::PrecisionConfig::DEFAULT);
      }
    }
  }
  EXPECT_EQ(num_dots, 1);
}

TEST_F(ConvertGraphDefToXlaWithTF32Disabled,
       EinsumIsConvertedToDotWithHighestPrecision) {
  GraphDef graph_def = EinsumGraph();
  tf2xla::Config config = EinsumConfig();

  TF_ASSERT_OK_AND_ASSIGN(auto compiler,
                          xla::GetDefaultPjRtCompiler(xla::CpuName()));
  xla::XlaComputation computation;
  TF_EXPECT_OK(ConvertGraphDefToXla(graph_def, config, compiler, &computation));

  int num_dots = 0;
  const xla::HloModuleProto& module_proto = computation.proto();
  for (const xla::HloComputationProto& computation_proto :
       module_proto.computations()) {
    for (const xla::HloInstructionProto& instruction_proto :
         computation_proto.instructions()) {
      if (instruction_proto.opcode() == "dot") {
        num_dots++;
        ASSERT_EQ(instruction_proto.precision_config().operand_precision_size(),
                  2);
        EXPECT_EQ(instruction_proto.precision_config().operand_precision(0),
                  xla::PrecisionConfig::HIGHEST);
        EXPECT_EQ(instruction_proto.precision_config().operand_precision(1),
                  xla::PrecisionConfig::HIGHEST);
      }
    }
  }
  EXPECT_EQ(num_dots, 1);
}

TEST_F(ConvertGraphDefToXlaWithTF32Disabled,
       EinsumIsConvertedToDotWithDefaultPrecisionIfNotF32) {
  GraphDef graph_def = EinsumGraph(DT_BFLOAT16);
  tf2xla::Config config = EinsumConfig();

  TF_ASSERT_OK_AND_ASSIGN(auto compiler,
                          xla::GetDefaultPjRtCompiler(xla::CpuName()));
  xla::XlaComputation computation;
  TF_EXPECT_OK(ConvertGraphDefToXla(graph_def, config, compiler, &computation));

  int num_dots = 0;
  const xla::HloModuleProto& module_proto = computation.proto();
  for (const xla::HloComputationProto& computation_proto :
       module_proto.computations()) {
    for (const xla::HloInstructionProto& instruction_proto :
         computation_proto.instructions()) {
      if (instruction_proto.opcode() == "dot") {
        num_dots++;
        ASSERT_EQ(instruction_proto.precision_config().operand_precision_size(),
                  2);
        EXPECT_EQ(instruction_proto.precision_config().operand_precision(0),
                  xla::PrecisionConfig::DEFAULT);
        EXPECT_EQ(instruction_proto.precision_config().operand_precision(1),
                  xla::PrecisionConfig::DEFAULT);
      }
    }
  }
  EXPECT_EQ(num_dots, 1);
}

GraphDef Conv2DGraph() {
  GraphDef graph_def;
  NodeDef* x = graph_def.add_node();
  x->set_name("x");
  x->set_op("Placeholder");
  (*x->mutable_attr())["dtype"] = TypeAttrValue(DT_FLOAT);
  NodeDef* y = graph_def.add_node();
  y->set_name("y");
  y->set_op("Placeholder");
  (*y->mutable_attr())["dtype"] = TypeAttrValue(DT_FLOAT);
  NodeDef* einsum = graph_def.add_node();
  einsum->set_name("conv2d");
  einsum->set_op("Conv2D");
  einsum->add_input("x");
  einsum->add_input("y");
  (*einsum->mutable_attr())["T"] = TypeAttrValue(DT_FLOAT);
  (*einsum->mutable_attr())["padding"] = StringAttrValue("VALID");
  (*einsum->mutable_attr())["strides"] = IntVectorAttrValue({1, 1, 1, 1});
  return graph_def;
}

tf2xla::Config Conv2DConfig() {
  tf2xla::Config config;
  tf2xla::Feed* x_feed = config.add_feed();
  x_feed->mutable_id()->set_node_name("x");
  *x_feed->mutable_shape() = TensorShape({1, 1, 2, 2});

  tf2xla::Feed* y_feed = config.add_feed();
  y_feed->mutable_id()->set_node_name("y");
  *y_feed->mutable_shape() = TensorShape({1, 1, 2, 2});
  config.add_fetch()->mutable_id()->set_node_name("conv2d");
  return config;
}

TEST(ConvertGraphDefToXla, Conv2DIsConvertedToConvolutionWithDefaultPrecision) {
  GraphDef graph_def = Conv2DGraph();
  tf2xla::Config config = Conv2DConfig();

  TF_ASSERT_OK_AND_ASSIGN(auto compiler,
                          xla::GetDefaultPjRtCompiler(xla::CpuName()));
  xla::XlaComputation computation;
  TF_EXPECT_OK(ConvertGraphDefToXla(graph_def, config, compiler, &computation));

  int num_convolutions = 0;
  const xla::HloModuleProto& module_proto = computation.proto();
  for (const xla::HloComputationProto& computation_proto :
       module_proto.computations()) {
    for (const xla::HloInstructionProto& instruction_proto :
         computation_proto.instructions()) {
      if (instruction_proto.opcode() == "convolution") {
        num_convolutions++;
        ASSERT_EQ(instruction_proto.precision_config().operand_precision_size(),
                  2);
        EXPECT_EQ(instruction_proto.precision_config().operand_precision(0),
                  xla::PrecisionConfig::DEFAULT);
        EXPECT_EQ(instruction_proto.precision_config().operand_precision(1),
                  xla::PrecisionConfig::DEFAULT);
      }
    }
  }
  EXPECT_EQ(num_convolutions, 1);
}

TEST_F(ConvertGraphDefToXlaWithTF32Disabled,
       Conv2DIsConvertedToConvolutionWithHighestPrecision) {
  GraphDef graph_def = Conv2DGraph();
  tf2xla::Config config = Conv2DConfig();

  TF_ASSERT_OK_AND_ASSIGN(auto compiler,
                          xla::GetDefaultPjRtCompiler(xla::CpuName()));
  xla::XlaComputation computation;
  TF_EXPECT_OK(ConvertGraphDefToXla(graph_def, config, compiler, &computation));

  int num_convolutions = 0;
  const xla::HloModuleProto& module_proto = computation.proto();
  for (const xla::HloComputationProto& computation_proto :
       module_proto.computations()) {
    for (const xla::HloInstructionProto& instruction_proto :
         computation_proto.instructions()) {
      if (instruction_proto.opcode() == "convolution") {
        num_convolutions++;
        ASSERT_EQ(instruction_proto.precision_config().operand_precision_size(),
                  2);
        EXPECT_EQ(instruction_proto.precision_config().operand_precision(0),
                  xla::PrecisionConfig::HIGHEST);
        EXPECT_EQ(instruction_proto.precision_config().operand_precision(1),
                  xla::PrecisionConfig::HIGHEST);
      }
    }
  }
  EXPECT_EQ(num_convolutions, 1);
}

TEST(ConvertGraphDefToXla, SumWithUnusedArgument) {
  GraphDef graph_def = SumGraph();
  tf2xla::Config config = SumConfig();
  NodeDef* unused = graph_def.add_node();
  unused->set_name("unused");
  unused->set_op("Placeholder");
  (*unused->mutable_attr())["dtype"] = TypeAttrValue(DT_INT32);
  config.add_feed()->mutable_id()->set_node_name("unused");

  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          xla::GetPjRtCpuClient(xla::CpuClientOptions()));
  TF_ASSERT_OK_AND_ASSIGN(auto compiler,
                          xla::GetDefaultPjRtCompiler(xla::CpuName()));

  xla::XlaComputation computation;
  TF_EXPECT_OK(ConvertGraphDefToXla(graph_def, config, compiler, &computation,
                                    client.get()));

  // Set up arguments.
  auto x_literal = xla::LiteralUtil::CreateR0<int32_t>(10);
  auto y_literal = xla::LiteralUtil::CreateR0<int32_t>(32);
  TF_ASSERT_OK_AND_ASSIGN(
      auto x_buf,
      client->BufferFromHostLiteral(x_literal, client->memory_spaces()[0]));
  TF_ASSERT_OK_AND_ASSIGN(
      auto y_buf,
      client->BufferFromHostLiteral(y_literal, client->memory_spaces()[0]));
  TF_ASSERT_OK_AND_ASSIGN(
      auto unused_buf,
      client->BufferFromHostLiteral(y_literal, client->memory_spaces()[0]));

  // Execute and check result.
  TF_ASSERT_OK_AND_ASSIGN(
      auto loaded_exec,
      client->CompileAndLoad(computation, xla::CompileOptions{}));
  xla::ExecuteOptions execute_options;
  TF_ASSERT_OK_AND_ASSIGN(
      auto results,
      loaded_exec->Execute({{x_buf.get(), y_buf.get(), unused_buf.get()}},
                           execute_options));
  TF_ASSERT_OK_AND_ASSIGN(auto result, results[0][0]->ToLiteralSync());
  EXPECT_EQ(42, result->Get<int32_t>({}));
}

}  // namespace
}  // namespace tensorflow
