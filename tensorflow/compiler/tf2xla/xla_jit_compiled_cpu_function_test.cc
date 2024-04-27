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

#include "tensorflow/compiler/tf2xla/xla_jit_compiled_cpu_function.h"

#include <memory>
#include <string>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/tf2xla/tf2xla.pb.h"
#include "xla/client/local_client.h"
#include "xla/service/compiler.h"
#include "xla/service/platform_util.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/test.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

using ::testing::HasSubstr;

PLATFORM_DEFINE_ID(kFakePlatformId);

AttrValue TypeAttrValue(DataType type) {
  AttrValue attr_value;
  SetAttrValue(type, &attr_value);
  return attr_value;
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
  tf2xla::Feed* x = config.add_feed();
  x->mutable_id()->set_node_name("x");
  x->set_name("x_name");
  tf2xla::Feed* y = config.add_feed();
  y->mutable_id()->set_node_name("y");
  y->set_name("y_name");
  tf2xla::Fetch* sum = config.add_fetch();
  sum->mutable_id()->set_node_name("sum");
  sum->set_name("sum_name");
  return config;
}

GraphDef SumGraphVariable() {
  constexpr char text_proto[] = R"pb(
    node {
      name: "x"
      op: "VarHandleOp"
      attr {
        key: "dtype"
        value { type: DT_INT32 }
      }
      attr {
        key: "shared_name"
        value { s: "myvar" }
      }
      attr {
        key: "shape"
        value { shape { dim { size: 1 } } }
      }
    }
    node {
      name: "read"
      op: "ReadVariableOp"
      input: "x"
      attr {
        key: "dtype"
        value { type: DT_INT32 }
      }
    }
    node {
      name: "y"
      op: "Placeholder"
      attr {
        key: "dtype"
        value { type: DT_INT32 }
      }
    }
    node {
      name: "sum"
      op: "Add"
      input: "read"
      input: "y"
      attr {
        key: "T"
        value { type: DT_INT32 }
      }
    }
    node {
      name: "assign"
      op: "AssignVariableOp"
      input: "x"
      input: "sum"
      attr {
        key: "dtype"
        value { type: DT_INT32 }
      }
    }
    # We use this identity op to make sure assign doesn't get pruned away.
    node {
      name: "out"
      op: "Identity"
      input: "y"
      input: "^assign"
      attr {
        key: "T"
        value { type: DT_INT32 }
      }
    })pb";
  GraphDef graph;
  CHECK(protobuf::TextFormat::ParseFromString(text_proto, &graph));
  return graph;
}

tf2xla::Config SumConfigVariable() {
  constexpr char text_proto[] = R"pb(feed { id { node_name: "y" } }
                                     variable {
                                       node_name: "myvar"
                                       shape { dim { size: 1 } }
                                       type: DT_INT32
                                     }
                                     fetch { id { node_name: "out" } })pb";
  tf2xla::Config config;
  CHECK(protobuf::TextFormat::ParseFromString(text_proto, &config));
  return config;
}

TEST(XlaJitCompiledCpuFunction, Sum) {
  GraphDef graph_def = SumGraph();
  tf2xla::Config config = SumConfig();

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<XlaJitCompiledCpuFunction> jit,
      XlaJitCompiledCpuFunction::Compile(graph_def, config,
                                         xla::ExecutableBuildOptions()));
  XlaCompiledCpuFunction function(jit->StaticData());
  ASSERT_EQ(function.num_args(), 2);
  ASSERT_EQ(function.num_results(), 1);

  // Run the function and check results.
  *static_cast<int32*>(function.arg_data(0)) = 10;
  *static_cast<int32*>(function.arg_data(1)) = 32;
  EXPECT_TRUE(function.Run());
  EXPECT_EQ(function.error_msg(), "");
  EXPECT_EQ(*static_cast<int32*>(function.result_data(0)), 42);

  // Run the function again.
  *static_cast<int32*>(function.arg_data(0)) = 100;
  *static_cast<int32*>(function.arg_data(1)) = 320;
  EXPECT_TRUE(function.Run());
  EXPECT_EQ(function.error_msg(), "");
  EXPECT_EQ(*static_cast<int32*>(function.result_data(0)), 420);

  // Check name to index lookups.
  EXPECT_TRUE(function.HasNameIndices());

  EXPECT_EQ(function.LookupArgIndex("x_name"), 0);
  EXPECT_EQ(function.LookupArgIndex("y_name"), 1);
  EXPECT_EQ(function.LookupArgIndex(""), -1);
  EXPECT_EQ(function.LookupArgIndex("x"), -1);
  EXPECT_EQ(function.LookupArgIndex("y"), -1);
  EXPECT_EQ(function.LookupArgIndex("sum"), -1);
  EXPECT_EQ(function.LookupArgIndex("sum_name"), -1);

  EXPECT_EQ(function.LookupResultIndex("sum_name"), 0);
  EXPECT_EQ(function.LookupResultIndex(""), -1);
  EXPECT_EQ(function.LookupResultIndex("x"), -1);
  EXPECT_EQ(function.LookupResultIndex("y"), -1);
  EXPECT_EQ(function.LookupResultIndex("sum"), -1);
  EXPECT_EQ(function.LookupResultIndex("x_name"), -1);
  EXPECT_EQ(function.LookupResultIndex("y_name"), -1);

  EXPECT_EQ(0, function.num_variables());
  EXPECT_EQ(function.LookupVariableIndex("x"), -1);

  // Expect that name and index lookups match.
  for (int i = 0; i < function.num_args(); ++i) {
    const char* name = function.GetArgName(i);
    ASSERT_NE(name, nullptr);
    const int roundtrip_i = function.LookupArgIndex(name);
    EXPECT_EQ(roundtrip_i, i) << " name= " << name;
  }
  for (int i = 0; i < function.num_results(); ++i) {
    const char* name = function.GetResultName(i);
    ASSERT_NE(name, nullptr);
    const int roundtrip_i = function.LookupResultIndex(name);
    EXPECT_EQ(roundtrip_i, i) << " name= " << name;
  }
  // Expect correct handling of invalid indices.
  EXPECT_EQ(function.GetArgName(-1), nullptr);
  EXPECT_EQ(function.GetArgName(function.num_args()), nullptr);
  EXPECT_EQ(function.GetResultName(-1), nullptr);
  EXPECT_EQ(function.GetResultName(function.num_results()), nullptr);
  EXPECT_EQ(function.GetVariableName(0), nullptr);

  // Check program shape.
  using xla::ShapeUtil;
  const xla::Shape s32 = ShapeUtil::MakeShape(xla::S32, {});
  ASSERT_TRUE(function.ProgramShape() != nullptr);
  const xla::ProgramShape program_shape(*function.ProgramShape());
  ASSERT_EQ(program_shape.parameters_size(), 2);
  EXPECT_TRUE(ShapeUtil::Compatible(program_shape.parameters(0), s32));
  EXPECT_TRUE(ShapeUtil::Compatible(program_shape.parameters(1), s32));

  const xla::Shape& result = program_shape.result();
  ASSERT_EQ(result.element_type(), xla::TUPLE);
  ASSERT_EQ(ShapeUtil::TupleElementCount(result), 1);
  const xla::Shape& result0 = ShapeUtil::GetTupleElementShape(result, 0);
  EXPECT_TRUE(ShapeUtil::Compatible(result0, s32));
}

TEST(XlaJitCompiledCpuFunction, SumVariable) {
  GraphDef graph_def = SumGraphVariable();
  tf2xla::Config config = SumConfigVariable();

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<XlaJitCompiledCpuFunction> jit,
      XlaJitCompiledCpuFunction::Compile(graph_def, config,
                                         xla::ExecutableBuildOptions()));
  XlaCompiledCpuFunction function(jit->StaticData());
  ASSERT_EQ(function.num_args(), 2);
  ASSERT_EQ(function.num_results(), 2);

  // Run the function and check results.
  *static_cast<int32*>(function.arg_data(0)) = 10;
  *static_cast<int32*>(function.arg_data(1)) = 32;
  EXPECT_TRUE(function.Run());
  EXPECT_EQ(function.error_msg(), "");
  EXPECT_EQ(*static_cast<int32*>(function.result_data(0)), 10);
  EXPECT_EQ(*static_cast<int32*>(function.result_data(1)), 42);

  // Run the function again.
  *static_cast<int32*>(function.arg_data(0)) = 100;
  *static_cast<int32*>(function.arg_data(1)) = 320;
  EXPECT_TRUE(function.Run());
  EXPECT_EQ(function.error_msg(), "");
  EXPECT_EQ(*static_cast<int32*>(function.result_data(0)), 100);
  EXPECT_EQ(*static_cast<int32*>(function.result_data(1)), 420);

  // Check name to index lookups.
  EXPECT_TRUE(function.HasNameIndices());

  EXPECT_EQ(2, function.num_args());

  EXPECT_EQ(1, function.num_variables());
  EXPECT_EQ(function.LookupVariableIndex("myvar"), 1);

  const char* name = function.GetVariableName(0);
  EXPECT_EQ(std::string(name), "myvar");
  EXPECT_EQ(function.GetVariableName(1), nullptr);
  EXPECT_EQ(function.GetVariableName(-1), nullptr);

  // Check program shape.
  using xla::ShapeUtil;
  const xla::Shape s32 = ShapeUtil::MakeShape(xla::S32, {});
  const xla::Shape s32_1 = ShapeUtil::MakeShape(xla::S32, {1});
  ASSERT_TRUE(function.ProgramShape() != nullptr);
  const xla::ProgramShape program_shape(*function.ProgramShape());
  ASSERT_EQ(program_shape.parameters_size(), 2);
  EXPECT_TRUE(ShapeUtil::Compatible(program_shape.parameters(0), s32));
  EXPECT_TRUE(ShapeUtil::Compatible(program_shape.parameters(1), s32_1));

  const xla::Shape& result = program_shape.result();
  ASSERT_EQ(result.element_type(), xla::TUPLE);
  ASSERT_EQ(ShapeUtil::TupleElementCount(result), 2);
  const xla::Shape& result0 = ShapeUtil::GetTupleElementShape(result, 0);
  EXPECT_TRUE(ShapeUtil::Compatible(result0, s32));
}

TEST(XlaJitCompiledCpuFunction, CanCompileWithAdditionalPlatform) {
  class FakePlatform : public se::Platform {
   public:
    FakePlatform() : name_("FakePlatform") {}
    ~FakePlatform() override {}

    se::Platform::Id id() const override { return kFakePlatformId; }

    int VisibleDeviceCount() const override { return 0; }

    const string& Name() const override { return name_; }

    absl::StatusOr<std::unique_ptr<se::DeviceDescription>> DescriptionForDevice(
        int ordinal) const override {
      return std::unique_ptr<se::DeviceDescription>(nullptr);
    }

    absl::StatusOr<se::StreamExecutor*> ExecutorForDevice(
        int ordinal) override {
      return nullptr;
    }

    absl::StatusOr<se::StreamExecutor*> GetExecutor(
        const se::StreamExecutorConfig& config) override {
      return nullptr;
    }

    absl::StatusOr<std::unique_ptr<se::StreamExecutor>> GetUncachedExecutor(
        const se::StreamExecutorConfig& config) override {
      return std::unique_ptr<se::StreamExecutor>(nullptr);
    }

   private:
    string name_;
  };

  TF_EXPECT_OK(
      se::PlatformManager::RegisterPlatform(std::make_unique<FakePlatform>()));
  xla::Compiler::RegisterCompilerFactory(kFakePlatformId, []() {
    return std::unique_ptr<xla::Compiler>(nullptr);
  });

  EXPECT_THAT(xla::PlatformUtil::GetDefaultPlatform().status().message(),
              HasSubstr("FakePlatform"));

  GraphDef graph_def = SumGraph();
  tf2xla::Config config = SumConfig();
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<XlaJitCompiledCpuFunction> jit,
      XlaJitCompiledCpuFunction::Compile(graph_def, config,
                                         xla::ExecutableBuildOptions()));
}

}  // namespace
}  // namespace tensorflow
