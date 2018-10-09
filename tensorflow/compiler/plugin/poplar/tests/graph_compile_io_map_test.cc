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

#include "tensorflow/compiler/plugin/poplar/driver/compiler.h"
#include "tensorflow/compiler/plugin/poplar/driver/executable.h"
#include "tensorflow/compiler/plugin/poplar/driver/input_output_aliasing_map.h"
#include "tensorflow/compiler/plugin/poplar/driver/platform.h"

#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace poplarplugin {

/* The compilation process produces an executable which contains a map of
 * which input tensors are also outputs.  This test checks that this map is
 * correct */

class GraphCompileIoMapTest : public HloTestBase {
 public:
 public:
  explicit GraphCompileIoMapTest(se::Platform* platform = nullptr)
      : HloTestBase() {}
  const InputOutputAliasingMap& GetInputOutputAliasingMap(PoplarExecutable* e) {
    return e->GetInputOutputAliasingMap();
  }

  static std::unique_ptr<HloModule> CreateNewModuleWithConfig(
      const HloModuleConfig& config, const string& name = TestName()) {
    return absl::make_unique<HloModule>(name, config);
  }
};

namespace {

TEST_F(GraphCompileIoMapTest, NoShared) {
  Shape image_shape = ShapeUtil::MakeShape(F32, {1, 4, 4, 2});

  auto builder = HloComputation::Builder(TestName());
  auto in1 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, image_shape, "input1"));
  auto in2 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, image_shape, "input2"));
  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(image_shape, HloOpcode::kAdd, in1, in2));
  builder.AddInstruction(HloInstruction::CreateTuple({add}));

  auto computation = builder.Build();

  auto hlo_module = CreateNewModule();
  hlo_module->AddEntryComputation(std::move(computation));

  auto* platform =
      se::MultiPlatformManager::PlatformWithName("Poplar").ConsumeValueOrDie();
  auto* stream_executor = platform->ExecutorForDevice(0).ConsumeValueOrDie();

  tensorflow::IPUOptions opts;
  auto* p = static_cast<PoplarPlatform*>(platform);
  EXPECT_TRUE(p->ConfigurePoplarDevice(0, opts).ok());

  PoplarCompiler compiler;

  hlo_module =
      compiler.RunHloPasses(std::move(hlo_module), stream_executor, nullptr)
          .ConsumeValueOrDie();

  std::unique_ptr<Executable> executable =
      compiler.RunBackend(std::move(hlo_module), stream_executor, nullptr)
          .ConsumeValueOrDie();

  PoplarExecutable* e = static_cast<PoplarExecutable*>(executable.get());
  const auto& input_output_aliasing_map = GetInputOutputAliasingMap(e);
  const auto& input_infos = input_output_aliasing_map.GetEntryInputInfos();
  const auto& output_infos = input_output_aliasing_map.GetEntryOutputInfos();

  EXPECT_EQ(2, input_infos.size());
  EXPECT_TRUE(input_infos[0].IsStreaming());
  EXPECT_TRUE(input_infos[1].IsStreaming());
  EXPECT_EQ(1, output_infos.size());
  EXPECT_TRUE(output_infos[0].IsStreaming());
}

TEST_F(GraphCompileIoMapTest, Input1Shared) {
  Shape image_shape = ShapeUtil::MakeShape(F32, {1, 4, 4, 2});

  auto builder = HloComputation::Builder(TestName());
  auto in1 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, image_shape, "input1"));
  auto in2 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, image_shape, "input2"));
  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(image_shape, HloOpcode::kAdd, in1, in2));
  builder.AddInstruction(HloInstruction::CreateTuple({add}));

  OpMetadata metadata1;
  metadata1.set_op_name("grad%1");
  metadata1.set_op_type("ResourceApplyGradientDescent");
  add->set_metadata(metadata1);

  auto computation = builder.Build();

  auto config = GetModuleConfigForTest();
  config.set_resource_input_count(1);
  config.set_resource_update_to_input_index({1});
  auto hlo_module = CreateNewModuleWithConfig(config);
  hlo_module->AddEntryComputation(std::move(computation));

  auto* platform =
      se::MultiPlatformManager::PlatformWithName("Poplar").ConsumeValueOrDie();
  auto* stream_executor = platform->ExecutorForDevice(0).ConsumeValueOrDie();

  tensorflow::IPUOptions opts;
  auto* p = static_cast<PoplarPlatform*>(platform);
  EXPECT_TRUE(p->ConfigurePoplarDevice(0, opts).ok());

  PoplarCompiler compiler;

  hlo_module =
      compiler.RunHloPasses(std::move(hlo_module), stream_executor, nullptr)
          .ConsumeValueOrDie();

  std::unique_ptr<Executable> executable =
      compiler.RunBackend(std::move(hlo_module), stream_executor, nullptr)
          .ConsumeValueOrDie();

  PoplarExecutable* e = static_cast<PoplarExecutable*>(executable.get());
  const auto& input_output_aliasing_map = GetInputOutputAliasingMap(e);
  const auto& input_infos = input_output_aliasing_map.GetEntryInputInfos();
  const auto& output_infos = input_output_aliasing_map.GetEntryOutputInfos();

  EXPECT_EQ(2, input_infos.size());
  EXPECT_TRUE(input_infos[0].IsStreaming());
  EXPECT_TRUE(input_infos[1].IsResource());
  EXPECT_EQ(0, input_infos[1].GetOutputIndex());
  EXPECT_EQ(1, output_infos.size());
  EXPECT_TRUE(output_infos[0].IsResourceModified());
  EXPECT_EQ(1, output_infos[0].GetInputIndex());
}

TEST_F(GraphCompileIoMapTest, TupleInTuple) {
  Shape image_shape = ShapeUtil::MakeShape(S32, {2, 2});

  auto builder = HloComputation::Builder(TestName());
  auto in1 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, image_shape, "input1"));
  auto in2 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, image_shape, "input2"));
  auto in3 = builder.AddInstruction(
      HloInstruction::CreateParameter(2, image_shape, "input3"));
  auto add1 = builder.AddInstruction(
      HloInstruction::CreateBinary(image_shape, HloOpcode::kAdd, in1, in2));
  auto add2 = builder.AddInstruction(
      HloInstruction::CreateBinary(image_shape, HloOpcode::kAdd, in2, in3));
  auto tup1 = builder.AddInstruction(HloInstruction::CreateTuple({add1, add2}));
  auto tup2 = builder.AddInstruction(HloInstruction::CreateTuple({add2, in3}));
  builder.AddInstruction(HloInstruction::CreateTuple({tup1, tup2}));

  OpMetadata metadata1;
  metadata1.set_op_name("grad%1");
  metadata1.set_op_type("ResourceApplyGradientDescent");
  add1->set_metadata(metadata1);

  OpMetadata metadata2;
  metadata2.set_op_name("grad%2");
  metadata2.set_op_type("ResourceApplyGradientDescent");
  add2->set_metadata(metadata2);

  auto computation = builder.Build();

  auto hlo_module = CreateNewModule();
  hlo_module->AddEntryComputation(std::move(computation));

  auto* platform =
      se::MultiPlatformManager::PlatformWithName("Poplar").ConsumeValueOrDie();
  auto* stream_executor = platform->ExecutorForDevice(0).ConsumeValueOrDie();

  tensorflow::IPUOptions opts;
  auto* p = static_cast<PoplarPlatform*>(platform);
  EXPECT_TRUE(p->ConfigurePoplarDevice(0, opts).ok());

  PoplarCompiler compiler;

  hlo_module =
      compiler.RunHloPasses(std::move(hlo_module), stream_executor, nullptr)
          .ConsumeValueOrDie();

  std::unique_ptr<Executable> executable =
      compiler.RunBackend(std::move(hlo_module), stream_executor, nullptr)
          .ConsumeValueOrDie();

  PoplarExecutable* e = static_cast<PoplarExecutable*>(executable.get());
  const auto& input_output_aliasing_map = GetInputOutputAliasingMap(e);
  const auto& input_infos = input_output_aliasing_map.GetEntryInputInfos();
  const auto& output_infos = input_output_aliasing_map.GetEntryOutputInfos();

  EXPECT_EQ(3, input_infos.size());
  EXPECT_TRUE(input_infos[0].IsStreaming());
  EXPECT_TRUE(input_infos[1].IsStreaming());
  EXPECT_TRUE(input_infos[2].IsStreaming());
  EXPECT_EQ(2, output_infos.size());
  EXPECT_TRUE(output_infos[0].IsStreaming());
  EXPECT_TRUE(output_infos[1].IsStreaming());
}

TEST_F(GraphCompileIoMapTest, GetTupleFromTuple) {
  Shape image_shape = ShapeUtil::MakeShape(S32, {2, 2});

  auto builder = HloComputation::Builder(TestName());
  auto in1 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, image_shape, "input1"));
  auto in2 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, image_shape, "input2"));
  auto in3 = builder.AddInstruction(
      HloInstruction::CreateParameter(2, image_shape, "input3"));
  auto add1 = builder.AddInstruction(
      HloInstruction::CreateBinary(image_shape, HloOpcode::kAdd, in1, in2));
  auto add2 = builder.AddInstruction(
      HloInstruction::CreateBinary(image_shape, HloOpcode::kAdd, in2, in3));
  auto tup1 = builder.AddInstruction(HloInstruction::CreateTuple({add1, add2}));
  auto tup2 = builder.AddInstruction(HloInstruction::CreateTuple({add2, in3}));
  auto tup3 = builder.AddInstruction(HloInstruction::CreateTuple({tup1, tup2}));
  builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(tup2->shape(), tup3, 1));

  auto computation = builder.Build();

  auto hlo_module = CreateNewModule();
  hlo_module->AddEntryComputation(std::move(computation));

  auto* platform =
      se::MultiPlatformManager::PlatformWithName("Poplar").ConsumeValueOrDie();
  auto* stream_executor = platform->ExecutorForDevice(0).ConsumeValueOrDie();

  tensorflow::IPUOptions opts;
  auto* p = static_cast<PoplarPlatform*>(platform);
  EXPECT_TRUE(p->ConfigurePoplarDevice(0, opts).ok());

  PoplarCompiler compiler;

  hlo_module =
      compiler.RunHloPasses(std::move(hlo_module), stream_executor, nullptr)
          .ConsumeValueOrDie();

  std::unique_ptr<Executable> executable =
      compiler.RunBackend(std::move(hlo_module), stream_executor, nullptr)
          .ConsumeValueOrDie();

  PoplarExecutable* e = static_cast<PoplarExecutable*>(executable.get());
  const auto& input_output_aliasing_map = GetInputOutputAliasingMap(e);
  const auto& input_infos = input_output_aliasing_map.GetEntryInputInfos();
  const auto& output_infos = input_output_aliasing_map.GetEntryOutputInfos();

  EXPECT_EQ(3, input_infos.size());
  EXPECT_TRUE(input_infos[0].IsStreaming());
  EXPECT_TRUE(input_infos[1].IsStreaming());
  EXPECT_TRUE(input_infos[2].IsStreaming());
  EXPECT_EQ(2, output_infos.size());
  EXPECT_TRUE(output_infos[0].IsStreaming());
  EXPECT_TRUE(output_infos[1].IsStreaming());
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
