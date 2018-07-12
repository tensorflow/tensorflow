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
  const OutputMap& GetMap(PoplarExecutable* e) { return e->output_map_; }
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
  EXPECT_EQ(0, GetMap(e).size());
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
  EXPECT_EQ(1, GetMap(e).size());
  EXPECT_EQ(0, GetMap(e).at(0));
}

TEST_F(GraphCompileIoMapTest, Input2Shared) {
  Shape image_shape = ShapeUtil::MakeShape(F32, {1, 4, 4, 2});

  auto builder = HloComputation::Builder(TestName());
  auto in1 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, image_shape, "input1"));
  auto in2 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, image_shape, "input2"));
  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(image_shape, HloOpcode::kAdd, in2, in1));
  builder.AddInstruction(HloInstruction::CreateTuple({add}));

  OpMetadata metadata1;
  metadata1.set_op_name("grad%1");
  metadata1.set_op_type("ResourceApplyGradientDescent");
  add->set_metadata(metadata1);

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
  EXPECT_EQ(1, GetMap(e).size());
  EXPECT_EQ(1, GetMap(e).at(0));
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
  ASSERT_EQ(4, GetMap(e).size());
  EXPECT_EQ(0, GetMap(e).at(0));
  EXPECT_EQ(1, GetMap(e).at(1));
  EXPECT_EQ(1, GetMap(e).at(2));
  EXPECT_EQ(2, GetMap(e).at(3));
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
      HloInstruction::CreateGetTupleElement(tup3->shape(), tup3, 1));

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
  ASSERT_EQ(1, GetMap(e).size());
  EXPECT_EQ(2, GetMap(e).at(1));
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
