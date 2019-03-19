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
#include "tensorflow/compiler/plugin/poplar/driver/tools/input_output_aliasing_map.h"

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

  static std::unique_ptr<HloModule> CreateNewVerifiedModuleWithConfig(
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

  auto hlo_module = CreateNewVerifiedModule();
  hlo_module->AddEntryComputation(std::move(computation));

  auto* platform =
      se::MultiPlatformManager::PlatformWithName("Poplar").ConsumeValueOrDie();
  auto* stream_executor = platform->ExecutorForDevice(0).ConsumeValueOrDie();

  IpuOptions opts;
  auto* p = static_cast<PoplarPlatform*>(platform);
  EXPECT_TRUE(p->ConfigurePoplarDevices(opts).ok());

  PoplarCompiler compiler;

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
  auto res = builder.AddInstruction(
      HloInstruction::CreateParameter(1, image_shape, "res"));
  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(image_shape, HloOpcode::kAdd, res, in1));
  builder.AddInstruction(HloInstruction::CreateTuple({add}));

  OpMetadata metadata1;
  metadata1.set_op_name("grad%1");
  metadata1.set_op_type("ResourceApplyGradientDescent");
  add->set_metadata(metadata1);

  auto computation = builder.Build();

  auto config = GetModuleConfigForTest();
  config.set_argument_count(2);
  config.set_resource_input_count(1);
  config.set_input_mapping({0, 1});
  config.set_resource_update_to_input_index({1});
  auto hlo_module = CreateNewVerifiedModuleWithConfig(config);
  hlo_module->AddEntryComputation(std::move(computation));

  auto* platform =
      se::MultiPlatformManager::PlatformWithName("Poplar").ConsumeValueOrDie();
  auto* stream_executor = platform->ExecutorForDevice(0).ConsumeValueOrDie();

  IpuOptions opts;
  auto* p = static_cast<PoplarPlatform*>(platform);
  EXPECT_TRUE(p->ConfigurePoplarDevices(opts).ok());

  PoplarCompiler compiler;

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

  auto hlo_module = CreateNewVerifiedModule();
  hlo_module->AddEntryComputation(std::move(computation));

  auto* platform =
      se::MultiPlatformManager::PlatformWithName("Poplar").ConsumeValueOrDie();
  auto* stream_executor = platform->ExecutorForDevice(0).ConsumeValueOrDie();

  IpuOptions opts;
  auto* p = static_cast<PoplarPlatform*>(platform);
  EXPECT_TRUE(p->ConfigurePoplarDevices(opts).ok());

  PoplarCompiler compiler;

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

  auto hlo_module = CreateNewVerifiedModule();
  hlo_module->AddEntryComputation(std::move(computation));

  auto* platform =
      se::MultiPlatformManager::PlatformWithName("Poplar").ConsumeValueOrDie();
  auto* stream_executor = platform->ExecutorForDevice(0).ConsumeValueOrDie();

  IpuOptions opts;
  auto* p = static_cast<PoplarPlatform*>(platform);
  EXPECT_TRUE(p->ConfigurePoplarDevices(opts).ok());

  PoplarCompiler compiler;

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

TEST_F(GraphCompileIoMapTest, ResourceInit) {
  Shape shape = ShapeUtil::MakeShape(F32, {2});

  auto builder = HloComputation::Builder(TestName());
  auto i1 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "input1"));
  auto i2 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, shape, "input2"));
  auto i3 = builder.AddInstruction(
      HloInstruction::CreateParameter(2, shape, "input3"));
  builder.AddInstruction(HloInstruction::CreateTuple({i1, i1, i2, i3}));

  auto computation = builder.Build();

  /* 3 inputs, 4 resources, all resources uninitialized, 2 set from one of the
   * inputs
   */
  auto config = GetModuleConfigForTest();
  config.set_argument_count(7);
  config.set_resource_input_count(4);
  config.set_input_mapping({0, 1, 2});
  config.set_resource_update_to_input_index({3, 4, 5, 6});
  auto hlo_module = CreateNewVerifiedModuleWithConfig(config);
  hlo_module->AddEntryComputation(std::move(computation));

  auto* platform =
      se::MultiPlatformManager::PlatformWithName("Poplar").ConsumeValueOrDie();
  auto* stream_executor = platform->ExecutorForDevice(0).ConsumeValueOrDie();

  se::Stream stream(stream_executor);
  stream.Init();

  IpuOptions opts;
  auto* p = static_cast<PoplarPlatform*>(platform);
  EXPECT_TRUE(p->ConfigurePoplarDevices(opts).ok());

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
  EXPECT_EQ(4, output_infos.size());
  EXPECT_FALSE(output_infos[0].IsStreaming());
  EXPECT_FALSE(output_infos[1].IsStreaming());
  EXPECT_FALSE(output_infos[2].IsStreaming());
  EXPECT_FALSE(output_infos[3].IsStreaming());
  EXPECT_FALSE(output_infos[0].IsResourceModified());
  EXPECT_FALSE(output_infos[1].IsResourceModified());
  EXPECT_FALSE(output_infos[2].IsResourceModified());
  EXPECT_FALSE(output_infos[3].IsResourceModified());

  StreamExecutorMemoryAllocator allocator(platform, {stream_executor});

  se::DeviceMemoryBase buf0 = allocator.Allocate(0, sizeof(float) * 2, false)
                                  .ConsumeValueOrDie()
                                  .Forget();
  se::DeviceMemoryBase buf1 = allocator.Allocate(0, sizeof(float) * 2, false)
                                  .ConsumeValueOrDie()
                                  .Forget();
  se::DeviceMemoryBase buf2 = allocator.Allocate(0, sizeof(float) * 2, false)
                                  .ConsumeValueOrDie()
                                  .Forget();

  float b0[2] = {1.0, 2.0};
  stream_executor->SynchronousMemcpyH2D(b0, sizeof(float) * 2, &buf0);

  float b1[2] = {3.0, 4.0};
  stream_executor->SynchronousMemcpyH2D(b1, sizeof(float) * 2, &buf1);

  float b2[2] = {5.0, 6.0};
  stream_executor->SynchronousMemcpyH2D(b2, sizeof(float) * 2, &buf2);

  ShapedBuffer arg0(shape, shape, platform, 0);
  arg0.set_buffer(buf0, {});
  ShapedBuffer arg1(shape, shape, platform, 0);
  arg1.set_buffer(buf1, {});
  ShapedBuffer arg2(shape, shape, platform, 0);
  arg2.set_buffer(buf2, {});

  std::vector<const ShapedBuffer*> args = {&arg0, &arg1, &arg2};

  ExecutableRunOptions ro;
  ro.set_stream(&stream).set_allocator(&allocator).set_device_ordinal(0);

  ServiceExecutableRunOptions sro(ro);

  auto ret = e->ExecuteOnStream(&sro, args, NULL).ConsumeValueOrDie();

  auto ret_buf0 = ret.buffer({0});
  auto ret_buf1 = ret.buffer({1});
  auto ret_buf2 = ret.buffer({2});
  auto ret_buf3 = ret.buffer({3});
  EXPECT_NE(ret_buf0.opaque(), buf0.opaque());
  EXPECT_NE(ret_buf1.opaque(), buf0.opaque());
  EXPECT_NE(ret_buf2.opaque(), buf1.opaque());
  EXPECT_NE(ret_buf3.opaque(), buf2.opaque());

  float ret_b[2];
  stream_executor->SynchronousMemcpyD2H(ret_buf0, sizeof(float) * 2, ret_b);
  EXPECT_EQ(ret_b[0], 1.0f);
  EXPECT_EQ(ret_b[1], 2.0f);
  stream_executor->SynchronousMemcpyD2H(ret_buf1, sizeof(float) * 2, ret_b);
  EXPECT_EQ(ret_b[0], 1.0f);
  EXPECT_EQ(ret_b[1], 2.0f);
  stream_executor->SynchronousMemcpyD2H(ret_buf2, sizeof(float) * 2, ret_b);
  EXPECT_EQ(ret_b[0], 3.0f);
  EXPECT_EQ(ret_b[1], 4.0f);
  stream_executor->SynchronousMemcpyD2H(ret_buf3, sizeof(float) * 2, ret_b);
  EXPECT_EQ(ret_b[0], 5.0f);
  EXPECT_EQ(ret_b[1], 6.0f);
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
