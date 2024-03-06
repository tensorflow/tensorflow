/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/compiler.h"

#include <gtest/gtest.h>
#include "xla/autotune_results.pb.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/stream_executor/gpu/gpu_init.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tests/test_macros.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

TEST(TargetConfigTest, DISABLED_ON_CPU(ExecutorConstructorFillsAllFields)) {
  TF_ASSERT_OK(stream_executor::ValidateGPUMachineManager());
  TF_ASSERT_OK_AND_ASSIGN(
      stream_executor::StreamExecutor * executor,
      stream_executor::GPUMachineManager()->ExecutorForDevice(0));
  Compiler::TargetConfig config(executor);
  stream_executor::GpuTargetConfigProto target = config.ToProto();

  // We don't attempt to validate values because doing so would require talking
  // to the driver directly.
  EXPECT_GT(target.dnn_version_info().major(), 0) << target.DebugString();
  EXPECT_GT(target.gpu_device_info().threads_per_block_limit(), 0)
      << target.DebugString();
  EXPECT_NE(target.device_description_str(), "") << target.DebugString();
  EXPECT_NE(target.platform_name(), "") << target.DebugString();
  EXPECT_EQ(target.autotune_results().version(), 0);

  EXPECT_EQ(5,
            stream_executor::GpuTargetConfigProto::descriptor()->field_count())
      << "Make sure all the fields in GpuTargetConfigProto are set and "
         "validated!";
}

TEST(TargetConfigTest, ProtoConstructorFillsAllFields) {
  stream_executor::GpuTargetConfigProto config_proto;
  config_proto.set_platform_name("platform");
  config_proto.mutable_dnn_version_info()->set_major(2);
  config_proto.mutable_gpu_device_info()->set_threads_per_block_limit(5);
  config_proto.set_device_description_str("foo");

  Compiler::TargetConfig config(config_proto);
  stream_executor::GpuTargetConfigProto target = config.ToProto();

  EXPECT_EQ(target.dnn_version_info().major(),
            config_proto.dnn_version_info().major())
      << target.DebugString();
  EXPECT_EQ(target.gpu_device_info().threads_per_block_limit(), 5)
      << target.DebugString();
  EXPECT_EQ(target.device_description_str(), "foo") << target.DebugString();
  EXPECT_EQ(target.platform_name(), "platform") << target.DebugString();
  EXPECT_EQ(target.autotune_results().version(), 0);

  EXPECT_EQ(5,
            stream_executor::GpuTargetConfigProto::descriptor()->field_count())
      << "Make sure all the fields in GpuTargetConfigProto are set and "
         "validated!";
}

}  // namespace
}  // namespace xla
