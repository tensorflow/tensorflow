/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/pjrt/cpu_device.h"

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/executable_build_options.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/core/platform/random.h"

namespace xla {
namespace {

TEST(CpuStreamDeviceTest, BlocksDeviceToHostStream) {  // b/214236179
  const float data[] = {1, 2, 3, 4};
  auto client = *GetCpuClient(true);
  auto* device = client->devices()[0];
  auto buffer = *client->BufferFromHostBuffer(
      &data, PrimitiveType::F32, {4}, absl::nullopt,
      PjRtClient::HostBufferSemantics::kZeroCopy, {}, device);
  auto literal = *buffer->ToLiteralSync();
}

}  // namespace
}  // namespace xla
