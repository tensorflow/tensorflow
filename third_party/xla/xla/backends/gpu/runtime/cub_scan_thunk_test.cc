/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/cub_scan_thunk.h"

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace xla::gpu {
namespace {

using ::tsl::proto_testing::EqualsProto;

TEST(CubScanThunkTest, ToProto) {
  TF_ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                          se::PlatformManager::PlatformWithName("CUDA"));
  ASSERT_NE(platform, nullptr);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<CubScanRunnerInterface> runner,
                          CubScanRunnerInterface::Create(S32, "CUDA"));

  BufferAllocation alloc0(0, 1024, 0);
  BufferAllocation alloc1(1, 1024, 0);
  BufferAllocation alloc2(2, 1024, 0);
  CubScanThunk thunk({}, std::move(runner), S32, "CUDA",
                     BufferAllocation::Slice(&alloc0, 0, 256),
                     BufferAllocation::Slice(&alloc1, 0, 256),
                     BufferAllocation::Slice(&alloc2, 0, 128), 64);

  TF_ASSERT_OK_AND_ASSIGN(ThunkProto proto, thunk.ToProto());
  EXPECT_THAT(
      proto,
      EqualsProto(
          R"pb(
            thunk_info {}
            cub_scan_thunk {
              type: S32
              platform_name: "CUDA"
              input_slice { offset: 0 size: 256 buffer_allocation_index: 0 }
              output_slice { offset: 0 size: 256 buffer_allocation_index: 1 }
              scratch_slice { offset: 0 size: 128 buffer_allocation_index: 2 }
              num_elements: 64
            }
          )pb"));
}

}  // namespace
}  // namespace xla::gpu
