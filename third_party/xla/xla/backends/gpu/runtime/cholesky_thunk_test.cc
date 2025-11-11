/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/cholesky_thunk.h"

#include <cstdint>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/gpu_solver_context.h"
#include "xla/stream_executor/platform/platform_object_registry.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::tsl::proto_testing::EqualsProto;

class CholeskyThunkTest : public HloTestBase {};

TEST_F(CholeskyThunkTest, ProtoRoundTrip) {
  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = "cholesky";
  CholeskyOptions options;
  options.set_lower(true);
  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(0, 256, 0), BufferAllocation(1, 128, 0),
      BufferAllocation(2, 4, 0)};
  BufferAllocation::Slice a_buffer(&buffer_allocations[0], 0, 256);
  BufferAllocation::Slice workspace_buffer(&buffer_allocations[1], 0, 128);
  BufferAllocation::Slice info_buffer(&buffer_allocations[2], 0, 4);
  PrimitiveType type = F32;
  int64_t batch_size = 1;
  int64_t n = 16;

  TF_ASSERT_OK_AND_ASSIGN(
      auto solver_creator,
      stream_executor::PlatformObjectRegistry::GetGlobalRegistry()
          .FindObject<stream_executor::GpuSolverContextFactory>(
              backend().platform()->id()));

  CholeskyThunk thunk(thunk_info, options, a_buffer, workspace_buffer,
                      info_buffer, type, batch_size, n, solver_creator);

  TF_ASSERT_OK_AND_ASSIGN(ThunkProto proto, thunk.ToProto());

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<CholeskyThunk> round_trip_thunk,
      CholeskyThunk::FromProto(thunk.thunk_info(), proto.cholesky_thunk(),
                               buffer_allocations, *backend().platform()));

  EXPECT_THAT(round_trip_thunk->ToProto(), IsOkAndHolds(EqualsProto(proto)));
}

}  // namespace
}  // namespace xla::gpu
