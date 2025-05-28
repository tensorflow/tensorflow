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

#include "xla/backends/gpu/runtime/kernel_thunk.h"

#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace xla::gpu {
namespace {

using ::tsl::proto_testing::EqualsProto;
using Kind = Thunk::Kind;

TEST(KernelThunkTest, CreateWithDefaultValues) {
  KernelThunk thunk(Thunk::ThunkInfo(),
                    /*kernel_name=*/"",
                    /*kernel_arguments=*/{},
                    /*launch_dimensions=*/LaunchDimensions(),
                    /*cluster_dim=*/se::ClusterDim(),
                    /*shmem_bytes=*/0,
                    /*tma_metadata=*/std::nullopt);
  EXPECT_EQ(thunk.kind(), Kind::kKernel);
  EXPECT_TRUE(thunk.kernel_name().empty());
  EXPECT_TRUE(thunk.arguments().empty());
  EXPECT_TRUE(thunk.written().empty());
  EXPECT_EQ(thunk.launch_dimensions().block_counts(), se::BlockDim(1, 1, 1));
  EXPECT_EQ(thunk.launch_dimensions().thread_counts_per_block(),
            se::ThreadDim(1, 1, 1));
  EXPECT_EQ(thunk.cluster_dim(), se::ClusterDim(1, 1, 1));
  EXPECT_EQ(thunk.shmem_bytes(), 0);
  EXPECT_EQ(thunk.ToString(0),
            ", kernel = , launch dimensions = blocks: {1, 1, 1}, "
            "threads/block: {1, 1, 1}, cluster_dim = ClusterDim{1, 1, 1}");
}

TEST(KernelThunkTest, CreateAndGettersAndToString) {
  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = "DotGeneral";
  thunk_info.execution_stream_id = 123;

  BufferAllocation alloc0(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation::Slice slice0(&alloc0, /*offset=*/0, /*size=*/1024);

  BufferAllocation alloc1(/*index=*/0, /*size=*/256, /*color=*/0);
  BufferAllocation::Slice slice1(&alloc1, /*offset=*/0, /*size=*/256);

  std::vector<emitters::KernelArgument> kernel_arguments = {
      emitters::KernelArgument(
          ShapeUtil::MakeValidatedShape(F32, {1024}).value(), slice0,
          /*written=*/false),
      emitters::KernelArgument(
          ShapeUtil::MakeValidatedShape(F32, {256}).value(), slice1,
          /*written=*/true)};

  LaunchDimensions launch_dimensions(se::BlockDim(32, 31, 30),
                                     se::ThreadDim(256, 255, 254));

  KernelThunk thunk(thunk_info,
                    /*kernel_name=*/"kernel123",
                    /*kernel_arguments=*/kernel_arguments,
                    /*launch_dimensions=*/launch_dimensions,
                    /*cluster_dim=*/se::ClusterDim(8, 7, 6),
                    /*shmem_bytes=*/1024,
                    /*tma_metadata=*/std::nullopt);
  EXPECT_EQ(thunk.kind(), Kind::kKernel);
  EXPECT_EQ(thunk.kernel_name(), "kernel123");
  EXPECT_EQ(thunk.arguments(),
            std::vector<BufferAllocation::Slice>({slice0, slice1}));
  EXPECT_EQ(thunk.written(), std::vector<bool>({false, true}));
  EXPECT_EQ(thunk.launch_dimensions().block_counts(), se::BlockDim(32, 31, 30));
  EXPECT_EQ(thunk.launch_dimensions().thread_counts_per_block(),
            se::ThreadDim(256, 255, 254));
  EXPECT_EQ(thunk.cluster_dim(), se::ClusterDim(8, 7, 6));
  EXPECT_EQ(thunk.shmem_bytes(), 1024);
  EXPECT_EQ(
      thunk.ToString(0),
      ", kernel = kernel123, launch dimensions = blocks: {32, 31, 30}, "
      "threads/block: {256, 255, 254}, cluster_dim = ClusterDim{8, 7, 6}");
}

TEST(KernelThunkTest, ToProto) {
  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = "DotGeneral";
  thunk_info.execution_stream_id = 123;

  BufferAllocation alloc0(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation::Slice slice0(&alloc0, /*offset=*/0, /*size=*/1024);

  BufferAllocation alloc1(/*index=*/0, /*size=*/256, /*color=*/0);
  BufferAllocation::Slice slice1(&alloc1, /*offset=*/0, /*size=*/256);

  std::vector<emitters::KernelArgument> kernel_arguments = {
      emitters::KernelArgument(
          ShapeUtil::MakeValidatedShape(F32, {1024}).value(), slice0,
          /*written=*/false),
      emitters::KernelArgument(
          ShapeUtil::MakeValidatedShape(F32, {256}).value(), slice1,
          /*written=*/true)};

  LaunchDimensions launch_dimensions(se::BlockDim(32, 31, 30),
                                     se::ThreadDim(256, 255, 254));
  KernelThunk thunk(thunk_info,
                    /*kernel_name=*/"kernel123",
                    /*kernel_arguments=*/kernel_arguments,
                    /*launch_dimensions=*/launch_dimensions,
                    /*cluster_dim=*/se::ClusterDim(8, 7, 6),
                    /*shmem_bytes=*/1024,
                    /*tma_metadata=*/std::nullopt);
  TF_ASSERT_OK_AND_ASSIGN(ThunkProto proto, thunk.ToProto());
  EXPECT_THAT(
      proto, EqualsProto(R"pb(
        thunk_info { profile_annotation: "DotGeneral" execution_stream_id: 123 }
        kernel_thunk {
          args { size: 1024 }
          args { size: 256 }
          written: false
          written: true
          kernel_name: "kernel123"
          launch_block_counts { x: 32 y: 31 z: 30 }
          launch_thread_counts_per_block { x: 256 y: 255 z: 254 }
          cluster_dim { x: 8 y: 7 z: 6 }
          shmem_bytes: 1024
        }
      )pb"));
}

}  // namespace
}  // namespace xla::gpu
