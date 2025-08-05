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

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/gpu/tma_metadata.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

using ::tsl::proto_testing::EqualsProto;
using Kind = Thunk::Kind;

TEST(KernelThunkTest, CreateWithDefaultValues) {
  KernelThunk thunk(Thunk::ThunkInfo(),
                    /*kernel_name=*/"",
                    /*kernel_arguments=*/emitters::KernelArguments({}),
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

  emitters::KernelArgument arg0(ShapeUtil::MakeShape(F32, {1024}), slice0);
  emitters::KernelArgument arg1(ShapeUtil::MakeShape(F32, {256}), slice1);
  arg0.set_written(false);
  arg1.set_written(true);

  emitters::KernelArguments kernel_arguments({arg0, arg1});

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

  emitters::KernelArgument arg0(ShapeUtil::MakeShape(F32, {1024}), slice0);
  emitters::KernelArgument arg1(ShapeUtil::MakeShape(F32, {256}), slice1);
  arg0.set_written(false);
  arg1.set_written(true);

  emitters::KernelArguments kernel_arguments({arg0, arg1});

  LaunchDimensions launch_dimensions(se::BlockDim(32, 31, 30),
                                     se::ThreadDim(256, 255, 254));

  TF_ASSERT_OK_AND_ASSIGN(stream_executor::gpu::TmaDescriptor descriptor,
                          stream_executor::gpu::TmaDescriptor::Create(
                              /*global_dims=*/{1024, 1024},
                              /*global_strides=*/{1024},
                              /*box_dims=*/{128, 128},
                              /*element_strides=*/{1, 1},
                              /*element_byte_width=*/4));
  stream_executor::gpu::TmaMetadata tma_metadata;
  tma_metadata.arg_index_to_tma_info.emplace(/*arg_index=*/0,
                                             std::move(descriptor));

  KernelThunk thunk(thunk_info,
                    /*kernel_name=*/"kernel123",
                    /*kernel_arguments=*/kernel_arguments,
                    /*launch_dimensions=*/launch_dimensions,
                    /*cluster_dim=*/se::ClusterDim(8, 7, 6),
                    /*shmem_bytes=*/1024,
                    /*tma_metadata=*/tma_metadata);
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
          launch_dimensions {
            block_counts { coordinates { x: 32 y: 31 z: 30 } }
            thread_counts_per_block { coordinates { x: 256 y: 255 z: 254 } }
          }
          cluster_dim { coordinates { x: 8 y: 7 z: 6 } }
          shmem_bytes: 1024
          tma_metadata {
            arg_index_to_tma_info {
              key: 0
              value {
                element_size: 4
                global_dims: 1024
                global_dims: 1024
                global_strides: 1024
                box_dims: 128
                box_dims: 128
                element_strides: 1
                element_strides: 1
                interleave: INTERLEAVE_NONE
                swizzle: SWIZZLE_NONE
                l2_promotion: L2_PROMOTION_NONE
                float_oob_fill: FLOAT_OOB_FILL_NONE
              }
            }
          }
        }
      )pb"));
}

TEST(KernelThunkTest, ToAndFromProto) {
  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = "DotGeneral";
  thunk_info.execution_stream_id = 123;

  std::array allocations{
      BufferAllocation{/*index=*/0, /*size=*/1024, /*color=*/0},
      BufferAllocation{/*index=*/0, /*size=*/256, /*color=*/0}};

  // Note that slices keep a pointer to the allocation. Therefore `allocations`
  // shouldn't be mutated afterwards.
  BufferAllocation::Slice slice0(&allocations.at(0), /*offset=*/0,
                                 /*size=*/1024);
  BufferAllocation::Slice slice1(&allocations.at(1), /*offset=*/0,
                                 /*size=*/256);

  emitters::KernelArgument arg0(ShapeUtil::MakeShape(F32, {1024}), slice0);
  emitters::KernelArgument arg1(ShapeUtil::MakeShape(F32, {256}), slice1);
  arg0.set_written(false);
  arg1.set_written(true);

  emitters::KernelArguments kernel_arguments({arg0, arg1});

  LaunchDimensions launch_dimensions(se::BlockDim(32, 31, 30),
                                     se::ThreadDim(256, 255, 254));
  se::ClusterDim cluster_dim(8, 7, 6);
  constexpr absl::string_view kKernelName = "kernel123";
  constexpr int kSharedMemoryBytes = 1024;

  TF_ASSERT_OK_AND_ASSIGN(stream_executor::gpu::TmaDescriptor descriptor,
                          stream_executor::gpu::TmaDescriptor::Create(
                              /*global_dims=*/{1024, 1024},
                              /*global_strides=*/{1024},
                              /*box_dims=*/{128, 128},
                              /*element_strides=*/{1, 1},
                              /*element_byte_width=*/4));
  stream_executor::gpu::TmaMetadata tma_metadata;
  tma_metadata.arg_index_to_tma_info.emplace(/*arg_index=*/0,
                                             std::move(descriptor));

  KernelThunk thunk(thunk_info, std::string{kKernelName}, kernel_arguments,
                    launch_dimensions, cluster_dim, kSharedMemoryBytes,
                    tma_metadata);
  TF_ASSERT_OK_AND_ASSIGN(ThunkProto proto, thunk.ToProto());
  ASSERT_TRUE(proto.has_kernel_thunk());
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<KernelThunk> reconstructed_thunk,
      KernelThunk::FromProto(thunk_info, proto.kernel_thunk(), allocations));

  EXPECT_THAT(reconstructed_thunk->cluster_dim(), cluster_dim);
  EXPECT_THAT(reconstructed_thunk->kernel_name(), kKernelName);
  EXPECT_THAT(reconstructed_thunk->launch_dimensions(), launch_dimensions);
  EXPECT_THAT(reconstructed_thunk->shmem_bytes(), kSharedMemoryBytes);
  EXPECT_THAT(reconstructed_thunk->written(),
              ::testing::ElementsAre(false, true));
  EXPECT_THAT(reconstructed_thunk->arguments(),
              ::testing::ElementsAre(slice0, slice1));
  EXPECT_THAT(reconstructed_thunk->tma_metadata(), tma_metadata);
}
}  // namespace
}  // namespace xla::gpu
