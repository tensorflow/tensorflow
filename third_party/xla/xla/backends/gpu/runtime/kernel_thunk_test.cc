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
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/command_buffer_cmd_emitter.h"
#include "xla/backends/gpu/runtime/command_buffer_thunk.h"
#include "xla/backends/gpu/runtime/command_executor.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/backends/gpu/runtime/thunk_id.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/literal.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/kernels/custom_kernel.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/platform_util.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/gpu/gpu_test_kernels.h"
#include "xla/stream_executor/gpu/tma_metadata.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

using ::tsl::proto_testing::EqualsProto;
using ::tsl::proto_testing::ParseTextProtoOrDie;
using Kind = Thunk::Kind;

TEST(KernelThunkTest, CreateWithDefaultValues) {
  KernelThunk thunk(Thunk::ThunkInfo(),
                    /*kernel_name=*/"",
                    /*kernel_arguments=*/emitters::KernelArguments({}),
                    /*launch_dimensions=*/LaunchDimensions(),
                    /*cluster_dim=*/se::ClusterDim(),
                    /*shmem_bytes=*/0,
                    /*tma_metadata=*/se::gpu::TmaMetadata());
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
                    /*tma_metadata=*/se::gpu::TmaMetadata());
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
          args_shape {
            element_type: F32
            dimensions: 1024
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          args_shape {
            element_type: F32
            dimensions: 256
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
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

TEST(KernelThunkTest, BufferUsesReturnsCorrectBuffers) {
  Shape arg_shape = ShapeUtil::MakeShape(F32, {512});
  BufferAllocation alloc(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation::Slice slice0(&alloc, /*offset=*/0, /*size=*/512);
  BufferAllocation::Slice slice1(&alloc, /*offset=*/512, /*size=*/512);
  emitters::KernelArgument arg0(arg_shape, slice0);
  emitters::KernelArgument arg1(arg_shape, slice1);
  arg0.set_written(false);
  arg1.set_written(true);
  emitters::KernelArguments kernel_arguments({arg0, arg1});
  KernelThunk thunk(Thunk::ThunkInfo(), "kernel", kernel_arguments,
                    LaunchDimensions(), se::ClusterDim(), /*shmem_bytes=*/0,
                    se::gpu::TmaMetadata());

  Thunk::BufferUses buffers = thunk.buffer_uses();

  ASSERT_THAT(buffers, testing::UnorderedElementsAre(
                           BufferUse::Read(slice0, arg_shape),
                           BufferUse::Write(slice1, arg_shape)));
}

TEST(KernelThunkTest, BufferUsesReturnsBuffersInConsistentOrder) {
  Shape arg_shape = ShapeUtil::MakeShape(F32, {512});
  BufferAllocation alloc(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation::Slice slice0(&alloc, /*offset=*/0, /*size=*/512);
  BufferAllocation::Slice slice1(&alloc, /*offset=*/512, /*size=*/512);
  emitters::KernelArgument arg0(arg_shape, slice0);
  emitters::KernelArgument arg1(arg_shape, slice1);
  arg0.set_written(false);
  arg1.set_written(true);
  emitters::KernelArguments kernel_arguments({arg0, arg1});
  KernelThunk thunk(Thunk::ThunkInfo(), "kernel", kernel_arguments,
                    LaunchDimensions(), se::ClusterDim(), /*shmem_bytes=*/0,
                    se::gpu::TmaMetadata());

  Thunk::BufferUses buffers1 = thunk.buffer_uses();
  Thunk::BufferUses buffers2 = thunk.buffer_uses();

  ASSERT_THAT(buffers1, testing::ContainerEq(buffers2));
}

class KernelThunkTmaPTXTest : public ::testing::TestWithParam<bool> {
 public:
  absl::StatusOr<std::unique_ptr<KernelThunk>> GetTmaKernelThunk() {
    std::string tma_kernel_thunk = R"pb(
      thunk_info { profile_annotation: "tma_kernel" execution_stream_id: 123 }
      kernel_thunk {
        args { size: 1048576 buffer_allocation_index: 0 }
        args { size: 1048576 offset: 1048576 }
        args { size: 4194304 offset: 2097152 }
        args_shape {
          element_type: F32
          dimensions: 262144
          layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
          is_dynamic_dimension: false
        }
        args_shape {
          element_type: F32
          dimensions: 262144
          layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
          is_dynamic_dimension: false
        }
        args_shape {
          element_type: F32
          dimensions: 1048576
          layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
          is_dynamic_dimension: false
        }
        written: false
        written: false
        written: true
        kernel_name: "tma_dot_kernel"
        launch_dimensions {
          block_counts { coordinates { x: 4096 y: 1 z: 1 } }
          thread_counts_per_block { coordinates { x: 128 y: 1 z: 1 } }
        }
        shmem_bytes: 24600
        tma_metadata {
          arg_index_to_tma_info {
            key: 0
            value {
              element_size: 2
              global_dims: 512
              global_dims: 1024
              global_strides: 1024
              box_dims: 64
              box_dims: 16
              element_strides: 1
              element_strides: 1
              swizzle: SWIZZLE_BYTES128
              l2_promotion: L2_PROMOTION_BYTES128
            }
          }
          arg_index_to_tma_info {
            key: 1
            value {
              element_size: 2
              global_dims: 1024
              global_dims: 512
              global_strides: 2048
              box_dims: 16
              box_dims: 128
              element_strides: 1
              element_strides: 1
              swizzle: SWIZZLE_BYTES32
              l2_promotion: L2_PROMOTION_BYTES128
            }
          }
          arg_index_to_tma_info {
            key: 2
            value {
              element_size: 4
              global_dims: 1024
              global_dims: 1024
              global_strides: 4096
              box_dims: 16
              box_dims: 16
              element_strides: 1
              element_strides: 1
              swizzle: SWIZZLE_BYTES64
              l2_promotion: L2_PROMOTION_BYTES128
            }
          }
        }
      }
    )pb";

    ThunkProto tma_kernel_thunk_proto =
        ParseTextProtoOrDie<ThunkProto>(tma_kernel_thunk);

    const size_t total_byte_size =
        tma_kernel_thunk_proto.kernel_thunk().args(0).size() +
        tma_kernel_thunk_proto.kernel_thunk().args(1).size() +
        tma_kernel_thunk_proto.kernel_thunk().args(2).size();
    allocations_.emplace_back(0, total_byte_size, 0);

    return KernelThunk::FromProto(Thunk::ThunkInfo(),
                                  tma_kernel_thunk_proto.kernel_thunk(),
                                  allocations_);
  }

  std::vector<BufferAllocation> allocations_;
};

TEST_P(KernelThunkTmaPTXTest, TmaPTX) {
  auto name = absl::AsciiStrToUpper(
      xla::PlatformUtil::CanonicalPlatformName("gpu").value());
  if (name == "ROCM") {
    GTEST_SKIP() << "TmaPTX cannot run on ROCm.";
  }
  TF_ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                          se::PlatformManager::PlatformWithName(name));
  TF_ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor,
                          platform->ExecutorForDevice(0));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream,
                          executor->CreateStream());
  if (!stream_executor::gpu::IsTmaAvailableForDevice(
          executor->GetDeviceDescription())) {
    GTEST_SKIP() << "TMA is not supported on this platform.";
  }

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<KernelThunk> kernel_thunk,
                          GetTmaKernelThunk());

  Thunk::ExecutableSource executable_source;
  executable_source.text = stream_executor::gpu::GetTmaPtxKernelSpec()
                               .cuda_ptx_in_memory()
                               .value()
                               .ptx;
  Thunk::InitializeParams initialize_params;
  initialize_params.executor = executor;
  initialize_params.src = executable_source;
  initialize_params.stream = stream.get();
  BufferAllocations buffer_allocations(
      /*buffers=*/{executor->Allocate(allocations_[0].size())},
      /*device_ordinal=*/0,
      /*memory_allocator=*/nullptr);

  initialize_params.buffer_allocations = &buffer_allocations;

  ServiceExecutableRunOptions run_options;
  run_options.mutable_run_options()->set_stream(stream.get());

  auto execute_params =
      Thunk::ExecuteParams::Create(run_options, buffer_allocations,
                                   stream.get(), nullptr, nullptr, nullptr, {});

  const bool use_command_buffer = GetParam();

  // We are checking both code paths for TMA kernels: through KernelThunk and
  // CommandBufferThunk.
  if (use_command_buffer) {
    ThunkSequence thunk_sequence;
    thunk_sequence.push_back(std::move(kernel_thunk));

    TF_ASSERT_OK_AND_ASSIGN(
        CommandBufferCmdExecutor cmds,
        ConvertToCommands(thunk_sequence, ConvertToCommandsOptions()));
    auto sequential_thunk = std::make_unique<SequentialThunk>(
        Thunk::ThunkInfo(), std::move(thunk_sequence));

    auto cmd_buffer_thunk = std::make_unique<CommandBufferThunk>(
        std::move(cmds), Thunk::ThunkInfo(), std::move(sequential_thunk), true);

    TF_ASSERT_OK(cmd_buffer_thunk->Initialize(initialize_params));
    TF_ASSERT_OK(cmd_buffer_thunk->ExecuteOnStream(execute_params));
  } else {
    TF_ASSERT_OK(kernel_thunk->Initialize(initialize_params));
    TF_ASSERT_OK(kernel_thunk->ExecuteOnStream(execute_params));
  }

  TF_ASSERT_OK(stream->BlockHostUntilDone());
}

INSTANTIATE_TEST_SUITE_P(KernelThunkTmaPTXTestSuite, KernelThunkTmaPTXTest,
                         ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool>& info) {
                           return info.param ? "in_command_buffer"
                                             : "in_kernel_thunk";
                         });

}  // namespace
}  // namespace xla::gpu
