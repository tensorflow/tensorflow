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
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_buffer_cmd_emitter.h"
#include "xla/backends/gpu/runtime/command_buffer_requests.h"
#include "xla/backends/gpu/runtime/command_buffer_thunk.h"
#include "xla/backends/gpu/runtime/command_executor.h"
#include "xla/backends/gpu/runtime/command_state.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/platform_util.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/gpu_test_kernels.h"
#include "xla/stream_executor/gpu/tma_metadata.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
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
                    /*tma_metadata=*/se::gpu::TmaMetadata(),
                    /*devices_per_host=*/1);
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
            "kernel=, profile_annotation=, launch dimensions="
            "blocks: {1, 1, 1}, threads/block: {1, 1, 1}, "
            "cluster_dim=ClusterDim{1, 1, 1}, shmem_bytes=0");
}

TEST(KernelThunkTest, CreateAndGettersAndToString) {
  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = "DotGeneral";

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
                    /*cluster_dim=*/se::ClusterDim(1, 1, 1),
                    /*shmem_bytes=*/1024,
                    /*tma_metadata=*/se::gpu::TmaMetadata(),
                    /*devices_per_host=*/1,
                    /*zeroed_output_buffer_indices=*/{},
                    /*use_pdl=*/true);
  EXPECT_EQ(thunk.kind(), Kind::kKernel);
  EXPECT_EQ(thunk.kernel_name(), "kernel123");
  EXPECT_EQ(thunk.arguments(),
            std::vector<ShapedSlice>(
                {{slice0, arg0.shape()}, {slice1, arg1.shape()}}));
  EXPECT_EQ(thunk.written(), std::vector<bool>({false, true}));
  EXPECT_EQ(thunk.launch_dimensions().block_counts(), se::BlockDim(32, 31, 30));
  EXPECT_EQ(thunk.launch_dimensions().thread_counts_per_block(),
            se::ThreadDim(256, 255, 254));
  EXPECT_EQ(thunk.cluster_dim(), se::ClusterDim(1, 1, 1));
  EXPECT_EQ(thunk.shmem_bytes(), 1024);
  EXPECT_EQ(
      thunk.ToString(0),
      "kernel=kernel123, profile_annotation=DotGeneral, "
      "launch dimensions=blocks: {32, 31, 30}, threads/block: {256, 255, 254}, "
      "cluster_dim=ClusterDim{1, 1, 1}, shmem_bytes=1024, use_pdl=true");
}

TEST(KernelThunkTest, ToProto) {
  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = "DotGeneral";

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

  ASSERT_OK_AND_ASSIGN(stream_executor::gpu::TmaDescriptor descriptor,
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
                    /*tma_metadata=*/tma_metadata,
                    /*devices_per_host=*/1,
                    /*zeroed_output_buffer_indices=*/{},
                    /*use_pdl=*/true);
  ASSERT_OK_AND_ASSIGN(ThunkProto proto, thunk.ToProto());
  EXPECT_THAT(
      proto, EqualsProto(R"pb(
        thunk_info { profile_annotation: "DotGeneral" }
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
          use_pdl: true
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

  ASSERT_OK_AND_ASSIGN(stream_executor::gpu::TmaDescriptor descriptor,
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
                    tma_metadata, /*devices_per_host=*/1,
                    /*zeroed_output_buffer_indices=*/{},
                    /*use_pdl=*/true);
  ASSERT_OK_AND_ASSIGN(ThunkProto proto, thunk.ToProto());
  ASSERT_TRUE(proto.has_kernel_thunk());
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<KernelThunk> reconstructed_thunk,
      KernelThunk::FromProto(thunk_info, proto.kernel_thunk(), allocations,
                             /*devices_per_host=*/1));

  EXPECT_THAT(reconstructed_thunk->cluster_dim(), cluster_dim);
  EXPECT_THAT(reconstructed_thunk->kernel_name(), kKernelName);
  EXPECT_THAT(reconstructed_thunk->launch_dimensions(), launch_dimensions);
  EXPECT_THAT(reconstructed_thunk->shmem_bytes(), kSharedMemoryBytes);
  EXPECT_THAT(reconstructed_thunk->written(),
              ::testing::ElementsAre(false, true));
  EXPECT_THAT(reconstructed_thunk->arguments(),
              ::testing::ElementsAre(ShapedSlice{slice0, arg0.shape()},
                                     ShapedSlice{slice1, arg1.shape()}));
  EXPECT_THAT(reconstructed_thunk->tma_metadata(), tma_metadata);
  EXPECT_TRUE(reconstructed_thunk->use_pdl());
}

TEST(KernelThunkTest, ConvertToCommandsPropagatesUsePdl) {
  BufferAllocation allocation(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation::Slice slice(&allocation, /*offset=*/0, /*size=*/1024);

  emitters::KernelArgument arg(ShapeUtil::MakeShape(F32, {256}), slice);
  arg.set_written(true);

  auto kernel_thunk = std::make_unique<KernelThunk>(
      Thunk::ThunkInfo(),
      /*kernel_name=*/"kernel",
      /*kernel_arguments=*/
      emitters::KernelArguments(std::vector<emitters::KernelArgument>{arg}),
      /*launch_dimensions=*/LaunchDimensions(),
      /*cluster_dim=*/se::ClusterDim(),
      /*shmem_bytes=*/0,
      /*tma_metadata=*/se::gpu::TmaMetadata(),
      /*devices_per_host=*/1,
      /*zeroed_output_buffer_indices=*/std::vector<int64_t>{},
      /*use_pdl=*/true);

  ThunkSequence thunks;
  thunks.push_back(std::move(kernel_thunk));

  ASSERT_OK_AND_ASSIGN(CommandExecutor commands,
                       ConvertToCommands(thunks, ConvertToCommandsOptions()));

  bool found_kernel_thunk = false;
  ASSERT_OK(commands.Walk([&](const Command* command) {
    if (auto* kernel_thunk = dynamic_cast<const KernelThunk*>(command);
        kernel_thunk != nullptr) {
      found_kernel_thunk = true;
      EXPECT_TRUE(kernel_thunk->use_pdl());
    }
    return absl::OkStatus();
  }));
  EXPECT_TRUE(found_kernel_thunk);
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
                    se::gpu::TmaMetadata(), /*devices_per_host=*/1);

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
                    se::gpu::TmaMetadata(), /*devices_per_host=*/1);

  Thunk::BufferUses buffers1 = thunk.buffer_uses();
  Thunk::BufferUses buffers2 = thunk.buffer_uses();

  ASSERT_THAT(buffers1, testing::ContainerEq(buffers2));
}

class KernelThunkTmaPTXTest : public ::testing::TestWithParam<bool> {
 public:
  absl::StatusOr<std::unique_ptr<KernelThunk>> GetTmaKernelThunk() {
    std::string tma_kernel_thunk = R"pb(
      thunk_info { profile_annotation: "tma_kernel" }
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
                                  allocations_, /*devices_per_host=*/1);
  }

  std::vector<BufferAllocation> allocations_;
};

TEST_P(KernelThunkTmaPTXTest, TmaPTX) {
  auto name = absl::AsciiStrToUpper(
      xla::PlatformUtil::CanonicalPlatformName("gpu").value());
  if (name == "ROCM") {
    GTEST_SKIP() << "TmaPTX cannot run on ROCm.";
  }
  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName(name));
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor,
                       platform->ExecutorForDevice(0));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream,
                       executor->CreateStream());
  if (!stream_executor::gpu::IsTmaAvailableForDevice(
          executor->GetDeviceDescription())) {
    GTEST_SKIP() << "TMA is not supported on this platform.";
  }

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<KernelThunk> kernel_thunk,
                       GetTmaKernelThunk());

  Thunk::ExecutableSource executable_source;
  executable_source.text = stream_executor::gpu::GetTmaPtxKernelSpec()
                               .cuda_ptx_in_memory()
                               .value()
                               .ptx;
  CommandBufferRequests command_buffer_requests;
  BufferAllocations buffer_allocations(
      /*buffers=*/{executor->Allocate(allocations_[0].size())},
      /*device_ordinal=*/0,
      /*memory_allocator=*/nullptr);

  Thunk::InitializeParams initialize_params(&command_buffer_requests);
  initialize_params.executor = executor;
  initialize_params.src = executable_source;
  initialize_params.buffer_allocations = &buffer_allocations;
  initialize_params.stream = stream.get();
  initialize_params.persistent_alloc_indices.emplace();

  ServiceExecutableRunOptions run_options;
  run_options.mutable_run_options()->set_stream(stream.get());

  auto execute_params = Thunk::ExecuteParams::Create(
      run_options, buffer_allocations, stream.get(), nullptr, nullptr, nullptr,
      nullptr, {},
      /*execution_scoped_state=*/nullptr,
      /*persistent_alloc_indices=*/
      initialize_params.persistent_alloc_indices);

  const bool use_command_buffer = GetParam();

  // We are checking both code paths for TMA kernels: through KernelThunk and
  // CommandBufferThunk.
  if (use_command_buffer) {
    ThunkSequence thunk_sequence;
    thunk_sequence.push_back(std::move(kernel_thunk));

    ASSERT_OK_AND_ASSIGN(
        CommandExecutor cmds,
        ConvertToCommands(thunk_sequence, ConvertToCommandsOptions()));
    auto sequential_thunk = std::make_unique<SequentialThunk>(
        Thunk::ThunkInfo(), std::move(thunk_sequence));

    CommandBufferThunkOptions options;
    options.enable_command_buffers_during_profiling = true;
    auto cmd_buffer_thunk = std::make_unique<CommandBufferThunk>(
        std::move(cmds), Thunk::ThunkInfo(), /*devices_in_process=*/1,
        std::move(sequential_thunk), options);

    ASSERT_OK(cmd_buffer_thunk->Initialize(initialize_params));
    ASSERT_OK(cmd_buffer_thunk->ExecuteOnStream(execute_params));
  } else {
    ASSERT_OK(kernel_thunk->Initialize(initialize_params));
    ASSERT_OK(kernel_thunk->ExecuteOnStream(execute_params));
  }

  ASSERT_OK(stream->BlockHostUntilDone());
}

INSTANTIATE_TEST_SUITE_P(KernelThunkTmaPTXTestSuite, KernelThunkTmaPTXTest,
                         ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool>& info) {
                           return info.param ? "in_command_buffer"
                                             : "in_kernel_thunk";
                         });

//===----------------------------------------------------------------------===//
// Command buffer tests (Record)
//===----------------------------------------------------------------------===//

static se::StreamExecutor* GpuExecutor() {
  auto name =
      absl::AsciiStrToUpper(PlatformUtil::CanonicalPlatformName("gpu").value());
  auto* platform = se::PlatformManager::PlatformWithName(name).value();
  return platform->ExecutorForDevice(0).value();
}

// Device buffers for one launch of the AddI32 test kernel, `c = a + b`. Each
// operand is a single int32.
struct AddI32Buffers {
  se::DeviceAddress<int32_t> a_dev;
  se::DeviceAddress<int32_t> b_dev;
  se::DeviceAddress<int32_t> c_dev;
};

// Allocates AddI32 operands on `executor`, initialized to `a`, `b` and c = 0.
static absl::StatusOr<AddI32Buffers> AllocateAddI32Buffers(
    se::StreamExecutor* executor, se::Stream* stream, int32_t a = 1,
    int32_t b = 2) {
  AddI32Buffers buffers{executor->AllocateArray<int32_t>(1, 0),
                        executor->AllocateArray<int32_t>(1, 0),
                        executor->AllocateArray<int32_t>(1, 0)};
  ABSL_RETURN_IF_ERROR(stream->Memcpy(&buffers.a_dev, &a, sizeof(int32_t)));
  ABSL_RETURN_IF_ERROR(stream->Memcpy(&buffers.b_dev, &b, sizeof(int32_t)));
  ABSL_RETURN_IF_ERROR(stream->MemZero(&buffers.c_dev, sizeof(int32_t)));
  // `a` and `b` go out of scope on return, so the copies must have read them.
  ABSL_RETURN_IF_ERROR(stream->BlockHostUntilDone());
  return buffers;
}

// The GPU executor, a stream on it, and AddI32 operands initialized to `a`, `b`
// and c = 0.
struct AddI32Setup {
  se::StreamExecutor* executor;
  std::unique_ptr<se::Stream> stream;
  se::DeviceAddress<int32_t> a_dev;
  se::DeviceAddress<int32_t> b_dev;
  se::DeviceAddress<int32_t> c_dev;
};

// Returns std::nullopt on platforms that cannot run the AddI32 PTX kernel
// (ROCm, oneAPI); the caller should skip. GTEST_SKIP has to be issued from the
// test body itself, which is why that decision is handed back rather than made
// here.
static absl::StatusOr<std::optional<AddI32Setup>> SetUpAddI32(int32_t a = 1,
                                                              int32_t b = 2) {
  ABSL_ASSIGN_OR_RETURN(std::string platform,
                   PlatformUtil::CanonicalPlatformName("gpu"));
  std::string name = absl::AsciiStrToUpper(platform);
  if (name == "ROCM" || name == "SYCL") {
    return std::optional<AddI32Setup>();
  }
  se::StreamExecutor* executor = GpuExecutor();
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<se::Stream> stream,
                   executor->CreateStream());
  ABSL_ASSIGN_OR_RETURN(AddI32Buffers buffers,
                   AllocateAddI32Buffers(executor, stream.get(), a, b));
  return std::optional<AddI32Setup>(AddI32Setup{executor, std::move(stream),
                                                buffers.a_dev, buffers.b_dev,
                                                buffers.c_dev});
}

// Helper: builds and initializes an AddI32 KernelThunk (c[0] = a[0] + b[0]).
// Returns {thunk, alloc_a, alloc_b, alloc_c}.
static absl::StatusOr<std::unique_ptr<KernelThunk>> MakeAddI32KernelThunk(
    const std::vector<BufferAllocation>& allocs, bool use_cluster_dim,
    std::vector<int64_t> zeroed_output_buffer_indices = {}) {
  emitters::KernelArgument arg_a(ShapeUtil::MakeShape(S32, {1}),
                                 BufferAllocation::Slice(&allocs[0], 0, 4));
  emitters::KernelArgument arg_b(ShapeUtil::MakeShape(S32, {1}),
                                 BufferAllocation::Slice(&allocs[1], 0, 4));
  emitters::KernelArgument arg_c(ShapeUtil::MakeShape(S32, {1}),
                                 BufferAllocation::Slice(&allocs[2], 0, 4));
  arg_a.set_written(false);
  arg_b.set_written(false);
  arg_c.set_written(true);

  return std::make_unique<KernelThunk>(
      Thunk::ThunkInfo(),
      /*kernel_name=*/"AddI32",
      /*kernel_arguments=*/emitters::KernelArguments({arg_a, arg_b, arg_c}),
      /*launch_dimensions=*/
      LaunchDimensions(se::BlockDim(1, 1, 1), se::ThreadDim(1, 1, 1)),
      /*cluster_dim=*/
      use_cluster_dim ? std::make_optional(se::ClusterDim(1, 1, 1))
                      : std::nullopt,
      /*shmem_bytes=*/0,
      /*tma_metadata=*/se::gpu::TmaMetadata(),
      /*devices_per_host=*/1,
      /*zeroed_output_buffer_indices=*/std::move(zeroed_output_buffer_indices));
}

// Mark input b as zeroed to observe the mechanism with c = a + b.
TEST(KernelThunkTest, ExecuteOnStreamZeroed) {
  ASSERT_OK_AND_ASSIGN(std::optional<AddI32Setup> setup, SetUpAddI32());
  if (!setup.has_value()) {
    GTEST_SKIP() << "AddI32 PTX kernel not supported on ROCm or oneAPI.";
  }
  auto& [executor, stream, a_dev, b_dev, c_dev] = *setup;

  std::vector<BufferAllocation> allocs = {
      BufferAllocation(/*index=*/0, /*size=*/4, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/4, /*color=*/0),
      BufferAllocation(/*index=*/2, /*size=*/4, /*color=*/0),
  };

  const bool use_cluster_dim = executor->GetDeviceDescription()
                                   .cuda_compute_capability()
                                   .IsAtLeastHopper();
  ASSERT_OK_AND_ASSIGN(auto thunk,
                       MakeAddI32KernelThunk(allocs, use_cluster_dim, {1}));

  // Initialize the thunk (loads kernel PTX).
  Thunk::ExecutableSource src;
  src.text = se::gpu::GetAddI32PtxKernelSpec().cuda_ptx_in_memory().value().ptx;
  se::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations buffer_allocations({a_dev, b_dev, c_dev}, 0, &allocator);
  CommandBufferRequests command_buffer_requests;
  Thunk::InitializeParams init_params(&command_buffer_requests);
  init_params.executor = executor;
  init_params.src = src;
  init_params.buffer_allocations = &buffer_allocations;
  init_params.stream = stream.get();
  ASSERT_OK(thunk->Initialize(init_params));

  ServiceExecutableRunOptions run_options;
  run_options.mutable_run_options()->set_stream(stream.get());
  auto execute_params = Thunk::ExecuteParams::Create(
      run_options, buffer_allocations, stream.get(), nullptr, nullptr, nullptr,
      nullptr, {});

  ASSERT_OK(thunk->ExecuteOnStream(execute_params));

  int32_t result = 0;
  ASSERT_OK(stream->Memcpy(&result, c_dev, sizeof(int32_t)));
  ASSERT_OK(stream->BlockHostUntilDone());
  RecordProperty("observed_c", result);
  EXPECT_EQ(result, 1);  // b must be zeroed before adding.
  int32_t observed_b = -1;
  ASSERT_OK(stream->Memcpy(&observed_b, b_dev, sizeof(int32_t)));
  ASSERT_OK(stream->BlockHostUntilDone());
  RecordProperty("observed_b", observed_b);
  EXPECT_EQ(observed_b, 0);
}

TEST(KernelThunkTest, RecordCommandBufferZeroed) {
  ASSERT_OK_AND_ASSIGN(std::optional<AddI32Setup> setup, SetUpAddI32());
  if (!setup.has_value()) {
    GTEST_SKIP() << "AddI32 PTX kernel not supported on ROCm or oneAPI.";
  }
  auto& [executor, stream, a_dev, b_dev, c_dev] = *setup;

  std::vector<BufferAllocation> allocs = {
      BufferAllocation(/*index=*/0, /*size=*/4, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/4, /*color=*/0),
      BufferAllocation(/*index=*/2, /*size=*/4, /*color=*/0),
  };

  const bool use_cluster_dim = executor->GetDeviceDescription()
                                   .cuda_compute_capability()
                                   .IsAtLeastHopper();
  ASSERT_OK_AND_ASSIGN(auto thunk,
                       MakeAddI32KernelThunk(allocs, use_cluster_dim, {1}));

  // Initialize the thunk (loads kernel PTX).
  Thunk::ExecutableSource src;
  src.text = se::gpu::GetAddI32PtxKernelSpec().cuda_ptx_in_memory().value().ptx;
  se::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations buffer_allocations({a_dev, b_dev, c_dev}, 0, &allocator);
  CommandBufferRequests command_buffer_requests;
  Thunk::InitializeParams init_params(&command_buffer_requests);
  init_params.executor = executor;
  init_params.src = src;
  init_params.buffer_allocations = &buffer_allocations;
  init_params.stream = stream.get();
  ASSERT_OK(thunk->Initialize(init_params));

  ServiceExecutableRunOptions run_options;
  run_options.mutable_run_options()->set_stream(stream.get());
  auto execute_params = Thunk::ExecuteParams::Create(
      run_options, buffer_allocations, stream.get(), nullptr, nullptr, nullptr,
      nullptr, {});

  CommandStateManager state;
  Command::RecordParams record_params = {state};

  ASSERT_OK_AND_ASSIGN(
      auto command_buffer,
      executor->CreateCommandBuffer(se::CommandBuffer::Mode::kPrimary));
  ASSERT_OK_AND_ASSIGN(const se::CommandBuffer::Command* cmd,
                       thunk->Record(execute_params, record_params,
                                     Command::RecordCreate{/*dependencies=*/{}},
                                     command_buffer.get()));
  ASSERT_NE(cmd, nullptr);
  ASSERT_OK(command_buffer->Finalize());
  ASSERT_OK(command_buffer->Submit(stream.get()));

  int32_t result = 0;
  ASSERT_OK(stream->Memcpy(&result, c_dev, sizeof(int32_t)));
  ASSERT_OK(stream->BlockHostUntilDone());
  RecordProperty("observed_c", result);
  EXPECT_EQ(result, 1);  // b must be zeroed before adding.
  int32_t observed_b = -1;
  ASSERT_OK(stream->Memcpy(&observed_b, b_dev, sizeof(int32_t)));
  ASSERT_OK(stream->BlockHostUntilDone());
  RecordProperty("observed_b", observed_b);
  EXPECT_EQ(observed_b, 0);
}

TEST(KernelThunkTest, RecordCommandBufferUpdateZeroed) {
  ASSERT_OK_AND_ASSIGN(std::optional<AddI32Setup> setup, SetUpAddI32());
  if (!setup.has_value()) {
    GTEST_SKIP() << "AddI32 PTX kernel not supported on ROCm or oneAPI.";
  }
  auto& [executor, stream, a_dev, b_dev, c_dev] = *setup;

  // Written back into b before each later submission.
  int32_t val_b = 2;

  std::vector<BufferAllocation> allocs = {
      BufferAllocation(/*index=*/0, /*size=*/4, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/4, /*color=*/0),
      BufferAllocation(/*index=*/2, /*size=*/4, /*color=*/0),
  };

  const bool use_cluster_dim = executor->GetDeviceDescription()
                                   .cuda_compute_capability()
                                   .IsAtLeastHopper();
  ASSERT_OK_AND_ASSIGN(auto thunk,
                       MakeAddI32KernelThunk(allocs, use_cluster_dim, {1}));

  Thunk::ExecutableSource src;
  src.text = se::gpu::GetAddI32PtxKernelSpec().cuda_ptx_in_memory().value().ptx;
  se::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations buffer_allocations({a_dev, b_dev, c_dev}, 0, &allocator);
  CommandBufferRequests command_buffer_requests;
  Thunk::InitializeParams init_params(&command_buffer_requests);
  init_params.executor = executor;
  init_params.src = src;
  init_params.buffer_allocations = &buffer_allocations;
  init_params.stream = stream.get();
  ASSERT_OK(thunk->Initialize(init_params));

  ServiceExecutableRunOptions run_options;
  run_options.mutable_run_options()->set_stream(stream.get());
  auto execute_params = Thunk::ExecuteParams::Create(
      run_options, buffer_allocations, stream.get(), nullptr, nullptr, nullptr,
      nullptr, {});

  CommandStateManager state;
  Command::RecordParams record_params = {state};

  // First recording: RecordCreate.
  ASSERT_OK_AND_ASSIGN(
      auto command_buffer,
      executor->CreateCommandBuffer(se::CommandBuffer::Mode::kPrimary));
  ASSERT_OK_AND_ASSIGN(const se::CommandBuffer::Command* cmd,
                       thunk->Record(execute_params, record_params,
                                     Command::RecordCreate{/*dependencies=*/{}},
                                     command_buffer.get()));
  ASSERT_NE(cmd, nullptr);
  ASSERT_OK(command_buffer->Finalize());
  ASSERT_OK(command_buffer->Submit(stream.get()));

  int32_t result = 0;
  ASSERT_OK(stream->Memcpy(&result, c_dev, sizeof(int32_t)));
  ASSERT_OK(stream->BlockHostUntilDone());
  RecordProperty("observed_c_RecordCreate", result);
  EXPECT_EQ(result, 1) << "RecordCreate";

  // Replay without updating, after dirtying b again.
  ASSERT_OK(stream->Memcpy(&b_dev, &val_b, sizeof(int32_t)));
  ASSERT_OK(command_buffer->Submit(stream.get()));
  ASSERT_OK(stream->Memcpy(&result, c_dev, sizeof(int32_t)));
  ASSERT_OK(stream->BlockHostUntilDone());
  RecordProperty("observed_c_Replay", result);
  EXPECT_EQ(result, 1) << "Replay";

  // Update and re-submit with same allocations, after dirtying b again.
  ASSERT_OK(stream->Memcpy(&b_dev, &val_b, sizeof(int32_t)));
  ASSERT_OK(command_buffer->Update());
  ASSERT_OK_AND_ASSIGN(
      const se::CommandBuffer::Command* updated_cmd,
      thunk->Record(execute_params, record_params, Command::RecordUpdate{cmd},
                    command_buffer.get()));
  EXPECT_EQ(updated_cmd, cmd);  // same command node is reused
  ASSERT_OK(command_buffer->Finalize());
  ASSERT_OK(command_buffer->Submit(stream.get()));

  result = 0;
  ASSERT_OK(stream->Memcpy(&result, c_dev, sizeof(int32_t)));
  ASSERT_OK(stream->BlockHostUntilDone());
  RecordProperty("observed_c_RecordUpdate", result);
  EXPECT_EQ(result, 1) << "RecordUpdate";
}

// One thunk recorded twice into the SAME command buffer, then both recordings
// updated with buffers that have moved.
//
// This is the shape an unrolled while loop produces: the loop body records the
// same commands into one child command buffer once per trip, each time creating
// fresh nodes. Command state is keyed by (command, command buffer) only -- the
// record id is not part of the key -- so both recordings share a single state
// object, and the zeroing memsets have to be distinguished by the node they
// were recorded alongside.
//
// If the memsets were tracked as one flat list instead, the create pass would
// overwrite it and both updates would repoint the second recording's memset,
// leaving the first aimed at its create-time address. Here that shows up as the
// first recording reading an unzeroed b (c == 3); in production it is a memset
// into whatever now occupies that address.
TEST(KernelThunkTest, RecordCommandBufferTwoRecordingsZeroed) {
  ASSERT_OK_AND_ASSIGN(std::optional<AddI32Setup> setup, SetUpAddI32());
  if (!setup.has_value()) {
    GTEST_SKIP() << "AddI32 PTX kernel not supported on ROCm or oneAPI.";
  }
  // Plain references rather than a structured binding: `params_for` below
  // captures `stream`, and C++17 lambdas cannot capture structured bindings.
  se::StreamExecutor* executor = setup->executor;
  std::unique_ptr<se::Stream>& stream = setup->stream;

  // One operand set per recording, plus a fresh b buffer for each so that the
  // update genuinely moves the memset destinations.
  AddI32Buffers first{setup->a_dev, setup->b_dev, setup->c_dev};
  ASSERT_OK_AND_ASSIGN(AddI32Buffers second,
                       AllocateAddI32Buffers(executor, stream.get()));
  se::DeviceAddress<int32_t> b0_new = executor->AllocateArray<int32_t>(1, 0);
  se::DeviceAddress<int32_t> b1_new = executor->AllocateArray<int32_t>(1, 0);
  int32_t val_b = 2;
  for (se::DeviceAddress<int32_t>* b : {&b0_new, &b1_new}) {
    ASSERT_OK(stream->Memcpy(b, &val_b, sizeof(int32_t)));
  }

  std::vector<BufferAllocation> allocs = {
      BufferAllocation(/*index=*/0, /*size=*/4, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/4, /*color=*/0),
      BufferAllocation(/*index=*/2, /*size=*/4, /*color=*/0),
  };

  const bool use_cluster_dim = executor->GetDeviceDescription()
                                   .cuda_compute_capability()
                                   .IsAtLeastHopper();
  ASSERT_OK_AND_ASSIGN(auto thunk,
                       MakeAddI32KernelThunk(allocs, use_cluster_dim, {1}));

  Thunk::ExecutableSource src;
  src.text = se::gpu::GetAddI32PtxKernelSpec().cuda_ptx_in_memory().value().ptx;

  se::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations allocations_0({first.a_dev, first.b_dev, first.c_dev}, 0,
                                  &allocator);
  BufferAllocations allocations_1({second.a_dev, second.b_dev, second.c_dev}, 0,
                                  &allocator);
  BufferAllocations allocations_0_new({first.a_dev, b0_new, first.c_dev}, 0,
                                      &allocator);
  BufferAllocations allocations_1_new({second.a_dev, b1_new, second.c_dev}, 0,
                                      &allocator);

  CommandBufferRequests command_buffer_requests;
  Thunk::InitializeParams init_params(&command_buffer_requests);
  init_params.executor = executor;
  init_params.src = src;
  init_params.buffer_allocations = &allocations_0;
  init_params.stream = stream.get();
  ASSERT_OK(thunk->Initialize(init_params));

  ServiceExecutableRunOptions run_options;
  run_options.mutable_run_options()->set_stream(stream.get());
  auto params_for = [&](const BufferAllocations& allocations) {
    return Thunk::ExecuteParams::Create(run_options, allocations, stream.get(),
                                        nullptr, nullptr, nullptr, nullptr, {});
  };

  CommandStateManager state;
  Command::RecordParams record_params = {state};

  ASSERT_OK_AND_ASSIGN(
      auto command_buffer,
      executor->CreateCommandBuffer(se::CommandBuffer::Mode::kPrimary));

  ASSERT_OK_AND_ASSIGN(const se::CommandBuffer::Command* cmd_0,
                       thunk->Record(params_for(allocations_0), record_params,
                                     Command::RecordCreate{/*dependencies=*/{}},
                                     command_buffer.get()));
  ASSERT_OK_AND_ASSIGN(const se::CommandBuffer::Command* cmd_1,
                       thunk->Record(params_for(allocations_1), record_params,
                                     Command::RecordCreate{/*dependencies=*/{}},
                                     command_buffer.get()));
  ASSERT_NE(cmd_0, nullptr);
  ASSERT_NE(cmd_1, nullptr);
  ASSERT_NE(cmd_0, cmd_1);

  ASSERT_OK(command_buffer->Finalize());
  ASSERT_OK(command_buffer->Submit(stream.get()));

  int32_t result_0 = 0, result_1 = 0;
  ASSERT_OK(stream->Memcpy(&result_0, first.c_dev, sizeof(int32_t)));
  ASSERT_OK(stream->Memcpy(&result_1, second.c_dev, sizeof(int32_t)));
  ASSERT_OK(stream->BlockHostUntilDone());
  EXPECT_EQ(result_0, 1) << "first recording, RecordCreate";
  EXPECT_EQ(result_1, 1) << "second recording, RecordCreate";

  // Point both recordings at their new b buffers. Each update must reach the
  // memset recorded alongside its own node.
  ASSERT_OK(command_buffer->Update());
  ASSERT_OK_AND_ASSIGN(
      const se::CommandBuffer::Command* updated_0,
      thunk->Record(params_for(allocations_0_new), record_params,
                    Command::RecordUpdate{cmd_0}, command_buffer.get()));
  ASSERT_OK_AND_ASSIGN(
      const se::CommandBuffer::Command* updated_1,
      thunk->Record(params_for(allocations_1_new), record_params,
                    Command::RecordUpdate{cmd_1}, command_buffer.get()));
  EXPECT_EQ(updated_0, cmd_0);
  EXPECT_EQ(updated_1, cmd_1);
  ASSERT_OK(command_buffer->Finalize());

  for (se::DeviceAddress<int32_t>* c : {&first.c_dev, &second.c_dev}) {
    ASSERT_OK(stream->MemZero(c, sizeof(int32_t)));
  }
  ASSERT_OK(command_buffer->Submit(stream.get()));

  result_0 = 0;
  result_1 = 0;
  ASSERT_OK(stream->Memcpy(&result_0, first.c_dev, sizeof(int32_t)));
  ASSERT_OK(stream->Memcpy(&result_1, second.c_dev, sizeof(int32_t)));
  ASSERT_OK(stream->BlockHostUntilDone());
  EXPECT_EQ(result_0, 1) << "first recording, RecordUpdate";
  EXPECT_EQ(result_1, 1) << "second recording, RecordUpdate";
}

TEST(KernelThunkTest, RecordCommandBuffer) {
  ASSERT_OK_AND_ASSIGN(std::optional<AddI32Setup> setup, SetUpAddI32());
  if (!setup.has_value()) {
    GTEST_SKIP() << "AddI32 PTX kernel not supported on ROCm or oneAPI.";
  }
  auto& [executor, stream, a_dev, b_dev, c_dev] = *setup;

  std::vector<BufferAllocation> allocs = {
      BufferAllocation(/*index=*/0, /*size=*/4, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/4, /*color=*/0),
      BufferAllocation(/*index=*/2, /*size=*/4, /*color=*/0),
  };

  const bool use_cluster_dim = executor->GetDeviceDescription()
                                   .cuda_compute_capability()
                                   .IsAtLeastHopper();
  ASSERT_OK_AND_ASSIGN(auto thunk,
                       MakeAddI32KernelThunk(allocs, use_cluster_dim));

  // Initialize the thunk (loads kernel PTX).
  Thunk::ExecutableSource src;
  src.text = se::gpu::GetAddI32PtxKernelSpec().cuda_ptx_in_memory().value().ptx;
  se::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations buffer_allocations({a_dev, b_dev, c_dev}, 0, &allocator);
  CommandBufferRequests command_buffer_requests;
  Thunk::InitializeParams init_params(&command_buffer_requests);
  init_params.executor = executor;
  init_params.src = src;
  init_params.buffer_allocations = &buffer_allocations;
  init_params.stream = stream.get();
  ASSERT_OK(thunk->Initialize(init_params));

  ServiceExecutableRunOptions run_options;
  run_options.mutable_run_options()->set_stream(stream.get());
  auto execute_params = Thunk::ExecuteParams::Create(
      run_options, buffer_allocations, stream.get(), nullptr, nullptr, nullptr,
      nullptr, {});

  CommandStateManager state;
  Command::RecordParams record_params = {state};

  ASSERT_OK_AND_ASSIGN(
      auto command_buffer,
      executor->CreateCommandBuffer(se::CommandBuffer::Mode::kPrimary));
  ASSERT_OK_AND_ASSIGN(const se::CommandBuffer::Command* cmd,
                       thunk->Record(execute_params, record_params,
                                     Command::RecordCreate{/*dependencies=*/{}},
                                     command_buffer.get()));
  ASSERT_NE(cmd, nullptr);
  ASSERT_OK(command_buffer->Finalize());
  ASSERT_OK(command_buffer->Submit(stream.get()));

  int32_t result = 0;
  ASSERT_OK(stream->Memcpy(&result, c_dev, sizeof(int32_t)));
  EXPECT_EQ(result, 3);  // 1 + 2 = 3
}

TEST(KernelThunkTest, RecordCommandBufferUpdate) {
  ASSERT_OK_AND_ASSIGN(std::optional<AddI32Setup> setup,
                       SetUpAddI32(/*a=*/10, /*b=*/20));
  if (!setup.has_value()) {
    GTEST_SKIP() << "AddI32 PTX kernel not supported on ROCm or oneAPI.";
  }
  auto& [executor, stream, a_dev, b_dev, c_dev] = *setup;

  std::vector<BufferAllocation> allocs = {
      BufferAllocation(/*index=*/0, /*size=*/4, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/4, /*color=*/0),
      BufferAllocation(/*index=*/2, /*size=*/4, /*color=*/0),
  };

  const bool use_cluster_dim = executor->GetDeviceDescription()
                                   .cuda_compute_capability()
                                   .IsAtLeastHopper();
  ASSERT_OK_AND_ASSIGN(auto thunk,
                       MakeAddI32KernelThunk(allocs, use_cluster_dim));

  Thunk::ExecutableSource src;
  src.text = se::gpu::GetAddI32PtxKernelSpec().cuda_ptx_in_memory().value().ptx;
  se::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations buffer_allocations({a_dev, b_dev, c_dev}, 0, &allocator);
  CommandBufferRequests command_buffer_requests;
  Thunk::InitializeParams init_params(&command_buffer_requests);
  init_params.executor = executor;
  init_params.src = src;
  init_params.buffer_allocations = &buffer_allocations;
  init_params.stream = stream.get();
  ASSERT_OK(thunk->Initialize(init_params));

  ServiceExecutableRunOptions run_options;
  run_options.mutable_run_options()->set_stream(stream.get());
  auto execute_params = Thunk::ExecuteParams::Create(
      run_options, buffer_allocations, stream.get(), nullptr, nullptr, nullptr,
      nullptr, {});

  CommandStateManager state;
  Command::RecordParams record_params = {state};

  // First recording: RecordCreate.
  ASSERT_OK_AND_ASSIGN(
      auto command_buffer,
      executor->CreateCommandBuffer(se::CommandBuffer::Mode::kPrimary));
  ASSERT_OK_AND_ASSIGN(const se::CommandBuffer::Command* cmd,
                       thunk->Record(execute_params, record_params,
                                     Command::RecordCreate{/*dependencies=*/{}},
                                     command_buffer.get()));
  ASSERT_NE(cmd, nullptr);
  ASSERT_OK(command_buffer->Finalize());
  ASSERT_OK(command_buffer->Submit(stream.get()));

  int32_t result = 0;
  ASSERT_OK(stream->Memcpy(&result, c_dev, sizeof(int32_t)));
  EXPECT_EQ(result, 30);  // 10 + 20 = 30

  // Update and re-submit with same allocations.
  ASSERT_OK(command_buffer->Update());
  ASSERT_OK_AND_ASSIGN(
      const se::CommandBuffer::Command* updated_cmd,
      thunk->Record(execute_params, record_params, Command::RecordUpdate{cmd},
                    command_buffer.get()));
  EXPECT_EQ(updated_cmd, cmd);  // same command node is reused
  ASSERT_OK(command_buffer->Finalize());
  ASSERT_OK(command_buffer->Submit(stream.get()));

  result = 0;
  ASSERT_OK(stream->Memcpy(&result, c_dev, sizeof(int32_t)));
  EXPECT_EQ(result, 30);  // still 10 + 20 = 30
}

TEST(KernelThunkTest, RecordCommandBufferUpdateWithNewOutputBuffer) {
  ASSERT_OK_AND_ASSIGN(std::optional<AddI32Setup> setup, SetUpAddI32());
  if (!setup.has_value()) {
    GTEST_SKIP() << "AddI32 PTX kernel not supported on ROCm or oneAPI.";
  }
  auto& [executor, stream, a_dev, b_dev, c_dev] = *setup;

  // A second output buffer for the second run.
  se::DeviceAddress<int32_t> c2_dev = executor->AllocateArray<int32_t>(1, 0);
  ASSERT_OK(stream->MemZero(&c2_dev, sizeof(int32_t)));

  std::vector<BufferAllocation> allocs = {
      BufferAllocation(/*index=*/0, /*size=*/4, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/4, /*color=*/0),
      BufferAllocation(/*index=*/2, /*size=*/4, /*color=*/0),
  };
  const bool use_cluster_dim = executor->GetDeviceDescription()
                                   .cuda_compute_capability()
                                   .IsAtLeastHopper();
  ASSERT_OK_AND_ASSIGN(auto thunk,
                       MakeAddI32KernelThunk(allocs, use_cluster_dim));

  Thunk::ExecutableSource src;
  src.text = se::gpu::GetAddI32PtxKernelSpec().cuda_ptx_in_memory().value().ptx;
  se::StreamExecutorAddressAllocator allocator(executor);

  // Initialize and run with c_dev as output.
  BufferAllocations alloc1({a_dev, b_dev, c_dev}, 0, &allocator);
  CommandBufferRequests command_buffer_requests;
  Thunk::InitializeParams init_params(&command_buffer_requests);
  init_params.executor = executor;
  init_params.src = src;
  init_params.buffer_allocations = &alloc1;
  init_params.stream = stream.get();
  ASSERT_OK(thunk->Initialize(init_params));

  ServiceExecutableRunOptions run_options;
  run_options.mutable_run_options()->set_stream(stream.get());
  auto execute_params1 =
      Thunk::ExecuteParams::Create(run_options, alloc1, stream.get(), nullptr,
                                   nullptr, nullptr, nullptr, {});

  CommandStateManager state;
  Command::RecordParams record_params = {state};

  ASSERT_OK_AND_ASSIGN(
      auto command_buffer,
      executor->CreateCommandBuffer(se::CommandBuffer::Mode::kPrimary));
  ASSERT_OK_AND_ASSIGN(const se::CommandBuffer::Command* cmd,
                       thunk->Record(execute_params1, record_params,
                                     Command::RecordCreate{/*dependencies=*/{}},
                                     command_buffer.get()));
  ASSERT_NE(cmd, nullptr);
  ASSERT_OK(command_buffer->Finalize());
  ASSERT_OK(command_buffer->Submit(stream.get()));

  int32_t result = 0;
  ASSERT_OK(stream->Memcpy(&result, c_dev, sizeof(int32_t)));
  EXPECT_EQ(result, 3);  // 1 + 2 = 3

  // Update to use c2_dev as the output buffer.
  BufferAllocations alloc2({a_dev, b_dev, c2_dev}, 0, &allocator);
  auto execute_params2 =
      Thunk::ExecuteParams::Create(run_options, alloc2, stream.get(), nullptr,
                                   nullptr, nullptr, nullptr, {});

  ASSERT_OK(command_buffer->Update());
  ASSERT_OK_AND_ASSIGN(
      const se::CommandBuffer::Command* updated_cmd,
      thunk->Record(execute_params2, record_params, Command::RecordUpdate{cmd},
                    command_buffer.get()));
  EXPECT_EQ(updated_cmd, cmd);  // same command node reused
  ASSERT_OK(command_buffer->Finalize());
  ASSERT_OK(command_buffer->Submit(stream.get()));

  // c2_dev should now have the result; c_dev should be unchanged.
  result = 0;
  ASSERT_OK(stream->Memcpy(&result, c2_dev, sizeof(int32_t)));
  EXPECT_EQ(result, 3);  // 1 + 2 = 3 written into the new buffer

  result = 0;
  ASSERT_OK(stream->Memcpy(&result, c_dev, sizeof(int32_t)));
  EXPECT_EQ(result, 3);  // original buffer still holds first-run result
}

TEST(KernelThunkTest, RecordFailsWithoutInitialize) {
  ASSERT_OK_AND_ASSIGN(std::optional<AddI32Setup> setup, SetUpAddI32());
  if (!setup.has_value()) {
    GTEST_SKIP() << "AddI32 PTX kernel not supported on ROCm or oneAPI.";
  }
  auto& [executor, stream, a_dev, b_dev, c_dev] = *setup;

  std::vector<BufferAllocation> allocs = {
      BufferAllocation(/*index=*/0, /*size=*/4, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/4, /*color=*/0),
      BufferAllocation(/*index=*/2, /*size=*/4, /*color=*/0),
  };
  const bool use_cluster_dim = executor->GetDeviceDescription()
                                   .cuda_compute_capability()
                                   .IsAtLeastHopper();
  ASSERT_OK_AND_ASSIGN(auto thunk,
                       MakeAddI32KernelThunk(allocs, use_cluster_dim));
  // Intentionally skip Initialize().

  se::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations buffer_allocations({a_dev, b_dev, c_dev}, 0, &allocator);
  ServiceExecutableRunOptions run_options;
  run_options.mutable_run_options()->set_stream(stream.get());
  auto execute_params = Thunk::ExecuteParams::Create(
      run_options, buffer_allocations, stream.get(), nullptr, nullptr, nullptr,
      nullptr, {});

  CommandStateManager state;
  Command::RecordParams record_params = {state};

  ASSERT_OK_AND_ASSIGN(
      auto command_buffer,
      executor->CreateCommandBuffer(se::CommandBuffer::Mode::kPrimary));
  auto status = thunk->Record(execute_params, record_params,
                              Command::RecordCreate{/*dependencies=*/{}},
                              command_buffer.get());
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.status().message(),
              ::testing::HasSubstr("Kernel not loaded"));
}

TEST(KernelThunkTest, ExecuteFailsWithoutInitialize) {
  ASSERT_OK_AND_ASSIGN(std::optional<AddI32Setup> setup, SetUpAddI32());
  if (!setup.has_value()) {
    GTEST_SKIP() << "AddI32 PTX kernel not supported on ROCm or oneAPI.";
  }
  auto& [executor, stream, a_dev, b_dev, c_dev] = *setup;

  int32_t sentinel = 99;
  ASSERT_OK(stream->Memcpy(&c_dev, &sentinel, sizeof(int32_t)));

  std::vector<BufferAllocation> allocs = {
      BufferAllocation(/*index=*/0, /*size=*/4, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/4, /*color=*/0),
      BufferAllocation(/*index=*/2, /*size=*/4, /*color=*/0),
  };
  emitters::KernelArgument arg_a(ShapeUtil::MakeShape(S32, {1}),
                                 BufferAllocation::Slice(&allocs[0], 0, 4));
  emitters::KernelArgument arg_b(ShapeUtil::MakeShape(S32, {1}),
                                 BufferAllocation::Slice(&allocs[1], 0, 4));
  emitters::KernelArgument arg_c(ShapeUtil::MakeShape(S32, {1}),
                                 BufferAllocation::Slice(&allocs[2], 0, 4));
  arg_a.set_written(false);
  arg_b.set_written(false);
  arg_c.set_written(true);

  KernelThunk thunk(
      Thunk::ThunkInfo(), /*kernel_name=*/"AddI32",
      emitters::KernelArguments({arg_a, arg_b, arg_c}),
      LaunchDimensions(se::BlockDim(1, 1, 1), se::ThreadDim(1, 1, 1)),
      /*cluster_dim=*/std::nullopt, /*shmem_bytes=*/0, se::gpu::TmaMetadata(),
      /*devices_per_host=*/1, /*zeroed_output_buffer_indices=*/{2});
  // Intentionally skip Initialize().

  se::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations buffer_allocations({a_dev, b_dev, c_dev}, 0, &allocator);
  ServiceExecutableRunOptions run_options;
  run_options.mutable_run_options()->set_stream(stream.get());
  auto execute_params = Thunk::ExecuteParams::Create(
      run_options, buffer_allocations, stream.get(), nullptr, nullptr, nullptr,
      nullptr, {});

  auto status = thunk.ExecuteOnStream(execute_params);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(), ::testing::HasSubstr("Kernel not loaded"));

  int32_t observed = 0;
  ASSERT_OK(stream->Memcpy(&observed, c_dev, sizeof(int32_t)));
  EXPECT_EQ(observed, sentinel);
}

}  // namespace
}  // namespace xla::gpu
