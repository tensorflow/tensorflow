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

#include "xla/backends/gpu/runtime/custom_kernel_thunk.h"

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "xla/backends/gpu/codegen/kernels/custom_kernel.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace xla::gpu {
namespace {
using absl_testing::IsOkAndHolds;
using ::testing::Field;
using ::testing::Optional;
using tsl::proto_testing::EqualsProto;
using tsl::proto_testing::ParseTextProtoOrDie;

TEST(CustomKernelThunkTest, BufferUsesReturnsCorrectBuffers) {
  Shape arg_shape = ShapeUtil::MakeShape(F32, {512});
  CustomKernel kernel(
      /*name=*/"",
      se::KernelLoaderSpec::CreateCudaPtxInMemorySpec(
          /*ptx=*/"", /*kernel_name=*/"", /*arity=*/0),
      se::BlockDim(), se::ThreadDim(), /*shared_memory_bytes=*/0);
  BufferAllocation alloc(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation::Slice slice0(&alloc, /*offset=*/0, /*size=*/512);
  BufferAllocation::Slice slice1(&alloc, /*offset=*/512, /*size=*/512);
  emitters::KernelArgument arg0(arg_shape, slice0);
  emitters::KernelArgument arg1(arg_shape, slice1);
  arg0.set_written(false);
  arg1.set_written(true);
  emitters::KernelArguments kernel_arguments({arg0, arg1});
  CustomKernelThunk thunk(Thunk::ThunkInfo{}, kernel, kernel_arguments);

  Thunk::BufferUses buffers = thunk.buffer_uses();

  ASSERT_THAT(buffers, testing::UnorderedElementsAre(
                           BufferUse::Read(slice0, arg_shape),
                           BufferUse::Write(slice1, arg_shape)));
}

TEST(CustomKernelThunkTest, BufferUsesReturnsBuffersInConsistentOrder) {
  CustomKernel kernel(
      /*name=*/"",
      se::KernelLoaderSpec::CreateCudaPtxInMemorySpec(
          /*ptx=*/"", /*kernel_name=*/"", /*arity=*/0),
      se::BlockDim(), se::ThreadDim(), /*shared_memory_bytes=*/0);
  BufferAllocation alloc(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation::Slice slice0(&alloc, /*offset=*/0, /*size=*/512);
  BufferAllocation::Slice slice1(&alloc, /*offset=*/512, /*size=*/512);
  emitters::KernelArgument arg0(ShapeUtil::MakeShape(F32, {512}), slice0);
  emitters::KernelArgument arg1(ShapeUtil::MakeShape(F32, {512}), slice1);
  arg0.set_written(false);
  arg1.set_written(true);
  emitters::KernelArguments kernel_arguments({arg0, arg1});
  CustomKernelThunk thunk(Thunk::ThunkInfo{}, kernel, kernel_arguments);

  Thunk::BufferUses buffers1 = thunk.buffer_uses();
  Thunk::BufferUses buffers2 = thunk.buffer_uses();

  ASSERT_THAT(buffers1, testing::ContainerEq(buffers2));
}

TEST(CustomKernelThunkTest, ToProto) {
  CustomKernel kernel("name",
                      se::KernelLoaderSpec::CreateCudaPtxInMemorySpec(
                          "PTX", "kernel_name", /*arity=*/1),
                      se::BlockDim(3, 2, 1), se::ThreadDim(4, 5, 6),
                      /*shared_memory_bytes=*/42);

  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = "profile_annotation";
  thunk_info.execution_stream_id = 7;
  thunk_info.thunk_id = 42;

  BufferAllocation alloc(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation::Slice slice0(&alloc, /*offset=*/0, /*size=*/512);
  emitters::KernelArgument arg0(ShapeUtil::MakeShape(F32, {512}), slice0);
  arg0.set_written(true);
  emitters::KernelArguments kernel_arguments({arg0});
  CustomKernelThunk thunk(thunk_info, kernel, kernel_arguments);

  EXPECT_THAT(
      thunk.ToProto(), IsOkAndHolds(EqualsProto(R"pb(
        thunk_info {
          profile_annotation: "profile_annotation"
          execution_stream_id: 7
          thunk_id: 42
        }
        custom_kernel_thunk {
          custom_kernel {
            name: "name"
            kernel_spec {
              kernel_name: "kernel_name"
              ptx { data: "PTX" }
              arity: 1
            }
            block_dims { coordinates { x: 3, y: 2, z: 1 } }
            thread_dims { coordinates { x: 4, y: 5, z: 6 } }
            shared_memory_bytes: 42
          }
          args {
            slice { buffer_allocation_index: 0, offset: 0, size: 512 }
            shape {
              element_type: F32
              dimensions: 512
              layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
              is_dynamic_dimension: false
            }
          }
          written: true
        }
      )pb")));
}

TEST(CustomKernelThunkTest, FromProto) {
  CustomKernelThunkProto proto = ParseTextProtoOrDie<CustomKernelThunkProto>(
      R"pb(
        custom_kernel {
          name: "test_kernel"
          kernel_spec {
            ptx { data: "PTX" }
            arity: 1
          }
          block_dims { coordinates { x: 1, y: 1, z: 1 } }
          thread_dims { coordinates { x: 1, y: 1, z: 1 } }
          shared_memory_bytes: 42
        }
        args {
          slice { buffer_allocation_index: 0, offset: 0, size: 1024 }
          shape {
            element_type: U8,
            dimensions: 1024,
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
        }
        written: true
      )pb");

  std::vector<BufferAllocation> buffer_allocations;
  buffer_allocations.emplace_back(/*index=*/0, /*size=*/1024, /*color=*/0);

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<CustomKernelThunk> thunk,
                          CustomKernelThunk::FromProto(
                              Thunk::ThunkInfo{}, proto, buffer_allocations));

  EXPECT_THAT(thunk->custom_kernel().name(), "test_kernel");
  EXPECT_THAT(thunk->arguments(),
              testing::ElementsAre(ShapedSlice{
                  BufferAllocation::Slice(&buffer_allocations[0], /*offset=*/0,
                                          /*size=*/1024),
                  ShapeUtil::MakeShape(U8, {1024})}));
  EXPECT_THAT(thunk->written(), testing::ElementsAre(true));
  EXPECT_THAT(thunk->custom_kernel().kernel_spec().cuda_ptx_in_memory(),
              Optional(Field(&se::CudaPtxInMemory::ptx, "PTX")));
}

}  // namespace
}  // namespace xla::gpu
