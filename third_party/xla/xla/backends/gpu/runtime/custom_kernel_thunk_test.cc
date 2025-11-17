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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/kernels/custom_kernel.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla::gpu {
namespace {

TEST(CustomKernelThunkTest, BufferUsesReturnsCorrectBuffers) {
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

  Thunk::BufferUses buffers = thunk.buffer_uses();

  ASSERT_THAT(buffers, testing::UnorderedElementsAre(BufferUse::Read(slice0),
                                                     BufferUse::Write(slice1)));
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

}  // namespace
}  // namespace xla::gpu
