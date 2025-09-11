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

#include "xla/backends/gpu/runtime/select_k_thunk.h"

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/literal_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace xla::gpu {
namespace {

using ::tsl::proto_testing::EqualsProto;

TEST(SelectKThunkTest, ToProto) {
  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = "profile_annotation";
  thunk_info.execution_stream_id = 123;

  BufferAllocation alloc0(/*index=*/0, /*size=*/20, /*color=*/0);
  BufferAllocation::Slice slice0(&alloc0, /*offset=*/0, /*size=*/20);

  BufferAllocation alloc1(/*index=*/1, /*size=*/12, /*color=*/0);
  BufferAllocation::Slice slice1(&alloc1, /*offset=*/0, /*size=*/12);

  BufferAllocation alloc2(/*index=*/2, /*size=*/12, /*color=*/0);
  BufferAllocation::Slice slice2(&alloc2, /*offset=*/0, /*size=*/12);

  emitters::KernelArgument arg0(ShapeUtil::MakeShape(F32, {1, 5}), slice0);
  emitters::KernelArgument arg1(ShapeUtil::MakeShape(F32, {1, 3}), slice1);
  emitters::KernelArgument arg2(ShapeUtil::MakeShape(U32, {1, 3}), slice2);
  arg0.set_written(false);
  arg1.set_written(true);
  arg2.set_written(true);

  emitters::KernelArguments kernel_arguments({arg0, arg1, arg2});

  auto c1 = HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<float>({{.125f, 0.875f, .5f, .25f, 0.75f}}));
  auto topKInst = HloInstruction::CreateCustomCall(
      ShapeUtil::MakeShape(F32, {1, 5}), {c1.get()}, "__gpu$TopK");

  SelectKThunk thunk(topKInst.get(), 1, 5, 3, F32, kernel_arguments);
  TF_ASSERT_OK_AND_ASSIGN(ThunkProto proto, thunk.ToProto());
  EXPECT_THAT(proto, EqualsProto(R"pb(
                thunk_info { profile_annotation: "custom-call" }
                select_k_thunk {}
              )pb"));
}

}  // namespace
}  // namespace xla::gpu
