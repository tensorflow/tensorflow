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
#include "xla/backends/gpu/runtime/thunk_buffer_debug_saver_inserter.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/runtime/custom_call_thunk.h"
#include "xla/backends/gpu/runtime/runtime_intrinsics.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/backends/gpu/runtime/thunk_proto_deserialization.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/util.h"

namespace xla::gpu {
namespace {

using ThunkBufferDebugSaverInserterTest = HloTestBase;
using ::testing::ElementsAre;
using ::testing::Eq;

MATCHER_P(ThunkKindIs, kind, "") {
  return ExplainMatchResult(Eq(kind), arg->kind(), result_listener);
}

MATCHER_P(IsCustomCallThunkWithTargetName, target_name, "") {
  return ExplainMatchResult(Eq(Thunk::Kind::kCustomCall), arg->kind(),
                            result_listener) &&
         ExplainMatchResult(
             Eq(target_name),
             static_cast<const CustomCallThunk&>(*arg).target_name(),
             result_listener);
}
MATCHER_P(IsSequentialThunkWith, thunk_matcher, "") {
  return ExplainMatchResult(Eq(Thunk::Kind::kSequential), arg->kind(),
                            result_listener) &&
         ExplainMatchResult(thunk_matcher,
                            static_cast<const SequentialThunk&>(*arg).thunks(),
                            result_listener);
}

absl::StatusOr<std::unique_ptr<SequentialThunk>> CloneRootThunk(
    const GpuExecutable* gpu_exec, std::vector<BufferAllocation>* allocations) {
  allocations->reserve(gpu_exec->GetAllocations().size());
  for (const BufferAllocation* alloc : gpu_exec->GetAllocations()) {
    allocations->emplace_back(BufferAllocation::FromProto(alloc->ToProto()));
  }
  TF_ASSIGN_OR_RETURN(ThunkProto proto, gpu_exec->GetThunk().ToProto());
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<Thunk> root_thunk,
      DeserializeThunkProto(proto, *allocations, &gpu_exec->module(), "GPU"));
  return unique_ptr_down_cast<SequentialThunk>(std::move(root_thunk));
}

TEST_F(ThunkBufferDebugSaverInserterTest, Insert) {
  std::string hlo = R"hlo(
HloModule m

ENTRY entry {
  p0 = f32[8,128] parameter(0)
  five_f = f32[] constant(5.0)
  fives_f = f32[8,128] broadcast(five_f), dimensions={}

  inst1 = f32[8,128] multiply(p0, fives_f)
  ROOT res = f32[8,128] add(inst1, p0)
})hlo";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> fused_module,
                          ParseAndReturnVerifiedModule(hlo));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<OpaqueExecutable> wrapped_exec,
                          CreateExecutable(fused_module->Clone(), false));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Executable> exec,
                          test_runner_as_hlo_runner().ExecutableFromWrapped(
                              std::move(wrapped_exec)));
  GpuExecutable* gpu_exec = dynamic_cast<GpuExecutable*>(exec.get());

  std::vector<BufferAllocation> allocations;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<SequentialThunk> root_thunk,
                          CloneRootThunk(gpu_exec, &allocations));

  absl::flat_hash_set<const Thunk*> before;
  root_thunk->ForAllThunks([&](const Thunk* thunk) { before.insert(thunk); });

  DebugOptions debug_options;
  debug_options.set_xla_gpu_experimental_debug_buffer_saver_directory(
      "/tmp/123");
  debug_options.mutable_xla_gpu_experimental_thunk_buffer_debug_filter()
      ->add_profile_annotation_regexes("wrapped_add");

  TF_EXPECT_OK(RunDebugSaverInserter(root_thunk.get(), debug_options,
                                     fused_module.get()));

  std::vector<const Thunk*> after;
  root_thunk->ForAllThunks([&](const Thunk* thunk) { after.push_back(thunk); });

  EXPECT_EQ(before.size() + 2, after.size());
  std::vector<const Thunk*> added;
  for (const Thunk* thunk : after) {
    if (before.contains(thunk)) {
      continue;
    }
    added.push_back(thunk);
  }

  EXPECT_THAT(added, ElementsAre(IsSequentialThunkWith(ElementsAre(
                                     ThunkKindIs(Thunk::Kind::kKernel),
                                     IsCustomCallThunkWithTargetName(
                                         kXlaGpuAppendToFileCustomCallTag))),
                                 IsCustomCallThunkWithTargetName(
                                     kXlaGpuAppendToFileCustomCallTag)));
}

}  // namespace
}  // namespace xla::gpu
