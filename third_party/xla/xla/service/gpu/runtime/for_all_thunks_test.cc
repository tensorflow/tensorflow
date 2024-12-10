/* Copyright 2024 The OpenXLA Authors.

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

#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/runtime/command_buffer_cmd.h"
#include "xla/service/gpu/runtime/command_buffer_thunk.h"
#include "xla/service/gpu/runtime/conditional_thunk.h"
#include "xla/service/gpu/runtime/dynamic_slice_thunk.h"
#include "xla/service/gpu/runtime/sequential_thunk.h"
#include "xla/service/gpu/runtime/thunk.h"
#include "xla/service/gpu/runtime/while_thunk.h"

namespace xla::gpu {
namespace {

using ::testing::IsSupersetOf;
using ::testing::UnorderedElementsAre;

// Invokes `ForAllThunks` on the `root` and returns a `vector` containing all
// iterated `Thunks`.
std::vector<const Thunk*> GetAllThunks(Thunk* root) {
  std::vector<const Thunk*> thunks;
  root->ForAllThunks([&](const Thunk* thunk) { thunks.push_back(thunk); });
  return thunks;
}

// A dummy `Thunk` that does nothing.
struct DummyThunk : public Thunk {
  DummyThunk() : Thunk(Thunk::Kind::kGemm, Thunk::ThunkInfo()) {}
  absl::Status ExecuteOnStream(const ExecuteParams& params) override {
    return absl::OkStatus();
  }
};

TEST(ForAllThunksTest, SingleThunk) {
  DummyThunk thunk;
  EXPECT_THAT(GetAllThunks(&thunk), UnorderedElementsAre(&thunk));
}

TEST(ForAllThunksTest, DynamicSliceThunk) {
  auto thunk = std::make_unique<DummyThunk>();
  Thunk* thunk_ptr = thunk.get();

  auto thunk_sequence = std::make_unique<ThunkSequence>();
  thunk_sequence->push_back(std::move(thunk));

  DynamicSliceThunk dynamic_slice_thunk(
      Thunk::ThunkInfo(), std::move(thunk_sequence), {}, {}, {}, {}, {}, {});
  EXPECT_THAT(GetAllThunks(&dynamic_slice_thunk),
              // `DynamicSliceThunk` wraps the `embedded_thunk` in a
              // `SequentialThunk`, which is why iterate over more than the
              // two expected `Thunks`.
              IsSupersetOf<const Thunk*>({thunk_ptr, &dynamic_slice_thunk}));
}

TEST(ForAllThunksTest, CommandBufferThunk) {
  auto thunk = std::make_unique<DummyThunk>();
  Thunk* thunk_ptr = thunk.get();

  ThunkSequence thunk_sequence;
  thunk_sequence.push_back(std::move(thunk));

  auto sequential_thunk = std::make_unique<SequentialThunk>(
      Thunk::ThunkInfo(), std::move(thunk_sequence));
  Thunk* sequential_thunk_ptr = sequential_thunk.get();

  CommandBufferThunk command_buffer_thunk(CommandBufferCmdSequence(),
                                          Thunk::ThunkInfo(),
                                          std::move(sequential_thunk));
  EXPECT_THAT(GetAllThunks(&command_buffer_thunk),
              UnorderedElementsAre(thunk_ptr, &command_buffer_thunk,
                                   sequential_thunk_ptr));
}

TEST(ForAllThunksTest, ConditionalThunk) {
  auto thunk = std::make_unique<DummyThunk>();
  Thunk* thunk_ptr = thunk.get();

  ThunkSequence thunk_sequence;
  thunk_sequence.push_back(std::move(thunk));

  auto sequential_thunk = std::make_unique<SequentialThunk>(
      Thunk::ThunkInfo(), std::move(thunk_sequence));
  SequentialThunk* sequential_thunk_ptr = sequential_thunk.get();

  ConditionalThunkConfig config;
  config.branch_thunks.push_back(std::move(sequential_thunk));
  ConditionalThunk conditional_thunk(Thunk::ThunkInfo(), std::move(config),
                                     BufferAllocation::Slice());

  EXPECT_THAT(GetAllThunks(&conditional_thunk),
              UnorderedElementsAre(thunk_ptr, sequential_thunk_ptr,
                                   &conditional_thunk));
}

TEST(ForAllThunksTest, WhileThunk) {
  auto condition_thunk = std::make_unique<DummyThunk>();
  Thunk* condition_thunk_ptr = condition_thunk.get();

  ThunkSequence condition_thunk_sequence;
  condition_thunk_sequence.push_back(std::move(condition_thunk));

  auto body_thunk = std::make_unique<DummyThunk>();
  Thunk* body_thunk_ptr = body_thunk.get();

  ThunkSequence body_thunk_sequence;
  body_thunk_sequence.push_back(std::move(body_thunk));

  WhileThunk while_thunk(
      Thunk::ThunkInfo(), BufferAllocation::Slice(),
      std::make_unique<SequentialThunk>(Thunk::ThunkInfo(),
                                        std::move(condition_thunk_sequence)),
      std::make_unique<SequentialThunk>(Thunk::ThunkInfo(),
                                        std::move(body_thunk_sequence)));

  EXPECT_THAT(GetAllThunks(&while_thunk),
              // `WhileThunk` wraps the `condition_thunk_sequence` and
              // `body_thunk_sequence` in `SequentialThunks`, which is why
              // iterate over more than the three expected `Thunks`.
              IsSupersetOf<const Thunk*>(
                  {condition_thunk_ptr, body_thunk_ptr, &while_thunk}));
}

}  // namespace
}  // namespace xla::gpu
