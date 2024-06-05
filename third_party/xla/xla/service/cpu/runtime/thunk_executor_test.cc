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

#include "xla/service/cpu/runtime/thunk_executor.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/runtime/thunk.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla::cpu {
namespace {

using ::testing::ElementsAre;

// A test-only thunk for testing thunk executor implementation.
class BufferUseThunk : public Thunk {
 public:
  using BufferUses = Thunk::BufferUses;

  BufferUseThunk(std::string name, BufferUses buffer_uses,
                 std::vector<std::string>* trace = nullptr)
      : Thunk(Kind::kKernel, Info{name}),
        buffer_uses_(std::move(buffer_uses)),
        trace_(trace) {}

  static std::unique_ptr<Thunk> Create(
      std::string name, BufferUses buffer_uses,
      std::vector<std::string>* trace = nullptr) {
    return std::make_unique<BufferUseThunk>(std::move(name),
                                            std::move(buffer_uses), trace);
  }

  BufferUses buffer_uses() const override { return buffer_uses_; }

  absl::Status Execute(const ExecuteParams&) final {
    if (trace_) trace_->push_back(info().op_name);
    return absl::OkStatus();
  }

 private:
  BufferUses buffer_uses_;
  std::vector<std::string>* trace_;
};

TEST(ThunkExecutorTest, Ordering) {
  BufferAllocation alloc(/*index=*/0, /*size=*/1024, /*color=*/0);

  BufferAllocation::Slice slice0(&alloc, /*offset=*/0, /*size=*/10);
  BufferAllocation::Slice slice1(&alloc, /*offset=*/5, /*size=*/10);
  BufferAllocation::Slice slice2(&alloc, /*offset=*/10, /*size=*/10);

  ThunkSequence sequence;
  sequence.push_back(BufferUseThunk::Create("a", {BufferUse::Read(slice0)}));
  sequence.push_back(BufferUseThunk::Create("b", {BufferUse::Read(slice1)}));
  sequence.push_back(BufferUseThunk::Create("c", {BufferUse::Write(slice2)}));

  TF_ASSERT_OK_AND_ASSIGN(auto executor,
                          ThunkExecutor::Create(std::move(sequence)));

  EXPECT_THAT(executor.source(), ElementsAre(0, 1));
  EXPECT_THAT(executor.sink(), ElementsAre(0, 2));
}

TEST(ThunkExecutorTest, Execute) {
  BufferAllocation alloc(/*index=*/0, /*size=*/1024, /*color=*/0);

  BufferAllocation::Slice slice0(&alloc, /*offset=*/0, /*size=*/10);
  BufferAllocation::Slice slice1(&alloc, /*offset=*/5, /*size=*/10);
  BufferAllocation::Slice slice2(&alloc, /*offset=*/10, /*size=*/10);

  std::vector<std::string> trace;

  ThunkSequence sequence;
  sequence.push_back(
      BufferUseThunk::Create("a", {BufferUse::Read(slice0)}, &trace));
  sequence.push_back(
      BufferUseThunk::Create("b", {BufferUse::Read(slice1)}, &trace));
  sequence.push_back(
      BufferUseThunk::Create("c", {BufferUse::Write(slice2)}, &trace));

  TF_ASSERT_OK_AND_ASSIGN(auto executor,
                          ThunkExecutor::Create(std::move(sequence)));
  Thunk::ExecuteParams params;
  TF_ASSERT_OK(executor.Execute(params, [&](ThunkExecutor::Task task) {
    trace.push_back("<TaskRunner>");
    task();
  }));

  EXPECT_THAT(trace, ElementsAre("<TaskRunner>", "b", "c", "a"));
}

}  // namespace
}  // namespace xla::cpu
