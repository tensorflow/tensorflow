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

#include "xla/service/executable.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_execution_profile.h"
#include "xla/service/hlo_module_executable.h"
#include "xla/service/service_executable_run_options.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

class TestExecutable : public HloModuleExecutable {
 public:
  explicit TestExecutable(std::shared_ptr<HloModule> module)
      : HloModuleExecutable{std::move(module)} {}

  absl::StatusOr<ExecutionOutput> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions* run_options,
      std::vector<ExecutionInput> arguments) override {
    return absl::UnimplementedError("Not needed for this test.");
  }

  const BufferAssignmentProto* buffer_assignment_proto() const override {
    return buffer_assignment_proto_.has_value() ? &*buffer_assignment_proto_
                                                : nullptr;
  }

  void set_buffer_assignment_proto(BufferAssignmentProto proto) {
    buffer_assignment_proto_ = std::move(proto);
  }

 private:
  std::optional<BufferAssignmentProto> buffer_assignment_proto_;
};

class ExecutableTest : public HloHardwareIndependentTestBase {};

TEST_F(ExecutableTest, DumpingHloSnapshotsFollowsDebugOptions) {
  constexpr absl::string_view kHloModule = R"(
    HloModule module

    ENTRY main {
      ROOT c = s32[] constant(1)
    }
  )";

  ASSERT_OK_AND_ASSIGN(std::shared_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloModule));
  EXPECT_FALSE(DumpingHloSnapshots(TestExecutable(module)));
  EXPECT_FALSE(DumpingHloSnapshots(TestExecutable(nullptr)));

  module->mutable_config().mutable_debug_options().set_xla_dump_hlo_snapshots(
      true);
  EXPECT_TRUE(DumpingHloSnapshots(TestExecutable(module)));
}

TEST_F(ExecutableTest, MakeHloProtoForSnapshotUsesModuleAndBufferAssignment) {
  constexpr absl::string_view kHloModule = R"(
    HloModule module

    ENTRY main {
      ROOT c = s32[] constant(1)
    }
  )";

  ASSERT_OK_AND_ASSIGN(std::shared_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloModule));
  TestExecutable executable(module);

  HloProto hlo_proto = MakeHloProtoForSnapshot(executable);
  EXPECT_EQ(hlo_proto.hlo_module().name(), "module");
  EXPECT_FALSE(hlo_proto.has_buffer_assignment());

  BufferAssignmentProto buffer_assignment;
  buffer_assignment.add_buffer_allocations()->set_index(42);
  executable.set_buffer_assignment_proto(std::move(buffer_assignment));

  hlo_proto = MakeHloProtoForSnapshot(executable);
  EXPECT_EQ(hlo_proto.hlo_module().name(), "module");
  ASSERT_EQ(hlo_proto.buffer_assignment().buffer_allocations_size(), 1);
  EXPECT_EQ(hlo_proto.buffer_assignment().buffer_allocations(0).index(), 42);
}

}  // namespace
}  // namespace xla
