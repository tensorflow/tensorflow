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

#include "xla/service/hlo_runner_pjrt.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/pjrt/interpreter/interpreter_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/service/computation_placer.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "tsl/platform/fingerprint.h"
#include "tsl/platform/notification.h"
#include "tsl/platform/path.h"

namespace xla {
namespace {

class FakeClient : public PjRtClient {
 public:
  class Executable : public PjRtExecutable {
   public:
    int num_replicas() const override { return 1; }
    int num_partitions() const override { return 1; }
    int64_t SizeOfGeneratedCodeInBytes() const override { return 0; }
    absl::string_view name() const override { return "FakeExecutable"; }
    absl::StatusOr<std::vector<std::shared_ptr<HloModule>>> GetHloModules()
        const override {
      return absl::UnimplementedError("GetHloModules is not implemented.");
    }
    absl::StatusOr<std::vector<std::vector<absl::string_view>>>
    GetOutputMemoryKinds() const override {
      return absl::UnimplementedError(
          "GetOutputMemoryKinds is not implemented.");
    }
    absl::StatusOr<std::string> SerializeExecutable() const override {
      return "serialized executable";
    }
  };

  FakeClient() : deserialize_callback_([](absl::string_view) {}) {}
  explicit FakeClient(
      std::function<void(absl::string_view)> deserialize_callback)
      : deserialize_callback_(std::move(deserialize_callback)) {}

  int process_index() const override { return 0; }
  int device_count() const override { return 1; }
  int addressable_device_count() const override { return 1; }
  absl::Span<PjRtDevice* const> devices() const override { return {}; }
  absl::Span<PjRtDevice* const> addressable_devices() const override {
    return {};
  }
  absl::Span<PjRtMemorySpace* const> memory_spaces() const override {
    return {};
  }
  PjRtPlatformId platform_id() const override {
    return tsl::Fingerprint64(platform_name_);
  }
  absl::string_view platform_name() const override { return platform_name_; }
  absl::string_view platform_version() const override { return "0.0.0"; }

  absl::StatusOr<DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const override {
    DeviceAssignment assignment(num_replicas, num_partitions);
    assignment.FillIota(0);
    return assignment;
  }

  absl::StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      const XlaComputation& computation, CompileOptions options) override {
    return std::make_unique<Executable>();
  }

  absl::StatusOr<std::unique_ptr<PjRtExecutable>> DeserializeExecutable(
      absl::string_view serialized,
      std::optional<CompileOptions> options) override {
    deserialize_callback_(serialized);
    return std::make_unique<Executable>();
  }

 private:
  std::function<void(absl::string_view)> deserialize_callback_;
  std::string platform_name_ = "fake";
  PjRtPlatformId platform_id_ = tsl::Fingerprint64(platform_name_);
};

absl::StatusOr<std::unique_ptr<HloModule>> CreateFakeModule() {
  return ParseAndReturnUnverifiedModule(R"(
HloModule constant_s32_module, entry_computation_layout={()->s32[]}

ENTRY %constant_s32 () -> s32[] {
  ROOT %constant = s32[] constant(-42)
}
)");
}

constexpr absl::string_view kModuleSerializedName =
    "4f22972e3e39d4470dc57f236347ca2d.bin";

class ArtifactDirTest : public ::testing::Test {
 public:
  void SetUp() override {
    TF_ASSERT_OK(tsl::Env::Default()->CreateDir(artifact_dir_));
  }
  void TearDown() override {
    int64_t num_files_deleted = 0;
    int64_t num_dirs_deleted = 0;
    TF_ASSERT_OK(tsl::Env::Default()->DeleteRecursively(
        artifact_dir_, &num_files_deleted, &num_dirs_deleted));
  }

  const std::string artifact_dir_ =
      tsl::io::JoinPath(testing::TempDir(), "artifact_dir");
};

using CompilePhaseHloRunnerPjRtTest = ArtifactDirTest;

// Tests that a call to CreateExecutable places the file in the right location.
TEST_F(CompilePhaseHloRunnerPjRtTest, CreateExecutablePlacesFileCorrectly) {
  CompilePhaseHloRunnerPjRt runner(std::make_unique<FakeClient>(),
                                   InterpreterClient::DeviceShapeRepresentation,
                                   InterpreterClient::ShapeSizeBytes,
                                   artifact_dir_);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m, CreateFakeModule());
  TF_ASSERT_OK(
      runner.CreateExecutable(std::move(m), /*run_hlo_passes=*/false).status());

  std::vector<std::string> children;
  TF_ASSERT_OK(tsl::Env::Default()->GetChildren(artifact_dir_, &children));
  ASSERT_EQ(children.size(), 1);
  ASSERT_EQ(children[0], kModuleSerializedName);
}

using ExecutePhaseHloRunnerPjRtTest = ArtifactDirTest;

// Tests that a call to CreateExecutable reads the file from the correct path
// and deserializes the right contents.
TEST_F(ExecutePhaseHloRunnerPjRtTest, CreateExecutableReadsFileCorrectly) {
  TF_ASSERT_OK(tsl::WriteStringToFile(
      tsl::Env::Default(),
      tsl::io::JoinPath(artifact_dir_, kModuleSerializedName), "hello world"));
  absl::Notification notification;
  std::optional<std::string> serialized_representation_read = std::nullopt;
  ExecutePhaseHloRunnerPjRt runner(
      std::make_unique<FakeClient>(
          [&notification,
           &serialized_representation_read](absl::string_view serialized) {
            serialized_representation_read = serialized;
            notification.Notify();
          }),
      InterpreterClient::DeviceShapeRepresentation,
      InterpreterClient::ShapeSizeBytes, artifact_dir_);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m, CreateFakeModule());
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OpaqueExecutable> executable,
      runner.CreateExecutable(std::move(m), /*run_hlo_passes=*/false));

  ASSERT_TRUE(notification.WaitForNotificationWithTimeout(absl::Seconds(5)));
  ASSERT_TRUE(serialized_representation_read.has_value());
  ASSERT_EQ(*serialized_representation_read, "hello world");
}

}  // namespace
}  // namespace xla
