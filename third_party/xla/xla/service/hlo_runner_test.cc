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

#include "xla/service/hlo_runner.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "google/protobuf/message_lite.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/service/device_assignment.h"
#include "xla/service/fake_multi_host_pjrt_client.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/service/test_compilation_environment.pb.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/fingerprint.h"
#include "tsl/platform/path.h"

namespace xla {
namespace {

using ::absl_testing::StatusIs;
using ::testing::StartsWith;

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
    GetParameterMemoryKinds() const override {
      return absl::UnimplementedError(
          "GetParameterMemoryKinds is not implemented.");
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

// Fake module with run_hlo_passes=false.
constexpr absl::string_view kModuleSerializedName =
    "a83d1d9a594b0f1fbc4227408abcdc6a.bin.gz";
// Fake module with run_hlo_passes=true.
constexpr absl::string_view kModuleWithRunHloPassesSerializedName =
    "1f51ba5f389ad07cbe268573dd21a94e.bin.gz";
// Fake module with a compilation environment set.
constexpr absl::string_view kModuleWithCompEnvSerializedName =
    "9385c25a58e7d6d47e56af1dd950b7d1.bin.gz";

class ArtifactDirTest : public ::testing::Test {
 public:
  void SetUp() override {
    ASSERT_OK(tsl::Env::Default()->CreateDir(artifact_dir_));
  }
  void TearDown() override {
    int64_t num_files_deleted = 0;
    int64_t num_dirs_deleted = 0;
    ASSERT_OK(tsl::Env::Default()->DeleteRecursively(
        artifact_dir_, &num_files_deleted, &num_dirs_deleted));
  }

  const std::string artifact_dir_ =
      tsl::io::JoinPath(testing::TempDir(), "artifact_dir");
};

using CompilePhaseHloRunnerTest = ArtifactDirTest;

// Tests that a call to CreateExecutable places the file in the right location.
TEST_F(CompilePhaseHloRunnerTest, CreateExecutablePlacesFileCorrectly) {
  CompilePhaseHloRunner runner(std::make_unique<FakeClient>(), artifact_dir_);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m, CreateFakeModule());
  ASSERT_OK(
      runner.CreateExecutable(std::move(m), /*run_hlo_passes=*/false).status());

  std::vector<std::string> children;
  ASSERT_OK(tsl::Env::Default()->GetChildren(artifact_dir_, &children));
  ASSERT_EQ(children.size(), 1);
  ASSERT_EQ(children[0], kModuleSerializedName);
}

// Tests that a CreateExecutable call with different run_hlo_passes value places
// the file in a different location.
TEST_F(CompilePhaseHloRunnerTest,
       CreateExecutablePlacesFilesCorrectlyWithDifferentRunHloPasses) {
  CompilePhaseHloRunner runner(std::make_unique<FakeClient>(), artifact_dir_);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m, CreateFakeModule());
  ASSERT_OK(runner.CreateExecutable(std::move(m), /*run_hlo_passes=*/true));

  std::vector<std::string> children;
  ASSERT_OK(tsl::Env::Default()->GetChildren(artifact_dir_, &children));
  ASSERT_EQ(children.size(), 1);
  ASSERT_EQ(children[0], kModuleWithRunHloPassesSerializedName);
}

// Tests that a CreateExecutable call with a different compilation
// environment places the file in a different location.
TEST_F(CompilePhaseHloRunnerTest,
       CreateExecutablePlacesFilesCorrectlyWithCompilationEnvironment) {
  CompilePhaseHloRunner runner(std::make_unique<FakeClient>(), artifact_dir_);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m, CreateFakeModule());
  m->comp_envs().RegisterProcessNewEnvFn(
      test::TestCompilationEnvironment1::GetDescriptor(),
      [](std::unique_ptr<tsl::protobuf::Message> msg) {
        std::unique_ptr<test::TestCompilationEnvironment1> env(
            google::protobuf::DownCastMessage<test::TestCompilationEnvironment1>(
                msg.release()));
        if (env == nullptr) {
          env = std::make_unique<test::TestCompilationEnvironment1>();
        }
        env->set_some_flag(42);
        return env;
      });
  ASSERT_OK(m->comp_envs().InitializeAllKnownEnvs());
  ASSERT_OK(runner.CreateExecutable(std::move(m), /*run_hlo_passes=*/false));

  std::vector<std::string> children;
  ASSERT_OK(tsl::Env::Default()->GetChildren(artifact_dir_, &children));
  ASSERT_EQ(children.size(), 1);
  ASSERT_EQ(children[0], kModuleWithCompEnvSerializedName);
}

// Tests that a CreateExecutable call with a different device assignment
// places the file in a different location.
TEST_F(CompilePhaseHloRunnerTest,
       CreateExecutablePlacesFilesCorrectlyWithDeviceAssignment) {
  CompilePhaseHloRunner runner(std::make_unique<FakeClient>(), artifact_dir_);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m1, CreateFakeModule());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m2, CreateFakeModule());

  DeviceAssignment da1(1, 2);
  da1.FillIota(0);
  m1->mutable_config().set_static_device_assignment(da1);

  DeviceAssignment da2(1, 2);
  da2.FillIota(1);
  m2->mutable_config().set_static_device_assignment(da2);

  ASSERT_OK(runner.CreateExecutable(std::move(m1), /*run_hlo_passes=*/false));
  ASSERT_OK(runner.CreateExecutable(std::move(m2), /*run_hlo_passes=*/false));

  std::vector<std::string> children;
  ASSERT_OK(tsl::Env::Default()->GetChildren(artifact_dir_, &children));
  EXPECT_EQ(children.size(), 2);
}

using ExecutePhaseHloRunnerTest = ArtifactDirTest;

// Tests that a call to CreateExecutable reads the file from the correct path
// and deserializes the right contents.
TEST_F(ExecutePhaseHloRunnerTest, CreateExecutableReadsFileCorrectly) {
  ASSERT_OK(CompilePhaseHloRunner::WriteCompressedExecutable(
      tsl::io::JoinPath(artifact_dir_, kModuleSerializedName), "hello world"));
  absl::Notification notification;
  std::optional<std::string> serialized_representation_read = std::nullopt;
  ExecutePhaseHloRunner runner(
      std::make_unique<FakeClient>(
          [&notification,
           &serialized_representation_read](absl::string_view serialized) {
            serialized_representation_read = serialized;
            notification.Notify();
          }),
      artifact_dir_);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m, CreateFakeModule());
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OpaqueExecutable> executable,
      runner.CreateExecutable(std::move(m), /*run_hlo_passes=*/false));

  ASSERT_TRUE(notification.WaitForNotificationWithTimeout(absl::Seconds(5)));
  ASSERT_TRUE(serialized_representation_read.has_value());
  ASSERT_EQ(*serialized_representation_read, "hello world");
}

TEST_F(ExecutePhaseHloRunnerTest,
       CreateExecutableFailsOnDuplicateLoadIfFeatureEnabled) {
  ASSERT_OK(CompilePhaseHloRunner::WriteCompressedExecutable(
      tsl::io::JoinPath(artifact_dir_, kModuleSerializedName), "hello world"));
  ExecutePhaseHloRunner runner(std::make_unique<FakeClient>(), artifact_dir_);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m, CreateFakeModule());
  ASSERT_OK(runner.CreateExecutable(m->Clone(""), /*run_hlo_passes=*/false));
  EXPECT_THAT(
      runner.CreateExecutable(std::move(m), /*run_hlo_passes=*/false),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          StartsWith(
              "ExecutePhaseHloRunner::CreateExecutable called with a "
              "module that loads an executable that was previously loaded.")));
}

TEST_F(ExecutePhaseHloRunnerTest,
       CreateExecutableSucceedsOnDuplicateLoadIfFeatureDisabled) {
  ASSERT_OK(CompilePhaseHloRunner::WriteCompressedExecutable(
      tsl::io::JoinPath(artifact_dir_, kModuleSerializedName), "hello world"));
  ExecutePhaseHloRunner runner(std::make_unique<FakeClient>(), artifact_dir_,
                               /*compile_if_not_found=*/false,
                               /*fail_duplicate_loads=*/false);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m, CreateFakeModule());
  ASSERT_OK(runner.CreateExecutable(m->Clone(""), /*run_hlo_passes=*/false));
  EXPECT_OK(runner.CreateExecutable(std::move(m), /*run_hlo_passes=*/false));
}

TEST(HloRunnerMultiHostTest, ExecuteReplicatedSkipsNonAddressableDevices) {
  auto client = std::make_unique<FakeMultiHostClient>(
      absl::Span<const bool>{false, false, true, true});
  FakeMultiHostClient* raw_client = client.get();
  HloRunner runner(std::move(client));

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> m,
      ParseAndReturnUnverifiedModule(FakeMultiHostClient::kIdentityHlo));
  DeviceAssignment device_assignment(4, 1);
  device_assignment.FillIota(0);

  Literal arg_literal = LiteralUtil::CreateR0<int32_t>(5);
  Literal infeed_literal = LiteralUtil::CreateR0<int32_t>(7);
  std::vector<Literal> outfeed_literals;
  HloRunnerInterface::ReplicatedExecuteOptions options;
  options.num_devices = 4;
  options.arguments = {&arg_literal};
  options.infeed_steps = 1;
  options.infeed_values = std::vector<const Literal*>(4, &infeed_literal);
  options.outfeed_shape = ShapeUtil::MakeShape(S32, {});
  options.outfeed_values = &outfeed_literals;

  ASSERT_OK_AND_ASSIGN(
      std::vector<Literal> results,
      runner.ExecuteReplicated(std::move(m), options, &device_assignment));

  ASSERT_EQ(results.size(), 4);
  ASSERT_EQ(outfeed_literals.size(), 4);
  for (int i : {0, 1}) {
    SCOPED_TRACE(i);
    EXPECT_EQ(raw_client->device(i).execute_count(), 0);
    EXPECT_EQ(raw_client->device(i).infeed_count(), 0);
    EXPECT_EQ(raw_client->device(i).outfeed_count(), 0);
    EXPECT_TRUE(ShapeUtil::IsEmptyTuple(results[i].shape()));
    EXPECT_TRUE(ShapeUtil::IsEmptyTuple(outfeed_literals[i].shape()));
  }
  for (int i : {2, 3}) {
    SCOPED_TRACE(i);
    EXPECT_EQ(raw_client->device(i).execute_count(), 1);
    EXPECT_EQ(raw_client->device(i).infeed_count(), 1);
    EXPECT_EQ(raw_client->device(i).outfeed_count(), 1);
    EXPECT_EQ(results[i], LiteralUtil::CreateR0<int32_t>(10 + i));
    EXPECT_EQ(outfeed_literals[i], LiteralUtil::CreateR0<int32_t>(100 + i));
  }
}

TEST(HloRunnerMultiHostTest,
     ExecuteReplicatedReturnsPlaceholdersWhenNoDevicesAreAddressable) {
  auto client = std::make_unique<FakeMultiHostClient>(
      absl::Span<const bool>{false, false});
  FakeMultiHostClient* raw_client = client.get();
  HloRunner runner(std::move(client));

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m, CreateFakeModule());
  DeviceAssignment device_assignment(2, 1);
  device_assignment.FillIota(0);

  Literal infeed_literal = LiteralUtil::CreateR0<int32_t>(7);
  std::vector<Literal> outfeed_literals;
  HloRunnerInterface::ReplicatedExecuteOptions options;
  options.num_devices = 2;
  options.infeed_steps = 1;
  options.infeed_values = std::vector<const Literal*>(2, &infeed_literal);
  options.outfeed_shape = ShapeUtil::MakeShape(S32, {});
  options.outfeed_values = &outfeed_literals;

  ASSERT_OK_AND_ASSIGN(
      std::vector<Literal> results,
      runner.ExecuteReplicated(std::move(m), options, &device_assignment));

  ASSERT_EQ(results.size(), 2);
  ASSERT_EQ(outfeed_literals.size(), 2);
  for (int i : {0, 1}) {
    SCOPED_TRACE(i);
    EXPECT_EQ(raw_client->device(i).execute_count(), 0);
    EXPECT_EQ(raw_client->device(i).infeed_count(), 0);
    EXPECT_EQ(raw_client->device(i).outfeed_count(), 0);
    EXPECT_TRUE(ShapeUtil::IsEmptyTuple(results[i].shape()));
    EXPECT_TRUE(ShapeUtil::IsEmptyTuple(outfeed_literals[i].shape()));
  }
}

}  // namespace
}  // namespace xla
