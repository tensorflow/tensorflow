/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/tests/aot_compatibility_experimental/test_lib.h"

#include <stdlib.h>

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/service/cpu/executable.pb.h"
#include "xla/tests/aot_interception_pjrt_client.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/util/split_proto/human_readable_aot_executable.pb.h"
#include "tsl/platform/path.h"
#include "tsl/platform/protobuf.h"

namespace xla {
namespace aot_compatibility_experimental {
namespace {

using ::absl_testing::StatusIs;
using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::TempDir;
using ::tsl::proto_testing::ParseTextProtoOrDie;

// Sets an environment variable for the lifetime of the object and restores the
// previous state -- including absence -- afterwards. Every case in this file
// runs in one process, so an unrestored variable silently changes the meaning
// of every case that follows it.
class ScopedEnv {
 public:
  // A null `value` unsets the variable instead of assigning to it.
  ScopedEnv(const char* name, const char* value) : name_(name) {
    if (const char* previous = std::getenv(name)) {
      had_previous_ = true;
      previous_ = previous;
    }
    if (value == nullptr) {
      unsetenv(name);
    } else {
      setenv(name, value, /*overwrite=*/1);
    }
  }

  ScopedEnv(const ScopedEnv&) = delete;
  ScopedEnv& operator=(const ScopedEnv&) = delete;

  ~ScopedEnv() {
    if (had_previous_) {
      setenv(name_, previous_.c_str(), /*overwrite=*/1);
    } else {
      unsetenv(name_);
    }
  }

 private:
  const char* name_;
  bool had_previous_ = false;
  std::string previous_;
};

// A temporary directory private to one test case. `TempDir()` is
// shared by the whole binary, so a case that scans it can otherwise observe
// entries another case created.
class ScopedTempDir {
 public:
  explicit ScopedTempDir(absl::string_view name)
      : path_(tsl::io::JoinPath(TempDir(), name)) {
    Remove();
    CHECK_OK(tsl::Env::Default()->RecursivelyCreateDir(path_));
  }

  ScopedTempDir(const ScopedTempDir&) = delete;
  ScopedTempDir& operator=(const ScopedTempDir&) = delete;

  ~ScopedTempDir() { Remove(); }

  const std::string& path() const { return path_; }

 private:
  void Remove() {
    int64_t undeleted_files = 0;
    int64_t undeleted_dirs = 0;
    tsl::Env::Default()
        ->DeleteRecursively(path_, &undeleted_files, &undeleted_dirs)
        .IgnoreError();
  }

  std::string path_;
};

TEST(TestLibTest, GetAotTestParamsForBackwardsCompatibility_With4Versions) {
  ScopedEnv all_versions("XLA_AOT_TEST_ALL_VERSIONS", nullptr);
  ASSERT_OK_AND_ASSIGN(std::vector<AotTestParam> params,
                       GetAotTestParamsForBackwardsCompatibility(
                           "test_dummy_test", AOTTestPlatform::kGpu));
  EXPECT_THAT(params,
              ElementsAre(AotTestParam{AOTTestMode::kBackwardsCompatibility, 1,
                                       "test_dummy_test"},
                          AotTestParam{AOTTestMode::kBackwardsCompatibility, 3,
                                       "test_dummy_test"}));
}

TEST(TestLibTest, GetAotTestParamsForBackwardsCompatibility_AllVersions) {
  ScopedEnv all_versions("XLA_AOT_TEST_ALL_VERSIONS", "1");
  ASSERT_OK_AND_ASSIGN(std::vector<AotTestParam> params,
                       GetAotTestParamsForBackwardsCompatibility(
                           "test_dummy_test", AOTTestPlatform::kGpu));
  EXPECT_THAT(params,
              ElementsAre(AotTestParam{AOTTestMode::kBackwardsCompatibility, 1,
                                       "test_dummy_test"},
                          AotTestParam{AOTTestMode::kBackwardsCompatibility, 2,
                                       "test_dummy_test"},
                          AotTestParam{AOTTestMode::kBackwardsCompatibility, 3,
                                       "test_dummy_test"},
                          AotTestParam{AOTTestMode::kBackwardsCompatibility, 4,
                                       "test_dummy_test"}));
}

// The kUpdateGolden control plane. Under XLA_AOT_UPDATE_GOLDENS the two
// parameter generators change behaviour completely, and none of it needs
// hardware or a PjRt client.

TEST(TestLibTest,
     GetAotTestParamsForBackwardsCompatibilityUpdateGoldensIsEmpty) {
  ScopedEnv update_goldens("XLA_AOT_UPDATE_GOLDENS", "1");
  // Regenerating goldens recompiles; replaying old executables is meaningless
  // in that mode, so the suite must be instantiated with no parameters at all.
  ASSERT_OK_AND_ASSIGN(std::vector<AotTestParam> params,
                       GetAotTestParamsForBackwardsCompatibility(
                           "test_dummy_test", AOTTestPlatform::kGpu));
  EXPECT_THAT(params, IsEmpty());
}

TEST(TestLibTest,
     GetAotTestParamsForGoldenFileVerificationUpdateGoldensYieldsOneParam) {
  ScopedEnv update_goldens("XLA_AOT_UPDATE_GOLDENS", "1");
  const std::string outputs_path = TempDir();
  ScopedEnv outputs_dir("TEST_UNDECLARED_OUTPUTS_DIR", outputs_path.c_str());
  // Version 0 is the sentinel: the dump does not read an existing version.
  ASSERT_OK_AND_ASSIGN(std::vector<AotTestParam> params,
                       GetAotTestParamsForGoldenFileVerification(
                           "test_dummy_test", AOTTestPlatform::kGpu));
  EXPECT_THAT(params, ElementsAre(AotTestParam{AOTTestMode::kUpdateGolden, 0,
                                               "test_dummy_test"}));
}

TEST(TestLibTest,
     GetAotTestParamsForGoldenFileVerificationUpdateGoldensNeedsOutputsDir) {
  ScopedEnv update_goldens("XLA_AOT_UPDATE_GOLDENS", "1");
  ScopedEnv outputs_dir("TEST_UNDECLARED_OUTPUTS_DIR", nullptr);
  // Without the outputs directory the dump has nowhere to write, which means
  // the test was invoked directly rather than through update_goldens.sh.
  EXPECT_THAT(GetAotTestParamsForGoldenFileVerification("test_dummy_test",
                                                        AOTTestPlatform::kGpu),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("TEST_UNDECLARED_OUTPUTS_DIR")));
}

TEST(TestLibTest,
     GetAotTestParamsForBackwardsCompatibilityMissingTargetDirIsNotFound) {
  ScopedEnv all_versions("XLA_AOT_TEST_ALL_VERSIONS", nullptr);
  ScopedEnv update_goldens("XLA_AOT_UPDATE_GOLDENS", nullptr);
  // A target whose executables directory does not exist is a mistake -- a
  // typo'd name, or goldens that were never generated. Returning no parameters
  // would leave the suite green while testing nothing, so it must be an error,
  // and the same error the verification generator produces.
  EXPECT_THAT(GetAotTestParamsForBackwardsCompatibility(
                  "target_that_does_not_exist", AOTTestPlatform::kGpu),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(TestLibTest,
     GetAotTestParamsForGoldenFileVerificationMissingTargetDirIsNotFound) {
  ScopedEnv update_goldens("XLA_AOT_UPDATE_GOLDENS", nullptr);
  EXPECT_THAT(GetAotTestParamsForGoldenFileVerification(
                  "target_that_does_not_exist", AOTTestPlatform::kGpu),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(TestLibTest, UpdateGoldensEnvFlagIsTruthyNotMerelyPresent) {
  ScopedEnv all_versions("XLA_AOT_TEST_ALL_VERSIONS", nullptr);
  // `=0` must not enable golden dumping. Presence-only checks made
  // `--test_env=XLA_AOT_UPDATE_GOLDENS=0` turn dumping on, which is the
  // opposite of what anyone passing it intends.
  ScopedEnv update_goldens("XLA_AOT_UPDATE_GOLDENS", "0");
  ASSERT_OK_AND_ASSIGN(std::vector<AotTestParam> params,
                       GetAotTestParamsForBackwardsCompatibility(
                           "test_dummy_test", AOTTestPlatform::kGpu));
  EXPECT_THAT(params, Not(IsEmpty()));
}

TEST(TestLibTest, GetAotTestParamsForGoldenFileVerification_With4Versions) {
  ASSERT_OK_AND_ASSIGN(std::vector<AotTestParam> params,
                       GetAotTestParamsForGoldenFileVerification(
                           "test_dummy_test", AOTTestPlatform::kGpu));
  EXPECT_THAT(params, ElementsAre(AotTestParam{AOTTestMode::kGoldenVerification,
                                               4, "test_dummy_test"}));
}

TEST(TestLibTest,
     AOTInterceptionPjrtClientPackArtifactForInnerClient_Succeeds) {
  std::string artifact_path = tsl::io::JoinPath(
      GetExecutablesDirectory("test_dummy_test", AOTTestPlatform::kGpu), "v1",
      "exec.pbtxt");
  AOTInterceptionPjrtClient client(
      nullptr, AOTTestMode::kBackwardsCompatibility, artifact_path);
  ASSERT_OK_AND_ASSIGN(std::string serialized,
                       client.PackArtifactForInnerClient());
  EXPECT_FALSE(serialized.empty());
}

TEST(TestLibTest, CompareGPUExecutables_NormalizesXlaDumpToAndReturnsOk) {
  auto golden = ParseTextProtoOrDie<HumanReadableAotExecutable>(R"pb(
    gpu_executable {
      binary: "golden_binary"
      hlo_module_with_config {
        config { debug_options { xla_dump_to: "/tmp/golden_dir" } }
      }
      asm_text: "test_asm"
    }
  )pb");
  auto fresh = ParseTextProtoOrDie<HumanReadableAotExecutable>(R"pb(
    gpu_executable {
      binary: "fresh_binary_that_should_be_ignored"
      hlo_module_with_config {
        config { debug_options { xla_dump_to: "/tmp/fresh_dir" } }
      }
      asm_text: "completely_different_asm"
    }
  )pb");
  EXPECT_OK(AOTInterceptionPjrtClient::CompareGPUExecutables(fresh, golden));
}

// TODO(b/528258781): Debug options are currently cleared wholesale before the
// structural comparison, so differing compiler flags are NOT detected. Once we
// decide which flags must be preserved, this test should assert that meaningful
// flag changes are detected again.
TEST(TestLibTest, CompareGPUExecutables_IgnoresDebugOptionsForNow) {
  auto golden = ParseTextProtoOrDie<HumanReadableAotExecutable>(R"pb(
    gpu_executable {
      binary: "same_binary"
      hlo_module_with_config {
        config { debug_options { xla_gpu_enable_fast_min_max: true } }
      }
    }
  )pb");
  auto fresh = ParseTextProtoOrDie<HumanReadableAotExecutable>(R"pb(
    gpu_executable {
      binary: "same_binary"
      hlo_module_with_config {
        config { debug_options { xla_gpu_enable_fast_min_max: false } }
      }
    }
  )pb");
  EXPECT_OK(AOTInterceptionPjrtClient::CompareGPUExecutables(fresh, golden));
}

// The compile-options copy of debug_options must also be cleared before the
// structural comparison; a flag difference there must not cause a spurious
// mismatch. See b/528258781.
TEST(TestLibTest,
     CompareGPUExecutables_IgnoresCompileOptionsDebugOptionsForNow) {
  auto golden = ParseTextProtoOrDie<HumanReadableAotExecutable>(R"pb(
    gpu_executable { binary: "same_binary" }
    executable_and_options {
      compile_options {
        executable_build_options {
          debug_options { xla_gpu_enable_fast_min_max: true }
        }
      }
    }
  )pb");
  auto fresh = ParseTextProtoOrDie<HumanReadableAotExecutable>(R"pb(
    gpu_executable { binary: "same_binary" }
    executable_and_options {
      compile_options {
        executable_build_options {
          debug_options { xla_gpu_enable_fast_min_max: false }
        }
      }
    }
  )pb");
  EXPECT_OK(AOTInterceptionPjrtClient::CompareGPUExecutables(fresh, golden));
}

// A different host-specific debug option (CUDA install path) must be normalized
// away and must not cause a spurious mismatch.
TEST(TestLibTest, CompareGPUExecutables_NormalizesHostPathFieldsAndReturnsOk) {
  auto golden = ParseTextProtoOrDie<HumanReadableAotExecutable>(R"pb(
    gpu_executable {
      binary: "same_binary"
      hlo_module_with_config {
        config { debug_options { xla_gpu_cuda_data_dir: "/host_a/cuda" } }
      }
    }
  )pb");
  auto fresh = ParseTextProtoOrDie<HumanReadableAotExecutable>(R"pb(
    gpu_executable {
      binary: "same_binary"
      hlo_module_with_config {
        config { debug_options { xla_gpu_cuda_data_dir: "/host_b/cuda" } }
      }
    }
  )pb");
  EXPECT_OK(AOTInterceptionPjrtClient::CompareGPUExecutables(fresh, golden));
}

TEST(TestLibTest, CompareGPUExecutables_FailsOnGenuineDifferences) {
  auto golden = ParseTextProtoOrDie<HumanReadableAotExecutable>(R"pb(
    gpu_executable { binary: "same_binary" module_name: "test_module" }
  )pb");
  auto fresh = ParseTextProtoOrDie<HumanReadableAotExecutable>(R"pb(
    gpu_executable { binary: "same_binary" module_name: "different_module" }
  )pb");

  EXPECT_THAT(
      AOTInterceptionPjrtClient::CompareGPUExecutables(fresh, golden),
      StatusIs(absl::StatusCode::kInternal,
               AllOf(HasSubstr("Golden Proto structural comparison failed"),
                     HasSubstr("module_name"), HasSubstr("test_module"),
                     HasSubstr("different_module"))));
}

// The CPU comparator must ignore both copies of debug_options as well as the
// compiled machine-code fields (object_files, target_machine_options,
// data_layout). See b/528258781.
TEST(TestLibTest, CompareGoldenCPUExecutable_IgnoresDebugOptionsAndReturnsOk) {
  HumanReadableAotExecutable golden;
  golden.mutable_cpu_executable()->set_entry_function_name("main");
  golden.mutable_cpu_executable()
      ->mutable_hlo_module()
      ->mutable_config()
      ->mutable_debug_options()
      ->set_xla_dump_to("/tmp/golden_dir");
  golden.mutable_cpu_executable()->set_data_layout("golden_layout");

  HumanReadableAotExecutable fresh;
  fresh.mutable_cpu_executable()->set_entry_function_name("main");
  fresh.mutable_cpu_executable()
      ->mutable_hlo_module()
      ->mutable_config()
      ->mutable_debug_options()
      ->set_xla_dump_to("/tmp/fresh_dir");
  fresh.mutable_cpu_executable()->set_data_layout("fresh_layout_ignored");

  EXPECT_OK(
      AOTInterceptionPjrtClient::CompareGoldenCPUExecutable(fresh, golden));
}

TEST(TestLibTest, CompareGoldenCPUExecutable_FailsOnGenuineDifferences) {
  HumanReadableAotExecutable golden;
  golden.mutable_cpu_executable()->set_entry_function_name("main");

  HumanReadableAotExecutable fresh;
  fresh.mutable_cpu_executable()->set_entry_function_name("different_entry");

  EXPECT_THAT(
      AOTInterceptionPjrtClient::CompareGoldenCPUExecutable(fresh, golden),
      StatusIs(absl::StatusCode::kInternal,
               AllOf(HasSubstr("Golden Proto structural comparison failed"),
                     HasSubstr("entry_function_name"),
                     HasSubstr("different_entry"))));
}

// CPU artifacts are plain (non-split) protos: an ExecutableAndOptionsProto
// whose serialized_executable holds a serialized cpu::CompilationResultProto.
// The CPU deserializer must unpack the inner proto and clear
// serialized_executable.
TEST(TestLibTest, DeserializeToHumanReadable_CpuUnpacksPlainProto) {
  cpu::CompilationResultProto cpu_proto;
  cpu_proto.set_entry_function_name("main");

  ExecutableAndOptionsProto outer;
  outer.set_serialized_executable(cpu_proto.SerializeAsString());
  std::string serialized;
  ASSERT_TRUE(outer.SerializeToString(&serialized));

  ASSERT_OK_AND_ASSIGN(HumanReadableAotExecutable unpacked,
                       AOTInterceptionPjrtClient::DeserializeToHumanReadable(
                           serialized, AOTTestPlatform::kCpu));
  EXPECT_TRUE(unpacked.has_cpu_executable());
  EXPECT_EQ(unpacked.cpu_executable().entry_function_name(), "main");
  EXPECT_TRUE(
      unpacked.executable_and_options().serialized_executable().empty());
}

TEST(TestLibTest, GetExecutableVersionsIgnoresEmptyDirs) {
  ScopedTempDir temp_dir("ignores_empty_dirs");
  tsl::Env* env = tsl::Env::Default();

  // v1 contains a golden.
  std::string v1_dir = tsl::io::JoinPath(temp_dir.path(), "v1");
  ASSERT_OK(env->RecursivelyCreateDir(v1_dir));
  ASSERT_OK(
      tsl::WriteStringToFile(env, tsl::io::JoinPath(v1_dir, "test.pbtxt"), ""));

  // v2 is an empty leftover, e.g. from syncing to a revision without goldens.
  std::string v2_dir = tsl::io::JoinPath(temp_dir.path(), "v2");
  ASSERT_OK(env->RecursivelyCreateDir(v2_dir));

  EXPECT_THAT(test_lib_internal::GetExecutableVersionsInDir(temp_dir.path()),
              absl_testing::IsOkAndHolds(ElementsAre(1)));
}

// A `v`-prefixed regular file and a non-numeric `v` directory are unrelated
// entries that must not wedge test discovery for the whole target: the scanner
// skips them and still reports the real versions.
TEST(TestLibTest, GetExecutableVersionsSkipsNonVersionEntries) {
  ScopedTempDir temp_dir("skips_non_version_entries");
  tsl::Env* env = tsl::Env::Default();

  std::string v1_dir = tsl::io::JoinPath(temp_dir.path(), "v1");
  ASSERT_OK(env->RecursivelyCreateDir(v1_dir));
  ASSERT_OK(
      tsl::WriteStringToFile(env, tsl::io::JoinPath(v1_dir, "test.pbtxt"), ""));

  // A `v`-prefixed regular file, not a directory.
  ASSERT_OK(tsl::WriteStringToFile(
      env, tsl::io::JoinPath(temp_dir.path(), "v9.txt"), ""));

  // A `v`-prefixed directory whose suffix is not a number. It holds a .pbtxt
  // so it survives the empty-directory filter and genuinely reaches the parse.
  std::string v_old_dir = tsl::io::JoinPath(temp_dir.path(), "v_old");
  ASSERT_OK(env->RecursivelyCreateDir(v_old_dir));
  ASSERT_OK(tsl::WriteStringToFile(
      env, tsl::io::JoinPath(v_old_dir, "stale.pbtxt"), ""));

  // Entries that do not start with `v` at all: a plain file and a directory
  // holding a .pbtxt, either of which could appear beside the goldens.
  ASSERT_OK(tsl::WriteStringToFile(
      env, tsl::io::JoinPath(temp_dir.path(), "README.md"), ""));
  std::string notes_dir = tsl::io::JoinPath(temp_dir.path(), "notes");
  ASSERT_OK(env->RecursivelyCreateDir(notes_dir));
  ASSERT_OK(
      tsl::WriteStringToFile(env, tsl::io::JoinPath(notes_dir, "n.pbtxt"), ""));

  EXPECT_THAT(test_lib_internal::GetExecutableVersionsInDir(temp_dir.path()),
              absl_testing::IsOkAndHolds(ElementsAre(1)));
}

// Versions must be ordered numerically, not lexically. With only v1..v4 on
// disk the two orderings agree, so this uses v10 to tell them apart: lexically
// "v10" sorts before "v2".
TEST(TestLibTest, GetExecutableVersionsSortsNumericallyNotLexically) {
  ScopedTempDir temp_dir("sorts_numerically");
  tsl::Env* env = tsl::Env::Default();

  for (absl::string_view name : {"v2", "v10", "v1"}) {
    std::string dir = tsl::io::JoinPath(temp_dir.path(), name);
    ASSERT_OK(env->RecursivelyCreateDir(dir));
    ASSERT_OK(
        tsl::WriteStringToFile(env, tsl::io::JoinPath(dir, "test.pbtxt"), ""));
  }

  EXPECT_THAT(test_lib_internal::GetExecutableVersionsInDir(temp_dir.path()),
              absl_testing::IsOkAndHolds(ElementsAre(1, 2, 10)));
}

TEST(TestLibTest, WriteGoldenTextProtoRoundTripWithHeader) {
  ScopedTempDir temp_dir("write_golden_round_trip");
  std::string filepath = tsl::io::JoinPath(temp_dir.path(), "roundtrip.pbtxt");

  HumanReadableAotExecutable original;
  original.mutable_cpu_executable()->set_entry_function_name("roundtrip_test");

  ASSERT_OK(
      AOTInterceptionPjrtClient::WriteGoldenTextProto(original, filepath));

  std::string file_content;
  ASSERT_OK(
      tsl::ReadFileToString(tsl::Env::Default(), filepath, &file_content));

  EXPECT_TRUE(absl::StartsWith(file_content, "# proto-file: "));
  EXPECT_THAT(file_content,
              HasSubstr("# proto-message: xla.HumanReadableAotExecutable\n"));

  HumanReadableAotExecutable parsed;
  ASSERT_TRUE(
      tsl::protobuf::TextFormat::ParseFromString(file_content, &parsed));
  EXPECT_EQ(parsed.cpu_executable().entry_function_name(),
            original.cpu_executable().entry_function_name());
}

}  // namespace
}  // namespace aot_compatibility_experimental
}  // namespace xla
