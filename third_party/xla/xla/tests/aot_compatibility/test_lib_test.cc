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

#include "xla/tests/aot_compatibility/test_lib.h"

#include <stdlib.h>

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/debug_options_runtime_fields.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/service/cpu/executable.pb.h"
#include "xla/service/gpu/gpu_executable.pb.h"
#include "xla/tests/aot_interception_pjrt_client.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/util/split_proto/human_readable_aot_executable.pb.h"
#include "xla/xla.pb.h"
#include "tsl/platform/path.h"
#include "tsl/platform/protobuf.h"

namespace xla {
namespace aot_compatibility {
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

TEST(TestLibTest, CompareGPUExecutables_IgnoresMachineCodeAndXlaDumpTo) {
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

TEST(TestLibTest, CompareGPUExecutables_IgnoresBufferAssignmentFields) {
  auto golden = ParseTextProtoOrDie<HumanReadableAotExecutable>(R"pb(
    gpu_executable {
      buffer_allocations { values { index: 0 size: 16 } }
      buffer_assignment {
        logical_buffers { id: 1 size: 16 }
        buffer_aliases {
          source_buffer_id: 1
          location { instruction_id: 10 }
        }
        buffer_aliases {
          source_buffer_id: 1
          location { instruction_id: 20 }
        }
      }
      buffer_allocations_debug_summary: "golden_summary"
    }
  )pb");
  auto fresh = ParseTextProtoOrDie<HumanReadableAotExecutable>(R"pb(
    gpu_executable {
      buffer_allocations { values { index: 0 size: 32 } }
      buffer_assignment {
        logical_buffers { id: 2 size: 32 }
        buffer_aliases {
          source_buffer_id: 2
          location { instruction_id: 10 }
        }
      }
      buffer_allocations_debug_summary: "fresh_summary"
    }
  )pb");
  EXPECT_OK(AOTInterceptionPjrtClient::CompareGPUExecutables(fresh, golden));
}

// Where in the executable a copy of the debug options lives. Both copies must
// be compared, except for the fields the comparator ignores.
enum class DebugOptionsCopy { kModuleConfig, kCompileOptions };

// A GPU executable whose `copy` of the debug options is `debug_options`.
HumanReadableAotExecutable GpuExecutableWithDebugOptions(
    DebugOptionsCopy copy, const DebugOptions& debug_options) {
  HumanReadableAotExecutable executable;
  executable.mutable_gpu_executable()->set_binary("same_binary");
  switch (copy) {
    case DebugOptionsCopy::kModuleConfig:
      *executable.mutable_gpu_executable()
           ->mutable_hlo_module_with_config()
           ->mutable_config()
           ->mutable_debug_options() = debug_options;
      break;
    case DebugOptionsCopy::kCompileOptions:
      *executable.mutable_executable_and_options()
           ->mutable_compile_options()
           ->mutable_executable_build_options()
           ->mutable_debug_options() = debug_options;
      break;
  }
  return executable;
}

// Returns an enum value of `field` other than its default, or null if the enum
// has no other value.
const tsl::protobuf::EnumValueDescriptor* NonDefaultEnumValue(
    const tsl::protobuf::FieldDescriptor* field) {
  const tsl::protobuf::EnumValueDescriptor* default_value =
      field->default_value_enum();
  const tsl::protobuf::EnumDescriptor* type = field->enum_type();
  for (int i = 0; i < type->value_count(); ++i) {
    if (default_value == nullptr ||
        type->value(i)->number() != default_value->number()) {
      return type->value(i);
    }
  }
  return nullptr;
}

// Sets `field` of `message` to a value that differs from its default: a
// singular field gets a value other than its default, a repeated field gets one
// such element, and a message gets its first field set the same way. Returns
// false if it cannot, e.g. for a type it does not know.
bool SetToNonDefault(const tsl::protobuf::FieldDescriptor* field,
                     tsl::protobuf::Message* message, int depth = 0) {
  using FieldDescriptor = tsl::protobuf::FieldDescriptor;
  const tsl::protobuf::Reflection* reflection = message->GetReflection();
  const bool repeated = field->is_repeated();
  switch (field->cpp_type()) {
    case FieldDescriptor::CPPTYPE_BOOL: {
      const bool value = !field->default_value_bool();
      if (repeated) {
        reflection->AddBool(message, field, value);
      } else {
        reflection->SetBool(message, field, value);
      }
      return true;
    }
    case FieldDescriptor::CPPTYPE_INT32: {
      const int32_t value = field->default_value_int32() + 1;
      if (repeated) {
        reflection->AddInt32(message, field, value);
      } else {
        reflection->SetInt32(message, field, value);
      }
      return true;
    }
    case FieldDescriptor::CPPTYPE_INT64: {
      const int64_t value = field->default_value_int64() + 1;
      if (repeated) {
        reflection->AddInt64(message, field, value);
      } else {
        reflection->SetInt64(message, field, value);
      }
      return true;
    }
    case FieldDescriptor::CPPTYPE_UINT32: {
      const uint32_t value = field->default_value_uint32() + 1;
      if (repeated) {
        reflection->AddUInt32(message, field, value);
      } else {
        reflection->SetUInt32(message, field, value);
      }
      return true;
    }
    case FieldDescriptor::CPPTYPE_UINT64: {
      const uint64_t value = field->default_value_uint64() + 1;
      if (repeated) {
        reflection->AddUInt64(message, field, value);
      } else {
        reflection->SetUInt64(message, field, value);
      }
      return true;
    }
    case FieldDescriptor::CPPTYPE_FLOAT: {
      const float value = field->default_value_float() + 1;
      if (repeated) {
        reflection->AddFloat(message, field, value);
      } else {
        reflection->SetFloat(message, field, value);
      }
      return true;
    }
    case FieldDescriptor::CPPTYPE_DOUBLE: {
      const double value = field->default_value_double() + 1;
      if (repeated) {
        reflection->AddDouble(message, field, value);
      } else {
        reflection->SetDouble(message, field, value);
      }
      return true;
    }
    case FieldDescriptor::CPPTYPE_STRING: {
      std::string value =
          absl::StrCat(field->default_value_string(), "_non_default");
      if (repeated) {
        reflection->AddString(message, field, std::move(value));
      } else {
        reflection->SetString(message, field, std::move(value));
      }
      return true;
    }
    case FieldDescriptor::CPPTYPE_ENUM: {
      const tsl::protobuf::EnumValueDescriptor* value =
          NonDefaultEnumValue(field);
      if (value == nullptr) {
        return false;
      }
      if (repeated) {
        reflection->AddEnum(message, field, value);
      } else {
        reflection->SetEnum(message, field, value);
      }
      return true;
    }
    case FieldDescriptor::CPPTYPE_MESSAGE: {
      // An empty submessage equals an unset one in an EQUIVALENT comparison,
      // so one of its fields must be set as well. The depth limit stops a
      // message type that contains itself.
      constexpr int kMaxDepth = 4;
      const tsl::protobuf::Descriptor* type = field->message_type();
      if (depth >= kMaxDepth || type->field_count() == 0) {
        return false;
      }
      tsl::protobuf::Message* child =
          repeated ? reflection->AddMessage(message, field)
                   : reflection->MutableMessage(message, field);
      return SetToNonDefault(type->field(0), child, depth + 1);
    }
  }
  return false;
}

// Returns `DebugOptions` with only `field` set, to a non-default value. Fails
// the test and returns nullopt if `SetToNonDefault` cannot set it.
std::optional<DebugOptions> DebugOptionsWithNonDefault(
    const tsl::protobuf::FieldDescriptor* field) {
  DebugOptions debug_options;
  if (!SetToNonDefault(field, &debug_options)) {
    ADD_FAILURE() << "Cannot set DebugOptions." << field->name() << " of type "
                  << field->type_name()
                  << " to a non-default value; teach SetToNonDefault to.";
    return std::nullopt;
  }
  // Guards against a helper bug that would make every "ignores" check pass.
  EXPECT_FALSE(tsl::protobuf::util::MessageDifferencer::Equivalent(
      debug_options, DebugOptions()))
      << field->name();
  return debug_options;
}

class CompareGPUExecutablesDebugOptionsTest
    : public ::testing::TestWithParam<DebugOptionsCopy> {
 protected:
  // Compares two GPU executables that differ only in the `GetParam()` copy of
  // their debug options.
  static absl::Status Compare(const DebugOptions& fresh,
                              const DebugOptions& golden) {
    return AOTInterceptionPjrtClient::CompareGPUExecutables(
        GpuExecutableWithDebugOptions(GetParam(), fresh),
        GpuExecutableWithDebugOptions(GetParam(), golden));
  }
};

// The fields whose defaults changed since the v3 goldens of the GPU collective
// AOT tests. None can change a loaded executable, so a difference in them must
// not cause a mismatch.
TEST_P(CompareGPUExecutablesDebugOptionsTest, IgnoresCompileOnlyFields) {
  constexpr int kCompileOnlyFields[] = {
      DebugOptions::kXlaGpuExperimentalAutotuneBackendsFieldNumber,
      DebugOptions::kXlaGpuExperimentalDeterministicOpsTritonFieldNumber,
      DebugOptions::kXlaGpuExperimentalEnableConvFusionFieldNumber,
      DebugOptions::kXlaGpuExperimentalGemmFusionV2FieldNumber,
      DebugOptions::kXlaGpuUnsupportedUseCrossHostOneShotKernelFieldNumber,
  };
  DebugOptions fresh_options;
  for (const int number : kCompileOnlyFields) {
    ASSERT_FALSE(IsRuntimeDebugOptionsField(number)) << number;
    const tsl::protobuf::FieldDescriptor* field =
        DebugOptions::descriptor()->FindFieldByNumber(number);
    ASSERT_NE(field, nullptr) << number;
    ASSERT_TRUE(SetToNonDefault(field, &fresh_options)) << field->name();
  }

  EXPECT_OK(Compare(fresh_options, DebugOptions()));
}

TEST_P(CompareGPUExecutablesDebugOptionsTest, DetectsNumericsFlagDifference) {
  ASSERT_TRUE(
      IsRuntimeDebugOptionsField(DebugOptions::kXlaGpuDetectNanFieldNumber));
  DebugOptions golden_options;
  golden_options.set_xla_gpu_detect_nan(DebugOptions::DETECTION_MODE_WARNING);
  DebugOptions fresh_options;
  fresh_options.set_xla_gpu_detect_nan(DebugOptions::DETECTION_MODE_FAIL);

  EXPECT_THAT(
      Compare(fresh_options, golden_options),
      StatusIs(absl::StatusCode::kInternal, HasSubstr("xla_gpu_detect_nan")));
}

TEST_P(CompareGPUExecutablesDebugOptionsTest, DetectsCollectiveFlagDifference) {
  ASSERT_TRUE(IsRuntimeDebugOptionsField(
      DebugOptions::kXlaGpuNcclTerminationTimeoutSecondsFieldNumber));
  DebugOptions golden_options;
  golden_options.set_xla_gpu_nccl_termination_timeout_seconds(600);
  DebugOptions fresh_options;
  fresh_options.set_xla_gpu_nccl_termination_timeout_seconds(30);

  EXPECT_THAT(Compare(fresh_options, golden_options),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("xla_gpu_nccl_termination_timeout_seconds")));
}

// Ignoring fields, `xla_dump_to` included, must not hide a runtime field that
// differs alongside them, and must keep the ignored ones out of the report.
TEST_P(CompareGPUExecutablesDebugOptionsTest,
       DetectsRuntimeFlagDifferenceNextToIgnoredOnes) {
  ASSERT_TRUE(
      IsRuntimeDebugOptionsField(DebugOptions::kXlaGpuDetectInfFieldNumber));
  ASSERT_FALSE(IsRuntimeDebugOptionsField(
      DebugOptions::kXlaGpuExperimentalEnableConvFusionFieldNumber));
  DebugOptions golden_options;
  golden_options.set_xla_dump_to("/tmp/golden_dir");
  golden_options.set_xla_gpu_experimental_enable_conv_fusion(false);
  golden_options.set_xla_gpu_detect_inf(DebugOptions::DETECTION_MODE_NONE);
  DebugOptions fresh_options;
  fresh_options.set_xla_dump_to("/tmp/fresh_dir");
  fresh_options.set_xla_gpu_experimental_enable_conv_fusion(true);
  fresh_options.set_xla_gpu_detect_inf(DebugOptions::DETECTION_MODE_FAIL);

  EXPECT_THAT(
      Compare(fresh_options, golden_options),
      StatusIs(
          absl::StatusCode::kInternal,
          AllOf(HasSubstr("xla_gpu_detect_inf"), Not(HasSubstr("xla_dump_to")),
                Not(HasSubstr("xla_gpu_experimental_enable_conv_fusion")))));
}

// `xla_dump_to` is ignored although it is a runtime field: the path is
// host-specific, and the loading process sets its own. So two different dump
// directories must not cause a mismatch.
TEST_P(CompareGPUExecutablesDebugOptionsTest, IgnoresXlaDumpToPath) {
  DebugOptions golden_options;
  golden_options.set_xla_dump_to("/tmp/golden_dir");
  DebugOptions fresh_options;
  fresh_options.set_xla_dump_to("/tmp/fresh_dir");

  EXPECT_OK(Compare(fresh_options, golden_options));
}

// Nor must an empty `xla_dump_to` vs a non-empty one, in either direction.
TEST_P(CompareGPUExecutablesDebugOptionsTest, IgnoresXlaDumpToEmptiness) {
  ASSERT_TRUE(IsRuntimeDebugOptionsField(DebugOptions::kXlaDumpToFieldNumber));
  DebugOptions empty_options;
  DebugOptions set_options;
  set_options.set_xla_dump_to("/tmp/dump_dir");

  EXPECT_OK(Compare(/*fresh=*/set_options, /*golden=*/empty_options));
  EXPECT_OK(Compare(/*fresh=*/empty_options, /*golden=*/set_options));
}

// Every runtime field must be compared, whatever its type, except
// `xla_dump_to`, which is ignored and has its own tests.
TEST_P(CompareGPUExecutablesDebugOptionsTest,
       DetectsEveryRuntimeFieldDifference) {
  for (const int number : RuntimeDebugOptionsFieldNumbers()) {
    if (number == DebugOptions::kXlaDumpToFieldNumber) {
      continue;
    }
    const tsl::protobuf::FieldDescriptor* field =
        DebugOptions::descriptor()->FindFieldByNumber(number);
    ASSERT_NE(field, nullptr) << "No DebugOptions field number " << number;
    SCOPED_TRACE(field->name());
    std::optional<DebugOptions> fresh_options =
        DebugOptionsWithNonDefault(field);
    if (!fresh_options.has_value()) {
      continue;
    }

    EXPECT_THAT(Compare(*fresh_options, DebugOptions()),
                StatusIs(absl::StatusCode::kInternal,
                         HasSubstr(std::string(field->name()))));
  }
}

// Every field outside the runtime list must be ignored, whatever its type.
TEST_P(CompareGPUExecutablesDebugOptionsTest,
       IgnoresEveryNonRuntimeFieldDifference) {
  const tsl::protobuf::Descriptor* descriptor = DebugOptions::descriptor();
  for (int i = 0; i < descriptor->field_count(); ++i) {
    const tsl::protobuf::FieldDescriptor* field = descriptor->field(i);
    if (IsRuntimeDebugOptionsField(field->number())) {
      continue;
    }
    SCOPED_TRACE(field->name());
    std::optional<DebugOptions> fresh_options =
        DebugOptionsWithNonDefault(field);
    if (!fresh_options.has_value()) {
      continue;
    }

    EXPECT_OK(Compare(*fresh_options, DebugOptions()));
  }
}

INSTANTIATE_TEST_SUITE_P(
    BothCopies, CompareGPUExecutablesDebugOptionsTest,
    ::testing::Values(DebugOptionsCopy::kModuleConfig,
                      DebugOptionsCopy::kCompileOptions),
    [](const ::testing::TestParamInfo<DebugOptionsCopy>& info) {
      return info.param == DebugOptionsCopy::kModuleConfig ? "ModuleConfig"
                                                           : "CompileOptions";
    });

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

// An executable whose payload is not a `gpu_executable` must still be compared
// in full, not treated as equal.
TEST(TestLibTest, CompareGPUExecutables_DetectsDifferencesInNonGpuPayload) {
  HumanReadableAotExecutable golden;
  golden.mutable_cpu_executable()->set_entry_function_name("golden_entry");
  HumanReadableAotExecutable fresh;
  fresh.mutable_cpu_executable()->set_entry_function_name("fresh_entry");

  EXPECT_THAT(AOTInterceptionPjrtClient::CompareGPUExecutables(fresh, golden),
              StatusIs(absl::StatusCode::kInternal,
                       AllOf(HasSubstr("entry_function_name"),
                             HasSubstr("fresh_entry"))));
}

// The CPU comparator must ignore both copies of debug_options as well as the
// compiled machine-code fields (object_files, target_machine_options,
// data_layout). See b/567825028.
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
}  // namespace aot_compatibility
}  // namespace xla
