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

#include "xla/hlo/tools/comparison/offline_utils.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/hlo/tools/comparison/comparison_result.pb.h"
#include "xla/hlo/tools/comparison/launch_info_compat.pb.h"
#include "xla/hlo/tools/comparison/original_tensor_summary_utils.h"
#include "xla/service/computation_placer.h"
#include "xla/service/hlo.pb.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/path.h"

namespace xla::numerics::comparison {
namespace {

using ::testing::HasSubstr;
using ::testing::UnorderedElementsAre;

class OfflineUtilsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    temp_dir_ = tsl::io::JoinPath(tsl::testing::TmpDir(), "offline_utils_test");
    ASSERT_OK(tsl::Env::Default()->RecursivelyCreateDir(temp_dir_));
  }

  void TearDown() override {
    int64_t undeleted_files, undeleted_dirs;
    ASSERT_OK(tsl::Env::Default()->DeleteRecursively(
        temp_dir_, &undeleted_files, &undeleted_dirs));
  }

  std::string temp_dir_;
};

TEST_F(OfflineUtilsTest, ProtoFromScopeInstruction) {
  ScopeInstruction si{/*instruction_name=*/"foo", /*iteration_index=*/1};
  ScopeInstructionProto proto = ProtoFromScopeInstruction(si);
  EXPECT_EQ(proto.instruction_name(), "foo");
  EXPECT_EQ(proto.iteration_index(), 1);
}

TEST_F(OfflineUtilsTest, FindFiles) {
  std::string file1 = tsl::io::JoinPath(temp_dir_, "file1.txt");
  std::string file2 = tsl::io::JoinPath(temp_dir_, "file2.txt");
  ASSERT_OK(tsl::WriteStringToFile(tsl::Env::Default(), file1, "data1"));
  ASSERT_OK(tsl::WriteStringToFile(tsl::Env::Default(), file2, "data2"));

  auto files_or = FindFiles({temp_dir_}, "*.txt");
  ASSERT_OK(files_or.status());
  EXPECT_THAT(*files_or, UnorderedElementsAre(file1, file2));
}

TEST_F(OfflineUtilsTest, FindOneFile) {
  std::string file1 = tsl::io::JoinPath(temp_dir_, "file1.txt");
  ASSERT_OK(tsl::WriteStringToFile(tsl::Env::Default(), file1, "data1"));

  auto file_or = FindOneFile({temp_dir_}, "file1.txt");
  ASSERT_OK(file_or.status());
  EXPECT_EQ(*file_or, file1);

  auto not_found_or = FindOneFile({temp_dir_}, "nonexistent.txt");
  EXPECT_FALSE(not_found_or.ok());
  EXPECT_EQ(not_found_or.status().code(), absl::StatusCode::kNotFound);
}

TEST_F(OfflineUtilsTest, FindLaunchInfoWithId) {
  std::string module_name = "test_module";
  int64_t launch_id = 12345;
  std::string launch_info_path = tsl::io::JoinPath(
      temp_dir_, absl::StrFormat("module_0.%s.%d.launch_info.pbtxt",
                                 module_name, launch_id));
  ASSERT_OK(
      tsl::WriteStringToFile(tsl::Env::Default(), launch_info_path, "dummy"));

  auto result_or = FindLaunchInfo({temp_dir_}, module_name, launch_id);
  ASSERT_OK(result_or.status());
  EXPECT_EQ(result_or->path, launch_info_path);
  EXPECT_EQ(result_or->launch_barrier_id, launch_id);
}

TEST_F(OfflineUtilsTest, FindLaunchInfoWithoutId) {
  std::string module_name = "test_module";
  int64_t launch_id = 12345;
  std::string launch_info_path = tsl::io::JoinPath(
      temp_dir_, absl::StrFormat("module_0.%s.%d.launch_info.pbtxt",
                                 module_name, launch_id));
  ASSERT_OK(
      tsl::WriteStringToFile(tsl::Env::Default(), launch_info_path, "dummy"));

  auto result_or = FindLaunchInfo({temp_dir_}, module_name, std::nullopt);
  ASSERT_OK(result_or.status());
  EXPECT_EQ(result_or->path, launch_info_path);
  EXPECT_EQ(result_or->launch_barrier_id, launch_id);
}

TEST_F(OfflineUtilsTest, LoadRunData) {
  tsl::Env* env = tsl::Env::Default();
  std::string module_name = "test_module";
  int64_t launch_id = 12345;

  // 1. Create LaunchInfo
  LaunchInfo launch_info_proto;
  auto device_assignment = std::make_unique<xla::DeviceAssignment>(1, 1);
  device_assignment->operator()(0, 0) = 100;
  device_assignment->Serialize(launch_info_proto.mutable_device_assignment());

  std::string launch_info_path = tsl::io::JoinPath(
      temp_dir_, absl::StrFormat("module_0.%s.%d.launch_info.pbtxt",
                                 module_name, launch_id));
  ASSERT_OK(tsl::WriteTextProto(env, launch_info_path, launch_info_proto));

  // 2. Create HLO Modules
  HloModuleProto hlo_proto;
  hlo_proto.set_name(module_name);
  hlo_proto.set_entry_computation_id(1);

  auto* computation = hlo_proto.add_computations();
  computation->set_name("main");
  computation->set_id(1);
  auto* root = computation->add_instructions();
  root->set_name("root");
  root->set_id(2);
  root->set_opcode("parameter");
  root->set_parameter_number(0);
  root->mutable_shape()->set_element_type(xla::F32);
  root->mutable_shape()->add_dimensions(1);
  computation->set_root_id(2);

  auto* program_shape = hlo_proto.mutable_host_program_shape();
  *program_shape->add_parameters() = root->shape();
  *program_shape->mutable_result() = root->shape();
  program_shape->add_parameter_names("p0");

  std::string original_hlo_path = tsl::io::JoinPath(
      temp_dir_, absl::StrFormat("module_0.%s.%d.original_hlo_module.pb",
                                 module_name, launch_id));
  std::string optimized_hlo_path = tsl::io::JoinPath(
      temp_dir_, absl::StrFormat("module_0.%s.%d.optimized_hlo_module.pb",
                                 module_name, launch_id));
  ASSERT_OK(tsl::WriteBinaryProto(env, original_hlo_path, hlo_proto));
  ASSERT_OK(tsl::WriteBinaryProto(env, optimized_hlo_path, hlo_proto));

  // 3. Create log files
  // FindFiles pattern: *%s*tpu_log-*_%d-*.riegeli
  std::string log_file_name = absl::StrFormat(
      "module_0.%s.tpu_log-100_%d-0.riegeli", module_name, launch_id);
  std::string log_path = tsl::io::JoinPath(temp_dir_, log_file_name);
  ASSERT_OK(tsl::WriteStringToFile(env, log_path, ""));

  // 4. Load Run Data
  auto run_data_or = LoadRunData({temp_dir_}, module_name, launch_id);
  ASSERT_OK(run_data_or.status());

  EXPECT_EQ(run_data_or->module_name, module_name);
  EXPECT_EQ(run_data_or->launch_barrier_id, launch_id);
  EXPECT_EQ(run_data_or->log_files.size(), 1);
  EXPECT_EQ(run_data_or->log_files[0], log_path);
  EXPECT_EQ(run_data_or->device_ids_for_log_files[0], 100);
}

TEST_F(OfflineUtilsTest, LoadRunDataMissingLogs) {
  tsl::Env* env = tsl::Env::Default();
  std::string module_name = "test_module";
  int64_t launch_id = 12345;

  // 1. Create LaunchInfo
  LaunchInfo launch_info_proto;
  auto device_assignment = std::make_unique<xla::DeviceAssignment>(1, 1);
  device_assignment->operator()(0, 0) = 100;
  device_assignment->Serialize(launch_info_proto.mutable_device_assignment());

  std::string launch_info_path = tsl::io::JoinPath(
      temp_dir_, absl::StrFormat("module_0.%s.%d.launch_info.pbtxt",
                                 module_name, launch_id));
  ASSERT_OK(tsl::WriteTextProto(env, launch_info_path, launch_info_proto));

  // 2. Create HLO Modules
  HloModuleProto hlo_proto;
  hlo_proto.set_name(module_name);
  hlo_proto.set_entry_computation_id(1);
  auto* computation = hlo_proto.add_computations();
  computation->set_name("main");
  computation->set_id(1);
  auto* root = computation->add_instructions();
  root->set_name("root");
  root->set_id(2);
  root->set_opcode("parameter");
  root->set_parameter_number(0);
  root->mutable_shape()->set_element_type(xla::F32);
  root->mutable_shape()->add_dimensions(1);
  computation->set_root_id(2);
  auto* program_shape = hlo_proto.mutable_host_program_shape();
  *program_shape->add_parameters() = root->shape();
  *program_shape->mutable_result() = root->shape();
  program_shape->add_parameter_names("p0");

  std::string original_hlo_path = tsl::io::JoinPath(
      temp_dir_, absl::StrFormat("module_0.%s.%d.original_hlo_module.pb",
                                 module_name, launch_id));
  std::string optimized_hlo_path = tsl::io::JoinPath(
      temp_dir_, absl::StrFormat("module_0.%s.%d.optimized_hlo_module.pb",
                                 module_name, launch_id));
  ASSERT_OK(tsl::WriteBinaryProto(env, original_hlo_path, hlo_proto));
  ASSERT_OK(tsl::WriteBinaryProto(env, optimized_hlo_path, hlo_proto));

  // Load data without creating log files
  auto run_data_or = LoadRunData({temp_dir_}, module_name, launch_id);
  EXPECT_FALSE(run_data_or.ok());
  EXPECT_THAT(run_data_or.status().message(),
              HasSubstr("Log files for device 100 not found"));
}

}  // namespace
}  // namespace xla::numerics::comparison
