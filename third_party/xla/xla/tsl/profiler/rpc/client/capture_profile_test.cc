/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/tsl/profiler/rpc/client/capture_profile.h"

#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "riegeli/bytes/fd_reader.h"
#include "riegeli/records/record_reader.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/status_matchers.h"  // IWYU pragma: keep
#include "xla/tsl/platform/test.h"
#include "xla/tsl/profiler/rpc/client/save_profile.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "tsl/platform/host_info.h"
#include "tsl/platform/path.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tsl {
namespace profiler {
namespace {

using ::absl_testing::IsOk;
using ::tensorflow::profiler::XSpace;
using ::testing::ElementsAre;
using ::testing::Not;
using ::tsl::proto_testing::EqualsProto;
using ::tsl::proto_testing::ParseTextProtoOrDie;
using ::tsl::proto_testing::Partially;

TEST(CaptureProfileTest, ExportToTensorBoardSingleXSpaceSuccess) {
  std::string temp_dir = testing::TmpDir();
  std::string run = "test_run_single";

  XSpace space = ParseTextProtoOrDie<XSpace>(R"pb(
    hostnames: "test_host_single"
    planes { name: "test_plane" }
  )pb");

  ASSERT_OK(ExportToTensorBoard(space, temp_dir, run,
                                /*also_export_trace_json=*/false));

  std::string file_path =
      io::JoinPath(GetTensorBoardProfilePluginDir(temp_dir), run,
                   absl::StrCat(tsl::port::Hostname(), ".xplane.pb"));
  EXPECT_OK(Env::Default()->FileExists(file_path));

  XSpace read_space;
  ASSERT_OK(ReadBinaryProto(Env::Default(), file_path, &read_space));
  EXPECT_THAT(read_space, Partially(EqualsProto(R"pb(
                hostnames: "test_host_single"
                planes { name: "test_plane" }
              )pb")));
}

TEST(CaptureProfileTest, ExportToTensorBoardSingleXSpaceWithTraceJsonSuccess) {
  std::string temp_dir = testing::TmpDir();
  std::string run = "test_run_single_json";

  XSpace space = ParseTextProtoOrDie<XSpace>(R"pb(
    hostnames: "test_host_single_json"
  )pb");

  ASSERT_OK(ExportToTensorBoard(space, temp_dir, run,
                                /*also_export_trace_json=*/true));

  std::string plugin_dir =
      io::JoinPath(GetTensorBoardProfilePluginDir(temp_dir), run);
  std::string pb_path = io::JoinPath(
      plugin_dir, absl::StrCat(tsl::port::Hostname(), ".xplane.pb"));
  std::string json_path = io::JoinPath(
      plugin_dir, absl::StrCat(tsl::port::Hostname(), ".trace.json.gz"));

  EXPECT_OK(Env::Default()->FileExists(pb_path));
  EXPECT_OK(Env::Default()->FileExists(json_path));
}

TEST(CaptureProfileTest, ExportToTensorBoardVectorXSpacesSuccess) {
  std::string temp_dir = testing::TmpDir();
  std::string run = "test_run_vector";

  XSpace space1 = ParseTextProtoOrDie<XSpace>(R"pb(
    hostnames: "host1"
    planes { name: "plane1" }
  )pb");

  XSpace space2 = ParseTextProtoOrDie<XSpace>(R"pb(
    hostnames: "host2"
    planes { name: "plane2" }
  )pb");

  std::vector<XSpace> xspaces = {space1, space2};
  ASSERT_OK(ExportToTensorBoard(temp_dir, run, xspaces));
  EXPECT_TRUE(xspaces.empty());

  std::string file_path =
      io::JoinPath(GetTensorBoardProfilePluginDir(temp_dir), run,
                   absl::StrCat(tsl::port::Hostname(), ".xplane.riegeli"));
  EXPECT_OK(Env::Default()->FileExists(file_path));

  std::vector<XSpace> read_spaces;
  riegeli::RecordReader<riegeli::FdReader<>> reader{
      riegeli::FdReader<>(file_path)};
  XSpace read_space;
  while (reader.ReadRecord(read_space)) {
    read_spaces.push_back(read_space);
    read_space.Clear();
  }
  ASSERT_OK(reader.status());
  reader.Close();

  ASSERT_EQ(read_spaces.size(), 2);
  EXPECT_THAT(read_spaces, ElementsAre(Partially(EqualsProto(R"pb(
                                         hostnames: "host1"
                                         planes { name: "plane1" }
                                       )pb")),
                                       Partially(EqualsProto(R"pb(
                                         hostnames: "host2"
                                         planes { name: "plane2" }
                                       )pb"))));
}

TEST(CaptureProfileTest, ExportToTensorBoardVectorXSpacesSuccessAndClear) {
  std::string temp_dir = testing::TmpDir();
  std::string run = "test_run_pointer";

  XSpace space = ParseTextProtoOrDie<XSpace>(R"pb(
    hostnames: "host1"
    planes { name: "plane1" }
  )pb");
  std::vector<XSpace> xspaces = {space};

  ASSERT_OK(ExportToTensorBoard(temp_dir, run, xspaces));
  EXPECT_TRUE(xspaces.empty());

  std::string file_path =
      io::JoinPath(GetTensorBoardProfilePluginDir(temp_dir), run,
                   absl::StrCat(tsl::port::Hostname(), ".xplane.riegeli"));
  EXPECT_OK(Env::Default()->FileExists(file_path));
}

TEST(CaptureProfileTest,
     ExportToTensorBoardVectorXSpacesDefaultRunSuccessAndClear) {
  std::string temp_dir = testing::TmpDir();

  XSpace space = ParseTextProtoOrDie<XSpace>(R"pb(
    hostnames: "host1"
    planes { name: "plane1" }
  )pb");
  std::vector<XSpace> xspaces = {space};

  ASSERT_OK(ExportToTensorBoard(temp_dir, xspaces));
  EXPECT_TRUE(xspaces.empty());
}

TEST(CaptureProfileTest, ExportToTensorBoardSingleXSpaceFailurePropagation) {
  std::string invalid_logdir = "/proc/invalid_nonexistent_dir_for_test";
  XSpace space = ParseTextProtoOrDie<XSpace>(R"pb(
    hostnames: "host1"
    planes { name: "plane1" }
  )pb");

  EXPECT_THAT(ExportToTensorBoard(space, invalid_logdir, "run",
                                  /*also_export_trace_json=*/false),
              Not(IsOk()));
}

TEST(CaptureProfileTest, ExportToTensorBoardVectorXSpacesEmpty) {
  std::string temp_dir = testing::TmpDir();
  std::vector<XSpace> empty_xspaces;
  EXPECT_OK(ExportToTensorBoard(temp_dir, "run", empty_xspaces));
  EXPECT_OK(ExportToTensorBoard(temp_dir, empty_xspaces));
}

TEST(CaptureProfileTest, ExportToTensorBoardVectorXSpacesFailurePropagation) {
  std::string invalid_logdir = "/proc/invalid_nonexistent_dir_for_test";
  XSpace space = ParseTextProtoOrDie<XSpace>(R"pb(
    hostnames: "host1"
    planes { name: "plane1" }
  )pb");
  std::vector<XSpace> xspaces = {space};

  EXPECT_THAT(ExportToTensorBoard(invalid_logdir, "run", xspaces), Not(IsOk()));
}

}  // namespace
}  // namespace profiler
}  // namespace tsl
