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
#include "tsl/platform/host_info.h"
#include "tsl/platform/path.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tsl {
namespace profiler {
namespace {

using ::absl_testing::IsOk;
using ::tensorflow::profiler::XPlane;
using ::tensorflow::profiler::XSpace;
using ::testing::Not;

TEST(CaptureProfileTest, ExportToTensorBoardSingleXSpaceSuccess) {
  std::string temp_dir = testing::TmpDir();
  std::string run = "test_run_single";

  XSpace space;
  space.add_hostnames("test_host_single");
  XPlane* plane = space.add_planes();
  plane->set_name("test_plane");

  ASSERT_OK(ExportToTensorBoard(space, temp_dir, run,
                                /*also_export_trace_json=*/false));

  std::string file_path =
      io::JoinPath(GetTensorBoardProfilePluginDir(temp_dir), run,
                   absl::StrCat(tsl::port::Hostname(), ".xplane.pb"));
  EXPECT_OK(Env::Default()->FileExists(file_path));

  XSpace read_space;
  ASSERT_OK(ReadBinaryProto(Env::Default(), file_path, &read_space));
  EXPECT_EQ(read_space.hostnames(0), "test_host_single");
  EXPECT_EQ(read_space.planes(0).name(), "test_plane");

  ASSERT_OK(Env::Default()->DeleteFile(file_path));
}

TEST(CaptureProfileTest, ExportToTensorBoardSingleXSpaceWithTraceJsonSuccess) {
  std::string temp_dir = testing::TmpDir();
  std::string run = "test_run_single_json";

  XSpace space;
  space.add_hostnames("test_host_json");

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

  ASSERT_OK(Env::Default()->DeleteFile(pb_path));
  ASSERT_OK(Env::Default()->DeleteFile(json_path));
}

TEST(CaptureProfileTest, ExportToTensorBoardVectorXSpacesSuccess) {
  std::string temp_dir = testing::TmpDir();
  std::string run = "test_run_vector";

  XSpace space1;
  space1.add_hostnames("host1");
  XPlane* plane1 = space1.add_planes();
  plane1->set_name("plane1");

  XSpace space2;
  space2.add_hostnames("host2");
  XPlane* plane2 = space2.add_planes();
  plane2->set_name("plane2");

  std::vector<XSpace> xspaces = {space1, space2};
  ASSERT_OK(ExportToTensorBoard(&xspaces, temp_dir, run));
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
  EXPECT_EQ(read_spaces[0].hostnames(0), "host1");
  EXPECT_EQ(read_spaces[0].planes(0).name(), "plane1");
  EXPECT_EQ(read_spaces[1].hostnames(0), "host2");
  EXPECT_EQ(read_spaces[1].planes(0).name(), "plane2");

  ASSERT_OK(Env::Default()->DeleteFile(file_path));
}

TEST(CaptureProfileTest, ExportToTensorBoardPointerXSpacesSuccessAndClear) {
  std::string temp_dir = testing::TmpDir();
  std::string run = "test_run_pointer";

  XSpace space;
  space.add_hostnames("host1");
  std::vector<XSpace> xspaces = {space};

  ASSERT_OK(ExportToTensorBoard(&xspaces, temp_dir, run));
  EXPECT_TRUE(xspaces.empty());

  std::string file_path =
      io::JoinPath(GetTensorBoardProfilePluginDir(temp_dir), run,
                   absl::StrCat(tsl::port::Hostname(), ".xplane.riegeli"));
  EXPECT_OK(Env::Default()->FileExists(file_path));

  ASSERT_OK(Env::Default()->DeleteFile(file_path));
}

TEST(CaptureProfileTest,
     ExportToTensorBoardPointerXSpacesDefaultRunSuccessAndClear) {
  std::string temp_dir = testing::TmpDir();

  XSpace space;
  space.add_hostnames("host1");
  std::vector<XSpace> xspaces = {space};

  ASSERT_OK(ExportToTensorBoard(&xspaces, temp_dir));
  EXPECT_TRUE(xspaces.empty());
}

TEST(CaptureProfileTest, ExportToTensorBoardSingleXSpaceFailurePropagation) {
  std::string invalid_logdir = "/proc/invalid_nonexistent_dir_for_test";
  XSpace space;
  space.add_hostnames("host1");

  EXPECT_THAT(ExportToTensorBoard(space, invalid_logdir, "run",
                                  /*also_export_trace_json=*/false),
              Not(IsOk()));
}

TEST(CaptureProfileTest, ExportToTensorBoardPointerXSpacesNullptr) {
  std::string temp_dir = testing::TmpDir();
  EXPECT_OK(ExportToTensorBoard(nullptr, temp_dir, "run"));
  EXPECT_OK(ExportToTensorBoard(nullptr, temp_dir));
}

TEST(CaptureProfileTest, ExportToTensorBoardPointerXSpacesFailurePropagation) {
  std::string invalid_logdir = "/proc/invalid_nonexistent_dir_for_test";
  XSpace space;
  space.add_hostnames("host1");
  std::vector<XSpace> xspaces = {space};

  EXPECT_THAT(ExportToTensorBoard(&xspaces, invalid_logdir, "run"),
              Not(IsOk()));
}

}  // namespace
}  // namespace profiler
}  // namespace tsl
