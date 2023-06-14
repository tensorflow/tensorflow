/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/profiler/convert/repository.h"

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/tsl/profiler/protobuf/xplane.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::testing::Eq;

TEST(Repository, GetHostName) {
  auto session_snapshot_or =
      SessionSnapshot::Create({"log/plugins/profile/hostname0.xplane.pb",
                               "log/plugins/profile/hostname1.xplane.pb"},
                              /*xspaces=*/std::nullopt);
  TF_CHECK_OK(session_snapshot_or.status());
  EXPECT_THAT(session_snapshot_or.value().GetHostname(0), Eq("hostname0"));
  EXPECT_THAT(session_snapshot_or.value().GetHostname(1), Eq("hostname1"));
  EXPECT_TRUE(session_snapshot_or.value().HasAccessibleRunDir());
}

TEST(Repository, GetSpaceByHostName) {
  std::vector<std::unique_ptr<XSpace>> xspaces;
  // prepare host 1.
  auto space1 = std::make_unique<XSpace>();
  *(space1->add_hostnames()) = "hostname1";
  // with index 0 which shouldn't impact the space finding by name.
  xspaces.push_back(std::move(space1));

  // prepare host 0.
  auto space0 = std::make_unique<XSpace>();
  *(space0->add_hostnames()) = "hostname0";
  // with index 1 which shouldn't impact the space finding by name.
  xspaces.push_back(std::move(space0));

  auto session_snapshot_or =
      SessionSnapshot::Create({"log/plugins/profile/hostname1.xplane.pb",
                               "log/plugins/profile/hostname0.xplane.pb"},
                              std::move(xspaces));
  TF_CHECK_OK(session_snapshot_or.status());
  auto xspace0_or = session_snapshot_or.value().GetXSpaceByName("hostname0");
  TF_CHECK_OK(xspace0_or.status());
  auto xspace1_or = session_snapshot_or.value().GetXSpaceByName("hostname1");
  EXPECT_FALSE(session_snapshot_or.value().HasAccessibleRunDir());
  TF_CHECK_OK(xspace1_or.status());
  EXPECT_THAT(xspace0_or.value()->hostnames(0), Eq("hostname0"));
  EXPECT_THAT(xspace1_or.value()->hostnames(0), Eq("hostname1"));
}

TEST(Repository, GetSSTableFile) {
  auto session_snapshot_or =
      SessionSnapshot::Create({"log/plugins/profile/hostname0.xplane.pb"},
                              /*xspaces=*/std::nullopt);
  TF_CHECK_OK(session_snapshot_or.status());
  auto sstable_path =
      session_snapshot_or.value().GetFilePath("trace_viewer@", "hostname0");
  auto not_found_path =
      session_snapshot_or.value().GetFilePath("memory_viewer", "hostname0");
  EXPECT_THAT(sstable_path, Eq("log/plugins/profile/hostname0.SSTABLE"));
  EXPECT_THAT(not_found_path, Eq(std::nullopt));
}

TEST(Repository, GetSSTableFileWithXSpace) {
  std::vector<std::unique_ptr<XSpace>> xspaces;
  // prepare host 0.
  auto space0 = std::make_unique<XSpace>();
  *(space0->add_hostnames()) = "hostname0";
  // with index 1 which shouldn't impact the space finding by name.
  xspaces.push_back(std::move(space0));
  auto session_snapshot_or = SessionSnapshot::Create(
      {"log/plugins/profile/hostname0.xplane.pb"}, std::move(xspaces));
  TF_CHECK_OK(session_snapshot_or.status());
  auto file_path_init_by_xspace =
      session_snapshot_or.value().GetFilePath("trace_viewer@", "hostname0");
  // The file path should be disabled in this mode.
  EXPECT_THAT(file_path_init_by_xspace, Eq(std::nullopt));
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
