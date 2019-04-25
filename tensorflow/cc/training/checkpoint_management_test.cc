/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/cc/training/checkpoint_management.h"

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/python/training/checkpoint_state.pb.h"

namespace tensorflow {
namespace {

TEST(CheckpointManagementTest, FailsToFindCheckpointState) {
  std::string path;
  EXPECT_FALSE(LatestCheckpointPath("/nonexistent/dir/", &path).ok());
}

TEST(CheckpointManagementTest, FailsToFindCheckpointPath) {
  const std::string dir = ::testing::TempDir();
  const std::string fname = io::JoinPath(dir, "checkpoint");

  CheckpointState ckpt_state;
  ckpt_state.add_all_model_checkpoint_paths("/path/to/model.ckpt-100");
  TF_ASSERT_OK(WriteTextProto(Env::Default(), fname, ckpt_state));

  std::string latest_path;
  EXPECT_FALSE(LatestCheckpointPath(dir, &latest_path).ok());
}

TEST(CheckpointManagementTest, FailsToFindCheckpoint) {
  const std::string dir = ::testing::TempDir();
  const std::string fname = io::JoinPath(dir, "checkpoint");

  CheckpointState ckpt_state;
  ckpt_state.set_model_checkpoint_path("abc.xyz-123");
  ckpt_state.add_all_model_checkpoint_paths("/path/to/model.ckpt-100");
  TF_ASSERT_OK(WriteTextProto(Env::Default(), fname, ckpt_state));

  std::string latest_path;
  EXPECT_FALSE(LatestCheckpointPath(dir, &latest_path).ok());
}

TEST(CheckpointManagementTest, FindsRelativeCheckpointPath) {
  const std::string dir = ::testing::TempDir();
  const std::string fname = io::JoinPath(dir, "checkpoint");

  CheckpointState ckpt_state;
  ckpt_state.set_model_checkpoint_path("abc.xyz-123");
  ckpt_state.add_all_model_checkpoint_paths("/path/to/model.ckpt-100");
  TF_ASSERT_OK(WriteTextProto(Env::Default(), fname, ckpt_state));
  TF_ASSERT_OK(
      WriteStringToFile(Env::Default(), io::JoinPath(dir, "abc.xyz-123"), ""));

  std::string latest_path;
  TF_CHECK_OK(LatestCheckpointPath(dir, &latest_path));

  EXPECT_EQ(latest_path, io::JoinPath(dir, "abc.xyz-123"));
}

TEST(CheckpointManagementTest, FindsAbsoluteCheckpointPath) {
  const std::string dir = ::testing::TempDir();
  const std::string fname = io::JoinPath(dir, "checkpoint");

  CheckpointState ckpt_state;
  ckpt_state.set_model_checkpoint_path(io::JoinPath(dir, "abc.xyz-123"));
  ckpt_state.add_all_model_checkpoint_paths("/path/to/model.ckpt-100");
  TF_ASSERT_OK(WriteTextProto(Env::Default(), fname, ckpt_state));
  TF_ASSERT_OK(
      WriteStringToFile(Env::Default(), io::JoinPath(dir, "abc.xyz-123"), ""));

  std::string latest_path;
  TF_CHECK_OK(LatestCheckpointPath(dir, &latest_path));

  EXPECT_EQ(latest_path, io::JoinPath(dir, "abc.xyz-123"));
}

TEST(CheckpointManagementTest, FindsCheckpointPathFromStateWithCustomName) {
  const std::string dir = ::testing::TempDir();
  const std::string fname = io::JoinPath(dir, "custom_name");

  CheckpointState ckpt_state;
  ckpt_state.set_model_checkpoint_path("abc.xyz-123");
  ckpt_state.add_all_model_checkpoint_paths("/path/to/model.ckpt-100");
  TF_ASSERT_OK(WriteTextProto(Env::Default(), fname, ckpt_state));
  TF_ASSERT_OK(
      WriteStringToFile(Env::Default(), io::JoinPath(dir, "abc.xyz-123"), ""));

  std::string latest_path;
  TF_CHECK_OK(LatestCheckpointPath(dir, "custom_name", &latest_path));

  EXPECT_EQ(latest_path, io::JoinPath(dir, "abc.xyz-123"));
}

}  // namespace
}  // namespace tensorflow
