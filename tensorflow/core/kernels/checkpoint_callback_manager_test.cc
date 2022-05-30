/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0(the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/core/kernels/checkpoint_callback_manager.h"

#include <string>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/resource_handle.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/managed_stack_trace.h"

namespace tensorflow {
namespace checkpoint {
namespace {

class CheckpointCallbackManagerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    checkpoint_callback_manager_ = new CheckpointCallbackManager();
    handle_ = ResourceHandle::MakeRefCountingHandle(
        checkpoint_callback_manager_, "cpu", {}, {});
  }

  CheckpointCallbackManager* checkpoint_callback_manager_;
  ResourceHandle handle_;
};

TEST_F(CheckpointCallbackManagerTest,
       GetCheckpointIdAndPathFromPrefixWithTempDir) {
  StatusOr<std::pair<std::string, std::string>> pair =
      CheckpointCallbackManager::GetCheckpointIdAndPathFromPrefix(
          "/foo/bar/model.ckpt-5_temp/part-00000-of-00001");
  TF_ASSERT_OK(pair.status());
  EXPECT_EQ(pair->first, "model.ckpt-5");
  EXPECT_EQ(pair->second, "/foo/bar");
}

TEST_F(CheckpointCallbackManagerTest,
       GetCheckpointIdAndPathFromPrefixWithPartFile) {
  StatusOr<std::pair<std::string, std::string>> pair =
      CheckpointCallbackManager::GetCheckpointIdAndPathFromPrefix(
          "/foo/bar/model.ckpt-5/part-00000-of-00001");
  TF_ASSERT_OK(pair.status());
  EXPECT_EQ(pair->first, "model.ckpt-5");
  EXPECT_EQ(pair->second, "/foo/bar");
}

TEST_F(CheckpointCallbackManagerTest,
       GetCheckpointIdAndPathFromPrefixWithoutPartFile) {
  StatusOr<std::pair<std::string, std::string>> pair =
      CheckpointCallbackManager::GetCheckpointIdAndPathFromPrefix(
          "/foo/bar/model.ckpt-5");
  TF_ASSERT_OK(pair.status());
  EXPECT_EQ(pair->first, "model.ckpt-5");
  EXPECT_EQ(pair->second, "/foo/bar");
}

TEST_F(CheckpointCallbackManagerTest,
       GetCheckpointIdAndPathFromPrefixUnrecognized) {
  EXPECT_FALSE(
      CheckpointCallbackManager::GetCheckpointIdAndPathFromPrefix("/foo/bar")
          .ok());
}

TEST_F(CheckpointCallbackManagerTest, RegisterSaveCallbackTwice) {
  SaveCallback first_callback = [](absl::string_view checkpoint_id) {
    return std::string("MockString");
  };

  SaveCallback second_callback = [](absl::string_view checkpoint_id) {
    return std::string("MockString");
  };

  TF_ASSERT_OK(checkpoint_callback_manager_->RegisterSaveCallback(
      "foo", std::move(first_callback)));

  EXPECT_FALSE(checkpoint_callback_manager_
                   ->RegisterSaveCallback("foo", std::move(second_callback))
                   .ok());
}

TEST_F(CheckpointCallbackManagerTest, RegisterRestoreCallbackTwice) {
  RestoreCallback first_callback = [](absl::string_view checkpoint_id,
                                      absl::string_view str) {
    return Status::OK();
  };
  RestoreCallback second_callback = [](absl::string_view checkpoint_id,
                                       absl::string_view str) {
    return Status::OK();
  };

  TF_ASSERT_OK(checkpoint_callback_manager_->RegisterRestoreCallback(
      "foo", std::move(first_callback)));

  EXPECT_FALSE(checkpoint_callback_manager_
                   ->RegisterRestoreCallback("foo", std::move(second_callback))
                   .ok());
}

TEST_F(CheckpointCallbackManagerTest, DoesSaveCallbackExist) {
  SaveCallback first_callback = [](absl::string_view checkpoint_id) {
    return std::string("MockString");
  };

  SaveCallback second_callback = [](absl::string_view checkpoint_id) {
    return std::string("MockString");
  };

  TF_ASSERT_OK(checkpoint_callback_manager_->RegisterSaveCallback(
      "foo", std::move(first_callback)));

  TF_ASSERT_OK(checkpoint_callback_manager_->RegisterSaveCallback(
      "bar", std::move(second_callback)));

  EXPECT_TRUE(checkpoint_callback_manager_->DoesSaveCallbackExist("foo"));
  EXPECT_TRUE(checkpoint_callback_manager_->DoesSaveCallbackExist("bar"));
  EXPECT_FALSE(
      checkpoint_callback_manager_->DoesSaveCallbackExist("not_exist"));
}

TEST_F(CheckpointCallbackManagerTest, DoesRestoreCallbackExist) {
  RestoreCallback first_callback = [](absl::string_view checkpoint_id,
                                      absl::string_view str) {
    return Status::OK();
  };
  RestoreCallback second_callback = [](absl::string_view checkpoint_id,
                                       absl::string_view str) {
    return Status::OK();
  };

  TF_ASSERT_OK(checkpoint_callback_manager_->RegisterRestoreCallback(
      "foo", std::move(first_callback)));

  TF_ASSERT_OK(checkpoint_callback_manager_->RegisterRestoreCallback(
      "bar", std::move(second_callback)));

  EXPECT_TRUE(checkpoint_callback_manager_->DoesRestoreCallbackExist("foo"));
  EXPECT_TRUE(checkpoint_callback_manager_->DoesRestoreCallbackExist("bar"));
  EXPECT_FALSE(
      checkpoint_callback_manager_->DoesRestoreCallbackExist("not_exist"));
}

TEST_F(CheckpointCallbackManagerTest, SaveTwoCallbacks) {
  SaveCallback save_callback1 = [](absl::string_view checkpoint_id) {
    return absl::StrCat("MockContent1::", checkpoint_id);
  };
  SaveCallback save_callback2 = [](absl::string_view checkpoint_id) {
    return absl::StrCat("MockContent2::", checkpoint_id);
  };

  TF_ASSERT_OK(checkpoint_callback_manager_->RegisterSaveCallback(
      "foo", std::move(save_callback1)));

  TF_ASSERT_OK(checkpoint_callback_manager_->RegisterSaveCallback(
      "bar", std::move(save_callback2)));

  checkpoint_callback_manager_->Save(io::JoinPath(
      testing::TmpDir(), "model.ckpt-123_temp/part-00000-of-00001"));
  std::string file_content1;
  TF_EXPECT_OK(ReadFileToString(
      Env::Default(), io::JoinPath(testing::TmpDir(), "model.ckpt-123.foo"),
      &file_content1));
  EXPECT_EQ(file_content1, "MockContent1::model.ckpt-123");

  std::string file_content2;
  TF_EXPECT_OK(ReadFileToString(
      Env::Default(), io::JoinPath(testing::TmpDir(), "model.ckpt-123.bar"),
      &file_content2));
  EXPECT_EQ(file_content2, "MockContent2::model.ckpt-123");
}

TEST_F(CheckpointCallbackManagerTest, SaveMultipleTimes) {
  SaveCallback save_callback = [](absl::string_view checkpoint_id) {
    return absl::StrCat("MockContent::", checkpoint_id);
  };

  TF_ASSERT_OK(checkpoint_callback_manager_->RegisterSaveCallback(
      "foo", std::move(save_callback)));

  checkpoint_callback_manager_->Save(io::JoinPath(
      testing::TmpDir(), "model.ckpt-100_temp/part-00000-of-00001"));

  checkpoint_callback_manager_->Save(io::JoinPath(
      testing::TmpDir(), "model.ckpt-100_temp/part-00000-of-00001"));

  checkpoint_callback_manager_->Save(io::JoinPath(
      testing::TmpDir(), "model.ckpt-200_temp/part-00000-of-00001"));

  std::string file_content;
  TF_EXPECT_OK(ReadFileToString(
      Env::Default(), io::JoinPath(testing::TmpDir(), "model.ckpt-100.foo"),
      &file_content));
  EXPECT_EQ(file_content, "MockContent::model.ckpt-100");

  TF_EXPECT_OK(ReadFileToString(
      Env::Default(), io::JoinPath(testing::TmpDir(), "model.ckpt-200.foo"),
      &file_content));
  EXPECT_EQ(file_content, "MockContent::model.ckpt-200");
}

TEST_F(CheckpointCallbackManagerTest, Restore) {
  int callback_call_count = 0;
  RestoreCallback restore_callback = [&callback_call_count](
                                         absl::string_view checkpoint_id,
                                         absl::string_view str) {
    EXPECT_EQ(checkpoint_id, "model.ckpt-100");
    EXPECT_EQ(str, "Apple");
    ++callback_call_count;
    return Status::OK();
  };

  TF_ASSERT_OK(checkpoint_callback_manager_->RegisterRestoreCallback(
      "foo", std::move(restore_callback)));

  TF_EXPECT_OK(WriteStringToFile(
      Env::Default(), io::JoinPath(testing::TmpDir(), "model.ckpt-100.foo"),
      "Apple"));

  EXPECT_EQ(callback_call_count, 0);
  checkpoint_callback_manager_->Restore(
      io::JoinPath(testing::TmpDir(), "model.ckpt-100"));
  EXPECT_EQ(callback_call_count, 1);

  checkpoint_callback_manager_->Restore(
      io::JoinPath(testing::TmpDir(), "model.ckpt-100"));
  EXPECT_EQ(callback_call_count, 2);
}

TEST_F(CheckpointCallbackManagerTest, SaveAndRestore) {
  SaveCallback save_callback = [](absl::string_view checkpoint_id) {
    return std::string("Apple");
  };

  TF_ASSERT_OK(checkpoint_callback_manager_->RegisterSaveCallback(
      "foo", std::move(save_callback)));

  int restore_callback_count = 0;
  RestoreCallback restore_callback = [&restore_callback_count](
                                         absl::string_view checkpoint_id,
                                         absl::string_view str) {
    EXPECT_EQ(checkpoint_id, "model.ckpt-500");
    EXPECT_EQ(str, "Apple");
    ++restore_callback_count;
    return Status::OK();
  };

  TF_ASSERT_OK(checkpoint_callback_manager_->RegisterRestoreCallback(
      "foo", std::move(restore_callback)));

  checkpoint_callback_manager_->Save(io::JoinPath(
      testing::TmpDir(), "model.ckpt-500_temp/part-00000-of-00001"));

  EXPECT_EQ(restore_callback_count, 0);
  checkpoint_callback_manager_->Restore(
      io::JoinPath(testing::TmpDir(), "model.ckpt-500"));
  EXPECT_EQ(restore_callback_count, 1);
}

TEST_F(CheckpointCallbackManagerTest, SaveLazyCallback) {
  SaveCallback save_callback = [](absl::string_view checkpoint_id) {
    return absl::StrCat("MockContent::", checkpoint_id);
  };

  checkpoint_callback_manager_->Save(io::JoinPath(
      testing::TmpDir(), "model.ckpt-456_temp/part-00000-of-00001"));

  TF_ASSERT_OK(checkpoint_callback_manager_->RegisterSaveCallback(
      "foo", std::move(save_callback)));

  std::string file_content;
  TF_EXPECT_OK(ReadFileToString(
      Env::Default(), io::JoinPath(testing::TmpDir(), "model.ckpt-456.foo"),
      &file_content));
  EXPECT_EQ(file_content, "MockContent::model.ckpt-456");
}

TEST_F(CheckpointCallbackManagerTest, RestoreLazyCallback) {
  int callback_call_count = 0;
  RestoreCallback restore_callback = [&callback_call_count](
                                         absl::string_view checkpoint_id,
                                         absl::string_view str) {
    EXPECT_EQ(checkpoint_id, "model.ckpt-100");
    EXPECT_EQ(str, "Apple");
    ++callback_call_count;
    return Status::OK();
  };

  TF_EXPECT_OK(WriteStringToFile(
      Env::Default(), io::JoinPath(testing::TmpDir(), "model.ckpt-100.foo"),
      "Apple"));

  EXPECT_EQ(callback_call_count, 0);
  checkpoint_callback_manager_->Restore(
      io::JoinPath(testing::TmpDir(), "model.ckpt-100"));
  EXPECT_EQ(callback_call_count, 0);

  TF_ASSERT_OK(checkpoint_callback_manager_->RegisterRestoreCallback(
      "foo", std::move(restore_callback)));

  EXPECT_EQ(callback_call_count, 1);
}

}  // namespace
}  // namespace checkpoint
}  // namespace tensorflow
