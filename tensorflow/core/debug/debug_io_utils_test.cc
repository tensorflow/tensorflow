/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/debug/debug_io_utils.h"

#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/util/event.pb.h"

namespace tensorflow {
namespace {

class DebugIOUtilsTest : public ::testing::Test {
 public:
  void Initialize() {
    env_ = Env::Default();

    tensor_a_.reset(new Tensor(DT_FLOAT, TensorShape({2, 2})));
    tensor_a_->flat<float>()(0) = 5.0;
    tensor_a_->flat<float>()(1) = 3.0;
    tensor_a_->flat<float>()(2) = -1.0;
    tensor_a_->flat<float>()(3) = 0.0;

    tensor_b_.reset(new Tensor(DT_STRING, TensorShape{2}));
    tensor_b_->flat<string>()(0) = "corge";
    tensor_b_->flat<string>()(1) = "garply";
  }

  Status ReadEventFromFile(const string& dump_file_path, Event* event) {
    string content;
    uint64 file_size = 0;

    Status s = env_->GetFileSize(dump_file_path, &file_size);
    if (!s.ok()) {
      return s;
    }

    content.resize(file_size);

    std::unique_ptr<RandomAccessFile> file;
    s = env_->NewRandomAccessFile(dump_file_path, &file);
    if (!s.ok()) {
      return s;
    }

    StringPiece result;
    s = file->Read(0, file_size, &result, &(content)[0]);
    if (!s.ok()) {
      return s;
    }

    event->ParseFromString(content);
    return Status::OK();
  }

  Env* env_;
  std::unique_ptr<Tensor> tensor_a_;
  std::unique_ptr<Tensor> tensor_b_;
};

TEST_F(DebugIOUtilsTest, DumpFloatTensorToFileSunnyDay) {
  Initialize();

  const string test_dir = testing::TmpDir();

  // Append levels of nonexisting directories, to test that the function can
  // create directories.
  const string kNodeName = "foo/bar/qux/tensor_a";
  const string kDebugOpName = "DebugIdentity";
  const int32 output_slot = 0;
  uint64 wall_time = env_->NowMicros();

  string dump_file_path;
  TF_ASSERT_OK(DebugFileIO::DumpTensorToDir(kNodeName, output_slot,
                                            kDebugOpName, *tensor_a_, wall_time,
                                            test_dir, &dump_file_path));

  // Read the file into a Event proto.
  Event event;
  TF_ASSERT_OK(ReadEventFromFile(dump_file_path, &event));

  ASSERT_GE(wall_time, event.wall_time());
  ASSERT_EQ(1, event.summary().value().size());
  ASSERT_EQ(strings::StrCat(kNodeName, ":", output_slot, ":", kDebugOpName),
            event.summary().value(0).node_name());

  Tensor a_prime(DT_FLOAT);
  ASSERT_TRUE(a_prime.FromProto(event.summary().value(0).tensor()));

  // Verify tensor shape and value.
  ASSERT_EQ(tensor_a_->shape(), a_prime.shape());
  for (int i = 0; i < a_prime.flat<float>().size(); ++i) {
    ASSERT_EQ(tensor_a_->flat<float>()(i), a_prime.flat<float>()(i));
  }

  // Tear down temporary file and directories.
  int64 undeleted_files = 0;
  int64 undeleted_dirs = 0;
  ASSERT_TRUE(
      env_->DeleteRecursively(test_dir, &undeleted_files, &undeleted_dirs)
          .ok());
  ASSERT_EQ(0, undeleted_files);
  ASSERT_EQ(0, undeleted_dirs);
}

TEST_F(DebugIOUtilsTest, DumpStringTensorToFileSunnyDay) {
  Initialize();

  const string test_dir = testing::TmpDir();

  const string kNodeName = "quux/grault/tensor_b";
  const string kDebugOpName = "DebugIdentity";
  const int32 output_slot = 1;
  uint64 wall_time = env_->NowMicros();

  string dump_file_name;
  Status s = DebugFileIO::DumpTensorToDir(kNodeName, output_slot, kDebugOpName,
                                          *tensor_b_, wall_time, test_dir,
                                          &dump_file_name);
  ASSERT_TRUE(s.ok());

  // Read the file into a Event proto.
  Event event;
  TF_ASSERT_OK(ReadEventFromFile(dump_file_name, &event));

  ASSERT_GE(wall_time, event.wall_time());
  ASSERT_EQ(1, event.summary().value().size());
  ASSERT_EQ(strings::StrCat(kNodeName, ":", output_slot, ":", kDebugOpName),
            event.summary().value(0).node_name());

  Tensor b_prime(DT_STRING);
  ASSERT_TRUE(b_prime.FromProto(event.summary().value(0).tensor()));

  // Verify tensor shape and value.
  ASSERT_EQ(tensor_b_->shape(), b_prime.shape());
  for (int i = 0; i < b_prime.flat<string>().size(); ++i) {
    ASSERT_EQ(tensor_b_->flat<string>()(i), b_prime.flat<string>()(i));
  }

  // Tear down temporary file and directories.
  int64 undeleted_files = 0;
  int64 undeleted_dirs = 0;
  ASSERT_TRUE(
      env_->DeleteRecursively(test_dir, &undeleted_files, &undeleted_dirs)
          .ok());
  ASSERT_EQ(0, undeleted_files);
  ASSERT_EQ(0, undeleted_dirs);
}

TEST_F(DebugIOUtilsTest, DumpTensorToFileCannotCreateDirectory) {
  Initialize();

  // First, create the file at the path.
  const string test_dir = testing::TmpDir();
  const string txt_file_name = strings::StrCat(test_dir, "/baz");

  if (!env_->FileExists(test_dir)) {
    ASSERT_TRUE(env_->CreateDir(test_dir).ok());
  }
  ASSERT_FALSE(env_->FileExists(txt_file_name));

  std::unique_ptr<WritableFile> file;
  ASSERT_TRUE(env_->NewWritableFile(txt_file_name, &file).ok());
  file->Append("text in baz");
  file->Flush();
  file->Close();

  // Verify that the path exists and that it is a file, not a directory.
  ASSERT_TRUE(env_->FileExists(txt_file_name));
  ASSERT_FALSE(env_->IsDirectory(txt_file_name).ok());

  // Second, try to dump the tensor to a path that requires "baz" to be a
  // directory, which should lead to an error.
  const string kNodeName = "baz/tensor_a";
  const string kDebugOpName = "DebugIdentity";
  const int32 output_slot = 0;
  uint64 wall_time = env_->NowMicros();

  string dump_file_name;
  Status s = DebugFileIO::DumpTensorToDir(kNodeName, output_slot, kDebugOpName,
                                          *tensor_a_, wall_time, test_dir,
                                          &dump_file_name);
  ASSERT_FALSE(s.ok());

  // Tear down temporary file and directories.
  int64 undeleted_files = 0;
  int64 undeleted_dirs = 0;
  ASSERT_TRUE(
      env_->DeleteRecursively(test_dir, &undeleted_files, &undeleted_dirs)
          .ok());
  ASSERT_EQ(0, undeleted_files);
  ASSERT_EQ(0, undeleted_dirs);
}

TEST_F(DebugIOUtilsTest, PublishTensorToMultipleFileURLs) {
  Initialize();

  const int kNumDumpRoots = 3;
  const string kNodeName = "foo/bar/qux/tensor_a";
  const string kDebugOpName = "DebugIdentity";
  const int32 output_slot = 0;

  uint64 wall_time = env_->NowMicros();

  std::vector<string> dump_roots;
  std::vector<string> dump_file_paths;
  std::vector<string> urls;
  for (int i = 0; i < kNumDumpRoots; ++i) {
    string dump_root = strings::StrCat(testing::TmpDir(), "/", i);

    dump_roots.push_back(dump_root);
    dump_file_paths.push_back(DebugFileIO::GetDumpFilePath(
        dump_root, kNodeName, output_slot, kDebugOpName, wall_time));
    urls.push_back(strings::StrCat("file://", dump_root));
  }

  for (int i = 1; i < kNumDumpRoots; ++i) {
    ASSERT_NE(dump_roots[0], dump_roots[i]);
  }

  const string tensor_name = strings::StrCat(kNodeName, ":", output_slot);
  const string debug_node_name =
      strings::StrCat(tensor_name, ":", kDebugOpName);
  Status s = DebugIO::PublishDebugTensor(tensor_name, kDebugOpName, *tensor_a_,
                                         wall_time, urls);
  ASSERT_TRUE(s.ok());

  // Try reading the file into a Event proto.
  for (int i = 0; i < kNumDumpRoots; ++i) {
    // Read the file into a Event proto.
    Event event;
    TF_ASSERT_OK(ReadEventFromFile(dump_file_paths[i], &event));

    ASSERT_GE(wall_time, event.wall_time());
    ASSERT_EQ(1, event.summary().value().size());
    ASSERT_EQ(debug_node_name, event.summary().value(0).node_name());

    Tensor a_prime(DT_FLOAT);
    ASSERT_TRUE(a_prime.FromProto(event.summary().value(0).tensor()));

    // Verify tensor shape and value.
    ASSERT_EQ(tensor_a_->shape(), a_prime.shape());
    for (int i = 0; i < a_prime.flat<float>().size(); ++i) {
      ASSERT_EQ(tensor_a_->flat<float>()(i), a_prime.flat<float>()(i));
    }
  }

  // Tear down temporary file and directories.
  for (int i = 0; i < kNumDumpRoots; ++i) {
    int64 undeleted_files = 0;
    int64 undeleted_dirs = 0;
    ASSERT_TRUE(env_->DeleteRecursively(dump_roots[i], &undeleted_files,
                                        &undeleted_dirs)
                    .ok());
    ASSERT_EQ(0, undeleted_files);
    ASSERT_EQ(0, undeleted_dirs);
  }
}

TEST_F(DebugIOUtilsTest, PublishTensorConcurrentlyToPartiallyOverlappingPaths) {
  Initialize();

  const int kConcurrentPubs = 3;
  const string kNodeName = "tensor_a";
  const string kDebugOpName = "DebugIdentity";
  const int32 kOutputSlot = 0;

  thread::ThreadPool* tp =
      new thread::ThreadPool(Env::Default(), "test", kConcurrentPubs);
  uint64 wall_time = env_->NowMicros();

  const string dump_root_base = testing::TmpDir();
  const string tensor_name = strings::StrCat(kNodeName, ":", kOutputSlot);
  const string debug_node_name =
      strings::StrCat(tensor_name, ":", kDebugOpName);

  mutex mu;
  std::vector<string> dump_roots GUARDED_BY(mu);
  std::vector<string> dump_file_paths GUARDED_BY(mu);

  int dump_count GUARDED_BY(mu) = 0;
  int done_count GUARDED_BY(mu) = 0;
  Notification all_done;

  auto fn = [this, &dump_count, &done_count, &mu, &dump_root_base, &dump_roots,
             &dump_file_paths, &wall_time, &tensor_name, &debug_node_name,
             &kNodeName, &kDebugOpName, &kConcurrentPubs, &all_done]() {
    // "gumpy" is the shared directory part of the path.
    string dump_root;
    string debug_url;
    {
      mutex_lock l(mu);
      dump_root =
          strings::StrCat(dump_root_base, "grumpy/", "dump_", dump_count++);

      dump_roots.push_back(dump_root);
      dump_file_paths.push_back(DebugFileIO::GetDumpFilePath(
          dump_root, kNodeName, kOutputSlot, kDebugOpName, wall_time));

      debug_url = strings::StrCat("file://", dump_root);
    }

    std::vector<string> urls;
    urls.push_back(debug_url);
    Status s = DebugIO::PublishDebugTensor(tensor_name, kDebugOpName,
                                           *tensor_a_, wall_time, urls);
    ASSERT_TRUE(s.ok());

    {
      mutex_lock l(mu);

      done_count++;
      if (done_count == kConcurrentPubs) {
        all_done.Notify();
      }
    }
  };

  for (int i = 0; i < kConcurrentPubs; ++i) {
    tp->Schedule(fn);
  }

  // Wait for all dumping calls to finish.
  all_done.WaitForNotification();
  delete tp;

  {
    mutex_lock l(mu);

    for (int i = 1; i < kConcurrentPubs; ++i) {
      ASSERT_NE(dump_roots[0], dump_roots[i]);
    }

    // Try reading the file into a Event proto.
    for (int i = 0; i < kConcurrentPubs; ++i) {
      // Read the file into a Event proto.
      Event event;
      TF_ASSERT_OK(ReadEventFromFile(dump_file_paths[i], &event));

      ASSERT_GE(wall_time, event.wall_time());
      ASSERT_EQ(1, event.summary().value().size());
      ASSERT_EQ(debug_node_name, event.summary().value(0).node_name());

      Tensor a_prime(DT_FLOAT);
      ASSERT_TRUE(a_prime.FromProto(event.summary().value(0).tensor()));

      // Verify tensor shape and value.
      ASSERT_EQ(tensor_a_->shape(), a_prime.shape());
      for (int i = 0; i < a_prime.flat<float>().size(); ++i) {
        ASSERT_EQ(tensor_a_->flat<float>()(i), a_prime.flat<float>()(i));
      }
    }

    // Tear down temporary file and directories.
    int64 undeleted_files = 0;
    int64 undeleted_dirs = 0;
    ASSERT_TRUE(env_->DeleteRecursively(dump_root_base, &undeleted_files,
                                        &undeleted_dirs)
                    .ok());
    ASSERT_EQ(0, undeleted_files);
    ASSERT_EQ(0, undeleted_dirs);
  }
}

}  // namespace
}  // namespace tensorflow
