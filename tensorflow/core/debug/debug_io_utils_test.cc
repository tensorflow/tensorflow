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

#include <cstdlib>
#include <unordered_set>

#include "tensorflow/core/debug/debug_io_utils.h"

#include "tensorflow/core/debug/debug_callback_registry.h"
#include "tensorflow/core/debug/debug_node_key.h"
#include "tensorflow/core/debug/debugger_event_metadata.pb.h"
#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
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
    tensor_b_->flat<tstring>()(0) = "corge";
    tensor_b_->flat<tstring>()(1) = "garply";
  }

  Env* env_;
  std::unique_ptr<Tensor> tensor_a_;
  std::unique_ptr<Tensor> tensor_b_;
};

TEST_F(DebugIOUtilsTest, ConstructDebugNodeKey) {
  DebugNodeKey debug_node_key("/job:worker/replica:1/task:0/device:GPU:2",
                              "hidden_1/MatMul", 0, "DebugIdentity");
  EXPECT_EQ("/job:worker/replica:1/task:0/device:GPU:2",
            debug_node_key.device_name);
  EXPECT_EQ("hidden_1/MatMul", debug_node_key.node_name);
  EXPECT_EQ(0, debug_node_key.output_slot);
  EXPECT_EQ("DebugIdentity", debug_node_key.debug_op);
  EXPECT_EQ("hidden_1/MatMul:0:DebugIdentity", debug_node_key.debug_node_name);
  EXPECT_EQ("_tfdbg_device_,job_worker,replica_1,task_0,device_GPU_2",
            debug_node_key.device_path);
}

TEST_F(DebugIOUtilsTest, EqualityOfDebugNodeKeys) {
  const DebugNodeKey debug_node_key_1("/job:worker/replica:1/task:0/gpu:2",
                                      "hidden_1/MatMul", 0, "DebugIdentity");
  const DebugNodeKey debug_node_key_2("/job:worker/replica:1/task:0/gpu:2",
                                      "hidden_1/MatMul", 0, "DebugIdentity");
  const DebugNodeKey debug_node_key_3("/job:worker/replica:1/task:0/gpu:2",
                                      "hidden_1/BiasAdd", 0, "DebugIdentity");
  const DebugNodeKey debug_node_key_4("/job:worker/replica:1/task:0/gpu:2",
                                      "hidden_1/MatMul", 0,
                                      "DebugNumericSummary");
  EXPECT_EQ(debug_node_key_1, debug_node_key_2);
  EXPECT_NE(debug_node_key_1, debug_node_key_3);
  EXPECT_NE(debug_node_key_1, debug_node_key_4);
  EXPECT_NE(debug_node_key_3, debug_node_key_4);
}

TEST_F(DebugIOUtilsTest, DebugNodeKeysIsHashable) {
  const DebugNodeKey debug_node_key_1("/job:worker/replica:1/task:0/gpu:2",
                                      "hidden_1/MatMul", 0, "DebugIdentity");
  const DebugNodeKey debug_node_key_2("/job:worker/replica:1/task:0/gpu:2",
                                      "hidden_1/MatMul", 0, "DebugIdentity");
  const DebugNodeKey debug_node_key_3("/job:worker/replica:1/task:0/gpu:2",
                                      "hidden_1/BiasAdd", 0, "DebugIdentity");

  std::unordered_set<DebugNodeKey> keys;
  keys.insert(debug_node_key_1);
  ASSERT_EQ(1, keys.size());
  keys.insert(debug_node_key_3);
  ASSERT_EQ(2, keys.size());
  keys.erase(debug_node_key_2);
  ASSERT_EQ(1, keys.size());
}

TEST_F(DebugIOUtilsTest, DumpFloatTensorToFileSunnyDay) {
  Initialize();

  const string test_dir = testing::TmpDir();

  // Append levels of nonexisting directories, to test that the function can
  // create directories.
  const uint64 wall_time = env_->NowMicros();
  const DebugNodeKey kDebugNodeKey("/job:localhost/replica:0/task:0/cpu:0",
                                   "foo/bar/qux/tensor_a", 0, "DebugIdentity");

  string dump_file_path;
  TF_ASSERT_OK(DebugFileIO::DumpTensorToDir(
      kDebugNodeKey, *tensor_a_, wall_time, test_dir, &dump_file_path));

  // Read the file into a Event proto.
  Event event;
  TF_ASSERT_OK(ReadEventFromFile(dump_file_path, &event));

  ASSERT_GE(wall_time, event.wall_time());
  ASSERT_EQ(1, event.summary().value().size());
  ASSERT_EQ(kDebugNodeKey.debug_node_name,
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

  const DebugNodeKey kDebugNodeKey("/job:localhost/replica:0/task:0/cpu:0",
                                   "quux/grault/tensor_b", 1, "DebugIdentity");
  const uint64 wall_time = env_->NowMicros();

  string dump_file_name;
  Status s = DebugFileIO::DumpTensorToDir(kDebugNodeKey, *tensor_b_, wall_time,
                                          test_dir, &dump_file_name);
  ASSERT_TRUE(s.ok());

  // Read the file into a Event proto.
  Event event;
  TF_ASSERT_OK(ReadEventFromFile(dump_file_name, &event));

  ASSERT_GE(wall_time, event.wall_time());
  ASSERT_EQ(1, event.summary().value().size());
  ASSERT_EQ(kDebugNodeKey.node_name, event.summary().value(0).tag());
  ASSERT_EQ(kDebugNodeKey.debug_node_name,
            event.summary().value(0).node_name());

  // Determine and validate some information from the metadata.
  third_party::tensorflow::core::debug::DebuggerEventMetadata metadata;
  auto status = tensorflow::protobuf::util::JsonStringToMessage(
      event.summary().value(0).metadata().plugin_data().content(), &metadata);
  ASSERT_TRUE(status.ok());
  ASSERT_EQ(kDebugNodeKey.device_name, metadata.device());
  ASSERT_EQ(kDebugNodeKey.output_slot, metadata.output_slot());

  Tensor b_prime(DT_STRING);
  ASSERT_TRUE(b_prime.FromProto(event.summary().value(0).tensor()));

  // Verify tensor shape and value.
  ASSERT_EQ(tensor_b_->shape(), b_prime.shape());
  for (int i = 0; i < b_prime.flat<tstring>().size(); ++i) {
    ASSERT_EQ(tensor_b_->flat<tstring>()(i), b_prime.flat<tstring>()(i));
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
  const string kDeviceName = "/job:localhost/replica:0/task:0/cpu:0";
  const DebugNodeKey kDebugNodeKey(kDeviceName, "baz/tensor_a", 0,
                                   "DebugIdentity");
  const string txt_file_dir =
      io::JoinPath(test_dir, DebugNodeKey::DeviceNameToDevicePath(kDeviceName));
  const string txt_file_name = io::JoinPath(txt_file_dir, "baz");
  if (!env_->FileExists(txt_file_dir).ok()) {
    ASSERT_TRUE(env_->RecursivelyCreateDir(txt_file_dir).ok());
  }
  ASSERT_EQ(error::Code::NOT_FOUND, env_->FileExists(txt_file_name).code());

  std::unique_ptr<WritableFile> file;
  ASSERT_TRUE(env_->NewWritableFile(txt_file_name, &file).ok());
  TF_EXPECT_OK(file->Append("text in baz"));
  TF_EXPECT_OK(file->Flush());
  TF_ASSERT_OK(file->Close());

  // Verify that the path exists and that it is a file, not a directory.
  ASSERT_TRUE(env_->FileExists(txt_file_name).ok());
  ASSERT_FALSE(env_->IsDirectory(txt_file_name).ok());

  // Second, try to dump the tensor to a path that requires "baz" to be a
  // directory, which should lead to an error.

  const uint64 wall_time = env_->NowMicros();

  string dump_file_name;
  Status s = DebugFileIO::DumpTensorToDir(kDebugNodeKey, *tensor_a_, wall_time,
                                          test_dir, &dump_file_name);
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
  const DebugNodeKey kDebugNodeKey("/job:localhost/replica:0/task:0/cpu:0",
                                   "foo/bar/qux/tensor_a", 0, "DebugIdentity");
  const uint64 wall_time = env_->NowMicros();

  std::vector<string> dump_roots;
  std::vector<string> dump_file_paths;
  std::vector<string> urls;
  for (int i = 0; i < kNumDumpRoots; ++i) {
    string dump_root = strings::StrCat(testing::TmpDir(), "/", i);

    dump_roots.push_back(dump_root);
    dump_file_paths.push_back(
        DebugFileIO::GetDumpFilePath(dump_root, kDebugNodeKey, wall_time));
    urls.push_back(strings::StrCat("file://", dump_root));
  }

  for (int i = 1; i < kNumDumpRoots; ++i) {
    ASSERT_NE(dump_roots[0], dump_roots[i]);
  }

  Status s =
      DebugIO::PublishDebugTensor(kDebugNodeKey, *tensor_a_, wall_time, urls);
  ASSERT_TRUE(s.ok());

  // Try reading the file into a Event proto.
  for (int i = 0; i < kNumDumpRoots; ++i) {
    // Read the file into a Event proto.
    Event event;
    TF_ASSERT_OK(ReadEventFromFile(dump_file_paths[i], &event));

    ASSERT_GE(wall_time, event.wall_time());
    ASSERT_EQ(1, event.summary().value().size());
    ASSERT_EQ(kDebugNodeKey.node_name, event.summary().value(0).tag());
    ASSERT_EQ(kDebugNodeKey.debug_node_name,
              event.summary().value(0).node_name());

    // Determine and validate some information from the metadata.
    third_party::tensorflow::core::debug::DebuggerEventMetadata metadata;
    auto status = tensorflow::protobuf::util::JsonStringToMessage(
        event.summary().value(0).metadata().plugin_data().content(), &metadata);
    ASSERT_TRUE(status.ok());
    ASSERT_EQ(kDebugNodeKey.device_name, metadata.device());
    ASSERT_EQ(kDebugNodeKey.output_slot, metadata.output_slot());

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

TEST_F(DebugIOUtilsTest, PublishTensorToMemoryCallback) {
  Initialize();

  const DebugNodeKey kDebugNodeKey("/job:localhost/replica:0/task:0/cpu:0",
                                   "foo/bar/qux/tensor_a", 0, "DebugIdentity");
  const uint64 wall_time = env_->NowMicros();

  bool called = false;
  std::vector<string> urls = {"memcbk://test_callback"};
  ;

  auto* callback_registry = DebugCallbackRegistry::singleton();
  callback_registry->RegisterCallback(
      "test_callback", [this, &kDebugNodeKey, &called](const DebugNodeKey& key,
                                                       const Tensor& tensor) {
        called = true;
        ASSERT_EQ(kDebugNodeKey.device_name, key.device_name);
        ASSERT_EQ(kDebugNodeKey.node_name, key.node_name);
        ASSERT_EQ(tensor_a_->shape(), tensor.shape());
        for (int i = 0; i < tensor.flat<float>().size(); ++i) {
          ASSERT_EQ(tensor_a_->flat<float>()(i), tensor.flat<float>()(i));
        }
      });

  Status s =
      DebugIO::PublishDebugTensor(kDebugNodeKey, *tensor_a_, wall_time, urls);
  ASSERT_TRUE(s.ok());
  ASSERT_TRUE(called);

  callback_registry->UnregisterCallback("test_callback");
}

TEST_F(DebugIOUtilsTest, PublishTensorConcurrentlyToPartiallyOverlappingPaths) {
  Initialize();

  const int kConcurrentPubs = 3;
  const DebugNodeKey kDebugNodeKey("/job:localhost/replica:0/task:0/cpu:0",
                                   "tensor_a", 0, "DebugIdentity");

  thread::ThreadPool* tp =
      new thread::ThreadPool(Env::Default(), "test", kConcurrentPubs);
  const uint64 wall_time = env_->NowMicros();
  const string dump_root_base = testing::TmpDir();

  mutex mu;
  std::vector<string> dump_roots TF_GUARDED_BY(mu);
  std::vector<string> dump_file_paths TF_GUARDED_BY(mu);

  int dump_count TF_GUARDED_BY(mu) = 0;
  int done_count TF_GUARDED_BY(mu) = 0;
  Notification all_done;

  auto fn = [this, &dump_count, &done_count, &mu, &dump_root_base, &dump_roots,
             &dump_file_paths, &wall_time, &kDebugNodeKey, &kConcurrentPubs,
             &all_done]() {
    // "gumpy" is the shared directory part of the path.
    string dump_root;
    string debug_url;
    {
      mutex_lock l(mu);
      dump_root =
          strings::StrCat(dump_root_base, "grumpy/", "dump_", dump_count++);

      dump_roots.push_back(dump_root);
      dump_file_paths.push_back(
          DebugFileIO::GetDumpFilePath(dump_root, kDebugNodeKey, wall_time));

      debug_url = strings::StrCat("file://", dump_root);
    }

    std::vector<string> urls;
    urls.push_back(debug_url);

    Status s =
        DebugIO::PublishDebugTensor(kDebugNodeKey, *tensor_a_, wall_time, urls);
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
      ASSERT_EQ(kDebugNodeKey.node_name, event.summary().value(0).tag());
      ASSERT_EQ(kDebugNodeKey.debug_node_name,
                event.summary().value(0).node_name());

      // Determine and validate some information from the metadata.
      third_party::tensorflow::core::debug::DebuggerEventMetadata metadata;
      auto status = tensorflow::protobuf::util::JsonStringToMessage(
          event.summary().value(0).metadata().plugin_data().content(),
          &metadata);
      ASSERT_TRUE(status.ok());
      ASSERT_EQ(kDebugNodeKey.device_name, metadata.device());
      ASSERT_EQ(kDebugNodeKey.output_slot, metadata.output_slot());

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

class DiskUsageLimitTest : public ::testing::Test {
 public:
  void Initialize() {
    setenv("TFDBG_DISK_BYTES_LIMIT", "", 1);
    DebugFileIO::resetDiskByteUsage();
    DebugFileIO::global_disk_bytes_limit_ = 0;
  }
};

TEST_F(DiskUsageLimitTest, RequestWithZeroByteIsOkay) {
  Initialize();
  ASSERT_TRUE(DebugFileIO::requestDiskByteUsage(0L));
}

TEST_F(DiskUsageLimitTest, ExceedingLimitAfterOneCall) {
  Initialize();
  ASSERT_FALSE(DebugFileIO::requestDiskByteUsage(100L * 1024L * 1024L * 1024L));
}

TEST_F(DiskUsageLimitTest, ExceedingLimitAfterTwoCalls) {
  Initialize();
  ASSERT_TRUE(DebugFileIO::requestDiskByteUsage(50L * 1024L * 1024L * 1024L));
  ASSERT_FALSE(DebugFileIO::requestDiskByteUsage(50L * 1024L * 1024L * 1024L));
  ASSERT_TRUE(DebugFileIO::requestDiskByteUsage(1024L));
}

TEST_F(DiskUsageLimitTest, ResetDiskByteUsageWorks) {
  Initialize();
  ASSERT_TRUE(DebugFileIO::requestDiskByteUsage(50L * 1024L * 1024L * 1024L));
  ASSERT_FALSE(DebugFileIO::requestDiskByteUsage(50L * 1024L * 1024L * 1024L));
  DebugFileIO::resetDiskByteUsage();
  ASSERT_TRUE(DebugFileIO::requestDiskByteUsage(50L * 1024L * 1024L * 1024L));
}

TEST_F(DiskUsageLimitTest, CustomEnvVarIsObeyed) {
  Initialize();
  setenv("TFDBG_DISK_BYTES_LIMIT", "1024", 1);
  ASSERT_FALSE(DebugFileIO::requestDiskByteUsage(1024L));
  ASSERT_TRUE(DebugFileIO::requestDiskByteUsage(1000L));
  ASSERT_TRUE(DebugFileIO::requestDiskByteUsage(23L));
  ASSERT_FALSE(DebugFileIO::requestDiskByteUsage(1L));
  DebugFileIO::resetDiskByteUsage();
  ASSERT_TRUE(DebugFileIO::requestDiskByteUsage(1023L));
}

}  // namespace
}  // namespace tensorflow
