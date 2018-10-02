// Copyright 2017 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "tensorflow/contrib/ffmpeg/ffmpeg_lib.h"

#include <array>
#include <set>
#include <string>
#include <vector>

#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace ffmpeg {
namespace {

TEST(FfmpegLibTest, TestTempDirectoryThreading) {
  // Testing a fix for a bug that allowed different threads to create
  // conflicting temp files.
  // See github.com/tensorflow/tensorflow/issues/5804 for details.
  const int32 kNumThreads = 10;
  const int32 kNumWorkItems = 10000;
  static constexpr size_t kStringsPerItem = 100;
  Env* environment = Env::Default();
  thread::ThreadPool pool(environment, "test", kNumThreads);

  mutex mu;
  std::vector<string> temp_filenames;
  temp_filenames.reserve(kNumWorkItems * kStringsPerItem);

  // Queue a large number of work items for the threads to process. Each work
  // item creates a temp file and then deletes it.
  for (int i = 0; i < kNumWorkItems; ++i) {
    pool.Schedule([&mu, &temp_filenames, environment]() {
      std::array<string, kStringsPerItem> buffer;
      for (int32 j = 0; j < kStringsPerItem; ++j) {
        buffer[j] = io::GetTempFilename("mp3");
        TF_QCHECK_OK(environment->DeleteFile(buffer[j]));
      }
      mutex_lock l(mu);
      for (const auto& fn : buffer) {
        temp_filenames.push_back(fn);
      }
    });
  }

  // Wait until all work items are complete.
  while (true) {
    mutex_lock l(mu);
    if (temp_filenames.size() == kNumWorkItems * kStringsPerItem) {
      break;
    }
  }

  // Check that no duplicates are created.
  std::set<string> unique_filenames;
  mutex_lock l(mu);
  for (const auto& fn : temp_filenames) {
    ASSERT_TRUE(unique_filenames.insert(fn).second);
  }
}

}  // namespace
}  // namespace ffmpeg
}  // namespace tensorflow
