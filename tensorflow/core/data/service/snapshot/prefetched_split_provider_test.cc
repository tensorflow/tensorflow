/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/data/service/snapshot/prefetched_split_provider.h"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/lib/io/compression.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/split_provider.h"
#include "tensorflow/core/data/service/test_util.h"
#include "tensorflow/core/data/snapshot_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/tensor.h"
#include "tsl/platform/path.h"

namespace tensorflow {
namespace data {
namespace {

using ::testing::ElementsAreArray;
using ::testing::IsSupersetOf;
using ::testing::UnorderedElementsAreArray;
using ::tsl::testing::IsOkAndHolds;
using ::tsl::testing::StatusIs;

absl::StatusOr<std::vector<std::string>> TestDirs(size_t num_dirs) {
  std::vector<std::string> test_dirs;
  std::string base_dir;
  if (!tsl::Env::Default()->LocalTempFilename(&base_dir)) {
    return absl::FailedPreconditionError("Failed to create local temp file.");
  }
  TF_RETURN_IF_ERROR(tsl::Env::Default()->RecursivelyCreateDir(base_dir));

  for (size_t i = 0; i < num_dirs; ++i) {
    std::string test_dir =
        tsl::io::JoinPath(base_dir, absl::StrCat("test_dir_", i));
    TF_RETURN_IF_ERROR(tsl::Env::Default()->RecursivelyCreateDir(test_dir));
    test_dirs.push_back(std::move(test_dir));
  }
  return test_dirs;
}

absl::StatusOr<std::unique_ptr<SplitProvider>> RangeSplitProvider(
    int64_t range) {
  DatasetDef range_dataset = testing::RangeDataset(range);
  std::vector<std::unique_ptr<SplitProvider>> split_providers;
  TF_RETURN_IF_ERROR(CreateSplitProviders(range_dataset, split_providers));
  if (split_providers.size() != 1) {
    return absl::InternalError(
        absl::StrCat("Range dataset should have one split provider, got ",
                     split_providers.size(), "."));
  }
  return std::move(split_providers[0]);
}

template <class T>
T GetValue(const Tensor& tensor) {
  return tensor.unaligned_flat<T>().data()[0];
}

template <class T>
absl::StatusOr<T> GetValueFromFile(const std::string& filename) {
  snapshot_util::TFRecordReaderImpl reader(filename,
                                           tsl::io::compression::kNone);
  TF_RETURN_IF_ERROR(reader.Initialize(tsl::Env::Default()));
  TF_ASSIGN_OR_RETURN(std::vector<Tensor> tensors, reader.GetTensors());
  if (tensors.size() != 1) {
    return absl::InternalError(absl::StrCat(
        "A snapshot split file is expected to contain 1 tensor. Got ",
        tensors.size(), " tensors from ", filename, "."));
  }
  return GetValue<T>(tensors[0]);
}

template <class T>
absl::StatusOr<std::vector<T>> GetSplits(
    PrefetchedSplitProvider& prefetched_split_provider,
    const std::string& test_dir) {
  std::vector<T> splits;
  for (size_t i = 0;; ++i) {
    std::string target_split_path =
        tsl::io::JoinPath(test_dir, absl::StrCat("split_", i));
    TF_ASSIGN_OR_RETURN(std::optional<Tensor> split,
                        prefetched_split_provider.GetNext(target_split_path));
    if (!split.has_value()) {
      return splits;
    }

    T split_value = GetValue<T>(*split);
    TF_ASSIGN_OR_RETURN(T split_from_file,
                        GetValueFromFile<T>(target_split_path));
    if (split_value != split_from_file) {
      return absl::InternalError(
          absl::StrCat("Inconsistent splits. From buffer: ", split_value,
                       ", from file: ", split_from_file, "."));
    }
    splits.push_back(split_value);
  }
  return splits;
}

std::vector<int64_t> Range(int64_t range) {
  std::vector<int64_t> result(range);
  std::iota(result.begin(), result.end(), 0);
  return result;
}

class PrefetchedSplitProviderParamTest
    : public ::testing::TestWithParam<
          std::tuple<int64_t, size_t, size_t, size_t>> {
 protected:
  int64_t NumElements() const { return std::get<0>(GetParam()); }
  size_t NumClients() const { return std::get<1>(GetParam()); }
  size_t NumWriteThreads() const { return std::get<2>(GetParam()); }
  size_t BufferSizePerThread() const { return std::get<3>(GetParam()); }
};

TEST_P(PrefetchedSplitProviderParamTest, GetSplits) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<SplitProvider> split_provider,
                          RangeSplitProvider(NumElements()));
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::string> test_dirs,
                          TestDirs(/*num_dirs=*/2));
  PrefetchedSplitProvider prefetched_split_provider(
      std::move(split_provider), test_dirs[0], tsl::Env::Default(),
      NumWriteThreads(), BufferSizePerThread());
  EXPECT_THAT(GetSplits<int64_t>(prefetched_split_provider, test_dirs[1]),
              IsOkAndHolds(ElementsAreArray(Range(NumElements()))));
}

TEST_P(PrefetchedSplitProviderParamTest, ConcurrentGetSplits) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<SplitProvider> split_provider,
                          RangeSplitProvider(NumElements()));
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::string> test_dirs,
                          TestDirs(/*num_dirs=*/1 + NumClients()));
  PrefetchedSplitProvider prefetched_split_provider(
      std::move(split_provider), test_dirs[0], tsl::Env::Default(),
      NumWriteThreads(), BufferSizePerThread());

  absl::Mutex mu;  // Protects `splits`.
  std::vector<int64_t> splits;
  std::vector<std::unique_ptr<tsl::Thread>> client_threads;
  for (int i = 0; i < NumClients(); ++i) {
    client_threads.push_back(absl::WrapUnique(tsl::Env::Default()->StartThread(
        /*thread_options=*/{}, /*name=*/absl::StrCat("Client_", i),
        [i, &prefetched_split_provider, &splits, &test_dirs, &mu]() {
          TF_ASSERT_OK_AND_ASSIGN(
              std::vector<int64_t> splits_per_thread,
              GetSplits<int64_t>(prefetched_split_provider, test_dirs[1 + i]));
          EXPECT_TRUE(absl::c_is_sorted(splits_per_thread));
          absl::MutexLock l(&mu);
          absl::c_move(splits_per_thread, std::back_inserter(splits));
        })));
  }

  client_threads.clear();
  EXPECT_THAT(splits, UnorderedElementsAreArray(Range(NumElements())));
}

TEST_P(PrefetchedSplitProviderParamTest, Reset) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<SplitProvider> split_provider,
                          RangeSplitProvider(NumElements()));
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::string> test_dirs,
                          TestDirs(/*num_dirs=*/2));
  PrefetchedSplitProvider prefetched_split_provider(
      std::move(split_provider), test_dirs[0], tsl::Env::Default(),
      NumWriteThreads(), BufferSizePerThread());

  // The split provider produces elements from the beginning after being reset.
  for (int i = 0; i < 3; ++i) {
    EXPECT_THAT(GetSplits<int64_t>(prefetched_split_provider, test_dirs[1]),
                IsOkAndHolds(ElementsAreArray(Range(NumElements()))));
    TF_EXPECT_OK(prefetched_split_provider.Reset());
  }
}

TEST_P(PrefetchedSplitProviderParamTest, ConcurrentGetSplitsAndReset) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<SplitProvider> split_provider,
                          RangeSplitProvider(NumElements()));
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::string> test_dirs,
                          TestDirs(/*num_dirs=*/1 + NumClients()));
  PrefetchedSplitProvider prefetched_split_provider(
      std::move(split_provider), test_dirs[0], tsl::Env::Default(),
      NumWriteThreads(), BufferSizePerThread());

  absl::Mutex mu;  // Protects `splits`.
  std::vector<int64_t> splits;
  std::vector<std::unique_ptr<tsl::Thread>> client_threads;
  for (int i = 0; i < NumClients(); ++i) {
    client_threads.push_back(absl::WrapUnique(tsl::Env::Default()->StartThread(
        /*thread_options=*/{}, /*name=*/absl::StrCat("Client_", i),
        [i, &prefetched_split_provider, &splits, &test_dirs, &mu]() {
          TF_ASSERT_OK_AND_ASSIGN(
              std::vector<int64_t> splits_per_thread,
              GetSplits<int64_t>(prefetched_split_provider, test_dirs[1 + i]));
          absl::MutexLock l(&mu);
          absl::c_move(splits_per_thread, std::back_inserter(splits));
        })));
  }

  TF_EXPECT_OK(prefetched_split_provider.Reset());
  client_threads.clear();
  // The result should be `Range(NumElements())` plus a few extra splits read
  // before the split provider was reset.
  EXPECT_THAT(splits, IsSupersetOf(Range(NumElements())));
}

INSTANTIATE_TEST_SUITE_P(
    PrefetchedSplitProviderParams, PrefetchedSplitProviderParamTest,
    ::testing::Combine(
        /*NumElements*/ ::testing::Values(0, 10, 1000),
        /*NumClients*/ ::testing::Values(1, 5),
        /*NumWriteThreads*/ ::testing::Values(1, 10),
        /*BufferSizePerThread*/ ::testing::Values(1, 10000)));

TEST(PrefetchedSplitProviderTest, Cancellation) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<SplitProvider> split_provider,
                          RangeSplitProvider(999999));
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::string> test_dirs,
                          TestDirs(/*num_dirs=*/2));
  PrefetchedSplitProvider prefetched_split_provider(
      std::move(split_provider), test_dirs[0], tsl::Env::Default(),
      /*num_write_threads=*/2, /*buffer_size_per_thread=*/1);

  std::unique_ptr<tsl::Thread> client_thread =
      absl::WrapUnique(tsl::Env::Default()->StartThread(
          /*thread_options=*/{}, /*name=*/"client_thread",
          [&prefetched_split_provider, &test_dirs]() {
            EXPECT_THAT(
                GetSplits<int64_t>(prefetched_split_provider, test_dirs[1]),
                StatusIs(absl::StatusCode::kCancelled));
          }));

  prefetched_split_provider.Cancel();
  client_thread.reset();
}

TEST(PrefetchedSplitProviderTest, ShutdownWithUnreadSplits) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<SplitProvider> split_provider,
                          RangeSplitProvider(100));
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::string> test_dirs,
                          TestDirs(/*num_dirs=*/2));
  PrefetchedSplitProvider prefetched_split_provider(
      std::move(split_provider), test_dirs[0], tsl::Env::Default());
  TF_EXPECT_OK(prefetched_split_provider.Reset());
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
