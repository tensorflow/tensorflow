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
#include "tensorflow/core/data/service/snapshot/parallel_tfrecord_writer.h"

#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/lib/io/compression.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "tensorflow/core/data/service/byte_size.h"
#include "tensorflow/core/data/snapshot_utils.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"

namespace tensorflow {
namespace data {
namespace {

using ::testing::Each;
using ::testing::Eq;
using ::testing::Field;
using ::testing::Gt;
using ::testing::IsEmpty;
using ::testing::Le;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAreArray;
using ::tsl::testing::IsOkAndHolds;
using ::tsl::testing::StatusIs;

absl::StatusOr<std::string> TestDir() {
  std::string test_dir;
  if (!tsl::Env::Default()->LocalTempFilename(&test_dir)) {
    return absl::FailedPreconditionError("Failed to create local temp file.");
  }
  TF_RETURN_IF_ERROR(tsl::Env::Default()->RecursivelyCreateDir(test_dir));
  return test_dir;
}

class RangeIterator {
 public:
  explicit RangeIterator(const int64_t range) : range_(range) {}

  absl::Status GetNext(std::vector<Tensor>& element, bool& end_of_sequence) {
    end_of_sequence = (next_ >= range_);
    if (end_of_sequence) {
      return absl::OkStatus();
    }
    element = {Tensor{next_++}};
    return absl::OkStatus();
  }

  int64_t Cardinality() const { return range_; }

 private:
  const int64_t range_;
  int64_t next_ = 0;
};

absl::StatusOr<ParallelTFRecordWriter::FileToStatsMap> WriteRecords(
    ParallelTFRecordWriter& writer, RangeIterator& iterator,
    bool finalize_writer = true) {
  std::vector<Tensor> record;
  bool end_of_sequence = false;
  TF_RETURN_IF_ERROR(iterator.GetNext(record, end_of_sequence));
  while (!end_of_sequence) {
    TF_RETURN_IF_ERROR(writer.Write(record));
    TF_RETURN_IF_ERROR(iterator.GetNext(record, end_of_sequence));
  }
  if (finalize_writer) {
    return writer.Finalize();
  }
  return ParallelTFRecordWriter::FileToStatsMap();
}

template <class T>
absl::StatusOr<std::vector<T>> ReadRecords(const std::string& filename,
                                           const std::string& compression) {
  snapshot_util::TFRecordReader reader(filename, compression,
                                       DataTypeVector{DT_INT64});
  TF_RETURN_IF_ERROR(reader.Initialize(tsl::Env::Default()));

  std::vector<T> result;
  while (true) {
    std::vector<Tensor> record;
    absl::Status status = reader.ReadTensors(&record);
    if (absl::IsOutOfRange(status)) {
      break;
    }
    TF_RETURN_IF_ERROR(status);

    for (const Tensor& tensor : record) {
      result.push_back(tensor.unaligned_flat<T>().data()[0]);
    }
  }
  return result;
}

template <class T>
absl::StatusOr<std::vector<T>> ReadRecords(
    const std::vector<std::string>& filenames, const std::string& compression) {
  std::vector<T> result;
  for (const std::string& filename : filenames) {
    TF_ASSIGN_OR_RETURN(std::vector<T> records,
                        ReadRecords<T>(filename, compression));
    absl::c_move(records, std::back_inserter(result));
  }
  return result;
}

std::vector<int64_t> Range(int64_t range) {
  std::vector<int64_t> result(range);
  std::iota(result.begin(), result.end(), 0);
  return result;
}

std::vector<int64_t> Repeat(const std::vector<int64_t>& values,
                            int64_t repeat) {
  std::vector<int64_t> result;
  for (int64_t i = 0; i < repeat; ++i) {
    absl::c_copy(values, std::back_inserter(result));
  }
  return result;
}

template <class K, class V>
std::pair<std::vector<K>, std::vector<V>> Unzip(
    const absl::flat_hash_map<K, V>& m) {
  std::vector<K> keys;
  std::vector<V> values;
  for (const auto& [k, v] : m) {
    keys.push_back(k);
    values.push_back(v);
  }
  return std::make_pair(keys, values);
}

class ParallelTFRecordWriterParamTest
    : public ::testing::TestWithParam<std::tuple<
          int64_t, int64_t, ByteSize, int64_t, int64_t, std::string>> {
 protected:
  int64_t NumElements() const { return std::get<0>(GetParam()); }
  int64_t NumClients() const { return std::get<1>(GetParam()); }
  ByteSize MaxFileSize() const { return std::get<2>(GetParam()); }
  int64_t NumWriteThreads() const { return std::get<3>(GetParam()); }
  int64_t BufferSize() const { return std::get<4>(GetParam()); }
  std::string Compression() const { return std::get<5>(GetParam()); }

  void VerifyFileStats(
      const std::vector<ParallelTFRecordWriter::FileStats>& file_stats,
      int64_t expected_num_elements) const {
    // Verifies total record counts.
    auto add_num_elements = [](int64_t num_elements,
                               const ParallelTFRecordWriter::FileStats& stats) {
      return num_elements + stats.num_records;
    };
    EXPECT_EQ(absl::c_accumulate(file_stats, 0, add_num_elements),
              expected_num_elements);

    // There should be no empty file.
    EXPECT_THAT(
        file_stats,
        Each(Field(&ParallelTFRecordWriter::FileStats::num_records, Gt(0))));

    // Each file contains at most `MaxFileSize()` + sizeof(one record) bytes.
    EXPECT_THAT(file_stats,
                Each(Field(&ParallelTFRecordWriter::FileStats::estimated_size,
                           Le(MaxFileSize() + ByteSize::Bytes(16)))));

    if (MaxFileSize() <= ByteSize::Bytes(1)) {
      // In this case, each file contains one record.
      EXPECT_THAT(
          file_stats,
          Each(Field(&ParallelTFRecordWriter::FileStats::num_records, Eq(1))));
      EXPECT_THAT(file_stats, SizeIs(expected_num_elements));
    }

    if (MaxFileSize() >= ByteSize::GB(1)) {
      // In this case, each thread writes at most one file.
      EXPECT_THAT(file_stats, SizeIs(Le(NumWriteThreads())));
    }
  }
};

TEST_P(ParallelTFRecordWriterParamTest, WriteRecords) {
  TF_ASSERT_OK_AND_ASSIGN(std::string test_dir, TestDir());
  ParallelTFRecordWriter parallel_tfrecord_writer(
      test_dir, Compression(), tsl::Env::Default(), MaxFileSize(),
      NumWriteThreads(), BufferSize());

  RangeIterator range_iterator(NumElements());
  TF_ASSERT_OK_AND_ASSIGN(
      ParallelTFRecordWriter::FileToStatsMap file_stats,
      WriteRecords(parallel_tfrecord_writer, range_iterator));

  const auto [files, stats] = Unzip(file_stats);
  EXPECT_THAT(ReadRecords<int64_t>(files, Compression()),
              absl_testing::IsOkAndHolds(
                  UnorderedElementsAreArray(Range(NumElements()))));
  VerifyFileStats(stats, NumElements());
}

TEST_P(ParallelTFRecordWriterParamTest, ConcurrentWrites) {
  TF_ASSERT_OK_AND_ASSIGN(std::string test_dir, TestDir());
  ParallelTFRecordWriter parallel_tfrecord_writer(
      test_dir, Compression(), tsl::Env::Default(), MaxFileSize(),
      NumWriteThreads(), BufferSize());

  std::vector<std::unique_ptr<tsl::Thread>> client_threads;
  for (int i = 0; i < NumClients(); ++i) {
    client_threads.push_back(absl::WrapUnique(tsl::Env::Default()->StartThread(
        /*thread_options=*/{}, /*name=*/absl::StrCat("Client_", i),
        [this, &parallel_tfrecord_writer]() {
          RangeIterator range_iterator(NumElements());
          TF_ASSERT_OK(WriteRecords(parallel_tfrecord_writer, range_iterator,
                                    /*finalize_writer=*/false)
                           .status());
        })));
  }
  client_threads.clear();
  TF_ASSERT_OK_AND_ASSIGN(ParallelTFRecordWriter::FileToStatsMap file_stats,
                          parallel_tfrecord_writer.Finalize());

  const auto [files, stats] = Unzip(file_stats);
  EXPECT_THAT(ReadRecords<int64_t>(files, Compression()),
              absl_testing::IsOkAndHolds(UnorderedElementsAreArray(
                  Repeat(Range(NumElements()), NumClients()))));
  VerifyFileStats(stats, NumElements() * NumClients());
}

INSTANTIATE_TEST_SUITE_P(ParallelTFRecordWriterParams,
                         ParallelTFRecordWriterParamTest,
                         ::testing::Combine(
                             /*NumElements*/ ::testing::Values(0, 1, 100),
                             /*NumClients*/ ::testing::Values(1, 5),
                             /*MaxFileSize*/
                             ::testing::Values(ByteSize::Bytes(1),
                                               ByteSize::Bytes(100),
                                               ByteSize::GB(1)),
                             /*NumWriteThreads*/ ::testing::Values(1, 5),
                             /*BufferSize*/ ::testing::Values(1, 10000),
                             /*Compression*/
                             ::testing::Values(tsl::io::compression::kNone,
                                               tsl::io::compression::kSnappy,
                                               tsl::io::compression::kZlib)));

TEST(ParallelTFRecordWriterTest, WriteNoRecord) {
  TF_ASSERT_OK_AND_ASSIGN(std::string test_dir, TestDir());
  ParallelTFRecordWriter parallel_tfrecord_writer(
      test_dir, tsl::io::compression::kNone, tsl::Env::Default());
  TF_ASSERT_OK_AND_ASSIGN(ParallelTFRecordWriter::FileToStatsMap file_stats,
                          parallel_tfrecord_writer.Finalize());

  const auto [files, stats] = Unzip(file_stats);
  EXPECT_THAT(ReadRecords<int64_t>(files, tsl::io::compression::kNone),
              absl_testing::IsOkAndHolds(IsEmpty()));
}

TEST(ParallelTFRecordWriterTest, CannotWriteFinalizedWriter) {
  TF_ASSERT_OK_AND_ASSIGN(std::string test_dir, TestDir());
  std::string file_prefix = "file";
  ParallelTFRecordWriter parallel_tfrecord_writer(
      test_dir, tsl::io::compression::kNone, tsl::Env::Default());

  std::unique_ptr<tsl::Thread> client_thread =
      absl::WrapUnique(tsl::Env::Default()->StartThread(
          /*thread_options=*/{}, /*name=*/"Client",
          [&parallel_tfrecord_writer]() {
            RangeIterator range_iterator(std::numeric_limits<int64_t>::max());
            EXPECT_THAT(
                WriteRecords(parallel_tfrecord_writer, range_iterator),
                absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition));
          }));

  parallel_tfrecord_writer.Finalize().status().IgnoreError();
  client_thread.reset();
}

TEST(ParallelTFRecordWriterTest, DirectoryDoesNotExist) {
  ParallelTFRecordWriter parallel_tfrecord_writer("/directory/does/not/exists",
                                                  tsl::io::compression::kNone,
                                                  tsl::Env::Default());

  RangeIterator range_iterator(10);
  std::vector<Tensor> element;
  bool end_of_sequence = false;
  TF_ASSERT_OK(range_iterator.GetNext(element, end_of_sequence));
  parallel_tfrecord_writer.Write(element).IgnoreError();
  EXPECT_THAT(parallel_tfrecord_writer.Finalize().status(),
              absl_testing::StatusIs(absl::StatusCode::kNotFound));
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
