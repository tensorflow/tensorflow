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
#include "tensorflow/core/data/service/snapshot/snapshot_chunk_provider.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/status_to_from_proto.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/protobuf/status.pb.h"
#include "tensorflow/core/data/serialization_utils.h"
#include "tensorflow/core/data/service/snapshot/file_utils.h"
#include "tensorflow/core/data/service/snapshot/path_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tsl/platform/path.h"
#include "tsl/platform/tstring.h"

namespace tensorflow {
namespace data {
namespace {

using ::testing::ElementsAreArray;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::UnorderedElementsAreArray;
using ::tsl::testing::IsOkAndHolds;
using ::tsl::testing::StatusIs;

absl::StatusOr<std::string> CreateSnapshotDirectory() {
  std::string snapshot_path;
  if (!tsl::Env::Default()->LocalTempFilename(&snapshot_path)) {
    return absl::FailedPreconditionError(
        "Failed to create local temp file for snapshot.");
  }
  TF_RETURN_IF_ERROR(tsl::Env::Default()->RecursivelyCreateDir(
      CommittedChunksDirectory(snapshot_path)));
  return snapshot_path;
}

absl::Status WriteChunk(absl::string_view snapshot_path,
                        absl::string_view chunk_file) {
  return AtomicallyWriteStringToFile(
      tsl::io::JoinPath(CommittedChunksDirectory(snapshot_path), chunk_file),
      "", tsl::Env::Default());
}

absl::Status SetDone(absl::string_view snapshot_path) {
  return AtomicallyWriteStringToFile(SnapshotDoneFilePath(snapshot_path), "",
                                     tsl::Env::Default());
}

absl::Status SetStatus(absl::string_view snapshot_path,
                       const absl::Status& status) {
  return AtomicallyWriteTextProto(SnapshotErrorFilePath(snapshot_path),
                                  tsl::StatusToProto(status),
                                  tsl::Env::Default());
}

absl::StatusOr<std::string> GetChunk(
    SnapshotChunkProvider& snapshot_chunk_provider) {
  Tensor split;
  bool end_of_splits = false;
  TF_RETURN_IF_ERROR(snapshot_chunk_provider.GetNext(&split, &end_of_splits));
  if (end_of_splits) {
    return absl::OutOfRangeError("No more available chunks.");
  }
  return split.unaligned_flat<tsl::tstring>().data()[0];
}

absl::StatusOr<std::vector<std::string>> GetAllChunks(
    SnapshotChunkProvider& snapshot_chunk_provider) {
  std::vector<std::string> chunks;
  while (true) {
    Tensor split;
    bool end_of_splits = false;
    TF_RETURN_IF_ERROR(snapshot_chunk_provider.GetNext(&split, &end_of_splits));
    if (end_of_splits) {
      return chunks;
    }
    chunks.push_back(split.unaligned_flat<tsl::tstring>().data()[0]);
  }
  return chunks;
}

std::vector<std::string> JoinPaths(absl::string_view snapshot_path,
                                   const std::vector<std::string> chunks) {
  std::vector<std::string> joined_chunks;
  for (absl::string_view chunk : chunks) {
    joined_chunks.push_back(
        tsl::io::JoinPath(CommittedChunksDirectory(snapshot_path), chunk));
  }
  return joined_chunks;
}

std::string full_name(const std::string& name) {
  return FullName("test", name);
}

absl::Status SaveAndRestore(SplitProvider& split_provider) {
  VariantTensorDataWriter writer;
  TF_RETURN_IF_ERROR(split_provider.Save(full_name, &writer));
  std::vector<const VariantTensorData*> variants;
  writer.GetData(&variants);
  VariantTensorDataReader reader(variants);
  TF_RETURN_IF_ERROR(split_provider.Restore(full_name, &reader));
  return absl::OkStatus();
}

TEST(SnapshotChunkProviderTest, EmptySnapshot) {
  TF_ASSERT_OK_AND_ASSIGN(std::string snapshot_path, CreateSnapshotDirectory());
  TF_ASSERT_OK(SetDone(snapshot_path));

  SnapshotChunkProvider snapshot_chunk_provider(snapshot_path,
                                                tsl::Env::Default());
  EXPECT_THAT(GetAllChunks(snapshot_chunk_provider), IsOkAndHolds(IsEmpty()));
  EXPECT_THAT(GetAllChunks(snapshot_chunk_provider), IsOkAndHolds(IsEmpty()));
}

TEST(SnapshotChunkProviderTest, SingleReader) {
  TF_ASSERT_OK_AND_ASSIGN(std::string snapshot_path, CreateSnapshotDirectory());
  std::vector<std::string> chunks = {"chunk_4_4_4", "chunk_3_3_3",
                                     "chunk_2_2_2", "chunk_1_1_1",
                                     "chunk_0_0_0"};
  for (absl::string_view chunk : chunks) {
    TF_ASSERT_OK(WriteChunk(snapshot_path, chunk));
  }
  TF_ASSERT_OK(SetDone(snapshot_path));

  SnapshotChunkProvider snapshot_chunk_provider(snapshot_path,
                                                tsl::Env::Default());
  // Chunks are ordered by chunk indices.
  absl::c_reverse(chunks);
  EXPECT_THAT(GetAllChunks(snapshot_chunk_provider),
              IsOkAndHolds(ElementsAreArray(JoinPaths(snapshot_path, chunks))));
}

TEST(SnapshotChunkProviderTest, Cardinality) {
  TF_ASSERT_OK_AND_ASSIGN(std::string snapshot_path, CreateSnapshotDirectory());
  TF_ASSERT_OK(WriteChunk(snapshot_path, "chunk_0_0_0"));
  SnapshotChunkProvider snapshot_chunk_provider(snapshot_path,
                                                tsl::Env::Default());
  // Cardinality is unknown when the snapshot is unfinished.
  EXPECT_EQ(snapshot_chunk_provider.Cardinality(), kUnknownCardinality);

  std::vector<std::string> chunks = {"chunk_1_1_1", "chunk_2_2_2",
                                     "chunk_3_3_3", "chunk_4_4_4"};
  for (absl::string_view chunk : chunks) {
    TF_ASSERT_OK(WriteChunk(snapshot_path, chunk));
  }
  // Cardinality is unknown when the snapshot is unfinished.
  EXPECT_EQ(snapshot_chunk_provider.Cardinality(), kUnknownCardinality);

  // Cardinality is 5 when the snapshot is finished.
  TF_ASSERT_OK(SetDone(snapshot_path));
  EXPECT_EQ(snapshot_chunk_provider.Cardinality(), 5);
}

TEST(SnapshotChunkProviderTest, WaitForSnapshot) {
  std::string snapshot_path;
  ASSERT_TRUE(tsl::Env::Default()->LocalTempFilename(&snapshot_path));

  absl::Mutex mu;
  std::vector<std::string> result;  // Guarded by `mu`.
  std::unique_ptr<tsl::Thread> reader_thread =
      absl::WrapUnique(tsl::Env::Default()->StartThread(
          /*thread_options=*/{}, /*name=*/"Reader",
          [&snapshot_path, &mu, &result]() {
            SnapshotChunkProvider snapshot_chunk_provider(snapshot_path,
                                                          tsl::Env::Default());
            TF_ASSERT_OK_AND_ASSIGN(std::vector<std::string> chunks,
                                    GetAllChunks(snapshot_chunk_provider));
            absl::MutexLock l(&mu);
            result = std::move(chunks);
          }));

  {  // The reader should wait when there are no chunks.
    absl::MutexLock l(&mu);
    EXPECT_TRUE(result.empty());
  }

  TF_ASSERT_OK(tsl::Env::Default()->RecursivelyCreateDir(
      CommittedChunksDirectory(snapshot_path)));
  TF_ASSERT_OK(WriteChunk(snapshot_path, "chunk_0_0_0"));
  TF_ASSERT_OK(SetDone(snapshot_path));

  // The reader should be able to get chunks now.
  reader_thread.reset();
  absl::MutexLock l(&mu);
  EXPECT_THAT(result,
              ElementsAreArray(JoinPaths(snapshot_path, {"chunk_0_0_0"})));
}

TEST(SnapshotChunkProviderTest, ConcurrentReadWrite) {
  TF_ASSERT_OK_AND_ASSIGN(std::string snapshot_path, CreateSnapshotDirectory());

  const int num_readers = 10;
  absl::Mutex mu;
  SnapshotChunkProvider snapshot_chunk_provider(snapshot_path,
                                                tsl::Env::Default());
  std::vector<std::string> result;  // Guarded by `mu`.
  std::vector<std::unique_ptr<tsl::Thread>> reader_threads;
  for (int i = 0; i < num_readers; ++i) {
    reader_threads.push_back(absl::WrapUnique(tsl::Env::Default()->StartThread(
        /*thread_options=*/{}, /*name=*/absl::StrCat("Reader_", i),
        [&snapshot_chunk_provider, &mu, &result]() {
          while (true) {
            tsl::Env::Default()->SleepForMicroseconds(25);
            Tensor split;
            bool end_of_splits = false;
            TF_ASSERT_OK(
                snapshot_chunk_provider.GetNext(&split, &end_of_splits));
            if (end_of_splits) {
              break;
            }
            absl::MutexLock l(&mu);
            result.push_back(split.unaligned_flat<tsl::tstring>().data()[0]);
          }
        })));
  }

  int num_streams = 10, num_chunks_per_stream = 50;
  std::vector<std::unique_ptr<tsl::Thread>> stream_threads;
  for (int i = 0; i < num_streams; ++i) {
    stream_threads.push_back(absl::WrapUnique(tsl::Env::Default()->StartThread(
        /*thread_options=*/{}, /*name=*/absl::StrCat("Writer_", i),
        [&snapshot_path, num_chunks_per_stream, i]() {
          for (int j = 0; j < num_chunks_per_stream; ++j) {
            std::string filename = absl::StrCat("chunk_", i, "_", j, "_1");
            TF_ASSERT_OK(WriteChunk(snapshot_path, filename));
            tsl::Env::Default()->SleepForMicroseconds(35);
          }
        })));
  }

  stream_threads.clear();
  TF_ASSERT_OK(SetDone(snapshot_path));

  reader_threads.clear();
  std::vector<std::string> expected;
  for (int i = 0; i < num_streams; ++i) {
    for (int j = 0; j < num_chunks_per_stream; ++j) {
      expected.push_back(absl::StrCat("chunk_", i, "_", j, "_1"));
    }
  }
  EXPECT_THAT(result,
              UnorderedElementsAreArray(JoinPaths(snapshot_path, expected)));
}

TEST(SnapshotChunkProviderTest, SaveRestore) {
  TF_ASSERT_OK_AND_ASSIGN(std::string snapshot_path, CreateSnapshotDirectory());
  std::vector<std::string> chunks = {"chunk_4_4_4", "chunk_3_3_3",
                                     "chunk_2_2_2", "chunk_1_1_1",
                                     "chunk_0_0_0"};
  for (absl::string_view chunk : chunks) {
    TF_ASSERT_OK(WriteChunk(snapshot_path, chunk));
  }
  TF_ASSERT_OK(SetDone(snapshot_path));

  SnapshotChunkProvider snapshot_chunk_provider(snapshot_path,
                                                tsl::Env::Default());
  EXPECT_THAT(GetChunk(snapshot_chunk_provider),
              IsOkAndHolds(tsl::io::JoinPath(
                  CommittedChunksDirectory(snapshot_path), "chunk_0_0_0")));
  TF_ASSERT_OK(SaveAndRestore(snapshot_chunk_provider));

  EXPECT_THAT(GetAllChunks(snapshot_chunk_provider),
              IsOkAndHolds(ElementsAreArray(
                  JoinPaths(snapshot_path, {"chunk_1_1_1", "chunk_2_2_2",
                                            "chunk_3_3_3", "chunk_4_4_4"}))));
}

TEST(SnapshotChunkProviderTest, SnapshotError) {
  TF_ASSERT_OK_AND_ASSIGN(std::string snapshot_path, CreateSnapshotDirectory());
  std::unique_ptr<tsl::Thread> reader_thread =
      absl::WrapUnique(tsl::Env::Default()->StartThread(
          /*thread_options=*/{}, /*name=*/"Reader", [&snapshot_path]() {
            SnapshotChunkProvider snapshot_chunk_provider(snapshot_path,
                                                          tsl::Env::Default());
            EXPECT_THAT(
                GetAllChunks(snapshot_chunk_provider),
                StatusIs(absl::StatusCode::kFailedPrecondition, "Test error."));
          }));

  TF_ASSERT_OK(WriteChunk(snapshot_path, "chunk_0_0_0"));
  TF_ASSERT_OK(WriteChunk(snapshot_path, "chunk_1_0_0"));
  TF_ASSERT_OK(WriteChunk(snapshot_path, "chunk_2_0_0"));
  TF_ASSERT_OK(
      SetStatus(snapshot_path, absl::FailedPreconditionError("Test error.")));
  reader_thread.reset();
}

TEST(SnapshotChunkProviderTest, Cancel) {
  TF_ASSERT_OK_AND_ASSIGN(std::string snapshot_path, CreateSnapshotDirectory());
  SnapshotChunkProvider snapshot_chunk_provider(snapshot_path,
                                                tsl::Env::Default());

  std::unique_ptr<tsl::Thread> reader_thread =
      absl::WrapUnique(tsl::Env::Default()->StartThread(
          /*thread_options=*/{}, /*name=*/"Reader",
          [&snapshot_chunk_provider]() {
            EXPECT_THAT(
                GetAllChunks(snapshot_chunk_provider),
                StatusIs(absl::StatusCode::kCancelled,
                         HasSubstr("Cancelled loading tf.data snapshot at")));
          }));

  TF_ASSERT_OK(WriteChunk(snapshot_path, "chunk_0_0_0"));
  TF_ASSERT_OK(WriteChunk(snapshot_path, "chunk_1_0_0"));
  TF_ASSERT_OK(WriteChunk(snapshot_path, "chunk_2_0_0"));
  snapshot_chunk_provider.Cancel();
  reader_thread.reset();
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
