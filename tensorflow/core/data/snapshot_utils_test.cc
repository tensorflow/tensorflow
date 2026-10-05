/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/data/snapshot_utils.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "tensorflow/core/data/service/test_util.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/compression.h"
#include "tensorflow/core/platform/coding.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/protobuf/snapshot.pb.h"

namespace tensorflow {
namespace data {
namespace snapshot_util {
namespace {

using ::tensorflow::data::testing::EqualsProto;
using ::tensorflow::data::testing::LocalTempFilename;

void GenerateTensorVector(tensorflow::DataTypeVector& dtypes,
                          std::vector<Tensor>& tensors) {
  std::string tensor_data(1024, 'a');
  for (int i = 0; i < 10; ++i) {
    Tensor t(tensor_data.data());
    dtypes.push_back(t.dtype());
    tensors.push_back(t);
  }
}

void SnapshotRoundTrip(std::string compression_type, int version) {
  // Generate ground-truth tensors for writing and reading.
  std::vector<Tensor> tensors;
  tensorflow::DataTypeVector dtypes;
  GenerateTensorVector(dtypes, tensors);

  std::string filename;
  EXPECT_TRUE(Env::Default()->LocalTempFilename(&filename));

  std::unique_ptr<Writer> writer;
  TF_ASSERT_OK(Writer::Create(tensorflow::Env::Default(), filename,
                              compression_type, version, dtypes, &writer));

  for (int i = 0; i < 100; ++i) {
    TF_ASSERT_OK(writer->WriteTensors(tensors));
  }
  TF_ASSERT_OK(writer->Close());

  std::unique_ptr<Reader> reader;
  TF_ASSERT_OK(Reader::Create(Env::Default(), filename, compression_type,
                              version, dtypes, &reader));

  for (int i = 0; i < 100; ++i) {
    std::vector<Tensor> read_tensors;
    TF_ASSERT_OK(reader->ReadTensors(&read_tensors));
    EXPECT_EQ(tensors.size(), read_tensors.size());
    for (int j = 0; j < read_tensors.size(); ++j) {
      TensorProto proto;
      TensorProto read_proto;

      tensors[j].AsProtoTensorContent(&proto);
      read_tensors[j].AsProtoTensorContent(&read_proto);

      std::string proto_serialized, read_proto_serialized;
      proto.AppendToString(&proto_serialized);
      read_proto.AppendToString(&read_proto_serialized);
      EXPECT_EQ(proto_serialized, read_proto_serialized);
    }
  }

  TF_ASSERT_OK(Env::Default()->DeleteFile(filename));
}

TEST(SnapshotUtilTest, CombinationRoundTripTest) {
  SnapshotRoundTrip(io::compression::kNone, 1);
  SnapshotRoundTrip(io::compression::kGzip, 1);
  SnapshotRoundTrip(io::compression::kSnappy, 1);

  SnapshotRoundTrip(io::compression::kNone, 2);
  SnapshotRoundTrip(io::compression::kGzip, 2);
  SnapshotRoundTrip(io::compression::kSnappy, 2);
}

TEST(SnapshotUtilTest, MetadataFileRoundTrip) {
  experimental::DistributedSnapshotMetadata metadata_in;
  metadata_in.set_compression(io::compression::kGzip);
  std::string dir = LocalTempFilename();
  TF_ASSERT_OK(WriteMetadataFile(Env::Default(), dir, &metadata_in));

  experimental::DistributedSnapshotMetadata metadata_out;
  bool file_exists;
  TF_ASSERT_OK(
      ReadMetadataFile(Env::Default(), dir, &metadata_out, &file_exists));
  EXPECT_THAT(metadata_in, EqualsProto(metadata_out));
}

TEST(SnapshotUtilTest, MetadataFileDoesntExist) {
  experimental::DistributedSnapshotMetadata metadata;
  bool file_exists;
  TF_ASSERT_OK(ReadMetadataFile(Env::Default(), LocalTempFilename(), &metadata,
                                &file_exists));
  EXPECT_FALSE(file_exists);
}

TEST(SnapshotUtilTest, SnappyTensorSizeMismatch) {
  // Generate ground-truth tensors for writing and reading.
  std::vector<Tensor> tensors;
  tensorflow::DataTypeVector dtypes;
  GenerateTensorVector(dtypes, tensors);

  std::string filename;
  EXPECT_TRUE(Env::Default()->LocalTempFilename(&filename));

  std::unique_ptr<Writer> writer;
  TF_ASSERT_OK(Writer::Create(tensorflow::Env::Default(), filename,
                              io::compression::kSnappy, 1, dtypes, &writer));

  TF_ASSERT_OK(writer->WriteTensors(tensors));
  TF_ASSERT_OK(writer->Close());

  std::unique_ptr<Reader> reader;
  tensorflow::DataTypeVector wrong_dtypes = dtypes;
  wrong_dtypes.push_back(DT_INT64);
  TF_ASSERT_OK(Reader::Create(Env::Default(), filename,
                              io::compression::kSnappy, 1, wrong_dtypes,
                              &reader));

  std::vector<Tensor> read_tensors;
  EXPECT_TRUE(absl::IsDataLoss(reader->ReadTensors(&read_tensors)));

  TF_ASSERT_OK(Env::Default()->DeleteFile(filename));
}

// Writes a snappy snapshot of two int64 tensors, then rewrites the dimensions
// of the first tensor's metadata to `patched_dim_sizes`.
//
// int64 is memcpy-able, so the tensors take the simple-tensor path in
// CustomReader::SnappyUncompress, which is where the shape is built. Records
// are an 8-byte little-endian length followed by the payload, and the metadata
// is the first record written for a snappy snapshot, so it can be rewritten
// without disturbing the compressed block that follows. The tensor count is
// left unchanged so the read reaches the shape instead of failing the earlier
// count check.
void WriteSnapshotWithPatchedDims(const std::vector<int64_t>& patched_dim_sizes,
                                  tensorflow::DataTypeVector* dtypes,
                                  std::string* filename) {
  std::vector<Tensor> tensors;
  for (int i = 0; i < 2; ++i) {
    Tensor t(DT_INT64, TensorShape({4}));
    t.flat<int64_t>().setZero();
    dtypes->push_back(DT_INT64);
    tensors.push_back(t);
  }

  ASSERT_TRUE(Env::Default()->LocalTempFilename(filename));

  std::unique_ptr<Writer> writer;
  TF_ASSERT_OK(Writer::Create(Env::Default(), *filename,
                              io::compression::kSnappy, /*version=*/1, *dtypes,
                              &writer));
  TF_ASSERT_OK(writer->WriteTensors(tensors));
  TF_ASSERT_OK(writer->Close());

  constexpr size_t kHeaderSize = sizeof(uint64_t);
  std::string contents;
  TF_ASSERT_OK(ReadFileToString(Env::Default(), *filename, &contents));
  ASSERT_GT(contents.size(), kHeaderSize);
  const uint64_t metadata_size = core::DecodeFixed64(contents.data());
  ASSERT_GE(contents.size(), kHeaderSize + metadata_size);

  experimental::SnapshotTensorMetadata metadata;
  ASSERT_TRUE(
      metadata.ParseFromString(contents.substr(kHeaderSize, metadata_size)));
  ASSERT_GT(metadata.tensor_metadata_size(), 0);
  auto* shape = metadata.mutable_tensor_metadata(0)->mutable_tensor_shape();
  shape->clear_dim();
  for (int64_t dim_size : patched_dim_sizes) {
    shape->add_dim()->set_size(dim_size);
  }

  const std::string patched = metadata.SerializeAsString();
  char header[kHeaderSize];
  core::EncodeFixed64(header, patched.size());
  TF_ASSERT_OK(WriteStringToFile(
      Env::Default(), *filename,
      std::string(header, kHeaderSize) + patched +
          contents.substr(kHeaderSize + metadata_size)));
}

TEST(SnapshotUtilTest, SnappyInvalidTensorShapeIsAnError) {
  // A corrupt snapshot whose metadata carries a negative dimension must be
  // reported as an error. Building a TensorShape directly from the proto
  // CHECK-fails instead, which aborts the process.
  std::string filename;
  tensorflow::DataTypeVector dtypes;
  ASSERT_NO_FATAL_FAILURE(WriteSnapshotWithPatchedDims(
      /*patched_dim_sizes=*/{-1}, &dtypes, &filename));

  std::unique_ptr<Reader> reader;
  TF_ASSERT_OK(Reader::Create(Env::Default(), filename,
                              io::compression::kSnappy, /*version=*/1, dtypes,
                              &reader));
  std::vector<Tensor> read_tensors;
  EXPECT_TRUE(absl::IsInvalidArgument(reader->ReadTensors(&read_tensors)));

  TF_ASSERT_OK(Env::Default()->DeleteFile(filename));
}

TEST(SnapshotUtilTest, SnappyZeroTensorDimensionIsAnError) {
  // Zero is a valid dimension, so the shape is built successfully, but the
  // tensor then holds no buffer. The read must report the resulting mismatch
  // between the metadata and the compressed payload rather than dereference
  // that null buffer.
  std::string filename;
  tensorflow::DataTypeVector dtypes;
  ASSERT_NO_FATAL_FAILURE(WriteSnapshotWithPatchedDims(
      /*patched_dim_sizes=*/{0}, &dtypes, &filename));

  std::unique_ptr<Reader> reader;
  TF_ASSERT_OK(Reader::Create(Env::Default(), filename,
                              io::compression::kSnappy, /*version=*/1, dtypes,
                              &reader));
  std::vector<Tensor> read_tensors;
  EXPECT_TRUE(absl::IsInternal(reader->ReadTensors(&read_tensors)));

  TF_ASSERT_OK(Env::Default()->DeleteFile(filename));
}

TEST(SnapshotUtilTest, SnappyOverflowingTensorShapeIsAnError) {
  // The other branch of BuildTensorShape: each dimension is non-negative, but
  // their product does not fit in int64, which the TensorShape proto
  // constructor would CHECK-fail on.
  std::string filename;
  tensorflow::DataTypeVector dtypes;
  ASSERT_NO_FATAL_FAILURE(WriteSnapshotWithPatchedDims(
      /*patched_dim_sizes=*/{1LL << 32, 1LL << 32}, &dtypes, &filename));

  std::unique_ptr<Reader> reader;
  TF_ASSERT_OK(Reader::Create(Env::Default(), filename,
                              io::compression::kSnappy, /*version=*/1, dtypes,
                              &reader));
  std::vector<Tensor> read_tensors;
  EXPECT_TRUE(absl::IsInvalidArgument(reader->ReadTensors(&read_tensors)));

  TF_ASSERT_OK(Env::Default()->DeleteFile(filename));
}

TEST(SnapshotUtilTest, SnappyZeroTensorDimensionRoundTrip) {
  // An empty tensor is legitimate rather than corrupt, so it must survive a
  // write and a read. Both sides hold a null buffer for it.
  std::string filename;
  ASSERT_TRUE(Env::Default()->LocalTempFilename(&filename));

  tensorflow::DataTypeVector dtypes = {DT_INT64};
  std::vector<Tensor> tensors;
  tensors.emplace_back(DT_INT64, TensorShape({0}));

  std::unique_ptr<Writer> writer;
  TF_ASSERT_OK(Writer::Create(Env::Default(), filename,
                              io::compression::kSnappy, /*version=*/1, dtypes,
                              &writer));
  TF_ASSERT_OK(writer->WriteTensors(tensors));
  TF_ASSERT_OK(writer->Close());

  std::unique_ptr<Reader> reader;
  TF_ASSERT_OK(Reader::Create(Env::Default(), filename,
                              io::compression::kSnappy, /*version=*/1, dtypes,
                              &reader));
  std::vector<Tensor> read_tensors;
  TF_ASSERT_OK(reader->ReadTensors(&read_tensors));
  ASSERT_EQ(read_tensors.size(), 1);
  EXPECT_EQ(read_tensors[0].NumElements(), 0);

  TF_ASSERT_OK(Env::Default()->DeleteFile(filename));
}

void SnapshotReaderBenchmarkLoop(::testing::benchmark::State& state,
                                 std::string compression_type, int version) {
  tensorflow::DataTypeVector dtypes;
  std::vector<Tensor> tensors;
  GenerateTensorVector(dtypes, tensors);

  std::string filename;
  EXPECT_TRUE(Env::Default()->LocalTempFilename(&filename));

  std::unique_ptr<Writer> writer;
  TF_ASSERT_OK(Writer::Create(tensorflow::Env::Default(), filename,
                              compression_type, version, dtypes, &writer));

  for (auto s : state) {
    writer->WriteTensors(tensors).IgnoreError();
  }
  TF_ASSERT_OK(writer->Close());

  std::unique_ptr<Reader> reader;
  TF_ASSERT_OK(Reader::Create(Env::Default(), filename, compression_type,
                              version, dtypes, &reader));

  for (auto s : state) {
    std::vector<Tensor> read_tensors;
    reader->ReadTensors(&read_tensors).IgnoreError();
  }

  TF_ASSERT_OK(Env::Default()->DeleteFile(filename));
}

void SnapshotCustomReaderNoneBenchmark(::testing::benchmark::State& state) {
  SnapshotReaderBenchmarkLoop(state, io::compression::kNone, 1);
}

void SnapshotCustomReaderGzipBenchmark(::testing::benchmark::State& state) {
  SnapshotReaderBenchmarkLoop(state, io::compression::kGzip, 1);
}

void SnapshotCustomReaderSnappyBenchmark(::testing::benchmark::State& state) {
  SnapshotReaderBenchmarkLoop(state, io::compression::kSnappy, 1);
}

void SnapshotTFRecordReaderNoneBenchmark(::testing::benchmark::State& state) {
  SnapshotReaderBenchmarkLoop(state, io::compression::kNone, 2);
}

void SnapshotTFRecordReaderGzipBenchmark(::testing::benchmark::State& state) {
  SnapshotReaderBenchmarkLoop(state, io::compression::kGzip, 2);
}

BENCHMARK(SnapshotCustomReaderNoneBenchmark);
BENCHMARK(SnapshotCustomReaderGzipBenchmark);
BENCHMARK(SnapshotCustomReaderSnappyBenchmark);
BENCHMARK(SnapshotTFRecordReaderNoneBenchmark);
BENCHMARK(SnapshotTFRecordReaderGzipBenchmark);

void SnapshotWriterBenchmarkLoop(::testing::benchmark::State& state,
                                 std::string compression_type, int version) {
  tensorflow::DataTypeVector dtypes;
  std::vector<Tensor> tensors;
  GenerateTensorVector(dtypes, tensors);

  std::string filename;
  EXPECT_TRUE(Env::Default()->LocalTempFilename(&filename));

  std::unique_ptr<Writer> writer;
  TF_ASSERT_OK(Writer::Create(tensorflow::Env::Default(), filename,
                              compression_type, version, dtypes, &writer));

  for (auto s : state) {
    writer->WriteTensors(tensors).IgnoreError();
  }
  writer->Close().IgnoreError();

  TF_ASSERT_OK(Env::Default()->DeleteFile(filename));
}

void SnapshotCustomWriterNoneBenchmark(::testing::benchmark::State& state) {
  SnapshotWriterBenchmarkLoop(state, io::compression::kNone, 1);
}

void SnapshotCustomWriterGzipBenchmark(::testing::benchmark::State& state) {
  SnapshotWriterBenchmarkLoop(state, io::compression::kGzip, 1);
}

void SnapshotCustomWriterSnappyBenchmark(::testing::benchmark::State& state) {
  SnapshotWriterBenchmarkLoop(state, io::compression::kSnappy, 1);
}

void SnapshotTFRecordWriterNoneBenchmark(::testing::benchmark::State& state) {
  SnapshotWriterBenchmarkLoop(state, io::compression::kNone, 2);
}

void SnapshotTFRecordWriterGzipBenchmark(::testing::benchmark::State& state) {
  SnapshotWriterBenchmarkLoop(state, io::compression::kGzip, 2);
}

void SnapshotTFRecordWriterSnappyBenchmark(::testing::benchmark::State& state) {
  SnapshotWriterBenchmarkLoop(state, io::compression::kSnappy, 2);
}

BENCHMARK(SnapshotCustomWriterNoneBenchmark);
BENCHMARK(SnapshotCustomWriterGzipBenchmark);
BENCHMARK(SnapshotCustomWriterSnappyBenchmark);
BENCHMARK(SnapshotTFRecordWriterNoneBenchmark);
BENCHMARK(SnapshotTFRecordWriterGzipBenchmark);
BENCHMARK(SnapshotTFRecordWriterSnappyBenchmark);

}  // namespace
}  // namespace snapshot_util
}  // namespace data
}  // namespace tensorflow
