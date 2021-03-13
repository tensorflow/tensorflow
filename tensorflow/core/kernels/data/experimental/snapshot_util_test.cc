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

#include "tensorflow/core/kernels/data/experimental/snapshot_util.h"

#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/compression.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace data {
namespace snapshot_util {
namespace {

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
