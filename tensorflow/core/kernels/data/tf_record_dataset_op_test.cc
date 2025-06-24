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
#include "tensorflow/core/kernels/data/tf_record_dataset_op.h"

#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status.h"
#include "tensorflow/core/data/dataset_test_base.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "tf_record_dataset";
constexpr char kOpVersion = 2;

// Returns the file offset for the record at the given index.
int64_t GetOffset(const std::string& filename, int64_t index) {
  Env* env_ = Env::Default();
  std::unique_ptr<RandomAccessFile> file_;
  std::unique_ptr<io::SequentialRecordReader> reader;
  absl::Status s1 = env_->NewRandomAccessFile(filename, &file_);
  TF_CHECK_OK(s1) << s1;
  reader = std::make_unique<io::SequentialRecordReader>(file_.get());
  for (int i = 0; i < index; ++i) {
    tstring record;
    absl::Status s2 = reader->ReadRecord(&record);
    TF_CHECK_OK(s2) << s2;
  }
  return reader->TellOffset();
}

class TFRecordDatasetParams : public DatasetParams {
 public:
  TFRecordDatasetParams(std::vector<tstring> filenames,
                        CompressionType compression_type, int64_t buffer_size,
                        std::vector<int64_t> byte_offsets, string node_name)
      : DatasetParams({DT_STRING}, {PartialTensorShape({})},
                      std::move(node_name)),
        filenames_(std::move(filenames)),
        compression_type_(compression_type),
        buffer_size_(buffer_size),
        byte_offsets_(std::move(byte_offsets)) {
    op_version_ = 2;
  }

  std::vector<Tensor> GetInputTensors() const override {
    int num_files = filenames_.size();
    int num_byte_offsets = byte_offsets_.size();
    return {
        CreateTensor<tstring>(TensorShape({num_files}), filenames_),
        CreateTensor<tstring>(TensorShape({}), {ToString(compression_type_)}),
        CreateTensor<int64_t>(TensorShape({}), {buffer_size_}),
        CreateTensor<int64_t>(TensorShape({num_byte_offsets}), byte_offsets_)};
  }

  absl::Status GetInputNames(std::vector<string>* input_names) const override {
    input_names->clear();
    *input_names = {
        TFRecordDatasetOp::kFileNames,
        TFRecordDatasetOp::kCompressionType,
        TFRecordDatasetOp::kBufferSize,
        TFRecordDatasetOp::kByteOffsets,
    };
    return absl::OkStatus();
  }

  absl::Status GetAttributes(AttributeVector* attr_vector) const override {
    attr_vector->clear();
    attr_vector->emplace_back("metadata", "");
    return absl::OkStatus();
  }

  string dataset_type() const override {
    return TFRecordDatasetOp::kDatasetType;
  }

 private:
  std::vector<tstring> filenames_;
  CompressionType compression_type_;
  int64_t buffer_size_;
  std::vector<int64_t> byte_offsets_;
};

class TFRecordDatasetOpTest : public DatasetOpsTestBase {};

absl::Status CreateTestFiles(const std::vector<tstring>& filenames,
                             const std::vector<std::vector<string>>& contents,
                             CompressionType compression_type) {
  if (filenames.size() != contents.size()) {
    return tensorflow::errors::InvalidArgument(
        "The number of files does not match with the contents");
  }
  for (int i = 0; i < filenames.size(); ++i) {
    CompressionParams params;
    params.output_buffer_size = 10;
    params.compression_type = compression_type;
    std::vector<absl::string_view> records(contents[i].begin(),
                                           contents[i].end());
    TF_RETURN_IF_ERROR(WriteDataToTFRecordFile(filenames[i], records, params));
  }
  return absl::OkStatus();
}

// Test case 1: multiple text files with ZLIB compression.
TFRecordDatasetParams TFRecordDatasetParams1() {
  std::vector<tstring> filenames = {
      absl::StrCat(testing::TmpDir(), "/tf_record_ZLIB_1"),
      absl::StrCat(testing::TmpDir(), "/tf_record_ZLIB_2")};
  std::vector<std::vector<string>> contents = {{"1", "22", "333"},
                                               {"a", "bb", "ccc"}};
  CompressionType compression_type = CompressionType::ZLIB;
  if (!CreateTestFiles(filenames, contents, compression_type).ok()) {
    LOG(WARNING) << "Failed to create the test files: "
                 << absl::StrJoin(filenames, ", ");
  }
  return TFRecordDatasetParams(filenames,
                               /*compression_type=*/compression_type,
                               /*buffer_size=*/10,
                               /*byte_offsets=*/{},
                               /*node_name=*/kNodeName);
}

// Test case 2: multiple text files with GZIP compression.
TFRecordDatasetParams TFRecordDatasetParams2() {
  std::vector<tstring> filenames = {
      absl::StrCat(testing::TmpDir(), "/tf_record_GZIP_1"),
      absl::StrCat(testing::TmpDir(), "/tf_record_GZIP_2")};
  std::vector<std::vector<string>> contents = {{"1", "22", "333"},
                                               {"a", "bb", "ccc"}};
  CompressionType compression_type = CompressionType::GZIP;
  if (!CreateTestFiles(filenames, contents, compression_type).ok()) {
    LOG(WARNING) << "Failed to create the test files: "
                 << absl::StrJoin(filenames, ", ");
  }
  return TFRecordDatasetParams(filenames,
                               /*compression_type=*/compression_type,
                               /*buffer_size=*/10,
                               /*byte_offsets=*/{},
                               /*node_name=*/kNodeName);
}

// Test case 3: multiple text files without compression.
TFRecordDatasetParams TFRecordDatasetParams3() {
  std::vector<tstring> filenames = {
      absl::StrCat(testing::TmpDir(), "/tf_record_UNCOMPRESSED_1"),
      absl::StrCat(testing::TmpDir(), "/tf_record_UNCOMPRESSED_2")};
  std::vector<std::vector<string>> contents = {{"1", "22", "333"},
                                               {"a", "bb", "ccc"}};
  CompressionType compression_type = CompressionType::UNCOMPRESSED;
  if (!CreateTestFiles(filenames, contents, compression_type).ok()) {
    LOG(WARNING) << "Failed to create the test files: "
                 << absl::StrJoin(filenames, ", ");
  }
  return TFRecordDatasetParams(filenames,
                               /*compression_type=*/compression_type,
                               /*buffer_size=*/10,
                               /*byte_offsets=*/{},
                               /*node_name=*/kNodeName);
}

// Test case 4: Read byte_offsets for records.
TFRecordDatasetParams TFRecordDatasetParams4() {
  std::vector<tstring> filenames = {
      absl::StrCat(testing::TmpDir(), "/tf_record_UNCOMPRESSED_1"),
      absl::StrCat(testing::TmpDir(), "/tf_record_UNCOMPRESSED_2"),
      absl::StrCat(testing::TmpDir(), "/tf_record_UNCOMPRESSED_3")};
  std::vector<std::vector<string>> contents = {
      {"1", "22", "333"}, {"a", "bb", "ccc"}, {"x", "yy", "zzz"}};
  CompressionType compression_type = CompressionType::UNCOMPRESSED;
  absl::Status status = CreateTestFiles(filenames, contents, compression_type);
  TF_CHECK_OK(status) << "Failed to create the test files: "
                      << absl::StrJoin(filenames, ", ") << ": " << status;
  std::vector<int64_t> byte_offsets = {};
  byte_offsets.push_back(GetOffset(filenames[0], 0));
  byte_offsets.push_back(GetOffset(filenames[1], 1));
  byte_offsets.push_back(GetOffset(filenames[1], 2));
  return TFRecordDatasetParams(filenames,
                               /*compression_type=*/compression_type,
                               /*buffer_size=*/10, byte_offsets,
                               /*node_name=*/kNodeName);
}

// Test case 5: Read invalid byte_offsets for records.
TFRecordDatasetParams InvalidByteOffsets() {
  std::vector<tstring> filenames = {
      absl::StrCat(testing::TmpDir(), "/tf_record_UNCOMPRESSED_1")};
  std::vector<std::vector<string>> contents = {{"1", "22", "333"}};
  CompressionType compression_type = CompressionType::UNCOMPRESSED;
  absl::Status status = CreateTestFiles(filenames, contents, compression_type);
  TF_CHECK_OK(status) << "Failed to create the test files: "
                      << absl::StrJoin(filenames, ", ") << ": " << status;
  return TFRecordDatasetParams(filenames,
                               /*compression_type=*/compression_type,
                               /*buffer_size=*/10, /*byte_offsets=*/{1},
                               /*node_name=*/kNodeName);
}

std::vector<GetNextTestCase<TFRecordDatasetParams>> GetNextTestCases() {
  return {
      {/*dataset_params=*/TFRecordDatasetParams1(),
       /*expected_outputs=*/
       CreateTensors<tstring>(
           TensorShape({}), {{"1"}, {"22"}, {"333"}, {"a"}, {"bb"}, {"ccc"}})},
      {/*dataset_params=*/TFRecordDatasetParams2(),
       CreateTensors<tstring>(
           TensorShape({}), {{"1"}, {"22"}, {"333"}, {"a"}, {"bb"}, {"ccc"}})},
      {/*dataset_params=*/TFRecordDatasetParams3(),
       CreateTensors<tstring>(
           TensorShape({}), {{"1"}, {"22"}, {"333"}, {"a"}, {"bb"}, {"ccc"}})},
      {/*dataset_params=*/TFRecordDatasetParams4(),
       CreateTensors<tstring>(
           TensorShape({}),
           {{"1"}, {"22"}, {"333"}, {"bb"}, {"ccc"}, {"zzz"}})}};
}

ITERATOR_GET_NEXT_TEST_P(TFRecordDatasetOpTest, TFRecordDatasetParams,
                         GetNextTestCases())

std::vector<SkipTestCase<TFRecordDatasetParams>> SkipTestCases() {
  return {{/*dataset_params=*/TFRecordDatasetParams1(),
           /*num_to_skip*/ 2, /*expected_num_skipped*/ 2, /*get_next*/ true,
           /*expected_outputs=*/
           CreateTensors<tstring>(TensorShape({}), {{"333"}})},
          {/*dataset_params=*/TFRecordDatasetParams1(),
           /*num_to_skip*/ 4, /*expected_num_skipped*/ 4, /*get_next*/ true,
           /*expected_outputs=*/
           CreateTensors<tstring>(TensorShape({}), {{"bb"}})},
          {/*dataset_params=*/TFRecordDatasetParams1(),
           /*num_to_skip*/ 7, /*expected_num_skipped*/ 6},

          {/*dataset_params=*/TFRecordDatasetParams2(),
           /*num_to_skip*/ 2, /*expected_num_skipped*/ 2, /*get_next*/ true,
           /*expected_outputs=*/
           CreateTensors<tstring>(TensorShape({}), {{"333"}})},
          {/*dataset_params=*/TFRecordDatasetParams2(),
           /*num_to_skip*/ 4, /*expected_num_skipped*/ 4, /*get_next*/ true,
           /*expected_outputs=*/
           CreateTensors<tstring>(TensorShape({}), {{"bb"}})},
          {/*dataset_params=*/TFRecordDatasetParams2(),
           /*num_to_skip*/ 7, /*expected_num_skipped*/ 6},

          {/*dataset_params=*/TFRecordDatasetParams3(),
           /*num_to_skip*/ 2, /*expected_num_skipped*/ 2, /*get_next*/ true,
           /*expected_outputs=*/
           CreateTensors<tstring>(TensorShape({}), {{"333"}})},
          {/*dataset_params=*/TFRecordDatasetParams3(),
           /*num_to_skip*/ 4, /*expected_num_skipped*/ 4, /*get_next*/ true,
           /*expected_outputs=*/
           CreateTensors<tstring>(TensorShape({}), {{"bb"}})},
          {/*dataset_params=*/TFRecordDatasetParams3(),
           /*num_to_skip*/ 7, /*expected_num_skipped*/ 6}};
}

ITERATOR_SKIP_TEST_P(TFRecordDatasetOpTest, TFRecordDatasetParams,
                     SkipTestCases())

TEST_F(TFRecordDatasetOpTest, DatasetNodeName) {
  auto dataset_params = TFRecordDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(dataset_params.node_name()));
}

TEST_F(TFRecordDatasetOpTest, DatasetTypeString) {
  auto dataset_params = TFRecordDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  name_utils::OpNameParams params;
  params.op_version = kOpVersion;
  TF_ASSERT_OK(CheckDatasetTypeString(
      name_utils::OpName(TFRecordDatasetOp::kDatasetType, params)));
}

TEST_F(TFRecordDatasetOpTest, DatasetOutputDtypes) {
  auto dataset_params = TFRecordDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputDtypes({DT_STRING}));
}

TEST_F(TFRecordDatasetOpTest, DatasetOutputShapes) {
  auto dataset_params = TFRecordDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputShapes({PartialTensorShape({})}));
}

TEST_F(TFRecordDatasetOpTest, Cardinality) {
  auto dataset_params = TFRecordDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetCardinality(kUnknownCardinality));
}

TEST_F(TFRecordDatasetOpTest, IteratorOutputDtypes) {
  auto dataset_params = TFRecordDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputDtypes({DT_STRING}));
}

TEST_F(TFRecordDatasetOpTest, IteratorOutputShapes) {
  auto dataset_params = TFRecordDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputShapes({PartialTensorShape({})}));
}

TEST_F(TFRecordDatasetOpTest, IteratorPrefix) {
  auto dataset_params = TFRecordDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  name_utils::IteratorPrefixParams iterator_prefix_params;
  iterator_prefix_params.op_version = kOpVersion;
  TF_ASSERT_OK(CheckIteratorPrefix(name_utils::IteratorPrefix(
      TFRecordDatasetOp::kDatasetType, dataset_params.iterator_prefix(),
      iterator_prefix_params)));
}

TEST_F(TFRecordDatasetOpTest, InvalidByteOffsetsToSeek) {
  auto dataset_params = InvalidByteOffsets();
  TF_ASSERT_OK(Initialize(dataset_params));
  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  EXPECT_EQ(
      iterator_->GetNext(iterator_ctx_.get(), &out_tensors, &end_of_sequence)
          .code(),
      absl::StatusCode::kDataLoss);
}

std::vector<IteratorSaveAndRestoreTestCase<TFRecordDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {
      {/*dataset_params=*/TFRecordDatasetParams1(),
       /*breakpoints=*/{0, 2, 7},
       /*expected_outputs=*/
       CreateTensors<tstring>(
           TensorShape({}), {{"1"}, {"22"}, {"333"}, {"a"}, {"bb"}, {"ccc"}})},
      {/*dataset_params=*/TFRecordDatasetParams2(),
       /*breakpoints=*/{0, 2, 7},
       CreateTensors<tstring>(
           TensorShape({}), {{"1"}, {"22"}, {"333"}, {"a"}, {"bb"}, {"ccc"}})},
      {/*dataset_params=*/TFRecordDatasetParams3(),
       /*breakpoints=*/{0, 2, 7},
       CreateTensors<tstring>(
           TensorShape({}), {{"1"}, {"22"}, {"333"}, {"a"}, {"bb"}, {"ccc"}})}};
}

ITERATOR_SAVE_AND_RESTORE_TEST_P(TFRecordDatasetOpTest, TFRecordDatasetParams,
                                 IteratorSaveAndRestoreTestCases())

}  // namespace
}  // namespace data
}  // namespace tensorflow
