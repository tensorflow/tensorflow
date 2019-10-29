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
#include "tensorflow/core/kernels/data/fixed_length_record_dataset_op.h"

#include "tensorflow/core/kernels/data/dataset_test_base.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "fixed_length_record_dataset";
constexpr int kOpVersion = 2;

class FixedLengthRecordDatasetParams : public DatasetParams {
 public:
  FixedLengthRecordDatasetParams(const std::vector<tstring>& filenames,
                                 int64 header_bytes, int64 record_bytes,
                                 int64 footer_bytes, int64 buffer_size,
                                 CompressionType compression_type,
                                 string node_name)
      : DatasetParams({DT_STRING}, {PartialTensorShape({})},
                      std::move(node_name)),
        filenames_(filenames),
        header_bytes_(header_bytes),
        record_bytes_(record_bytes),
        footer_bytes_(footer_bytes),
        buffer_size_(buffer_size),
        compression_type_(compression_type) {
    op_version_ = 2;
  }

  std::vector<Tensor> GetInputTensors() const override {
    int num_files = filenames_.size();
    return {
        CreateTensor<tstring>(TensorShape({num_files}), filenames_),
        CreateTensor<int64>(TensorShape({}), {header_bytes_}),
        CreateTensor<int64>(TensorShape({}), {record_bytes_}),
        CreateTensor<int64>(TensorShape({}), {footer_bytes_}),
        CreateTensor<int64>(TensorShape({}), {buffer_size_}),
        CreateTensor<tstring>(TensorShape({}), {ToString(compression_type_)})};
  }

  Status GetInputNames(std::vector<string>* input_names) const override {
    input_names->clear();
    *input_names = {FixedLengthRecordDatasetOp::kFileNames,
                    FixedLengthRecordDatasetOp::kHeaderBytes,
                    FixedLengthRecordDatasetOp::kRecordBytes,
                    FixedLengthRecordDatasetOp::kFooterBytes,
                    FixedLengthRecordDatasetOp::kBufferSize,
                    FixedLengthRecordDatasetOp::kCompressionType};
    return Status::OK();
  }

  Status GetAttributes(AttributeVector* attr_vector) const override {
    *attr_vector = {};
    return Status::OK();
  }

  string dataset_type() const override {
    return FixedLengthRecordDatasetOp::kDatasetType;
  }

 private:
  std::vector<tstring> filenames_;
  int64 header_bytes_;
  int64 record_bytes_;
  int64 footer_bytes_;
  int64 buffer_size_;
  CompressionType compression_type_;
};

class FixedLengthRecordDatasetOpTest : public DatasetOpsTestBaseV2 {};

Status CreateTestFiles(const std::vector<tstring>& filenames,
                       const std::vector<string>& contents,
                       CompressionType compression_type) {
  if (filenames.size() != contents.size()) {
    return tensorflow::errors::InvalidArgument(
        "The number of files does not match with the contents");
  }
  if (compression_type == CompressionType::UNCOMPRESSED) {
    for (int i = 0; i < filenames.size(); ++i) {
      TF_RETURN_IF_ERROR(WriteDataToFile(filenames[i], contents[i].data()));
    }
  } else {
    CompressionParams params;
    params.output_buffer_size = 10;
    params.compression_type = compression_type;
    for (int i = 0; i < filenames.size(); ++i) {
      TF_RETURN_IF_ERROR(
          WriteDataToFile(filenames[i], contents[i].data(), params));
    }
  }
  return Status::OK();
}

// Test case 1: multiple fixed-length record files with ZLIB compression.
FixedLengthRecordDatasetParams FixedLengthRecordDatasetParams1() {
  std::vector<tstring> filenames = {
      absl::StrCat(testing::TmpDir(), "/text_line_ZLIB_1"),
      absl::StrCat(testing::TmpDir(), "/text_line_ZLIB_2")};
  std::vector<string> contents = {
      absl::StrCat("HHHHH", "111", "222", "333", "FF"),
      absl::StrCat("HHHHH", "aaa", "bbb", "FF")};
  CompressionType compression_type = CompressionType::ZLIB;
  if (!CreateTestFiles(filenames, contents, compression_type).ok()) {
    VLOG(WARNING) << "Failed to create the test files: "
                  << absl::StrJoin(filenames, ", ");
  }

  return FixedLengthRecordDatasetParams(filenames,
                                        /*header_bytes=*/5,
                                        /*record_bytes=*/3,
                                        /*footer_bytes=*/2,
                                        /*buffer_size=*/10,
                                        /*compression_type=*/compression_type,
                                        /*node_name=*/kNodeName);
}

// Test case 2: multiple fixed-length record files with GZIP compression.
FixedLengthRecordDatasetParams FixedLengthRecordDatasetParams2() {
  std::vector<tstring> filenames = {
      absl::StrCat(testing::TmpDir(), "/text_line_GZIP_1"),
      absl::StrCat(testing::TmpDir(), "/text_line_GZIP_2")};
  std::vector<string> contents = {
      absl::StrCat("HHHHH", "111", "222", "333", "FF"),
      absl::StrCat("HHHHH", "aaa", "bbb", "FF")};
  CompressionType compression_type = CompressionType::GZIP;
  if (!CreateTestFiles(filenames, contents, compression_type).ok()) {
    VLOG(WARNING) << "Failed to create the test files: "
                  << absl::StrJoin(filenames, ", ");
  }
  return FixedLengthRecordDatasetParams(filenames,
                                        /*header_bytes=*/5,
                                        /*record_bytes=*/3,
                                        /*footer_bytes=*/2,
                                        /*buffer_size=*/10,
                                        /*compression_type=*/compression_type,
                                        /*node_name=*/kNodeName);
}

// Test case 3: multiple fixed-length record files without compression.
FixedLengthRecordDatasetParams FixedLengthRecordDatasetParams3() {
  std::vector<tstring> filenames = {
      absl::StrCat(testing::TmpDir(), "/text_line_UNCOMPRESSED_1"),
      absl::StrCat(testing::TmpDir(), "/text_line_UNCOMPRESSED_2")};
  std::vector<string> contents = {
      absl::StrCat("HHHHH", "111", "222", "333", "FF"),
      absl::StrCat("HHHHH", "aaa", "bbb", "FF")};
  CompressionType compression_type = CompressionType::UNCOMPRESSED;
  if (!CreateTestFiles(filenames, contents, compression_type).ok()) {
    VLOG(WARNING) << "Failed to create the test files: "
                  << absl::StrJoin(filenames, ", ");
  }
  return FixedLengthRecordDatasetParams(filenames,
                                        /*header_bytes=*/5,
                                        /*record_bytes=*/3,
                                        /*footer_bytes=*/2,
                                        /*buffer_size=*/10,
                                        /*compression_type=*/compression_type,
                                        /*node_name=*/kNodeName);
}

std::vector<GetNextTestCase<FixedLengthRecordDatasetParams>>
GetNextTestCases() {
  return {
      {/*dataset_params=*/FixedLengthRecordDatasetParams1(),
       /*expected_outputs=*/
       CreateTensors<tstring>(TensorShape({}),
                              {{"111"}, {"222"}, {"333"}, {"aaa"}, {"bbb"}})},
      {/*dataset_params=*/FixedLengthRecordDatasetParams2(),
       CreateTensors<tstring>(TensorShape({}),
                              {{"111"}, {"222"}, {"333"}, {"aaa"}, {"bbb"}})},
      {/*dataset_params=*/FixedLengthRecordDatasetParams3(),
       CreateTensors<tstring>(TensorShape({}),
                              {{"111"}, {"222"}, {"333"}, {"aaa"}, {"bbb"}})}};
}

ITERATOR_GET_NEXT_TEST_P(FixedLengthRecordDatasetOpTest,
                         FixedLengthRecordDatasetParams, GetNextTestCases())

TEST_F(FixedLengthRecordDatasetOpTest, DatasetNodeName) {
  auto dataset_params = FixedLengthRecordDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(dataset_params.node_name()));
}

TEST_F(FixedLengthRecordDatasetOpTest, DatasetTypeString) {
  auto dataset_params = FixedLengthRecordDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  name_utils::OpNameParams params;
  params.op_version = kOpVersion;
  TF_ASSERT_OK(CheckDatasetTypeString(
      name_utils::OpName(FixedLengthRecordDatasetOp::kDatasetType, params)));
}

TEST_F(FixedLengthRecordDatasetOpTest, DatasetOutputDtypes) {
  auto dataset_params = FixedLengthRecordDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputDtypes({DT_STRING}));
}

TEST_F(FixedLengthRecordDatasetOpTest, DatasetOutputShapes) {
  auto dataset_params = FixedLengthRecordDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputShapes({PartialTensorShape({})}));
}

TEST_F(FixedLengthRecordDatasetOpTest, Cardinality) {
  auto dataset_params = FixedLengthRecordDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetCardinality(kUnknownCardinality));
}

TEST_F(FixedLengthRecordDatasetOpTest, IteratorOutputDtypes) {
  auto dataset_params = FixedLengthRecordDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputDtypes({DT_STRING}));
}

TEST_F(FixedLengthRecordDatasetOpTest, IteratorOutputShapes) {
  auto dataset_params = FixedLengthRecordDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputShapes({PartialTensorShape({})}));
}

TEST_F(FixedLengthRecordDatasetOpTest, IteratorPrefix) {
  auto dataset_params = FixedLengthRecordDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  name_utils::IteratorPrefixParams iterator_prefix_params;
  iterator_prefix_params.op_version = kOpVersion;
  TF_ASSERT_OK(CheckIteratorPrefix(name_utils::IteratorPrefix(
      FixedLengthRecordDatasetOp::kDatasetType,
      dataset_params.iterator_prefix(), iterator_prefix_params)));
}

std::vector<IteratorSaveAndRestoreTestCase<FixedLengthRecordDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {
      {/*dataset_params=*/FixedLengthRecordDatasetParams1(),
       /*breakpoints=*/{0, 2, 6},
       /*expected_outputs=*/
       CreateTensors<tstring>(TensorShape({}),
                              {{"111"}, {"222"}, {"333"}, {"aaa"}, {"bbb"}})},
      {/*dataset_params=*/FixedLengthRecordDatasetParams2(),
       /*breakpoints=*/{0, 2, 6},
       CreateTensors<tstring>(TensorShape({}),
                              {{"111"}, {"222"}, {"333"}, {"aaa"}, {"bbb"}})},
      {/*dataset_params=*/FixedLengthRecordDatasetParams3(),
       /*breakpoints=*/{0, 2, 6},
       CreateTensors<tstring>(TensorShape({}),
                              {{"111"}, {"222"}, {"333"}, {"aaa"}, {"bbb"}})}};
}

ITERATOR_SAVE_AND_RESTORE_TEST_P(FixedLengthRecordDatasetOpTest,
                                 FixedLengthRecordDatasetParams,
                                 IteratorSaveAndRestoreTestCases())

}  // namespace
}  // namespace data
}  // namespace tensorflow
