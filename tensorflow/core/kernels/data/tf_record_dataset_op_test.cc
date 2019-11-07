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

#include "tensorflow/core/kernels/data/dataset_test_base.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "tf_record_dataset";

class TFRecordDatasetParams : public DatasetParams {
 public:
  TFRecordDatasetParams(std::vector<tstring> filenames,
                        CompressionType compression_type, int64 buffer_size,
                        string node_name)
      : DatasetParams({DT_STRING}, {PartialTensorShape({})},
                      std::move(node_name)),
        filenames_(std::move(filenames)),
        compression_type_(compression_type),
        buffer_size_(buffer_size) {}

  std::vector<Tensor> GetInputTensors() const override {
    int num_files = filenames_.size();
    return {
        CreateTensor<tstring>(TensorShape({num_files}), filenames_),
        CreateTensor<tstring>(TensorShape({}), {ToString(compression_type_)}),
        CreateTensor<int64>(TensorShape({}), {buffer_size_})};
  }

  Status GetInputNames(std::vector<string>* input_names) const override {
    input_names->clear();
    *input_names = {
        TFRecordDatasetOp::kFileNames,
        TFRecordDatasetOp::kCompressionType,
        TFRecordDatasetOp::kBufferSize,
    };
    return Status::OK();
  }

  Status GetAttributes(AttributeVector* attr_vector) const override {
    *attr_vector = {};
    return Status::OK();
  }

  string dataset_type() const override {
    return TFRecordDatasetOp::kDatasetType;
  }

 private:
  std::vector<tstring> filenames_;
  CompressionType compression_type_;
  int64 buffer_size_;
};

class TFRecordDatasetOpTest : public DatasetOpsTestBaseV2 {};

Status CreateTestFiles(const std::vector<tstring>& filenames,
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
  return Status::OK();
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
    VLOG(WARNING) << "Failed to create the test files: "
                  << absl::StrJoin(filenames, ", ");
  }
  return TFRecordDatasetParams(filenames,
                               /*compression_type=*/compression_type,
                               /*buffer_size=*/10,
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
    VLOG(WARNING) << "Failed to create the test files: "
                  << absl::StrJoin(filenames, ", ");
  }
  return TFRecordDatasetParams(filenames,
                               /*compression_type=*/compression_type,
                               /*buffer_size=*/10,
                               /*node_name=*/kNodeName);
}

// Test case 3: multiple text files without compression.
TFRecordDatasetParams TFRecordDatasetParams3() {
  std::vector<tstring> filenames = {
      absl::StrCat(testing::TmpDir(), "/tf_record_UNCOMPRESSED_1"),
      absl::StrCat(testing::TmpDir(), "/tf_record_UNCOMPRESSED_2")};
  std::vector<std::vector<string>> contents = {{"1", "22", "333"},
                                               {"a", "bb", "ccc"}};
  CompressionType compression_type = CompressionType::GZIP;
  if (!CreateTestFiles(filenames, contents, compression_type).ok()) {
    VLOG(WARNING) << "Failed to create the test files: "
                  << absl::StrJoin(filenames, ", ");
  }
  return TFRecordDatasetParams(filenames,
                               /*compression_type=*/compression_type,
                               /*buffer_size=*/10,
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
           TensorShape({}), {{"1"}, {"22"}, {"333"}, {"a"}, {"bb"}, {"ccc"}})}};
}

ITERATOR_GET_NEXT_TEST_P(TFRecordDatasetOpTest, TFRecordDatasetParams,
                         GetNextTestCases())

TEST_F(TFRecordDatasetOpTest, DatasetNodeName) {
  auto dataset_params = TFRecordDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(dataset_params.node_name()));
}

TEST_F(TFRecordDatasetOpTest, DatasetTypeString) {
  auto dataset_params = TFRecordDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetTypeString(
      name_utils::OpName(TFRecordDatasetOp::kDatasetType)));
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
  TF_ASSERT_OK(CheckIteratorPrefix(name_utils::IteratorPrefix(
      TFRecordDatasetOp::kDatasetType, dataset_params.iterator_prefix())));
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
