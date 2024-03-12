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
#include "tensorflow/core/kernels/data/text_line_dataset_op.h"

#include "tensorflow/core/data/dataset_test_base.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "text_line_dataset";

tstring LocalTempFilename() {
  std::string path;
  CHECK(Env::Default()->LocalTempFilename(&path));
  return tstring(path);
}

class TextLineDatasetParams : public DatasetParams {
 public:
  TextLineDatasetParams(std::vector<tstring> filenames,
                        CompressionType compression_type, int64_t buffer_size,
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
        CreateTensor<int64_t>(TensorShape({}), {buffer_size_})};
  }

  Status GetInputNames(std::vector<string>* input_names) const override {
    input_names->clear();
    *input_names = {
        TextLineDatasetOp::kFileNames,
        TextLineDatasetOp::kCompressionType,
        TextLineDatasetOp::kBufferSize,
    };
    return absl::OkStatus();
  }

  Status GetAttributes(AttributeVector* attr_vector) const override {
    attr_vector->clear();
    attr_vector->emplace_back("metadata", "");
    return absl::OkStatus();
  }

  string dataset_type() const override {
    return TextLineDatasetOp::kDatasetType;
  }

 private:
  std::vector<tstring> filenames_;
  CompressionType compression_type_;
  int64_t buffer_size_;
};

class TextLineDatasetOpTest : public DatasetOpsTestBase {};

Status CreateTestFiles(const std::vector<tstring>& filenames,
                       const std::vector<tstring>& contents,
                       CompressionType compression_type) {
  if (filenames.size() != contents.size()) {
    return tensorflow::errors::InvalidArgument(
        "The number of files does not match with the contents");
  }
  CompressionParams params;
  params.output_buffer_size = 10;
  params.compression_type = compression_type;
  for (int i = 0; i < filenames.size(); ++i) {
    TF_RETURN_IF_ERROR(
        WriteDataToFile(filenames[i], contents[i].data(), params));
  }
  return absl::OkStatus();
}

// Test case 1: multiple text files with ZLIB compression.
TextLineDatasetParams TextLineDatasetParams1() {
  std::vector<tstring> filenames = {LocalTempFilename(), LocalTempFilename()};
  std::vector<tstring> contents = {
      absl::StrCat("hello world\n", "11223334455\n"),
      absl::StrCat("abcd, EFgH\n", "           \n", "$%^&*()\n")};
  CompressionType compression_type = CompressionType::ZLIB;
  if (!CreateTestFiles(filenames, contents, compression_type).ok()) {
    VLOG(WARNING) << "Failed to create the test files: "
                  << absl::StrJoin(filenames, ", ");
  }
  return TextLineDatasetParams(filenames,
                               /*compression_type=*/compression_type,
                               /*buffer_size=*/10,
                               /*node_name=*/kNodeName);
}

// Test case 2: multiple text files with GZIP compression.
TextLineDatasetParams TextLineDatasetParams2() {
  std::vector<tstring> filenames = {LocalTempFilename(), LocalTempFilename()};
  std::vector<tstring> contents = {
      absl::StrCat("hello world\n", "11223334455\n"),
      absl::StrCat("abcd, EFgH\n", "           \n", "$%^&*()\n")};
  CompressionType compression_type = CompressionType::GZIP;
  if (!CreateTestFiles(filenames, contents, compression_type).ok()) {
    VLOG(WARNING) << "Failed to create the test files: "
                  << absl::StrJoin(filenames, ", ");
  }
  return TextLineDatasetParams(filenames,
                               /*compression_type=*/compression_type,
                               /*buffer_size=*/10,
                               /*node_name=*/kNodeName);
}

// Test case 3: multiple text files without compression.
TextLineDatasetParams TextLineDatasetParams3() {
  std::vector<tstring> filenames = {LocalTempFilename(), LocalTempFilename()};
  std::vector<tstring> contents = {
      absl::StrCat("hello world\n", "11223334455\n"),
      absl::StrCat("abcd, EFgH\n", "           \n", "$%^&*()\n")};
  CompressionType compression_type = CompressionType::UNCOMPRESSED;
  if (!CreateTestFiles(filenames, contents, compression_type).ok()) {
    VLOG(WARNING) << "Failed to create the test files: "
                  << absl::StrJoin(filenames, ", ");
  }
  return TextLineDatasetParams(filenames,
                               /*compression_type=*/compression_type,
                               /*buffer_size=*/10,
                               /*node_name=*/kNodeName);
}

std::vector<GetNextTestCase<TextLineDatasetParams>> GetNextTestCases() {
  return {{/*dataset_params=*/TextLineDatasetParams1(),
           /*expected_outputs=*/
           CreateTensors<tstring>(TensorShape({}), {{"hello world"},
                                                    {"11223334455"},
                                                    {"abcd, EFgH"},
                                                    {"           "},
                                                    {"$%^&*()"}})},
          {/*dataset_params=*/TextLineDatasetParams2(),
           CreateTensors<tstring>(TensorShape({}), {{"hello world"},
                                                    {"11223334455"},
                                                    {"abcd, EFgH"},
                                                    {"           "},
                                                    {"$%^&*()"}})},
          {/*dataset_params=*/TextLineDatasetParams3(),
           CreateTensors<tstring>(TensorShape({}), {{"hello world"},
                                                    {"11223334455"},
                                                    {"abcd, EFgH"},
                                                    {"           "},
                                                    {"$%^&*()"}})}};
}

ITERATOR_GET_NEXT_TEST_P(TextLineDatasetOpTest, TextLineDatasetParams,
                         GetNextTestCases())

TEST_F(TextLineDatasetOpTest, DatasetNodeName) {
  auto dataset_params = TextLineDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(dataset_params.node_name()));
}

TEST_F(TextLineDatasetOpTest, DatasetTypeString) {
  auto dataset_params = TextLineDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetTypeString(
      name_utils::OpName(TextLineDatasetOp::kDatasetType)));
}

TEST_F(TextLineDatasetOpTest, DatasetOutputDtypes) {
  auto dataset_params = TextLineDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputDtypes({DT_STRING}));
}

TEST_F(TextLineDatasetOpTest, DatasetOutputShapes) {
  auto dataset_params = TextLineDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputShapes({PartialTensorShape({})}));
}

TEST_F(TextLineDatasetOpTest, Cardinality) {
  auto dataset_params = TextLineDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetCardinality(kUnknownCardinality));
}

TEST_F(TextLineDatasetOpTest, IteratorOutputDtypes) {
  auto dataset_params = TextLineDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputDtypes({DT_STRING}));
}

TEST_F(TextLineDatasetOpTest, IteratorOutputShapes) {
  auto dataset_params = TextLineDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputShapes({PartialTensorShape({})}));
}

TEST_F(TextLineDatasetOpTest, IteratorPrefix) {
  auto dataset_params = TextLineDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));

  TF_ASSERT_OK(CheckIteratorPrefix(name_utils::IteratorPrefix(
      TextLineDatasetOp::kDatasetType, dataset_params.iterator_prefix())));
}

std::vector<IteratorSaveAndRestoreTestCase<TextLineDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {{/*dataset_params=*/TextLineDatasetParams1(),
           /*breakpoints=*/{0, 2, 6},
           /*expected_outputs=*/
           CreateTensors<tstring>(TensorShape({}), {{"hello world"},
                                                    {"11223334455"},
                                                    {"abcd, EFgH"},
                                                    {"           "},
                                                    {"$%^&*()"}})},
          {/*dataset_params=*/TextLineDatasetParams2(),
           /*breakpoints=*/{0, 2, 6},
           CreateTensors<tstring>(TensorShape({}), {{"hello world"},
                                                    {"11223334455"},
                                                    {"abcd, EFgH"},
                                                    {"           "},
                                                    {"$%^&*()"}})},
          {/*dataset_params=*/TextLineDatasetParams3(),
           /*breakpoints=*/{0, 2, 6},
           CreateTensors<tstring>(TensorShape({}), {{"hello world"},
                                                    {"11223334455"},
                                                    {"abcd, EFgH"},
                                                    {"           "},
                                                    {"$%^&*()"}})}};
}

ITERATOR_SAVE_AND_RESTORE_TEST_P(TextLineDatasetOpTest, TextLineDatasetParams,
                                 IteratorSaveAndRestoreTestCases())

}  // namespace
}  // namespace data
}  // namespace tensorflow
