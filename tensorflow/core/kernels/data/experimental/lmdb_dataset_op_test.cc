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
#include "tensorflow/core/kernels/data/experimental/lmdb_dataset_op.h"

#include "tensorflow/core/kernels/data/dataset_test_base.h"

#include "tensorflow/core/platform/posix/posix_file_system.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace {

constexpr char kNodeName[] = "lmdb_dataset";
constexpr char kIteratorPrefix[] = "Iterator";
constexpr char kDataFileName[] = "data.mdb";
constexpr char kDataFileLoc[] = "tensorflow/core/lib/lmdb/testdata";

class LMDBDatasetParams : public DatasetParams {
 public:
  LMDBDatasetParams(std::vector<std::string> filenames,
                    DataTypeVector output_dtypes,
                    std::vector<PartialTensorShape> output_shapes,
                    string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(kNodeName)),
        filenames(CreateTensor<std::string>(
            TensorShape({static_cast<int>(filenames.size())}), filenames)) {}

  Status MakeInputs(gtl::InlinedVector<TensorValue, 4>* inputs) override {
    *inputs = {TensorValue(&filenames)};
    return Status::OK();
  }

 private:
  // Names of binary database files to read, boxed up inside a Tensor of
  // strings
  Tensor filenames;
};

class LMDBDatasetOpTest : public DatasetOpsTestBaseV2<LMDBDatasetParams> {
 public:
  Status Initialize(LMDBDatasetParams* dataset_params) override {
    // Step 1: Set up enough of a TF runtime to be able to invoke a kernel.
    TF_RETURN_IF_ERROR(InitThreadPool(thread_num_));
    TF_RETURN_IF_ERROR(InitFunctionLibraryRuntime({}, cpu_num_));

    // Step 2: Box up the inputs to the kernel inside TensorValue objects inside
    // a vector.
    gtl::InlinedVector<TensorValue, 4> inputs;
    TF_RETURN_IF_ERROR(dataset_params->MakeInputs(&inputs));

    // Step 3: Create a dataset kernel to test, passing in attributes of the
    // kernel.
    TF_RETURN_IF_ERROR(
        CreateLMDBDatasetOpKernel(*dataset_params, &dataset_kernel_));

    // Step 4: Create a context in which the kernel will operate. This is where
    // the kernel gets initialized with its inputs
    TF_RETURN_IF_ERROR(
        CreateDatasetContext(dataset_kernel_.get(), &inputs, &dataset_ctx_));

    // Step 5: Unbox the DatasetBase object inside the variant tensor backing
    // the kernel.
    TF_RETURN_IF_ERROR(
        CreateDataset(dataset_kernel_.get(), dataset_ctx_.get(), &dataset_));

    // Step 6: Create an iterator in case the test needs to read the output of
    // the dataset.
    TF_RETURN_IF_ERROR(
        CreateIteratorContext(dataset_ctx_.get(), &iterator_ctx_));
    TF_RETURN_IF_ERROR(dataset_->MakeIterator(iterator_ctx_.get(),
                                              kIteratorPrefix, &iterator_));

    return Status::OK();
  }

 protected:
  Status CreateLMDBDatasetOpKernel(const LMDBDatasetParams& dataset_params,
                                   std::unique_ptr<OpKernel>* op_kernel) {
    NodeDef node_def = test::function::NDef(
        kNodeName, name_utils::OpName(LMDBDatasetOp::kDatasetType),
        // Inputs
        {LMDBDatasetOp::kFileNames},
        // Attributes
        {{LMDBDatasetOp::kOutputTypes, dataset_params.output_dtypes},
         {LMDBDatasetOp::kOutputShapes, dataset_params.output_shapes}});
    TF_RETURN_IF_ERROR(CreateOpKernel(node_def, op_kernel));
    return Status::OK();
  }
};

// Copy our test data file to the current test program's temporary
// directory, and return the full path to the copied file.
// This copying is necessary because LMDB creates lock files adjacent
// to files that it reads.
string MaybeCopyDataFile() {
  string src_loc = io::JoinPath(kDataFileLoc, kDataFileName);
  string dest_loc = io::JoinPath(testing::TmpDir(), kDataFileName);

  PosixFileSystem fs;
  // PosixFileSystem::FileExists currently returns Status::OK() if the file
  // exists and errors::NotFound() if the file doesn't exist. There's no
  // indication in the code or docs about whether other error codes may be
  // added in the future, so we code defensively here.
  Status exists_status = fs.FileExists(dest_loc);
  if (exists_status.code() == error::NOT_FOUND) {
    TF_EXPECT_OK(fs.CopyFile(src_loc, dest_loc));
  } else {
    TF_EXPECT_OK(exists_status);
  }

  return dest_loc;
}

LMDBDatasetParams SingleValidInput() {
  return {/*filenames=*/{MaybeCopyDataFile()},
          /*output_dtypes=*/{DT_STRING, DT_STRING},
          /*output_shapes=*/{PartialTensorShape({}), PartialTensorShape({})},
          /*node_name=*/kNodeName};
}

LMDBDatasetParams TwoValidInputs() {
  return {/*filenames*/ {MaybeCopyDataFile(), MaybeCopyDataFile()},
          /*output_dtypes*/ {DT_STRING, DT_STRING},
          /*output_shapes*/ {PartialTensorShape({}), PartialTensorShape({})},
          /*node_name=*/kNodeName};
}

LMDBDatasetParams EmptyInput() {
  return {/*filenames*/ {},
          /*output_dtypes*/ {DT_STRING, DT_STRING},
          /*output_shapes*/ {PartialTensorShape({}), PartialTensorShape({})},
          /*node_name=*/kNodeName};
}

LMDBDatasetParams InvalidPathAtStart() {
  return {/*filenames*/ {"This is not a valid filename", MaybeCopyDataFile()},
          /*output_dtypes*/ {DT_STRING, DT_STRING},
          /*output_shapes*/ {PartialTensorShape({}), PartialTensorShape({})},
          /*node_name=*/kNodeName};
}

LMDBDatasetParams InvalidPathInMiddle() {
  return {/*filenames*/ {MaybeCopyDataFile(), "This is not a valid filename",
                         MaybeCopyDataFile()},
          /*output_dtypes*/ {DT_STRING, DT_STRING},
          /*output_shapes*/ {PartialTensorShape({}), PartialTensorShape({})},
          /*node_name=*/kNodeName};
}
class ParameterizedGetNextTest
    : public LMDBDatasetOpTest,
      public ::testing::WithParamInterface<GetNextTestCase<LMDBDatasetParams>> {
};

std::vector<Tensor> StringsToTensors(const std::vector<string> strs) {
  std::vector<Tensor> ret;
  for (string str : strs) {
    ret.push_back(CreateTensor<string>(TensorShape({}), {str}));
  }
  return ret;
}

// The tensors we expect to see each time we read through the input
// data file.
const std::vector<Tensor> kFileOutput = StringsToTensors({
    // Each call to GetNext() produces two scalar string tensors, but the test
    // harness expects to receive a flat vector
    "0", "a",  //
    "1", "b",  //
    "2", "c",  //
    "3", "d",  //
    "4", "e",  //
    "5", "f",  //
    "6", "g",  //
    "7", "h",  //
    "8", "i",  //
    "9", "j",  //
});

std::vector<GetNextTestCase<LMDBDatasetParams>> GetNextTestCases() {
  // STL vectors don't have a "concatenate two vectors into a new vector"
  // operation, so...
  std::vector<Tensor> output_twice;
  output_twice.insert(output_twice.end(), kFileOutput.cbegin(),
                      kFileOutput.cend());
  output_twice.insert(output_twice.end(), kFileOutput.cbegin(),
                      kFileOutput.cend());

  return {
      {/*dataset_params=*/SingleValidInput(), /*expected_outputs=*/kFileOutput},
      {/*dataset_params=*/TwoValidInputs(), /*expected_outputs=*/output_twice},
      {/*dataset_params=*/EmptyInput(), /*expected_outputs=*/{}}};
}

TEST_P(ParameterizedGetNextTest, GetNext) {
  auto test_case = GetParam();
  TF_ASSERT_OK(Initialize(&test_case.dataset_params));
  TF_ASSERT_OK(
      CheckIteratorGetNext(test_case.expected_outputs, /*compare_order=*/true));
}

INSTANTIATE_TEST_SUITE_P(LMDBDatasetOpTest, ParameterizedGetNextTest,
                         ::testing::ValuesIn(GetNextTestCases()));

TEST_F(LMDBDatasetOpTest, InvalidPathAtStart) {
  auto dataset_params = InvalidPathAtStart();
  TF_ASSERT_OK(Initialize(&dataset_params));

  // Errors about invalid files are only raised when attempting to read data.
  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  std::vector<Tensor> next;

  Status get_next_status =
      iterator_->GetNext(iterator_ctx_.get(), &next, &end_of_sequence);

  EXPECT_EQ(get_next_status.code(), error::INVALID_ARGUMENT);
}

TEST_F(LMDBDatasetOpTest, InvalidPathInMiddle) {
  auto dataset_params = InvalidPathInMiddle();
  TF_ASSERT_OK(Initialize(&dataset_params));

  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  std::vector<Tensor> next;

  // First 10 rows should be ok
  for (int i = 0; i < 10; ++i) {
    TF_ASSERT_OK(
        iterator_->GetNext(iterator_ctx_.get(), &next, &end_of_sequence));
    EXPECT_FALSE(end_of_sequence);
  }

  // Next read operation should raise an error
  Status get_next_status =
      iterator_->GetNext(iterator_ctx_.get(), &next, &end_of_sequence);
  EXPECT_EQ(get_next_status.code(), error::INVALID_ARGUMENT);
}

TEST_F(LMDBDatasetOpTest, DatasetNodeName) {
  auto dataset_params = SingleValidInput();
  TF_ASSERT_OK(Initialize(&dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(dataset_params.node_name));
}

TEST_F(LMDBDatasetOpTest, DatasetTypeString) {
  auto dataset_params = SingleValidInput();
  TF_ASSERT_OK(Initialize(&dataset_params));
  TF_ASSERT_OK(
      CheckDatasetTypeString(name_utils::OpName(LMDBDatasetOp::kDatasetType)));
}

TEST_F(LMDBDatasetOpTest, DatasetOutputDtypes) {
  auto dataset_params = SingleValidInput();
  TF_ASSERT_OK(Initialize(&dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputDtypes({DT_STRING, DT_STRING}));
}

TEST_F(LMDBDatasetOpTest, DatasetOutputShapes) {
  auto dataset_params = SingleValidInput();
  TF_ASSERT_OK(Initialize(&dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputShapes(
      {PartialTensorShape({}), PartialTensorShape({})}));
}

TEST_F(LMDBDatasetOpTest, Cardinality) {
  auto dataset_params = SingleValidInput();
  TF_ASSERT_OK(Initialize(&dataset_params));
  TF_ASSERT_OK(CheckDatasetCardinality(kUnknownCardinality));
}

TEST_F(LMDBDatasetOpTest, IteratorOutputDtypes) {
  auto dataset_params = SingleValidInput();
  TF_ASSERT_OK(Initialize(&dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputDtypes({DT_STRING, DT_STRING}));
}

TEST_F(LMDBDatasetOpTest, IteratorOutputShapes) {
  auto dataset_params = SingleValidInput();
  TF_ASSERT_OK(Initialize(&dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputShapes(
      {PartialTensorShape({}), PartialTensorShape({})}));
}

TEST_F(LMDBDatasetOpTest, IteratorOutputPrefix) {
  auto dataset_params = SingleValidInput();
  TF_ASSERT_OK(Initialize(&dataset_params));
  TF_ASSERT_OK(CheckIteratorPrefix(name_utils::IteratorPrefix(
      LMDBDatasetOp::kDatasetType, kIteratorPrefix)));
}

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
