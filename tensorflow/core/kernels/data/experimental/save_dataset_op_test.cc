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
#include "tensorflow/core/kernels/data/experimental/save_dataset_op.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/data/dataset_test_base.h"
#include "tensorflow/core/data/serialization_utils.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"

namespace tensorflow {
namespace data {
namespace experimental {

constexpr char kSaveDatasetV2NodeName[] = "save_dataset_v2";

class SaveDatasetV2Params : public DatasetParams {
 public:
  template <typename T>
  SaveDatasetV2Params(T input_dataset_params, const tstring& path,
                      const std::string& compression,
                      FunctionDefHelper::AttrValueWrapper shard_func,
                      std::vector<FunctionDef> func_lib, bool use_shard_func,
                      DataTypeVector output_dtypes,
                      std::vector<PartialTensorShape> output_shapes,
                      string node_name, DataTypeVector type_arguments)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        path_(path),
        compression_(compression),
        shard_func_(shard_func),
        func_lib_(std::move(func_lib)),
        use_shard_func_(use_shard_func),
        type_arguments_(std::move(type_arguments)) {
    input_dataset_params_.push_back(std::make_unique<T>(input_dataset_params));
    iterator_prefix_ =
        name_utils::IteratorPrefix(input_dataset_params.dataset_type(),
                                   input_dataset_params.iterator_prefix());
  }

  std::vector<Tensor> GetInputTensors() const override {
    std::vector<Tensor> input_tensors;
    input_tensors.emplace_back(CreateTensor<tstring>(TensorShape({}), {path_}));
    return input_tensors;
  }

  Status GetInputNames(std::vector<string>* input_names) const override {
    input_names->clear();
    input_names->emplace_back(SaveDatasetV2Op::kInputDataset);
    input_names->emplace_back(SaveDatasetV2Op::kPath);
    return OkStatus();
  }

  Status GetAttributes(AttributeVector* attr_vector) const override {
    attr_vector->clear();
    attr_vector->emplace_back(SaveDatasetV2Op::kCompression, compression_);
    attr_vector->emplace_back(SaveDatasetV2Op::kShardFunc, shard_func_);
    attr_vector->emplace_back(SaveDatasetV2Op::kUseShardFunc, use_shard_func_);
    attr_vector->emplace_back(SaveDatasetV2Op::kShardFuncTarguments,
                              type_arguments_);
    attr_vector->emplace_back(SaveDatasetV2Op::kOutputTypes, output_dtypes_);
    attr_vector->emplace_back(SaveDatasetV2Op::kOutputShapes, output_shapes_);
    return OkStatus();
  }

  string path() const { return path_; }

  string dataset_type() const override { return SaveDatasetV2Op::kDatasetType; }

  string op_name() const override { return "SaveDatasetV2"; }

  std::vector<FunctionDef> func_lib() const override { return func_lib_; }

 private:
  std::string path_;
  std::string compression_;
  FunctionDefHelper::AttrValueWrapper shard_func_;
  std::vector<FunctionDef> func_lib_;
  bool use_shard_func_;
  DataTypeVector type_arguments_;
};

class SaveDatasetV2OpTest : public DatasetOpsTestBase {
 public:
  Status Initialize(const DatasetParams& dataset_params) {
    TF_RETURN_IF_ERROR(DatasetOpsTestBase::Initialize(dataset_params));
    auto params = static_cast<const SaveDatasetV2Params&>(dataset_params);
    save_filename_ = params.path();
    return OkStatus();
  }

 protected:
  std::string save_filename_;
};

// Test case 1. Basic save parameters.
SaveDatasetV2Params SaveDatasetV2Params1() {
  return SaveDatasetV2Params(
      RangeDatasetParams(0, 10, 2),
      /*path=*/io::JoinPath(testing::TmpDir(), "save_data"),
      /*compression=*/"",
      /*shard_func=*/
      FunctionDefHelper::FunctionRef("XTimesTwo", {{"T", DT_INT64}}),
      /*func_lib=*/{test::function::XTimesTwo()},
      /*use_shard_func=*/false,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({})},
      /*node_name=*/kSaveDatasetV2NodeName,
      /*type_arguments=*/{});
}

// Test case 2. Tests custom compression settings and uses shard func.
SaveDatasetV2Params SaveDatasetV2Params2() {
  return SaveDatasetV2Params(
      RangeDatasetParams(0, 5, 1),
      /*path=*/io::JoinPath(testing::TmpDir(), "save_data"),
      /*compression=*/"GZIP",
      /*shard_func=*/
      FunctionDefHelper::FunctionRef("XTimesTwo", {{"T", DT_INT64}}),
      /*func_lib=*/{test::function::XTimesTwo()},
      /*use_shard_func=*/true,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({})},
      /*node_name=*/kSaveDatasetV2NodeName,
      /*type_arguments=*/{});
}

std::vector<GetNextTestCase<SaveDatasetV2Params>> GetNextTestCases() {
  return {{/*dataset_params=*/
           SaveDatasetV2Params1(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({}), {{0}, {2}, {4}, {6}, {8}})},
          {/*dataset_params=*/SaveDatasetV2Params2(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({}), {{0}, {1}, {2}, {3}, {4}})}};
}

class ParameterizedGetNextTest : public SaveDatasetV2OpTest,
                                 public ::testing::WithParamInterface<
                                     GetNextTestCase<SaveDatasetV2Params>> {};

TEST_P(ParameterizedGetNextTest, GetNext) {
  auto test_case = GetParam();
  TF_ASSERT_OK(Initialize(test_case.dataset_params));

  // Test the write mode.
  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;

  while (!end_of_sequence) {
    std::vector<Tensor> next;
    TF_EXPECT_OK(
        iterator_->GetNext(iterator_ctx_.get(), &next, &end_of_sequence));
    out_tensors.insert(out_tensors.end(), next.begin(), next.end());
  }
  TF_EXPECT_OK(ExpectEqual(out_tensors, test_case.expected_outputs,
                           /*compare_order=*/true));
}

INSTANTIATE_TEST_SUITE_P(SaveDatasetV2OpTest, ParameterizedGetNextTest,
                         ::testing::ValuesIn(GetNextTestCases()));

TEST_F(SaveDatasetV2OpTest, DatasetNodeName) {
  auto dataset_params = SaveDatasetV2Params1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(dataset_params.node_name()));
}

TEST_F(SaveDatasetV2OpTest, DatasetTypeString) {
  auto dataset_params = SaveDatasetV2Params1();
  TF_ASSERT_OK(Initialize(dataset_params));
  name_utils::OpNameParams params;
  params.op_version = dataset_params.op_version();
  TF_ASSERT_OK(CheckDatasetTypeString("SaveDatasetV2"));
}

TEST_F(SaveDatasetV2OpTest, DatasetOutputDtypes) {
  auto dataset_params = SaveDatasetV2Params1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputDtypes(dataset_params.output_dtypes()));
}

std::vector<DatasetOutputDtypesTestCase<SaveDatasetV2Params>>
DatasetOutputDtypesTestCases() {
  return {{/*dataset_params=*/SaveDatasetV2Params1(),
           /*expected_output_dtypes=*/{DT_INT64}},
          {/*dataset_params=*/SaveDatasetV2Params2(),
           /*expected_output_dtypes=*/{DT_INT64}}};
}

DATASET_OUTPUT_DTYPES_TEST_P(SaveDatasetV2OpTest, SaveDatasetV2Params,
                             DatasetOutputDtypesTestCases())

std::vector<DatasetOutputShapesTestCase<SaveDatasetV2Params>>
DatasetOutputShapesTestCases() {
  return {{/*dataset_params=*/SaveDatasetV2Params1(),
           /*expected_output_shapes=*/{PartialTensorShape({})}},
          {/*dataset_params=*/SaveDatasetV2Params2(),
           /*expected_output_shapes=*/{PartialTensorShape({})}}};
}

DATASET_OUTPUT_SHAPES_TEST_P(SaveDatasetV2OpTest, SaveDatasetV2Params,
                             DatasetOutputShapesTestCases())

std::vector<CardinalityTestCase<SaveDatasetV2Params>> CardinalityTestCases() {
  return {{/*dataset_params=*/SaveDatasetV2Params1(),
           /*expected_cardinality=*/5},
          {/*dataset_params=*/SaveDatasetV2Params2(),
           /*expected_cardinality=*/5}};
}

DATASET_CARDINALITY_TEST_P(SaveDatasetV2OpTest, SaveDatasetV2Params,
                           CardinalityTestCases())

TEST_F(SaveDatasetV2OpTest, IteratorPrefix) {
  auto dataset_params = SaveDatasetV2Params1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorPrefix(name_utils::IteratorPrefix(
      SaveDatasetV2Op::kDatasetType, dataset_params.iterator_prefix())));
}

std::vector<IteratorSaveAndRestoreTestCase<SaveDatasetV2Params>>
IteratorSaveAndRestoreTestCases() {
  return {{/*dataset_params=*/SaveDatasetV2Params1(),
           /*breakpoints=*/{0, 2, 4, 6, 8},
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({}), {{0}, {2}, {4}, {6}, {8}})},
          {/*dataset_params=*/SaveDatasetV2Params2(),
           /*breakpoints=*/{0, 2, 5},
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({}), {{0}, {1}, {2}, {3}, {4}})}};
}

class ParameterizedIteratorSaveAndRestoreTest
    : public SaveDatasetV2OpTest,
      public ::testing::WithParamInterface<
          IteratorSaveAndRestoreTestCase<SaveDatasetV2Params>> {};

TEST_P(ParameterizedIteratorSaveAndRestoreTest, SaveAndRestore) {
  auto test_case = GetParam();
  TF_ASSERT_OK(Initialize(test_case.dataset_params));

  std::unique_ptr<SerializationContext> serialization_ctx;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_ctx));

  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  int cur_iteration = 0;
  const std::vector<int>& breakpoints = test_case.breakpoints;
  for (int breakpoint : breakpoints) {
    VariantTensorDataWriter writer;
    TF_EXPECT_OK(iterator_->Save(serialization_ctx.get(), &writer));
    std::vector<const VariantTensorData*> data;
    writer.GetData(&data);
    VariantTensorDataReader reader(data);
    TF_EXPECT_OK(RestoreIterator(iterator_ctx_.get(), &reader,
                                 test_case.dataset_params.iterator_prefix(),
                                 *dataset_, &iterator_));

    while (cur_iteration <= breakpoint) {
      std::vector<Tensor> next;
      TF_EXPECT_OK(
          iterator_->GetNext(iterator_ctx_.get(), &next, &end_of_sequence));
      out_tensors.insert(out_tensors.end(), next.begin(), next.end());
      cur_iteration++;
    }
  }

  TF_EXPECT_OK(ExpectEqual(out_tensors, test_case.expected_outputs,
                           /*compare_order=*/true));
}

INSTANTIATE_TEST_CASE_P(SaveDatasetV2OpTest,
                        ParameterizedIteratorSaveAndRestoreTest,
                        ::testing::ValuesIn(IteratorSaveAndRestoreTestCases()));

}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
