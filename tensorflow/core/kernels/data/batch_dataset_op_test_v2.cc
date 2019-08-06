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
#include "tensorflow/core/kernels/data/batch_dataset_op.h"

#include "tensorflow/core/kernels/data/dataset_test_base.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "batch_dataset_v2";
constexpr int kOpVersion = 2;
constexpr char kIteratorPrefix[] = "Iterator";

class BatchDatasetParams : public DatasetParams {
 public:
  BatchDatasetParams() = default;

  BatchDatasetParams(int64 num_input_elements, int64 batch_size,
                     bool drop_remainder, bool parallel_copy,
                     DataTypeVector output_dtypes,
                     std::vector<PartialTensorShape> output_shapes,
                     string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        num_input_elements(num_input_elements),
        batch_size(CreateTensor<int64>(TensorShape({}), {batch_size})),
        drop_remainder(CreateTensor<bool>(TensorShape({}), {drop_remainder})),
        parallel_copy(parallel_copy) {}

  Status MakeInputs(gtl::InlinedVector<TensorValue, 4>* inputs) override {
    if (input_dataset.NumElements() == 0 ||
        input_dataset.dtype() != DT_VARIANT) {
      return tensorflow::errors::Internal(
          "The input dataset is not populated as the dataset tensor yet.");
    }
    *inputs = {TensorValue(&input_dataset), TensorValue(&batch_size),
               TensorValue(&drop_remainder)};
    return Status::OK();
  }

  int64 num_input_elements;
  Tensor input_dataset;
  Tensor batch_size;
  Tensor drop_remainder;
  bool parallel_copy;
};

class BatchDatasetOpTest : public DatasetOpsTestBase {
 public:
  Status SetupTestEnv(int thread_num, int cpu_num,
                      const std::vector<FunctionDef>& flib) {
    TF_RETURN_IF_ERROR(InitThreadPool(thread_num));
    TF_RETURN_IF_ERROR(InitFunctionLibraryRuntime(flib, cpu_num));
  }

  Status MakeDatasetAndIterator(BatchDatasetParams batch_dataset_params,
                                bool create_iterator) {
    batch_dataset_params_ = std::move(batch_dataset_params);

    // Populate the `input_dataset` in `batch_dataset_params_`.
    RangeDatasetParams input_dataset_params(
        0, batch_dataset_params_.num_input_elements, 1, {DT_INT64},
        {PartialTensorShape({})}, "");
    TF_RETURN_IF_ERROR(MakeRangeDataset(input_dataset_params,
                                        &batch_dataset_params_.input_dataset));
    // Create the dataset kernel.
    TF_RETURN_IF_ERROR(CreateBatchDatasetOpKernel(batch_dataset_params_,
                                                  &batch_dataset_kernel_));
    // Create the inputs for the dataset op.
    gtl::InlinedVector<TensorValue, 4> inputs;
    TF_RETURN_IF_ERROR(batch_dataset_params_.MakeInputs(&inputs));
    // Creat the dataset context.
    TF_RETURN_IF_ERROR(CreateBatchDatasetContext(
        batch_dataset_kernel_.get(), &inputs, &batch_dataset_context_));
    // Create the dataset.
    DatasetBase* batch_dataset;
    TF_RETURN_IF_ERROR(CreateDataset(batch_dataset_kernel_.get(),
                                     batch_dataset_context_.get(),
                                     &batch_dataset));
    batch_dataset_.reset(batch_dataset);

    if (!create_iterator) {
      return Status::OK();
    }

    // Create the iterator context.
    TF_RETURN_IF_ERROR(
        CreateIteratorContext(batch_dataset_context_.get(), &iterator_ctx_));
    // Create the iterator.
    TF_RETURN_IF_ERROR(batch_dataset_->MakeIterator(
        iterator_ctx_.get(), kIteratorPrefix, &iterator_));
  }

  Status CheckIteratorGetNext(const std::vector<Tensor>& expected_outputs,
                              bool compare_order) {
    TF_RETURN_IF_ERROR(DatasetOpsTestBase::CheckIteratorGetNext(
        iterator_.get(), iterator_ctx_.get(), expected_outputs, compare_order));
    return Status::OK();
  }

  Status CheckDatasetNodeName(const string& expected_node_name) {
    TF_RETURN_IF_ERROR(DatasetOpsTestBase::CheckDatasetNodeName(
        *batch_dataset_, expected_node_name));
    return Status::OK();
  }

  Status CheckDatasetTypeString(const string& expected_dataset_type_str) {
    TF_RETURN_IF_ERROR(DatasetOpsTestBase::CheckDatasetTypeString(
        *batch_dataset_, expected_dataset_type_str));
    return Status::OK();
  }

  Status CheckDatasetOutputDtypes(const DataTypeVector& expected_output_dtype) {
    TF_RETURN_IF_ERROR(DatasetOpsTestBase::CheckDatasetOutputDtypes(
        *batch_dataset_, expected_output_dtype));
    return Status::OK();
  }

  Status CheckDatasetOutputShapes(
      const std::vector<PartialTensorShape>& expected_output_shapes) {
    TF_RETURN_IF_ERROR(DatasetOpsTestBase::CheckDatasetOutputShapes(
        *batch_dataset_, expected_output_shapes));
    return Status::OK();
  }

  Status CheckDatasetCardinality(int expected_cardinality) {
    TF_RETURN_IF_ERROR(DatasetOpsTestBase::CheckDatasetCardinality(
        *batch_dataset_, expected_cardinality));
    return Status::OK();
  }

  Status CheckDatasetSave() {
    TF_RETURN_IF_ERROR(DatasetOpsTestBase::CheckDatasetSave(*batch_dataset_));
    return Status::OK();
  }

  Status CheckDatasetIsStateful(bool expected_stateful) {
    TF_RETURN_IF_ERROR(DatasetOpsTestBase::CheckDatasetIsStateful(
        *batch_dataset_, expected_stateful));
    return Status::OK();
  }

  Status CheckIteratorOutputDtypes(
      const DataTypeVector& expected_output_dtype) {
    TF_RETURN_IF_ERROR(DatasetOpsTestBase::CheckIteratorOutputDtypes(
        *iterator_, expected_output_dtype));
    return Status::OK();
  }

  Status CheckIteratorOutputShapes(
      const std::vector<PartialTensorShape>& expected_output_shapes) {
    TF_RETURN_IF_ERROR(DatasetOpsTestBase::CheckIteratorOutputShapes(
        *iterator_, expected_output_shapes));
    return Status::OK();
  }

  Status CheckIteratorPrefix(const string& expected_iterator_prefix) {
    TF_RETURN_IF_ERROR(DatasetOpsTestBase::CheckIteratorPrefix(
        *iterator_, expected_iterator_prefix));
    return Status::OK();
  }

  Status CheckIteratorSaveAndRestore(
      const string& iterator_prefix,
      const std::vector<Tensor>& expected_outputs,
      const std::vector<int>& breakpoints) {
    TF_RETURN_IF_ERROR(DatasetOpsTestBase::CheckIteratorSaveAndRestore(
        *batch_dataset_, iterator_ctx_.get(), iterator_prefix, expected_outputs,
        breakpoints));
    return Status::OK();
  }

 protected:
  // Creates a new `BatchDataset` op kernel.
  Status CreateBatchDatasetOpKernel(
      const BatchDatasetParams& dataset_params,
      std::unique_ptr<OpKernel>* batch_dataset_op_kernel) {
    name_utils::OpNameParams params;
    params.op_version = kOpVersion;
    NodeDef node_def = test::function::NDef(
        kNodeName, name_utils::OpName(BatchDatasetOp::kDatasetType, params),
        {BatchDatasetOp::kInputDataset, BatchDatasetOp::kBatchSize,
         BatchDatasetOp::kDropRemainder},
        {{BatchDatasetOp::kParallelCopy, dataset_params.parallel_copy},
         {BatchDatasetOp::kOutputTypes, dataset_params.output_dtypes},
         {BatchDatasetOp::kOutputShapes, dataset_params.output_shapes}});
    TF_RETURN_IF_ERROR(CreateOpKernel(node_def, batch_dataset_op_kernel));
    return Status::OK();
  }

  // Create a new `BatchDataset` op kernel context
  Status CreateBatchDatasetContext(
      OpKernel* const op_kernel,
      gtl::InlinedVector<TensorValue, 4>* const inputs,
      std::unique_ptr<OpKernelContext>* context) {
    TF_RETURN_IF_ERROR(CheckOpKernelInput(*op_kernel, *inputs));
    TF_RETURN_IF_ERROR(CreateOpKernelContext(op_kernel, inputs, context));
    return Status::OK();
  }

 private:
  BatchDatasetParams batch_dataset_params_;
  std::unique_ptr<OpKernel> batch_dataset_kernel_;
  std::unique_ptr<OpKernelContext> batch_dataset_context_;
  std::unique_ptr<DatasetBase> batch_dataset_;
  std::unique_ptr<IteratorContext> iterator_ctx_;
  std::unique_ptr<IteratorBase> iterator_;
};

// Test Case 1: test BatchDatasetV2 with `drop_remainder` = false and a batch
// size that can evenly split the input dataset.
BatchDatasetParams BatchDataset1() {
  return {/*num_input_elements=*/12,
          /*batch_size=*/4,
          /*drop_remainder=*/false,
          /*parallel_copy=*/true,
          /*output_dtypes=*/{DT_INT64},
          /*output_shapes=*/{PartialTensorShape({4})},
          /*node_name=*/"batch_dataset"};
}

// Test Case 2: test BatchDatasetV2 with `drop_remainder` = true and a batch
// size that can evenly split the input dataset.
BatchDatasetParams BatchDataset2() {
  return {/*num_input_elements=*/12,
          /*batch_size=*/4,
          /*drop_remainder=*/true,
          /*parallel_copy=*/false,
          /*output_dtypes=*/{DT_INT64},
          /*output_shapes=*/{PartialTensorShape({4})},
          /*node_name=*/"batch_dataset"};
}

// Test Case 3: test BatchDatasetV2 with `drop_remainder` = false and a batch
// size that can not evenly split the input dataset.
BatchDatasetParams BatchDataset3() {
  return {/*num_input_elements=*/10,
          /*batch_size=*/3,
          /*drop_remainder=*/false,
          /*parallel_copy=*/false,
          /*output_dtypes=*/{DT_INT64},
          /*output_shapes=*/{PartialTensorShape({-1})},
          /*node_name=*/"batch_dataset"};
}

// Test Case 4: test BatchDatasetV2 with `drop_remainder` = true and a batch
// size that can not evenly split the input dataset.
BatchDatasetParams BatchDataset4() {
  return {/*num_input_elements=*/10,
          /*batch_size=*/3,
          /*drop_remainder=*/true,
          /*parallel_copy=*/true,
          /*output_dtypes=*/{DT_INT64},
          /*output_shapes=*/{PartialTensorShape({3})},
          /*node_name=*/"batch_dataset"};
}

// Test Case 5: test BatchDatasetV2 with `drop_remainder` = true and
// `batch_size` > the cardinality of the input dataset.
BatchDatasetParams BatchDataset5() {
  return {/*num_input_elements=*/10,
          /*batch_size=*/12,
          /*drop_remainder=*/true,
          /*parallel_copy=*/true,
          /*output_dtypes=*/{DT_INT64},
          /*output_shapes=*/{PartialTensorShape({12})},
          /*node_name=*/"batch_dataset"};
}

// Test Case 6: test BatchDatasetV2 with `drop_remainder` = false and
// `batch_size` > the cardinality of the input dataset.
BatchDatasetParams BatchDataset6() {
  return {/*num_input_elements=*/10,
          /*batch_size=*/12,
          /*drop_remainder=*/false,
          /*parallel_copy=*/true,
          /*output_dtypes=*/{DT_INT64},
          /*output_shapes=*/{PartialTensorShape({-1})},
          /*node_name=*/"batch_dataset"};
}

// Test Case 7: test BatchDatasetV2 with `drop_remainder` = false and
// the output of the input dataset is empty.
BatchDatasetParams BatchDataset7() {
  return {/*num_input_elements=*/0,
          /*batch_size=*/4,
          /*drop_remainder=*/false,
          /*parallel_copy=*/false,
          /*output_dtypes=*/{DT_INT64},
          /*output_shapes=*/{PartialTensorShape({4})},
          /*node_name=*/"batch_dataset"};
}

// Test Case 8: test BatchDatasetV2 with an invalid batch size
BatchDatasetParams InvalidBatchSizeBatchDataset() {
  return {/*num_input_elements=*/10,
          /*batch_size=*/-1,
          /*drop_remainder=*/false,
          /*parallel_copy=*/false,
          /*output_dtypes=*/{DT_INT64},
          /*output_shapes=*/{PartialTensorShape({3})},
          /*node_name=*/"batch_dataset"};
}

class ParameterizedGetNextTest : public BatchDatasetOpTest,
                                 public ::testing::WithParamInterface<
                                     GetNextTestCase<BatchDatasetParams>> {};

std::vector<GetNextTestCase<BatchDatasetParams>> GetNextTestCases() {
  return {{/*dataset_params=*/BatchDataset1(),
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({4}),
                                {{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}})},
          {/*dataset_params=*/BatchDataset2(),
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({4}),
                                {{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}})},
          {/*dataset_params=*/BatchDataset3(),
           /*expected_outputs=*/
           {CreateTensor<int64>(TensorShape({3}), {0, 1, 2}),
            CreateTensor<int64>(TensorShape({3}), {3, 4, 5}),
            CreateTensor<int64>(TensorShape({3}), {6, 7, 8}),
            CreateTensor<int64>(TensorShape({1}), {9})}},
          {/*dataset_params=*/BatchDataset4(),
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({3}),
                                {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}})},
          {/*dataset_params=*/BatchDataset5(),
           /*expected_outputs=*/{}},
          {/*dataset_params=*/BatchDataset6(),
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({10}),
                                {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}})},

          {/*dataset_params=*/BatchDataset7(),
           /*expected_outputs=*/{}}};
}

TEST_P(ParameterizedGetNextTest, GetNext) {
  auto test_case = GetParam();
  TF_ASSERT_OK(SetupTestEnv(/*thread_num=*/2, /*cpu_num=*/2, {}));
  TF_ASSERT_OK(MakeDatasetAndIterator(test_case.dataset_params, true));
  TF_ASSERT_OK(
      CheckIteratorGetNext(test_case.expected_outputs, /*compare_order=*/true));
}

INSTANTIATE_TEST_SUITE_P(
    BatchDatasetOpTest, ParameterizedGetNextTest,
    ::testing::ValuesIn(
        std::vector<GetNextTestCase<BatchDatasetParams>>(GetNextTestCases())));

DatasetNodeNameTestCase<BatchDatasetParams> DatasetNodeNameTestCase1() {
  return {/*dataset_params=*/BatchDataset1(),
          /*expected_node_name=*/kNodeName};
}

TEST_F(BatchDatasetOpTest, DatasetNodeName) {
  auto test_case = DatasetNodeNameTestCase1();
  TF_ASSERT_OK(SetupTestEnv(/*thread_num=*/2, /*cpu_num=*/2, {}));
  TF_ASSERT_OK(MakeDatasetAndIterator(test_case.dataset_params, false));

  TF_ASSERT_OK(CheckDatasetNodeName(kNodeName));
}

DatasetTypeStringTestCase<BatchDatasetParams> DatasetTypeStringTestCase1() {
  name_utils::OpNameParams params;
  params.op_version = kOpVersion;
  return {/*dataset_params=*/BatchDataset1(),
          /*expected_dataset_type_string=*/
          name_utils::OpName(BatchDatasetOp::kDatasetType, params)};
}

TEST_F(BatchDatasetOpTest, DatasetTypeString) {
  auto test_case = DatasetTypeStringTestCase1();
  TF_ASSERT_OK(SetupTestEnv(/*thread_num=*/2, /*cpu_num=*/2, {}));
  TF_ASSERT_OK(MakeDatasetAndIterator(test_case.dataset_params, false));

  TF_ASSERT_OK(CheckDatasetTypeString(test_case.expected_dataset_type_string));
}

DatasetOutputDtypesTestCase<BatchDatasetParams> DatasetOutputDtypesTestCase1() {
  return {/*dataset_params=*/BatchDataset1(),
          /*expected_output_dtypes=*/{DT_INT64}};
}

TEST_F(BatchDatasetOpTest, DatasetOutputDtypes) {
  auto test_case = DatasetOutputDtypesTestCase1();

  TF_ASSERT_OK(SetupTestEnv(/*thread_num=*/2, /*cpu_num=*/2, {}));
  TF_ASSERT_OK(MakeDatasetAndIterator(test_case.dataset_params, false));

  TF_ASSERT_OK(CheckDatasetOutputDtypes(test_case.expected_output_dtypes));
}

class ParameterizedDatasetOutputShapesTest
    : public BatchDatasetOpTest,
      public ::testing::WithParamInterface<
          DatasetOutputShapesTestCase<BatchDatasetParams>> {};

std::vector<DatasetOutputShapesTestCase<BatchDatasetParams>>
DatasetOutputShapesTestCases() {
  return {{/*dataset_params=*/BatchDataset1(),
           /*expected_output_shapes=*/{PartialTensorShape({4})}},
          {/*dataset_params=*/BatchDataset2(),
           /*expected_output_shapes=*/{PartialTensorShape({4})}},
          {/*dataset_params=*/BatchDataset3(),
           /*expected_output_shapes=*/{PartialTensorShape({-1})}},
          {/*dataset_params=*/BatchDataset4(),
           /*expected_output_shapes=*/{PartialTensorShape({3})}},
          {/*dataset_params=*/BatchDataset5(),
           /*expected_output_shapes=*/{PartialTensorShape({12})}},
          {/*dataset_params=*/BatchDataset6(),
           /*expected_output_shapes=*/{PartialTensorShape({-1})}},
          {/*dataset_params=*/BatchDataset7(),
           /*expected_output_shapes=*/{PartialTensorShape({4})}}};
}

TEST_P(ParameterizedDatasetOutputShapesTest, DatasetOutputShapes) {
  auto test_case = GetParam();
  TF_ASSERT_OK(SetupTestEnv(/*thread_num=*/2, /*cpu_num=*/2, {}));
  TF_ASSERT_OK(MakeDatasetAndIterator(test_case.dataset_params, false));

  TF_ASSERT_OK(CheckDatasetOutputShapes(test_case.expected_output_shapes));
}

INSTANTIATE_TEST_SUITE_P(
    BatchDatasetOpTest, ParameterizedDatasetOutputShapesTest,
    ::testing::ValuesIn(
        std::vector<DatasetOutputShapesTestCase<BatchDatasetParams>>(
            DatasetOutputShapesTestCases())));

class ParameterizedCardinalityTest
    : public BatchDatasetOpTest,
      public ::testing::WithParamInterface<
          CardinalityTestCase<BatchDatasetParams>> {};

std::vector<CardinalityTestCase<BatchDatasetParams>> CardinalityTestCases() {
  return {{/*dataset_params=*/BatchDataset1(), /*expected_cardinality=*/3},
          {/*dataset_params=*/BatchDataset2(), /*expected_cardinality=*/3},
          {/*dataset_params=*/BatchDataset3(), /*expected_cardinality=*/4},
          {/*dataset_params=*/BatchDataset4(), /*expected_cardinality=*/3},
          {/*dataset_params=*/BatchDataset5(), /*expected_cardinality=*/0},
          {/*dataset_params=*/BatchDataset6(), /*expected_cardinality=*/1},
          {/*dataset_params=*/BatchDataset7(), /*expected_cardinality=*/0}};
}

TEST_P(ParameterizedCardinalityTest, Cardinality) {
  auto test_case = GetParam();
  TF_ASSERT_OK(SetupTestEnv(/*thread_num=*/2, /*cpu_num=*/2, {}));
  TF_ASSERT_OK(MakeDatasetAndIterator(test_case.dataset_params, false));
  TF_ASSERT_OK(CheckDatasetCardinality(test_case.expected_cardinality));
}

INSTANTIATE_TEST_SUITE_P(
    BatchDatasetOpTest, ParameterizedCardinalityTest,
    ::testing::ValuesIn(std::vector<CardinalityTestCase<BatchDatasetParams>>(
        CardinalityTestCases())));

DatasetSaveTestCase<BatchDatasetParams> DatasetSaveTestCase1() {
  return {/*dataset_params=*/BatchDataset1()};
}

TEST_F(BatchDatasetOpTest, DatasetSave) {
  auto test_case = DatasetSaveTestCase1();
  TF_ASSERT_OK(SetupTestEnv(/*thread_num=*/2, /*cpu_num=*/2, {}));
  TF_ASSERT_OK(MakeDatasetAndIterator(test_case.dataset_params, false));

  TF_ASSERT_OK(CheckDatasetSave());
}

IsStatefulTestCase<BatchDatasetParams> IsStatefulTestCase1() {
  return {/*dataset_params=*/BatchDataset1(),
          /*expected_stateful=*/false};
}

TEST_F(BatchDatasetOpTest, IsStateful) {
  auto test_case = IsStatefulTestCase1();
  TF_ASSERT_OK(SetupTestEnv(/*thread_num=*/2, /*cpu_num=*/2, {}));
  TF_ASSERT_OK(MakeDatasetAndIterator(test_case.dataset_params, false));

  TF_ASSERT_OK(CheckDatasetIsStateful(test_case.expected_stateful));
}

IteratorOutputDtypesTestCase<BatchDatasetParams>
IteratorOutputDtypesTestCase1() {
  return {/*dataset_params=*/BatchDataset1(),
          /*expected_output_dtypes=*/{DT_INT64}};
}

TEST_F(BatchDatasetOpTest, IteratorOutputDtypes) {
  auto test_case = IteratorOutputDtypesTestCase1();
  TF_ASSERT_OK(SetupTestEnv(/*thread_num=*/2, /*cpu_num=*/2, {}));
  TF_ASSERT_OK(MakeDatasetAndIterator(test_case.dataset_params, true));

  TF_ASSERT_OK(CheckIteratorOutputDtypes(test_case.expected_output_dtypes));
}

class ParameterizedIteratorOutputShapesTest
    : public BatchDatasetOpTest,
      public ::testing::WithParamInterface<
          IteratorOutputShapesTestCase<BatchDatasetParams>> {};

std::vector<IteratorOutputShapesTestCase<BatchDatasetParams>>
IteratorOutputShapesTestCases() {
  return {{/*dataset_params=*/BatchDataset1(),
           /*expected_output_shapes=*/{PartialTensorShape({4})}},
          {/*dataset_params=*/BatchDataset2(),
           /*expected_output_shapes=*/{PartialTensorShape({4})}},
          {/*dataset_params=*/BatchDataset3(),
           /*expected_output_shapes=*/{PartialTensorShape({-1})}},
          {/*dataset_params=*/BatchDataset4(),
           /*expected_output_shapes=*/{PartialTensorShape({3})}},
          {/*dataset_params=*/BatchDataset5(),
           /*expected_output_shapes=*/{PartialTensorShape({12})}},
          {/*dataset_params=*/BatchDataset6(),
           /*expected_output_shapes=*/{PartialTensorShape({-1})}},
          {/*dataset_params=*/BatchDataset7(),
           /*expected_output_shapes=*/{PartialTensorShape({4})}}};
}

TEST_P(ParameterizedIteratorOutputShapesTest, IteratorOutputShapes) {
  auto test_case = GetParam();
  TF_ASSERT_OK(SetupTestEnv(/*thread_num=*/2, /*cpu_num=*/2, {}));
  TF_ASSERT_OK(MakeDatasetAndIterator(test_case.dataset_params, true));

  TF_ASSERT_OK(CheckIteratorOutputShapes(test_case.expected_output_shapes));
}

INSTANTIATE_TEST_SUITE_P(
    BatchDatasetOpTest, ParameterizedIteratorOutputShapesTest,
    ::testing::ValuesIn(
        std::vector<IteratorOutputShapesTestCase<BatchDatasetParams>>(
            IteratorOutputShapesTestCases())));

IteratorPrefixTestCase<BatchDatasetParams> IteratorOutputPrefixTestCase1() {
  name_utils::IteratorPrefixParams params;
  params.op_version = kOpVersion;
  return {/*dataset_params=*/BatchDataset1(),
          /*expected_iterator_prefix=*/name_utils::IteratorPrefix(
              BatchDatasetOp::kDatasetType, kIteratorPrefix, params)};
}

TEST_F(BatchDatasetOpTest, IteratorOutputPrefix) {
  auto test_case = IteratorOutputPrefixTestCase1();

  TF_ASSERT_OK(SetupTestEnv(/*thread_num=*/2, /*cpu_num=*/2, {}));
  TF_ASSERT_OK(MakeDatasetAndIterator(test_case.dataset_params, true));
  TF_ASSERT_OK(CheckIteratorPrefix(test_case.expected_iterator_prefix));
}

class ParameterizedIteratorSaveAndRestoreTest
    : public BatchDatasetOpTest,
      public ::testing::WithParamInterface<
          IteratorSaveAndRestoreTestCase<BatchDatasetParams>> {};

std::vector<IteratorSaveAndRestoreTestCase<BatchDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {{/*dataset_params=*/BatchDataset1(),
           /*breakpoints=*/{0, 1, 5},
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({4}),
                                {{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}})},
          {/*dataset_params=*/BatchDataset2(),
           /*breakpoints=*/{0, 1, 5},
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({4}),
                                {{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}})},
          {/*dataset_params=*/BatchDataset3(),
           /*breakpoints=*/{0, 1, 5},
           /*expected_outputs=*/
           {CreateTensor<int64>(TensorShape({3}), {0, 1, 2}),
            CreateTensor<int64>(TensorShape({3}), {3, 4, 5}),
            CreateTensor<int64>(TensorShape({3}), {6, 7, 8}),
            CreateTensor<int64>(TensorShape({1}), {9})}},
          {/*dataset_params=*/BatchDataset4(),
           /*breakpoints=*/{0, 1, 5},
           /*expected_outputs=*/
           {CreateTensor<int64>(TensorShape({3}), {0, 1, 2}),
            CreateTensor<int64>(TensorShape({3}), {3, 4, 5}),
            CreateTensor<int64>(TensorShape({3}), {6, 7, 8})}},
          {/*dataset_params=*/BatchDataset5(),
           /*breakpoints=*/{0, 1, 5},
           /*expected_outputs=*/{}},
          {/*dataset_params=*/BatchDataset6(),
           /*breakpoints=*/{0, 1, 5},
           /*expected_outputs=*/
           {CreateTensor<int64>(TensorShape({10}),
                                {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})}},
          {/*dataset_params=*/BatchDataset7(),
           /*breakpoints=*/{0, 1, 5},
           /*expected_outputs=*/{}}};
}

TEST_P(ParameterizedIteratorSaveAndRestoreTest, IteratorSaveAndRestore) {
  auto test_case = GetParam();

  TF_ASSERT_OK(SetupTestEnv(/*thread_num=*/2, /*cpu_num=*/2, {}));
  TF_ASSERT_OK(MakeDatasetAndIterator(test_case.dataset_params, true));

  TF_ASSERT_OK(CheckIteratorSaveAndRestore(
      kIteratorPrefix, test_case.expected_outputs, test_case.breakpoints));
}

INSTANTIATE_TEST_SUITE_P(
    BatchDatasetOpTest, ParameterizedIteratorSaveAndRestoreTest,
    ::testing::ValuesIn(
        std::vector<IteratorSaveAndRestoreTestCase<BatchDatasetParams>>(
            IteratorSaveAndRestoreTestCases())));

GetNextTestCase<BatchDatasetParams> InvalidBatchSizeTestCase() {
  return {/*dataset_params=*/InvalidBatchSizeBatchDataset(),
          /*expected_outputs=*/{}};
}

TEST_F(BatchDatasetOpTest, InvalidBatchSize) {
  auto test_case = InvalidBatchSizeTestCase();
  TF_ASSERT_OK(SetupTestEnv(/*thread_num=*/2, /*cpu_num=*/2, {}));
  EXPECT_EQ(MakeDatasetAndIterator(test_case.dataset_params, true).code(),
            tensorflow::error::INVALID_ARGUMENT);
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
