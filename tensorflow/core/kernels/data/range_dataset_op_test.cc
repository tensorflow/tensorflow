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
#include "tensorflow/core/kernels/data/range_dataset_op.h"

#include "tensorflow/core/kernels/data/dataset_test_base.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "range_dataset";
constexpr char kIteratorPrefix[] = "Iterator";

class RangeDatasetOpTest : public DatasetOpsTestBase {
 protected:
  // Creates a new RangeDataset op kernel context.
  Status CreateRangeDatasetContext(
      OpKernel* const range_kernel,
      gtl::InlinedVector<TensorValue, 4>* const inputs,
      std::unique_ptr<OpKernelContext>* range_context) {
    TF_RETURN_IF_ERROR(CheckOpKernelInput(*range_kernel, *inputs));
    TF_RETURN_IF_ERROR(
        CreateOpKernelContext(range_kernel, inputs, range_context));
    return Status::OK();
  }
};

class RangeDatasetParams : public DatasetParams {
 public:
  RangeDatasetParams(int64 start, int64 stop, int64 step,
                     DataTypeVector output_dtypes,
                     std::vector<PartialTensorShape> output_shapes,
                     string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        start(CreateTensor<int64>(TensorShape({}), {start})),
        stop(CreateTensor<int64>(TensorShape({}), {stop})),
        step(CreateTensor<int64>(TensorShape({}), {step})) {}

  Status MakeInputs(gtl::InlinedVector<TensorValue, 4>* inputs) override {
    *inputs = {TensorValue(&start), TensorValue(&stop), TensorValue(&step)};
    return Status::OK();
  }

  Tensor start;
  Tensor stop;
  Tensor step;
};

RangeDatasetParams PositiveStepRangeDataset() {
  return {/*start=*/0,
          /*stop=*/10,
          /*step=*/3,
          /*output_dtypes=*/{DT_INT64},
          /*output_shapes=*/{PartialTensorShape({})},
          /*node_name=*/kNodeName};
}

RangeDatasetParams NegativeStepRangeDataset() {
  return {/*start=*/10,
          /*stop=*/0,
          /*step=*/-3,
          /*output_dtypes=*/{DT_INT64},
          /*output_shapes=*/{PartialTensorShape({})},
          /*node_name=*/kNodeName};
}

RangeDatasetParams ZeroStepRangeDataset() {
  return {/*start=*/10,
          /*stop=*/0,
          /*step=*/0,
          /*output_dtypes=*/{DT_INT64},
          /*output_shapes=*/{PartialTensorShape({})},
          /*node_name=*/kNodeName};
}

class ParameterizedGetNextRangeDatasetOpTest
    : public RangeDatasetOpTest,
      public ::testing::WithParamInterface<
          GetNextTestCase<RangeDatasetParams>> {};

GetNextTestCase<RangeDatasetParams> GetNextTestCase1() {
  return {/*dataset_params=*/PositiveStepRangeDataset(),
          /*expected_outputs=*/
          CreateTensors<int64>(TensorShape({}), {{0}, {3}, {6}, {9}})};
}

GetNextTestCase<RangeDatasetParams> GetNextTestCase2() {
  return {/*dataset_params=*/NegativeStepRangeDataset(),
          /*expected_outputs=*/
          CreateTensors<int64>(TensorShape({}), {{10}, {7}, {4}, {1}})};
}

TEST_P(ParameterizedGetNextRangeDatasetOpTest, GetNext) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  GetNextTestCase<RangeDatasetParams> test_case = GetParam();
  gtl::InlinedVector<TensorValue, 4> inputs;
  TF_ASSERT_OK(test_case.dataset_params.MakeInputs(&inputs));
  std::unique_ptr<OpKernel> range_dataset_kernel;
  TF_ASSERT_OK(CreateRangeDatasetOpKernel<int64>(
      test_case.dataset_params.node_name, &range_dataset_kernel));
  std::unique_ptr<OpKernelContext> range_dataset_context;
  TF_ASSERT_OK(CreateRangeDatasetContext(range_dataset_kernel.get(), &inputs,
                                         &range_dataset_context));
  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateDataset(range_dataset_kernel.get(),
                             range_dataset_context.get(), &range_dataset));
  core::ScopedUnref scoped_unref(range_dataset);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(
      CreateIteratorContext(range_dataset_context.get(), &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(range_dataset->MakeIterator(iterator_context.get(),
                                           kIteratorPrefix, &iterator));

  TF_ASSERT_OK(CheckIteratorGetNext(iterator.get(), iterator_context.get(),
                                    test_case.expected_outputs,
                                    /*compare_order=*/true));
}

INSTANTIATE_TEST_SUITE_P(
    RangeDatasetOpTest, ParameterizedGetNextRangeDatasetOpTest,
    ::testing::ValuesIn(std::vector<GetNextTestCase<RangeDatasetParams>>(
        {GetNextTestCase1(), GetNextTestCase2()})));

DatasetNodeNameTestCase<RangeDatasetParams> DatasetNodeNameTestCase1() {
  return {/*dataset_params=*/PositiveStepRangeDataset(),
          /*expected_node_name=*/kNodeName};
}

TEST_F(RangeDatasetOpTest, DatasetNodeName) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  auto test_case = DatasetNodeNameTestCase1();
  gtl::InlinedVector<TensorValue, 4> inputs;
  TF_ASSERT_OK(test_case.dataset_params.MakeInputs(&inputs));
  std::unique_ptr<OpKernel> range_dataset_kernel;
  TF_ASSERT_OK(CreateRangeDatasetOpKernel<int64>(
      test_case.dataset_params.node_name, &range_dataset_kernel));
  std::unique_ptr<OpKernelContext> range_dataset_context;
  TF_ASSERT_OK(CreateRangeDatasetContext(range_dataset_kernel.get(), &inputs,
                                         &range_dataset_context));
  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateDataset(range_dataset_kernel.get(),
                             range_dataset_context.get(), &range_dataset));
  core::ScopedUnref scoped_unref(range_dataset);

  TF_ASSERT_OK(
      CheckDatasetNodeName(*range_dataset, test_case.expected_node_name));
}

DatasetTypeStringTestCase<RangeDatasetParams> DatasetTypeStringTestCase1() {
  return {/*dataset_params=*/PositiveStepRangeDataset(),
          /*expected_dataset_type_string=*/
          name_utils::OpName(RangeDatasetOp::kDatasetType)};
}

TEST_F(RangeDatasetOpTest, DatasetTypeString) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  auto test_case = DatasetTypeStringTestCase1();
  gtl::InlinedVector<TensorValue, 4> inputs;
  TF_ASSERT_OK(test_case.dataset_params.MakeInputs(&inputs));
  std::unique_ptr<OpKernel> range_dataset_kernel;
  TF_ASSERT_OK(CreateRangeDatasetOpKernel<int64>(
      test_case.dataset_params.node_name, &range_dataset_kernel));
  std::unique_ptr<OpKernelContext> range_dataset_context;
  TF_ASSERT_OK(CreateRangeDatasetContext(range_dataset_kernel.get(), &inputs,
                                         &range_dataset_context));
  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateDataset(range_dataset_kernel.get(),
                             range_dataset_context.get(), &range_dataset));
  core::ScopedUnref scoped_unref(range_dataset);

  TF_ASSERT_OK(CheckDatasetTypeString(*range_dataset,
                                      test_case.expected_dataset_type_string));
}

DatasetOutputDtypesTestCase<RangeDatasetParams> DatasetOutputDtypesTestCase1() {
  return {/*dataset_params=*/PositiveStepRangeDataset(),
          /*expected_output_dtypes=*/{DT_INT64}};
}

TEST_F(RangeDatasetOpTest, DatasetOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  auto test_case = DatasetOutputDtypesTestCase1();
  gtl::InlinedVector<TensorValue, 4> inputs;
  TF_ASSERT_OK(test_case.dataset_params.MakeInputs(&inputs));
  std::unique_ptr<OpKernel> range_dataset_kernel;
  TF_ASSERT_OK(CreateRangeDatasetOpKernel<int64>(
      test_case.dataset_params.node_name, &range_dataset_kernel));
  std::unique_ptr<OpKernelContext> range_dataset_context;
  TF_ASSERT_OK(CreateRangeDatasetContext(range_dataset_kernel.get(), &inputs,
                                         &range_dataset_context));
  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateDataset(range_dataset_kernel.get(),
                             range_dataset_context.get(), &range_dataset));
  core::ScopedUnref scoped_unref(range_dataset);

  TF_ASSERT_OK(CheckDatasetOutputDtypes(*range_dataset,
                                        test_case.expected_output_dtypes));
}

DatasetOutputShapesTestCase<RangeDatasetParams> DatasetOutputShapesTestCase1() {
  return {/*dataset_params=*/PositiveStepRangeDataset(),
          /*expected_output_shapes=*/{PartialTensorShape({})}};
}

TEST_F(RangeDatasetOpTest, DatasetOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  auto test_case = DatasetOutputShapesTestCase1();
  gtl::InlinedVector<TensorValue, 4> inputs;
  TF_ASSERT_OK(test_case.dataset_params.MakeInputs(&inputs));
  std::unique_ptr<OpKernel> range_dataset_kernel;
  TF_ASSERT_OK(CreateRangeDatasetOpKernel<int64>(
      test_case.dataset_params.node_name, &range_dataset_kernel));
  std::unique_ptr<OpKernelContext> range_dataset_context;
  TF_ASSERT_OK(CreateRangeDatasetContext(range_dataset_kernel.get(), &inputs,
                                         &range_dataset_context));
  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateDataset(range_dataset_kernel.get(),
                             range_dataset_context.get(), &range_dataset));
  core::ScopedUnref scoped_unref(range_dataset);

  TF_ASSERT_OK(CheckDatasetOutputShapes(*range_dataset,
                                        test_case.expected_output_shapes));
}

class ParameterizedCardinalityRangeDatasetOpTest
    : public RangeDatasetOpTest,
      public ::testing::WithParamInterface<
          CardinalityTestCase<RangeDatasetParams>> {};

CardinalityTestCase<RangeDatasetParams> CardinalityTestCase1() {
  return {/*dataset_params=*/PositiveStepRangeDataset(),
          /*expected_cardinality=*/4};
}

CardinalityTestCase<RangeDatasetParams> CardinalityTestCase2() {
  return {/*dataset_params=*/NegativeStepRangeDataset(),
          /*expected_cardinality=*/4};
}

TEST_P(ParameterizedCardinalityRangeDatasetOpTest, Cardinality) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  auto test_case = GetParam();
  gtl::InlinedVector<TensorValue, 4> inputs;
  TF_ASSERT_OK(test_case.dataset_params.MakeInputs(&inputs));
  std::unique_ptr<OpKernel> range_dataset_kernel;
  TF_ASSERT_OK(CreateRangeDatasetOpKernel<int64>(
      test_case.dataset_params.node_name, &range_dataset_kernel));
  std::unique_ptr<OpKernelContext> range_dataset_context;
  TF_ASSERT_OK(CreateRangeDatasetContext(range_dataset_kernel.get(), &inputs,
                                         &range_dataset_context));
  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateDataset(range_dataset_kernel.get(),
                             range_dataset_context.get(), &range_dataset));
  core::ScopedUnref scoped_unref(range_dataset);

  TF_ASSERT_OK(
      CheckDatasetCardinality(*range_dataset, test_case.expected_cardinality));
}

INSTANTIATE_TEST_SUITE_P(
    RangeDatasetOpTest, ParameterizedCardinalityRangeDatasetOpTest,
    ::testing::ValuesIn(std::vector<CardinalityTestCase<RangeDatasetParams>>(
        {CardinalityTestCase1(), CardinalityTestCase2()})));

DatasetSaveTestCase<RangeDatasetParams> DatasetSaveTestCase1() {
  return {/*dataset_params=*/PositiveStepRangeDataset()};
}

TEST_F(RangeDatasetOpTest, DatasetSave) {
  int64 thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  auto test_case = DatasetSaveTestCase1();
  gtl::InlinedVector<TensorValue, 4> inputs;
  TF_ASSERT_OK(test_case.dataset_params.MakeInputs(&inputs));
  std::unique_ptr<OpKernel> range_dataset_kernel;
  TF_ASSERT_OK(CreateRangeDatasetOpKernel<int64>(
      test_case.dataset_params.node_name, &range_dataset_kernel));
  std::unique_ptr<OpKernelContext> range_dataset_context;
  TF_ASSERT_OK(CreateRangeDatasetContext(range_dataset_kernel.get(), &inputs,
                                         &range_dataset_context));
  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateDataset(range_dataset_kernel.get(),
                             range_dataset_context.get(), &range_dataset));
  core::ScopedUnref scoped_unref(range_dataset);

  TF_ASSERT_OK(CheckDatasetSave(*range_dataset));
}

IsStatefulTestCase<RangeDatasetParams> IsStatefulTestCase1() {
  return {/*dataset_params=*/PositiveStepRangeDataset(),
          /*expected_stateful=*/false};
}

TEST_F(RangeDatasetOpTest, IsStateful) {
  int64 thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  auto test_case = IsStatefulTestCase1();
  gtl::InlinedVector<TensorValue, 4> inputs;
  TF_ASSERT_OK(test_case.dataset_params.MakeInputs(&inputs));
  std::unique_ptr<OpKernel> range_dataset_kernel;
  TF_ASSERT_OK(CreateRangeDatasetOpKernel<int64>(
      test_case.dataset_params.node_name, &range_dataset_kernel));
  std::unique_ptr<OpKernelContext> range_dataset_context;
  TF_ASSERT_OK(CreateRangeDatasetContext(range_dataset_kernel.get(), &inputs,
                                         &range_dataset_context));
  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateDataset(range_dataset_kernel.get(),
                             range_dataset_context.get(), &range_dataset));
  core::ScopedUnref scoped_unref(range_dataset);

  TF_ASSERT_OK(
      CheckDatasetIsStateful(*range_dataset, test_case.expected_stateful));
}

IteratorOutputDtypesTestCase<RangeDatasetParams>
IteratorOutputDtypesTestCase1() {
  return {/*dataset_params=*/PositiveStepRangeDataset(),
          /*expected_output_dtypes=*/{DT_INT64}};
}

TEST_F(RangeDatasetOpTest, IteratorOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  auto test_case = IteratorOutputDtypesTestCase1();
  gtl::InlinedVector<TensorValue, 4> inputs;
  TF_ASSERT_OK(test_case.dataset_params.MakeInputs(&inputs));
  std::unique_ptr<OpKernel> range_dataset_kernel;
  TF_ASSERT_OK(CreateRangeDatasetOpKernel<int64>(
      test_case.dataset_params.node_name, &range_dataset_kernel));
  std::unique_ptr<OpKernelContext> range_dataset_context;
  TF_ASSERT_OK(CreateRangeDatasetContext(range_dataset_kernel.get(), &inputs,
                                         &range_dataset_context));
  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateDataset(range_dataset_kernel.get(),
                             range_dataset_context.get(), &range_dataset));
  core::ScopedUnref scoped_unref(range_dataset);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(
      CreateIteratorContext(range_dataset_context.get(), &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(range_dataset->MakeIterator(iterator_context.get(),
                                           kIteratorPrefix, &iterator));
  TF_ASSERT_OK(
      CheckIteratorOutputDtypes(*iterator, test_case.expected_output_dtypes));
}

IteratorOutputShapesTestCase<RangeDatasetParams>
IteratorOutputShapesTestCase1() {
  return {/*dataset_params=*/PositiveStepRangeDataset(),
          /*expected_output_shapes=*/{PartialTensorShape({})}};
}

TEST_F(RangeDatasetOpTest, IteratorOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  auto test_case = IteratorOutputShapesTestCase1();
  gtl::InlinedVector<TensorValue, 4> inputs;
  TF_ASSERT_OK(test_case.dataset_params.MakeInputs(&inputs));
  std::unique_ptr<OpKernel> range_dataset_kernel;
  TF_ASSERT_OK(CreateRangeDatasetOpKernel<int64>(
      test_case.dataset_params.node_name, &range_dataset_kernel));
  std::unique_ptr<OpKernelContext> range_dataset_context;
  TF_ASSERT_OK(CreateRangeDatasetContext(range_dataset_kernel.get(), &inputs,
                                         &range_dataset_context));
  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateDataset(range_dataset_kernel.get(),
                             range_dataset_context.get(), &range_dataset));
  core::ScopedUnref scoped_unref(range_dataset);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(
      CreateIteratorContext(range_dataset_context.get(), &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(range_dataset->MakeIterator(iterator_context.get(),
                                           kIteratorPrefix, &iterator));
  TF_ASSERT_OK(
      CheckIteratorOutputShapes(*iterator, test_case.expected_output_shapes));
}

IteratorOutputPrefixTestCase<RangeDatasetParams>
IteratorOutputPrefixTestCase1() {
  return {/*dataset_params=*/PositiveStepRangeDataset(),
          /*expected_iterator_prefix=*/name_utils::IteratorPrefix(
              RangeDatasetOp::kDatasetType, kIteratorPrefix)};
}

TEST_F(RangeDatasetOpTest, IteratorOutputPrefix) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  auto test_case = IteratorOutputPrefixTestCase1();
  gtl::InlinedVector<TensorValue, 4> inputs;
  TF_ASSERT_OK(test_case.dataset_params.MakeInputs(&inputs));
  std::unique_ptr<OpKernel> range_dataset_kernel;
  TF_ASSERT_OK(CreateRangeDatasetOpKernel<int64>(
      test_case.dataset_params.node_name, &range_dataset_kernel));
  std::unique_ptr<OpKernelContext> range_dataset_context;
  TF_ASSERT_OK(CreateRangeDatasetContext(range_dataset_kernel.get(), &inputs,
                                         &range_dataset_context));
  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateDataset(range_dataset_kernel.get(),
                             range_dataset_context.get(), &range_dataset));
  core::ScopedUnref scoped_unref(range_dataset);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(
      CreateIteratorContext(range_dataset_context.get(), &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(range_dataset->MakeIterator(iterator_context.get(),
                                           kIteratorPrefix, &iterator));
  TF_ASSERT_OK(
      CheckIteratorPrefix(*iterator, test_case.expected_iterator_prefix));
}

class ParameterizedIteratorSaveAndRestoreRangeDatasetOpTest
    : public RangeDatasetOpTest,
      public ::testing::WithParamInterface<
          IteratorSaveAndRestoreTestCase<RangeDatasetParams>> {};

IteratorSaveAndRestoreTestCase<RangeDatasetParams>
IteratorSaveAndRestoreTestCase1() {
  return {/*dataset_params=*/PositiveStepRangeDataset(),
          /*breakpoints=*/{0, 1, 4},
          /*expected_outputs=*/
          CreateTensors<int64>(TensorShape({}), {{0}, {3}, {6}, {9}})};
}

IteratorSaveAndRestoreTestCase<RangeDatasetParams>
IteratorSaveAndRestoreTestCase2() {
  return {/*dataset_params=*/NegativeStepRangeDataset(),
          /*breakpoints=*/{0, 1, 4},
          /*expected_outputs=*/
          CreateTensors<int64>(TensorShape({}), {{10}, {7}, {4}, {1}})};
}

TEST_P(ParameterizedIteratorSaveAndRestoreRangeDatasetOpTest,
       IteratorSaveAndRestore) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  auto test_case = GetParam();
  gtl::InlinedVector<TensorValue, 4> inputs;
  TF_ASSERT_OK(test_case.dataset_params.MakeInputs(&inputs));
  std::unique_ptr<OpKernel> range_dataset_kernel;
  TF_ASSERT_OK(CreateRangeDatasetOpKernel<int64>(
      test_case.dataset_params.node_name, &range_dataset_kernel));
  std::unique_ptr<OpKernelContext> range_dataset_context;
  TF_ASSERT_OK(CreateRangeDatasetContext(range_dataset_kernel.get(), &inputs,
                                         &range_dataset_context));
  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateDataset(range_dataset_kernel.get(),
                             range_dataset_context.get(), &range_dataset));
  core::ScopedUnref scoped_unref(range_dataset);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(
      CreateIteratorContext(range_dataset_context.get(), &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(range_dataset->MakeIterator(iterator_context.get(),
                                           kIteratorPrefix, &iterator));
  TF_ASSERT_OK(CheckIteratorSaveAndRestore(
      *range_dataset, iterator_context.get(), kIteratorPrefix,
      test_case.expected_outputs, test_case.breakpoints));
}

INSTANTIATE_TEST_SUITE_P(
    RangeDatasetOpTest, ParameterizedIteratorSaveAndRestoreRangeDatasetOpTest,
    ::testing::ValuesIn(
        std::vector<IteratorSaveAndRestoreTestCase<RangeDatasetParams>>(
            {IteratorSaveAndRestoreTestCase1(),
             IteratorSaveAndRestoreTestCase2()})));

GetNextTestCase<RangeDatasetParams> ZeroStepTestCase1() {
  return {/*dataset_params=*/ZeroStepRangeDataset(),
          /*expected_outputs=*/{}};
}

TEST_F(RangeDatasetOpTest, ZeroStep) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  auto test_case = ZeroStepTestCase1();
  gtl::InlinedVector<TensorValue, 4> inputs;
  TF_ASSERT_OK(test_case.dataset_params.MakeInputs(&inputs));
  std::unique_ptr<OpKernel> range_dataset_kernel;
  TF_ASSERT_OK(CreateRangeDatasetOpKernel<int64>(
      test_case.dataset_params.node_name, &range_dataset_kernel));
  std::unique_ptr<OpKernelContext> range_dataset_context;
  TF_ASSERT_OK(CreateRangeDatasetContext(range_dataset_kernel.get(), &inputs,
                                         &range_dataset_context));
  DatasetBase* range_dataset;
  EXPECT_EQ(CreateDataset(range_dataset_kernel.get(),
                          range_dataset_context.get(), &range_dataset)
                .code(),
            tensorflow::error::INVALID_ARGUMENT);
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
