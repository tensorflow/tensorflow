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

#include "tensorflow/core/kernels/data/dataset_test_base.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "window_dataset";
constexpr char kOpName[] = "WindowDataset";

class WindowDatasetOpTest : public DatasetOpsTestBase {
 protected:
  // Creates a new `WindowDataset` op kernel
  Status CreateWindowDatasetKernel(
      const DataTypeVector& output_types,
      const std::vector<PartialTensorShape>& output_shapes,
      std::unique_ptr<OpKernel>* op_kernel) {
    NodeDef node_def = test::function::NDef(
        kNodeName, kOpName,
        {"input_dataset", "size", "shift", "stride", "drop_remainder"},
        {{"output_types", output_types}, {"output_shapes", output_shapes}});
    TF_RETURN_IF_ERROR(CreateOpKernel(node_def, op_kernel));
    return Status::OK();
  }

  // Creates a new `WindowDataset` op kernel context.
  Status CreateWindowDatasetContext(
      OpKernel* const op_kernel,
      gtl::InlinedVector<TensorValue, 4>* const inputs,
      std::unique_ptr<OpKernelContext>* context) {
    TF_RETURN_IF_ERROR(CheckOpKernelInput(*op_kernel, *inputs));
    TF_RETURN_IF_ERROR(CreateOpKernelContext(op_kernel, inputs, context));
    return Status::OK();
  }
};

struct RangeDatasetParam {
  int64 start;
  int64 end;
  int64 step;
};

struct TestCase {
  RangeDatasetParam range_data_param;
  Tensor size;
  Tensor shift;
  Tensor stride;
  Tensor drop_remainder;
  std::vector<std::vector<Tensor>> expected_outputs;
  DataTypeVector expected_output_dtypes;
  std::vector<PartialTensorShape> expected_output_shapes;
  int64 expected_cardinality;
  std::vector<int> breakpoints;
};

// Test case 1: size=2, shift=2, stride=1, drop_remainder=false.
TestCase TestCase1() {
  return {
      /*range_data_param*/ {0, 7, 1},
      /*size*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {2}),
      /*shift*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {2}),
      /*stride*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {1}),
      /*drop_remainder*/
      DatasetOpsTestBase::CreateTensor<bool>(TensorShape({}), {false}),
      /*expected_outputs*/
      {{DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {0}),
        DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {1})},
       {DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {2}),
        DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {3})},
       {DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {4}),
        DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {5})},
       {DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {6})}},
      /*expected_output_dtypes*/ {DT_VARIANT},
      /*expected_output_shapes*/ {PartialTensorShape({})},
      /*expected_cardinality*/ 4,
      /*breakpoints*/ {0, 1, 9}};
}

// Test case 2: size=2, shift=2, stride=2, drop_remainder=true.
TestCase TestCase2() {
  return {
      /*range_data_param*/ {0, 7, 1},
      /*size*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {2}),
      /*shift*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {2}),
      /*stride*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {2}),
      /*drop_remainder*/
      DatasetOpsTestBase::CreateTensor<bool>(TensorShape({}), {true}),
      /*expected_outputs*/
      {{DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {0}),
        DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {2})},
       {DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {2}),
        DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {4})},
       {DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {4}),
        DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {6})}},
      /*expected_output_dtypes*/ {DT_VARIANT},
      /*expected_output_shapes*/ {PartialTensorShape({})},
      /*expected_cardinality*/ 3,
      /*breakpoints*/ {0, 1, 9}};
}

// Test case 3: size=8, shift=3, stride=1, drop_remainder=false.
TestCase TestCase3() {
  return {
      /*range_data_param*/ {0, 7, 1},
      /*size*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {8}),
      /*shift*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {3}),
      /*stride*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {1}),
      /*drop_remainder*/
      DatasetOpsTestBase::CreateTensor<bool>(TensorShape({}), {false}),
      /*expected_outputs*/
      {{DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {0}),
        DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {1}),
        DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {2}),
        DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {3}),
        DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {4}),
        DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {5}),
        DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {6})},
       {DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {3}),
        DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {4}),
        DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {5}),
        DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {6})},
       {DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {6})}},
      /*expected_output_dtypes*/ {DT_VARIANT},
      /*expected_output_shapes*/ {PartialTensorShape({})},
      /*expected_cardinality*/ 3,
      /*breakpoints*/ {0, 1, 9}};
}

// Test case 4: size=8, shift=3, stride=1, drop_remainder=true.
TestCase TestCase4() {
  return {
      /*range_data_param*/ {0, 7, 1},
      /*size*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {8}),
      /*shift*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {3}),
      /*stride*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {1}),
      /*drop_remainder*/
      DatasetOpsTestBase::CreateTensor<bool>(TensorShape({}), {true}),
      /*expected_outputs*/ {},
      /*expected_output_dtypes*/ {DT_VARIANT},
      /*expected_output_shapes*/ {PartialTensorShape({})},
      /*expected_cardinality*/ 0,
      /*breakpoints*/ {0, 1, 9}};
}

// Test case 5: size=2, shift=8, stride=1, drop_remainder=false.
TestCase TestCase5() {
  return {
      /*range_data_param*/ {0, 7, 1},
      /*size*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {2}),
      /*shift*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {8}),
      /*stride*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {1}),
      /*drop_remainder*/
      DatasetOpsTestBase::CreateTensor<bool>(TensorShape({}), {false}),
      /*expected_outputs*/
      {{DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {0}),
        DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {1})}},
      /*expected_output_dtypes*/ {DT_VARIANT},
      /*expected_output_shapes*/ {PartialTensorShape({})},
      /*expected_cardinality*/ 1,
      /*breakpoints*/ {0, 1, 9}};
}

// Test case 6: size=2, shift=8, stride=1, drop_remainder=true.
TestCase TestCase6() {
  return {
      /*range_data_param*/ {0, 7, 1},
      /*size*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {2}),
      /*shift*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {8}),
      /*stride*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {1}),
      /*drop_remainder*/
      DatasetOpsTestBase::CreateTensor<bool>(TensorShape({}), {true}),
      /*expected_outputs*/
      {{DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {0}),
        DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {1})}},
      /*expected_output_dtypes*/ {DT_VARIANT},
      /*expected_output_shapes*/ {PartialTensorShape({})},
      /*expected_cardinality*/ 1,
      /*breakpoints*/ {0, 1, 9}};
}

// Test case 7: size=2, shift=2, stride=8, drop_remainder=false.
TestCase TestCase7() {
  return {
      /*range_data_param*/ {0, 7, 1},
      /*size*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {2}),
      /*shift*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {2}),
      /*stride*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {8}),
      /*drop_remainder*/
      DatasetOpsTestBase::CreateTensor<bool>(TensorShape({}), {false}),
      /*expected_outputs*/
      {{DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {0})},
       {DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {2})},
       {DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {4})},
       {DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {6})}},
      /*expected_output_dtypes*/ {DT_VARIANT},
      /*expected_output_shapes*/ {PartialTensorShape({})},
      /*expected_cardinality*/ 4,
      /*breakpoints*/ {0, 1, 9}};
}

// Test case 8: size=2, shift=2, stride=8, drop_remainder=true.
TestCase TestCase8() {
  return {
      /*range_data_param*/ {0, 7, 1},
      /*size*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {2}),
      /*shift*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {2}),
      /*stride*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {8}),
      /*drop_remainder*/
      DatasetOpsTestBase::CreateTensor<bool>(TensorShape({}), {true}),
      /*expected_outputs*/ {},
      /*expected_output_dtypes*/ {DT_VARIANT},
      /*expected_output_shapes*/ {PartialTensorShape({})},
      /*expected_cardinality*/ 0,
      /*breakpoints*/ {0, 1, 9}};
}

// Test case 9: size=4, shift=2, stride=2, drop_remainder=true.
TestCase TestCase9() {
  return {
      /*range_data_param*/ {0, 7, 1},
      /*size*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {4}),
      /*shift*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {2}),
      /*stride*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {2}),
      /*drop_remainder*/
      DatasetOpsTestBase::CreateTensor<bool>(TensorShape({}), {true}),
      /*expected_outputs*/
      {{DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {0}),
        DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {2}),
        DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {4}),
        DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {6})}},
      /*expected_output_dtypes*/ {DT_VARIANT},
      /*expected_output_shapes*/ {PartialTensorShape({})},
      /*expected_cardinality*/ 1,
      /*breakpoints*/ {0, 1, 9}};
}

// Test case 10: size=5, shift=2, stride=2, drop_remainder=true.
TestCase TestCase10() {
  return {
      /*range_data_param*/ {0, 7, 1},
      /*size*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {5}),
      /*shift*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {2}),
      /*stride*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {2}),
      /*drop_remainder*/
      DatasetOpsTestBase::CreateTensor<bool>(TensorShape({}), {true}),
      /*expected_outputs*/ {},
      /*expected_output_dtypes*/ {DT_VARIANT},
      /*expected_output_shapes*/ {PartialTensorShape({})},
      /*expected_cardinality*/ 0,
      /*breakpoints*/ {0, 1, 9}};
}

// Test case 11: size=0, shift=2, stride=2, drop_remainder=true.
TestCase InvalidWindowSizeTestCase() {
  return {
      /*range_data_param*/ {0, 7, 1},
      /*size*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {0}),
      /*shift*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {2}),
      /*stride*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {2}),
      /*drop_remainder*/
      DatasetOpsTestBase::CreateTensor<bool>(TensorShape({}), {true}),
      /*expected_outputs*/ {},
      /*expected_output_dtypes*/ {DT_VARIANT},
      /*expected_output_shapes*/ {PartialTensorShape({})},
      /*expected_cardinality*/ 0,
      /*breakpoints*/ {0, 1, 9}};
}

// Test case 12: size=2, shift=0, stride=2, drop_remainder=true.
TestCase InvalidWindowShiftTestCase() {
  return {
      /*range_data_param*/ {0, 7, 1},
      /*size*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {2}),
      /*shift*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {0}),
      /*stride*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {2}),
      /*drop_remainder*/
      DatasetOpsTestBase::CreateTensor<bool>(TensorShape({}), {true}),
      /*expected_outputs*/ {},
      /*expected_output_dtypes*/ {DT_VARIANT},
      /*expected_output_shapes*/ {PartialTensorShape({})},
      /*expected_cardinality*/ 0,
      /*breakpoints*/ {0, 1, 9}};
}

// Test case 13: size=2, shift=2, stride=0, drop_remainder=true.
TestCase InvalidWindowStrideTestCase() {
  return {
      /*range_data_param*/ {0, 7, 1},
      /*size*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {2}),
      /*shift*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {2}),
      /*stride*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {0}),
      /*drop_remainder*/
      DatasetOpsTestBase::CreateTensor<bool>(TensorShape({}), {true}),
      /*expected_outputs*/ {},
      /*expected_output_dtypes*/ {DT_VARIANT},
      /*expected_output_shapes*/ {PartialTensorShape({})},
      /*expected_cardinality*/ 0,
      /*breakpoints*/ {0, 1, 9}};
}

class ParameterizedWindowDatasetOpTest
    : public WindowDatasetOpTest,
      public ::testing::WithParamInterface<TestCase> {};

TEST_P(ParameterizedWindowDatasetOpTest, GetNext) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> window_dataset_kernel;
  TF_ASSERT_OK(CreateWindowDatasetKernel(test_case.expected_output_dtypes,
                                         test_case.expected_output_shapes,
                                         &window_dataset_kernel));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_data_param.start, test_case.range_data_param.end,
      test_case.range_data_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));
  Tensor size = test_case.size;
  Tensor shift = test_case.shift;
  Tensor stride = test_case.stride;
  Tensor drop_remainder = test_case.drop_remainder;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&range_dataset_tensor), TensorValue(&size),
       TensorValue(&shift), TensorValue(&stride),
       TensorValue(&drop_remainder)});

  std::unique_ptr<OpKernelContext> window_dataset_op_ctx;
  TF_ASSERT_OK(CreateWindowDatasetContext(window_dataset_kernel.get(), &inputs,
                                          &window_dataset_op_ctx));
  DatasetBase* dataset;
  TF_ASSERT_OK(CreateDataset(window_dataset_kernel.get(),
                             window_dataset_op_ctx.get(), &dataset));
  core::ScopedUnref scoped_unref_dataset(dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(window_dataset_op_ctx.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));

  bool end_of_sequence = false;
  auto expected_outputs_it = test_case.expected_outputs.begin();
  while (!end_of_sequence) {
    // Owns the window_datasets, which are stored as the variant tensors in the
    // vector.
    std::vector<Tensor> out_tensors;
    TF_EXPECT_OK(
        iterator->GetNext(iterator_ctx.get(), &out_tensors, &end_of_sequence));
    if (!end_of_sequence) {
      for (const auto& window_dataset_tensor : out_tensors) {
        // Not owned.
        DatasetBase* window_dataset;
        TF_ASSERT_OK(GetDatasetFromVariantTensor(window_dataset_tensor,
                                                 &window_dataset));
        std::unique_ptr<IteratorBase> window_dataset_iterator;
        TF_ASSERT_OK(window_dataset->MakeIterator(
            iterator_ctx.get(), "Iterator", &window_dataset_iterator));
        bool end_of_window_dataset = false;
        std::vector<Tensor> window_elements;
        // Fetches all the elements in window_dataset.
        while (!end_of_window_dataset) {
          std::vector<Tensor> next_element;
          TF_EXPECT_OK(window_dataset_iterator->GetNext(
              iterator_ctx.get(), &next_element, &end_of_window_dataset));
          window_elements.insert(window_elements.end(), next_element.begin(),
                                 next_element.end());
        }
        EXPECT_LT(expected_outputs_it, test_case.expected_outputs.end());
        TF_EXPECT_OK(ExpectEqual(window_elements, *expected_outputs_it, false));
        expected_outputs_it++;
      }
    }
  }
  EXPECT_EQ(expected_outputs_it, test_case.expected_outputs.end());
}

TEST_F(WindowDatasetOpTest, DatasetNodeName) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = TestCase1();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> window_dataset_kernel;
  TF_ASSERT_OK(CreateWindowDatasetKernel(test_case.expected_output_dtypes,
                                         test_case.expected_output_shapes,
                                         &window_dataset_kernel));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_data_param.start, test_case.range_data_param.end,
      test_case.range_data_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));
  Tensor size = test_case.size;
  Tensor shift = test_case.shift;
  Tensor stride = test_case.stride;
  Tensor drop_remainder = test_case.drop_remainder;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&range_dataset_tensor), TensorValue(&size),
       TensorValue(&shift), TensorValue(&stride),
       TensorValue(&drop_remainder)});

  std::unique_ptr<OpKernelContext> window_dataset_op_ctx;
  TF_ASSERT_OK(CreateWindowDatasetContext(window_dataset_kernel.get(), &inputs,
                                          &window_dataset_op_ctx));
  DatasetBase* dataset;
  TF_ASSERT_OK(CreateDataset(window_dataset_kernel.get(),
                             window_dataset_op_ctx.get(), &dataset));
  core::ScopedUnref scoped_unref_dataset(dataset);

  EXPECT_EQ(dataset->node_name(), kNodeName);
}

TEST_F(WindowDatasetOpTest, DatasetTypeString) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = TestCase1();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> window_dataset_kernel;
  TF_ASSERT_OK(CreateWindowDatasetKernel(test_case.expected_output_dtypes,
                                         test_case.expected_output_shapes,
                                         &window_dataset_kernel));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_data_param.start, test_case.range_data_param.end,
      test_case.range_data_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));
  Tensor size = test_case.size;
  Tensor shift = test_case.shift;
  Tensor stride = test_case.stride;
  Tensor drop_remainder = test_case.drop_remainder;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&range_dataset_tensor), TensorValue(&size),
       TensorValue(&shift), TensorValue(&stride),
       TensorValue(&drop_remainder)});

  std::unique_ptr<OpKernelContext> window_dataset_op_ctx;
  TF_ASSERT_OK(CreateWindowDatasetContext(window_dataset_kernel.get(), &inputs,
                                          &window_dataset_op_ctx));
  DatasetBase* dataset;
  TF_ASSERT_OK(CreateDataset(window_dataset_kernel.get(),
                             window_dataset_op_ctx.get(), &dataset));
  core::ScopedUnref scoped_unref_dataset(dataset);

  EXPECT_EQ(dataset->type_string(), kOpName);
}

TEST_P(ParameterizedWindowDatasetOpTest, DatasetOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> window_dataset_kernel;
  TF_ASSERT_OK(CreateWindowDatasetKernel(test_case.expected_output_dtypes,
                                         test_case.expected_output_shapes,
                                         &window_dataset_kernel));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_data_param.start, test_case.range_data_param.end,
      test_case.range_data_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));
  Tensor size = test_case.size;
  Tensor shift = test_case.shift;
  Tensor stride = test_case.stride;
  Tensor drop_remainder = test_case.drop_remainder;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&range_dataset_tensor), TensorValue(&size),
       TensorValue(&shift), TensorValue(&stride),
       TensorValue(&drop_remainder)});

  std::unique_ptr<OpKernelContext> window_dataset_op_ctx;
  TF_ASSERT_OK(CreateWindowDatasetContext(window_dataset_kernel.get(), &inputs,
                                          &window_dataset_op_ctx));
  DatasetBase* dataset;
  TF_ASSERT_OK(CreateDataset(window_dataset_kernel.get(),
                             window_dataset_op_ctx.get(), &dataset));
  core::ScopedUnref scoped_unref_dataset(dataset);

  TF_EXPECT_OK(VerifyTypesMatch(dataset->output_dtypes(),
                                test_case.expected_output_dtypes));
}

TEST_P(ParameterizedWindowDatasetOpTest, DatasetOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> window_dataset_kernel;
  TF_ASSERT_OK(CreateWindowDatasetKernel(test_case.expected_output_dtypes,
                                         test_case.expected_output_shapes,
                                         &window_dataset_kernel));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_data_param.start, test_case.range_data_param.end,
      test_case.range_data_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));
  Tensor size = test_case.size;
  Tensor shift = test_case.shift;
  Tensor stride = test_case.stride;
  Tensor drop_remainder = test_case.drop_remainder;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&range_dataset_tensor), TensorValue(&size),
       TensorValue(&shift), TensorValue(&stride),
       TensorValue(&drop_remainder)});

  std::unique_ptr<OpKernelContext> window_dataset_op_ctx;
  TF_ASSERT_OK(CreateWindowDatasetContext(window_dataset_kernel.get(), &inputs,
                                          &window_dataset_op_ctx));
  DatasetBase* dataset;
  TF_ASSERT_OK(CreateDataset(window_dataset_kernel.get(),
                             window_dataset_op_ctx.get(), &dataset));
  core::ScopedUnref scoped_unref_dataset(dataset);

  TF_EXPECT_OK(VerifyShapesCompatible(dataset->output_shapes(),
                                      test_case.expected_output_shapes));
}

TEST_P(ParameterizedWindowDatasetOpTest, Cardinality) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> window_dataset_kernel;
  TF_ASSERT_OK(CreateWindowDatasetKernel(test_case.expected_output_dtypes,
                                         test_case.expected_output_shapes,
                                         &window_dataset_kernel));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_data_param.start, test_case.range_data_param.end,
      test_case.range_data_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));
  Tensor size = test_case.size;
  Tensor shift = test_case.shift;
  Tensor stride = test_case.stride;
  Tensor drop_remainder = test_case.drop_remainder;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&range_dataset_tensor), TensorValue(&size),
       TensorValue(&shift), TensorValue(&stride),
       TensorValue(&drop_remainder)});

  std::unique_ptr<OpKernelContext> window_dataset_op_ctx;
  TF_ASSERT_OK(CreateWindowDatasetContext(window_dataset_kernel.get(), &inputs,
                                          &window_dataset_op_ctx));
  DatasetBase* dataset;
  TF_ASSERT_OK(CreateDataset(window_dataset_kernel.get(),
                             window_dataset_op_ctx.get(), &dataset));
  core::ScopedUnref scoped_unref_dataset(dataset);

  EXPECT_EQ(dataset->Cardinality(), test_case.expected_cardinality);
}

TEST_P(ParameterizedWindowDatasetOpTest, DatasetSave) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> window_dataset_kernel;
  TF_ASSERT_OK(CreateWindowDatasetKernel(test_case.expected_output_dtypes,
                                         test_case.expected_output_shapes,
                                         &window_dataset_kernel));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_data_param.start, test_case.range_data_param.end,
      test_case.range_data_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));
  Tensor size = test_case.size;
  Tensor shift = test_case.shift;
  Tensor stride = test_case.stride;
  Tensor drop_remainder = test_case.drop_remainder;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&range_dataset_tensor), TensorValue(&size),
       TensorValue(&shift), TensorValue(&stride),
       TensorValue(&drop_remainder)});

  std::unique_ptr<OpKernelContext> window_dataset_op_ctx;
  TF_ASSERT_OK(CreateWindowDatasetContext(window_dataset_kernel.get(), &inputs,
                                          &window_dataset_op_ctx));
  DatasetBase* dataset;
  TF_ASSERT_OK(CreateDataset(window_dataset_kernel.get(),
                             window_dataset_op_ctx.get(), &dataset));
  core::ScopedUnref scoped_unref_dataset(dataset);

  std::unique_ptr<SerializationContext> serialization_context;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_context));
  VariantTensorData data;
  VariantTensorDataWriter writer(&data);
  TF_ASSERT_OK(dataset->Save(serialization_context.get(), &writer));
  TF_ASSERT_OK(writer.Flush());
}

TEST_P(ParameterizedWindowDatasetOpTest, IteratorOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> window_dataset_kernel;
  TF_ASSERT_OK(CreateWindowDatasetKernel(test_case.expected_output_dtypes,
                                         test_case.expected_output_shapes,
                                         &window_dataset_kernel));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_data_param.start, test_case.range_data_param.end,
      test_case.range_data_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));
  Tensor size = test_case.size;
  Tensor shift = test_case.shift;
  Tensor stride = test_case.stride;
  Tensor drop_remainder = test_case.drop_remainder;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&range_dataset_tensor), TensorValue(&size),
       TensorValue(&shift), TensorValue(&stride),
       TensorValue(&drop_remainder)});

  std::unique_ptr<OpKernelContext> window_dataset_op_ctx;
  TF_ASSERT_OK(CreateWindowDatasetContext(window_dataset_kernel.get(), &inputs,
                                          &window_dataset_op_ctx));
  DatasetBase* dataset;
  TF_ASSERT_OK(CreateDataset(window_dataset_kernel.get(),
                             window_dataset_op_ctx.get(), &dataset));
  core::ScopedUnref scoped_unref_dataset(dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(window_dataset_op_ctx.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));

  TF_EXPECT_OK(VerifyTypesMatch(iterator->output_dtypes(),
                                test_case.expected_output_dtypes));
}

TEST_P(ParameterizedWindowDatasetOpTest, IteratorOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> window_dataset_kernel;
  TF_ASSERT_OK(CreateWindowDatasetKernel(test_case.expected_output_dtypes,
                                         test_case.expected_output_shapes,
                                         &window_dataset_kernel));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_data_param.start, test_case.range_data_param.end,
      test_case.range_data_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));
  Tensor size = test_case.size;
  Tensor shift = test_case.shift;
  Tensor stride = test_case.stride;
  Tensor drop_remainder = test_case.drop_remainder;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&range_dataset_tensor), TensorValue(&size),
       TensorValue(&shift), TensorValue(&stride),
       TensorValue(&drop_remainder)});

  std::unique_ptr<OpKernelContext> window_dataset_op_ctx;
  TF_ASSERT_OK(CreateWindowDatasetContext(window_dataset_kernel.get(), &inputs,
                                          &window_dataset_op_ctx));
  DatasetBase* dataset;
  TF_ASSERT_OK(CreateDataset(window_dataset_kernel.get(),
                             window_dataset_op_ctx.get(), &dataset));
  core::ScopedUnref scoped_unref_dataset(dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(window_dataset_op_ctx.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));

  TF_EXPECT_OK(VerifyShapesCompatible(iterator->output_shapes(),
                                      test_case.expected_output_shapes));
}

TEST_F(WindowDatasetOpTest, IteratorOutputPrefix) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = TestCase1();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> window_dataset_kernel;
  TF_ASSERT_OK(CreateWindowDatasetKernel(test_case.expected_output_dtypes,
                                         test_case.expected_output_shapes,
                                         &window_dataset_kernel));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_data_param.start, test_case.range_data_param.end,
      test_case.range_data_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));
  Tensor size = test_case.size;
  Tensor shift = test_case.shift;
  Tensor stride = test_case.stride;
  Tensor drop_remainder = test_case.drop_remainder;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&range_dataset_tensor), TensorValue(&size),
       TensorValue(&shift), TensorValue(&stride),
       TensorValue(&drop_remainder)});

  std::unique_ptr<OpKernelContext> window_dataset_op_ctx;
  TF_ASSERT_OK(CreateWindowDatasetContext(window_dataset_kernel.get(), &inputs,
                                          &window_dataset_op_ctx));
  DatasetBase* dataset;
  TF_ASSERT_OK(CreateDataset(window_dataset_kernel.get(),
                             window_dataset_op_ctx.get(), &dataset));
  core::ScopedUnref scoped_unref_dataset(dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(window_dataset_op_ctx.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));

  EXPECT_EQ(iterator->prefix(), "Iterator::Window");
}

TEST_P(ParameterizedWindowDatasetOpTest, Roundtrip) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> window_dataset_kernel;
  TF_ASSERT_OK(CreateWindowDatasetKernel(test_case.expected_output_dtypes,
                                         test_case.expected_output_shapes,
                                         &window_dataset_kernel));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_data_param.start, test_case.range_data_param.end,
      test_case.range_data_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));
  Tensor size = test_case.size;
  Tensor shift = test_case.shift;
  Tensor stride = test_case.stride;
  Tensor drop_remainder = test_case.drop_remainder;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&range_dataset_tensor), TensorValue(&size),
       TensorValue(&shift), TensorValue(&stride),
       TensorValue(&drop_remainder)});

  std::unique_ptr<OpKernelContext> window_dataset_op_ctx;
  TF_ASSERT_OK(CreateWindowDatasetContext(window_dataset_kernel.get(), &inputs,
                                          &window_dataset_op_ctx));
  DatasetBase* dataset;
  TF_ASSERT_OK(CreateDataset(window_dataset_kernel.get(),
                             window_dataset_op_ctx.get(), &dataset));
  core::ScopedUnref scoped_unref_dataset(dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(window_dataset_op_ctx.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));

  std::unique_ptr<SerializationContext> serialization_ctx;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_ctx));

  bool end_of_sequence = false;
  auto expected_outputs_it = test_case.expected_outputs.begin();
  int cur_iteration = 0;
  for (int breakpoint : test_case.breakpoints) {
    VariantTensorData data;
    VariantTensorDataWriter writer(&data);
    TF_EXPECT_OK(iterator->Save(serialization_ctx.get(), &writer));
    TF_EXPECT_OK(writer.Flush());
    VariantTensorDataReader reader(&data);
    TF_EXPECT_OK(RestoreIterator(iterator_ctx.get(), &reader, "Iterator",
                                 *dataset, &iterator));
    while (cur_iteration <= breakpoint) {
      while (!end_of_sequence) {
        // Owns the datasets, which are stored as the variant tensors in the
        // vector.
        std::vector<Tensor> out_tensors;
        TF_EXPECT_OK(iterator->GetNext(iterator_ctx.get(), &out_tensors,
                                       &end_of_sequence));
        if (!end_of_sequence) {
          for (const auto& window_dataset_tensor : out_tensors) {
            // Not owned.
            DatasetBase* window_dataset;
            TF_ASSERT_OK(GetDatasetFromVariantTensor(window_dataset_tensor,
                                                     &window_dataset));
            std::unique_ptr<IteratorBase> window_dataset_iterator;
            TF_ASSERT_OK(window_dataset->MakeIterator(
                iterator_ctx.get(), "Iterator", &window_dataset_iterator));
            bool end_of_window_dataset = false;
            std::vector<Tensor> window_elements;
            while (!end_of_window_dataset) {
              std::vector<Tensor> next_element;
              TF_EXPECT_OK(window_dataset_iterator->GetNext(
                  iterator_ctx.get(), &next_element, &end_of_window_dataset));
              window_elements.insert(window_elements.end(),
                                     next_element.begin(), next_element.end());
            }
            EXPECT_LT(expected_outputs_it, test_case.expected_outputs.end());
            TF_EXPECT_OK(
                ExpectEqual(window_elements, *expected_outputs_it, false));
            expected_outputs_it++;
          }
        }
      }
      cur_iteration++;
    }
  }
  EXPECT_EQ(expected_outputs_it, test_case.expected_outputs.end());
}

INSTANTIATE_TEST_SUITE_P(
    WindowDatasetOpTest, ParameterizedWindowDatasetOpTest,
    ::testing::ValuesIn(std::vector<TestCase>(
        {TestCase1(), TestCase2(), TestCase3(), TestCase4(), TestCase5(),
         TestCase6(), TestCase7(), TestCase8(), TestCase9(), TestCase10()})));

TEST_F(WindowDatasetOpTest, InvalidArguments) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  std::vector<TestCase> test_cases({InvalidWindowSizeTestCase(),
                                    InvalidWindowShiftTestCase(),
                                    InvalidWindowStrideTestCase()});
  for (const auto& test_case : test_cases) {
    std::unique_ptr<OpKernel> window_dataset_kernel;
    TF_ASSERT_OK(CreateWindowDatasetKernel(test_case.expected_output_dtypes,
                                           test_case.expected_output_shapes,
                                           &window_dataset_kernel));
    DatasetBase* range_dataset;
    TF_ASSERT_OK(CreateRangeDataset<int64>(
        test_case.range_data_param.start, test_case.range_data_param.end,
        test_case.range_data_param.step, "range", &range_dataset));
    Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
    TF_ASSERT_OK(
        StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));
    Tensor size = test_case.size;
    Tensor shift = test_case.shift;
    Tensor stride = test_case.stride;
    Tensor drop_remainder = test_case.drop_remainder;
    gtl::InlinedVector<TensorValue, 4> inputs(
        {TensorValue(&range_dataset_tensor), TensorValue(&size),
         TensorValue(&shift), TensorValue(&stride),
         TensorValue(&drop_remainder)});

    std::unique_ptr<OpKernelContext> window_dataset_op_ctx;
    TF_ASSERT_OK(CreateWindowDatasetContext(window_dataset_kernel.get(),
                                            &inputs, &window_dataset_op_ctx));
    DatasetBase* dataset;
    EXPECT_EQ(CreateDataset(window_dataset_kernel.get(),
                            window_dataset_op_ctx.get(), &dataset)
                  .code(),
              tensorflow::error::INVALID_ARGUMENT);
  }
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
