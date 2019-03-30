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

constexpr char kNodeName[] = "zip_dataset";
constexpr char kOpName[] = "ZipDataset";

struct RangeDatasetParam {
  int64 start;
  int64 end;
  int64 step;
};

class ZipDatasetOpTest : public DatasetOpsTestBase {
 protected:
  // Creates `RangeDataset` variant tensors from the input vector of
  // `RangeDatasetParam`.
  Status CreateRangeDatasetTensors(const std::vector<RangeDatasetParam> &params,
                                   std::vector<Tensor> *const dataset_tensors) {
    for (int i = 0; i < params.size(); ++i) {
      DatasetBase *range_dataset;
      TF_RETURN_IF_ERROR(CreateRangeDataset<int64>(
          params[i].start, params[i].end, params[i].step,
          strings::StrCat("range_", i), &range_dataset));
      Tensor dataset_tensor(DT_VARIANT, TensorShape({}));
      TF_RETURN_IF_ERROR(
          StoreDatasetInVariantTensor(range_dataset, &dataset_tensor));
      dataset_tensors->emplace_back(std::move(dataset_tensor));
    }
    return Status::OK();
  }

  // Creates a new ZipDataset op kernel.
  Status CreateZipDatasetKernel(
      const DataTypeVector &dtypes,
      const std::vector<PartialTensorShape> &output_shapes, int n,
      std::unique_ptr<OpKernel> *op_kernel) {
    std::vector<string> input_datasets;
    input_datasets.reserve(n);
    for (int i = 0; i < n; ++i) {
      // Create the placeholder names for the input components of `ZipDataset`.
      input_datasets.emplace_back(strings::StrCat("input_dataset_", i));
    }
    NodeDef node_def = test::function::NDef(
        kNodeName, kOpName, input_datasets,
        {{"output_types", dtypes}, {"output_shapes", output_shapes}, {"N", n}});
    TF_RETURN_IF_ERROR(CreateOpKernel(node_def, op_kernel));
    return Status::OK();
  }

  // Creates a new ZipDataset op kernel context.
  Status CreateZipDatasetContext(
      OpKernel *const op_kernel,
      gtl::InlinedVector<TensorValue, 4> *const inputs,
      std::unique_ptr<OpKernelContext> *context) {
    TF_RETURN_IF_ERROR(CheckOpKernelInput(*op_kernel, *inputs));
    TF_RETURN_IF_ERROR(CreateOpKernelContext(op_kernel, inputs, context));
    return Status::OK();
  }
};

struct TestParam {
  std::vector<RangeDatasetParam> input_range_dataset_params;
  std::vector<Tensor> expected_outputs;
  std::vector<int> breakpoints;
};

// Test case 1: the input datasets with same number of outputs.
TestParam TestCase1() {
  return {/*input_range_dataset_params*/
          {RangeDatasetParam{0, 3, 1}, RangeDatasetParam{10, 13, 1}},
          /*expected_outputs*/
          {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{}, {0}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape{}, {10}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape{}, {1}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape{}, {11}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape{}, {2}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape{}, {12})},
          /*breakpoints*/ {0, 1, 4}};
}

// Test case 2: the input datasets with different number of outputs.
TestParam TestCase2() {
  return {/*input_range_dataset_params*/
          {RangeDatasetParam{0, 3, 1}, RangeDatasetParam{10, 15, 1}},
          /*expected_outputs*/
          {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{}, {0}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape{}, {10}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape{}, {1}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape{}, {11}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape{}, {2}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape{}, {12})},
          /*breakpoints*/ {0, 1, 4}};
}

class ParameterizedZipDatasetOpTest
    : public ZipDatasetOpTest,
      public ::testing::WithParamInterface<TestParam> {};

TEST_P(ParameterizedZipDatasetOpTest, GetNext) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestParam &test_case = GetParam();
  std::vector<Tensor> range_dataset_tensors;
  range_dataset_tensors.reserve(test_case.input_range_dataset_params.size());
  TF_ASSERT_OK(CreateRangeDatasetTensors(test_case.input_range_dataset_params,
                                         &range_dataset_tensors));
  gtl::InlinedVector<TensorValue, 4> inputs;
  inputs.reserve(range_dataset_tensors.size());
  for (auto &tensor : range_dataset_tensors) {
    inputs.emplace_back(&tensor);
  }
  std::unique_ptr<OpKernel> dataset_kernel;
  int num_tensors_per_slice = test_case.input_range_dataset_params.size();
  TF_ASSERT_OK(CreateZipDatasetKernel({DT_INT64}, {{num_tensors_per_slice}},
                                      inputs.size(), &dataset_kernel));
  std::unique_ptr<OpKernelContext> dataset_kernel_ctx;
  TF_ASSERT_OK(CreateZipDatasetContext(dataset_kernel.get(), &inputs,
                                       &dataset_kernel_ctx));
  DatasetBase *zip_dataset;
  TF_ASSERT_OK(CreateDataset(dataset_kernel.get(), dataset_kernel_ctx.get(),
                             &zip_dataset));
  core::ScopedUnref scoped_unref(zip_dataset);
  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(CreateIteratorContext(dataset_kernel_ctx.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      zip_dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));

  auto expected_outputs_it = test_case.expected_outputs.begin();
  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  while (!end_of_sequence) {
    TF_EXPECT_OK(
        iterator->GetNext(iterator_ctx.get(), &out_tensors, &end_of_sequence));
    if (!end_of_sequence) {
      for (const auto &tensor : out_tensors) {
        EXPECT_NE(expected_outputs_it, test_case.expected_outputs.end());
        TF_EXPECT_OK(ExpectEqual(tensor, *expected_outputs_it));
        expected_outputs_it++;
      }
    }
  }
  EXPECT_EQ(expected_outputs_it, test_case.expected_outputs.end());
}

TEST_F(ZipDatasetOpTest, DatasetNodeName) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestParam &test_case = TestCase1();
  std::vector<Tensor> range_dataset_tensors;
  range_dataset_tensors.reserve(test_case.input_range_dataset_params.size());
  TF_ASSERT_OK(CreateRangeDatasetTensors(test_case.input_range_dataset_params,
                                         &range_dataset_tensors));
  gtl::InlinedVector<TensorValue, 4> inputs;
  inputs.reserve(range_dataset_tensors.size());
  for (auto &tensor : range_dataset_tensors) {
    inputs.emplace_back(&tensor);
  }
  std::unique_ptr<OpKernel> dataset_kernel;
  int num_tensors_per_slice = test_case.input_range_dataset_params.size();
  TF_ASSERT_OK(CreateZipDatasetKernel({DT_INT64}, {{num_tensors_per_slice}},
                                      inputs.size(), &dataset_kernel));
  std::unique_ptr<OpKernelContext> dataset_kernel_ctx;
  TF_ASSERT_OK(CreateZipDatasetContext(dataset_kernel.get(), &inputs,
                                       &dataset_kernel_ctx));
  DatasetBase *zip_dataset;
  TF_ASSERT_OK(CreateDataset(dataset_kernel.get(), dataset_kernel_ctx.get(),
                             &zip_dataset));
  core::ScopedUnref scoped_unref(zip_dataset);

  EXPECT_EQ(zip_dataset->node_name(), kNodeName);
}

TEST_F(ZipDatasetOpTest, DatasetTypeString) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestParam &test_case = TestCase1();
  std::vector<Tensor> range_dataset_tensors;
  range_dataset_tensors.reserve(test_case.input_range_dataset_params.size());
  TF_ASSERT_OK(CreateRangeDatasetTensors(test_case.input_range_dataset_params,
                                         &range_dataset_tensors));
  gtl::InlinedVector<TensorValue, 4> inputs;
  inputs.reserve(range_dataset_tensors.size());
  for (auto &tensor : range_dataset_tensors) {
    inputs.emplace_back(&tensor);
  }
  std::unique_ptr<OpKernel> dataset_kernel;
  int num_tensors_per_slice = test_case.input_range_dataset_params.size();
  TF_ASSERT_OK(CreateZipDatasetKernel({DT_INT64}, {{num_tensors_per_slice}},
                                      inputs.size(), &dataset_kernel));
  std::unique_ptr<OpKernelContext> dataset_kernel_ctx;
  TF_ASSERT_OK(CreateZipDatasetContext(dataset_kernel.get(), &inputs,
                                       &dataset_kernel_ctx));
  DatasetBase *zip_dataset;
  TF_ASSERT_OK(CreateDataset(dataset_kernel.get(), dataset_kernel_ctx.get(),
                             &zip_dataset));
  core::ScopedUnref scoped_unref(zip_dataset);

  EXPECT_EQ(zip_dataset->type_string(), kOpName);
}

TEST_P(ParameterizedZipDatasetOpTest, DatasetOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestParam &test_case = GetParam();
  std::vector<Tensor> range_dataset_tensors;
  range_dataset_tensors.reserve(test_case.input_range_dataset_params.size());
  TF_ASSERT_OK(CreateRangeDatasetTensors(test_case.input_range_dataset_params,
                                         &range_dataset_tensors));
  gtl::InlinedVector<TensorValue, 4> inputs;
  inputs.reserve(range_dataset_tensors.size());
  for (auto &tensor : range_dataset_tensors) {
    inputs.emplace_back(&tensor);
  }
  std::unique_ptr<OpKernel> dataset_kernel;
  int num_tensors_per_slice = test_case.input_range_dataset_params.size();
  TF_ASSERT_OK(CreateZipDatasetKernel({DT_INT64}, {{num_tensors_per_slice}},
                                      inputs.size(), &dataset_kernel));
  std::unique_ptr<OpKernelContext> dataset_kernel_ctx;
  TF_ASSERT_OK(CreateZipDatasetContext(dataset_kernel.get(), &inputs,
                                       &dataset_kernel_ctx));
  DatasetBase *zip_dataset;
  TF_ASSERT_OK(CreateDataset(dataset_kernel.get(), dataset_kernel_ctx.get(),
                             &zip_dataset));
  core::ScopedUnref scoped_unref(zip_dataset);

  DataTypeVector expected_output_dtypes;
  expected_output_dtypes.reserve(num_tensors_per_slice);
  for (int i = 0; i < num_tensors_per_slice; ++i) {
    expected_output_dtypes.emplace_back(test_case.expected_outputs[i].dtype());
  }

  TF_EXPECT_OK(
      VerifyTypesMatch(zip_dataset->output_dtypes(), expected_output_dtypes));
}

TEST_P(ParameterizedZipDatasetOpTest, DatasetOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestParam &test_case = GetParam();
  std::vector<Tensor> range_dataset_tensors;
  range_dataset_tensors.reserve(test_case.input_range_dataset_params.size());
  TF_ASSERT_OK(CreateRangeDatasetTensors(test_case.input_range_dataset_params,
                                         &range_dataset_tensors));
  gtl::InlinedVector<TensorValue, 4> inputs;
  inputs.reserve(range_dataset_tensors.size());
  for (auto &tensor : range_dataset_tensors) {
    inputs.emplace_back(&tensor);
  }
  std::unique_ptr<OpKernel> dataset_kernel;
  int num_tensors_per_slice = test_case.input_range_dataset_params.size();
  TF_ASSERT_OK(CreateZipDatasetKernel({DT_INT64}, {{num_tensors_per_slice}},
                                      inputs.size(), &dataset_kernel));
  std::unique_ptr<OpKernelContext> dataset_kernel_ctx;
  TF_ASSERT_OK(CreateZipDatasetContext(dataset_kernel.get(), &inputs,
                                       &dataset_kernel_ctx));
  DatasetBase *zip_dataset;
  TF_ASSERT_OK(CreateDataset(dataset_kernel.get(), dataset_kernel_ctx.get(),
                             &zip_dataset));
  core::ScopedUnref scoped_unref(zip_dataset);

  std::vector<PartialTensorShape> expected_output_shapes;
  expected_output_shapes.reserve(num_tensors_per_slice);
  for (int i = 0; i < num_tensors_per_slice; ++i) {
    expected_output_shapes.emplace_back(test_case.expected_outputs[i].shape());
  }

  TF_EXPECT_OK(VerifyShapesCompatible(zip_dataset->output_shapes(),
                                      expected_output_shapes));
}

TEST_P(ParameterizedZipDatasetOpTest, Cardinality) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestParam &test_case = GetParam();
  std::vector<Tensor> range_dataset_tensors;
  range_dataset_tensors.reserve(test_case.input_range_dataset_params.size());
  TF_ASSERT_OK(CreateRangeDatasetTensors(test_case.input_range_dataset_params,
                                         &range_dataset_tensors));
  gtl::InlinedVector<TensorValue, 4> inputs;
  inputs.reserve(range_dataset_tensors.size());
  for (auto &tensor : range_dataset_tensors) {
    inputs.emplace_back(&tensor);
  }
  std::unique_ptr<OpKernel> dataset_kernel;
  int num_tensors_per_slice = test_case.input_range_dataset_params.size();
  TF_ASSERT_OK(CreateZipDatasetKernel({DT_INT64}, {{num_tensors_per_slice}},
                                      inputs.size(), &dataset_kernel));
  std::unique_ptr<OpKernelContext> dataset_kernel_ctx;
  TF_ASSERT_OK(CreateZipDatasetContext(dataset_kernel.get(), &inputs,
                                       &dataset_kernel_ctx));
  DatasetBase *zip_dataset;
  TF_ASSERT_OK(CreateDataset(dataset_kernel.get(), dataset_kernel_ctx.get(),
                             &zip_dataset));
  core::ScopedUnref scoped_unref(zip_dataset);

  EXPECT_EQ(zip_dataset->Cardinality(),
            test_case.expected_outputs.size() / num_tensors_per_slice);
}

TEST_F(ZipDatasetOpTest, DatasetSave) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestParam &test_case = TestCase1();
  std::vector<Tensor> range_dataset_tensors;
  range_dataset_tensors.reserve(test_case.input_range_dataset_params.size());
  TF_ASSERT_OK(CreateRangeDatasetTensors(test_case.input_range_dataset_params,
                                         &range_dataset_tensors));
  gtl::InlinedVector<TensorValue, 4> inputs;
  inputs.reserve(range_dataset_tensors.size());
  for (auto &tensor : range_dataset_tensors) {
    inputs.emplace_back(&tensor);
  }
  std::unique_ptr<OpKernel> dataset_kernel;
  int num_tensors_per_slice = test_case.input_range_dataset_params.size();
  TF_ASSERT_OK(CreateZipDatasetKernel({DT_INT64}, {{num_tensors_per_slice}},
                                      inputs.size(), &dataset_kernel));
  std::unique_ptr<OpKernelContext> dataset_kernel_ctx;
  TF_ASSERT_OK(CreateZipDatasetContext(dataset_kernel.get(), &inputs,
                                       &dataset_kernel_ctx));
  DatasetBase *zip_dataset;
  TF_ASSERT_OK(CreateDataset(dataset_kernel.get(), dataset_kernel_ctx.get(),
                             &zip_dataset));
  core::ScopedUnref scoped_unref(zip_dataset);

  std::unique_ptr<SerializationContext> serialization_ctx;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_ctx));
  VariantTensorData data;
  VariantTensorDataWriter writer(&data);
  TF_ASSERT_OK(zip_dataset->Save(serialization_ctx.get(), &writer));
  TF_ASSERT_OK(writer.Flush());
}

TEST_P(ParameterizedZipDatasetOpTest, IteratorOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestParam &test_case = GetParam();
  std::vector<Tensor> range_dataset_tensors;
  range_dataset_tensors.reserve(test_case.input_range_dataset_params.size());
  TF_ASSERT_OK(CreateRangeDatasetTensors(test_case.input_range_dataset_params,
                                         &range_dataset_tensors));
  gtl::InlinedVector<TensorValue, 4> inputs;
  inputs.reserve(range_dataset_tensors.size());
  for (auto &tensor : range_dataset_tensors) {
    inputs.emplace_back(&tensor);
  }
  std::unique_ptr<OpKernel> dataset_kernel;
  int num_tensors_per_slice = test_case.input_range_dataset_params.size();
  TF_ASSERT_OK(CreateZipDatasetKernel({DT_INT64}, {{num_tensors_per_slice}},
                                      inputs.size(), &dataset_kernel));
  std::unique_ptr<OpKernelContext> dataset_kernel_ctx;
  TF_ASSERT_OK(CreateZipDatasetContext(dataset_kernel.get(), &inputs,
                                       &dataset_kernel_ctx));
  DatasetBase *zip_dataset;
  TF_ASSERT_OK(CreateDataset(dataset_kernel.get(), dataset_kernel_ctx.get(),
                             &zip_dataset));
  core::ScopedUnref scoped_unref(zip_dataset);
  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(CreateIteratorContext(dataset_kernel_ctx.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      zip_dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));

  DataTypeVector expected_output_dtypes;
  expected_output_dtypes.reserve(num_tensors_per_slice);
  for (int i = 0; i < num_tensors_per_slice; ++i) {
    expected_output_dtypes.emplace_back(test_case.expected_outputs[i].dtype());
  }

  TF_EXPECT_OK(
      VerifyTypesMatch(iterator->output_dtypes(), expected_output_dtypes));
}

TEST_P(ParameterizedZipDatasetOpTest, IteratorOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestParam &test_case = GetParam();
  std::vector<Tensor> range_dataset_tensors;
  range_dataset_tensors.reserve(test_case.input_range_dataset_params.size());
  TF_ASSERT_OK(CreateRangeDatasetTensors(test_case.input_range_dataset_params,
                                         &range_dataset_tensors));
  gtl::InlinedVector<TensorValue, 4> inputs;
  inputs.reserve(range_dataset_tensors.size());
  for (auto &tensor : range_dataset_tensors) {
    inputs.emplace_back(&tensor);
  }
  std::unique_ptr<OpKernel> dataset_kernel;
  int num_tensors_per_slice = test_case.input_range_dataset_params.size();
  TF_ASSERT_OK(CreateZipDatasetKernel({DT_INT64}, {{num_tensors_per_slice}},
                                      inputs.size(), &dataset_kernel));
  std::unique_ptr<OpKernelContext> dataset_kernel_ctx;
  TF_ASSERT_OK(CreateZipDatasetContext(dataset_kernel.get(), &inputs,
                                       &dataset_kernel_ctx));
  DatasetBase *zip_dataset;
  TF_ASSERT_OK(CreateDataset(dataset_kernel.get(), dataset_kernel_ctx.get(),
                             &zip_dataset));
  core::ScopedUnref scoped_unref(zip_dataset);
  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(CreateIteratorContext(dataset_kernel_ctx.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      zip_dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));

  std::vector<PartialTensorShape> expected_output_shapes;
  expected_output_shapes.reserve(num_tensors_per_slice);
  for (int i = 0; i < num_tensors_per_slice; ++i) {
    expected_output_shapes.emplace_back(test_case.expected_outputs[i].shape());
  }

  TF_EXPECT_OK(VerifyShapesCompatible(iterator->output_shapes(),
                                      expected_output_shapes));
}

TEST_F(ZipDatasetOpTest, IteratorOutputPrefix) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestParam &test_case = TestCase1();
  std::vector<Tensor> range_dataset_tensors;
  range_dataset_tensors.reserve(test_case.input_range_dataset_params.size());
  TF_ASSERT_OK(CreateRangeDatasetTensors(test_case.input_range_dataset_params,
                                         &range_dataset_tensors));
  gtl::InlinedVector<TensorValue, 4> inputs;
  inputs.reserve(range_dataset_tensors.size());
  for (auto &tensor : range_dataset_tensors) {
    inputs.emplace_back(&tensor);
  }
  std::unique_ptr<OpKernel> dataset_kernel;
  int num_tensors_per_slice = test_case.input_range_dataset_params.size();
  TF_ASSERT_OK(CreateZipDatasetKernel({DT_INT64}, {{num_tensors_per_slice}},
                                      inputs.size(), &dataset_kernel));
  std::unique_ptr<OpKernelContext> dataset_kernel_ctx;
  TF_ASSERT_OK(CreateZipDatasetContext(dataset_kernel.get(), &inputs,
                                       &dataset_kernel_ctx));
  DatasetBase *zip_dataset;
  TF_ASSERT_OK(CreateDataset(dataset_kernel.get(), dataset_kernel_ctx.get(),
                             &zip_dataset));
  core::ScopedUnref scoped_unref(zip_dataset);
  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(CreateIteratorContext(dataset_kernel_ctx.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      zip_dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));

  EXPECT_EQ(iterator->prefix(), "Iterator::Zip");
}

TEST_P(ParameterizedZipDatasetOpTest, Roundtrip) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestParam &test_case = GetParam();
  std::vector<Tensor> range_dataset_tensors;
  range_dataset_tensors.reserve(test_case.input_range_dataset_params.size());
  TF_ASSERT_OK(CreateRangeDatasetTensors(test_case.input_range_dataset_params,
                                         &range_dataset_tensors));
  gtl::InlinedVector<TensorValue, 4> inputs;
  inputs.reserve(range_dataset_tensors.size());
  for (auto &tensor : range_dataset_tensors) {
    inputs.emplace_back(&tensor);
  }
  std::unique_ptr<OpKernel> dataset_kernel;
  int num_tensors_per_slice = test_case.input_range_dataset_params.size();
  TF_ASSERT_OK(CreateZipDatasetKernel({DT_INT64}, {{num_tensors_per_slice}},
                                      inputs.size(), &dataset_kernel));
  std::unique_ptr<OpKernelContext> dataset_kernel_ctx;
  TF_ASSERT_OK(CreateZipDatasetContext(dataset_kernel.get(), &inputs,
                                       &dataset_kernel_ctx));
  DatasetBase *zip_dataset;
  TF_ASSERT_OK(CreateDataset(dataset_kernel.get(), dataset_kernel_ctx.get(),
                             &zip_dataset));
  core::ScopedUnref scoped_unref(zip_dataset);
  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(CreateIteratorContext(dataset_kernel_ctx.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      zip_dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));

  std::unique_ptr<SerializationContext> serialization_ctx;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_ctx));

  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  auto expected_outputs_it = test_case.expected_outputs.begin();
  int cur_iteration = 0;
  for (int breakpoint : test_case.breakpoints) {
    VariantTensorData data;
    VariantTensorDataWriter writer(&data);
    TF_EXPECT_OK(iterator->Save(serialization_ctx.get(), &writer));
    TF_EXPECT_OK(writer.Flush());
    VariantTensorDataReader reader(&data);
    TF_EXPECT_OK(iterator->Restore(iterator_ctx.get(), &reader));

    while (cur_iteration < breakpoint) {
      TF_EXPECT_OK(iterator->GetNext(iterator_ctx.get(), &out_tensors,
                                     &end_of_sequence));
      if (!end_of_sequence) {
        for (auto &tensor : out_tensors) {
          EXPECT_NE(expected_outputs_it, test_case.expected_outputs.end());
          TF_EXPECT_OK(ExpectEqual(tensor, *expected_outputs_it));
          expected_outputs_it++;
        }
      }
      cur_iteration++;
    }

    if (breakpoint >= zip_dataset->Cardinality()) {
      EXPECT_TRUE(end_of_sequence);
      EXPECT_EQ(expected_outputs_it, test_case.expected_outputs.end());
    } else {
      EXPECT_FALSE(end_of_sequence);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    ZipDatasetOpTest, ParameterizedZipDatasetOpTest,
    ::testing::ValuesIn(std::vector<TestParam>({TestCase1(), TestCase2()})));

}  // namespace
}  // namespace data
}  // namespace tensorflow
