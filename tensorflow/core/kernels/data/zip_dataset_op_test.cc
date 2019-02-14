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
  // Creates `RangeDataset` variant tensors
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
  }

  // Creates a new ZipDataset op kernel.
  Status CreateZipDatasetKernel(
      const DataTypeVector &dtypes,
      const std::vector<PartialTensorShape> &output_shapes, int n,
      std::unique_ptr<OpKernel> *op_kernel) {
    std::vector<string> input_datasets;
    input_datasets.reserve(n);
    for (int i = 0; i < n; ++i) {
      input_datasets.emplace_back(strings::StrCat("input_dataset_", i));
    }
    node_def_ = test::function::NDef(
        kNodeName, kOpName, input_datasets,
        {{"output_types", dtypes}, {"output_shapes", output_shapes}, {"N", n}});
    TF_RETURN_IF_ERROR(CreateOpKernel(node_def_, op_kernel));
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

 private:
  NodeDef node_def_;
};

struct TestParam {
  std::vector<RangeDatasetParam> testcases;
  std::vector<Tensor> expected_outputs;
  std::vector<int> breakpoints;
} TestCases[] = {
    // Test the input datasets with same number of outputs.
    {{RangeDatasetParam{0, 3, 1}, RangeDatasetParam{10, 13, 1}},
     {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{}, {0}),
      DatasetOpsTestBase::CreateTensor<int64>(TensorShape{}, {10}),
      DatasetOpsTestBase::CreateTensor<int64>(TensorShape{}, {1}),
      DatasetOpsTestBase::CreateTensor<int64>(TensorShape{}, {11}),
      DatasetOpsTestBase::CreateTensor<int64>(TensorShape{}, {2}),
      DatasetOpsTestBase::CreateTensor<int64>(TensorShape{}, {12})},
     {0, 1, 4}},
    // Test the input datasets with different number of outputs.
    {{RangeDatasetParam{0, 3, 1}, RangeDatasetParam{10, 15, 1}},
     {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{}, {0}),
      DatasetOpsTestBase::CreateTensor<int64>(TensorShape{}, {10}),
      DatasetOpsTestBase::CreateTensor<int64>(TensorShape{}, {1}),
      DatasetOpsTestBase::CreateTensor<int64>(TensorShape{}, {11}),
      DatasetOpsTestBase::CreateTensor<int64>(TensorShape{}, {2}),
      DatasetOpsTestBase::CreateTensor<int64>(TensorShape{}, {12})},
     {0, 1, 4}}};

struct DatasetGetNextTest : ZipDatasetOpTest,
                            ::testing::WithParamInterface<TestParam> {};

TEST_P(DatasetGetNextTest, GetNext) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::vector<RangeDatasetParam> testcases = GetParam().testcases;
  std::vector<Tensor> expected_outputs = GetParam().expected_outputs;
  auto expected_outputs_it = expected_outputs.begin();

  std::vector<Tensor> range_dataset_tensors;
  TF_ASSERT_OK(CreateRangeDatasetTensors(testcases, &range_dataset_tensors));
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto &tensor : range_dataset_tensors) {
    inputs.emplace_back(&tensor);
  }
  int num_tensors_per_slice = testcases.size();
  std::unique_ptr<OpKernel> dataset_kernel;
  TF_ASSERT_OK(CreateZipDatasetKernel({DT_INT64}, {{num_tensors_per_slice}},
                                      inputs.size(), &dataset_kernel));
  std::unique_ptr<OpKernelContext> dataset_kernel_ctx;
  TF_ASSERT_OK(CreateZipDatasetContext(dataset_kernel.get(), &inputs,
                                       &dataset_kernel_ctx));
  DatasetBase *dataset;
  TF_ASSERT_OK(
      CreateDataset(dataset_kernel.get(), dataset_kernel_ctx.get(), &dataset));
  core::ScopedUnref scoped_unref(dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(CreateIteratorContext(dataset_kernel_ctx.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));

  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  while (!end_of_sequence) {
    TF_EXPECT_OK(
        iterator->GetNext(iterator_ctx.get(), &out_tensors, &end_of_sequence));
    if (!end_of_sequence) {
      EXPECT_NE(expected_outputs_it, expected_outputs.end());
      for (int i = 0; i < num_tensors_per_slice; ++i) {
        TF_EXPECT_OK(ExpectEqual(out_tensors[i], *expected_outputs_it));
        expected_outputs_it++;
      }
    }
  }
  EXPECT_EQ(expected_outputs_it, expected_outputs.end());
}

INSTANTIATE_TEST_CASE_P(ZipDatasetOpTest, DatasetGetNextTest,
                        ::testing::ValuesIn(TestCases));

TEST_F(ZipDatasetOpTest, DatasetName) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::vector<RangeDatasetParam> testcases = TestCases[0].testcases;
  std::vector<Tensor> range_dataset_tensors;
  TF_ASSERT_OK(CreateRangeDatasetTensors(testcases, &range_dataset_tensors));
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto &tensor : range_dataset_tensors) {
    inputs.emplace_back(&tensor);
  }

  std::unique_ptr<OpKernel> dataset_kernel;
  TF_ASSERT_OK(CreateZipDatasetKernel({DT_INT64}, {{2}}, inputs.size(),
                                      &dataset_kernel));
  std::unique_ptr<OpKernelContext> dataset_kernel_ctx;
  TF_ASSERT_OK(CreateZipDatasetContext(dataset_kernel.get(), &inputs,
                                       &dataset_kernel_ctx));
  DatasetBase *dataset;
  TF_ASSERT_OK(
      CreateDataset(dataset_kernel.get(), dataset_kernel_ctx.get(), &dataset));
  core::ScopedUnref scoped_unref(dataset);

  EXPECT_EQ(dataset->name(), kOpName);
}

struct DatasetOutputDtypesTest : ZipDatasetOpTest,
                                 ::testing::WithParamInterface<TestParam> {};

TEST_P(DatasetOutputDtypesTest, DatasetOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::vector<RangeDatasetParam> testcases = GetParam().testcases;
  std::vector<Tensor> expected_outputs = GetParam().expected_outputs;

  std::vector<Tensor> range_dataset_tensors;
  TF_ASSERT_OK(CreateRangeDatasetTensors(testcases, &range_dataset_tensors));
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto &tensor : range_dataset_tensors) {
    inputs.emplace_back(&tensor);
  }
  int num_tensors_per_slice = testcases.size();
  std::unique_ptr<OpKernel> dataset_kernel;
  TF_ASSERT_OK(CreateZipDatasetKernel({DT_INT64}, {{num_tensors_per_slice}},
                                      inputs.size(), &dataset_kernel));
  std::unique_ptr<OpKernelContext> dataset_kernel_ctx;
  TF_ASSERT_OK(CreateZipDatasetContext(dataset_kernel.get(), &inputs,
                                       &dataset_kernel_ctx));
  DatasetBase *dataset;
  TF_ASSERT_OK(
      CreateDataset(dataset_kernel.get(), dataset_kernel_ctx.get(), &dataset));
  core::ScopedUnref scoped_unref(dataset);

  DataTypeVector expected_output_dtypes;
  for (int i = 0; i < num_tensors_per_slice; ++i) {
    expected_output_dtypes.emplace_back(expected_outputs[i].dtype());
  }

  TF_EXPECT_OK(
      VerifyTypesMatch(dataset->output_dtypes(), expected_output_dtypes));
}

INSTANTIATE_TEST_CASE_P(ZipDatasetOpTest, DatasetOutputDtypesTest,
                        ::testing::ValuesIn(TestCases));

struct DatasetOutputShapesTest : ZipDatasetOpTest,
                                 ::testing::WithParamInterface<TestParam> {};

TEST_P(DatasetOutputShapesTest, DatasetOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::vector<RangeDatasetParam> testcases = GetParam().testcases;
  std::vector<Tensor> expected_outputs = GetParam().expected_outputs;

  std::vector<Tensor> range_dataset_tensors;
  TF_ASSERT_OK(CreateRangeDatasetTensors(testcases, &range_dataset_tensors));
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto &tensor : range_dataset_tensors) {
    inputs.emplace_back(&tensor);
  }
  int num_tensors_per_slice = testcases.size();
  std::unique_ptr<OpKernel> dataset_kernel;
  TF_ASSERT_OK(CreateZipDatasetKernel({DT_INT64}, {{num_tensors_per_slice}},
                                      inputs.size(), &dataset_kernel));
  std::unique_ptr<OpKernelContext> dataset_kernel_ctx;
  TF_ASSERT_OK(CreateZipDatasetContext(dataset_kernel.get(), &inputs,
                                       &dataset_kernel_ctx));
  DatasetBase *dataset;
  TF_ASSERT_OK(
      CreateDataset(dataset_kernel.get(), dataset_kernel_ctx.get(), &dataset));
  core::ScopedUnref scoped_unref(dataset);

  std::vector<PartialTensorShape> expected_output_shapes;
  for (int i = 0; i < num_tensors_per_slice; ++i) {
    expected_output_shapes.emplace_back(expected_outputs[i].shape());
  }

  TF_EXPECT_OK(
      VerifyShapesCompatible(dataset->output_shapes(), expected_output_shapes));
}

INSTANTIATE_TEST_CASE_P(ZipDatasetOpTest, DatasetOutputShapesTest,
                        ::testing::ValuesIn(TestCases));

struct DatasetCardinalityTest : ZipDatasetOpTest,
                                ::testing::WithParamInterface<TestParam> {};

TEST_P(DatasetCardinalityTest, Cardinality) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::vector<RangeDatasetParam> testcases = GetParam().testcases;
  std::vector<Tensor> expected_outputs = GetParam().expected_outputs;

  std::vector<Tensor> range_dataset_tensors;
  TF_ASSERT_OK(CreateRangeDatasetTensors(testcases, &range_dataset_tensors));
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto &tensor : range_dataset_tensors) {
    inputs.emplace_back(&tensor);
  }
  int num_tensors_per_slice = testcases.size();
  std::unique_ptr<OpKernel> dataset_kernel;
  TF_ASSERT_OK(CreateZipDatasetKernel({DT_INT64}, {{num_tensors_per_slice}},
                                      inputs.size(), &dataset_kernel));
  std::unique_ptr<OpKernelContext> dataset_kernel_ctx;
  TF_ASSERT_OK(CreateZipDatasetContext(dataset_kernel.get(), &inputs,
                                       &dataset_kernel_ctx));
  DatasetBase *dataset;
  TF_ASSERT_OK(
      CreateDataset(dataset_kernel.get(), dataset_kernel_ctx.get(), &dataset));
  core::ScopedUnref scoped_unref(dataset);

  EXPECT_EQ(dataset->Cardinality(),
            expected_outputs.size() / num_tensors_per_slice);
}

INSTANTIATE_TEST_CASE_P(ZipDatasetOpTest, DatasetCardinalityTest,
                        ::testing::ValuesIn(TestCases));

TEST_F(ZipDatasetOpTest, DatasetSave) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::vector<RangeDatasetParam> testcases = TestCases[0].testcases;
  std::vector<Tensor> range_dataset_tensors;
  TF_ASSERT_OK(CreateRangeDatasetTensors(testcases, &range_dataset_tensors));
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto &tensor : range_dataset_tensors) {
    inputs.emplace_back(&tensor);
  }

  std::unique_ptr<OpKernel> dataset_kernel;
  TF_ASSERT_OK(CreateZipDatasetKernel({DT_INT64}, {{2}}, inputs.size(),
                                      &dataset_kernel));
  std::unique_ptr<OpKernelContext> dataset_kernel_ctx;
  TF_ASSERT_OK(CreateZipDatasetContext(dataset_kernel.get(), &inputs,
                                       &dataset_kernel_ctx));
  DatasetBase *dataset;
  TF_ASSERT_OK(
      CreateDataset(dataset_kernel.get(), dataset_kernel_ctx.get(), &dataset));
  core::ScopedUnref scoped_unref(dataset);

  std::unique_ptr<SerializationContext> serialization_ctx;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_ctx));
  VariantTensorData data;
  VariantTensorDataWriter writer(&data);
  TF_ASSERT_OK(dataset->Save(serialization_ctx.get(), &writer));
  TF_ASSERT_OK(writer.Flush());
}

struct IteratorOutputDtypesTest : ZipDatasetOpTest,
                                  ::testing::WithParamInterface<TestParam> {};

TEST_P(IteratorOutputDtypesTest, IteratorOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::vector<RangeDatasetParam> testcases = GetParam().testcases;
  std::vector<Tensor> expected_outputs = GetParam().expected_outputs;

  std::vector<Tensor> range_dataset_tensors;
  TF_ASSERT_OK(CreateRangeDatasetTensors(testcases, &range_dataset_tensors));
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto &tensor : range_dataset_tensors) {
    inputs.emplace_back(&tensor);
  }
  int num_tensors_per_slice = testcases.size();
  std::unique_ptr<OpKernel> dataset_kernel;
  TF_ASSERT_OK(CreateZipDatasetKernel({DT_INT64}, {{num_tensors_per_slice}},
                                      inputs.size(), &dataset_kernel));
  std::unique_ptr<OpKernelContext> dataset_kernel_ctx;
  TF_ASSERT_OK(CreateZipDatasetContext(dataset_kernel.get(), &inputs,
                                       &dataset_kernel_ctx));
  DatasetBase *dataset;
  TF_ASSERT_OK(
      CreateDataset(dataset_kernel.get(), dataset_kernel_ctx.get(), &dataset));
  core::ScopedUnref scoped_unref(dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(CreateIteratorContext(dataset_kernel_ctx.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));

  DataTypeVector expected_output_dtypes;
  for (int i = 0; i < num_tensors_per_slice; ++i) {
    expected_output_dtypes.emplace_back(expected_outputs[i].dtype());
  }

  TF_EXPECT_OK(
      VerifyTypesMatch(iterator->output_dtypes(), expected_output_dtypes));
}

INSTANTIATE_TEST_CASE_P(ZipDatasetOpTest, IteratorOutputDtypesTest,
                        ::testing::ValuesIn(TestCases));

struct IteratorOutputShapesTest : ZipDatasetOpTest,
                                  ::testing::WithParamInterface<TestParam> {};

TEST_P(IteratorOutputShapesTest, IteratorOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::vector<RangeDatasetParam> testcases = GetParam().testcases;
  std::vector<Tensor> expected_outputs = GetParam().expected_outputs;

  std::vector<Tensor> range_dataset_tensors;
  TF_ASSERT_OK(CreateRangeDatasetTensors(testcases, &range_dataset_tensors));
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto &tensor : range_dataset_tensors) {
    inputs.emplace_back(&tensor);
  }
  int num_tensors_per_slice = testcases.size();
  std::unique_ptr<OpKernel> dataset_kernel;
  TF_ASSERT_OK(CreateZipDatasetKernel({DT_INT64}, {{num_tensors_per_slice}},
                                      inputs.size(), &dataset_kernel));
  std::unique_ptr<OpKernelContext> dataset_kernel_ctx;
  TF_ASSERT_OK(CreateZipDatasetContext(dataset_kernel.get(), &inputs,
                                       &dataset_kernel_ctx));
  DatasetBase *dataset;
  TF_ASSERT_OK(
      CreateDataset(dataset_kernel.get(), dataset_kernel_ctx.get(), &dataset));
  core::ScopedUnref scoped_unref(dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(CreateIteratorContext(dataset_kernel_ctx.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));

  std::vector<PartialTensorShape> expected_output_shapes;
  expected_output_shapes.reserve(num_tensors_per_slice);
  for (int i = 0; i < num_tensors_per_slice; ++i) {
    expected_output_shapes.emplace_back(expected_outputs[i].shape());
  }

  TF_EXPECT_OK(VerifyShapesCompatible(iterator->output_shapes(),
                                      expected_output_shapes));
}

INSTANTIATE_TEST_CASE_P(ZipDatasetOpTest, IteratorOutputShapesTest,
                        ::testing::ValuesIn(TestCases));

TEST_F(ZipDatasetOpTest, IteratorOutputPrefix) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::vector<RangeDatasetParam> testcases(
      {RangeDatasetParam{0, 10, 1}, RangeDatasetParam{10, 20, 1}});
  std::vector<Tensor> range_dataset_tensors;
  TF_ASSERT_OK(CreateRangeDatasetTensors(testcases, &range_dataset_tensors));
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto &tensor : range_dataset_tensors) {
    inputs.emplace_back(&tensor);
  }

  std::unique_ptr<OpKernel> dataset_kernel;
  TF_ASSERT_OK(CreateZipDatasetKernel({DT_INT64}, {{2}}, inputs.size(),
                                      &dataset_kernel));
  std::unique_ptr<OpKernelContext> dataset_kernel_ctx;
  TF_ASSERT_OK(CreateZipDatasetContext(dataset_kernel.get(), &inputs,
                                       &dataset_kernel_ctx));
  DatasetBase *dataset;
  TF_ASSERT_OK(
      CreateDataset(dataset_kernel.get(), dataset_kernel_ctx.get(), &dataset));
  core::ScopedUnref scoped_unref(dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(CreateIteratorContext(dataset_kernel_ctx.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));
  EXPECT_EQ(iterator->prefix(), "Iterator::Zip");
}

struct IteratorRoundtripTest : ZipDatasetOpTest,
                               ::testing::WithParamInterface<TestParam> {};

TEST_P(IteratorRoundtripTest, Roundtrip) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::vector<RangeDatasetParam> testcases = GetParam().testcases;
  std::vector<Tensor> expected_outputs = GetParam().expected_outputs;
  auto expected_outputs_it = expected_outputs.begin();
  std::vector<int> breakpoints = GetParam().breakpoints;

  std::vector<Tensor> range_dataset_tensors;
  TF_ASSERT_OK(CreateRangeDatasetTensors(testcases, &range_dataset_tensors));
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto &tensor : range_dataset_tensors) {
    inputs.emplace_back(&tensor);
  }
  int num_tensors_per_slice = testcases.size();
  std::unique_ptr<OpKernel> dataset_kernel;
  TF_ASSERT_OK(CreateZipDatasetKernel({DT_INT64}, {{num_tensors_per_slice}},
                                      inputs.size(), &dataset_kernel));
  std::unique_ptr<OpKernelContext> dataset_kernel_ctx;
  TF_ASSERT_OK(CreateZipDatasetContext(dataset_kernel.get(), &inputs,
                                       &dataset_kernel_ctx));
  DatasetBase *dataset;
  TF_ASSERT_OK(
      CreateDataset(dataset_kernel.get(), dataset_kernel_ctx.get(), &dataset));
  core::ScopedUnref scoped_unref(dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(CreateIteratorContext(dataset_kernel_ctx.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));

  std::unique_ptr<SerializationContext> serialization_ctx;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_ctx));

  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  int cur_iteration = 0;
  for (int breakpoint : breakpoints) {
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
          EXPECT_NE(expected_outputs_it, expected_outputs.end());
          TF_EXPECT_OK(ExpectEqual(tensor, *expected_outputs_it));
          expected_outputs_it++;
        }
      }
      cur_iteration++;
    }

    if (breakpoint >= dataset->Cardinality()) {
      EXPECT_TRUE(end_of_sequence);
      EXPECT_EQ(expected_outputs_it, expected_outputs.end());
    } else {
      EXPECT_FALSE(end_of_sequence);
    }
  }
}

INSTANTIATE_TEST_CASE_P(ZipDatasetOpTest, IteratorRoundtripTest,
                        ::testing::ValuesIn(TestCases));

}  // namespace
}  // namespace data
}  // namespace tensorflow
