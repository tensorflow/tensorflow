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
  std::vector<RangeDatasetParam> input_range_dataset_params;
  std::vector<Tensor> expected_outputs;
  std::vector<int> breakpoints;
};

TestParam TestCase1() {
  // Test case 1: the input datasets with same number of outputs.
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

TestParam TestCase2() {
  // Test case 2: the input datasets with different number of outputs.
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

class ZipDatasetOpTestHelper : public ZipDatasetOpTest {
 public:
  ~ZipDatasetOpTestHelper() override {
    if (dataset_) dataset_->Unref();
  }

 protected:
  Status CreateDatasetFromTestCase(const TestParam &test_case) {
    std::vector<Tensor> range_dataset_tensors;
    range_dataset_tensors.reserve(test_case.input_range_dataset_params.size());
    TF_RETURN_IF_ERROR(CreateRangeDatasetTensors(
        test_case.input_range_dataset_params, &range_dataset_tensors));
    gtl::InlinedVector<TensorValue, 4> inputs;
    inputs.reserve(range_dataset_tensors.size());
    for (auto &tensor : range_dataset_tensors) {
      inputs.emplace_back(&tensor);
    }
    int num_tensors_per_slice = test_case.input_range_dataset_params.size();
    TF_RETURN_IF_ERROR(CreateZipDatasetKernel({DT_INT64},
                                              {{num_tensors_per_slice}},
                                              inputs.size(), &dataset_kernel_));
    TF_RETURN_IF_ERROR(CreateZipDatasetContext(dataset_kernel_.get(), &inputs,
                                               &dataset_kernel_ctx_));
    TF_RETURN_IF_ERROR(CreateDataset(dataset_kernel_.get(),
                                     dataset_kernel_ctx_.get(), &dataset_));
    return Status::OK();
  }

  Status CreateIteratorFromTestCase(const TestParam &test_case) {
    TF_RETURN_IF_ERROR(CreateDatasetFromTestCase(test_case));
    TF_RETURN_IF_ERROR(
        CreateIteratorContext(dataset_kernel_ctx_.get(), &iterator_ctx_));
    TF_RETURN_IF_ERROR(
        dataset_->MakeIterator(iterator_ctx_.get(), "Iterator", &iterator_));
    return Status::OK();
  }

  std::unique_ptr<OpKernel> dataset_kernel_;
  std::unique_ptr<OpKernelContext> dataset_kernel_ctx_;
  DatasetBase *dataset_ = nullptr;  // owned by this class.
  std::unique_ptr<IteratorContext> iterator_ctx_;
  std::unique_ptr<IteratorBase> iterator_;
};

class ParameterizedDatasetTest
    : public ZipDatasetOpTestHelper,
      public ::testing::WithParamInterface<TestParam> {};

TEST_P(ParameterizedDatasetTest, GetNext) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  const TestParam &test_case = GetParam();
  TF_ASSERT_OK(CreateIteratorFromTestCase(test_case));

  auto expected_outputs_it = test_case.expected_outputs.begin();
  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  while (!end_of_sequence) {
    TF_EXPECT_OK(iterator_->GetNext(iterator_ctx_.get(), &out_tensors,
                                    &end_of_sequence));
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

TEST_F(ZipDatasetOpTestHelper, DatasetName) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  TF_ASSERT_OK(CreateDatasetFromTestCase(TestCase1()));

  EXPECT_EQ(dataset_->type_string(), kOpName);
}

TEST_P(ParameterizedDatasetTest, DatasetOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  const TestParam &test_case = GetParam();
  int num_tensors_per_slice = test_case.input_range_dataset_params.size();
  TF_ASSERT_OK(CreateDatasetFromTestCase(test_case));

  DataTypeVector expected_output_dtypes;
  expected_output_dtypes.reserve(num_tensors_per_slice);
  for (int i = 0; i < num_tensors_per_slice; ++i) {
    expected_output_dtypes.emplace_back(test_case.expected_outputs[i].dtype());
  }

  TF_EXPECT_OK(
      VerifyTypesMatch(dataset_->output_dtypes(), expected_output_dtypes));
}

TEST_P(ParameterizedDatasetTest, DatasetOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  const TestParam &test_case = GetParam();
  int num_tensors_per_slice = test_case.input_range_dataset_params.size();
  TF_ASSERT_OK(CreateDatasetFromTestCase(test_case));

  std::vector<PartialTensorShape> expected_output_shapes;
  expected_output_shapes.reserve(num_tensors_per_slice);
  for (int i = 0; i < num_tensors_per_slice; ++i) {
    expected_output_shapes.emplace_back(test_case.expected_outputs[i].shape());
  }

  TF_EXPECT_OK(VerifyShapesCompatible(dataset_->output_shapes(),
                                      expected_output_shapes));
}

TEST_P(ParameterizedDatasetTest, Cardinality) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  const TestParam &test_case = GetParam();
  int num_tensors_per_slice = test_case.input_range_dataset_params.size();
  TF_ASSERT_OK(CreateDatasetFromTestCase(test_case));

  EXPECT_EQ(dataset_->Cardinality(),
            test_case.expected_outputs.size() / num_tensors_per_slice);
}

TEST_F(ZipDatasetOpTestHelper, DatasetSave) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  TF_ASSERT_OK(CreateDatasetFromTestCase(TestCase1()));

  std::unique_ptr<SerializationContext> serialization_ctx;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_ctx));
  VariantTensorData data;
  VariantTensorDataWriter writer(&data);
  TF_ASSERT_OK(dataset_->Save(serialization_ctx.get(), &writer));
  TF_ASSERT_OK(writer.Flush());
}

TEST_P(ParameterizedDatasetTest, IteratorOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  const TestParam &test_case = GetParam();
  int num_tensors_per_slice = test_case.input_range_dataset_params.size();
  TF_ASSERT_OK(CreateIteratorFromTestCase(test_case));

  DataTypeVector expected_output_dtypes;
  expected_output_dtypes.reserve(num_tensors_per_slice);
  for (int i = 0; i < num_tensors_per_slice; ++i) {
    expected_output_dtypes.emplace_back(test_case.expected_outputs[i].dtype());
  }

  TF_EXPECT_OK(
      VerifyTypesMatch(iterator_->output_dtypes(), expected_output_dtypes));
}

TEST_P(ParameterizedDatasetTest, IteratorOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  const TestParam &test_case = GetParam();
  int num_tensors_per_slice = test_case.input_range_dataset_params.size();
  TF_ASSERT_OK(CreateIteratorFromTestCase(test_case));

  std::vector<PartialTensorShape> expected_output_shapes;
  expected_output_shapes.reserve(num_tensors_per_slice);
  for (int i = 0; i < num_tensors_per_slice; ++i) {
    expected_output_shapes.emplace_back(test_case.expected_outputs[i].shape());
  }

  TF_EXPECT_OK(VerifyShapesCompatible(iterator_->output_shapes(),
                                      expected_output_shapes));
}

TEST_F(ZipDatasetOpTestHelper, IteratorOutputPrefix) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  TF_ASSERT_OK(CreateIteratorFromTestCase(TestCase1()));
  EXPECT_EQ(iterator_->prefix(), "Iterator::Zip");
}

TEST_P(ParameterizedDatasetTest, Roundtrip) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  const TestParam &test_case = GetParam();
  auto expected_outputs_it = test_case.expected_outputs.begin();
  TF_ASSERT_OK(CreateIteratorFromTestCase(test_case));

  std::unique_ptr<SerializationContext> serialization_ctx;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_ctx));

  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  int cur_iteration = 0;
  for (int breakpoint : test_case.breakpoints) {
    VariantTensorData data;
    VariantTensorDataWriter writer(&data);
    TF_EXPECT_OK(iterator_->Save(serialization_ctx.get(), &writer));
    TF_EXPECT_OK(writer.Flush());
    VariantTensorDataReader reader(&data);
    TF_EXPECT_OK(iterator_->Restore(iterator_ctx_.get(), &reader));

    while (cur_iteration < breakpoint) {
      TF_EXPECT_OK(iterator_->GetNext(iterator_ctx_.get(), &out_tensors,
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

    if (breakpoint >= dataset_->Cardinality()) {
      EXPECT_TRUE(end_of_sequence);
      EXPECT_EQ(expected_outputs_it, test_case.expected_outputs.end());
    } else {
      EXPECT_FALSE(end_of_sequence);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    ZipDatasetOpTest, ParameterizedDatasetTest,
    ::testing::ValuesIn(std::vector<TestParam>({TestCase1(), TestCase2()})));

}  // namespace
}  // namespace data
}  // namespace tensorflow
