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
#include "tensorflow/core/kernels/data/tensor_slice_dataset_op.h"

#include "tensorflow/core/kernels/data/dataset_test_base.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "tensor_slice_dataset";

class TensorSliceDatasetOpTest : public DatasetOpsTestBase {
 protected:
  // Creates a new TensorSliceDataset op kernel.
  Status CreateTensorSliceDatasetKernel(
      DataTypeVector dtypes, std::vector<PartialTensorShape> shapes,
      std::unique_ptr<OpKernel> *tensor_dataset_kernel) {
    std::vector<string> components;
    components.reserve(dtypes.size());
    for (int i = 0; i < dtypes.size(); i++) {
      components.emplace_back(
          strings::StrCat(TensorSliceDatasetOp::kComponents, "_", i));
    }

    NodeDef node_def = test::function::NDef(
        kNodeName, name_utils::OpName(TensorSliceDatasetOp::kDatasetType),
        components,
        {{TensorSliceDatasetOp::kToutputTypes, dtypes},
         {TensorSliceDatasetOp::kOutputShapes, shapes}});
    TF_RETURN_IF_ERROR(CreateOpKernel(node_def, tensor_dataset_kernel));
    return Status::OK();
  }

  // Creates a new TensorSliceDataset op kernel context.
  Status CreateTensorSliceDatasetContext(
      OpKernel *const tensor_dataset_kernel,
      gtl::InlinedVector<TensorValue, 4> *inputs,
      std::unique_ptr<OpKernelContext> *context) {
    TF_RETURN_IF_ERROR(CheckOpKernelInput(*tensor_dataset_kernel, *inputs));
    TF_RETURN_IF_ERROR(
        CreateOpKernelContext(tensor_dataset_kernel, inputs, context));
    return Status::OK();
  }
};

struct TestCase {
  std::vector<Tensor> components;
  std::vector<Tensor> expected_outputs;
  std::vector<int> breakpoints;
};

TestCase PlainTensorTestCase() {
  return {/*components*/
          {CreateTensor<int64>(TensorShape({2}), {1, 2}),
           CreateTensor<int64>(TensorShape({2, 2}), {1, 2, 3, 4}),
           CreateTensor<uint32>(TensorShape({2}), {2, 3}),
           CreateTensor<uint32>(TensorShape({2, 2}), {2, 3, 4, 5}),
           CreateTensor<uint64>(TensorShape({2}), {3, 4}),
           CreateTensor<uint64>(TensorShape({2, 2}), {3, 4, 5, 6}),
           CreateTensor<double>(TensorShape({2, 1}), {37.0, 38.0}),
           CreateTensor<tstring>(TensorShape({2, 1}), {"a", "b"})},
          /*expected_outputs*/
          {CreateTensor<int64>(TensorShape({}), {1}),
           CreateTensor<int64>(TensorShape({2}), {1, 2}),
           CreateTensor<uint32>(TensorShape({}), {2}),
           CreateTensor<uint32>(TensorShape({2}), {2, 3}),
           CreateTensor<uint64>(TensorShape({}), {3}),
           CreateTensor<uint64>(TensorShape({2}), {3, 4}),
           CreateTensor<double>(TensorShape({1}), {37.0}),
           CreateTensor<tstring>(TensorShape({1}), {"a"}),
           CreateTensor<int64>(TensorShape({}), {2}),
           CreateTensor<int64>(TensorShape({2}), {3, 4}),
           CreateTensor<uint32>(TensorShape({}), {3}),
           CreateTensor<uint32>(TensorShape({2}), {4, 5}),
           CreateTensor<uint64>(TensorShape({}), {4}),
           CreateTensor<uint64>(TensorShape({2}), {5, 6}),
           CreateTensor<double>(TensorShape({1}), {38.0}),
           CreateTensor<tstring>(TensorShape({1}), {"b"})},
          /*breakpoints*/ {0, 1, 3}};
}

TestCase NestedTensorTestCase() {
  return {
      /*components*/
      {CreateTensor<Variant>(
           TensorShape({2, 1}),
           {CreateTensor<double>(TensorShape({2, 2}), {1.0, 2.0, 3.0, 4.0}),
            CreateTensor<double>(TensorShape({2, 2}), {5.0, 6.0, 7.0, 8.0})}),
       CreateTensor<Variant>(
           TensorShape({2, 1}),
           {CreateTensor<tstring>(TensorShape({1, 2}), {"a", "b"}),
            CreateTensor<tstring>(TensorShape({1, 2}), {"c", "d"})}),
       CreateTensor<int64>(TensorShape({2, 3}), {1, 2, 3, 4, 5, 6})},
      /*expected_outputs*/
      {CreateTensor<Variant>(
           TensorShape({1}),
           {CreateTensor<double>(TensorShape({2, 2}), {1.0, 2.0, 3.0, 4.0})}),
       CreateTensor<Variant>(
           TensorShape({1}),
           {CreateTensor<tstring>(TensorShape({1, 2}), {"a", "b"})}),
       CreateTensor<int64>(TensorShape({3}), {1, 2, 3}),
       CreateTensor<Variant>(
           TensorShape({1}),
           {CreateTensor<double>(TensorShape({2, 2}), {5.0, 6.0, 7.0, 8.0})}),
       CreateTensor<Variant>(
           TensorShape({1}),
           {CreateTensor<tstring>(TensorShape({1, 2}), {"c", "d"})}),
       CreateTensor<int64>(TensorShape({3}), {4, 5, 6})},
      /*breakpoints*/ {0, 1, 2}};
}

class ParameterizedTensorSliceDatasetOpTest
    : public TensorSliceDatasetOpTest,
      public ::testing::WithParamInterface<TestCase> {};

TEST_P(ParameterizedTensorSliceDatasetOpTest, GetNext) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestCase &test_case = GetParam();
  const std::vector<Tensor> &expected_outputs = test_case.expected_outputs;
  std::vector<Tensor> components = test_case.components;
  DataTypeVector dtypes;
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto &component : components) {
    inputs.emplace_back(&component);
    dtypes.emplace_back(component.dtype());
  }
  size_t num_tensors_per_slice = components.size();
  std::vector<PartialTensorShape> shapes;
  shapes.reserve(num_tensors_per_slice);
  for (int i = 0; i < num_tensors_per_slice; ++i) {
    shapes.emplace_back(expected_outputs[i].shape());
  }
  std::unique_ptr<OpKernel> tensor_slice_dataset_kernel;
  TF_ASSERT_OK(CreateTensorSliceDatasetKernel(dtypes, shapes,
                                              &tensor_slice_dataset_kernel));
  std::unique_ptr<OpKernelContext> tensor_slice_dataset_context;
  TF_ASSERT_OK(
      CreateTensorSliceDatasetContext(tensor_slice_dataset_kernel.get(),
                                      &inputs, &tensor_slice_dataset_context));
  DatasetBase *tensor_slice_dataset;
  TF_ASSERT_OK(CreateDataset(tensor_slice_dataset_kernel.get(),
                             tensor_slice_dataset_context.get(),
                             &tensor_slice_dataset));
  core::ScopedUnref scoped_unref(tensor_slice_dataset);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(CreateIteratorContext(tensor_slice_dataset_context.get(),
                                     &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(tensor_slice_dataset->MakeIterator(iterator_context.get(),
                                                  "Iterator", &iterator));
  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  int cur_slice = 0;

  while (!end_of_sequence) {
    TF_EXPECT_OK(iterator->GetNext(iterator_context.get(), &out_tensors,
                                   &end_of_sequence));
    for (int i = 0; i < out_tensors.size(); ++i) {
      EXPECT_LT(i + num_tensors_per_slice * cur_slice, expected_outputs.size());
      if (out_tensors[i].dtype() == DT_VARIANT) {
        // Currently `ExpectEqual()` does not support the variant tensor
        // yet, so we manually cast the variant to numeric/string tensor.
        const Tensor *output = out_tensors[i].scalar<Variant>()().get<Tensor>();
        const Tensor *expected_output =
            expected_outputs[i + num_tensors_per_slice * cur_slice]
                .scalar<Variant>()()
                .get<Tensor>();
        TF_EXPECT_OK(ExpectEqual(*output, *expected_output));
      } else {
        TF_EXPECT_OK(ExpectEqual(
            out_tensors[i],
            expected_outputs[i + num_tensors_per_slice * cur_slice]));
      }
    }
    out_tensors.clear();
    cur_slice++;
  }
}

TEST_F(TensorSliceDatasetOpTest, DatasetNodeName) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestCase &test_case = PlainTensorTestCase();
  const std::vector<Tensor> &expected_outputs = test_case.expected_outputs;
  std::vector<Tensor> components = test_case.components;
  DataTypeVector dtypes;
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto &component : components) {
    inputs.emplace_back(&component);
    dtypes.emplace_back(component.dtype());
  }
  size_t num_tensors_per_slice = components.size();
  std::vector<PartialTensorShape> shapes;
  shapes.reserve(num_tensors_per_slice);
  for (int i = 0; i < num_tensors_per_slice; ++i) {
    shapes.emplace_back(expected_outputs[i].shape());
  }
  std::unique_ptr<OpKernel> tensor_slice_dataset_kernel;
  TF_ASSERT_OK(CreateTensorSliceDatasetKernel(dtypes, shapes,
                                              &tensor_slice_dataset_kernel));
  std::unique_ptr<OpKernelContext> tensor_slice_dataset_context;
  TF_ASSERT_OK(
      CreateTensorSliceDatasetContext(tensor_slice_dataset_kernel.get(),
                                      &inputs, &tensor_slice_dataset_context));
  DatasetBase *tensor_slice_dataset;
  TF_ASSERT_OK(CreateDataset(tensor_slice_dataset_kernel.get(),
                             tensor_slice_dataset_context.get(),
                             &tensor_slice_dataset));
  core::ScopedUnref scoped_unref(tensor_slice_dataset);

  EXPECT_EQ(tensor_slice_dataset->node_name(), kNodeName);
}

TEST_F(TensorSliceDatasetOpTest, DatasetTypeString) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestCase &test_case = PlainTensorTestCase();
  const std::vector<Tensor> &expected_outputs = test_case.expected_outputs;
  std::vector<Tensor> components = test_case.components;
  DataTypeVector dtypes;
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto &component : components) {
    inputs.emplace_back(&component);
    dtypes.emplace_back(component.dtype());
  }
  size_t num_tensors_per_slice = components.size();
  std::vector<PartialTensorShape> shapes;
  shapes.reserve(num_tensors_per_slice);
  for (int i = 0; i < num_tensors_per_slice; ++i) {
    shapes.emplace_back(expected_outputs[i].shape());
  }
  std::unique_ptr<OpKernel> tensor_slice_dataset_kernel;
  TF_ASSERT_OK(CreateTensorSliceDatasetKernel(dtypes, shapes,
                                              &tensor_slice_dataset_kernel));
  std::unique_ptr<OpKernelContext> tensor_slice_dataset_context;
  TF_ASSERT_OK(
      CreateTensorSliceDatasetContext(tensor_slice_dataset_kernel.get(),
                                      &inputs, &tensor_slice_dataset_context));
  DatasetBase *tensor_slice_dataset;
  TF_ASSERT_OK(CreateDataset(tensor_slice_dataset_kernel.get(),
                             tensor_slice_dataset_context.get(),
                             &tensor_slice_dataset));
  core::ScopedUnref scoped_unref(tensor_slice_dataset);

  EXPECT_EQ(tensor_slice_dataset->type_string(),
            name_utils::OpName(TensorSliceDatasetOp::kDatasetType));
}

TEST_P(ParameterizedTensorSliceDatasetOpTest, DatasetOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestCase &test_case = GetParam();
  const std::vector<Tensor> &expected_outputs = test_case.expected_outputs;
  std::vector<Tensor> components = test_case.components;
  DataTypeVector dtypes;
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto &component : components) {
    inputs.emplace_back(&component);
    dtypes.emplace_back(component.dtype());
  }
  size_t num_tensors_per_slice = components.size();
  std::vector<PartialTensorShape> shapes;
  shapes.reserve(num_tensors_per_slice);
  for (int i = 0; i < num_tensors_per_slice; ++i) {
    shapes.emplace_back(expected_outputs[i].shape());
  }
  std::unique_ptr<OpKernel> tensor_slice_dataset_kernel;
  TF_ASSERT_OK(CreateTensorSliceDatasetKernel(dtypes, shapes,
                                              &tensor_slice_dataset_kernel));
  std::unique_ptr<OpKernelContext> tensor_slice_dataset_context;
  TF_ASSERT_OK(
      CreateTensorSliceDatasetContext(tensor_slice_dataset_kernel.get(),
                                      &inputs, &tensor_slice_dataset_context));
  DatasetBase *tensor_slice_dataset;
  TF_ASSERT_OK(CreateDataset(tensor_slice_dataset_kernel.get(),
                             tensor_slice_dataset_context.get(),
                             &tensor_slice_dataset));
  core::ScopedUnref scoped_unref(tensor_slice_dataset);

  const DataTypeVector produced_output_dtypes =
      tensor_slice_dataset->output_dtypes();
  EXPECT_EQ(produced_output_dtypes.size(), num_tensors_per_slice);
  for (int i = 0; i < num_tensors_per_slice; ++i) {
    EXPECT_EQ(produced_output_dtypes[i], expected_outputs[i].dtype());
  }
}

TEST_P(ParameterizedTensorSliceDatasetOpTest, DatasetOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestCase &test_case = GetParam();
  const std::vector<Tensor> &expected_outputs = test_case.expected_outputs;
  std::vector<Tensor> components = test_case.components;
  DataTypeVector dtypes;
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto &component : components) {
    inputs.emplace_back(&component);
    dtypes.emplace_back(component.dtype());
  }
  size_t num_tensors_per_slice = components.size();
  std::vector<PartialTensorShape> shapes;
  shapes.reserve(num_tensors_per_slice);
  for (int i = 0; i < num_tensors_per_slice; ++i) {
    shapes.emplace_back(expected_outputs[i].shape());
  }
  std::unique_ptr<OpKernel> tensor_slice_dataset_kernel;
  TF_ASSERT_OK(CreateTensorSliceDatasetKernel(dtypes, shapes,
                                              &tensor_slice_dataset_kernel));
  std::unique_ptr<OpKernelContext> tensor_slice_dataset_context;
  TF_ASSERT_OK(
      CreateTensorSliceDatasetContext(tensor_slice_dataset_kernel.get(),
                                      &inputs, &tensor_slice_dataset_context));
  DatasetBase *tensor_slice_dataset;
  TF_ASSERT_OK(CreateDataset(tensor_slice_dataset_kernel.get(),
                             tensor_slice_dataset_context.get(),
                             &tensor_slice_dataset));
  core::ScopedUnref scoped_unref(tensor_slice_dataset);

  const std::vector<PartialTensorShape> produced_output_shapes =
      tensor_slice_dataset->output_shapes();
  std::vector<PartialTensorShape> expected_output_shapes;
  EXPECT_EQ(produced_output_shapes.size(), num_tensors_per_slice);
  for (int i = 0; i < num_tensors_per_slice; ++i) {
    EXPECT_TRUE(
        produced_output_shapes[i].IsIdenticalTo(expected_outputs[i].shape()));
  }
}

TEST_P(ParameterizedTensorSliceDatasetOpTest, Cardinality) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestCase &test_case = GetParam();
  const std::vector<Tensor> &expected_outputs = test_case.expected_outputs;
  std::vector<Tensor> components = test_case.components;
  DataTypeVector dtypes;
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto &component : components) {
    inputs.emplace_back(&component);
    dtypes.emplace_back(component.dtype());
  }
  size_t num_tensors_per_slice = components.size();
  std::vector<PartialTensorShape> shapes;
  shapes.reserve(num_tensors_per_slice);
  for (int i = 0; i < num_tensors_per_slice; ++i) {
    shapes.emplace_back(expected_outputs[i].shape());
  }
  std::unique_ptr<OpKernel> tensor_slice_dataset_kernel;
  TF_ASSERT_OK(CreateTensorSliceDatasetKernel(dtypes, shapes,
                                              &tensor_slice_dataset_kernel));
  std::unique_ptr<OpKernelContext> tensor_slice_dataset_context;
  TF_ASSERT_OK(
      CreateTensorSliceDatasetContext(tensor_slice_dataset_kernel.get(),
                                      &inputs, &tensor_slice_dataset_context));
  DatasetBase *tensor_slice_dataset;
  TF_ASSERT_OK(CreateDataset(tensor_slice_dataset_kernel.get(),
                             tensor_slice_dataset_context.get(),
                             &tensor_slice_dataset));
  core::ScopedUnref scoped_unref(tensor_slice_dataset);

  EXPECT_EQ(tensor_slice_dataset->Cardinality(), inputs[0].tensor->dim_size(0));
}

TEST_P(ParameterizedTensorSliceDatasetOpTest, IteratorOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestCase &test_case = GetParam();
  const std::vector<Tensor> &expected_outputs = test_case.expected_outputs;
  std::vector<Tensor> components = test_case.components;
  DataTypeVector dtypes;
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto &component : components) {
    inputs.emplace_back(&component);
    dtypes.emplace_back(component.dtype());
  }
  size_t num_tensors_per_slice = components.size();
  std::vector<PartialTensorShape> shapes;
  shapes.reserve(num_tensors_per_slice);
  for (int i = 0; i < num_tensors_per_slice; ++i) {
    shapes.emplace_back(expected_outputs[i].shape());
  }
  std::unique_ptr<OpKernel> tensor_slice_dataset_kernel;
  TF_ASSERT_OK(CreateTensorSliceDatasetKernel(dtypes, shapes,
                                              &tensor_slice_dataset_kernel));
  std::unique_ptr<OpKernelContext> tensor_slice_dataset_context;
  TF_ASSERT_OK(
      CreateTensorSliceDatasetContext(tensor_slice_dataset_kernel.get(),
                                      &inputs, &tensor_slice_dataset_context));
  DatasetBase *tensor_slice_dataset;
  TF_ASSERT_OK(CreateDataset(tensor_slice_dataset_kernel.get(),
                             tensor_slice_dataset_context.get(),
                             &tensor_slice_dataset));
  core::ScopedUnref scoped_unref(tensor_slice_dataset);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(CreateIteratorContext(tensor_slice_dataset_context.get(),
                                     &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(tensor_slice_dataset->MakeIterator(iterator_context.get(),
                                                  "Iterator", &iterator));
  const DataTypeVector produced_output_dtypes = iterator->output_dtypes();

  EXPECT_EQ(produced_output_dtypes.size(), num_tensors_per_slice);
  for (int i = 0; i < num_tensors_per_slice; ++i) {
    EXPECT_EQ(produced_output_dtypes[i], expected_outputs[i].dtype());
  }
}

TEST_P(ParameterizedTensorSliceDatasetOpTest, IteratorOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestCase &test_case = GetParam();
  const std::vector<Tensor> &expected_outputs = test_case.expected_outputs;
  std::vector<Tensor> components = test_case.components;
  DataTypeVector dtypes;
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto &component : components) {
    inputs.emplace_back(&component);
    dtypes.emplace_back(component.dtype());
  }
  size_t num_tensors_per_slice = components.size();
  std::vector<PartialTensorShape> shapes;
  shapes.reserve(num_tensors_per_slice);
  for (int i = 0; i < num_tensors_per_slice; ++i) {
    shapes.emplace_back(expected_outputs[i].shape());
  }
  std::unique_ptr<OpKernel> tensor_slice_dataset_kernel;
  TF_ASSERT_OK(CreateTensorSliceDatasetKernel(dtypes, shapes,
                                              &tensor_slice_dataset_kernel));
  std::unique_ptr<OpKernelContext> tensor_slice_dataset_context;
  TF_ASSERT_OK(
      CreateTensorSliceDatasetContext(tensor_slice_dataset_kernel.get(),
                                      &inputs, &tensor_slice_dataset_context));
  DatasetBase *tensor_slice_dataset;
  TF_ASSERT_OK(CreateDataset(tensor_slice_dataset_kernel.get(),
                             tensor_slice_dataset_context.get(),
                             &tensor_slice_dataset));
  core::ScopedUnref scoped_unref(tensor_slice_dataset);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(CreateIteratorContext(tensor_slice_dataset_context.get(),
                                     &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(tensor_slice_dataset->MakeIterator(iterator_context.get(),
                                                  "Iterator", &iterator));
  const std::vector<PartialTensorShape> produced_output_shapes =
      iterator->output_shapes();
  EXPECT_EQ(produced_output_shapes.size(), num_tensors_per_slice);
  for (int i = 0; i < num_tensors_per_slice; ++i) {
    EXPECT_TRUE(
        produced_output_shapes[i].IsIdenticalTo(expected_outputs[i].shape()));
  }
}

TEST_F(TensorSliceDatasetOpTest, IteratorOutputPrefix) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestCase &test_case = PlainTensorTestCase();
  const std::vector<Tensor> &expected_outputs = test_case.expected_outputs;
  std::vector<Tensor> components = test_case.components;
  DataTypeVector dtypes;
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto &component : components) {
    inputs.emplace_back(&component);
    dtypes.emplace_back(component.dtype());
  }
  size_t num_tensors_per_slice = components.size();
  std::vector<PartialTensorShape> shapes;
  shapes.reserve(num_tensors_per_slice);
  for (int i = 0; i < num_tensors_per_slice; ++i) {
    shapes.emplace_back(expected_outputs[i].shape());
  }
  std::unique_ptr<OpKernel> tensor_slice_dataset_kernel;
  TF_ASSERT_OK(CreateTensorSliceDatasetKernel(dtypes, shapes,
                                              &tensor_slice_dataset_kernel));
  std::unique_ptr<OpKernelContext> tensor_slice_dataset_context;
  TF_ASSERT_OK(
      CreateTensorSliceDatasetContext(tensor_slice_dataset_kernel.get(),
                                      &inputs, &tensor_slice_dataset_context));
  DatasetBase *tensor_slice_dataset;
  TF_ASSERT_OK(CreateDataset(tensor_slice_dataset_kernel.get(),
                             tensor_slice_dataset_context.get(),
                             &tensor_slice_dataset));
  core::ScopedUnref scoped_unref(tensor_slice_dataset);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(CreateIteratorContext(tensor_slice_dataset_context.get(),
                                     &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(tensor_slice_dataset->MakeIterator(iterator_context.get(),
                                                  "Iterator", &iterator));
  EXPECT_EQ(iterator->prefix(),
            name_utils::IteratorPrefix(TensorSliceDatasetOp::kDatasetType,
                                       "Iterator"));
}

TEST_P(ParameterizedTensorSliceDatasetOpTest, Roundtrip) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestCase &test_case = GetParam();
  const std::vector<Tensor> &expected_outputs = test_case.expected_outputs;
  std::vector<Tensor> components = test_case.components;
  DataTypeVector dtypes;
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto &component : components) {
    inputs.emplace_back(&component);
    dtypes.emplace_back(component.dtype());
  }
  size_t num_tensors_per_slice = components.size();
  std::vector<PartialTensorShape> shapes;
  shapes.reserve(num_tensors_per_slice);
  for (int i = 0; i < num_tensors_per_slice; ++i) {
    shapes.emplace_back(expected_outputs[i].shape());
  }
  std::unique_ptr<OpKernel> tensor_slice_dataset_kernel;
  TF_ASSERT_OK(CreateTensorSliceDatasetKernel(dtypes, shapes,
                                              &tensor_slice_dataset_kernel));
  std::unique_ptr<OpKernelContext> tensor_slice_dataset_context;
  TF_ASSERT_OK(
      CreateTensorSliceDatasetContext(tensor_slice_dataset_kernel.get(),
                                      &inputs, &tensor_slice_dataset_context));
  DatasetBase *tensor_slice_dataset;
  TF_ASSERT_OK(CreateDataset(tensor_slice_dataset_kernel.get(),
                             tensor_slice_dataset_context.get(),
                             &tensor_slice_dataset));
  core::ScopedUnref scoped_unref(tensor_slice_dataset);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(CreateIteratorContext(tensor_slice_dataset_context.get(),
                                     &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(tensor_slice_dataset->MakeIterator(iterator_context.get(),
                                                  "Iterator", &iterator));
  std::unique_ptr<SerializationContext> serialization_context;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_context));

  int cur_iteration = 0;
  bool end_of_sequence = false;
  int64 num_slices = inputs[0].tensor->dim_size(0);
  std::vector<Tensor> out_tensors;
  const std::vector<int> &breakpoints = test_case.breakpoints;
  for (int breakpoint : breakpoints) {
    while (cur_iteration < breakpoint) {
      TF_EXPECT_OK(iterator->GetNext(iterator_context.get(), &out_tensors,
                                     &end_of_sequence));
      cur_iteration++;
    }

    if (breakpoint == 0) {
      EXPECT_FALSE(end_of_sequence);
    } else if (breakpoint <= num_slices) {
      for (int i = 0; i < out_tensors.size(); ++i) {
        if (out_tensors[i].dtype() == DT_VARIANT) {
          const Tensor *output =
              out_tensors[i].scalar<Variant>()().get<Tensor>();
          const Tensor *expected_output =
              expected_outputs[i + num_tensors_per_slice * (cur_iteration - 1)]
                  .scalar<Variant>()()
                  .get<Tensor>();
          TF_EXPECT_OK(ExpectEqual(*output, *expected_output));
        } else {
          TF_EXPECT_OK(ExpectEqual(
              out_tensors[i], expected_outputs[i + num_tensors_per_slice *
                                                       (cur_iteration - 1)]));
        }
      }
    } else {
      EXPECT_TRUE(end_of_sequence);
    }

    VariantTensorData data;
    VariantTensorDataWriter writer(&data);
    TF_ASSERT_OK(iterator->Save(serialization_context.get(), &writer));
    TF_ASSERT_OK(writer.Flush());
    VariantTensorDataReader reader(&data);
    TF_EXPECT_OK(RestoreIterator(iterator_context.get(), &reader, "Iterator",
                                 *tensor_slice_dataset, &iterator));
  }
}

INSTANTIATE_TEST_SUITE_P(TensorSliceDatasetOpTest,
                         ParameterizedTensorSliceDatasetOpTest,
                         ::testing::ValuesIn(std::vector<TestCase>(
                             {PlainTensorTestCase(), NestedTensorTestCase()})));

}  // namespace
}  // namespace data
}  // namespace tensorflow
