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

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/kernels/data/dataset_test_base.h"
#include "tensorflow/core/kernels/data/dataset_utils.h"
#include "tensorflow/core/kernels/data/iterator_ops.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "tensor_dataset";
constexpr char kOpName[] = "TensorDataset";

class TensorDatasetOpTest : public DatasetOpsTestBase {
 protected:
  // Creates a new TensorDataset op kernel.
  Status CreateTensorDatasetKernel(
      DataTypeVector dtypes, std::vector<PartialTensorShape> shapes,
      std::unique_ptr<OpKernel> *tensor_dataset_kernel) {
    std::vector<string> components;
    components.reserve(dtypes.size());
    for (int i = 0; i < dtypes.size(); i++) {
      components.emplace_back(strings::StrCat("component_", i));
    }
    node_def_ = test::function::NDef(
        kNodeName, kOpName, components,
        {{"Toutput_types", dtypes}, {"output_shapes", shapes}});
    TF_RETURN_IF_ERROR(CreateOpKernel(node_def_, tensor_dataset_kernel));
    return Status::OK();
  }

  // Creates a new TensorDataset op kernel context.
  Status CreateTensorDatasetContext(OpKernel *const tensor_dataset_kernel,
                                    gtl::InlinedVector<TensorValue, 4> *inputs,
                                    std::unique_ptr<OpKernelContext> *context) {
    TF_RETURN_IF_ERROR(CheckOpKernelInput(*tensor_dataset_kernel, *inputs));
    TF_RETURN_IF_ERROR(
        CreateOpKernelContext(tensor_dataset_kernel, inputs, context));
    return Status::OK();
  }

 private:
  NodeDef node_def_;
};

struct TestCase {
  std::vector<Tensor> components;
  std::vector<Tensor> expected_outputs;
  DataTypeVector expected_output_dtypes;
  std::vector<PartialTensorShape> expected_output_shapes;
  int64 expected_cardinality;
  std::vector<int> breakpoints;
};

// Test case 1: test a dataset that represents a single tuple of plain tensors.
TestCase PlainTensorsTestCase() {
  return {
      /*components*/
      {DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {1}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape({1, 3}), {1, 2, 3}),
       DatasetOpsTestBase::CreateTensor<double>(TensorShape({}), {37.0}),
       DatasetOpsTestBase::CreateTensor<string>(TensorShape({1, 2}),
                                                {"a", "b"})},
      /*expected_outputs*/
      {DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {1}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape({1, 3}), {1, 2, 3}),
       DatasetOpsTestBase::CreateTensor<double>(TensorShape({}), {37.0}),
       DatasetOpsTestBase::CreateTensor<string>(TensorShape({1, 2}),
                                                {"a", "b"})},
      /*expected_output_dtypes*/
      {DT_INT64, DT_INT64, DT_DOUBLE, DT_STRING},
      /*expected_output_shapes*/
      {PartialTensorShape({}), PartialTensorShape({1, 3}),
       PartialTensorShape({}), PartialTensorShape({1, 2})},
      /*expected_cardinality*/ 1,
      /*breakpoints*/ {0, 1, 2}};
}

// Test case 2: test a dataset that represents a tuple of nested tensors.
TestCase NestedTensorsTestCase() {
  return {
      /*components*/
      {DatasetOpsTestBase::CreateTensor<Variant>(
           TensorShape({}), {DatasetOpsTestBase::CreateTensor<double>(
                                TensorShape({2, 2}), {1.0, 2.0, 3.0, 4.0})}),
       DatasetOpsTestBase::CreateTensor<Variant>(
           TensorShape({}), {DatasetOpsTestBase::CreateTensor<string>(
                                TensorShape({1, 2}), {"a", "b"})}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape({1, 3}), {1, 2, 3})},
      /*expected_outputs*/
      {DatasetOpsTestBase::CreateTensor<Variant>(
           TensorShape({}), {DatasetOpsTestBase::CreateTensor<double>(
                                TensorShape({2, 2}), {1.0, 2.0, 3.0, 4.0})}),
       DatasetOpsTestBase::CreateTensor<Variant>(
           TensorShape({}), {DatasetOpsTestBase::CreateTensor<string>(
                                TensorShape({1, 2}), {"a", "b"})}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape({1, 3}), {1, 2, 3})},
      /*expected_output_dtypes*/
      {DT_VARIANT, DT_VARIANT, DT_INT64},
      /*expected_output_shapes*/
      {PartialTensorShape({}), PartialTensorShape({}),
       PartialTensorShape({1, 3})},
      /*expected_cardinality*/ 1,
      /*breakpoints*/ {0, 1, 2}};
}

class ParametrizedTensorDatasetOpTest
    : public TensorDatasetOpTest,
      public ::testing::WithParamInterface<TestCase> {};

TEST_P(ParametrizedTensorDatasetOpTest, GetNext) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestCase &test_case = GetParam();
  std::vector<Tensor> components = test_case.components;
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto &component : components) {
    inputs.push_back(&component);
  }
  std::unique_ptr<OpKernel> tensor_dataset_kernel;
  TF_ASSERT_OK(CreateTensorDatasetKernel(test_case.expected_output_dtypes,
                                         test_case.expected_output_shapes,
                                         &tensor_dataset_kernel));
  std::unique_ptr<OpKernelContext> tensor_dataset_context;
  TF_ASSERT_OK(CreateTensorDatasetContext(tensor_dataset_kernel.get(), &inputs,
                                          &tensor_dataset_context));
  DatasetBase *tensor_dataset;
  TF_ASSERT_OK(CreateDataset(tensor_dataset_kernel.get(),
                             tensor_dataset_context.get(), &tensor_dataset));
  core::ScopedUnref scoped_unref(tensor_dataset);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(
      CreateIteratorContext(tensor_dataset_context.get(), &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(tensor_dataset->MakeIterator(iterator_context.get(), "Iterator",
                                            &iterator));
  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  while (!end_of_sequence) {
    TF_EXPECT_OK(iterator->GetNext(iterator_context.get(), &out_tensors,
                                   &end_of_sequence));
  }
  EXPECT_EQ(out_tensors.size(), test_case.expected_outputs.size());
  for (int i = 0; i < out_tensors.size(); ++i) {
    if (out_tensors[i].dtype() == DT_VARIANT) {
      // Currently `ExpectEqual()` does not support the variant tensor
      // yet, so we manually cast the variant to numeric/string tensor.
      const Tensor *output = out_tensors[i].scalar<Variant>()().get<Tensor>();
      const Tensor *expected_output =
          test_case.expected_outputs[i].scalar<Variant>()().get<Tensor>();
      TF_EXPECT_OK(ExpectEqual(*output, *expected_output));
    } else {
      TF_EXPECT_OK(ExpectEqual(out_tensors[i], test_case.expected_outputs[i]));
    }
  }
}

TEST_F(TensorDatasetOpTest, DatasetTypeString) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestCase &test_case = PlainTensorsTestCase();
  std::vector<Tensor> components = test_case.components;
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto &component : components) {
    inputs.push_back(&component);
  }
  std::unique_ptr<OpKernel> tensor_dataset_kernel;
  TF_ASSERT_OK(CreateTensorDatasetKernel(test_case.expected_output_dtypes,
                                         test_case.expected_output_shapes,
                                         &tensor_dataset_kernel));
  std::unique_ptr<OpKernelContext> tensor_dataset_context;
  TF_ASSERT_OK(CreateTensorDatasetContext(tensor_dataset_kernel.get(), &inputs,
                                          &tensor_dataset_context));
  DatasetBase *tensor_dataset;
  TF_ASSERT_OK(CreateDataset(tensor_dataset_kernel.get(),
                             tensor_dataset_context.get(), &tensor_dataset));
  core::ScopedUnref scoped_unref(tensor_dataset);

  EXPECT_EQ(tensor_dataset->type_string(), kOpName);
}

TEST_F(TensorDatasetOpTest, DatasetNodeName) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestCase &test_case = PlainTensorsTestCase();
  std::vector<Tensor> components = test_case.components;
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto &component : components) {
    inputs.push_back(&component);
  }
  std::unique_ptr<OpKernel> tensor_dataset_kernel;
  TF_ASSERT_OK(CreateTensorDatasetKernel(test_case.expected_output_dtypes,
                                         test_case.expected_output_shapes,
                                         &tensor_dataset_kernel));
  std::unique_ptr<OpKernelContext> tensor_dataset_context;
  TF_ASSERT_OK(CreateTensorDatasetContext(tensor_dataset_kernel.get(), &inputs,
                                          &tensor_dataset_context));
  DatasetBase *tensor_dataset;
  TF_ASSERT_OK(CreateDataset(tensor_dataset_kernel.get(),
                             tensor_dataset_context.get(), &tensor_dataset));
  core::ScopedUnref scoped_unref(tensor_dataset);

  EXPECT_EQ(tensor_dataset->node_name(), kNodeName);
}

TEST_F(TensorDatasetOpTest, DatasetOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestCase &test_case = PlainTensorsTestCase();
  std::vector<Tensor> components = test_case.components;
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto &component : components) {
    inputs.push_back(&component);
  }
  std::unique_ptr<OpKernel> tensor_dataset_kernel;
  TF_ASSERT_OK(CreateTensorDatasetKernel(test_case.expected_output_dtypes,
                                         test_case.expected_output_shapes,
                                         &tensor_dataset_kernel));
  std::unique_ptr<OpKernelContext> tensor_dataset_context;
  TF_ASSERT_OK(CreateTensorDatasetContext(tensor_dataset_kernel.get(), &inputs,
                                          &tensor_dataset_context));
  DatasetBase *tensor_dataset;
  TF_ASSERT_OK(CreateDataset(tensor_dataset_kernel.get(),
                             tensor_dataset_context.get(), &tensor_dataset));
  core::ScopedUnref scoped_unref(tensor_dataset);

  EXPECT_EQ(tensor_dataset->output_dtypes(), test_case.expected_output_dtypes);
}

TEST_F(TensorDatasetOpTest, DatasetOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestCase &test_case = PlainTensorsTestCase();
  std::vector<Tensor> components = test_case.components;
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto &component : components) {
    inputs.push_back(&component);
  }
  std::unique_ptr<OpKernel> tensor_dataset_kernel;
  TF_ASSERT_OK(CreateTensorDatasetKernel(test_case.expected_output_dtypes,
                                         test_case.expected_output_shapes,
                                         &tensor_dataset_kernel));
  std::unique_ptr<OpKernelContext> tensor_dataset_context;
  TF_ASSERT_OK(CreateTensorDatasetContext(tensor_dataset_kernel.get(), &inputs,
                                          &tensor_dataset_context));
  DatasetBase *tensor_dataset;
  TF_ASSERT_OK(CreateDataset(tensor_dataset_kernel.get(),
                             tensor_dataset_context.get(), &tensor_dataset));
  core::ScopedUnref scoped_unref(tensor_dataset);

  EXPECT_EQ(tensor_dataset->output_shapes().size(),
            test_case.expected_output_shapes.size());
  for (int i = 0; i < test_case.expected_output_shapes.size(); i++) {
    EXPECT_TRUE(test_case.expected_output_shapes[i].IsIdenticalTo(
        tensor_dataset->output_shapes()[i]));
  }
}

TEST_F(TensorDatasetOpTest, Cardinality) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestCase &test_case = PlainTensorsTestCase();
  std::vector<Tensor> components = test_case.components;
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto &component : components) {
    inputs.push_back(&component);
  }
  std::unique_ptr<OpKernel> tensor_dataset_kernel;
  TF_ASSERT_OK(CreateTensorDatasetKernel(test_case.expected_output_dtypes,
                                         test_case.expected_output_shapes,
                                         &tensor_dataset_kernel));
  std::unique_ptr<OpKernelContext> tensor_dataset_context;
  TF_ASSERT_OK(CreateTensorDatasetContext(tensor_dataset_kernel.get(), &inputs,
                                          &tensor_dataset_context));
  DatasetBase *tensor_dataset;
  TF_ASSERT_OK(CreateDataset(tensor_dataset_kernel.get(),
                             tensor_dataset_context.get(), &tensor_dataset));
  core::ScopedUnref scoped_unref(tensor_dataset);

  EXPECT_EQ(tensor_dataset->Cardinality(), test_case.expected_cardinality);
}

TEST_P(ParametrizedTensorDatasetOpTest, DatasetSave) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestCase &test_case = GetParam();
  std::vector<Tensor> components = test_case.components;
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto &component : components) {
    inputs.push_back(&component);
  }
  std::unique_ptr<OpKernel> tensor_dataset_kernel;
  TF_ASSERT_OK(CreateTensorDatasetKernel(test_case.expected_output_dtypes,
                                         test_case.expected_output_shapes,
                                         &tensor_dataset_kernel));
  std::unique_ptr<OpKernelContext> tensor_dataset_context;
  TF_ASSERT_OK(CreateTensorDatasetContext(tensor_dataset_kernel.get(), &inputs,
                                          &tensor_dataset_context));
  DatasetBase *tensor_dataset;
  TF_ASSERT_OK(CreateDataset(tensor_dataset_kernel.get(),
                             tensor_dataset_context.get(), &tensor_dataset));
  core::ScopedUnref scoped_unref(tensor_dataset);

  std::unique_ptr<SerializationContext> serialization_context;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_context));
  VariantTensorData data;
  VariantTensorDataWriter writer(&data);
  TF_ASSERT_OK(tensor_dataset->Save(serialization_context.get(), &writer));
  TF_ASSERT_OK(writer.Flush());
}

TEST_P(ParametrizedTensorDatasetOpTest, IteratorOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestCase &test_case = GetParam();
  std::vector<Tensor> components = test_case.components;
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto &component : components) {
    inputs.push_back(&component);
  }
  std::unique_ptr<OpKernel> tensor_dataset_kernel;
  TF_ASSERT_OK(CreateTensorDatasetKernel(test_case.expected_output_dtypes,
                                         test_case.expected_output_shapes,
                                         &tensor_dataset_kernel));
  std::unique_ptr<OpKernelContext> tensor_dataset_context;
  TF_ASSERT_OK(CreateTensorDatasetContext(tensor_dataset_kernel.get(), &inputs,
                                          &tensor_dataset_context));
  DatasetBase *tensor_dataset;
  TF_ASSERT_OK(CreateDataset(tensor_dataset_kernel.get(),
                             tensor_dataset_context.get(), &tensor_dataset));
  core::ScopedUnref scoped_unref(tensor_dataset);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(
      CreateIteratorContext(tensor_dataset_context.get(), &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(tensor_dataset->MakeIterator(iterator_context.get(), "Iterator",
                                            &iterator));
  EXPECT_EQ(iterator->output_dtypes(), test_case.expected_output_dtypes);
}

TEST_P(ParametrizedTensorDatasetOpTest, IteratorOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestCase &test_case = GetParam();
  std::vector<Tensor> components = test_case.components;
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto &component : components) {
    inputs.push_back(&component);
  }
  std::unique_ptr<OpKernel> tensor_dataset_kernel;
  TF_ASSERT_OK(CreateTensorDatasetKernel(test_case.expected_output_dtypes,
                                         test_case.expected_output_shapes,
                                         &tensor_dataset_kernel));
  std::unique_ptr<OpKernelContext> tensor_dataset_context;
  TF_ASSERT_OK(CreateTensorDatasetContext(tensor_dataset_kernel.get(), &inputs,
                                          &tensor_dataset_context));
  DatasetBase *tensor_dataset;
  TF_ASSERT_OK(CreateDataset(tensor_dataset_kernel.get(),
                             tensor_dataset_context.get(), &tensor_dataset));
  core::ScopedUnref scoped_unref(tensor_dataset);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(
      CreateIteratorContext(tensor_dataset_context.get(), &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(tensor_dataset->MakeIterator(iterator_context.get(), "Iterator",
                                            &iterator));

  EXPECT_EQ(iterator->output_shapes().size(),
            test_case.expected_output_shapes.size());
  for (int i = 0; i < test_case.expected_output_shapes.size(); ++i) {
    EXPECT_TRUE(test_case.expected_output_shapes[i].IsIdenticalTo(
        iterator->output_shapes()[i]));
  }
}

TEST_F(TensorDatasetOpTest, IteratorOutputPrefix) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestCase &test_case = PlainTensorsTestCase();
  std::vector<Tensor> components = test_case.components;
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto &component : components) {
    inputs.push_back(&component);
  }
  std::unique_ptr<OpKernel> tensor_dataset_kernel;
  TF_ASSERT_OK(CreateTensorDatasetKernel(test_case.expected_output_dtypes,
                                         test_case.expected_output_shapes,
                                         &tensor_dataset_kernel));
  std::unique_ptr<OpKernelContext> tensor_dataset_context;
  TF_ASSERT_OK(CreateTensorDatasetContext(tensor_dataset_kernel.get(), &inputs,
                                          &tensor_dataset_context));
  DatasetBase *tensor_dataset;
  TF_ASSERT_OK(CreateDataset(tensor_dataset_kernel.get(),
                             tensor_dataset_context.get(), &tensor_dataset));
  core::ScopedUnref scoped_unref(tensor_dataset);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(
      CreateIteratorContext(tensor_dataset_context.get(), &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(tensor_dataset->MakeIterator(iterator_context.get(), "Iterator",
                                            &iterator));

  EXPECT_EQ(iterator->prefix(), "Iterator::FromTensor");
}

TEST_P(ParametrizedTensorDatasetOpTest, Roundtrip) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestCase &test_case = GetParam();
  std::vector<Tensor> components = test_case.components;
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto &component : components) {
    inputs.push_back(&component);
  }
  std::unique_ptr<OpKernel> tensor_dataset_kernel;
  TF_ASSERT_OK(CreateTensorDatasetKernel(test_case.expected_output_dtypes,
                                         test_case.expected_output_shapes,
                                         &tensor_dataset_kernel));
  std::unique_ptr<OpKernelContext> tensor_dataset_context;
  TF_ASSERT_OK(CreateTensorDatasetContext(tensor_dataset_kernel.get(), &inputs,
                                          &tensor_dataset_context));
  DatasetBase *tensor_dataset;
  TF_ASSERT_OK(CreateDataset(tensor_dataset_kernel.get(),
                             tensor_dataset_context.get(), &tensor_dataset));
  core::ScopedUnref scoped_unref(tensor_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(tensor_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      tensor_dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));

  std::unique_ptr<SerializationContext> serialization_ctx;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_ctx));
  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  int cur_iteration = 0;
  const std::vector<int> &breakpoints = test_case.breakpoints;
  for (int breakpoint : breakpoints) {
    VariantTensorData data;
    VariantTensorDataWriter writer(&data);
    TF_EXPECT_OK(iterator->Save(serialization_ctx.get(), &writer));
    TF_EXPECT_OK(writer.Flush());
    VariantTensorDataReader reader(&data);
    TF_EXPECT_OK(iterator->Restore(iterator_ctx.get(), &reader));

    while (cur_iteration <= breakpoint) {
      TF_EXPECT_OK(iterator->GetNext(iterator_ctx.get(), &out_tensors,
                                     &end_of_sequence));
      if (!end_of_sequence) {
        EXPECT_EQ(out_tensors.size(), test_case.expected_outputs.size());
        for (int i = 0; i < out_tensors.size(); ++i) {
          if (out_tensors[i].dtype() == DT_VARIANT) {
            // Currently `ExpectEqual()` does not support the variant tensor
            // yet, so we manually cast the variant to numeric/string tensor.
            const Tensor *output =
                out_tensors[i].scalar<Variant>()().get<Tensor>();
            const Tensor *expected_output =
                test_case.expected_outputs[i].scalar<Variant>()().get<Tensor>();
            TF_EXPECT_OK(ExpectEqual(*output, *expected_output));
          } else {
            TF_EXPECT_OK(
                ExpectEqual(out_tensors[i], test_case.expected_outputs[i]));
          }
        }
      }
      cur_iteration++;
    }

    if (breakpoint >= test_case.expected_cardinality) {
      EXPECT_TRUE(end_of_sequence);
    } else {
      EXPECT_FALSE(end_of_sequence);
    }
  }
}

INSTANTIATE_TEST_CASE_P(
    TensorDatasetOpTest, ParametrizedTensorDatasetOpTest,
    ::testing::ValuesIn(std::vector<TestCase>({PlainTensorsTestCase(),
                                               NestedTensorsTestCase()})));

}  // namespace
}  // namespace data
}  // namespace tensorflow
