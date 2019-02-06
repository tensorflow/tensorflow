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
    TF_CHECK_OK(CreateOpKernel(node_def_, tensor_dataset_kernel));
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

template <typename T>
Tensor CreateTensor(TensorShape input_shape,
                    const gtl::ArraySlice<T> &input_data) {
  Tensor tensor(DataTypeToEnum<T>::value, input_shape);
  test::FillValues<T>(&tensor, input_data);
  return tensor;
}

struct TestParam {
  std::vector<Tensor> components;
} TestCases[] = {
    {{CreateTensor<int64>(TensorShape({}), {1}),
      CreateTensor<int64>(TensorShape({1, 3}), {1, 2, 3}),
      CreateTensor<double>(TensorShape({}), {37.0}),
      CreateTensor<string>(TensorShape({1, 2}),
                           {"a", "b"})}},  // A single tuple of tensors
    {{CreateTensor<Variant>(
          TensorShape({}),
          {CreateTensor<double>(TensorShape({2, 2}), {1.0, 2.0, 3.0, 4.0})}),
      CreateTensor<Variant>(
          TensorShape({}),
          {CreateTensor<string>(TensorShape({1, 2}), {"a", "b"})}),
      CreateTensor<int64>(TensorShape({1, 3}), {1, 2, 3})}}  // Nested tensors
};

struct DatasetGetNextTest : TensorDatasetOpTest,
                            ::testing::WithParamInterface<TestParam> {};

TEST_P(DatasetGetNextTest, GetNext) {
  int thread_num = 2, cpu_num = 2;
  std::vector<Tensor> components = GetParam().components;

  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  DataTypeVector dtypes;
  std::vector<PartialTensorShape> shapes;
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto &component : components) {
    inputs.push_back(&component);
    dtypes.push_back(component.dtype());
    shapes.push_back(component.shape());
  }

  std::unique_ptr<OpKernel> tensor_dataset_kernel;
  TF_ASSERT_OK(
      CreateTensorDatasetKernel(dtypes, shapes, &tensor_dataset_kernel));
  std::unique_ptr<OpKernelContext> tensor_dataset_context;
  TF_ASSERT_OK(CreateTensorDatasetContext(tensor_dataset_kernel.get(), &inputs,
                                          &tensor_dataset_context));
  DatasetBase *tensor_dataset;
  TF_ASSERT_OK(CreateDataset(tensor_dataset_kernel.get(),
                             tensor_dataset_context.get(), &tensor_dataset));
  core::ScopedUnref scored_unref(tensor_dataset);

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

  EXPECT_EQ(out_tensors.size(), inputs.size());
  for (int i = 0; i < out_tensors.size(); i++) {
    if (out_tensors[i].dtype() == DT_VARIANT) {
      // Currently `ExpectEqual()` does not support the variant tensor
      // yet, so we manually cast the variant to numeric/string tensor.
      const Tensor *output = out_tensors[i].scalar<Variant>()().get<Tensor>();
      const Tensor *input = inputs[i].tensor->scalar<Variant>()().get<Tensor>();
      TF_EXPECT_OK(ExpectEqual(*output, *input));
    } else {
      TF_EXPECT_OK(ExpectEqual(out_tensors[i], *(inputs[i].tensor)));
    }
  }
}

INSTANTIATE_TEST_CASE_P(TensorDatasetOpTest, DatasetGetNextTest,
                        ::testing::ValuesIn(TestCases));

TEST_F(TensorDatasetOpTest, DatasetName) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  Tensor t1 = CreateTensor<int64>(TensorShape({2, 2}), {1, 2, 3, 4});
  Tensor t2 = CreateTensor<int64>(TensorShape({2, 2}), {5, 6, 7, 8});
  gtl::InlinedVector<TensorValue, 4> inputs;
  inputs.push_back(&t1);
  inputs.push_back(&t2);

  DataTypeVector dtypes;
  std::vector<PartialTensorShape> shapes;
  for (auto &input : inputs) {
    dtypes.push_back(input.tensor->dtype());
    shapes.push_back(input.tensor->shape());
  }

  std::unique_ptr<OpKernel> tensor_dataset_kernel;
  TF_ASSERT_OK(
      CreateTensorDatasetKernel(dtypes, shapes, &tensor_dataset_kernel));
  std::unique_ptr<OpKernelContext> tensor_dataset_context;
  TF_ASSERT_OK(CreateTensorDatasetContext(tensor_dataset_kernel.get(), &inputs,
                                          &tensor_dataset_context));
  DatasetBase *tensor_dataset;
  TF_ASSERT_OK(CreateDataset(tensor_dataset_kernel.get(),
                             tensor_dataset_context.get(), &tensor_dataset));
  core::ScopedUnref scored_unref(tensor_dataset);

  EXPECT_EQ(tensor_dataset->name(), kOpName);
}

TEST_F(TensorDatasetOpTest, DatasetOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  Tensor t1 = CreateTensor<int64>(TensorShape({2, 2}), {1, 2, 3, 4});
  Tensor t2 = CreateTensor<int64>(TensorShape({2, 2}), {5, 6, 7, 8});
  gtl::InlinedVector<TensorValue, 4> inputs;
  inputs.push_back(&t1);
  inputs.push_back(&t2);

  DataTypeVector dtypes;
  std::vector<PartialTensorShape> shapes;
  for (auto &input : inputs) {
    dtypes.push_back(input.tensor->dtype());
    shapes.push_back(input.tensor->shape());
  }

  std::unique_ptr<OpKernel> tensor_dataset_kernel;
  TF_ASSERT_OK(
      CreateTensorDatasetKernel(dtypes, shapes, &tensor_dataset_kernel));
  std::unique_ptr<OpKernelContext> tensor_dataset_context;
  TF_ASSERT_OK(CreateTensorDatasetContext(tensor_dataset_kernel.get(), &inputs,
                                          &tensor_dataset_context));
  DatasetBase *tensor_dataset;
  TF_ASSERT_OK(CreateDataset(tensor_dataset_kernel.get(),
                             tensor_dataset_context.get(), &tensor_dataset));
  core::ScopedUnref scored_unref(tensor_dataset);

  EXPECT_EQ(tensor_dataset->output_dtypes(), dtypes);
}

TEST_F(TensorDatasetOpTest, DatasetOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  Tensor t1 = CreateTensor<int64>(TensorShape({2, 2}), {1, 2, 3, 4});
  Tensor t2 = CreateTensor<int64>(TensorShape({2, 2}), {5, 6, 7, 8});
  gtl::InlinedVector<TensorValue, 4> inputs;
  inputs.push_back(&t1);
  inputs.push_back(&t2);

  DataTypeVector dtypes;
  std::vector<PartialTensorShape> shapes;
  for (auto &input : inputs) {
    dtypes.push_back(input.tensor->dtype());
    shapes.push_back(input.tensor->shape());
  }

  std::unique_ptr<OpKernel> tensor_dataset_kernel;
  TF_ASSERT_OK(
      CreateTensorDatasetKernel(dtypes, shapes, &tensor_dataset_kernel));
  std::unique_ptr<OpKernelContext> tensor_dataset_context;
  TF_ASSERT_OK(CreateTensorDatasetContext(tensor_dataset_kernel.get(), &inputs,
                                          &tensor_dataset_context));
  DatasetBase *tensor_dataset;
  TF_ASSERT_OK(CreateDataset(tensor_dataset_kernel.get(),
                             tensor_dataset_context.get(), &tensor_dataset));
  core::ScopedUnref scored_unref(tensor_dataset);

  EXPECT_EQ(tensor_dataset->output_shapes().size(), shapes.size());
  for (int i = 0; i < shapes.size(); i++) {
    EXPECT_TRUE(shapes[i].IsIdenticalTo(tensor_dataset->output_shapes()[i]));
  }
}

TEST_F(TensorDatasetOpTest, Cardinality) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  Tensor t1 = CreateTensor<int64>(TensorShape({2, 2}), {1, 2, 3, 4});
  Tensor t2 = CreateTensor<int64>(TensorShape({2, 2}), {5, 6, 7, 8});
  gtl::InlinedVector<TensorValue, 4> inputs;
  inputs.push_back(&t1);
  inputs.push_back(&t2);

  DataTypeVector dtypes;
  std::vector<PartialTensorShape> shapes;
  for (auto &input : inputs) {
    dtypes.push_back(input.tensor->dtype());
    shapes.push_back(input.tensor->shape());
  }

  std::unique_ptr<OpKernel> tensor_dataset_kernel;
  TF_ASSERT_OK(
      CreateTensorDatasetKernel(dtypes, shapes, &tensor_dataset_kernel));
  std::unique_ptr<OpKernelContext> tensor_dataset_context;
  TF_ASSERT_OK(CreateTensorDatasetContext(tensor_dataset_kernel.get(), &inputs,
                                          &tensor_dataset_context));
  DatasetBase *tensor_dataset;
  TF_ASSERT_OK(CreateDataset(tensor_dataset_kernel.get(),
                             tensor_dataset_context.get(), &tensor_dataset));
  core::ScopedUnref scored_unref(tensor_dataset);

  EXPECT_EQ(tensor_dataset->Cardinality(), 1LL);
}

TEST_F(TensorDatasetOpTest, DatasetSave) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  Tensor t1 = CreateTensor<int64>(TensorShape({2, 2}), {1, 2, 3, 4});
  Tensor t2 = CreateTensor<int64>(TensorShape({2, 2}), {5, 6, 7, 8});
  gtl::InlinedVector<TensorValue, 4> inputs;
  inputs.push_back(&t1);
  inputs.push_back(&t2);

  DataTypeVector dtypes;
  std::vector<PartialTensorShape> shapes;
  for (auto &input : inputs) {
    dtypes.push_back(input.tensor->dtype());
    shapes.push_back(input.tensor->shape());
  }

  std::unique_ptr<OpKernel> tensor_dataset_kernel;
  TF_ASSERT_OK(
      CreateTensorDatasetKernel(dtypes, shapes, &tensor_dataset_kernel));
  std::unique_ptr<OpKernelContext> tensor_dataset_context;
  TF_ASSERT_OK(CreateTensorDatasetContext(tensor_dataset_kernel.get(), &inputs,
                                          &tensor_dataset_context));
  DatasetBase *tensor_dataset;
  TF_ASSERT_OK(CreateDataset(tensor_dataset_kernel.get(),
                             tensor_dataset_context.get(), &tensor_dataset));
  core::ScopedUnref scored_unref(tensor_dataset);

  std::unique_ptr<SerializationContext> serialization_context;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_context));
  VariantTensorData data;
  VariantTensorDataWriter writer(&data);
  TF_ASSERT_OK(tensor_dataset->Save(serialization_context.get(), &writer));
  TF_ASSERT_OK(writer.Flush());
}

TEST_F(TensorDatasetOpTest, IteratorOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  Tensor t1 = CreateTensor<int64>(TensorShape({2, 2}), {1, 2, 3, 4});
  Tensor t2 = CreateTensor<int64>(TensorShape({2, 2}), {5, 6, 7, 8});
  gtl::InlinedVector<TensorValue, 4> inputs;
  inputs.push_back(&t1);
  inputs.push_back(&t2);

  DataTypeVector dtypes;
  std::vector<PartialTensorShape> shapes;
  for (auto &input : inputs) {
    dtypes.push_back(input.tensor->dtype());
    shapes.push_back(input.tensor->shape());
  }

  std::unique_ptr<OpKernel> tensor_dataset_kernel;
  TF_ASSERT_OK(
      CreateTensorDatasetKernel(dtypes, shapes, &tensor_dataset_kernel));
  std::unique_ptr<OpKernelContext> tensor_dataset_context;
  TF_ASSERT_OK(CreateTensorDatasetContext(tensor_dataset_kernel.get(), &inputs,
                                          &tensor_dataset_context));
  DatasetBase *tensor_dataset;
  TF_ASSERT_OK(CreateDataset(tensor_dataset_kernel.get(),
                             tensor_dataset_context.get(), &tensor_dataset));
  core::ScopedUnref scored_unref(tensor_dataset);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(
      CreateIteratorContext(tensor_dataset_context.get(), &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(tensor_dataset->MakeIterator(iterator_context.get(), "Iterator",
                                            &iterator));
  EXPECT_EQ(iterator->output_dtypes(), dtypes);
}

TEST_F(TensorDatasetOpTest, IteratorOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  Tensor t1 = CreateTensor<int64>(TensorShape({2, 2}), {1, 2, 3, 4});
  Tensor t2 = CreateTensor<int64>(TensorShape({2, 2}), {5, 6, 7, 8});
  gtl::InlinedVector<TensorValue, 4> inputs;
  inputs.push_back(&t1);
  inputs.push_back(&t2);

  DataTypeVector dtypes;
  std::vector<PartialTensorShape> shapes;
  for (auto &input : inputs) {
    dtypes.push_back(input.tensor->dtype());
    shapes.push_back(input.tensor->shape());
  }

  std::unique_ptr<OpKernel> tensor_dataset_kernel;
  TF_ASSERT_OK(
      CreateTensorDatasetKernel(dtypes, shapes, &tensor_dataset_kernel));
  std::unique_ptr<OpKernelContext> tensor_dataset_context;
  TF_ASSERT_OK(CreateTensorDatasetContext(tensor_dataset_kernel.get(), &inputs,
                                          &tensor_dataset_context));
  DatasetBase *tensor_dataset;
  TF_ASSERT_OK(CreateDataset(tensor_dataset_kernel.get(),
                             tensor_dataset_context.get(), &tensor_dataset));
  core::ScopedUnref scored_unref(tensor_dataset);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(
      CreateIteratorContext(tensor_dataset_context.get(), &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      CreateIteratorContext(tensor_dataset_context.get(), &iterator_context));
  TF_ASSERT_OK(tensor_dataset->MakeIterator(iterator_context.get(), "Iterator",
                                            &iterator));
  EXPECT_EQ(iterator->output_shapes().size(), shapes.size());
  for (int i = 0; i < shapes.size(); i++) {
    EXPECT_TRUE(shapes[i].IsIdenticalTo(iterator->output_shapes()[i]));
  }
}

TEST_F(TensorDatasetOpTest, IteratorOutputPrefix) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  Tensor t1 = CreateTensor<int64>(TensorShape({2, 2}), {1, 2, 3, 4});
  Tensor t2 = CreateTensor<int64>(TensorShape({2, 2}), {5, 6, 7, 8});
  gtl::InlinedVector<TensorValue, 4> inputs;
  inputs.push_back(&t1);
  inputs.push_back(&t2);

  DataTypeVector dtypes;
  std::vector<PartialTensorShape> shapes;
  for (auto &input : inputs) {
    dtypes.push_back(input.tensor->dtype());
    shapes.push_back(input.tensor->shape());
  }

  std::unique_ptr<OpKernel> tensor_dataset_kernel;
  TF_ASSERT_OK(
      CreateTensorDatasetKernel(dtypes, shapes, &tensor_dataset_kernel));
  std::unique_ptr<OpKernelContext> tensor_dataset_context;
  TF_ASSERT_OK(CreateTensorDatasetContext(tensor_dataset_kernel.get(), &inputs,
                                          &tensor_dataset_context));
  DatasetBase *tensor_dataset;
  TF_ASSERT_OK(CreateDataset(tensor_dataset_kernel.get(),
                             tensor_dataset_context.get(), &tensor_dataset));
  core::ScopedUnref scored_unref(tensor_dataset);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(
      CreateIteratorContext(tensor_dataset_context.get(), &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(tensor_dataset->MakeIterator(iterator_context.get(), "Iterator",
                                            &iterator));

  EXPECT_EQ(iterator->prefix(), "Iterator::FromTensor");
}

struct IteratorRoundtripTest : TensorDatasetOpTest,
                               ::testing::WithParamInterface<TestParam> {};

TEST_P(IteratorRoundtripTest, Roundtrip) {
  int thread_num = 2, cpu_num = 2;
  std::vector<Tensor> components = GetParam().components;

  gtl::InlinedVector<TensorValue, 4> inputs;

  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  DataTypeVector dtypes;
  std::vector<PartialTensorShape> shapes;
  for (auto &component : components) {
    inputs.push_back(&component);
    dtypes.push_back(component.dtype());
    shapes.push_back(component.shape());
  }

  std::unique_ptr<OpKernel> tensor_dataset_kernel;
  TF_ASSERT_OK(
      CreateTensorDatasetKernel(dtypes, shapes, &tensor_dataset_kernel));
  std::unique_ptr<OpKernelContext> tensor_dataset_context;
  TF_ASSERT_OK(CreateTensorDatasetContext(tensor_dataset_kernel.get(), &inputs,
                                          &tensor_dataset_context));
  DatasetBase *tensor_dataset;
  TF_ASSERT_OK(CreateDataset(tensor_dataset_kernel.get(),
                             tensor_dataset_context.get(), &tensor_dataset));
  core::ScopedUnref scored_unref(tensor_dataset);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(
      CreateIteratorContext(tensor_dataset_context.get(), &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(tensor_dataset->MakeIterator(iterator_context.get(), "Iterator",
                                            &iterator));

  std::unique_ptr<SerializationContext> serialization_context;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_context));

  // Test the serialization before running GetNext
  VariantTensorData data1;
  VariantTensorDataWriter writer1(&data1);
  TF_ASSERT_OK(iterator->Save(serialization_context.get(), &writer1));
  TF_ASSERT_OK(writer1.Flush());
  VariantTensorDataReader reader1(&data1);
  TF_ASSERT_OK(iterator->Restore(iterator_context.get(), &reader1));

  std::vector<Tensor> out_tensors;
  bool end_of_sequence = false;
  TF_EXPECT_OK(iterator->GetNext(iterator_context.get(), &out_tensors,
                                 &end_of_sequence));
  EXPECT_FALSE(end_of_sequence);
  EXPECT_EQ(out_tensors.size(), inputs.size());
  for (int i = 0; i < out_tensors.size(); i++) {
    if (out_tensors[i].dtype() == DT_VARIANT) {
      // Currently `ExpectEqual()` does not support the variant tensor
      // yet, so we manually cast the variant to numeric/string tensor.
      const Tensor *output = out_tensors[i].scalar<Variant>()().get<Tensor>();
      const Tensor *input = inputs[i].tensor->scalar<Variant>()().get<Tensor>();
      TF_EXPECT_OK(ExpectEqual(*output, *input));
    } else {
      TF_EXPECT_OK(ExpectEqual(out_tensors[i], *(inputs[i].tensor)));
    }
  }

  // Test the serialization after running GetNext.
  out_tensors.clear();
  VariantTensorData data2;
  VariantTensorDataWriter writer2(&data2);
  TF_ASSERT_OK(iterator->Save(serialization_context.get(), &writer2));
  TF_ASSERT_OK(writer2.Flush());
  VariantTensorDataReader reader2(&data2);
  TF_ASSERT_OK(iterator->Restore(iterator_context.get(), &reader2));

  TF_EXPECT_OK(iterator->GetNext(iterator_context.get(), &out_tensors,
                                 &end_of_sequence));
  EXPECT_TRUE(end_of_sequence);
  EXPECT_EQ(out_tensors.size(), 0);
}

INSTANTIATE_TEST_CASE_P(TensorDatasetOpTest, IteratorRoundtripTest,
                        ::testing::ValuesIn(TestCases));

}  // namespace
}  // namespace data
}  // namespace tensorflow
