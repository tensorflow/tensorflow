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
#include "tensorflow/core/kernels/data/cache_dataset_ops.h"

#include "tensorflow/core/kernels/data/dataset_test_base.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "cache_dataset";
constexpr char kIteratorPrefix[] = "Iterator";
constexpr char kFileDatasetPrefix[] = "File";
constexpr char kMemoryDatasetPrefix[] = "Memory";

class CacheDatasetOpTest : public DatasetOpsTestBase {
 public:
  ~CacheDatasetOpTest() {
    if (!filename_.empty()) {
      std::vector<string> cache_files;
      Status s = device_->env()->GetMatchingPaths(
          strings::StrCat(filename_, "*"), &cache_files);
      if (!s.ok()) {
        LOG(WARNING) << "Failed to get matching files on " << filename_
                     << "* : " << s.ToString();
      }
      for (const string& path : cache_files) {
        s = device_->env()->DeleteFile(path);
        if (!s.ok()) {
          LOG(WARNING) << "Failed to delete " << path << " : " << s.ToString();
        }
      }
    }
  }

 protected:
  // Creates `TensorSliceDataset` variant tensor from the input vector of
  // tensors.
  Status CreateTensorSliceDatasetTensor(
      std::vector<Tensor>* const tensor_vector, Tensor* dataset_tensor) {
    DatasetBase* tensor_slice_dataset;
    TF_RETURN_IF_ERROR(CreateTensorSliceDataset(
        "tensor_slice_node", tensor_vector, &tensor_slice_dataset));
    TF_RETURN_IF_ERROR(
        StoreDatasetInVariantTensor(tensor_slice_dataset, dataset_tensor));
    return Status::OK();
  }

  // Create a new `CacheDataset` op kernel.
  Status CreateCacheDatasetOpKernel(
      const DataTypeVector& output_types,
      const std::vector<PartialTensorShape>& output_shapes,
      std::unique_ptr<OpKernel>* cache_dataset_op_kernel) {
    NodeDef node_def = test::function::NDef(
        kNodeName, name_utils::OpName(CacheDatasetOp::kDatasetType),
        {CacheDatasetOp::kInputDataset, CacheDatasetOp::kFileName},
        {{CacheDatasetOp::kOutputTypes, output_types},
         {CacheDatasetOp::kOutputShapes, output_shapes}});
    TF_RETURN_IF_ERROR(CreateOpKernel(node_def, cache_dataset_op_kernel));
    return Status::OK();
  }

  // Create a new `CacheDataset` op kernel context.
  Status CreateCacheDatasetContext(
      OpKernel* const op_kernel,
      gtl::InlinedVector<TensorValue, 4>* const inputs,
      std::unique_ptr<OpKernelContext>* context) {
    TF_RETURN_IF_ERROR(CheckOpKernelInput(*op_kernel, *inputs));
    TF_RETURN_IF_ERROR(CreateOpKernelContext(op_kernel, inputs, context));
    TF_RETURN_IF_ERROR(ParseScalarArgument<tstring>(
        context->get(), CacheDatasetOp::kFileName, &filename_));
    return Status::OK();
  }

 private:
  tstring filename_ = "";
};

struct TestCase {
  std::vector<Tensor> input_tensors;
  string file_name;
  std::vector<Tensor> expected_outputs;
  DataTypeVector expected_output_dtypes;
  std::vector<PartialTensorShape> expected_output_shapes;
  int64 expected_cardinality;
  std::vector<int> breakpoints;
};

// Test case 1: cache data in file.
TestCase TestCase1() {
  return {/*input_tensors*/ {CreateTensor<int64>(TensorShape{3, 3, 1},
                                                 {0, 1, 2, 3, 4, 5, 6, 7, 8})},
          /*file_name*/ absl::StrCat(testing::TmpDir(), "/cache_data"),
          /*expected_outputs*/
          {CreateTensor<int64>(TensorShape{3, 1}, {0, 1, 2}),
           CreateTensor<int64>(TensorShape{3, 1}, {3, 4, 5}),
           CreateTensor<int64>(TensorShape{3, 1}, {6, 7, 8})},
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({3, 1})},
          /*expected_cardinality*/ 3,
          /*breakpoints*/ {0, 2, 4, 11}};
}

// Test case 2: cache empty data in file.
TestCase TestCase2() {
  return {/*input_tensors*/ {CreateTensor<int64>(TensorShape{0}, {})},
          /*file_name*/ absl::StrCat(testing::TmpDir(), "/empty_cache_data"),
          /*expected_outputs*/ {},
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({})},
          /*expected_cardinality*/ 0,
          /*breakpoints*/ {0, 2, 4, 11}};
}

// Test case 3: cache data in memory.
TestCase TestCase3() {
  return {/*input_tensors*/ {CreateTensor<int64>(TensorShape{3, 3, 1},
                                                 {0, 1, 2, 3, 4, 5, 6, 7, 8})},
          /*file_name*/ "",
          /*expected_outputs*/
          {CreateTensor<int64>(TensorShape{3, 1}, {0, 1, 2}),
           CreateTensor<int64>(TensorShape{3, 1}, {3, 4, 5}),
           CreateTensor<int64>(TensorShape{3, 1}, {6, 7, 8})},
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({3, 1})},
          /*expected_cardinality*/ 3,
          /*breakpoints*/ {0, 2, 4, 11}};
}

// Test case 4: cache empty data in memory.
TestCase TestCase4() {
  return {/*input_tensors*/ {CreateTensor<int64>(TensorShape{0}, {})},
          /*file_name*/ "",
          /*expected_outputs*/ {},
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({})},
          /*expected_cardinality*/ 0,
          /*breakpoints*/ {0, 2, 4, 11}};
}

class ParameterizedCacheDatasetOpTest
    : public CacheDatasetOpTest,
      public ::testing::WithParamInterface<TestCase> {};

TEST_P(ParameterizedCacheDatasetOpTest, GetNext) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> cache_dataset_kernel;
  TF_ASSERT_OK(CreateCacheDatasetOpKernel(test_case.expected_output_dtypes,
                                          test_case.expected_output_shapes,
                                          &cache_dataset_kernel));
  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  Tensor file_name =
      CreateTensor<tstring>(TensorShape{}, {test_case.file_name});
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&tensor_slice_dataset_tensor), TensorValue(&file_name)});
  std::unique_ptr<OpKernelContext> cache_dataset_context;
  TF_ASSERT_OK(CreateCacheDatasetContext(cache_dataset_kernel.get(), &inputs,
                                         &cache_dataset_context));
  DatasetBase* cache_dataset;
  TF_ASSERT_OK(CreateDataset(cache_dataset_kernel.get(),
                             cache_dataset_context.get(), &cache_dataset));
  core::ScopedUnref scoped_unref(cache_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(cache_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(cache_dataset->MakeIterator(iterator_ctx.get(), kIteratorPrefix,
                                           &iterator));

  // Test the write mode.
  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  while (!end_of_sequence) {
    std::vector<Tensor> next;
    TF_EXPECT_OK(
        iterator->GetNext(iterator_ctx.get(), &next, &end_of_sequence));
    out_tensors.insert(out_tensors.end(), next.begin(), next.end());
  }
  TF_EXPECT_OK(ExpectEqual(out_tensors, test_case.expected_outputs,
                           /*compare_order*/ true));

  // Test the read mode.
  TF_ASSERT_OK(cache_dataset->MakeIterator(iterator_ctx.get(), kIteratorPrefix,
                                           &iterator));
  end_of_sequence = false;
  out_tensors.clear();
  while (!end_of_sequence) {
    std::vector<Tensor> next;
    TF_EXPECT_OK(
        iterator->GetNext(iterator_ctx.get(), &next, &end_of_sequence));
    out_tensors.insert(out_tensors.end(), next.begin(), next.end());
  }
  TF_EXPECT_OK(ExpectEqual(out_tensors, test_case.expected_outputs,
                           /*compare_order*/ true));
}

TEST_F(CacheDatasetOpTest, DatasetNodeName) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = TestCase1();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> cache_dataset_kernel;
  TF_ASSERT_OK(CreateCacheDatasetOpKernel(test_case.expected_output_dtypes,
                                          test_case.expected_output_shapes,
                                          &cache_dataset_kernel));
  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  Tensor file_name =
      CreateTensor<tstring>(TensorShape{}, {test_case.file_name});
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&tensor_slice_dataset_tensor), TensorValue(&file_name)});
  std::unique_ptr<OpKernelContext> cache_dataset_context;
  TF_ASSERT_OK(CreateCacheDatasetContext(cache_dataset_kernel.get(), &inputs,
                                         &cache_dataset_context));
  DatasetBase* cache_dataset;
  TF_ASSERT_OK(CreateDataset(cache_dataset_kernel.get(),
                             cache_dataset_context.get(), &cache_dataset));
  core::ScopedUnref scoped_unref(cache_dataset);

  EXPECT_EQ(cache_dataset->node_name(), kNodeName);
}

TEST_P(ParameterizedCacheDatasetOpTest, DatasetTypeString) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> cache_dataset_kernel;
  TF_ASSERT_OK(CreateCacheDatasetOpKernel(test_case.expected_output_dtypes,
                                          test_case.expected_output_shapes,
                                          &cache_dataset_kernel));
  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  Tensor file_name =
      CreateTensor<tstring>(TensorShape{}, {test_case.file_name});
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&tensor_slice_dataset_tensor), TensorValue(&file_name)});
  std::unique_ptr<OpKernelContext> cache_dataset_context;
  TF_ASSERT_OK(CreateCacheDatasetContext(cache_dataset_kernel.get(), &inputs,
                                         &cache_dataset_context));
  DatasetBase* cache_dataset;
  TF_ASSERT_OK(CreateDataset(cache_dataset_kernel.get(),
                             cache_dataset_context.get(), &cache_dataset));
  core::ScopedUnref scoped_unref(cache_dataset);

  EXPECT_EQ(cache_dataset->type_string(),
            name_utils::OpName(CacheDatasetOp::kDatasetType));
}

TEST_P(ParameterizedCacheDatasetOpTest, DatasetOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> cache_dataset_kernel;
  TF_ASSERT_OK(CreateCacheDatasetOpKernel(test_case.expected_output_dtypes,
                                          test_case.expected_output_shapes,
                                          &cache_dataset_kernel));
  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  Tensor file_name =
      CreateTensor<tstring>(TensorShape{}, {test_case.file_name});
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&tensor_slice_dataset_tensor), TensorValue(&file_name)});
  std::unique_ptr<OpKernelContext> cache_dataset_context;
  TF_ASSERT_OK(CreateCacheDatasetContext(cache_dataset_kernel.get(), &inputs,
                                         &cache_dataset_context));
  DatasetBase* cache_dataset;
  TF_ASSERT_OK(CreateDataset(cache_dataset_kernel.get(),
                             cache_dataset_context.get(), &cache_dataset));
  core::ScopedUnref scoped_unref(cache_dataset);

  TF_EXPECT_OK(VerifyTypesMatch(cache_dataset->output_dtypes(),
                                test_case.expected_output_dtypes));
}

TEST_P(ParameterizedCacheDatasetOpTest, DatasetOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> cache_dataset_kernel;
  TF_ASSERT_OK(CreateCacheDatasetOpKernel(test_case.expected_output_dtypes,
                                          test_case.expected_output_shapes,
                                          &cache_dataset_kernel));
  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  Tensor file_name =
      CreateTensor<tstring>(TensorShape{}, {test_case.file_name});
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&tensor_slice_dataset_tensor), TensorValue(&file_name)});
  std::unique_ptr<OpKernelContext> cache_dataset_context;
  TF_ASSERT_OK(CreateCacheDatasetContext(cache_dataset_kernel.get(), &inputs,
                                         &cache_dataset_context));
  DatasetBase* cache_dataset;
  TF_ASSERT_OK(CreateDataset(cache_dataset_kernel.get(),
                             cache_dataset_context.get(), &cache_dataset));
  core::ScopedUnref scoped_unref(cache_dataset);

  TF_EXPECT_OK(VerifyShapesCompatible(cache_dataset->output_shapes(),
                                      test_case.expected_output_shapes));
}

TEST_P(ParameterizedCacheDatasetOpTest, Cardinality) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> cache_dataset_kernel;
  TF_ASSERT_OK(CreateCacheDatasetOpKernel(test_case.expected_output_dtypes,
                                          test_case.expected_output_shapes,
                                          &cache_dataset_kernel));
  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  Tensor file_name =
      CreateTensor<tstring>(TensorShape{}, {test_case.file_name});
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&tensor_slice_dataset_tensor), TensorValue(&file_name)});
  std::unique_ptr<OpKernelContext> cache_dataset_context;
  TF_ASSERT_OK(CreateCacheDatasetContext(cache_dataset_kernel.get(), &inputs,
                                         &cache_dataset_context));
  DatasetBase* cache_dataset;
  TF_ASSERT_OK(CreateDataset(cache_dataset_kernel.get(),
                             cache_dataset_context.get(), &cache_dataset));
  core::ScopedUnref scoped_unref(cache_dataset);

  EXPECT_EQ(cache_dataset->Cardinality(), test_case.expected_cardinality);
}

TEST_P(ParameterizedCacheDatasetOpTest, IteratorOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> cache_dataset_kernel;
  TF_ASSERT_OK(CreateCacheDatasetOpKernel(test_case.expected_output_dtypes,
                                          test_case.expected_output_shapes,
                                          &cache_dataset_kernel));
  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  Tensor file_name =
      CreateTensor<tstring>(TensorShape{}, {test_case.file_name});
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&tensor_slice_dataset_tensor), TensorValue(&file_name)});
  std::unique_ptr<OpKernelContext> cache_dataset_context;
  TF_ASSERT_OK(CreateCacheDatasetContext(cache_dataset_kernel.get(), &inputs,
                                         &cache_dataset_context));
  DatasetBase* cache_dataset;
  TF_ASSERT_OK(CreateDataset(cache_dataset_kernel.get(),
                             cache_dataset_context.get(), &cache_dataset));
  core::ScopedUnref scoped_unref(cache_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(cache_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(cache_dataset->MakeIterator(iterator_ctx.get(), kIteratorPrefix,
                                           &iterator));

  TF_EXPECT_OK(VerifyTypesMatch(iterator->output_dtypes(),
                                test_case.expected_output_dtypes));
}

TEST_P(ParameterizedCacheDatasetOpTest, IteratorOutputPrefix) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> cache_dataset_kernel;
  TF_ASSERT_OK(CreateCacheDatasetOpKernel(test_case.expected_output_dtypes,
                                          test_case.expected_output_shapes,
                                          &cache_dataset_kernel));
  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  Tensor file_name =
      CreateTensor<tstring>(TensorShape{}, {test_case.file_name});
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&tensor_slice_dataset_tensor), TensorValue(&file_name)});
  std::unique_ptr<OpKernelContext> cache_dataset_context;
  TF_ASSERT_OK(CreateCacheDatasetContext(cache_dataset_kernel.get(), &inputs,
                                         &cache_dataset_context));
  DatasetBase* cache_dataset;
  TF_ASSERT_OK(CreateDataset(cache_dataset_kernel.get(),
                             cache_dataset_context.get(), &cache_dataset));
  core::ScopedUnref scoped_unref(cache_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(cache_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(cache_dataset->MakeIterator(iterator_ctx.get(), kIteratorPrefix,
                                           &iterator));

  name_utils::IteratorPrefixParams params;
  params.dataset_prefix =
      test_case.file_name.empty() ? kMemoryDatasetPrefix : kFileDatasetPrefix;
  EXPECT_EQ(iterator->prefix(),
            name_utils::IteratorPrefix(CacheDatasetOp::kDatasetType,
                                       kIteratorPrefix, params));
}

TEST_P(ParameterizedCacheDatasetOpTest, Roundtrip) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> cache_dataset_kernel;
  TF_ASSERT_OK(CreateCacheDatasetOpKernel(test_case.expected_output_dtypes,
                                          test_case.expected_output_shapes,
                                          &cache_dataset_kernel));
  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  Tensor file_name =
      CreateTensor<tstring>(TensorShape{}, {test_case.file_name});
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&tensor_slice_dataset_tensor), TensorValue(&file_name)});
  std::unique_ptr<OpKernelContext> cache_dataset_context;
  TF_ASSERT_OK(CreateCacheDatasetContext(cache_dataset_kernel.get(), &inputs,
                                         &cache_dataset_context));
  DatasetBase* cache_dataset;
  TF_ASSERT_OK(CreateDataset(cache_dataset_kernel.get(),
                             cache_dataset_context.get(), &cache_dataset));
  core::ScopedUnref scoped_unref(cache_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(cache_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(cache_dataset->MakeIterator(iterator_ctx.get(), kIteratorPrefix,
                                           &iterator));

  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  // For MemoryIterator in the read mode, the cache needs to be completed before
  // it has been read.
  if (test_case.file_name.empty()) {
    while (!end_of_sequence) {
      TF_EXPECT_OK(iterator->GetNext(iterator_ctx.get(), &out_tensors,
                                     &end_of_sequence));
    }
    end_of_sequence = false;
    out_tensors.clear();
    TF_ASSERT_OK(cache_dataset->MakeIterator(iterator_ctx.get(),
                                             kIteratorPrefix, &iterator));
  }

  std::unique_ptr<SerializationContext> serialization_ctx;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_ctx));
  int cur_iteration = 0;
  auto expected_outputs_it = test_case.expected_outputs.begin();
  for (int breakpoint : test_case.breakpoints) {
    VariantTensorData data;
    VariantTensorDataWriter writer(&data);
    TF_EXPECT_OK(iterator->Save(serialization_ctx.get(), &writer));
    TF_EXPECT_OK(writer.Flush());
    VariantTensorDataReader reader(&data);
    TF_EXPECT_OK(RestoreIterator(iterator_ctx.get(), &reader, kIteratorPrefix,
                                 *cache_dataset, &iterator));

    while (cur_iteration <= breakpoint) {
      out_tensors.clear();
      TF_EXPECT_OK(iterator->GetNext(iterator_ctx.get(), &out_tensors,
                                     &end_of_sequence));
      if (!end_of_sequence) {
        EXPECT_LT(expected_outputs_it, test_case.expected_outputs.end());
        TF_EXPECT_OK(ExpectEqual(out_tensors.back(), *expected_outputs_it));
        expected_outputs_it++;
      }
      cur_iteration++;
    }

    if (breakpoint >= test_case.expected_cardinality) {
      EXPECT_TRUE(end_of_sequence);
      EXPECT_EQ(expected_outputs_it, test_case.expected_outputs.end());
    } else {
      EXPECT_FALSE(end_of_sequence);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    CacheDatasetOpTest, ParameterizedCacheDatasetOpTest,
    ::testing::ValuesIn(std::vector<TestCase>({TestCase1(), TestCase2(),
                                               TestCase3(), TestCase4()})));

}  // namespace
}  // namespace data
}  // namespace tensorflow
