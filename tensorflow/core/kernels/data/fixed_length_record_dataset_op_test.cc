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
#include "tensorflow/core/kernels/data/text_line_dataset_op.h"

#include "tensorflow/core/kernels/data/dataset_test_base.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "fixed_length_record_dataset";
constexpr char kIteratorPrefix[] = "Iterator";

class FixedLengthRecordDatasetOpTest : public DatasetOpsTestBase {
 protected:
  // Create a new `TextLineDataset` op kernel.
  Status CreateFixedLengthRecordDatasetOpKernel(
      std::unique_ptr<OpKernel>* text_line_dataset_op_kernel) {
    NodeDef node_def = test::function::NDef(
        kNodeName, "FixedLengthRecordDatasetV2",
        {"filenames", "header_bytes", "record_bytes", "footer_bytes",
         "buffer_size", "compression_type"},
        {});
    TF_RETURN_IF_ERROR(CreateOpKernel(node_def, text_line_dataset_op_kernel));
    return Status::OK();
  }

  // Create a new `FixedLengthRecordDataset` op kernel context
  Status CreateFixedLengthRecordDatasetContext(
      OpKernel* const op_kernel,
      gtl::InlinedVector<TensorValue, 4>* const inputs,
      std::unique_ptr<OpKernelContext>* context) {
    TF_RETURN_IF_ERROR(CheckOpKernelInput(*op_kernel, *inputs));
    TF_RETURN_IF_ERROR(CreateOpKernelContext(op_kernel, inputs, context));
    return Status::OK();
  }
};

struct TestCase {
  std::vector<string> filenames;
  int64 header_bytes;
  int64 record_bytes;
  int64 footer_bytes;
  int64 buffer_size;
  CompressionType compression_type;
  string header;
  string footer;
  std::vector<Tensor> expected_outputs;
  DataTypeVector expected_output_dtypes;
  std::vector<PartialTensorShape> expected_output_shapes;
  int64 expected_cardinality;
  std::vector<int> breakpoints;
};

std::vector<string> ToStringVector(const std::vector<Tensor>& str_tensors) {
  std::vector<string> str_vec;
  for (const Tensor& tensor : str_tensors) {
    for (int i = 0; i < tensor.NumElements(); ++i) {
      str_vec.push_back(tensor.flat<string>()(i));
    }
  }
  return str_vec;
}

// Test case 1: multiple text files with ZLIB compression.
TestCase TestCase1() {
  return {
      /*filenames*/ {absl::StrCat(testing::TmpDir(), "/text_line_ZLIB_1"),
                     absl::StrCat(testing::TmpDir(), "/text_line_ZLIB_2")},
      /*header_bytes*/ 5,
      /*record_bytes*/ 3,
      /*footer_bytes*/ 2,
      /*buffer_size*/ 10,
      /*compression_type*/ ZLIB,
      /*header*/ "HHHHH",
      /*footer*/ "FF",
      /*expected_outputs*/
      {DatasetOpsTestBase::CreateTensor<string>(TensorShape({}),
                                                {"111"}),
       DatasetOpsTestBase::CreateTensor<string>(TensorShape({}),
                                                {"222"}),
       DatasetOpsTestBase::CreateTensor<string>(TensorShape({}),
                                                {"333"}),
       DatasetOpsTestBase::CreateTensor<string>(TensorShape({}),
                                                {"444"}),
       DatasetOpsTestBase::CreateTensor<string>(TensorShape({}), {"555"})},
      /*expected_output_dtypes*/ {DT_STRING},
      /*expected_output_shapes*/ {PartialTensorShape({})},
      /*expected_cardinality*/ kUnknownCardinality,
      /*breakpoints*/ {0, 2, 6}};
}

// Test case 2: multiple text files with GZIP compression.
TestCase TestCase2() {
  return {/*filenames*/ {absl::StrCat(testing::TmpDir(), "/text_line_GZIP_1"),
                         absl::StrCat(testing::TmpDir(), "/text_line_GZIP_2")},
          /*header_bytes*/ 5,
          /*record_bytes*/ 3,
          /*footer_bytes*/ 2,
          /*buffer_size*/ 10,
          /*compression_type*/ GZIP,
          /*header*/ "HHHHH",
          /*footer*/ "FF",
          /*expected_outputs*/
          {DatasetOpsTestBase::CreateTensor<string>(TensorShape({}),
                                                    {"aaa"}),
           DatasetOpsTestBase::CreateTensor<string>(TensorShape({}),
                                                    {"bbb"}),
           DatasetOpsTestBase::CreateTensor<string>(TensorShape({}),
                                                    {"ccc"})},
          /*expected_output_dtypes*/ {DT_STRING},
          /*expected_output_shapes*/ {PartialTensorShape({})},
          /*expected_cardinality*/ kUnknownCardinality,
          /*breakpoints*/ {0, 2, 6}};
}

// Test case 3: a single text file without compression.
TestCase TestCase3() {
  return {/*filenames*/ {
              absl::StrCat(testing::TmpDir(), "/text_line_UNCOMPRESSED")},
          /*header_bytes*/ 5,
          /*record_bytes*/ 3,
          /*footer_bytes*/ 2,
          /*buffer_size*/ 10,
          /*compression_type*/ UNCOMPRESSED,
          /*header*/ "HHHHH",
          /*footer*/ "FF",
          /*expected_outputs*/
          {DatasetOpsTestBase::CreateTensor<string>(TensorShape({}),
                                                    {"aa1"}),
           DatasetOpsTestBase::CreateTensor<string>(TensorShape({}),
                                                    {"b2b"}),
           DatasetOpsTestBase::CreateTensor<string>(TensorShape({}),
                                                    {"3cc"})},
          /*expected_output_dtypes*/ {DT_STRING},
          /*expected_output_shapes*/ {PartialTensorShape({})},
          /*expected_cardinality*/ kUnknownCardinality,
          /*breakpoints*/ {0, 2, 6}};
}

class ParameterizedFixedLengthRecordDatasetOpTest
    : public FixedLengthRecordDatasetOpTest,
      public ::testing::WithParamInterface<TestCase> {};

TEST_P(ParameterizedFixedLengthRecordDatasetOpTest, GetNext) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::vector<string> multi_texts = ToStringVector(test_case.expected_outputs);
  TF_ASSERT_OK(CreateMultiTextFiles(
      test_case.filenames, multi_texts, test_case.buffer_size,
      test_case.buffer_size, test_case.compression_type));

  std::unique_ptr<OpKernel> text_line_dataset_kernel;
  TF_ASSERT_OK(
      CreateFixedLengthRecordDatasetOpKernel(&text_line_dataset_kernel));

  int64 num_files = test_case.filenames.size();
  Tensor filenames =
      CreateTensor<string>(TensorShape({num_files}), test_case.filenames);
  Tensor compression_type = CreateTensor<string>(
      TensorShape({}), {CompressionName(test_case.compression_type)});
  Tensor buffer_size =
      CreateTensor<int64>(TensorShape({}), {test_case.buffer_size});
  gtl::InlinedVector<TensorValue, 4> inputs{TensorValue(&filenames),
                                            TensorValue(&compression_type),
                                            TensorValue(&buffer_size)};
  std::unique_ptr<OpKernelContext> text_line_dataset_context;
  TF_ASSERT_OK(CreateFixedLengthRecordDatasetContext(
      text_line_dataset_kernel.get(), &inputs, &text_line_dataset_context));

  DatasetBase* text_line_dataset;
  TF_ASSERT_OK(CreateDataset(text_line_dataset_kernel.get(),
                             text_line_dataset_context.get(),
                             &text_line_dataset));
  core::ScopedUnref scoped_unref(text_line_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(text_line_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(text_line_dataset->MakeIterator(iterator_ctx.get(),
                                               kIteratorPrefix, &iterator));
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
}

TEST_F(FixedLengthRecordDatasetOpTest, DatasetNodeName) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = TestCase1();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::vector<string> multi_texts = ToStringVector(test_case.expected_outputs);
  TF_ASSERT_OK(CreateMultiTextFiles(
      test_case.filenames, multi_texts, test_case.buffer_size,
      test_case.buffer_size, test_case.compression_type));

  std::unique_ptr<OpKernel> text_line_dataset_kernel;
  TF_ASSERT_OK(
      CreateFixedLengthRecordDatasetOpKernel(&text_line_dataset_kernel));

  int64 num_files = test_case.filenames.size();
  Tensor filenames =
      CreateTensor<string>(TensorShape({num_files}), test_case.filenames);
  Tensor compression_type = CreateTensor<string>(
      TensorShape({}), {CompressionName(test_case.compression_type)});
  Tensor buffer_size =
      CreateTensor<int64>(TensorShape({}), {test_case.buffer_size});
  gtl::InlinedVector<TensorValue, 4> inputs{TensorValue(&filenames),
                                            TensorValue(&compression_type),
                                            TensorValue(&buffer_size)};
  std::unique_ptr<OpKernelContext> text_line_dataset_context;
  TF_ASSERT_OK(CreateFixedLengthRecordDatasetContext(
      text_line_dataset_kernel.get(), &inputs, &text_line_dataset_context));

  DatasetBase* text_line_dataset;
  TF_ASSERT_OK(CreateDataset(text_line_dataset_kernel.get(),
                             text_line_dataset_context.get(),
                             &text_line_dataset));
  core::ScopedUnref scoped_unref(text_line_dataset);
  EXPECT_EQ(text_line_dataset->node_name(), kNodeName);
}

TEST_F(FixedLengthRecordDatasetOpTest, DatasetTypeString) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = TestCase1();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::vector<string> multi_texts = ToStringVector(test_case.expected_outputs);
  TF_ASSERT_OK(CreateMultiTextFiles(
      test_case.filenames, multi_texts, test_case.buffer_size,
      test_case.buffer_size, test_case.compression_type));

  std::unique_ptr<OpKernel> text_line_dataset_kernel;
  TF_ASSERT_OK(
      CreateFixedLengthRecordDatasetOpKernel(&text_line_dataset_kernel));

  int64 num_files = test_case.filenames.size();
  Tensor filenames =
      CreateTensor<string>(TensorShape({num_files}), test_case.filenames);
  Tensor compression_type = CreateTensor<string>(
      TensorShape({}), {CompressionName(test_case.compression_type)});
  Tensor buffer_size =
      CreateTensor<int64>(TensorShape({}), {test_case.buffer_size});
  gtl::InlinedVector<TensorValue, 4> inputs{TensorValue(&filenames),
                                            TensorValue(&compression_type),
                                            TensorValue(&buffer_size)};
  std::unique_ptr<OpKernelContext> text_line_dataset_context;
  TF_ASSERT_OK(CreateFixedLengthRecordDatasetContext(
      text_line_dataset_kernel.get(), &inputs, &text_line_dataset_context));

  DatasetBase* text_line_dataset;
  TF_ASSERT_OK(CreateDataset(text_line_dataset_kernel.get(),
                             text_line_dataset_context.get(),
                             &text_line_dataset));
  core::ScopedUnref scoped_unref(text_line_dataset);
  EXPECT_EQ(text_line_dataset->type_string(),
            name_utils::OpName(FixedLengthRecordDatasetOp::kDatasetType));
}

TEST_P(ParameterizedFixedLengthRecordDatasetOpTest, DatasetOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::vector<string> multi_texts = ToStringVector(test_case.expected_outputs);
  TF_ASSERT_OK(CreateMultiTextFiles(
      test_case.filenames, multi_texts, test_case.buffer_size,
      test_case.buffer_size, test_case.compression_type));

  std::unique_ptr<OpKernel> text_line_dataset_kernel;
  TF_ASSERT_OK(
      CreateFixedLengthRecordDatasetOpKernel(&text_line_dataset_kernel));

  int64 num_files = test_case.filenames.size();
  Tensor filenames =
      CreateTensor<string>(TensorShape({num_files}), test_case.filenames);
  Tensor compression_type = CreateTensor<string>(
      TensorShape({}), {CompressionName(test_case.compression_type)});
  Tensor buffer_size =
      CreateTensor<int64>(TensorShape({}), {test_case.buffer_size});
  gtl::InlinedVector<TensorValue, 4> inputs{TensorValue(&filenames),
                                            TensorValue(&compression_type),
                                            TensorValue(&buffer_size)};
  std::unique_ptr<OpKernelContext> text_line_dataset_context;
  TF_ASSERT_OK(CreateFixedLengthRecordDatasetContext(
      text_line_dataset_kernel.get(), &inputs, &text_line_dataset_context));

  DatasetBase* text_line_dataset;
  TF_ASSERT_OK(CreateDataset(text_line_dataset_kernel.get(),
                             text_line_dataset_context.get(),
                             &text_line_dataset));
  core::ScopedUnref scoped_unref(text_line_dataset);
  TF_EXPECT_OK(VerifyTypesMatch(text_line_dataset->output_dtypes(),
                                test_case.expected_output_dtypes));
}

TEST_P(ParameterizedFixedLengthRecordDatasetOpTest, DatasetOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::vector<string> multi_texts = ToStringVector(test_case.expected_outputs);
  TF_ASSERT_OK(CreateMultiTextFiles(
      test_case.filenames, multi_texts, test_case.buffer_size,
      test_case.buffer_size, test_case.compression_type));

  std::unique_ptr<OpKernel> text_line_dataset_kernel;
  TF_ASSERT_OK(
      CreateFixedLengthRecordDatasetOpKernel(&text_line_dataset_kernel));

  int64 num_files = test_case.filenames.size();
  Tensor filenames =
      CreateTensor<string>(TensorShape({num_files}), test_case.filenames);
  Tensor compression_type = CreateTensor<string>(
      TensorShape({}), {CompressionName(test_case.compression_type)});
  Tensor buffer_size =
      CreateTensor<int64>(TensorShape({}), {test_case.buffer_size});
  gtl::InlinedVector<TensorValue, 4> inputs{TensorValue(&filenames),
                                            TensorValue(&compression_type),
                                            TensorValue(&buffer_size)};
  std::unique_ptr<OpKernelContext> text_line_dataset_context;
  TF_ASSERT_OK(CreateFixedLengthRecordDatasetContext(
      text_line_dataset_kernel.get(), &inputs, &text_line_dataset_context));

  DatasetBase* text_line_dataset;
  TF_ASSERT_OK(CreateDataset(text_line_dataset_kernel.get(),
                             text_line_dataset_context.get(),
                             &text_line_dataset));
  core::ScopedUnref scoped_unref(text_line_dataset);
  TF_EXPECT_OK(VerifyShapesCompatible(text_line_dataset->output_shapes(),
                                      test_case.expected_output_shapes));
}

TEST_P(ParameterizedFixedLengthRecordDatasetOpTest, Cardinality) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::vector<string> multi_texts = ToStringVector(test_case.expected_outputs);
  TF_ASSERT_OK(CreateMultiTextFiles(
      test_case.filenames, multi_texts, test_case.buffer_size,
      test_case.buffer_size, test_case.compression_type));

  std::unique_ptr<OpKernel> text_line_dataset_kernel;
  TF_ASSERT_OK(
      CreateFixedLengthRecordDatasetOpKernel(&text_line_dataset_kernel));

  int64 num_files = test_case.filenames.size();
  Tensor filenames =
      CreateTensor<string>(TensorShape({num_files}), test_case.filenames);
  Tensor compression_type = CreateTensor<string>(
      TensorShape({}), {CompressionName(test_case.compression_type)});
  Tensor buffer_size =
      CreateTensor<int64>(TensorShape({}), {test_case.buffer_size});
  gtl::InlinedVector<TensorValue, 4> inputs{TensorValue(&filenames),
                                            TensorValue(&compression_type),
                                            TensorValue(&buffer_size)};
  std::unique_ptr<OpKernelContext> text_line_dataset_context;
  TF_ASSERT_OK(CreateFixedLengthRecordDatasetContext(
      text_line_dataset_kernel.get(), &inputs, &text_line_dataset_context));

  DatasetBase* text_line_dataset;
  TF_ASSERT_OK(CreateDataset(text_line_dataset_kernel.get(),
                             text_line_dataset_context.get(),
                             &text_line_dataset));
  core::ScopedUnref scoped_unref(text_line_dataset);
  EXPECT_EQ(text_line_dataset->Cardinality(), test_case.expected_cardinality);
}

TEST_P(ParameterizedFixedLengthRecordDatasetOpTest, DatasetSave) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::vector<string> multi_texts = ToStringVector(test_case.expected_outputs);
  TF_ASSERT_OK(CreateMultiTextFiles(
      test_case.filenames, multi_texts, test_case.buffer_size,
      test_case.buffer_size, test_case.compression_type));

  std::unique_ptr<OpKernel> text_line_dataset_kernel;
  TF_ASSERT_OK(
      CreateFixedLengthRecordDatasetOpKernel(&text_line_dataset_kernel));

  int64 num_files = test_case.filenames.size();
  Tensor filenames =
      CreateTensor<string>(TensorShape({num_files}), test_case.filenames);
  Tensor compression_type = CreateTensor<string>(
      TensorShape({}), {CompressionName(test_case.compression_type)});
  Tensor buffer_size =
      CreateTensor<int64>(TensorShape({}), {test_case.buffer_size});
  gtl::InlinedVector<TensorValue, 4> inputs{TensorValue(&filenames),
                                            TensorValue(&compression_type),
                                            TensorValue(&buffer_size)};
  std::unique_ptr<OpKernelContext> text_line_dataset_context;
  TF_ASSERT_OK(CreateFixedLengthRecordDatasetContext(
      text_line_dataset_kernel.get(), &inputs, &text_line_dataset_context));

  DatasetBase* text_line_dataset;
  TF_ASSERT_OK(CreateDataset(text_line_dataset_kernel.get(),
                             text_line_dataset_context.get(),
                             &text_line_dataset));
  core::ScopedUnref scoped_unref(text_line_dataset);

  std::unique_ptr<SerializationContext> serialization_context;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_context));
  VariantTensorData data;
  VariantTensorDataWriter writer(&data);
  TF_ASSERT_OK(text_line_dataset->Save(serialization_context.get(), &writer));
  TF_ASSERT_OK(writer.Flush());
}

TEST_P(ParameterizedFixedLengthRecordDatasetOpTest, IteratorOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::vector<string> multi_texts = ToStringVector(test_case.expected_outputs);
  TF_ASSERT_OK(CreateMultiTextFiles(
      test_case.filenames, multi_texts, test_case.buffer_size,
      test_case.buffer_size, test_case.compression_type));

  std::unique_ptr<OpKernel> text_line_dataset_kernel;
  TF_ASSERT_OK(
      CreateFixedLengthRecordDatasetOpKernel(&text_line_dataset_kernel));

  int64 num_files = test_case.filenames.size();
  Tensor filenames =
      CreateTensor<string>(TensorShape({num_files}), test_case.filenames);
  Tensor compression_type = CreateTensor<string>(
      TensorShape({}), {CompressionName(test_case.compression_type)});
  Tensor buffer_size =
      CreateTensor<int64>(TensorShape({}), {test_case.buffer_size});
  gtl::InlinedVector<TensorValue, 4> inputs{TensorValue(&filenames),
                                            TensorValue(&compression_type),
                                            TensorValue(&buffer_size)};
  std::unique_ptr<OpKernelContext> text_line_dataset_context;
  TF_ASSERT_OK(CreateFixedLengthRecordDatasetContext(
      text_line_dataset_kernel.get(), &inputs, &text_line_dataset_context));

  DatasetBase* text_line_dataset;
  TF_ASSERT_OK(CreateDataset(text_line_dataset_kernel.get(),
                             text_line_dataset_context.get(),
                             &text_line_dataset));
  core::ScopedUnref scoped_unref(text_line_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(text_line_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(text_line_dataset->MakeIterator(iterator_ctx.get(),
                                               kIteratorPrefix, &iterator));

  TF_EXPECT_OK(VerifyTypesMatch(iterator->output_dtypes(),
                                test_case.expected_output_dtypes));
}

TEST_P(ParameterizedFixedLengthRecordDatasetOpTest, IteratorOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::vector<string> multi_texts = ToStringVector(test_case.expected_outputs);
  TF_ASSERT_OK(CreateMultiTextFiles(
      test_case.filenames, multi_texts, test_case.buffer_size,
      test_case.buffer_size, test_case.compression_type));

  std::unique_ptr<OpKernel> text_line_dataset_kernel;
  TF_ASSERT_OK(
      CreateFixedLengthRecordDatasetOpKernel(&text_line_dataset_kernel));

  int64 num_files = test_case.filenames.size();
  Tensor filenames =
      CreateTensor<string>(TensorShape({num_files}), test_case.filenames);
  Tensor compression_type = CreateTensor<string>(
      TensorShape({}), {CompressionName(test_case.compression_type)});
  Tensor buffer_size =
      CreateTensor<int64>(TensorShape({}), {test_case.buffer_size});
  gtl::InlinedVector<TensorValue, 4> inputs{TensorValue(&filenames),
                                            TensorValue(&compression_type),
                                            TensorValue(&buffer_size)};
  std::unique_ptr<OpKernelContext> text_line_dataset_context;
  TF_ASSERT_OK(CreateFixedLengthRecordDatasetContext(
      text_line_dataset_kernel.get(), &inputs, &text_line_dataset_context));

  DatasetBase* text_line_dataset;
  TF_ASSERT_OK(CreateDataset(text_line_dataset_kernel.get(),
                             text_line_dataset_context.get(),
                             &text_line_dataset));
  core::ScopedUnref scoped_unref(text_line_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(text_line_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(text_line_dataset->MakeIterator(iterator_ctx.get(),
                                               kIteratorPrefix, &iterator));

  TF_EXPECT_OK(VerifyShapesCompatible(iterator->output_shapes(),
                                      test_case.expected_output_shapes));
}

TEST_P(ParameterizedFixedLengthRecordDatasetOpTest, IteratorOutputPrefix) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::vector<string> multi_texts = ToStringVector(test_case.expected_outputs);
  TF_ASSERT_OK(CreateMultiTextFiles(
      test_case.filenames, multi_texts, test_case.buffer_size,
      test_case.buffer_size, test_case.compression_type));

  std::unique_ptr<OpKernel> text_line_dataset_kernel;
  TF_ASSERT_OK(
      CreateFixedLengthRecordDatasetOpKernel(&text_line_dataset_kernel));

  int64 num_files = test_case.filenames.size();
  Tensor filenames =
      CreateTensor<string>(TensorShape({num_files}), test_case.filenames);
  Tensor compression_type = CreateTensor<string>(
      TensorShape({}), {CompressionName(test_case.compression_type)});
  Tensor buffer_size =
      CreateTensor<int64>(TensorShape({}), {test_case.buffer_size});
  gtl::InlinedVector<TensorValue, 4> inputs{TensorValue(&filenames),
                                            TensorValue(&compression_type),
                                            TensorValue(&buffer_size)};
  std::unique_ptr<OpKernelContext> text_line_dataset_context;
  TF_ASSERT_OK(CreateFixedLengthRecordDatasetContext(
      text_line_dataset_kernel.get(), &inputs, &text_line_dataset_context));

  DatasetBase* text_line_dataset;
  TF_ASSERT_OK(CreateDataset(text_line_dataset_kernel.get(),
                             text_line_dataset_context.get(),
                             &text_line_dataset));
  core::ScopedUnref scoped_unref(text_line_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(text_line_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(text_line_dataset->MakeIterator(iterator_ctx.get(),
                                               kIteratorPrefix, &iterator));

  EXPECT_EQ(iterator->prefix(),
            name_utils::IteratorPrefix(FixedLengthRecordDatasetOp::kDatasetType,
                                       kIteratorPrefix));
}

TEST_P(ParameterizedFixedLengthRecordDatasetOpTest, Roundtrip) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::vector<string> multi_texts = ToStringVector(test_case.expected_outputs);
  TF_ASSERT_OK(CreateMultiTextFiles(
      test_case.filenames, multi_texts, test_case.buffer_size,
      test_case.buffer_size, test_case.compression_type));

  std::unique_ptr<OpKernel> text_line_dataset_kernel;
  TF_ASSERT_OK(
      CreateFixedLengthRecordDatasetOpKernel(&text_line_dataset_kernel));

  int64 num_files = test_case.filenames.size();
  Tensor filenames =
      CreateTensor<string>(TensorShape({num_files}), test_case.filenames);
  Tensor compression_type = CreateTensor<string>(
      TensorShape({}), {CompressionName(test_case.compression_type)});
  Tensor buffer_size =
      CreateTensor<int64>(TensorShape({}), {test_case.buffer_size});
  gtl::InlinedVector<TensorValue, 4> inputs{TensorValue(&filenames),
                                            TensorValue(&compression_type),
                                            TensorValue(&buffer_size)};
  std::unique_ptr<OpKernelContext> text_line_dataset_context;
  TF_ASSERT_OK(CreateFixedLengthRecordDatasetContext(
      text_line_dataset_kernel.get(), &inputs, &text_line_dataset_context));

  DatasetBase* text_line_dataset;
  TF_ASSERT_OK(CreateDataset(text_line_dataset_kernel.get(),
                             text_line_dataset_context.get(),
                             &text_line_dataset));
  core::ScopedUnref scoped_unref(text_line_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(text_line_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(text_line_dataset->MakeIterator(iterator_ctx.get(),
                                               kIteratorPrefix, &iterator));

  std::unique_ptr<SerializationContext> serialization_ctx;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_ctx));

  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  int cur_iteration = 0;
  const std::vector<int>& breakpoints = test_case.breakpoints;
  for (int breakpoint : breakpoints) {
    VariantTensorData data;
    VariantTensorDataWriter writer(&data);
    TF_EXPECT_OK(iterator->Save(serialization_ctx.get(), &writer));
    TF_EXPECT_OK(writer.Flush());
    VariantTensorDataReader reader(&data);
    TF_EXPECT_OK(RestoreIterator(iterator_ctx.get(), &reader, kIteratorPrefix,
                                 *text_line_dataset, &iterator));

    while (cur_iteration <= breakpoint) {
      std::vector<Tensor> next;
      TF_EXPECT_OK(
          iterator->GetNext(iterator_ctx.get(), &next, &end_of_sequence));
      out_tensors.insert(out_tensors.end(), next.begin(), next.end());
      cur_iteration++;
    }
  }

  TF_EXPECT_OK(ExpectEqual(out_tensors, test_case.expected_outputs,
                           /*compare_order*/ true));
}

INSTANTIATE_TEST_SUITE_P(FixedLengthRecordDatasetOpTest,
                         ParameterizedFixedLengthRecordDatasetOpTest,
                         ::testing::ValuesIn(std::vector<TestCase>(
                             {TestCase1(), TestCase2(), TestCase3()})));

}  // namespace
}  // namespace data
}  // namespace tensorflow
