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
#include "tensorflow/core/kernels/data/fixed_length_record_dataset_op.h"

#include "tensorflow/core/kernels/data/dataset_test_base.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "fixed_length_record_dataset";
constexpr char kIteratorPrefix[] = "Iterator";
constexpr int kOpVersion = 2;

class FixedLengthRecordDatasetOpTest : public DatasetOpsTestBase {
 protected:
  // Create a new `TextLineDataset` op kernel.
  Status CreateFixedLengthRecordDatasetOpKernel(
      std::unique_ptr<OpKernel>* fixed_length_record_dataset_op_kernel) {
    name_utils::OpNameParams params;
    params.op_version = kOpVersion;
    NodeDef node_def = test::function::NDef(
        kNodeName,
        name_utils::OpName(FixedLengthRecordDatasetOp::kDatasetType, params),
        {FixedLengthRecordDatasetOp::kFileNames,
         FixedLengthRecordDatasetOp::kHeaderBytes,
         FixedLengthRecordDatasetOp::kRecordBytes,
         FixedLengthRecordDatasetOp::kFooterBytes,
         FixedLengthRecordDatasetOp::kBufferSize,
         FixedLengthRecordDatasetOp::kCompressionType},
        {});
    TF_RETURN_IF_ERROR(
        CreateOpKernel(node_def, fixed_length_record_dataset_op_kernel));
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
  std::vector<string> contents;
  int64 header_bytes;
  int64 record_bytes;
  int64 footer_bytes;
  int64 buffer_size;
  CompressionType compression_type;
  std::vector<Tensor> expected_outputs;
  DataTypeVector expected_output_dtypes;
  std::vector<PartialTensorShape> expected_output_shapes;
  int64 expected_cardinality;
  std::vector<int> breakpoints;
};

Status CreateTestFiles(const TestCase& test_case) {
  if (test_case.filenames.size() != test_case.contents.size()) {
    return tensorflow::errors::InvalidArgument(
        "The number of files does not match with the contents");
  }
  if (test_case.compression_type == CompressionType::UNCOMPRESSED) {
    for (int i = 0; i < test_case.filenames.size(); ++i) {
      TF_RETURN_IF_ERROR(WriteDataToFile(test_case.filenames[i],
                                         test_case.contents[i].data()));
    }
  } else {
    CompressionParams params;
    params.compression_type = test_case.compression_type;
    params.input_buffer_size = test_case.buffer_size;
    params.output_buffer_size = test_case.buffer_size;
    for (int i = 0; i < test_case.filenames.size(); ++i) {
      TF_RETURN_IF_ERROR(WriteDataToFile(test_case.filenames[i],
                                         test_case.contents[i].data(), params));
    }
  }
  return Status::OK();
}

// Test case 1: multiple fixed-length record files with ZLIB compression.
TestCase TestCase1() {
  return {/*filenames*/ {absl::StrCat(testing::TmpDir(), "/text_line_ZLIB_1"),
                         absl::StrCat(testing::TmpDir(), "/text_line_ZLIB_2")},
          /*contents*/
          {absl::StrCat("HHHHH", "111", "222", "333", "FF"),
           absl::StrCat("HHHHH", "aaa", "bbb", "FF")},
          /*header_bytes*/ 5,
          /*record_bytes*/ 3,
          /*footer_bytes*/ 2,
          /*buffer_size*/ 10,
          /*compression_type*/ CompressionType::ZLIB,
          /*expected_outputs*/
          {CreateTensor<string>(TensorShape({}), {"111"}),
           CreateTensor<string>(TensorShape({}), {"222"}),
           CreateTensor<string>(TensorShape({}), {"333"}),
           CreateTensor<string>(TensorShape({}), {"aaa"}),
           CreateTensor<string>(TensorShape({}), {"bbb"})},
          /*expected_output_dtypes*/ {DT_STRING},
          /*expected_output_shapes*/ {PartialTensorShape({})},
          /*expected_cardinality*/ kUnknownCardinality,
          /*breakpoints*/ {0, 2, 6}};
}

// Test case 2: multiple fixed-length record files with GZIP compression.
TestCase TestCase2() {
  return {/*filenames*/ {absl::StrCat(testing::TmpDir(), "/text_line_GZIP_1"),
                         absl::StrCat(testing::TmpDir(), "/text_line_GZIP_2")},
          /*contents*/
          {absl::StrCat("HHHHH", "111", "222", "333", "FF"),
           absl::StrCat("HHHHH", "aaa", "bbb", "FF")},
          /*header_bytes*/ 5,
          /*record_bytes*/ 3,
          /*footer_bytes*/ 2,
          /*buffer_size*/ 10,
          /*compression_type*/ CompressionType::GZIP,
          /*expected_outputs*/
          {CreateTensor<string>(TensorShape({}), {"111"}),
           CreateTensor<string>(TensorShape({}), {"222"}),
           CreateTensor<string>(TensorShape({}), {"333"}),
           CreateTensor<string>(TensorShape({}), {"aaa"}),
           CreateTensor<string>(TensorShape({}), {"bbb"})},
          /*expected_output_dtypes*/ {DT_STRING},
          /*expected_output_shapes*/ {PartialTensorShape({})},
          /*expected_cardinality*/ kUnknownCardinality,
          /*breakpoints*/ {0, 2, 6}};
}

// Test case 3: multiple fixed-length record files without compression.
TestCase TestCase3() {
  return {/*filenames*/ {
              absl::StrCat(testing::TmpDir(), "/text_line_UNCOMPRESSED_1"),
              absl::StrCat(testing::TmpDir(), "/text_line_UNCOMPRESSED_2")},
          /*contents*/
          {absl::StrCat("HHHHH", "111", "222", "333", "FF"),
           absl::StrCat("HHHHH", "aaa", "bbb", "FF")},
          /*header_bytes*/ 5,
          /*record_bytes*/ 3,
          /*footer_bytes*/ 2,
          /*buffer_size*/ 10,
          /*compression_type*/ CompressionType::UNCOMPRESSED,
          /*expected_outputs*/
          {CreateTensor<string>(TensorShape({}), {"111"}),
           CreateTensor<string>(TensorShape({}), {"222"}),
           CreateTensor<string>(TensorShape({}), {"333"}),
           CreateTensor<string>(TensorShape({}), {"aaa"}),
           CreateTensor<string>(TensorShape({}), {"bbb"})},
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

  TF_ASSERT_OK(CreateTestFiles(test_case));

  std::unique_ptr<OpKernel> fixed_length_record_dataset_kernel;
  TF_ASSERT_OK(CreateFixedLengthRecordDatasetOpKernel(
      &fixed_length_record_dataset_kernel));

  int64 num_files = test_case.filenames.size();
  Tensor filenames =
      CreateTensor<string>(TensorShape({num_files}), test_case.filenames);
  Tensor header_bytes =
      CreateTensor<int64>(TensorShape({}), {test_case.header_bytes});
  Tensor record_bytes =
      CreateTensor<int64>(TensorShape({}), {test_case.record_bytes});
  Tensor footer_bytes =
      CreateTensor<int64>(TensorShape({}), {test_case.footer_bytes});
  Tensor buffer_size =
      CreateTensor<int64>(TensorShape({}), {test_case.buffer_size});
  Tensor compression_type = CreateTensor<string>(
      TensorShape({}), {ToString(test_case.compression_type)});
  gtl::InlinedVector<TensorValue, 4> inputs{
      TensorValue(&filenames),    TensorValue(&header_bytes),
      TensorValue(&record_bytes), TensorValue(&footer_bytes),
      TensorValue(&buffer_size),  TensorValue(&compression_type),
  };
  std::unique_ptr<OpKernelContext> fixed_length_record_dataset_context;
  TF_ASSERT_OK(CreateFixedLengthRecordDatasetContext(
      fixed_length_record_dataset_kernel.get(), &inputs,
      &fixed_length_record_dataset_context));

  DatasetBase* fixed_length_record_dataset;
  TF_ASSERT_OK(CreateDataset(fixed_length_record_dataset_kernel.get(),
                             fixed_length_record_dataset_context.get(),
                             &fixed_length_record_dataset));
  core::ScopedUnref scoped_unref(fixed_length_record_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(CreateIteratorContext(fixed_length_record_dataset_context.get(),
                                     &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(fixed_length_record_dataset->MakeIterator(
      iterator_ctx.get(), kIteratorPrefix, &iterator));
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

  TF_ASSERT_OK(CreateTestFiles(test_case));

  std::unique_ptr<OpKernel> fixed_length_record_dataset_kernel;
  TF_ASSERT_OK(CreateFixedLengthRecordDatasetOpKernel(
      &fixed_length_record_dataset_kernel));

  int64 num_files = test_case.filenames.size();
  Tensor filenames =
      CreateTensor<string>(TensorShape({num_files}), test_case.filenames);
  Tensor header_bytes =
      CreateTensor<int64>(TensorShape({}), {test_case.header_bytes});
  Tensor record_bytes =
      CreateTensor<int64>(TensorShape({}), {test_case.record_bytes});
  Tensor footer_bytes =
      CreateTensor<int64>(TensorShape({}), {test_case.footer_bytes});
  Tensor buffer_size =
      CreateTensor<int64>(TensorShape({}), {test_case.buffer_size});
  Tensor compression_type = CreateTensor<string>(
      TensorShape({}), {ToString(test_case.compression_type)});
  gtl::InlinedVector<TensorValue, 4> inputs{
      TensorValue(&filenames),    TensorValue(&header_bytes),
      TensorValue(&record_bytes), TensorValue(&footer_bytes),
      TensorValue(&buffer_size),  TensorValue(&compression_type),
  };
  std::unique_ptr<OpKernelContext> fixed_length_record_dataset_context;
  TF_ASSERT_OK(CreateFixedLengthRecordDatasetContext(
      fixed_length_record_dataset_kernel.get(), &inputs,
      &fixed_length_record_dataset_context));

  DatasetBase* fixed_length_record_dataset;
  TF_ASSERT_OK(CreateDataset(fixed_length_record_dataset_kernel.get(),
                             fixed_length_record_dataset_context.get(),
                             &fixed_length_record_dataset));
  core::ScopedUnref scoped_unref(fixed_length_record_dataset);
  EXPECT_EQ(fixed_length_record_dataset->node_name(), kNodeName);
}

TEST_F(FixedLengthRecordDatasetOpTest, DatasetTypeString) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = TestCase1();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  TF_ASSERT_OK(CreateTestFiles(test_case));

  std::unique_ptr<OpKernel> fixed_length_record_dataset_kernel;
  TF_ASSERT_OK(CreateFixedLengthRecordDatasetOpKernel(
      &fixed_length_record_dataset_kernel));

  int64 num_files = test_case.filenames.size();
  Tensor filenames =
      CreateTensor<string>(TensorShape({num_files}), test_case.filenames);
  Tensor header_bytes =
      CreateTensor<int64>(TensorShape({}), {test_case.header_bytes});
  Tensor record_bytes =
      CreateTensor<int64>(TensorShape({}), {test_case.record_bytes});
  Tensor footer_bytes =
      CreateTensor<int64>(TensorShape({}), {test_case.footer_bytes});
  Tensor buffer_size =
      CreateTensor<int64>(TensorShape({}), {test_case.buffer_size});
  Tensor compression_type = CreateTensor<string>(
      TensorShape({}), {ToString(test_case.compression_type)});
  gtl::InlinedVector<TensorValue, 4> inputs{
      TensorValue(&filenames),    TensorValue(&header_bytes),
      TensorValue(&record_bytes), TensorValue(&footer_bytes),
      TensorValue(&buffer_size),  TensorValue(&compression_type),
  };
  std::unique_ptr<OpKernelContext> fixed_length_record_dataset_context;
  TF_ASSERT_OK(CreateFixedLengthRecordDatasetContext(
      fixed_length_record_dataset_kernel.get(), &inputs,
      &fixed_length_record_dataset_context));

  DatasetBase* fixed_length_record_dataset;
  TF_ASSERT_OK(CreateDataset(fixed_length_record_dataset_kernel.get(),
                             fixed_length_record_dataset_context.get(),
                             &fixed_length_record_dataset));
  core::ScopedUnref scoped_unref(fixed_length_record_dataset);
  name_utils::OpNameParams params;
  params.op_version = kOpVersion;
  EXPECT_EQ(
      fixed_length_record_dataset->type_string(),
      name_utils::OpName(FixedLengthRecordDatasetOp::kDatasetType, params));
}

TEST_P(ParameterizedFixedLengthRecordDatasetOpTest, DatasetOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  TF_ASSERT_OK(CreateTestFiles(test_case));

  std::unique_ptr<OpKernel> fixed_length_record_dataset_kernel;
  TF_ASSERT_OK(CreateFixedLengthRecordDatasetOpKernel(
      &fixed_length_record_dataset_kernel));

  int64 num_files = test_case.filenames.size();
  Tensor filenames =
      CreateTensor<string>(TensorShape({num_files}), test_case.filenames);
  Tensor header_bytes =
      CreateTensor<int64>(TensorShape({}), {test_case.header_bytes});
  Tensor record_bytes =
      CreateTensor<int64>(TensorShape({}), {test_case.record_bytes});
  Tensor footer_bytes =
      CreateTensor<int64>(TensorShape({}), {test_case.footer_bytes});
  Tensor buffer_size =
      CreateTensor<int64>(TensorShape({}), {test_case.buffer_size});
  Tensor compression_type = CreateTensor<string>(
      TensorShape({}), {ToString(test_case.compression_type)});
  gtl::InlinedVector<TensorValue, 4> inputs{
      TensorValue(&filenames),    TensorValue(&header_bytes),
      TensorValue(&record_bytes), TensorValue(&footer_bytes),
      TensorValue(&buffer_size),  TensorValue(&compression_type),
  };
  std::unique_ptr<OpKernelContext> fixed_length_record_dataset_context;
  TF_ASSERT_OK(CreateFixedLengthRecordDatasetContext(
      fixed_length_record_dataset_kernel.get(), &inputs,
      &fixed_length_record_dataset_context));

  DatasetBase* fixed_length_record_dataset;
  TF_ASSERT_OK(CreateDataset(fixed_length_record_dataset_kernel.get(),
                             fixed_length_record_dataset_context.get(),
                             &fixed_length_record_dataset));
  core::ScopedUnref scoped_unref(fixed_length_record_dataset);
  TF_EXPECT_OK(VerifyTypesMatch(fixed_length_record_dataset->output_dtypes(),
                                test_case.expected_output_dtypes));
}

TEST_P(ParameterizedFixedLengthRecordDatasetOpTest, DatasetOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  TF_ASSERT_OK(CreateTestFiles(test_case));

  std::unique_ptr<OpKernel> fixed_length_record_dataset_kernel;
  TF_ASSERT_OK(CreateFixedLengthRecordDatasetOpKernel(
      &fixed_length_record_dataset_kernel));

  int64 num_files = test_case.filenames.size();
  Tensor filenames =
      CreateTensor<string>(TensorShape({num_files}), test_case.filenames);
  Tensor header_bytes =
      CreateTensor<int64>(TensorShape({}), {test_case.header_bytes});
  Tensor record_bytes =
      CreateTensor<int64>(TensorShape({}), {test_case.record_bytes});
  Tensor footer_bytes =
      CreateTensor<int64>(TensorShape({}), {test_case.footer_bytes});
  Tensor buffer_size =
      CreateTensor<int64>(TensorShape({}), {test_case.buffer_size});
  Tensor compression_type = CreateTensor<string>(
      TensorShape({}), {ToString(test_case.compression_type)});
  gtl::InlinedVector<TensorValue, 4> inputs{
      TensorValue(&filenames),    TensorValue(&header_bytes),
      TensorValue(&record_bytes), TensorValue(&footer_bytes),
      TensorValue(&buffer_size),  TensorValue(&compression_type),
  };
  std::unique_ptr<OpKernelContext> fixed_length_record_dataset_context;
  TF_ASSERT_OK(CreateFixedLengthRecordDatasetContext(
      fixed_length_record_dataset_kernel.get(), &inputs,
      &fixed_length_record_dataset_context));

  DatasetBase* fixed_length_record_dataset;
  TF_ASSERT_OK(CreateDataset(fixed_length_record_dataset_kernel.get(),
                             fixed_length_record_dataset_context.get(),
                             &fixed_length_record_dataset));
  core::ScopedUnref scoped_unref(fixed_length_record_dataset);
  TF_EXPECT_OK(
      VerifyShapesCompatible(fixed_length_record_dataset->output_shapes(),
                             test_case.expected_output_shapes));
}

TEST_P(ParameterizedFixedLengthRecordDatasetOpTest, Cardinality) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  TF_ASSERT_OK(CreateTestFiles(test_case));

  std::unique_ptr<OpKernel> fixed_length_record_dataset_kernel;
  TF_ASSERT_OK(CreateFixedLengthRecordDatasetOpKernel(
      &fixed_length_record_dataset_kernel));

  int64 num_files = test_case.filenames.size();
  Tensor filenames =
      CreateTensor<string>(TensorShape({num_files}), test_case.filenames);
  Tensor header_bytes =
      CreateTensor<int64>(TensorShape({}), {test_case.header_bytes});
  Tensor record_bytes =
      CreateTensor<int64>(TensorShape({}), {test_case.record_bytes});
  Tensor footer_bytes =
      CreateTensor<int64>(TensorShape({}), {test_case.footer_bytes});
  Tensor buffer_size =
      CreateTensor<int64>(TensorShape({}), {test_case.buffer_size});
  Tensor compression_type = CreateTensor<string>(
      TensorShape({}), {ToString(test_case.compression_type)});
  gtl::InlinedVector<TensorValue, 4> inputs{
      TensorValue(&filenames),    TensorValue(&header_bytes),
      TensorValue(&record_bytes), TensorValue(&footer_bytes),
      TensorValue(&buffer_size),  TensorValue(&compression_type),
  };
  std::unique_ptr<OpKernelContext> fixed_length_record_dataset_context;
  TF_ASSERT_OK(CreateFixedLengthRecordDatasetContext(
      fixed_length_record_dataset_kernel.get(), &inputs,
      &fixed_length_record_dataset_context));

  DatasetBase* fixed_length_record_dataset;
  TF_ASSERT_OK(CreateDataset(fixed_length_record_dataset_kernel.get(),
                             fixed_length_record_dataset_context.get(),
                             &fixed_length_record_dataset));
  core::ScopedUnref scoped_unref(fixed_length_record_dataset);
  EXPECT_EQ(fixed_length_record_dataset->Cardinality(),
            test_case.expected_cardinality);
}

TEST_P(ParameterizedFixedLengthRecordDatasetOpTest, IteratorOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  TF_ASSERT_OK(CreateTestFiles(test_case));

  std::unique_ptr<OpKernel> fixed_length_record_dataset_kernel;
  TF_ASSERT_OK(CreateFixedLengthRecordDatasetOpKernel(
      &fixed_length_record_dataset_kernel));

  int64 num_files = test_case.filenames.size();
  Tensor filenames =
      CreateTensor<string>(TensorShape({num_files}), test_case.filenames);
  Tensor header_bytes =
      CreateTensor<int64>(TensorShape({}), {test_case.header_bytes});
  Tensor record_bytes =
      CreateTensor<int64>(TensorShape({}), {test_case.record_bytes});
  Tensor footer_bytes =
      CreateTensor<int64>(TensorShape({}), {test_case.footer_bytes});
  Tensor buffer_size =
      CreateTensor<int64>(TensorShape({}), {test_case.buffer_size});
  Tensor compression_type = CreateTensor<string>(
      TensorShape({}), {ToString(test_case.compression_type)});
  gtl::InlinedVector<TensorValue, 4> inputs{
      TensorValue(&filenames),    TensorValue(&header_bytes),
      TensorValue(&record_bytes), TensorValue(&footer_bytes),
      TensorValue(&buffer_size),  TensorValue(&compression_type),
  };
  std::unique_ptr<OpKernelContext> fixed_length_record_dataset_context;
  TF_ASSERT_OK(CreateFixedLengthRecordDatasetContext(
      fixed_length_record_dataset_kernel.get(), &inputs,
      &fixed_length_record_dataset_context));

  DatasetBase* fixed_length_record_dataset;
  TF_ASSERT_OK(CreateDataset(fixed_length_record_dataset_kernel.get(),
                             fixed_length_record_dataset_context.get(),
                             &fixed_length_record_dataset));
  core::ScopedUnref scoped_unref(fixed_length_record_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(CreateIteratorContext(fixed_length_record_dataset_context.get(),
                                     &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(fixed_length_record_dataset->MakeIterator(
      iterator_ctx.get(), kIteratorPrefix, &iterator));

  TF_EXPECT_OK(VerifyTypesMatch(iterator->output_dtypes(),
                                test_case.expected_output_dtypes));
}

TEST_P(ParameterizedFixedLengthRecordDatasetOpTest, IteratorOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  TF_ASSERT_OK(CreateTestFiles(test_case));

  std::unique_ptr<OpKernel> fixed_length_record_dataset_kernel;
  TF_ASSERT_OK(CreateFixedLengthRecordDatasetOpKernel(
      &fixed_length_record_dataset_kernel));

  int64 num_files = test_case.filenames.size();
  Tensor filenames =
      CreateTensor<string>(TensorShape({num_files}), test_case.filenames);
  Tensor header_bytes =
      CreateTensor<int64>(TensorShape({}), {test_case.header_bytes});
  Tensor record_bytes =
      CreateTensor<int64>(TensorShape({}), {test_case.record_bytes});
  Tensor footer_bytes =
      CreateTensor<int64>(TensorShape({}), {test_case.footer_bytes});
  Tensor buffer_size =
      CreateTensor<int64>(TensorShape({}), {test_case.buffer_size});
  Tensor compression_type = CreateTensor<string>(
      TensorShape({}), {ToString(test_case.compression_type)});
  gtl::InlinedVector<TensorValue, 4> inputs{
      TensorValue(&filenames),    TensorValue(&header_bytes),
      TensorValue(&record_bytes), TensorValue(&footer_bytes),
      TensorValue(&buffer_size),  TensorValue(&compression_type),
  };
  std::unique_ptr<OpKernelContext> fixed_length_record_dataset_context;
  TF_ASSERT_OK(CreateFixedLengthRecordDatasetContext(
      fixed_length_record_dataset_kernel.get(), &inputs,
      &fixed_length_record_dataset_context));

  DatasetBase* fixed_length_record_dataset;
  TF_ASSERT_OK(CreateDataset(fixed_length_record_dataset_kernel.get(),
                             fixed_length_record_dataset_context.get(),
                             &fixed_length_record_dataset));
  core::ScopedUnref scoped_unref(fixed_length_record_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(CreateIteratorContext(fixed_length_record_dataset_context.get(),
                                     &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(fixed_length_record_dataset->MakeIterator(
      iterator_ctx.get(), kIteratorPrefix, &iterator));

  TF_EXPECT_OK(VerifyShapesCompatible(iterator->output_shapes(),
                                      test_case.expected_output_shapes));
}

TEST_P(ParameterizedFixedLengthRecordDatasetOpTest, IteratorOutputPrefix) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  TF_ASSERT_OK(CreateTestFiles(test_case));

  std::unique_ptr<OpKernel> fixed_length_record_dataset_kernel;
  TF_ASSERT_OK(CreateFixedLengthRecordDatasetOpKernel(
      &fixed_length_record_dataset_kernel));

  int64 num_files = test_case.filenames.size();
  Tensor filenames =
      CreateTensor<string>(TensorShape({num_files}), test_case.filenames);
  Tensor header_bytes =
      CreateTensor<int64>(TensorShape({}), {test_case.header_bytes});
  Tensor record_bytes =
      CreateTensor<int64>(TensorShape({}), {test_case.record_bytes});
  Tensor footer_bytes =
      CreateTensor<int64>(TensorShape({}), {test_case.footer_bytes});
  Tensor buffer_size =
      CreateTensor<int64>(TensorShape({}), {test_case.buffer_size});
  Tensor compression_type = CreateTensor<string>(
      TensorShape({}), {ToString(test_case.compression_type)});
  gtl::InlinedVector<TensorValue, 4> inputs{
      TensorValue(&filenames),    TensorValue(&header_bytes),
      TensorValue(&record_bytes), TensorValue(&footer_bytes),
      TensorValue(&buffer_size),  TensorValue(&compression_type),
  };
  std::unique_ptr<OpKernelContext> fixed_length_record_dataset_context;
  TF_ASSERT_OK(CreateFixedLengthRecordDatasetContext(
      fixed_length_record_dataset_kernel.get(), &inputs,
      &fixed_length_record_dataset_context));

  DatasetBase* fixed_length_record_dataset;
  TF_ASSERT_OK(CreateDataset(fixed_length_record_dataset_kernel.get(),
                             fixed_length_record_dataset_context.get(),
                             &fixed_length_record_dataset));
  core::ScopedUnref scoped_unref(fixed_length_record_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(CreateIteratorContext(fixed_length_record_dataset_context.get(),
                                     &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(fixed_length_record_dataset->MakeIterator(
      iterator_ctx.get(), kIteratorPrefix, &iterator));
  name_utils::IteratorPrefixParams params;
  params.op_version = kOpVersion;
  EXPECT_EQ(iterator->prefix(),
            name_utils::IteratorPrefix(FixedLengthRecordDatasetOp::kDatasetType,
                                       kIteratorPrefix, params));
}

TEST_P(ParameterizedFixedLengthRecordDatasetOpTest, Roundtrip) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  TF_ASSERT_OK(CreateTestFiles(test_case));

  std::unique_ptr<OpKernel> fixed_length_record_dataset_kernel;
  TF_ASSERT_OK(CreateFixedLengthRecordDatasetOpKernel(
      &fixed_length_record_dataset_kernel));

  int64 num_files = test_case.filenames.size();
  Tensor filenames =
      CreateTensor<string>(TensorShape({num_files}), test_case.filenames);
  Tensor header_bytes =
      CreateTensor<int64>(TensorShape({}), {test_case.header_bytes});
  Tensor record_bytes =
      CreateTensor<int64>(TensorShape({}), {test_case.record_bytes});
  Tensor footer_bytes =
      CreateTensor<int64>(TensorShape({}), {test_case.footer_bytes});
  Tensor buffer_size =
      CreateTensor<int64>(TensorShape({}), {test_case.buffer_size});
  Tensor compression_type = CreateTensor<string>(
      TensorShape({}), {ToString(test_case.compression_type)});
  gtl::InlinedVector<TensorValue, 4> inputs{
      TensorValue(&filenames),    TensorValue(&header_bytes),
      TensorValue(&record_bytes), TensorValue(&footer_bytes),
      TensorValue(&buffer_size),  TensorValue(&compression_type),
  };
  std::unique_ptr<OpKernelContext> fixed_length_record_dataset_context;
  TF_ASSERT_OK(CreateFixedLengthRecordDatasetContext(
      fixed_length_record_dataset_kernel.get(), &inputs,
      &fixed_length_record_dataset_context));

  DatasetBase* fixed_length_record_dataset;
  TF_ASSERT_OK(CreateDataset(fixed_length_record_dataset_kernel.get(),
                             fixed_length_record_dataset_context.get(),
                             &fixed_length_record_dataset));
  core::ScopedUnref scoped_unref(fixed_length_record_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(CreateIteratorContext(fixed_length_record_dataset_context.get(),
                                     &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(fixed_length_record_dataset->MakeIterator(
      iterator_ctx.get(), kIteratorPrefix, &iterator));

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
                                 *fixed_length_record_dataset, &iterator));

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
