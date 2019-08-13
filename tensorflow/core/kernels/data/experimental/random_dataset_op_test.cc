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
#include "tensorflow/core/kernels/data/experimental/random_dataset_op.h"

#include "tensorflow/core/kernels/data/dataset_test_base.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/random/random_distributions.h"


namespace tensorflow {
namespace data {
namespace experimental {
namespace {

constexpr char kNodeName[] = "sampling_dataset";
constexpr char kIteratorPrefix[] = "Iterator";

// Number of random samples generated per test
constexpr int kCount = 10;

// Generate the sequence of numbers that the kernel should produce for a given
// seed. 
// For compatibility with the test harness, return value is a vector of scalar
// Tensors.
std::vector<Tensor> GenerateExpectedData(int64 seed, int64 seed2, int count) {
  std::vector<Tensor> ret;
  auto parent_generator = random::PhiloxRandom(seed, seed2);
  auto generator = random::SingleSampleAdapter<random::PhiloxRandom>(&parent_generator);

  for (int i = 0; i < count; ++i) {
    ret.push_back(CreateTensor<int64>(TensorShape({}), { generator() }));
  }

  return ret;
}

// Main test harness class for all tests in this file
class RandomDatasetOpTest : public DatasetOpsTestBase {
protected:
  // Creates a new `RandomDataset` op kernel.
  // Doesn't initialize the random seeds because they are inputs, not attributes.
  Status CreateRandomDatasetOpKernel(
      const DataTypeVector& output_types,
      const std::vector<PartialTensorShape>& output_shapes,
      std::unique_ptr<OpKernel>* random_dataset_op_kernel) {
    NodeDef node_def = test::function::NDef(
        kNodeName, name_utils::OpName(RandomDatasetOp::kDatasetType),
        // Inputs
        {RandomDatasetOp::kSeed, RandomDatasetOp::kSeed2},
        // Attributes
        {{RandomDatasetOp::kOutputTypes, output_types},
         {RandomDatasetOp::kOutputShapes, output_shapes}});
    TF_RETURN_IF_ERROR(CreateOpKernel(node_def, random_dataset_op_kernel));
    return Status::OK();
  }

  // Creates an OpKernel context suitable for running a `RandomDataset`
  // kernel.
  Status CreateRandomDatasetContext(
      OpKernel* const op_kernel,
      gtl::InlinedVector<TensorValue, 4>* const inputs,
      std::unique_ptr<OpKernelContext>* context) {
    TF_RETURN_IF_ERROR(CheckOpKernelInput(*op_kernel, *inputs));
    TF_RETURN_IF_ERROR(CreateOpKernelContext(op_kernel, inputs, context));
    return Status::OK();
  }
};


class RandomDatasetParams : public DatasetParams {
  public:
    RandomDatasetParams(int64 seed, int64 seed2,
                        DataTypeVector output_dtypes,
                        std::vector<PartialTensorShape> output_shapes,
                        string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
                      seed(CreateTensor<int64>(TensorShape({}), {seed})),
                      seed2(CreateTensor<int64>(TensorShape({}), {seed2})) {}

  
    Status MakeInputs(gtl::InlinedVector<TensorValue, 4>* inputs) override {
      *inputs = {TensorValue(&seed), TensorValue(&seed2)}; return Status::OK();
    }

    Tensor seed;
    Tensor seed2;
};


RandomDatasetParams FortyTwo() {
  return {/*seed*/ 42,
          /*seed2*/ 42,
          /*output_dtypes*/ {DT_INT64},
          /*output_shapes*/ {PartialTensorShape({})},
          /*node_name=*/kNodeName};
}

// Change just first seed relative to FortyTwo
RandomDatasetParams ChangeSeed() {
  return {/*seed*/ 1000,
          /*seed2*/ 42,
          /*output_dtypes*/ {DT_INT64},
          /*output_shapes*/ {PartialTensorShape({})},
          /*node_name=*/kNodeName};
}

// Change just second seed relative to FortyTwo
RandomDatasetParams ChangeSeed2() {
  return {/*seed*/ 42,
          /*seed2*/ 1000,
          /*output_dtypes*/ {DT_INT64},
          /*output_shapes*/ {PartialTensorShape({})},
          /*node_name=*/kNodeName};
}

class ParameterizedGetNextRandomDatasetOpTest
    : public RandomDatasetOpTest,
      public ::testing::WithParamInterface<
          GetNextTestCase<RandomDatasetParams>> {};


GetNextTestCase<RandomDatasetParams> GetNextFortyTwo() {
  return {/*dataset_params=*/FortyTwo(),
          /*expected_outputs=*/GenerateExpectedData(42, 42, kCount)};
}

GetNextTestCase<RandomDatasetParams> GetNextChangeSeed() {
  return {/*dataset_params=*/ChangeSeed(),
          /*expected_outputs=*/GenerateExpectedData(1000, 42, kCount)};
}

GetNextTestCase<RandomDatasetParams> GetNextChangeSeed2() {
  return {/*dataset_params=*/ChangeSeed2(),
          /*expected_outputs=*/GenerateExpectedData(42, 1000, kCount)};
}


TEST_P(ParameterizedGetNextRandomDatasetOpTest, GetNext) {
  // BEGIN INITIALIZATION CODE
  // Step 1: Set up enough of a TF runtime to be able to invoke a kernel.
  const int thread_num = 2, cpu_num = 2;
  auto test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  // Step 2: Box up the four inputs to the kernel inside TensorValue objects
  // inside a vector.
  gtl::InlinedVector<TensorValue, 4> inputs;
  TF_ASSERT_OK(test_case.dataset_params.MakeInputs(&inputs));

  // Step 3: Create a RandomDataset kernel to test, passing in static
  // attributes of the kernel.
  std::unique_ptr<OpKernel> random_dataset_kernel;
  TF_ASSERT_OK(CreateRandomDatasetOpKernel(
      test_case.dataset_params.output_dtypes,
      test_case.dataset_params.output_shapes, &random_dataset_kernel));

  // Step 4: Create a context in which the kernel will operate. This is where
  // the kernel gets initialized with its inputs.
  std::unique_ptr<OpKernelContext> random_dataset_context;
  TF_ASSERT_OK(CreateRandomDatasetContext(
      random_dataset_kernel.get(), &inputs, &random_dataset_context));

  // Step 6: Unbox the DatasetBase inside the variant tensor backing the
  // kernel.
  DatasetBase* random_dataset;
  TF_ASSERT_OK(CreateDataset(random_dataset_kernel.get(),
                             random_dataset_context.get(),
                             &random_dataset));
  core::ScopedUnref scoped_unref(random_dataset);

  // Step 7: Create an iterator to read the output of the dataset.
  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(
      CreateIteratorContext(random_dataset_context.get(), &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(random_dataset->MakeIterator(iterator_context.get(),
                                            kIteratorPrefix, &iterator));

  // END INITIALIZATION CODE

  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  for (int i = 0; i < kCount; i++) {
    std::vector<Tensor> next;
    TF_EXPECT_OK(
        iterator->GetNext(iterator_context.get(), &next, &end_of_sequence));
    out_tensors.insert(out_tensors.end(), next.begin(), next.end());
    EXPECT_FALSE(end_of_sequence); // RandomDataset output never ends
  }

  TF_EXPECT_OK(ExpectEqual(out_tensors, test_case.expected_outputs,
                           /*compare_order*/ true));

}

INSTANTIATE_TEST_SUITE_P(
    RandomDatasetOpTest, ParameterizedGetNextRandomDatasetOpTest,
    ::testing::ValuesIn(std::vector<GetNextTestCase<RandomDatasetParams>>(
        {GetNextFortyTwo(), GetNextChangeSeed(), GetNextChangeSeed2()})));



}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow

