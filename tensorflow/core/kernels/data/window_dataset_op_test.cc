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
#include "tensorflow/core/kernels/data/window_dataset_op.h"

#include <string>
#include <utility>

#include "tensorflow/core/data/dataset_test_base.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/serialization_utils.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "window_dataset";

class WindowDatasetParams : public DatasetParams {
 public:
  template <typename T>
  WindowDatasetParams(T input_dataset_params, int64_t size, int64_t shift,
                      int64_t stride, bool drop_remainder,
                      DataTypeVector output_dtypes,
                      std::vector<PartialTensorShape> output_shapes,
                      string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        size_(size),
        shift_(shift),
        stride_(stride),
        drop_remainder_(drop_remainder) {
    input_dataset_params_.push_back(absl::make_unique<T>(input_dataset_params));
    iterator_prefix_ =
        name_utils::IteratorPrefix(input_dataset_params.dataset_type(),
                                   input_dataset_params.iterator_prefix());
  }

  std::vector<Tensor> GetInputTensors() const override {
    return {CreateTensor<int64_t>(TensorShape({}), {size_}),
            CreateTensor<int64_t>(TensorShape({}), {shift_}),
            CreateTensor<int64_t>(TensorShape({}), {stride_}),
            CreateTensor<bool>(TensorShape({}), {drop_remainder_})};
  }

  Status GetInputNames(std::vector<string>* input_names) const override {
    input_names->clear();
    input_names->emplace_back(WindowDatasetOp::kInputDataset);
    input_names->emplace_back(WindowDatasetOp::kSize);
    input_names->emplace_back(WindowDatasetOp::kShift);
    input_names->emplace_back(WindowDatasetOp::kStride);
    input_names->emplace_back(WindowDatasetOp::kDropRemainder);
    return Status::OK();
  }

  Status GetAttributes(AttributeVector* attr_vector) const override {
    attr_vector->clear();
    attr_vector->emplace_back("output_types", output_dtypes_);
    attr_vector->emplace_back("output_shapes", output_shapes_);
    attr_vector->emplace_back("metadata", "");
    return Status::OK();
  }

  string dataset_type() const override { return WindowDatasetOp::kDatasetType; }

 private:
  int64_t size_;
  int64_t shift_;
  int64_t stride_;
  bool drop_remainder_;
};

class WindowDatasetOpTest : public DatasetOpsTestBase {};

// Test case 1: size=2, shift=2, stride=1, drop_remainder=false.
WindowDatasetParams WindowDatasetParams1() {
  return WindowDatasetParams(RangeDatasetParams(0, 7, 1),
                             /*size=*/2,
                             /*shift=*/2,
                             /*stride=*/1,
                             /*drop_remainder=*/false,
                             /*output_dtypes=*/{DT_VARIANT},
                             /*output_shapes=*/{PartialTensorShape({})},
                             /*node_name=*/kNodeName);
}

// Test case 2: size=2, shift=2, stride=2, drop_remainder=true.
WindowDatasetParams WindowDatasetParams2() {
  return WindowDatasetParams(RangeDatasetParams(0, 7, 1),
                             /*size=*/2,
                             /*shift=*/2,
                             /*stride=*/2,
                             /*drop_remainder=*/true,
                             /*output_dtypes=*/{DT_VARIANT},
                             /*output_shapes=*/{PartialTensorShape({})},
                             /*node_name=*/kNodeName);
}

// Test case 3: size=8, shift=3, stride=1, drop_remainder=false.
WindowDatasetParams WindowDatasetParams3() {
  return WindowDatasetParams(RangeDatasetParams(0, 7, 1),
                             /*size=*/8,
                             /*shift=*/3,
                             /*stride=*/1,
                             /*drop_remainder=*/false,
                             /*output_dtypes=*/{DT_VARIANT},
                             /*output_shapes=*/{PartialTensorShape({})},
                             /*node_name=*/kNodeName);
}

// Test case 4: size=8, shift=3, stride=1, drop_remainder=true.
WindowDatasetParams WindowDatasetParams4() {
  return WindowDatasetParams(RangeDatasetParams(0, 7, 1),
                             /*size=*/8,
                             /*shift=*/3,
                             /*stride=*/1,
                             /*drop_remainder=*/true,
                             /*output_dtypes=*/{DT_VARIANT},
                             /*output_shapes=*/{PartialTensorShape({})},
                             /*node_name=*/kNodeName);
}

// Test case 5: size=2, shift=8, stride=1, drop_remainder=false.
WindowDatasetParams WindowDatasetParams5() {
  return WindowDatasetParams(RangeDatasetParams(0, 7, 1),
                             /*size=*/2,
                             /*shift=*/8,
                             /*stride=*/1,
                             /*drop_remainder=*/false,
                             /*output_dtypes=*/{DT_VARIANT},
                             /*output_shapes=*/{PartialTensorShape({})},
                             /*node_name=*/kNodeName);
}

// Test case 6: size=2, shift=8, stride=1, drop_remainder=true.
WindowDatasetParams WindowDatasetParams6() {
  return WindowDatasetParams(RangeDatasetParams(0, 7, 1),
                             /*size=*/2,
                             /*shift=*/8,
                             /*stride=*/1,
                             /*drop_remainder=*/true,
                             /*output_dtypes=*/{DT_VARIANT},
                             /*output_shapes=*/{PartialTensorShape({})},
                             /*node_name=*/kNodeName);
}

// Test case 7: size=2, shift=2, stride=8, drop_remainder=false.
WindowDatasetParams WindowDatasetParams7() {
  return WindowDatasetParams(RangeDatasetParams(0, 7, 1),
                             /*size=*/2,
                             /*shift=*/2,
                             /*stride=*/8,
                             /*drop_remainder=*/false,
                             /*output_dtypes=*/{DT_VARIANT},
                             /*output_shapes=*/{PartialTensorShape({})},
                             /*node_name=*/kNodeName);
}

// Test case 8: size=2, shift=2, stride=8, drop_remainder=true.
WindowDatasetParams WindowDatasetParams8() {
  return WindowDatasetParams(RangeDatasetParams(0, 7, 1),
                             /*size=*/2,
                             /*shift=*/2,
                             /*stride=*/8,
                             /*drop_remainder=*/true,
                             /*output_dtypes=*/{DT_VARIANT},
                             /*output_shapes=*/{PartialTensorShape({})},
                             /*node_name=*/kNodeName);
}

// Test case 9: size=4, shift=2, stride=2, drop_remainder=true.
WindowDatasetParams WindowDatasetParams9() {
  return WindowDatasetParams(RangeDatasetParams(0, 7, 1),
                             /*size=*/4,
                             /*shift=*/2,
                             /*stride=*/2,
                             /*drop_remainder=*/true,
                             /*output_dtypes=*/{DT_VARIANT},
                             /*output_shapes=*/{PartialTensorShape({})},
                             /*node_name=*/kNodeName);
}

// Test case 10: size=5, shift=2, stride=2, drop_remainder=true.
WindowDatasetParams WindowDatasetParams10() {
  return WindowDatasetParams(RangeDatasetParams(0, 7, 1),
                             /*size=*/5,
                             /*shift=*/2,
                             /*stride=*/2,
                             /*drop_remainder=*/true,
                             /*output_dtypes=*/{DT_VARIANT},
                             /*output_shapes=*/{PartialTensorShape({})},
                             /*node_name=*/kNodeName);
}

// Test case 11: size=0, shift=2, stride=2, drop_remainder=true.
WindowDatasetParams WindowDatasetParamsWithInvalidWindowSize() {
  return WindowDatasetParams(RangeDatasetParams(0, 7, 1),
                             /*size=*/0,
                             /*shift=*/2,
                             /*stride=*/2,
                             /*drop_remainder=*/true,
                             /*output_dtypes=*/{DT_VARIANT},
                             /*output_shapes=*/{PartialTensorShape({})},
                             /*node_name=*/kNodeName);
}

// Test case 12: size=2, shift=0, stride=2, drop_remainder=true.
WindowDatasetParams WindowDatasetParamswithInvalidWindowShift() {
  return WindowDatasetParams(RangeDatasetParams(0, 7, 1),
                             /*size=*/2,
                             /*shift=*/0,
                             /*stride=*/2,
                             /*drop_remainder=*/true,
                             /*output_dtypes=*/{DT_VARIANT},
                             /*output_shapes=*/{PartialTensorShape({})},
                             /*node_name=*/kNodeName);
}

// Test case 13: size=2, shift=2, stride=0, drop_remainder=true.
WindowDatasetParams WindowDatasetParamsWithInvalidWindowStride() {
  return WindowDatasetParams(RangeDatasetParams(0, 7, 1),
                             /*size=*/2,
                             /*shift=*/2,
                             /*stride=*/0,
                             /*drop_remainder=*/true,
                             /*output_dtypes=*/{DT_VARIANT},
                             /*output_shapes=*/{PartialTensorShape({})},
                             /*node_name=*/kNodeName);
}

template <typename T>
struct GetNextTestCase {
  T dataset_params;
  std::vector<std::vector<Tensor>> expected_outputs;
};

std::vector<GetNextTestCase<WindowDatasetParams>> GetNextTestCases() {
  return {{/*dataset_params=*/WindowDatasetParams1(),
           /*expected_outputs=*/
           {CreateTensors<int64_t>(TensorShape{}, {{0}, {1}}),
            CreateTensors<int64_t>(TensorShape{}, {{2}, {3}}),
            CreateTensors<int64_t>(TensorShape{}, {{4}, {5}}),
            CreateTensors<int64_t>(TensorShape{}, {{6}})}},
          {/*dataset_params=*/WindowDatasetParams2(),
           /*expected_outputs=*/
           {CreateTensors<int64_t>(TensorShape{}, {{0}, {2}}),
            CreateTensors<int64_t>(TensorShape{}, {{2}, {4}}),
            CreateTensors<int64_t>(TensorShape{}, {{4}, {6}})}},
          {/*dataset_params=*/WindowDatasetParams3(),
           /*expected_outputs=*/
           {CreateTensors<int64_t>(TensorShape({}),
                                   {{0}, {1}, {2}, {3}, {4}, {5}, {6}}),
            CreateTensors<int64_t>(TensorShape({}), {{3}, {4}, {5}, {6}}),
            CreateTensors<int64_t>(TensorShape({}), {{6}})}},
          {/*dataset_params=*/WindowDatasetParams4(),
           /*expected_outputs=*/{}},
          {/*dataset_params=*/WindowDatasetParams5(),
           /*expected_outputs=*/
           {CreateTensors<int64_t>(TensorShape({}), {{0}, {1}})}},
          {/*dataset_params=*/WindowDatasetParams6(),
           /*expected_outputs=*/
           {CreateTensors<int64_t>(TensorShape({}), {{0}, {1}})}},
          {/*dataset_params=*/WindowDatasetParams7(),
           /*expected_outputs=*/
           {CreateTensors<int64_t>(TensorShape({}), {{0}}),
            CreateTensors<int64_t>(TensorShape({}), {{2}}),
            CreateTensors<int64_t>(TensorShape({}), {{4}}),
            CreateTensors<int64_t>(TensorShape({}), {{6}})}},
          {/*dataset_params=*/WindowDatasetParams8(),
           /*expected_outputs=*/{}},
          {/*dataset_params=*/WindowDatasetParams9(),
           /*expected_outputs=*/
           {CreateTensors<int64_t>(TensorShape({}), {{0}, {2}, {4}, {6}})}},
          {/*dataset_params=*/WindowDatasetParams10(),
           /*expected_outputs=*/{}}};
}

class ParameterizedGetNextTest : public WindowDatasetOpTest,
                                 public ::testing::WithParamInterface<
                                     GetNextTestCase<WindowDatasetParams>> {};

TEST_P(ParameterizedGetNextTest, GetNext) {
  auto test_case = GetParam();
  TF_ASSERT_OK(Initialize(test_case.dataset_params));

  bool end_of_sequence = false;
  auto expected_outputs_it = test_case.expected_outputs.begin();
  while (!end_of_sequence) {
    // Owns the window_datasets, which are stored as the variant tensors in the
    // vector.
    std::vector<Tensor> out_tensors;
    TF_EXPECT_OK(iterator_->GetNext(iterator_ctx_.get(), &out_tensors,
                                    &end_of_sequence));
    if (!end_of_sequence) {
      for (const auto& window_dataset_tensor : out_tensors) {
        // Not owned.
        DatasetBase* window_dataset;
        TF_ASSERT_OK(GetDatasetFromVariantTensor(window_dataset_tensor,
                                                 &window_dataset));
        std::unique_ptr<IteratorBase> window_dataset_iterator;
        TF_ASSERT_OK(window_dataset->MakeIterator(
            iterator_ctx_.get(), /*parent=*/nullptr,
            test_case.dataset_params.iterator_prefix(),
            &window_dataset_iterator));
        bool end_of_window_dataset = false;
        std::vector<Tensor> window_elements;
        // Fetches all the elements in window_dataset.
        while (!end_of_window_dataset) {
          std::vector<Tensor> next_element;
          TF_EXPECT_OK(window_dataset_iterator->GetNext(
              iterator_ctx_.get(), &next_element, &end_of_window_dataset));
          window_elements.insert(window_elements.end(), next_element.begin(),
                                 next_element.end());
        }
        EXPECT_LT(expected_outputs_it, test_case.expected_outputs.end());
        TF_EXPECT_OK(ExpectEqual(window_elements, *expected_outputs_it, false));
        expected_outputs_it++;
      }
    }
  }
  EXPECT_EQ(expected_outputs_it, test_case.expected_outputs.end());
}

INSTANTIATE_TEST_CASE_P(WindowDatasetOpTest, ParameterizedGetNextTest,
                        ::testing::ValuesIn(GetNextTestCases()));

TEST_F(WindowDatasetOpTest, DatasetTypeString) {
  auto dataset_params = WindowDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetTypeString(
      name_utils::OpName(WindowDatasetOp::kDatasetType)));
}

TEST_F(WindowDatasetOpTest, DatasetNodeName) {
  auto dataset_params = WindowDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(dataset_params.node_name()));
}

TEST_F(WindowDatasetOpTest, DatasetOutputDtypes) {
  auto dataset_params = WindowDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputDtypes(dataset_params.output_dtypes()));
}

TEST_F(WindowDatasetOpTest, DatasetOutputShapes) {
  auto dataset_params = WindowDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputShapes(dataset_params.output_shapes()));
}

std::vector<CardinalityTestCase<WindowDatasetParams>>
DatasetCardinalityTestCases() {
  return {{/*dataset_params=*/WindowDatasetParams1(),
           /*expected_cardinality=*/4},
          {/*dataset_params=*/WindowDatasetParams2(),
           /*expected_cardinality=*/3},
          {/*dataset_params=*/WindowDatasetParams3(),
           /*expected_cardinality=*/3},
          {/*dataset_params=*/WindowDatasetParams4(),
           /*expected_cardinality=*/0},
          {/*dataset_params=*/WindowDatasetParams5(),
           /*expected_cardinality=*/1},
          {/*dataset_params=*/WindowDatasetParams6(),
           /*expected_cardinality=*/1},
          {/*dataset_params=*/WindowDatasetParams7(),
           /*expected_cardinality=*/4},
          {/*dataset_params=*/WindowDatasetParams8(),
           /*expected_cardinality=*/0},
          {/*dataset_params=*/WindowDatasetParams9(),
           /*expected_cardinality=*/1},
          {/*dataset_params=*/WindowDatasetParams10(),
           /*expected_cardinality=*/0}};
}

DATASET_CARDINALITY_TEST_P(WindowDatasetOpTest, WindowDatasetParams,
                           DatasetCardinalityTestCases())

TEST_F(WindowDatasetOpTest, IteratorOutputDtypes) {
  auto dataset_params = WindowDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputDtypes(dataset_params.output_dtypes()));
}

TEST_F(WindowDatasetOpTest, IteratorOutputShapes) {
  auto dataset_params = WindowDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputShapes(dataset_params.output_shapes()));
}

TEST_F(WindowDatasetOpTest, IteratorOutputPrefix) {
  auto dataset_params = WindowDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorPrefix(name_utils::IteratorPrefix(
      WindowDatasetOp::kDatasetType, dataset_params.iterator_prefix())));
}

template <typename T>
struct IteratorSaveAndRestoreTestCase {
  T dataset_params;
  std::vector<int> breakpoints;
  std::vector<std::vector<Tensor>> expected_outputs;
};

std::vector<IteratorSaveAndRestoreTestCase<WindowDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {{/*dataset_params=*/WindowDatasetParams1(),
           /*breakpoints=*/{0, 1, 9},
           /*expected_outputs=*/
           {CreateTensors<int64_t>(TensorShape{}, {{0}, {1}}),
            CreateTensors<int64_t>(TensorShape{}, {{2}, {3}}),
            CreateTensors<int64_t>(TensorShape{}, {{4}, {5}}),
            CreateTensors<int64_t>(TensorShape{}, {{6}})}},
          {/*dataset_params=*/WindowDatasetParams2(),
           /*breakpoints=*/{0, 1, 9},
           /*expected_outputs=*/
           {CreateTensors<int64_t>(TensorShape{}, {{0}, {2}}),
            CreateTensors<int64_t>(TensorShape{}, {{2}, {4}}),
            CreateTensors<int64_t>(TensorShape{}, {{4}, {6}})}},
          {/*dataset_params=*/WindowDatasetParams3(),
           /*breakpoints=*/{0, 1, 9},
           /*expected_outputs=*/
           {CreateTensors<int64_t>(TensorShape({}),
                                   {{0}, {1}, {2}, {3}, {4}, {5}, {6}}),
            CreateTensors<int64_t>(TensorShape({}), {{3}, {4}, {5}, {6}}),
            CreateTensors<int64_t>(TensorShape({}), {{6}})}},
          {/*dataset_params=*/WindowDatasetParams4(),
           /*breakpoints=*/{0, 1, 9},
           /*expected_outputs=*/{}},
          {/*dataset_params=*/WindowDatasetParams5(),
           /*breakpoints=*/{0, 1, 9},
           /*expected_outputs=*/
           {CreateTensors<int64_t>(TensorShape({}), {{0}, {1}})}},
          {/*dataset_params=*/WindowDatasetParams6(),
           /*breakpoints=*/{0, 1, 9},
           /*expected_outputs=*/
           {CreateTensors<int64_t>(TensorShape({}), {{0}, {1}})}},
          {/*dataset_params=*/WindowDatasetParams7(),
           /*breakpoints=*/{0, 1, 9},
           /*expected_outputs=*/
           {CreateTensors<int64_t>(TensorShape({}), {{0}}),
            CreateTensors<int64_t>(TensorShape({}), {{2}}),
            CreateTensors<int64_t>(TensorShape({}), {{4}}),
            CreateTensors<int64_t>(TensorShape({}), {{6}})}},
          {/*dataset_params=*/WindowDatasetParams8(),
           /*breakpoints=*/{0, 1, 9},
           /*expected_outputs=*/{}},
          {/*dataset_params=*/WindowDatasetParams9(),
           /*breakpoints=*/{0, 1, 9},
           /*expected_outputs=*/
           {CreateTensors<int64_t>(TensorShape({}), {{0}, {2}, {4}, {6}})}},
          {/*dataset_params=*/WindowDatasetParams10(),
           /*breakpoints=*/{0, 1, 9},
           /*expected_outputs=*/{}}};
}

class ParameterizedIteratorSaveAndRestoreTest
    : public WindowDatasetOpTest,
      public ::testing::WithParamInterface<
          IteratorSaveAndRestoreTestCase<WindowDatasetParams>> {};

TEST_P(ParameterizedIteratorSaveAndRestoreTest, IteratorSaveAndRestore) {
  auto test_case = GetParam();
  TF_ASSERT_OK(Initialize(test_case.dataset_params));

  std::unique_ptr<SerializationContext> serialization_ctx;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_ctx));

  bool end_of_sequence = false;
  auto expected_outputs_it = test_case.expected_outputs.begin();
  int cur_iteration = 0;
  for (int breakpoint : test_case.breakpoints) {
    VariantTensorDataWriter writer;
    TF_EXPECT_OK(iterator_->Save(serialization_ctx.get(), &writer));
    std::vector<const VariantTensorData*> data;
    writer.GetData(&data);
    VariantTensorDataReader reader(data);
    TF_EXPECT_OK(RestoreIterator(iterator_ctx_.get(), &reader,
                                 test_case.dataset_params.iterator_prefix(),
                                 *dataset_, &iterator_));
    while (cur_iteration <= breakpoint) {
      while (!end_of_sequence) {
        // Owns the datasets, which are stored as the variant tensors in the
        // vector.
        std::vector<Tensor> out_tensors;
        TF_EXPECT_OK(iterator_->GetNext(iterator_ctx_.get(), &out_tensors,
                                        &end_of_sequence));
        if (!end_of_sequence) {
          for (const auto& window_dataset_tensor : out_tensors) {
            // Not owned.
            DatasetBase* window_dataset;
            TF_ASSERT_OK(GetDatasetFromVariantTensor(window_dataset_tensor,
                                                     &window_dataset));
            std::unique_ptr<IteratorBase> window_dataset_iterator;
            TF_ASSERT_OK(window_dataset->MakeIterator(
                iterator_ctx_.get(), /*parent=*/nullptr,
                test_case.dataset_params.iterator_prefix(),
                &window_dataset_iterator));
            bool end_of_window_dataset = false;
            std::vector<Tensor> window_elements;
            while (!end_of_window_dataset) {
              std::vector<Tensor> next_element;
              TF_EXPECT_OK(window_dataset_iterator->GetNext(
                  iterator_ctx_.get(), &next_element, &end_of_window_dataset));
              window_elements.insert(window_elements.end(),
                                     next_element.begin(), next_element.end());
            }
            EXPECT_LT(expected_outputs_it, test_case.expected_outputs.end());
            TF_EXPECT_OK(
                ExpectEqual(window_elements, *expected_outputs_it, false));
            expected_outputs_it++;
          }
        }
      }
      cur_iteration++;
    }
  }
  EXPECT_EQ(expected_outputs_it, test_case.expected_outputs.end());
}

INSTANTIATE_TEST_CASE_P(WindowDatasetOpTest,
                        ParameterizedIteratorSaveAndRestoreTest,
                        ::testing::ValuesIn(IteratorSaveAndRestoreTestCases()));

TEST_F(WindowDatasetOpTest, InvalidArguments) {
  std::vector<WindowDatasetParams> dataset_params_vec(
      {WindowDatasetParamsWithInvalidWindowSize(),
       WindowDatasetParamswithInvalidWindowShift(),
       WindowDatasetParamsWithInvalidWindowStride()});
  for (const auto& dataset_params : dataset_params_vec) {
    EXPECT_EQ(Initialize(dataset_params).code(),
              tensorflow::error::INVALID_ARGUMENT);
  }
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
