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
#include "tensorflow/core/kernels/data/padded_batch_dataset_op.h"

#include "tensorflow/core/data/dataset_test_base.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "padded_batch_dataset";
constexpr int kOpVersion = 2;

class PaddedBatchDatasetOpTest : public DatasetOpsTestBase {};

class PaddedBatchDatasetParams : public DatasetParams {
 public:
  template <typename T>
  PaddedBatchDatasetParams(T input_dataset_params, int64_t batch_size,
                           std::vector<Tensor> padded_shapes,
                           std::vector<Tensor> padded_values,
                           bool drop_remainder, bool parallel_copy,
                           DataTypeVector output_dtypes,
                           std::vector<PartialTensorShape> output_shapes,
                           int num_padded_shapes, string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        batch_size_(batch_size),
        padded_shapes_(std::move(padded_shapes)),
        padded_values_(std::move(padded_values)),
        drop_remainder_(drop_remainder),
        parallel_copy_(parallel_copy),
        num_padded_shapes_(num_padded_shapes) {
    input_dataset_params_.push_back(std::make_unique<T>(input_dataset_params));
    op_version_ = kOpVersion;
    iterator_prefix_ =
        name_utils::IteratorPrefix(input_dataset_params.dataset_type(),
                                   input_dataset_params.iterator_prefix());
  }

  std::vector<Tensor> GetInputTensors() const override {
    std::vector<Tensor> input_tensors;
    input_tensors.emplace_back(
        CreateTensor<int64_t>(TensorShape({}), {batch_size_}));
    for (auto& padded_shape : padded_shapes_) {
      input_tensors.emplace_back(padded_shape);
    }
    for (auto& padded_value : padded_values_) {
      input_tensors.emplace_back(padded_value);
    }
    input_tensors.emplace_back(
        CreateTensor<bool>(TensorShape({}), {drop_remainder_}));
    return input_tensors;
  }

  Status GetInputNames(std::vector<string>* input_names) const override {
    *input_names = {PaddedBatchDatasetOp::kInputDataset,
                    PaddedBatchDatasetOp::kBatchSize};
    // Create the input names for the input padded_shapes.
    for (int i = 0; i < num_padded_shapes_; ++i) {
      input_names->emplace_back(
          strings::StrCat(PaddedBatchDatasetOp::kPaddedShapes, "_", i));
    }
    // Create the input names for the input padding_values.
    for (int j = 0; j < padded_values_.size(); ++j) {
      input_names->emplace_back(
          strings::StrCat(PaddedBatchDatasetOp::kPaddingValues, "_", j));
    }
    input_names->push_back(PaddedBatchDatasetOp::kDropRemainder);
    return OkStatus();
  }

  Status GetAttributes(AttributeVector* attr_vector) const override {
    *attr_vector = {{"parallel_copy", parallel_copy_},
                    {"Toutput_types", output_dtypes_},
                    {"output_shapes", output_shapes_},
                    {"N", num_padded_shapes_},
                    {"metadata", ""}};
    return OkStatus();
  }

  string dataset_type() const override {
    return PaddedBatchDatasetOp::kDatasetType;
  }

 private:
  int64_t batch_size_;
  std::vector<Tensor> padded_shapes_;
  std::vector<Tensor> padded_values_;
  bool drop_remainder_;
  bool parallel_copy_;
  int num_padded_shapes_;
};

// Test case 1: input elements with same shapes.
PaddedBatchDatasetParams PaddedBatchDatasetParams1() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(
          TensorShape{7, 2}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13})},
      /*node_name=*/"tensor_slice");
  return PaddedBatchDatasetParams(
      /*input_dataset_params=*/tensor_slice_dataset_params,
      /*batch_size=*/2,
      /*padded_shapes=*/{CreateTensor<int64_t>(TensorShape{1}, {3})},
      /*padded_values=*/{CreateTensor<int64_t>(TensorShape{}, {1})},
      /*drop_remainder=*/true,
      /*parallel_copy=*/true,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({2, 3})},
      /*num_padded_shapes=*/1,
      /*node_name=*/kNodeName);
}

// Test case 2: input elements with different shapes.
PaddedBatchDatasetParams PaddedBatchDatasetParams2() {
  auto tensor_slice_dataset_params_0 = TensorSliceDatasetParams(
      /*components=*/CreateTensors<int64_t>(TensorShape{3, 2},
                                            {{0, 1, 2, 3, 4, 5}}),
      /*node_name=*/"tensor_slice_0");
  auto tensor_slice_dataset_params_1 = TensorSliceDatasetParams(
      /*components=*/CreateTensors<int64_t>(TensorShape{4, 1}, {{6, 7, 8, 9}}),
      /*node_name=*/"tensor_slice_1");
  auto concatenate_dataset_params =
      ConcatenateDatasetParams(std::move(tensor_slice_dataset_params_0),
                               std::move(tensor_slice_dataset_params_1),
                               /*output_dtypes=*/{DT_INT64},
                               /*output_shapes=*/{PartialTensorShape({-1})},
                               /*node_name=*/"concatenate");
  return PaddedBatchDatasetParams(
      /*input_dataset_params=*/concatenate_dataset_params,
      /*batch_size=*/2,
      /*padded_shapes=*/{CreateTensor<int64_t>(TensorShape{1}, {3})},
      /*padded_values=*/{CreateTensor<int64_t>(TensorShape{}, {1})},
      /*drop_remainder=*/true,
      /*parallel_copy=*/true,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({2, 3})},
      /*num_padded_shapes=*/1,
      /*node_name=*/kNodeName);
}

// Test case 3: similar with the test case 2 but drop_remainder = false.
PaddedBatchDatasetParams PaddedBatchDatasetParams3() {
  auto tensor_slice_dataset_params_0 = TensorSliceDatasetParams(
      /*components=*/CreateTensors<int64_t>(TensorShape{3, 2},
                                            {{0, 1, 2, 3, 4, 5}}),
      /*node_name=*/"tensor_slice_0");
  auto tensor_slice_dataset_params_1 = TensorSliceDatasetParams(
      /*components=*/CreateTensors<int64_t>(TensorShape{4, 1}, {{6, 7, 8, 9}}),
      /*node_name=*/"tensor_slice_1");
  auto concatenate_dataset_params =
      ConcatenateDatasetParams(std::move(tensor_slice_dataset_params_0),
                               std::move(tensor_slice_dataset_params_1),
                               /*output_dtypes=*/{DT_INT64},
                               /*output_shapes=*/{PartialTensorShape({-1})},
                               /*node_name=*/"concatenate");
  return PaddedBatchDatasetParams(
      /*input_dataset_params=*/concatenate_dataset_params,
      /*batch_size=*/2,
      /*padded_shapes=*/{CreateTensor<int64_t>(TensorShape{1}, {3})},
      /*padded_values=*/{CreateTensor<int64_t>(TensorShape{}, {1})},
      /*drop_remainder=*/false,
      /*parallel_copy=*/true,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({2, 3})},
      /*num_padded_shapes=*/1,
      /*node_name=*/kNodeName);
}

// Test case 4: similar with the test case 3 but the input elements can be
// divided by the batch size evenly. As drop_remainder = false, the output
// shape is still {-1, 3} instead of {2, 3}.
PaddedBatchDatasetParams PaddedBatchDatasetParams4() {
  auto tensor_slice_dataset_params_0 = TensorSliceDatasetParams(
      /*components=*/CreateTensors<int64_t>(TensorShape{3, 2},
                                            {{0, 1, 2, 3, 4, 5}}),
      /*node_name=*/"tensor_slice_0");
  auto tensor_slice_dataset_params_1 = TensorSliceDatasetParams(
      /*components=*/CreateTensors<int64_t>(TensorShape{3, 1}, {{6, 7, 8}}),
      /*node_name=*/"tensor_slice_1");
  auto concatenate_dataset_params =
      ConcatenateDatasetParams(std::move(tensor_slice_dataset_params_0),
                               std::move(tensor_slice_dataset_params_1),
                               /*output_dtypes=*/{DT_INT64},
                               /*output_shapes=*/{PartialTensorShape({-1})},
                               /*node_name=*/"concatenate");
  return PaddedBatchDatasetParams(
      /*input_dataset_params=*/concatenate_dataset_params,
      /*batch_size=*/2,
      /*padded_shapes=*/{CreateTensor<int64_t>(TensorShape{1}, {3})},
      /*padded_values=*/{CreateTensor<int64_t>(TensorShape{}, {1})},
      /*drop_remainder=*/false,
      /*parallel_copy=*/true,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({-1, 3})},
      /*num_padded_shapes=*/1,
      /*node_name=*/kNodeName);
}

// Test case 5: similar with the test case 3 but padded_shapes = {-1}.
PaddedBatchDatasetParams PaddedBatchDatasetParams5() {
  auto tensor_slice_dataset_params_0 = TensorSliceDatasetParams(
      /*components=*/CreateTensors<int64_t>(TensorShape{3, 2},
                                            {{0, 1, 2, 3, 4, 5}}),
      /*node_name=*/"tensor_slice_0");
  auto tensor_slice_dataset_params_1 = TensorSliceDatasetParams(
      /*components=*/CreateTensors<int64_t>(TensorShape{4, 1}, {{6, 7, 8, 9}}),
      /*node_name=*/"tensor_slice_1");
  auto concatenate_dataset_params =
      ConcatenateDatasetParams(std::move(tensor_slice_dataset_params_0),
                               std::move(tensor_slice_dataset_params_1),
                               /*output_dtypes=*/{DT_INT64},
                               /*output_shapes=*/{PartialTensorShape({-1})},
                               /*node_name=*/"concatenate");
  return PaddedBatchDatasetParams(
      /*input_dataset_params=*/concatenate_dataset_params,
      /*batch_size=*/2,
      /*padded_shapes=*/{CreateTensor<int64_t>(TensorShape{1}, {-1})},
      /*padded_values=*/{CreateTensor<int64_t>(TensorShape{}, {1})},
      /*drop_remainder=*/false,
      /*parallel_copy=*/false,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({-1, -1})},
      /*num_padded_shapes=*/1,
      /*node_name=*/kNodeName);
}

// Test case 6: similar with the test case 5 but parallel_copy = true.
PaddedBatchDatasetParams PaddedBatchDatasetParams6() {
  auto tensor_slice_dataset_params_0 = TensorSliceDatasetParams(
      /*components=*/CreateTensors<int64_t>(TensorShape{3, 2},
                                            {{0, 1, 2, 3, 4, 5}}),
      /*node_name=*/"tensor_slice_0");
  auto tensor_slice_dataset_params_1 = TensorSliceDatasetParams(
      /*components=*/CreateTensors<int64_t>(TensorShape{4, 1}, {{6, 7, 8, 9}}),
      /*node_name=*/"tensor_slice_1");
  auto concatenate_dataset_params =
      ConcatenateDatasetParams(std::move(tensor_slice_dataset_params_0),
                               std::move(tensor_slice_dataset_params_1),
                               /*output_dtypes=*/{DT_INT64},
                               /*output_shapes=*/{PartialTensorShape({-1})},
                               /*node_name=*/"concatenate");
  return PaddedBatchDatasetParams(
      /*input_dataset_params=*/concatenate_dataset_params,
      /*batch_size=*/2,
      /*padded_shapes=*/{CreateTensor<int64_t>(TensorShape{1}, {-1})},
      /*padded_values=*/{CreateTensor<int64_t>(TensorShape{}, {1})},
      /*drop_remainder=*/false,
      /*parallel_copy=*/true,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({-1, -1})},
      /*num_padded_shapes=*/1,
      /*node_name=*/kNodeName);
}

// Test case 7: empty input elements.
PaddedBatchDatasetParams PaddedBatchDatasetParams7() {
  return PaddedBatchDatasetParams(
      /*input_dataset_params=*/RangeDatasetParams(0, 0, 1),
      /*batch_size=*/2,
      /*padded_shapes=*/{CreateTensor<int64_t>(TensorShape{1}, {-1})},
      /*padded_values=*/{CreateTensor<int64_t>(TensorShape{}, {1})},
      /*drop_remainder=*/false,
      /*parallel_copy=*/true,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({-1, -1})},
      /*num_padded_shapes=*/1,
      /*node_name=*/kNodeName);
}

// Test case 8: short padding shape.
PaddedBatchDatasetParams PaddedBatchDatasetParamsWithShortPaddingShape() {
  auto tensor_slice_dataset_params_0 = TensorSliceDatasetParams(
      /*components=*/CreateTensors<int64_t>(TensorShape{3, 2},
                                            {{0, 1, 2, 3, 4, 5}}),
      /*node_name=*/"tensor_slice_0");
  auto tensor_slice_dataset_params_1 = TensorSliceDatasetParams(
      /*components=*/CreateTensors<int64_t>(TensorShape{3, 2},
                                            {{6, 7, 8, 9, 10, 11}}),
      /*node_name=*/"tensor_slice_1");
  auto concatenate_dataset_params =
      ConcatenateDatasetParams(std::move(tensor_slice_dataset_params_0),
                               std::move(tensor_slice_dataset_params_1),
                               /*output_dtypes=*/{DT_INT64},
                               /*output_shapes=*/{PartialTensorShape({2})},
                               /*node_name=*/"concatenate");
  return PaddedBatchDatasetParams(
      /*input_dataset_params=*/concatenate_dataset_params,
      /*batch_size=*/2,
      /*padded_shapes=*/{CreateTensor<int64_t>(TensorShape{1}, {1})},
      /*padded_values=*/{CreateTensor<int64_t>(TensorShape{}, {1})},
      /*drop_remainder=*/false,
      /*parallel_copy=*/true,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({-1, -1})},
      /*num_padded_shapes=*/1,
      /*node_name=*/kNodeName);
}

PaddedBatchDatasetParams PaddedBatchDatasetParamsWithInvalidPaddingShape() {
  auto tensor_slice_dataset_params_0 = TensorSliceDatasetParams(
      /*components=*/CreateTensors<int64_t>(TensorShape{3, 2},
                                            {{0, 1, 2, 3, 4, 5}}),
      /*node_name=*/"tensor_slice_0");
  auto tensor_slice_dataset_params_1 = TensorSliceDatasetParams(
      /*components=*/CreateTensors<int64_t>(TensorShape{3, 2},
                                            {{6, 7, 8, 9, 10, 11}}),
      /*node_name=*/"tensor_slice_1");
  auto concatenate_dataset_params =
      ConcatenateDatasetParams(std::move(tensor_slice_dataset_params_0),
                               std::move(tensor_slice_dataset_params_1),
                               /*output_dtypes=*/{DT_INT64},
                               /*output_shapes=*/{PartialTensorShape({2})},
                               /*node_name=*/"concatenate");
  return PaddedBatchDatasetParams(
      /*input_dataset_params=*/concatenate_dataset_params,
      /*batch_size=*/2,
      /*padded_shapes=*/{CreateTensor<int64_t>(TensorShape{2}, {1, 2})},
      /*padded_values=*/{CreateTensor<int64_t>(TensorShape{}, {1})},
      /*drop_remainder=*/false,
      /*parallel_copy=*/true,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({-1, -1})},
      /*num_padded_shapes=*/1,
      /*node_name=*/kNodeName);
}

PaddedBatchDatasetParams PaddedBatchDatasetParamsWithInvalidBatchSize() {
  auto tensor_slice_dataset_params_0 = TensorSliceDatasetParams(
      /*components=*/CreateTensors<int64_t>(TensorShape{3, 2},
                                            {{0, 1, 2, 3, 4, 5}}),
      /*node_name=*/"tensor_slice_0");
  auto tensor_slice_dataset_params_1 = TensorSliceDatasetParams(
      /*components=*/CreateTensors<int64_t>(TensorShape{3, 2},
                                            {{6, 7, 8, 9, 10, 11}}),
      /*node_name=*/"tensor_slice_1");
  auto concatenate_dataset_params =
      ConcatenateDatasetParams(std::move(tensor_slice_dataset_params_0),
                               std::move(tensor_slice_dataset_params_1),
                               /*output_dtypes=*/{DT_INT64},
                               /*output_shapes=*/{PartialTensorShape({2})},
                               /*node_name=*/"concatenate");
  return PaddedBatchDatasetParams(
      /*input_dataset_params=*/concatenate_dataset_params,
      /*batch_size=*/-1,
      /*padded_shapes=*/{CreateTensor<int64_t>(TensorShape{1}, {3})},
      /*padded_values=*/{CreateTensor<int64_t>(TensorShape{}, {1})},
      /*drop_remainder=*/false,
      /*parallel_copy=*/true,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({-1, -1})},
      /*num_padded_shapes=*/1,
      /*node_name=*/kNodeName);
}

PaddedBatchDatasetParams
PaddedBatchDatasetParamsWithInvalidPaddingShapesSize() {
  auto tensor_slice_dataset_params_0 = TensorSliceDatasetParams(
      /*components=*/CreateTensors<int64_t>(TensorShape{3, 2},
                                            {{0, 1, 2, 3, 4, 5}}),
      /*node_name=*/"tensor_slice_0");
  auto tensor_slice_dataset_params_1 = TensorSliceDatasetParams(
      /*components=*/CreateTensors<int64_t>(TensorShape{3, 2},
                                            {{6, 7, 8, 9, 10, 11}}),
      /*node_name=*/"tensor_slice_1");
  auto concatenate_dataset_params =
      ConcatenateDatasetParams(std::move(tensor_slice_dataset_params_0),
                               std::move(tensor_slice_dataset_params_1),
                               /*output_dtypes=*/{DT_INT64},
                               /*output_shapes=*/{PartialTensorShape({2})},
                               /*node_name=*/"concatenate");
  return PaddedBatchDatasetParams(
      /*input_dataset_params=*/concatenate_dataset_params,
      /*batch_size=*/2,
      /*padded_shapes=*/
      {CreateTensor<int64_t>(TensorShape{1}, {3}),
       CreateTensor<int64_t>(TensorShape{1}, {3})},
      /*padded_values=*/{CreateTensor<int64_t>(TensorShape{}, {1})},
      /*drop_remainder=*/false,
      /*parallel_copy=*/true,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({-1, -1})},
      /*num_padded_shapes=*/2,
      /*node_name=*/kNodeName);
}

PaddedBatchDatasetParams
PaddedBatchDatasetParamsWithInvalidPaddingValuesSize() {
  auto tensor_slice_dataset_params_0 = TensorSliceDatasetParams(
      /*components=*/CreateTensors<int64_t>(TensorShape{3, 2},
                                            {{0, 1, 2, 3, 4, 5}}),
      /*node_name=*/"tensor_slice_0");
  auto tensor_slice_dataset_params_1 = TensorSliceDatasetParams(
      /*components=*/CreateTensors<int64_t>(TensorShape{3, 2},
                                            {{6, 7, 8, 9, 10, 11}}),
      /*node_name=*/"tensor_slice_1");
  auto concatenate_dataset_params =
      ConcatenateDatasetParams(std::move(tensor_slice_dataset_params_0),
                               std::move(tensor_slice_dataset_params_1),
                               /*output_dtypes=*/{DT_INT64},
                               /*output_shapes=*/{PartialTensorShape({2})},
                               /*node_name=*/"concatenate");
  return PaddedBatchDatasetParams(
      /*input_dataset_params=*/concatenate_dataset_params,
      /*batch_size=*/2,
      /*padded_shapes=*/
      {CreateTensor<int64_t>(TensorShape{1}, {3})},
      /*padded_values=*/
      {CreateTensor<int64_t>(TensorShape{}, {1}),
       CreateTensor<int64_t>(TensorShape{}, {1})},
      /*drop_remainder=*/false,
      /*parallel_copy=*/true,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({-1, -1})},
      /*num_padded_shapes=*/2,
      /*node_name=*/kNodeName);
}

PaddedBatchDatasetParams
PaddedBatchDatasetParamsWithInvalidPaddingValuesDType() {
  auto tensor_slice_dataset_params_0 = TensorSliceDatasetParams(
      /*components=*/CreateTensors<int64_t>(TensorShape{3, 2},
                                            {{0, 1, 2, 3, 4, 5}}),
      /*node_name=*/"tensor_slice_0");
  auto tensor_slice_dataset_params_1 = TensorSliceDatasetParams(
      /*components=*/CreateTensors<int64_t>(TensorShape{3, 2},
                                            {{6, 7, 8, 9, 10, 11}}),
      /*node_name=*/"tensor_slice_1");
  auto concatenate_dataset_params =
      ConcatenateDatasetParams(std::move(tensor_slice_dataset_params_0),
                               std::move(tensor_slice_dataset_params_1),
                               /*output_dtypes=*/{DT_INT64},
                               /*output_shapes=*/{PartialTensorShape({2})},
                               /*node_name=*/"concatenate");
  return PaddedBatchDatasetParams(
      /*input_dataset_params=*/concatenate_dataset_params,
      /*batch_size=*/2,
      /*padded_shapes=*/
      {CreateTensor<int64_t>(TensorShape{1}, {3})},
      /*padded_values=*/
      {CreateTensor<tstring>(TensorShape{}, {"a"})},
      /*drop_remainder=*/false,
      /*parallel_copy=*/true,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({-1, -1})},
      /*num_padded_shapes=*/1,
      /*node_name=*/kNodeName);
}

PaddedBatchDatasetParams
PaddedBatchDatasetParamsWithInvalidPaddingValuesShape() {
  auto tensor_slice_dataset_params_0 = TensorSliceDatasetParams(
      /*components=*/CreateTensors<int64_t>(TensorShape{3, 2},
                                            {{0, 1, 2, 3, 4, 5}}),
      /*node_name=*/"tensor_slice_0");
  auto tensor_slice_dataset_params_1 = TensorSliceDatasetParams(
      /*components=*/CreateTensors<int64_t>(TensorShape{3, 2},
                                            {{6, 7, 8, 9, 10, 11}}),
      /*node_name=*/"tensor_slice_1");
  auto concatenate_dataset_params =
      ConcatenateDatasetParams(std::move(tensor_slice_dataset_params_0),
                               std::move(tensor_slice_dataset_params_1),
                               /*output_dtypes=*/{DT_INT64},
                               /*output_shapes=*/{PartialTensorShape({2})},
                               /*node_name=*/"concatenate");
  return PaddedBatchDatasetParams(
      /*input_dataset_params=*/concatenate_dataset_params,
      /*batch_size=*/2,
      /*padded_shapes=*/
      {CreateTensor<int64_t>(TensorShape{1}, {3})},
      /*padded_values=*/
      {CreateTensor<int64_t>(TensorShape{1}, {1})},
      /*drop_remainder=*/false,
      /*parallel_copy=*/true,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({-1, -1})},
      /*num_padded_shapes=*/1,
      /*node_name=*/kNodeName);
}

std::vector<GetNextTestCase<PaddedBatchDatasetParams>> GetNextTestCases() {
  return {{/*dataset_params=*/PaddedBatchDatasetParams1(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(
               TensorShape{2, 3},
               {{0, 1, 1, 2, 3, 1}, {4, 5, 1, 6, 7, 1}, {8, 9, 1, 10, 11, 1}})},
          {/*dataset_params=*/PaddedBatchDatasetParams2(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(
               TensorShape{2, 3},
               {{0, 1, 1, 2, 3, 1}, {4, 5, 1, 6, 1, 1}, {7, 1, 1, 8, 1, 1}})},
          {/*dataset_params=*/PaddedBatchDatasetParams3(),
           /*expected_outputs=*/
           {CreateTensor<int64_t>(TensorShape{2, 3}, {0, 1, 1, 2, 3, 1}),
            CreateTensor<int64_t>(TensorShape{2, 3}, {4, 5, 1, 6, 1, 1}),
            CreateTensor<int64_t>(TensorShape{2, 3}, {7, 1, 1, 8, 1, 1}),
            CreateTensor<int64_t>(TensorShape{1, 3}, {9, 1, 1})}},
          {/*dataset_params=*/PaddedBatchDatasetParams4(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(
               TensorShape{2, 3},
               {{0, 1, 1, 2, 3, 1}, {4, 5, 1, 6, 1, 1}, {7, 1, 1, 8, 1, 1}})},
          {/*dataset_params=*/PaddedBatchDatasetParams5(),
           /*expected_outputs=*/
           {CreateTensor<int64_t>(TensorShape{2, 2}, {0, 1, 2, 3}),
            CreateTensor<int64_t>(TensorShape{2, 2}, {4, 5, 6, 1}),
            CreateTensor<int64_t>(TensorShape{2, 1}, {7, 8}),
            CreateTensor<int64_t>(TensorShape{1, 1}, {9})}},
          {/*dataset_params=*/PaddedBatchDatasetParams6(),
           /*expected_outputs=*/
           {CreateTensor<int64_t>(TensorShape{2, 2}, {0, 1, 2, 3}),
            CreateTensor<int64_t>(TensorShape{2, 2}, {4, 5, 6, 1}),
            CreateTensor<int64_t>(TensorShape{2, 1}, {7, 8}),
            CreateTensor<int64_t>(TensorShape{1, 1}, {9})}},
          {/*dataset_params=*/PaddedBatchDatasetParams7(),
           /*expected_outputs=*/{}}};
}

ITERATOR_GET_NEXT_TEST_P(PaddedBatchDatasetOpTest, PaddedBatchDatasetParams,
                         GetNextTestCases())

TEST_F(PaddedBatchDatasetOpTest, DatasetNodeName) {
  auto dataset_params = PaddedBatchDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(dataset_params.node_name()));
}

TEST_F(PaddedBatchDatasetOpTest, DatasetTypeString) {
  auto dataset_params = PaddedBatchDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  name_utils::OpNameParams params;
  params.op_version = dataset_params.op_version();
  TF_ASSERT_OK(CheckDatasetTypeString(
      name_utils::OpName(PaddedBatchDatasetOp::kDatasetType, params)));
}

std::vector<DatasetOutputDtypesTestCase<PaddedBatchDatasetParams>>
DatasetOutputDtypesTestCases() {
  return {{/*dataset_params=*/PaddedBatchDatasetParams1(),
           /*expected_output_dtypes=*/{DT_INT64}},
          {/*dataset_params=*/PaddedBatchDatasetParams2(),
           /*expected_output_dtypes=*/{DT_INT64}},
          {/*dataset_params=*/PaddedBatchDatasetParams3(),
           /*expected_output_dtypes=*/{DT_INT64}},
          {/*dataset_params=*/PaddedBatchDatasetParams4(),
           /*expected_output_dtypes=*/{DT_INT64}},
          {/*dataset_params=*/PaddedBatchDatasetParams5(),
           /*expected_output_dtypes=*/{DT_INT64}},
          {/*dataset_params=*/PaddedBatchDatasetParams6(),
           /*expected_output_dtypes=*/{DT_INT64}},
          {/*dataset_params=*/PaddedBatchDatasetParams7(),
           /*expected_output_dtypes=*/{DT_INT64}}};
}

DATASET_OUTPUT_DTYPES_TEST_P(PaddedBatchDatasetOpTest, PaddedBatchDatasetParams,
                             DatasetOutputDtypesTestCases())

std::vector<DatasetOutputShapesTestCase<PaddedBatchDatasetParams>>
DatasetOutputShapesTestCases() {
  return {{/*dataset_params=*/PaddedBatchDatasetParams1(),
           /*expected_output_shapes=*/{PartialTensorShape({2, 3})}},
          {/*dataset_params=*/PaddedBatchDatasetParams2(),
           /*expected_output_shapes=*/{PartialTensorShape({2, 3})}},
          {/*dataset_params=*/PaddedBatchDatasetParams3(),
           /*expected_output_shapes=*/{PartialTensorShape({-1, 3})}},
          {/*dataset_params=*/PaddedBatchDatasetParams4(),
           /*expected_output_shapes=*/{PartialTensorShape({-1, 3})}},
          {/*dataset_params=*/PaddedBatchDatasetParams5(),
           /*expected_output_shapes=*/{PartialTensorShape({-1, -1})}},
          {/*dataset_params=*/PaddedBatchDatasetParams6(),
           /*expected_output_shapes=*/{PartialTensorShape({-1, -1})}},
          {/*dataset_params=*/PaddedBatchDatasetParams7(),
           /*expected_output_shapes=*/{PartialTensorShape({-1, -1})}}};
}

DATASET_OUTPUT_SHAPES_TEST_P(PaddedBatchDatasetOpTest, PaddedBatchDatasetParams,
                             DatasetOutputShapesTestCases())

std::vector<CardinalityTestCase<PaddedBatchDatasetParams>>
CardinalityTestCases() {
  return {{/*dataset_params=*/PaddedBatchDatasetParams1(),
           /*expected_cardinality=*/3},
          {/*dataset_params=*/PaddedBatchDatasetParams2(),
           /*expected_cardinality=*/3},
          {/*dataset_params=*/PaddedBatchDatasetParams3(),
           /*expected_cardinality=*/4},
          {/*dataset_params=*/PaddedBatchDatasetParams4(),
           /*expected_cardinality=*/3},
          {/*dataset_params=*/PaddedBatchDatasetParams5(),
           /*expected_cardinality=*/4},
          {/*dataset_params=*/PaddedBatchDatasetParams6(),
           /*expected_cardinality=*/4},
          {/*dataset_params=*/PaddedBatchDatasetParams7(),
           /*expected_cardinality=*/0}};
}

DATASET_CARDINALITY_TEST_P(PaddedBatchDatasetOpTest, PaddedBatchDatasetParams,
                           CardinalityTestCases())

std::vector<IteratorOutputDtypesTestCase<PaddedBatchDatasetParams>>
IteratorOutputDtypesTestCases() {
  return {{/*dataset_params=*/PaddedBatchDatasetParams1(),
           /*expected_output_dtypes=*/{DT_INT64}},
          {/*dataset_params=*/PaddedBatchDatasetParams2(),
           /*expected_output_dtypes=*/{DT_INT64}},
          {/*dataset_params=*/PaddedBatchDatasetParams3(),
           /*expected_output_dtypes=*/{DT_INT64}},
          {/*dataset_params=*/PaddedBatchDatasetParams4(),
           /*expected_output_dtypes=*/{DT_INT64}},
          {/*dataset_params=*/PaddedBatchDatasetParams5(),
           /*expected_output_dtypes=*/{DT_INT64}},
          {/*dataset_params=*/PaddedBatchDatasetParams6(),
           /*expected_output_dtypes=*/{DT_INT64}},
          {/*dataset_params=*/PaddedBatchDatasetParams7(),
           /*expected_output_dtypes=*/{DT_INT64}}};
}

ITERATOR_OUTPUT_DTYPES_TEST_P(PaddedBatchDatasetOpTest,
                              PaddedBatchDatasetParams,
                              IteratorOutputDtypesTestCases())

std::vector<IteratorOutputShapesTestCase<PaddedBatchDatasetParams>>
IteratorOutputShapesTestCases() {
  return {{/*dataset_params=*/PaddedBatchDatasetParams1(),
           /*expected_output_shapes=*/{PartialTensorShape({2, 3})}},
          {/*dataset_params=*/PaddedBatchDatasetParams2(),
           /*expected_output_shapes=*/{PartialTensorShape({2, 3})}},
          {/*dataset_params=*/PaddedBatchDatasetParams3(),
           /*expected_output_shapes=*/{PartialTensorShape({-1, 3})}},
          {/*dataset_params=*/PaddedBatchDatasetParams4(),
           /*expected_output_shapes=*/{PartialTensorShape({-1, 3})}},
          {/*dataset_params=*/PaddedBatchDatasetParams5(),
           /*expected_output_shapes=*/{PartialTensorShape({-1, -1})}},
          {/*dataset_params=*/PaddedBatchDatasetParams6(),
           /*expected_output_shapes=*/{PartialTensorShape({-1, -1})}},
          {/*dataset_params=*/PaddedBatchDatasetParams7(),
           /*expected_output_shapes=*/{PartialTensorShape({-1, -1})}}};
}

ITERATOR_OUTPUT_SHAPES_TEST_P(PaddedBatchDatasetOpTest,
                              PaddedBatchDatasetParams,
                              IteratorOutputShapesTestCases())

TEST_F(PaddedBatchDatasetOpTest, IteratorPrefix) {
  auto dataset_params = PaddedBatchDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  name_utils::IteratorPrefixParams params;
  params.op_version = dataset_params.op_version();
  TF_ASSERT_OK(CheckIteratorPrefix(
      name_utils::IteratorPrefix(PaddedBatchDatasetOp::kDatasetType,
                                 dataset_params.iterator_prefix(), params)));
}

std::vector<IteratorSaveAndRestoreTestCase<PaddedBatchDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {{/*dataset_params=*/PaddedBatchDatasetParams1(),
           /*breakpoints=*/{0, 2, 5},
           /*expected_outputs=*/
           CreateTensors<int64_t>(
               TensorShape{2, 3},
               {{0, 1, 1, 2, 3, 1}, {4, 5, 1, 6, 7, 1}, {8, 9, 1, 10, 11, 1}})},
          {/*dataset_params=*/PaddedBatchDatasetParams2(),
           /*breakpoints=*/{0, 2, 5},
           /*expected_outputs=*/
           CreateTensors<int64_t>(
               TensorShape{2, 3},
               {{0, 1, 1, 2, 3, 1}, {4, 5, 1, 6, 1, 1}, {7, 1, 1, 8, 1, 1}})},
          {/*dataset_params=*/PaddedBatchDatasetParams3(),
           /*breakpoints=*/{0, 2, 5},
           /*expected_outputs=*/
           {CreateTensor<int64_t>(TensorShape{2, 3}, {0, 1, 1, 2, 3, 1}),
            CreateTensor<int64_t>(TensorShape{2, 3}, {4, 5, 1, 6, 1, 1}),
            CreateTensor<int64_t>(TensorShape{2, 3}, {7, 1, 1, 8, 1, 1}),
            CreateTensor<int64_t>(TensorShape{1, 3}, {9, 1, 1})}},
          {/*dataset_params=*/PaddedBatchDatasetParams4(),
           /*breakpoints=*/{0, 2, 5},
           /*expected_outputs=*/
           CreateTensors<int64_t>(
               TensorShape{2, 3},
               {{0, 1, 1, 2, 3, 1}, {4, 5, 1, 6, 1, 1}, {7, 1, 1, 8, 1, 1}})},
          {/*dataset_params=*/PaddedBatchDatasetParams5(),
           /*breakpoints=*/{0, 2, 5},
           /*expected_outputs=*/
           {CreateTensor<int64_t>(TensorShape{2, 2}, {0, 1, 2, 3}),
            CreateTensor<int64_t>(TensorShape{2, 2}, {4, 5, 6, 1}),
            CreateTensor<int64_t>(TensorShape{2, 1}, {7, 8}),
            CreateTensor<int64_t>(TensorShape{1, 1}, {9})}},
          {/*dataset_params=*/PaddedBatchDatasetParams6(),
           /*breakpoints=*/{0, 2, 5},
           /*expected_outputs=*/
           {CreateTensor<int64_t>(TensorShape{2, 2}, {0, 1, 2, 3}),
            CreateTensor<int64_t>(TensorShape{2, 2}, {4, 5, 6, 1}),
            CreateTensor<int64_t>(TensorShape{2, 1}, {7, 8}),
            CreateTensor<int64_t>(TensorShape{1, 1}, {9})}},
          {/*dataset_params=*/PaddedBatchDatasetParams7(),
           /*breakpoints=*/{0, 2, 5},
           /*expected_outputs=*/{}}};
}

ITERATOR_SAVE_AND_RESTORE_TEST_P(PaddedBatchDatasetOpTest,
                                 PaddedBatchDatasetParams,
                                 IteratorSaveAndRestoreTestCases())

TEST_F(PaddedBatchDatasetOpTest, ShortPadding) {
  auto dataset_params = PaddedBatchDatasetParamsWithShortPaddingShape();
  TF_ASSERT_OK(Initialize(dataset_params));
  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  EXPECT_EQ(
      iterator_->GetNext(iterator_ctx_.get(), &out_tensors, &end_of_sequence)
          .code(),
      tensorflow::error::DATA_LOSS);
}

TEST_F(PaddedBatchDatasetOpTest, InvalidPaddedShapes) {
  auto dataset_params = PaddedBatchDatasetParamsWithInvalidPaddingShape();
  TF_ASSERT_OK(Initialize(dataset_params));
  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  EXPECT_EQ(
      iterator_->GetNext(iterator_ctx_.get(), &out_tensors, &end_of_sequence)
          .code(),
      absl::StatusCode::kInvalidArgument);
}

class ParameterizedInvalidArgumentTest
    : public PaddedBatchDatasetOpTest,
      public ::testing::WithParamInterface<PaddedBatchDatasetParams> {};

TEST_P(ParameterizedInvalidArgumentTest, InvalidPredicateFunc) {
  auto dataset_params = GetParam();
  EXPECT_EQ(Initialize(dataset_params).code(),
            absl::StatusCode::kInvalidArgument);
}

INSTANTIATE_TEST_SUITE_P(
    PaddedBatchDatasetOpTest, ParameterizedInvalidArgumentTest,
    ::testing::ValuesIn(
        {PaddedBatchDatasetParamsWithInvalidBatchSize(),
         PaddedBatchDatasetParamsWithInvalidPaddingShapesSize(),
         PaddedBatchDatasetParamsWithInvalidPaddingValuesSize(),
         PaddedBatchDatasetParamsWithInvalidPaddingValuesDType(),
         PaddedBatchDatasetParamsWithInvalidPaddingValuesShape()}));

}  // namespace
}  // namespace data
}  // namespace tensorflow
