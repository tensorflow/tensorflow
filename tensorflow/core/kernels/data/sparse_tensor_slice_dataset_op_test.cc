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
#include "tensorflow/core/kernels/data/dataset_utils.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "sparse_tensor_slice_dataset";
constexpr char kDatasetType[] = "SparseTensorSlice";

class SparseTensorSliceDatasetParams : public DatasetParams {
 public:
  SparseTensorSliceDatasetParams(Tensor indices, Tensor values,
                                 Tensor dense_shape, DataType tvalues,
                                 string node_name)
      : DatasetParams({tvalues}, {PartialTensorShape({})},
                      std::move(node_name)),
        indices_(std::move(indices)),
        values_(std::move(values)),
        dense_shape_(std::move(dense_shape)),
        tvalues_(tvalues) {
    iterator_prefix_ = "Iterator";
  }

  std::vector<Tensor> GetInputTensors() const override {
    return {indices_, values_, dense_shape_};
  }

  Status GetInputNames(std::vector<string>* input_names) const override {
    input_names->clear();
    input_names->emplace_back("indices");
    input_names->emplace_back("values");
    input_names->emplace_back("dense_shape");
    return Status::OK();
  }

  Status GetAttributes(AttributeVector* attr_vector) const override {
    attr_vector->clear();
    attr_vector->emplace_back("Tvalues", tvalues_);
    return Status::OK();
  }

  string dataset_type() const override { return kDatasetType; }

 private:
  Tensor indices_;
  Tensor values_;
  Tensor dense_shape_;
  DataType tvalues_;
};

class SparseTensorSliceDatasetOpTest : public DatasetOpsTestBase {};

SparseTensorSliceDatasetParams TwoDimsSparseTensorSliceDatasetParams() {
  return SparseTensorSliceDatasetParams(
      /*indices=*/CreateTensor<int64>({2, 2}, {0, 0, 1, 1}),
      /*values=*/CreateTensor<int32>({2}, {888, 999}),
      /*dense_shape=*/CreateTensor<int64>({2}, {2, 2}),
      /*tvalues=*/DT_INT32,
      /*node_name=*/kNodeName);
}

SparseTensorSliceDatasetParams ThreeDimsSparseTensorSliceDatasetParams() {
  return SparseTensorSliceDatasetParams(
      /*indices=*/CreateTensor<int64>({2, 3}, {0, 0, 0, 1, 1, 1}),
      /*values=*/CreateTensor<double>({2}, {888.0, 999.0}),
      /*dense_shape=*/CreateTensor<int64>({3}, {2, 2, 2}),
      /*tvalues=*/DT_DOUBLE,
      /*node_name=*/kNodeName);
}

SparseTensorSliceDatasetParams FourDimsSparseTensorSliceDatasetParams() {
  return SparseTensorSliceDatasetParams(
      /*indices=*/CreateTensor<int64>({2, 4}, {0, 0, 0, 0, 1, 1, 1, 1}),
      /*values=*/CreateTensor<tstring>({2}, {"a", "b"}),
      /*dense_shape=*/CreateTensor<int64>({4}, {3, 2, 2, 2}),
      /*tvalues=*/DT_STRING,
      /*node_name=*/kNodeName);
}

SparseTensorSliceDatasetParams FiveDimsSparseTensorSliceDatasetParams() {
  return SparseTensorSliceDatasetParams(
      /*indices=*/CreateTensor<int64>({2, 5}, {0, 0, 0, 0, 0, 1, 1, 1, 1, 1}),
      /*values=*/CreateTensor<int32>({2}, {888, 999}),
      /*dense_shape=*/CreateTensor<int64>({5}, {3, 2, 2, 2, 2}),
      /*tvalues=*/DT_INT32,
      /*node_name=*/kNodeName);
}

template <typename T>
struct GetNextTestCase {
  T dataset_params;
  std::vector<std::vector<Tensor>> expected_outputs;
};

std::vector<GetNextTestCase<SparseTensorSliceDatasetParams>>
GetNextTestCases() {
  return {{/*dataset_params=*/TwoDimsSparseTensorSliceDatasetParams(),
           /*expected_outputs=*/
           {{/*indices*/ CreateTensor<int64>({1, 1}, {0}),
             /*values*/ CreateTensor<int32>({1}, {888}),
             /*dense_shape*/ CreateTensor<int64>({1}, {2})},
            {/*indices*/ CreateTensor<int64>({1, 1}, {1}),
             /*values*/ CreateTensor<int32>({1}, {999}),
             /*dense_shape*/ CreateTensor<int64>({1}, {2})}}},
          {/*dataset_params=*/ThreeDimsSparseTensorSliceDatasetParams(),
           /*expected_outputs=*/
           {{/*indices*/ CreateTensor<int64>({1, 2}, {0, 0}),
             /*values*/ CreateTensor<double>({1}, {888.0}),
             /*dense_shape*/ CreateTensor<int64>({2}, {2, 2})},
            {{/*indices*/ CreateTensor<int64>({1, 2}, {1, 1})},
             {/*values*/ CreateTensor<double>({1}, {999.0})},
             {/*dense_shape*/ CreateTensor<int64>({2}, {2, 2})}}}},
          {/*dataset_params=*/FourDimsSparseTensorSliceDatasetParams(),
           /*expected_outputs=*/
           {{/*indices*/ CreateTensor<int64>({1, 3}, {0, 0, 0}),
             /*values*/ CreateTensor<tstring>({1}, {"a"}),
             /*dense_shape*/
             CreateTensor<int64>({3}, {2, 2, 2})},
            {/*indices*/ CreateTensor<int64>({1, 3}, {1, 1, 1}),
             /*values*/ CreateTensor<tstring>({1}, {"b"}),
             /*dense_shape*/
             CreateTensor<int64>({3}, {2, 2, 2})},
            {/*indices*/ CreateTensor<int64>({0, 3}, {}),
             /*values*/ CreateTensor<tstring>({0}, {}),
             /*dense_shape*/
             CreateTensor<int64>({3}, {2, 2, 2})}}},
          {/*dataset_params=*/FiveDimsSparseTensorSliceDatasetParams(),
           /*expected_outputs=*/{
               {/*indices*/ CreateTensor<int64>({1, 4}, {0, 0, 0, 0}),
                /*values*/ CreateTensor<int32>({1}, {888}),
                /*dense_shape*/
                CreateTensor<int64>({4}, {2, 2, 2, 2})},
               {/*indices*/ CreateTensor<int64>({1, 4}, {1, 1, 1, 1}),
                /*values*/ CreateTensor<int32>({1}, {999}),
                /*dense_shape*/
                CreateTensor<int64>({4}, {2, 2, 2, 2})},
               {/*indices*/ CreateTensor<int64>({0, 4}, {}),
                /*values*/ CreateTensor<int32>({0}, {}),
                /*dense_shape*/
                CreateTensor<int64>({4}, {2, 2, 2, 2})}}}};
}

class ParameterizedGetNextTest
    : public SparseTensorSliceDatasetOpTest,
      public ::testing::WithParamInterface<
          GetNextTestCase<SparseTensorSliceDatasetParams>> {};

TEST_P(ParameterizedGetNextTest, GetNext) {
  auto test_case = GetParam();
  TF_ASSERT_OK(Initialize(test_case.dataset_params));

  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  auto expected_outputs_it = test_case.expected_outputs.begin();
  while (!end_of_sequence) {
    TF_EXPECT_OK(iterator_->GetNext(iterator_ctx_.get(), &out_tensors,
                                    &end_of_sequence));
    if (!end_of_sequence) {
      TF_EXPECT_OK(ExpectEqual(out_tensors[0], expected_outputs_it->at(0)));
      TF_EXPECT_OK(ExpectEqual(out_tensors[1], expected_outputs_it->at(1)));
      TF_EXPECT_OK(ExpectEqual(out_tensors[2], expected_outputs_it->at(2)));
      expected_outputs_it++;
    }
  }
  EXPECT_EQ(expected_outputs_it, test_case.expected_outputs.end());
}

INSTANTIATE_TEST_CASE_P(SparseTensorSliceDatasetOpTest,
                        ParameterizedGetNextTest,
                        ::testing::ValuesIn(GetNextTestCases()));

TEST_F(SparseTensorSliceDatasetOpTest, DatasetTypeString) {
  auto dataset_params = TwoDimsSparseTensorSliceDatasetParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetTypeString(name_utils::OpName(kDatasetType)));
}

TEST_F(SparseTensorSliceDatasetOpTest, DatasetNodeName) {
  auto dataset_params = TwoDimsSparseTensorSliceDatasetParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(dataset_params.node_name()));
}

std::vector<DatasetOutputDtypesTestCase<SparseTensorSliceDatasetParams>>
DatasetOutputDtypesTestCases() {
  return {{/*dataset_params=*/TwoDimsSparseTensorSliceDatasetParams(),
           /*expected_output_dtypes=*/{DT_INT64, DT_INT32, DT_INT64}},
          {/*dataset_params=*/ThreeDimsSparseTensorSliceDatasetParams(),
           /*expected_output_dtypes=*/{DT_INT64, DT_DOUBLE, DT_INT64}},
          {/*dataset_params=*/FourDimsSparseTensorSliceDatasetParams(),
           /*expected_output_dtypes=*/{DT_INT64, DT_STRING, DT_INT64}},
          {/*dataset_params=*/FiveDimsSparseTensorSliceDatasetParams(),
           /*expected_output_dtypes=*/{DT_INT64, DT_INT32, DT_INT64}}};
}

DATASET_OUTPUT_DTYPES_TEST_P(SparseTensorSliceDatasetOpTest,
                             SparseTensorSliceDatasetParams,
                             DatasetOutputDtypesTestCases())

std::vector<DatasetOutputShapesTestCase<SparseTensorSliceDatasetParams>>
DatasetOutputShapesTestCases() {
  return {{/*dataset_params=*/TwoDimsSparseTensorSliceDatasetParams(),
           /*expected_output_shapes=*/{PartialTensorShape({1, 1}),
                                       PartialTensorShape({1}),
                                       PartialTensorShape({1})}},
          {/*dataset_params=*/ThreeDimsSparseTensorSliceDatasetParams(),
           /*expected_output_shapes=*/{PartialTensorShape({1, 2}),
                                       PartialTensorShape({1}),
                                       PartialTensorShape({2})}},
          {/*dataset_params=*/FourDimsSparseTensorSliceDatasetParams(),
           /*expected_output_shapes=*/{PartialTensorShape({1, 3}),
                                       PartialTensorShape({1}),
                                       PartialTensorShape({3})}},
          {/*dataset_params=*/FiveDimsSparseTensorSliceDatasetParams(),
           /*expected_output_shapes=*/{PartialTensorShape({1, 4}),
                                       PartialTensorShape({1}),
                                       PartialTensorShape({4})}}};
}

DATASET_OUTPUT_SHAPES_TEST_P(SparseTensorSliceDatasetOpTest,
                             SparseTensorSliceDatasetParams,
                             DatasetOutputShapesTestCases())

std::vector<CardinalityTestCase<SparseTensorSliceDatasetParams>>
CardinalityTestCases() {
  return {{/*dataset_params=*/TwoDimsSparseTensorSliceDatasetParams(),
           /*expected_cardinality=*/2},
          {/*dataset_params=*/ThreeDimsSparseTensorSliceDatasetParams(),
           /*expected_cardinality=*/2},
          {/*dataset_params=*/FourDimsSparseTensorSliceDatasetParams(),
           /*expected_cardinality=*/3},
          {/*dataset_params=*/FiveDimsSparseTensorSliceDatasetParams(),
           /*expected_cardinality=*/3}};
}

DATASET_CARDINALITY_TEST_P(SparseTensorSliceDatasetOpTest,
                           SparseTensorSliceDatasetParams,
                           CardinalityTestCases())

std::vector<IteratorOutputDtypesTestCase<SparseTensorSliceDatasetParams>>
IteratorOutputDtypesTestCases() {
  return {{/*dataset_params=*/TwoDimsSparseTensorSliceDatasetParams(),
           /*expected_output_dtypes=*/{DT_INT64, DT_INT32, DT_INT64}},
          {/*dataset_params=*/ThreeDimsSparseTensorSliceDatasetParams(),
           /*expected_output_dtypes=*/{DT_INT64, DT_DOUBLE, DT_INT64}},
          {/*dataset_params=*/FourDimsSparseTensorSliceDatasetParams(),
           /*expected_output_dtypes=*/{DT_INT64, DT_STRING, DT_INT64}},
          {/*dataset_params=*/FiveDimsSparseTensorSliceDatasetParams(),
           /*expected_output_dtypes=*/{DT_INT64, DT_INT32, DT_INT64}}};
}

ITERATOR_OUTPUT_DTYPES_TEST_P(SparseTensorSliceDatasetOpTest,
                              SparseTensorSliceDatasetParams,
                              IteratorOutputDtypesTestCases())

std::vector<IteratorOutputShapesTestCase<SparseTensorSliceDatasetParams>>
IteratorOutputShapesTestCases() {
  return {{/*dataset_params=*/TwoDimsSparseTensorSliceDatasetParams(),
           /*expected_output_shapes=*/{PartialTensorShape({1, 1}),
                                       PartialTensorShape({1}),
                                       PartialTensorShape({1})}},
          {/*dataset_params=*/ThreeDimsSparseTensorSliceDatasetParams(),
           /*expected_output_shapes=*/{PartialTensorShape({1, 2}),
                                       PartialTensorShape({1}),
                                       PartialTensorShape({2})}},
          {/*dataset_params=*/FourDimsSparseTensorSliceDatasetParams(),
           /*expected_output_shapes=*/{PartialTensorShape({1, 3}),
                                       PartialTensorShape({1}),
                                       PartialTensorShape({3})}},
          {/*dataset_params=*/FiveDimsSparseTensorSliceDatasetParams(),
           /*expected_output_shapes=*/{PartialTensorShape({1, 4}),
                                       PartialTensorShape({1}),
                                       PartialTensorShape({4})}}};
}

ITERATOR_OUTPUT_SHAPES_TEST_P(SparseTensorSliceDatasetOpTest,
                              SparseTensorSliceDatasetParams,
                              IteratorOutputShapesTestCases())

TEST_F(SparseTensorSliceDatasetOpTest, IteratorPrefix) {
  auto dataset_params = TwoDimsSparseTensorSliceDatasetParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorPrefix(name_utils::IteratorPrefix(
      kDatasetType, dataset_params.iterator_prefix())));
}

template <typename T>
struct IteratorSaveAndRestoreTestCase {
  T dataset_params;
  std::vector<int> breakpoints;
  std::vector<std::vector<Tensor>> expected_outputs;
};

std::vector<IteratorSaveAndRestoreTestCase<SparseTensorSliceDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {{/*dataset_params=*/TwoDimsSparseTensorSliceDatasetParams(),
           /*breakpoints=*/{0, 1, 2},
           /*expected_outputs=*/
           {{/*indices*/ CreateTensor<int64>({1, 1}, {0}),
             /*values*/ CreateTensor<int32>({1}, {888}),
             /*dense_shape*/ CreateTensor<int64>({1}, {2})},
            {/*indices*/ CreateTensor<int64>({1, 1}, {1}),
             /*values*/ CreateTensor<int32>({1}, {999}),
             /*dense_shape*/ CreateTensor<int64>({1}, {2})}}},
          {/*dataset_params=*/ThreeDimsSparseTensorSliceDatasetParams(),
           /*breakpoints=*/{0, 1, 2},
           /*expected_outputs=*/
           {{/*indices*/ CreateTensor<int64>({1, 2}, {0, 0}),
             /*values*/ CreateTensor<double>({1}, {888.0}),
             /*dense_shape*/ CreateTensor<int64>({2}, {2, 2})},
            {{/*indices*/ CreateTensor<int64>({1, 2}, {1, 1})},
             {/*values*/ CreateTensor<double>({1}, {999.0})},
             {/*dense_shape*/ CreateTensor<int64>({2}, {2, 2})}}}},
          {/*dataset_params=*/FourDimsSparseTensorSliceDatasetParams(),
           /*breakpoints=*/{0, 1, 3},
           /*expected_outputs=*/
           {{/*indices*/ CreateTensor<int64>({1, 3}, {0, 0, 0}),
             /*values*/ CreateTensor<tstring>({1}, {"a"}),
             /*dense_shape*/
             CreateTensor<int64>({3}, {2, 2, 2})},
            {/*indices*/ CreateTensor<int64>({1, 3}, {1, 1, 1}),
             /*values*/ CreateTensor<tstring>({1}, {"b"}),
             /*dense_shape*/
             CreateTensor<int64>({3}, {2, 2, 2})},
            {/*indices*/ CreateTensor<int64>({0, 3}, {}),
             /*values*/ CreateTensor<tstring>({0}, {}),
             /*dense_shape*/
             CreateTensor<int64>({3}, {2, 2, 2})}}},
          {/*dataset_params=*/FiveDimsSparseTensorSliceDatasetParams(),
           /*breakpoints=*/{0, 1, 2},
           /*expected_outputs=*/
           {{/*indices*/ CreateTensor<int64>({1, 4}, {0, 0, 0, 0}),
             /*values*/ CreateTensor<int32>({1}, {888}),
             /*dense_shape*/
             CreateTensor<int64>({4}, {2, 2, 2, 2})},
            {/*indices*/ CreateTensor<int64>({1, 4}, {1, 1, 1, 1}),
             /*values*/ CreateTensor<int32>({1}, {999}),
             /*dense_shape*/
             CreateTensor<int64>({4}, {2, 2, 2, 2})},
            {/*indices*/ CreateTensor<int64>({0, 4}, {}),
             /*values*/ CreateTensor<int32>({0}, {}),
             /*dense_shape*/
             CreateTensor<int64>({4}, {2, 2, 2, 2})}}}};
}

class ParameterizedIteratorSaveAndRestoreTest
    : public SparseTensorSliceDatasetOpTest,
      public ::testing::WithParamInterface<
          IteratorSaveAndRestoreTestCase<SparseTensorSliceDatasetParams>> {};

TEST_P(ParameterizedIteratorSaveAndRestoreTest, IteratorSaveAndRestore) {
  auto test_case = GetParam();
  TF_ASSERT_OK(Initialize(test_case.dataset_params));

  std::unique_ptr<SerializationContext> serialization_ctx;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_ctx));

  int cur_iteration = 0;
  bool end_of_sequence = false;
  int64 num_slices = dataset_->Cardinality();
  std::vector<Tensor> out_tensors;

  for (int breakpoint : test_case.breakpoints) {
    while (cur_iteration < breakpoint) {
      TF_EXPECT_OK(iterator_->GetNext(iterator_ctx_.get(), &out_tensors,
                                      &end_of_sequence));
      cur_iteration++;
    }

    if (breakpoint == 0) {
      EXPECT_FALSE(end_of_sequence);
    } else if (breakpoint <= num_slices) {
      for (int i = 0; i < out_tensors.size(); ++i) {
        TF_EXPECT_OK(ExpectEqual(
            out_tensors[0], test_case.expected_outputs[cur_iteration - 1][0]));
        TF_EXPECT_OK(ExpectEqual(
            out_tensors[1], test_case.expected_outputs[cur_iteration - 1][1]));
        TF_EXPECT_OK(ExpectEqual(
            out_tensors[2], test_case.expected_outputs[cur_iteration - 1][2]));
      }
    } else {
      EXPECT_TRUE(end_of_sequence);
    }

    VariantTensorDataWriter writer;
    TF_ASSERT_OK(iterator_->Save(serialization_ctx.get(), &writer));
    std::vector<const VariantTensorData*> data;
    writer.GetData(&data);
    VariantTensorDataReader reader(data);
    TF_EXPECT_OK(RestoreIterator(iterator_ctx_.get(), &reader,
                                 test_case.dataset_params.iterator_prefix(),
                                 *dataset_, &iterator_));
  }
}

INSTANTIATE_TEST_CASE_P(SparseTensorSliceDatasetOpTest,
                        ParameterizedIteratorSaveAndRestoreTest,
                        ::testing::ValuesIn(IteratorSaveAndRestoreTestCases()));

}  // namespace
}  // namespace data
}  // namespace tensorflow
