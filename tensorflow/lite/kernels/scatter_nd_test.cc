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
#include <stdint.h>

#include <initializer_list>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class ScatterNdOpModel : public SingleOpModel {
 public:
  ScatterNdOpModel(const TensorData& indices, const TensorData& updates,
                   const TensorData& shape) {
    indices_ = AddInput(indices);
    updates_ = AddInput(updates);
    shape_ = AddInput(shape);
    output_ = AddOutput(updates.type);
    SetBuiltinOp(BuiltinOperator_SCATTER_ND, BuiltinOptions_ScatterNdOptions,
                 CreateScatterNdOptions(builder_).Union());
    BuildInterpreter(
        {GetShape(indices_), GetShape(updates_), GetShape(shape_)});
  }

  template <typename T>
  void SetIndices(std::initializer_list<T> data) {
    PopulateTensor<T>(indices_, data);
  }

  template <typename T>
  void SetUpdates(std::initializer_list<T> data) {
    PopulateTensor<T>(updates_, data);
  }

  template <typename T>
  void SetShape(std::initializer_list<T> data) {
    PopulateTensor<T>(shape_, data);
  }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  int indices_;
  int updates_;
  int shape_;
  int output_;
};

TEST(ScatterNdOpTest, ScatterElementIntoVector) {
  ScatterNdOpModel m({TensorType_INT32, {4, 1}}, {TensorType_FLOAT32, {4}},
                     {TensorType_INT32, {1}});
  m.SetIndices<int32_t>({4, 3, 1, 7});
  m.SetUpdates<float>({9, 10, 11, 12});
  m.SetShape<int32_t>({8});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({8}));
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray({0, 11, 0, 10, 9, 0, 0, 12}));
}

TEST(ScatterNdOpTest, ScatterMatrixIntoRank3Tensor) {
  ScatterNdOpModel m({TensorType_INT32, {2, 1}},
                     {TensorType_FLOAT32, {2, 4, 4}}, {TensorType_INT32, {3}});
  m.SetIndices<int32_t>({0, 2});
  m.SetUpdates<float>({5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8,
                       5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8});
  m.SetShape<int32_t>({4, 4, 4});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({4, 4, 4}));
  EXPECT_THAT(
      m.GetOutput<float>(),
      ElementsAreArray({5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
}

TEST(ScatterNdOpTest, ScatterVectorIntoMatrix) {
  ScatterNdOpModel m({TensorType_INT32, {4, 1}}, {TensorType_FLOAT32, {4, 4}},
                     {TensorType_INT32, {2}});
  m.SetIndices<int32_t>({/*0*/ 9, /*1*/ 8, /*2*/ 0, /*3*/ 1});
  m.SetUpdates<float>({/*0*/ 1, 2, 3, 4,
                       /*1*/ 5, 6, 7, 8,
                       /*2*/ 9, 10, 11, 12,
                       /*3*/ 13, 14, 15, 16});
  m.SetShape<int32_t>({10, 4});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({10, 4}));
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray({/*0*/ 9,  10, 11, 12,
                                /*1*/ 13, 14, 15, 16,
                                /*2*/ 0,  0,  0,  0,
                                /*3*/ 0,  0,  0,  0,
                                /*4*/ 0,  0,  0,  0,
                                /*5*/ 0,  0,  0,  0,
                                /*6*/ 0,  0,  0,  0,
                                /*7*/ 0,  0,  0,  0,
                                /*8*/ 5,  6,  7,  8,
                                /*9*/ 1,  2,  3,  4}));
}

TEST(ScatterNdOpTest, ScatterMatricesIntoRank4Tensor) {
  ScatterNdOpModel m({TensorType_INT32, {2, 2, 2}},
                     {TensorType_FLOAT32, {2, 2, 2, 2}},
                     {TensorType_INT32, {4}});
  m.SetIndices<int32_t>(
      {/*0,0*/ 1, 1, /*0,1*/ 0, 1, /*1,0*/ 0, 0, /*1,1*/ 1, 0});
  m.SetUpdates<float>({/*0,0*/ 1, 2, 3, 4, /*0,1*/ 5, 6, 7, 8,
                       /*1,0*/ 9, 10, 11, 12, /*1,1*/ 13, 14, 15, 16});
  m.SetShape<int32_t>({2, 2, 2, 2});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 2, 2}));
  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray({/*0, 0*/ 9, 10, 11, 12,
                                                      /*0, 1*/ 5, 6, 7, 8,
                                                      /*1, 0*/ 13, 14, 15, 16,
                                                      /*1, 1*/ 1, 2, 3, 4}));
}

TEST(ScatterNdOpTest, ScatterVectorIntoRank4Tensor) {
  ScatterNdOpModel m({TensorType_INT32, {2, 2, 3}},
                     {TensorType_FLOAT32, {2, 2, 5}}, {TensorType_INT32, {4}});
  m.SetIndices<int32_t>(
      {/*0,0*/ 2, 2, 2, /*0,1*/ 1, 0, 1, /*1,0*/ 0, 2, 0, /*1,0*/ 2, 2, 0});
  m.SetUpdates<float>(
      {/*0,0*/ 1,  2,  3,  4,  5,  /*0,1*/ 6,  7,  8,  9,  10,
       /*1,0*/ 11, 12, 13, 14, 15, /*1,1*/ 16, 17, 18, 19, 20});
  m.SetShape<int32_t>({3, 3, 3, 5});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 3, 3, 5}));
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray({
                  /*0, 0, 0*/ 0,  0,  0,  0,  0,
                  /*0, 0, 1*/ 0,  0,  0,  0,  0,
                  /*0, 0, 2*/ 0,  0,  0,  0,  0,
                  /*0, 1, 0*/ 0,  0,  0,  0,  0,
                  /*0, 1, 1*/ 0,  0,  0,  0,  0,
                  /*0, 1, 2*/ 0,  0,  0,  0,  0,
                  /*0, 2, 0*/ 11, 12, 13, 14, 15,
                  /*0, 2, 1*/ 0,  0,  0,  0,  0,
                  /*0, 2, 2*/ 0,  0,  0,  0,  0,
                  /*1, 0, 0*/ 0,  0,  0,  0,  0,
                  /*1, 0, 1*/ 6,  7,  8,  9,  10,
                  /*1, 0, 2*/ 0,  0,  0,  0,  0,
                  /*1, 1, 0*/ 0,  0,  0,  0,  0,
                  /*1, 1, 1*/ 0,  0,  0,  0,  0,
                  /*1, 1, 2*/ 0,  0,  0,  0,  0,
                  /*1, 2, 0*/ 0,  0,  0,  0,  0,
                  /*1, 2, 1*/ 0,  0,  0,  0,  0,
                  /*1, 2, 2*/ 0,  0,  0,  0,  0,
                  /*2, 0, 0*/ 0,  0,  0,  0,  0,
                  /*2, 0, 1*/ 0,  0,  0,  0,  0,
                  /*2, 0, 2*/ 0,  0,  0,  0,  0,
                  /*2, 1, 0*/ 0,  0,  0,  0,  0,
                  /*2, 1, 1*/ 0,  0,  0,  0,  0,
                  /*2, 1, 2*/ 0,  0,  0,  0,  0,
                  /*2, 2, 0*/ 16, 17, 18, 19, 20,
                  /*2, 2, 1*/ 0,  0,  0,  0,  0,
                  /*2, 2, 2*/ 1,  2,  3,  4,  5,
              }));
}

TEST(ScatterNdOpTest, ScatterVectorIntoRank3Tensor) {
  ScatterNdOpModel m({TensorType_INT32, {4, 2}}, {TensorType_FLOAT32, {4, 5}},
                     {TensorType_INT32, {3}});
  m.SetIndices<int32_t>({/*0*/ 0, 0, /*1*/ 1, 0, /*2*/ 0, 2, /*3*/ 1, 2});
  m.SetUpdates<float>(
      {/*0*/ 1,  2,  3,  4,  5,  /*1*/ 6,  7,  8,  9,  10,
       /*2*/ 11, 12, 13, 14, 15, /*3*/ 16, 17, 18, 19, 20});
  m.SetShape<int32_t>({2, 3, 5});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 3, 5}));
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray({/*0, 0*/ 1,  2,  3,  4,  5,
                                /*0, 1*/ 0,  0,  0,  0,  0,
                                /*0, 2*/ 11, 12, 13, 14, 15,
                                /*1, 0*/ 6,  7,  8,  9,  10,
                                /*1, 1*/ 0,  0,  0,  0,  0,
                                /*1, 2*/ 16, 17, 18, 19, 20}));
}

TEST(ScatterNdOpTest, OverlappedIndicesSummed) {
  ScatterNdOpModel m({TensorType_INT32, {4, 2}}, {TensorType_FLOAT32, {4, 5}},
                     {TensorType_INT32, {3}});
  m.SetIndices<int32_t>({/*0*/ 1, 0, /*1*/ 0, 2, /*2*/ 0, 2, /*3*/ 1, 0});
  m.SetUpdates<float>(
      {/*0*/ 1,  2,  3,  4,  5,  /*1*/ 6,  7,  8,  9,  10,
       /*2*/ 11, 12, 13, 14, 15, /*3*/ 16, 17, 18, 19, 20});
  m.SetShape<int32_t>({2, 3, 5});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 3, 5}));
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray({/*0, 0*/ 0,  0,  0,  0,  0,
                                /*0, 1*/ 0,  0,  0,  0,  0,
                                /*0, 2*/ 17, 19, 21, 23, 25,
                                /*1, 0*/ 17, 19, 21, 23, 25,
                                /*1, 1*/ 0,  0,  0,  0,  0,
                                /*1, 2*/ 0,  0,  0,  0,  0}));
}

TEST(ScatterNdOpTest, Int32IndicesUint8Updates) {
  ScatterNdOpModel m({TensorType_INT32, {4, 2}}, {TensorType_UINT8, {4, 5}},
                     {TensorType_INT32, {3}});
  m.SetIndices<int32_t>({/*0*/ 0, 0, /*1*/ 1, 0, /*2*/ 0, 2, /*3*/ 1, 2});
  m.SetUpdates<uint8_t>(
      {/*0*/ 1,  2,  3,  4,  5,  /*1*/ 6,  7,  8,  9,  10,
       /*2*/ 11, 12, 13, 14, 15, /*3*/ 16, 17, 18, 19, 20});
  m.SetShape<int32_t>({2, 3, 5});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 3, 5}));
  EXPECT_THAT(m.GetOutput<uint8_t>(),
              ElementsAreArray({/*0, 0*/ 1,  2,  3,  4,  5,
                                /*0, 1*/ 0,  0,  0,  0,  0,
                                /*0, 2*/ 11, 12, 13, 14, 15,
                                /*1, 0*/ 6,  7,  8,  9,  10,
                                /*1, 1*/ 0,  0,  0,  0,  0,
                                /*1, 2*/ 16, 17, 18, 19, 20}));
}

TEST(ScatterNdOpTest, Int32IndicesInt8Updates) {
  ScatterNdOpModel m({TensorType_INT32, {4, 2}}, {TensorType_INT8, {4, 5}},
                     {TensorType_INT32, {3}});
  m.SetIndices<int32_t>({/*0*/ 0, 0, /*1*/ 1, 0, /*2*/ 0, 2, /*3*/ 1, 2});
  m.SetUpdates<int8_t>(
      {/*0*/ 1,  2,  3,  4,  5,  /*1*/ 6,  7,  8,  9,  10,
       /*2*/ 11, 12, 13, 14, 15, /*3*/ 16, 17, 18, 19, 20});
  m.SetShape<int32_t>({2, 3, 5});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 3, 5}));
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({/*0, 0*/ 1,  2,  3,  4,  5,
                                /*0, 1*/ 0,  0,  0,  0,  0,
                                /*0, 2*/ 11, 12, 13, 14, 15,
                                /*1, 0*/ 6,  7,  8,  9,  10,
                                /*1, 1*/ 0,  0,  0,  0,  0,
                                /*1, 2*/ 16, 17, 18, 19, 20}));
}

TEST(ScatterNdOpTest, Int32IndicesInt32Updates) {
  ScatterNdOpModel m({TensorType_INT32, {4, 2}}, {TensorType_INT32, {4, 5}},
                     {TensorType_INT32, {3}});
  m.SetIndices<int32_t>({/*0*/ 0, 0, /*1*/ 1, 0, /*2*/ 0, 2, /*3*/ 1, 2});
  m.SetUpdates<int32_t>(
      {/*0*/ 1,  2,  3,  4,  5,  /*1*/ 6,  7,  8,  9,  10,
       /*2*/ 11, 12, 13, 14, 15, /*3*/ 16, 17, 18, 19, 20});
  m.SetShape<int32_t>({2, 3, 5});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 3, 5}));
  EXPECT_THAT(m.GetOutput<int32_t>(),
              ElementsAreArray({/*0, 0*/ 1,  2,  3,  4,  5,
                                /*0, 1*/ 0,  0,  0,  0,  0,
                                /*0, 2*/ 11, 12, 13, 14, 15,
                                /*1, 0*/ 6,  7,  8,  9,  10,
                                /*1, 1*/ 0,  0,  0,  0,  0,
                                /*1, 2*/ 16, 17, 18, 19, 20}));
}

TEST(ScatterNdOpTest, Int32IndicesInt64Updates) {
  ScatterNdOpModel m({TensorType_INT32, {4, 2}}, {TensorType_INT64, {4, 5}},
                     {TensorType_INT32, {3}});
  m.SetIndices<int32_t>({/*0*/ 0, 0, /*1*/ 1, 0, /*2*/ 0, 2, /*3*/ 1, 2});
  m.SetUpdates<int64_t>(
      {/*0*/ 1,  2,  3,  4,  5,  /*1*/ 6,  7,  8,  9,  10,
       /*2*/ 11, 12, 13, 14, 15, /*3*/ 16, 17, 18, 19, 20});
  m.SetShape<int32_t>({2, 3, 5});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 3, 5}));
  EXPECT_THAT(m.GetOutput<int64_t>(),
              ElementsAreArray({/*0, 0*/ 1,  2,  3,  4,  5,
                                /*0, 1*/ 0,  0,  0,  0,  0,
                                /*0, 2*/ 11, 12, 13, 14, 15,
                                /*1, 0*/ 6,  7,  8,  9,  10,
                                /*1, 1*/ 0,  0,  0,  0,  0,
                                /*1, 2*/ 16, 17, 18, 19, 20}));
}

TEST(ScatterNdOpTest, Int32IndicesBoolUpdates) {
  ScatterNdOpModel m({TensorType_INT32, {4, 1}}, {TensorType_BOOL, {4}},
                     {TensorType_INT32, {1}});
  m.SetIndices<int32_t>({4, 3, 1, 7});
  m.SetUpdates<bool>({true, false, true, false});
  m.SetShape<int32_t>({8});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({8}));
  EXPECT_THAT(
      m.GetOutput<bool>(),
      ElementsAreArray({false, true, false, false, true, false, false, false}));
}

TEST(ScatterNdOpTest, DynamicShape) {
  ScatterNdOpModel m({TensorType_INT32, {4, 2}}, {TensorType_INT64, {4, 5}},
                     {TensorType_INT32, {3}});
  m.SetIndices<int32_t>({/*0*/ 0, 0, /*1*/ 1, 0, /*2*/ 0, 2, /*3*/ 1, 2});
  m.SetUpdates<int64_t>(
      {/*0*/ 1,  2,  3,  4,  5,  /*1*/ 6,  7,  8,  9,  10,
       /*2*/ 11, 12, 13, 14, 15, /*3*/ 16, 17, 18, 19, 20});
  m.SetShape<int32_t>({2, 3, 5});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 3, 5}));
  EXPECT_THAT(m.GetOutput<int64_t>(),
              ElementsAreArray({/*0, 0*/ 1,  2,  3,  4,  5,
                                /*0, 1*/ 0,  0,  0,  0,  0,
                                /*0, 2*/ 11, 12, 13, 14, 15,
                                /*1, 0*/ 6,  7,  8,  9,  10,
                                /*1, 1*/ 0,  0,  0,  0,  0,
                                /*1, 2*/ 16, 17, 18, 19, 20}));

  m.SetIndices<int32_t>({/*0*/ 2, 3, /*1*/ 1, 0, /*2*/ 2, 0, /*3*/ 1, 2});
  m.SetShape<int32_t>({3, 4, 5});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 4, 5}));
  EXPECT_THAT(m.GetOutput<int64_t>(),
              ElementsAreArray({/*0, 0*/ 0,  0,  0,  0,  0,
                                /*0, 1*/ 0,  0,  0,  0,  0,
                                /*0, 2*/ 0,  0,  0,  0,  0,
                                /*0, 3*/ 0,  0,  0,  0,  0,
                                /*1, 0*/ 6,  7,  8,  9,  10,
                                /*1, 1*/ 0,  0,  0,  0,  0,
                                /*1, 2*/ 16, 17, 18, 19, 20,
                                /*1, 3*/ 0,  0,  0,  0,  0,
                                /*2, 0*/ 11, 12, 13, 14, 15,
                                /*2, 1*/ 0,  0,  0,  0,  0,
                                /*2, 2*/ 0,  0,  0,  0,  0,
                                /*2, 3*/ 1,  2,  3,  4,  5}));
}

}  // namespace
}  // namespace tflite
