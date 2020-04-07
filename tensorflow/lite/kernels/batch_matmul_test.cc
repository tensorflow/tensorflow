/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

template <typename T>
class BatchMatMulOpModel : public SingleOpModel {
 public:
  BatchMatMulOpModel(const TensorData& lhs, const TensorData& rhs,
                     bool adjoint_lhs = false, bool adjoint_rhs = false) {
    lhs_id_ = AddInput(lhs);
    rhs_id_ = AddInput(rhs);
    output_id_ = AddOutput(lhs.type);
    SetBuiltinOp(
        BuiltinOperator_BATCH_MATMUL, BuiltinOptions_BatchMatMulOptions,
        CreateBatchMatMulOptions(builder_, adjoint_lhs, adjoint_rhs).Union());
    BuildInterpreter({GetShape(lhs_id_), GetShape(rhs_id_)});
  }

  int lhs() const { return lhs_id_; }
  int rhs() const { return rhs_id_; }
  std::vector<T> GetOutput() { return ExtractVector<T>(output_id_); }
  std::vector<int32_t> GetOutputShape() { return GetTensorShape(output_id_); }

 protected:
  int lhs_id_;
  int rhs_id_;
  int output_id_;
};

TEST(BatchMatMulOpModelTest, Float32Test_Simple) {
  BatchMatMulOpModel<float> model({TensorType_FLOAT32, {1, 2, 3}},
                                  {TensorType_FLOAT32, {1, 3, 4}});
  model.PopulateTensor<float>(model.lhs(), {1, 2, 3, 4, 5, 6});
  model.PopulateTensor<float>(model.rhs(),
                              {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18});
  model.Invoke();
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({74., 80., 86., 92., 173., 188., 203., 218.}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 2, 4}));
}

TEST(BatchMatMulOpModelTest, Float32Test_SimpleRHSAdjoint) {
  BatchMatMulOpModel<float> model({TensorType_FLOAT32, {1, 2, 3}},
                                  {TensorType_FLOAT32, {1, 4, 3}}, false, true);
  model.PopulateTensor<float>(model.lhs(), {1, 2, 3, 4, 5, 6});
  model.PopulateTensor<float>(model.rhs(),
                              {7, 11, 15, 8, 12, 16, 9, 13, 17, 10, 14, 18});
  model.Invoke();
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({74., 80., 86., 92., 173., 188., 203., 218.}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 2, 4}));
}

TEST(BatchMatMulOpModelTest, Float32Test_SimpleLHSAdjoint) {
  BatchMatMulOpModel<float> model({TensorType_FLOAT32, {1, 3, 2}},
                                  {TensorType_FLOAT32, {1, 3, 4}}, true, false);
  model.PopulateTensor<float>(model.lhs(), {1, 4, 2, 5, 3, 6});
  model.PopulateTensor<float>(model.rhs(),
                              {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18});
  model.Invoke();
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({74., 80., 86., 92., 173., 188., 203., 218.}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 2, 4}));
}

TEST(BatchMatMulOpModelTest, Float32Test_BatchSizeTwo) {
  BatchMatMulOpModel<float> model({TensorType_FLOAT32, {2, 2, 3}},
                                  {TensorType_FLOAT32, {2, 3, 4}});
  model.PopulateTensor<float>(model.lhs(),
                              {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  model.PopulateTensor<float>(model.rhs(),
                              {7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18,
                               19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30});
  model.Invoke();
  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray({74., 80., 86., 92., 173., 188., 203., 218., 560., 584.,
                        608., 632., 767., 800., 833., 866.}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 2, 4}));
}

TEST(BatchMatMulOpModelTest, Float32Test_Broadcast) {
  BatchMatMulOpModel<float> model({TensorType_FLOAT32, {2, 2, 3}},
                                  {TensorType_FLOAT32, {3, 4}});
  model.PopulateTensor<float>(model.lhs(),
                              {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  model.PopulateTensor<float>(model.rhs(),
                              {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18});

  model.Invoke();
  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray({74., 80., 86., 92., 173., 188., 203., 218., 272., 296.,
                        320., 344., 371., 404., 437., 470.}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 2, 4}));
}

TEST(BatchMatMulOpModelTest, Float32Test_BroadcastLHSAdjoint) {
  BatchMatMulOpModel<float> model({TensorType_FLOAT32, {2, 3, 2}},
                                  {TensorType_FLOAT32, {3, 4}}, true, false);
  model.PopulateTensor<float>(model.lhs(),
                              {1, 4, 2, 5, 3, 6, 7, 10, 8, 11, 9, 12});
  model.PopulateTensor<float>(model.rhs(),
                              {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18});

  model.Invoke();
  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray({74., 80., 86., 92., 173., 188., 203., 218., 272., 296.,
                        320., 344., 371., 404., 437., 470.}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 2, 4}));
}

TEST(BatchMatMulOpModelTest, Float32Test_Broadcast2) {
  BatchMatMulOpModel<float> model({TensorType_FLOAT32, {2, 1, 3, 2}},
                                  {TensorType_FLOAT32, {3, 2, 4}});
  model.PopulateTensor<float>(model.lhs(),
                              {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  model.PopulateTensor<float>(model.rhs(),
                              {7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18,
                               19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30});

  model.Invoke();
  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray({29.,  32.,  35.,  38.,  65.,  72.,  79.,  86.,  101.,
                        112., 123., 134., 53.,  56.,  59.,  62.,  121., 128.,
                        135., 142., 189., 200., 211., 222., 77.,  80.,  83.,
                        86.,  177., 184., 191., 198., 277., 288., 299., 310.,
                        137., 152., 167., 182., 173., 192., 211., 230., 209.,
                        232., 255., 278., 257., 272., 287., 302., 325., 344.,
                        363., 382., 393., 416., 439., 462., 377., 392., 407.,
                        422., 477., 496., 515., 534., 577., 600., 623., 646.}));

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 3, 3, 4}));
}

TEST(BatchMatMulOpModelTest, Float32Test_Broadcast2LHSAdjoint) {
  BatchMatMulOpModel<float> model({TensorType_FLOAT32, {2, 1, 2, 3}},
                                  {TensorType_FLOAT32, {3, 2, 4}}, true, false);
  model.PopulateTensor<float>(model.lhs(),
                              {1, 3, 5, 2, 4, 6, 7, 9, 11, 8, 10, 12});
  model.PopulateTensor<float>(model.rhs(),
                              {7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18,
                               19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30});

  model.Invoke();
  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray({29.,  32.,  35.,  38.,  65.,  72.,  79.,  86.,  101.,
                        112., 123., 134., 53.,  56.,  59.,  62.,  121., 128.,
                        135., 142., 189., 200., 211., 222., 77.,  80.,  83.,
                        86.,  177., 184., 191., 198., 277., 288., 299., 310.,
                        137., 152., 167., 182., 173., 192., 211., 230., 209.,
                        232., 255., 278., 257., 272., 287., 302., 325., 344.,
                        363., 382., 393., 416., 439., 462., 377., 392., 407.,
                        422., 477., 496., 515., 534., 577., 600., 623., 646.}));

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 3, 3, 4}));
}

TEST(BatchMatMulOpModelTest, Float32Test_Broadcast2RHSAdjoint) {
  BatchMatMulOpModel<float> model({TensorType_FLOAT32, {2, 1, 3, 2}},
                                  {TensorType_FLOAT32, {3, 4, 2}}, false, true);
  model.PopulateTensor<float>(model.lhs(),
                              {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  model.PopulateTensor<float>(model.rhs(),
                              {7,  11, 8,  12, 9,  13, 10, 14, 15, 19, 16, 20,
                               17, 21, 18, 22, 23, 27, 24, 28, 25, 29, 26, 30});
  model.Invoke();
  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray({29.,  32.,  35.,  38.,  65.,  72.,  79.,  86.,  101.,
                        112., 123., 134., 53.,  56.,  59.,  62.,  121., 128.,
                        135., 142., 189., 200., 211., 222., 77.,  80.,  83.,
                        86.,  177., 184., 191., 198., 277., 288., 299., 310.,
                        137., 152., 167., 182., 173., 192., 211., 230., 209.,
                        232., 255., 278., 257., 272., 287., 302., 325., 344.,
                        363., 382., 393., 416., 439., 462., 377., 392., 407.,
                        422., 477., 496., 515., 534., 577., 600., 623., 646.}));

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 3, 3, 4}));
}

TEST(BatchMatMulOpModelTest, Float32Test_Broadcast2BothAdjoint) {
  BatchMatMulOpModel<float> model({TensorType_FLOAT32, {2, 1, 2, 3}},
                                  {TensorType_FLOAT32, {3, 4, 2}}, true, true);
  model.PopulateTensor<float>(model.lhs(),
                              {1, 3, 5, 2, 4, 6, 7, 9, 11, 8, 10, 12});
  model.PopulateTensor<float>(model.rhs(),
                              {7,  11, 8,  12, 9,  13, 10, 14, 15, 19, 16, 20,
                               17, 21, 18, 22, 23, 27, 24, 28, 25, 29, 26, 30});
  model.Invoke();
  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray({29.,  32.,  35.,  38.,  65.,  72.,  79.,  86.,  101.,
                        112., 123., 134., 53.,  56.,  59.,  62.,  121., 128.,
                        135., 142., 189., 200., 211., 222., 77.,  80.,  83.,
                        86.,  177., 184., 191., 198., 277., 288., 299., 310.,
                        137., 152., 167., 182., 173., 192., 211., 230., 209.,
                        232., 255., 278., 257., 272., 287., 302., 325., 344.,
                        363., 382., 393., 416., 439., 462., 377., 392., 407.,
                        422., 477., 496., 515., 534., 577., 600., 623., 646.}));

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 3, 3, 4}));
}

TEST(BatchMatMulOpModelTest, Float32Test_BroadcastFromRHS) {
  BatchMatMulOpModel<float> model({TensorType_FLOAT32, {4, 5}},
                                  {TensorType_FLOAT32, {3, 1, 5, 2}});
  model.PopulateTensor<float>(
      model.lhs(),
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
  model.PopulateTensor<float>(
      model.rhs(),
      {7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
       22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36});

  model.Invoke();
  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray({185., 200., 460.,  500.,  735.,  800.,  1010., 1100.,
                        335., 350., 860.,  900.,  1385., 1450., 1910., 2000.,
                        485., 500., 1260., 1300., 2035., 2100., 2810., 2900.}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({3, 1, 4, 2}));
}

}  // namespace
}  // namespace tflite
