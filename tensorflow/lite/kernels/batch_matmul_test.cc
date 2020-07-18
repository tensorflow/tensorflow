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
#include <stddef.h>
#include <stdint.h>

#include <initializer_list>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {

namespace ops {
namespace builtin {

TfLiteRegistration* Register_BATCH_MATMUL_REF();
TfLiteRegistration* Register_BATCH_MATMUL_GENERIC_OPTIMIZED();

}  // namespace builtin
}  // namespace ops

namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

template <typename T>
class BatchMatMulOpModel : public SingleOpModel {
 public:
  BatchMatMulOpModel(const TensorData& lhs, const TensorData& rhs,
                     bool adj_x = false, bool adj_y = false) {
    lhs_id_ = AddInput(lhs);
    rhs_id_ = AddInput(rhs);
    output_id_ = AddOutput(lhs.type);
    SetBuiltinOp(BuiltinOperator_BATCH_MATMUL,
                 BuiltinOptions_BatchMatMulOptions,
                 CreateBatchMatMulOptions(builder_, adj_x, adj_y).Union());
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

const auto kKernelMap = new std::map<string, TfLiteRegistration*>({
    {"Reference", ops::builtin::Register_BATCH_MATMUL_REF()},
    {"GenericOptimized",
     ops::builtin::Register_BATCH_MATMUL_GENERIC_OPTIMIZED()},
});

class BatchMatMulOpTest : public SingleOpTest {
 protected:
  const std::map<string, TfLiteRegistration*>& GetKernelMap() override {
    return *kKernelMap;
  }
};

TEST_P(BatchMatMulOpTest, Float32Test_Simple) {
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

TEST_P(BatchMatMulOpTest, Float32Test_SimpleRHSAdjoint) {
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

TEST_P(BatchMatMulOpTest, Float32Test_SimpleLHSAdjoint) {
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

TEST_P(BatchMatMulOpTest, Float32Test_BatchSizeTwo) {
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

TEST_P(BatchMatMulOpTest, Float32Test_Broadcast) {
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

TEST_P(BatchMatMulOpTest, Float32Test_BroadcastLHSAdjoint) {
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

TEST_P(BatchMatMulOpTest, Float32Test_Broadcast2) {
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

TEST_P(BatchMatMulOpTest, Float32Test_Broadcast2LHSAdjoint) {
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

TEST_P(BatchMatMulOpTest, Float32Test_Broadcast2RHSAdjoint) {
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

TEST_P(BatchMatMulOpTest, Float32Test_Broadcast2BothAdjoint) {
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

TEST_P(BatchMatMulOpTest, Float32Test_BroadcastFromRHS) {
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

INSTANTIATE_TEST_SUITE_P(
    BatchMatMulOpTest, BatchMatMulOpTest,
    ::testing::ValuesIn(SingleOpTest::GetKernelTags(*kKernelMap)));

// In the hybrid model the weights are quantized int8. But the input
// and output are expected to be in float precision.
class HybridAsymmetricBatchMatMulOpModel : public SingleOpModel {
 public:
  HybridAsymmetricBatchMatMulOpModel(
      int units, int batches, const TensorData& lhs, const TensorData& rhs,
      const TensorData& output = {TensorType_FLOAT32}, bool adj_x = false,
      bool adj_y = false)
      : units_(units), batches_(batches) {
    int total_input_size = 1;
    for (size_t i = 0; i < lhs.shape.size(); ++i) {
      total_input_size *= lhs.shape[i];
    }
    input_size_ = total_input_size / batches_;

    lhs_id_ = AddInput(lhs);
    rhs_id_ = AddInput(rhs);

    output_id_ = AddOutput(output);

    SetBuiltinOp(BuiltinOperator_BATCH_MATMUL,
                 BuiltinOptions_BatchMatMulOptions,
                 CreateBatchMatMulOptions(builder_, adj_x, adj_y).Union());
    BuildInterpreter({GetShape(lhs_id_), GetShape(rhs_id_)});
  }
  void SetWeights(const std::vector<float>& data) {
    SymmetricQuantizeAndPopulate(rhs_id_, data);
  }

  void SetSignedWeights(std::initializer_list<float> f) {
    SignedSymmetricQuantizeAndPopulate(rhs_id_, f);
  }

  void SetInput(const std::vector<float>& f) { PopulateTensor(lhs_id_, f); }
  std::vector<float> GetOutput() { return ExtractVector<float>(output_id_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_id_); }

  int input_size() { return input_size_; }
  int num_units() { return units_; }
  int num_batches() { return batches_; }

  int lhs() const { return lhs_id_; }
  int rhs() const { return rhs_id_; }

 protected:
  int lhs_id_;
  int rhs_id_;
  int output_id_;
  int units_;
  int batches_;
  int input_size_;
};

class HybridAsymmetricBatchMatMulOpTest : public SingleOpTest {
 protected:
  const std::map<string, TfLiteRegistration*>& GetKernelMap() override {
    return *kKernelMap;
  }
};

TEST_P(HybridAsymmetricBatchMatMulOpTest, SimpleTestQuantizedInt8) {
  HybridAsymmetricBatchMatMulOpModel m(
      /*units=*/3, /*batches=*/2,
      /*lhs=*/{TensorType_FLOAT32, {2, 10}},
      /*rhs=*/{TensorType_INT8, {10, 3}, 0, 0, 10.0 / 127.0, 0});

  m.SetSignedWeights({
      1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5,  5,  5,
      6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10,
  });

  m.SetInput({
      11, 12, 13, 14, 15, 16, 17, 18,  -19, -20,  // batch 1, 0
      11, 12, 13, 14, 15, 16, 17, -18, 19,  -20,  // batch 1, 1
  });

  m.Invoke();

  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(
                                 {
                                     196,
                                     196,
                                     196,
                                     246,
                                     246,
                                     246,
                                 },
                                 /*max_abs_error=*/0.64f)));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 3}));
}

TEST_P(HybridAsymmetricBatchMatMulOpTest, QuantizedInt8BroadcastWeights) {
  HybridAsymmetricBatchMatMulOpModel m(
      /*units=*/3, /*batches=*/2,
      /*lhs=*/{TensorType_FLOAT32, {2, 2, 10}},
      /*rhs=*/{TensorType_INT8, {10, 3}, 0, 0, 10.0 / 127.0, 0});

  m.SetSignedWeights({
      1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5,  5,  5,
      6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10,
  });

  m.SetInput({
      1,  2,  3,  4,  5,  6,  7,  8,   -9,  -10,  // batch 0, 0
      1,  2,  3,  4,  5,  6,  7,  -8,  9,   -10,  // batch 0, 1
      11, 12, 13, 14, 15, 16, 17, 18,  -19, -20,  // batch 1, 0
      11, 12, 13, 14, 15, 16, 17, -18, 19,  -20,  // batch 1, 1
  });

  m.Invoke();

  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(
                                 {
                                     24, 24, 24,     //
                                     58, 58, 58,     //
                                     196, 196, 196,  //
                                     246, 246, 246,  //
                                 },
                                 /*max_abs_error=*/1.3f)));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 3}));
}

TEST_P(HybridAsymmetricBatchMatMulOpTest, QuantizedInt8BroadcastBigWeights) {
  HybridAsymmetricBatchMatMulOpModel m(
      /*units=*/9, /*batches=*/2,
      /*lhs=*/{TensorType_FLOAT32, {2, 2, 10}},
      /*rhs=*/{TensorType_INT8, {10, 9}, 0, 0, 10.0 / 127.0, 0});

  m.SetSignedWeights({
      1, 1, 1, 17, 17, 17, 26, 26, 26, 2,  2,  2,  18, 18, 18, 27, 27, 27,
      3, 3, 3, 19, 19, 19, 28, 28, 28, 4,  4,  4,  20, 20, 20, 29, 29, 29,
      5, 5, 5, 21, 21, 21, 30, 30, 30, 6,  6,  6,  22, 22, 22, 31, 31, 31,
      7, 7, 7, 23, 23, 23, 32, 32, 32, 8,  8,  8,  24, 24, 24, 33, 33, 33,
      9, 9, 9, 25, 25, 25, 34, 34, 34, 10, 10, 10, 26, 26, 26, 35, 35, 35,
  });

  m.SetInput({
      1,  2,  3,  4,  5,  6,  7,  8,   -9,  -10,  // batch 0, 0
      1,  2,  3,  4,  5,  6,  7,  -8,  9,   -10,  // batch 0, 1
      11, 12, 13, 14, 15, 16, 17, 18,  -19, -20,  // batch 1, 0
      11, 12, 13, 14, 15, 16, 17, -18, 19,  -20,  // batch 1, 1
  });

  m.Invoke();

  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      23,  23,  23,  295,  295,  295,  449,  449,  449,   //
                      60,  60,  60,  364,  364,  364,  533,  533,  533,   //
                      195, 195, 195, 1429, 1429, 1429, 2124, 2124, 2124,  //
                      250, 250, 250, 1512, 1512, 1512, 2213, 2213, 2213   //
                  },
                  /*max_abs_error=*/1.3f)));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 9}));
}

TEST_P(HybridAsymmetricBatchMatMulOpTest, QuantizedInt8BroadcastInputs) {
  HybridAsymmetricBatchMatMulOpModel m(
      /*units=*/3, /*batches=*/2,
      /*lhs=*/{TensorType_FLOAT32, {2, 10}},
      /*rhs=*/{TensorType_INT8, {2, 10, 3}, 0, 0, 10.0 / 127.0, 0});

  m.SetSignedWeights({
      1, -3, 1, 2, -2, 2, 3, -1, 3, 4,  0, 4, 5, 1, 5, 6, 2, 6,  7,  3,
      7, 8,  4, 8, 9,  5, 9, 10, 6, 10, 1, 1, 1, 2, 2, 2, 3, 3,  3,  4,
      4, 4,  5, 5, 5,  6, 6, 6,  7, 7,  7, 8, 8, 8, 9, 9, 9, 10, 10, 10,
  });

  m.SetInput({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // batch 0, 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // batch 0, 1
  });

  m.Invoke();

  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(
                                 {
                                     24, -45, 24,  //
                                     58, -18, 58,  //
                                     24, 24, 24,   //
                                     58, 58, 58,   //
                                 },
                                 /*max_abs_error=*/0.64f)));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 3}));
}

INSTANTIATE_TEST_SUITE_P(
    HybridAsymmetricBatchMatMulOpTest, HybridAsymmetricBatchMatMulOpTest,
    ::testing::ValuesIn(SingleOpTest::GetKernelTags(*kKernelMap)));

class QuantizedBatchMatMulOpModel : public SingleOpModel {
 public:
  QuantizedBatchMatMulOpModel(int units, int batches, const TensorData& lhs,
                              const TensorData& output = {TensorType_INT8},
                              bool adj_x = false, bool adj_y = false)
      : units_(units), batches_(batches) {
    int total_input_size = 1;
    for (size_t i = 0; i < lhs.shape.size(); ++i) {
      total_input_size *= lhs.shape[i];
    }
    input_size_ = total_input_size / batches_;

    lhs_id_ = AddInput(lhs);
    rhs_id_ = AddInput({lhs.type, {input_size_, units_}, lhs.min, lhs.max});

    output_id_ = AddOutput(output);

    SetBuiltinOp(BuiltinOperator_BATCH_MATMUL,
                 BuiltinOptions_BatchMatMulOptions,
                 CreateBatchMatMulOptions(builder_, adj_x, adj_y).Union());
    BuildInterpreter({GetShape(lhs_id_), GetShape(rhs_id_)});
  }

  template <typename T>
  void SetWeights(const std::vector<float>& data) {
    QuantizeAndPopulate<T>(rhs_id_, data);
  }

  template <typename T>
  void SetInput(const std::vector<float>& data) {
    QuantizeAndPopulate<T>(lhs_id_, data);
  }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_id_);
  }

  template <typename T>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<T>(ExtractVector<T>(output_id_), GetScale(output_id_),
                         GetZeroPoint(output_id_));
  }

 protected:
  int lhs_id_;
  int rhs_id_;
  int output_id_;
  int units_;
  int batches_;
  int input_size_;
};

class QuantizedBatchMatMulOpTest : public SingleOpTest {
 protected:
  const std::map<string, TfLiteRegistration*>& GetKernelMap() override {
    return *kKernelMap;
  }
};

TEST_P(QuantizedBatchMatMulOpTest, SimpleTestQuantizedInt8) {
  QuantizedBatchMatMulOpModel m(
      /*units=*/3, /*batches*/ 2,
      /*lhs=*/{TensorType_INT8, {2, 10}, -63.5, 64},
      /*output=*/{TensorType_INT8, {}, -127, 128});

  m.SetWeights<int8_t>({
      1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5,  5,  5,
      6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10,
  });

  m.SetInput<int8_t>({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
  });

  m.Invoke();

  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear({23, 23, 23, 57, 57, 57})));
  EXPECT_THAT(m.GetOutput<int8_t>(), ElementsAre(22, 22, 22, 56, 56, 56));
}

INSTANTIATE_TEST_SUITE_P(
    QuantizedBatchMatMulOpTest, QuantizedBatchMatMulOpTest,
    ::testing::ValuesIn(SingleOpTest::GetKernelTags(*kKernelMap)));

}  // namespace
}  // namespace tflite
