/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
// Unit test for TFLite sparse output fully connected op.
#include <iomanip>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/flexbuffers.h"  // TF:flatbuffers
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"

namespace tflite {

namespace ops {
namespace custom {

TfLiteRegistration* Register_SPARSE_OUTPUT_FULLY_CONNECTED();

namespace {

using ::testing::ElementsAreArray;

class BaseSparseOutputFullyConnectedOpModel : public SingleOpModel {
 public:
  BaseSparseOutputFullyConnectedOpModel(const TensorData& input,
                                        const TensorData& weights,
                                        const TensorData& output = {
                                            TensorType_FLOAT32}) {
    input_ = AddInput(input);
    lookup_ = AddInput({TensorType_INT32, {1}});
    weights_ = AddInput(weights);
    int bias_size = GetShape(weights_)[0];
    bias_ = AddInput({TensorType_FLOAT32, {bias_size}});
    output_ = AddOutput(output);

    // Create empty (required) options map.
    flexbuffers::Builder fbb;
    fbb.Map([&]() {});
    fbb.Finish();

    SetCustomOp("SPARSE_OUTPUT_FULLY_CONNECTED", fbb.GetBuffer(),
                Register_SPARSE_OUTPUT_FULLY_CONNECTED);
    BuildInterpreter({GetShape(input_), GetShape(lookup_), GetShape(weights_),
                      GetShape(bias_)});
  }

  void SetInput(const std::vector<float>& data) {
    PopulateTensor(input_, data);
  }

  void SetLookup(const std::vector<int32_t>& f) { PopulateTensor(lookup_, f); }

  void SetBias(const std::vector<float>& f) { PopulateTensor(bias_, f); }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

 protected:
  int input_;
  int lookup_;
  int weights_;
  int bias_;
  int output_;
};

class FloatSparseOutputFullyConnectedOpModel
    : public BaseSparseOutputFullyConnectedOpModel {
 public:
  using BaseSparseOutputFullyConnectedOpModel::
      BaseSparseOutputFullyConnectedOpModel;

  void SetWeights(const std::vector<float>& f) { PopulateTensor(weights_, f); }
};

class HybridSparseOutputFullyConnectedOpModel
    : public BaseSparseOutputFullyConnectedOpModel {
 public:
  using BaseSparseOutputFullyConnectedOpModel::
      BaseSparseOutputFullyConnectedOpModel;

  void SetWeights(const std::vector<float>& f) {
    SymmetricQuantizeAndPopulate(weights_, f);
  }

  void SetSignedWeights(const std::vector<float>& f) {
    SignedSymmetricQuantizeAndPopulate(weights_, f);
  }
};

TEST(SparseOutputFullyConnectedOpTest, SimpleTestFloat) {
  FloatSparseOutputFullyConnectedOpModel m({TensorType_FLOAT32, {1, 5}},
                                           {TensorType_FLOAT32, {3, 5}},
                                           {TensorType_FLOAT32, {}});

  m.SetInput({-1.0, 0.0, 1.0, 2.0, 3.0});

  m.SetLookup({2});

  m.SetWeights({
      -1.0, 0.0, 1.0, 2.0, 3.0,  //
      0.0, 1.0, 2.0, 3.0, 4.0,   //
      1.0, 2.0, 3.0, 4.0, 5.0,   //
  });

  m.SetBias({1.0, 2.0, 3.0});

  m.Invoke();

  EXPECT_THAT(m.GetOutput(), ElementsAreArray({28}));
}

TEST(SparseOutputFullyConnectedOpTest, SimpleTestHybridUint8) {
  HybridSparseOutputFullyConnectedOpModel m({TensorType_FLOAT32, {1, 5}},
                                            {TensorType_UINT8, {3, 5}},
                                            {TensorType_FLOAT32, {}});

  m.SetInput({-1.0, 0.0, 1.0, 2.0, 3.0});

  m.SetLookup({2});

  m.SetWeights({
      -1.0, 0.0, 1.0, 2.0, 3.0,  //
      0.0, 1.0, 2.0, 3.0, 4.0,   //
      1.0, 2.0, 3.0, 4.0, 5.0,   //
  });

  m.SetBias({1.0, 2.0, 3.0});

  m.Invoke();

  // We get 28.0552 instead of 28.
  //
  // Input -> -42, 0, 42, 85, 127 with scale factor of 127/3.
  // Looked up weights ->  25, 51, 76, 102, 127 with scale factor of 127/5.
  //
  // (-42 * 25 + 0 * 51 + 42 * 76 + 85 * 102 + 127 * 127) * (3*5/127^2) + 3.0
  // gives us the expected result.
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({28}, 0.0553)));
}

TEST(SparseOutputFullyConnectedOpTest, SimpleTestHybridInt8) {
  HybridSparseOutputFullyConnectedOpModel m({TensorType_FLOAT32, {1, 5}},
                                            {TensorType_INT8, {3, 5}},
                                            {TensorType_FLOAT32, {}});

  m.SetInput({-1.0, 0.0, 1.0, 2.0, 3.0});

  m.SetLookup({2});

  m.SetSignedWeights({
      -1.0, 0.0, 1.0, 2.0, 3.0,  //
      0.0, 1.0, 2.0, 3.0, 4.0,   //
      1.0, 2.0, 3.0, 4.0, 5.0,   //
  });

  m.SetBias({1.0, 2.0, 3.0});

  m.Invoke();

  // We get 28.0552 instead of 28.
  //
  // Input -> -42, 0, 42, 85, 127 with scale factor of 127/3.
  // Looked up weights ->  25, 51, 76, 102, 127 with scale factor of 127/5.
  //
  // (-42 * 25 + 0 * 51 + 42 * 76 + 85 * 102 + 127 * 127) * (3*5/127^2) + 3.0
  // gives us the expected result.
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({28}, 0.0553)));
}

}  // namespace
}  // namespace custom
}  // namespace ops
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
