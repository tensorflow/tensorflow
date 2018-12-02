/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"
#include <farmhash.h>

namespace tflite {

namespace ops {
namespace custom {
TfLiteRegistration* Register_EXTRACT_FEATURES();

namespace {

using ::testing::ElementsAre;

class ExtractFeatureOpModel : public SingleOpModel {
 public:
  explicit ExtractFeatureOpModel(const std::vector<string>& input) {
    input_ = AddInput(TensorType_STRING);
    signature_ = AddOutput(TensorType_INT32);
    weight_ = AddOutput(TensorType_FLOAT32);

    SetCustomOp("ExtractFeatures", {}, Register_EXTRACT_FEATURES);
    BuildInterpreter({{static_cast<int>(input.size())}});
    PopulateStringTensor(input_, input);
  }

  std::vector<int> GetSignature() { return ExtractVector<int>(signature_); }
  std::vector<float> GetWeight() { return ExtractVector<float>(weight_); }

 private:
  int input_;
  int signature_;
  int weight_;
};

int CalcFeature(const string& str) {
  return ::util::Fingerprint64(str) % 1000000;
}

TEST(ExtractFeatureOpTest, RegularInput) {
  ExtractFeatureOpModel m({"<S>", "<S> Hi", "Hi", "Hi !", "!", "! <E>", "<E>"});
  m.Invoke();
  EXPECT_THAT(m.GetSignature(),
              ElementsAre(0, CalcFeature("<S> Hi"), CalcFeature("Hi"),
                          CalcFeature("Hi !"), CalcFeature("!"),
                          CalcFeature("! <E>"), 0));
  EXPECT_THAT(m.GetWeight(), ElementsAre(0, 2, 1, 2, 1, 2, 0));
}

TEST(ExtractFeatureOpTest, OneInput) {
  ExtractFeatureOpModel m({"Hi"});
  m.Invoke();
  EXPECT_THAT(m.GetSignature(), ElementsAre(CalcFeature("Hi")));
  EXPECT_THAT(m.GetWeight(), ElementsAre(1));
}

TEST(ExtractFeatureOpTest, ZeroInput) {
  ExtractFeatureOpModel m({});
  m.Invoke();
  EXPECT_THAT(m.GetSignature(), ElementsAre(0));
  EXPECT_THAT(m.GetWeight(), ElementsAre(0));
}

TEST(ExtractFeatureOpTest, AllBlacklistInput) {
  ExtractFeatureOpModel m({"<S>", "<E>"});
  m.Invoke();
  EXPECT_THAT(m.GetSignature(), ElementsAre(0, 0));
  EXPECT_THAT(m.GetWeight(), ElementsAre(0, 0));
}

}  // namespace
}  // namespace custom
}  // namespace ops
}  // namespace tflite

int main(int argc, char** argv) {
  // On Linux, add: tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
