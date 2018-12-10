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
#include "tensorflow/lite/string_util.h"

namespace tflite {

namespace ops {
namespace custom {
TfLiteRegistration* Register_NORMALIZE();

namespace {

using ::testing::ElementsAreArray;

class NormalizeOpModel : public SingleOpModel {
 public:
  explicit NormalizeOpModel(const string& input) {
    input_ = AddInput(TensorType_STRING);
    output_ = AddOutput(TensorType_STRING);

    SetCustomOp("Normalize", {}, Register_NORMALIZE);
    BuildInterpreter({{static_cast<int>(input.size())}});
    PopulateStringTensor(input_, {input});
  }

  std::vector<string> GetStringOutput() {
    TfLiteTensor* output = interpreter_->tensor(output_);
    int num = GetStringCount(output);
    std::vector<string> result(num);
    for (int i = 0; i < num; i++) {
      auto ref = GetString(output, i);
      result[i] = string(ref.str, ref.len);
    }
    return result;
  }

 private:
  int input_;
  int output_;
};

TEST(NormalizeOpTest, RegularInput) {
  NormalizeOpModel m("I'm good; you're welcome");
  m.Invoke();
  EXPECT_THAT(m.GetStringOutput(),
              ElementsAreArray({"<S> i am good; you are welcome <E>"}));
}

TEST(NormalizeOpTest, OneInput) {
  NormalizeOpModel m("Hi!!!!");
  m.Invoke();
  EXPECT_THAT(m.GetStringOutput(), ElementsAreArray({"<S> hi ! <E>"}));
}

TEST(NormalizeOpTest, EmptyInput) {
  NormalizeOpModel m("");
  m.Invoke();
  EXPECT_THAT(m.GetStringOutput(), ElementsAreArray({"<S>  <E>"}));
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
