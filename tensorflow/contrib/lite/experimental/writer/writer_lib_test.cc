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

#include "tensorflow/contrib/lite/experimental/writer/writer_lib.h"
#include <gtest/gtest.h>
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/testing/util.h"

namespace tflite {
// Make an interpreter that has no tensors and no nodes
// TODO(b/113731921): add more tests.
TEST(Writer, BasicTest) {
  Interpreter interpreter;
  interpreter.AddTensors(3);
  float foo[] = {1, 2, 3};
  interpreter.SetTensorParametersReadWrite(0, kTfLiteFloat32, "a", {3},
                                           TfLiteQuantizationParams());
  interpreter.SetTensorParametersReadOnly(
      1, kTfLiteFloat32, "b", {3}, TfLiteQuantizationParams(),
      reinterpret_cast<char*>(foo), sizeof(foo));
  interpreter.SetTensorParametersReadWrite(2, kTfLiteFloat32, "c", {3},
                                           TfLiteQuantizationParams());
  interpreter.SetInputs({0, 1});
  interpreter.SetOutputs({2});
  const char* initial_data = "";
  tflite::ops::builtin::BuiltinOpResolver resolver;
  TfLiteAddParams* builtin_data =
      reinterpret_cast<TfLiteAddParams*>(malloc(sizeof(TfLiteAddParams)));
  builtin_data->activation = kTfLiteActNone;
  const TfLiteRegistration* reg = resolver.FindOp(BuiltinOperator_ADD, 1);
  interpreter.AddNodeWithParameters({0, 1}, {2}, initial_data, 0,
                                    reinterpret_cast<void*>(builtin_data), reg);

  InterpreterWriter writer(&interpreter);
  writer.Write("/tmp/test.tflite");
  std::unique_ptr<FlatBufferModel> model =
      FlatBufferModel::BuildFromFile("/tmp/test.tflite");
  InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<Interpreter> new_interpreter;
  builder(&new_interpreter);
}

}  // namespace tflite

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
