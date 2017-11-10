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
// Unit test for speech SpeakerId model using TFLite Ops.

#include <string.h>

#include <memory>
#include <string>

#include "base/logging.h"
#include "file/base/path.h"
#include "testing/base/public/googletest.h"
#include <gtest/gtest.h>
#include "tensorflow/contrib/lite/context.h"
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/models/test_utils.h"
#include "tensorflow/contrib/lite/tools/mutable_op_resolver.h"

void RegisterSelectedOps(::tflite::MutableOpResolver* resolver);

namespace tflite {
namespace models {

constexpr int kModelInputTensor = 0;
constexpr int kLstmLayer1OutputStateTensor = 19;
constexpr int kLstmLayer1CellStateTensor = 20;
constexpr int kLstmLayer2OutputStateTensor = 40;
constexpr int kLstmLayer2CellStateTensor = 41;
constexpr int kLstmLayer3OutputStateTensor = 61;
constexpr int kLstmLayer3CellStateTensor = 62;
constexpr int kModelOutputTensor = 66;

TEST(SpeechSpeakerId, OkGoogleTest) {
  // Read the model.
  string tflite_file_path =
      file::JoinPath(TestDataPath(), "speech_speakerid_model.tflite");
  auto model = FlatBufferModel::BuildFromFile(tflite_file_path.c_str());
  CHECK(model) << "Failed to read model from file " << tflite_file_path;

  // Initialize the interpreter.
  ::tflite::MutableOpResolver resolver;
  RegisterSelectedOps(&resolver);
  std::unique_ptr<Interpreter> interpreter;
  InterpreterBuilder(*model, resolver)(&interpreter);
  CHECK(interpreter != nullptr);
  interpreter->AllocateTensors();

  // Load the input frames.
  Frames input_frames;
  const string input_file_path =
      file::JoinPath(TestDataPath(), "speech_speakerid_model_in.csv");
  ReadFrames(input_file_path, &input_frames);

  // Load the golden output results.
  Frames output_frames;
  const string output_file_path =
      file::JoinPath(TestDataPath(), "speech_speakerid_model_out.csv");
  ReadFrames(output_file_path, &output_frames);

  const int speech_batch_size =
      interpreter->tensor(kModelInputTensor)->dims->data[0];
  const int speech_input_size =
      interpreter->tensor(kModelInputTensor)->dims->data[1];
  const int speech_output_size =
      interpreter->tensor(kModelOutputTensor)->dims->data[1];

  float* input_ptr = interpreter->tensor(kModelInputTensor)->data.f;
  float* output_ptr = interpreter->tensor(kModelOutputTensor)->data.f;

  // Clear the LSTM state for layers.
  memset(interpreter->tensor(kLstmLayer1OutputStateTensor)->data.raw, 0,
         interpreter->tensor(kLstmLayer1OutputStateTensor)->bytes);
  memset(interpreter->tensor(kLstmLayer1CellStateTensor)->data.raw, 0,
         interpreter->tensor(kLstmLayer1CellStateTensor)->bytes);

  memset(interpreter->tensor(kLstmLayer2OutputStateTensor)->data.raw, 0,
         interpreter->tensor(kLstmLayer2OutputStateTensor)->bytes);
  memset(interpreter->tensor(kLstmLayer2CellStateTensor)->data.raw, 0,
         interpreter->tensor(kLstmLayer2CellStateTensor)->bytes);

  memset(interpreter->tensor(kLstmLayer3OutputStateTensor)->data.raw, 0,
         interpreter->tensor(kLstmLayer3OutputStateTensor)->bytes);
  memset(interpreter->tensor(kLstmLayer3CellStateTensor)->data.raw, 0,
         interpreter->tensor(kLstmLayer3CellStateTensor)->bytes);
  for (int i = 0; i < input_frames.size(); i++) {
    // Feed the input to model.
    int frame_ptr = 0;
    for (int k = 0; k < speech_input_size * speech_batch_size; k++) {
      input_ptr[k] = input_frames[i][frame_ptr++];
    }
    // Run the model.
    interpreter->Invoke();
    // Validate the output.
    for (int k = 0; k < speech_output_size; k++) {
      ASSERT_NEAR(output_ptr[k], output_frames[i][k], 1e-5);
    }
  }
}

}  // namespace models
}  // namespace tflite
