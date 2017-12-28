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
// Unit test for speech ASR LM model using TFLite Ops.

#include <string.h>

#include <memory>
#include <string>

#include "base/logging.h"
#include "file/base/path.h"
#include "testing/base/public/googletest.h"
#include <gtest/gtest.h>
#include "tensorflow/contrib/lite/context.h"
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/models/test_utils.h"

namespace tflite {
namespace models {

constexpr int kModelInput1Tensor = 0;
constexpr int kModelInput2Tensor = 66;
constexpr int kLstmLayer1OutputStateTensor = 21;
constexpr int kLstmLayer1CellStateTensor = 22;
constexpr int kLstmLayer2OutputStateTensor = 42;
constexpr int kLstmLayer2CellStateTensor = 43;
constexpr int kLstmLayer3OutputStateTensor = 63;
constexpr int kLstmLayer3CellStateTensor = 64;
constexpr int kModelOutputTensor = 75;

static void ClearLstmStates(Interpreter* interpreter) {
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
}

TEST(SpeechAsrLm, EndToEndTest) {
  // Read the model.
  string tflite_file_path =
      file::JoinPath(TestDataPath(), "speech_asr_lm_model.tflite");
  auto model = FlatBufferModel::BuildFromFile(tflite_file_path.c_str());
  CHECK(model) << "Failed to mmap model " << tflite_file_path;

  // Initialize the interpreter.
  ops::builtin::BuiltinOpResolver builtins;
  std::unique_ptr<Interpreter> interpreter;
  InterpreterBuilder(*model, builtins)(&interpreter);
  CHECK(interpreter != nullptr);
  interpreter->AllocateTensors();

  // Load the input frames.
  Frames input_frames;
  const string input_file_path =
      file::JoinPath(TestDataPath(), "speech_asr_lm_model_in.csv");
  ReadFrames(input_file_path, &input_frames);

  // Load the golden output results.
  Frames output_frames;
  const string output_file_path =
      file::JoinPath(TestDataPath(), "speech_asr_lm_model_out.csv");
  ReadFrames(output_file_path, &output_frames);

  CHECK_EQ(interpreter->tensor(kModelInput1Tensor)->dims->size, 1);
  const int input1_size =
      interpreter->tensor(kModelInput1Tensor)->dims->data[0];
  CHECK_EQ(input1_size, 1);
  CHECK_EQ(interpreter->tensor(kModelInput2Tensor)->dims->size, 1);
  const int output_size =
      interpreter->tensor(kModelOutputTensor)->dims->data[0];
  CHECK_EQ(output_size, 1);

  int* input_lookup_ptr = interpreter->tensor(kModelInput1Tensor)->data.i32;
  int* output_lookup_ptr = interpreter->tensor(kModelInput2Tensor)->data.i32;
  float* output_ptr = interpreter->tensor(kModelOutputTensor)->data.f;


  for (int i = 0; i < input_frames.size(); i++) {
    float output_score = 0.0f;
    // Reset LSTM states for each sequence.
    ClearLstmStates(interpreter.get());
    // For subsequent inputs feed them sequentially, one-by-one.
    for (int k = 1; k < input_frames[i].size(); k++) {
      // Feed the inputs to model.
      input_lookup_ptr[0] = static_cast<int32>(input_frames[i][k - 1]);
      output_lookup_ptr[0] = static_cast<int32>(input_frames[i][k]);
      // Run the model.
      interpreter->Invoke();
      // Sum up the outputs.
      output_score += output_ptr[0];
    }
    // Validate the output.
    ASSERT_NEAR(output_score, output_frames[i][0], 1.4e-5);
  }
}

}  // namespace models
}  // namespace tflite
