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
// Unit test for speech Hotword model using TFLite Ops.

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

void RunTest(int model_input_tensor, int svdf_layer_state_tensor,
             int model_output_tensor, const string& model_name,
             const string& golden_in_name, const string& golden_out_name) {
  // Read the model.
  string tflite_file_path = file::JoinPath(TestDataPath(), model_name);
  auto model = FlatBufferModel::BuildFromFile(tflite_file_path.c_str());
  CHECK(model) << "Failed to read model from file " << tflite_file_path;

  // Initialize the interpreter.
  ops::builtin::BuiltinOpResolver builtins;
  std::unique_ptr<Interpreter> interpreter;
  InterpreterBuilder(*model, builtins)(&interpreter);
  CHECK(interpreter != nullptr);
  interpreter->AllocateTensors();

  // Reset the SVDF layer state.
  memset(interpreter->tensor(svdf_layer_state_tensor)->data.raw, 0,
         interpreter->tensor(svdf_layer_state_tensor)->bytes);

  // Load the input frames.
  Frames input_frames;
  const string input_file_path = file::JoinPath(TestDataPath(), golden_in_name);
  ReadFrames(input_file_path, &input_frames);

  // Load the golden output results.
  Frames output_frames;
  const string output_file_path =
      file::JoinPath(TestDataPath(), golden_out_name);
  ReadFrames(output_file_path, &output_frames);

  const int speech_batch_size =
      interpreter->tensor(model_input_tensor)->dims->data[0];
  const int speech_input_size =
      interpreter->tensor(model_input_tensor)->dims->data[1];
  const int speech_output_size =
      interpreter->tensor(model_output_tensor)->dims->data[1];
  const int input_sequence_size =
      input_frames[0].size() / (speech_input_size * speech_batch_size);
  float* input_ptr = interpreter->tensor(model_input_tensor)->data.f;
  float* output_ptr = interpreter->tensor(model_output_tensor)->data.f;

  // The first layer (SVDF) input size is 40 (speech_input_size). Each speech
  // input frames for this model is 1280 floats, which can be fed to input in a
  // sequence of size 32 (input_sequence_size).
  for (int i = 0; i < TestInputSize(input_frames); i++) {
    int frame_ptr = 0;
    for (int s = 0; s < input_sequence_size; s++) {
      for (int k = 0; k < speech_input_size * speech_batch_size; k++) {
        input_ptr[k] = input_frames[i][frame_ptr++];
      }
      interpreter->Invoke();
    }
    // After the whole frame (1280 floats) is fed, we can check the output frame
    // matches with the golden output frame.
    for (int k = 0; k < speech_output_size; k++) {
      ASSERT_NEAR(output_ptr[k], output_frames[i][k], 1e-5);
    }
  }
}

TEST(SpeechHotword, OkGoogleTestRank1) {
  constexpr int kModelInputTensor = 0;
  constexpr int kSvdfLayerStateTensor = 4;
  constexpr int kModelOutputTensor = 18;

  RunTest(kModelInputTensor, kSvdfLayerStateTensor, kModelOutputTensor,
          "speech_hotword_model_rank1.tflite", "speech_hotword_model_in.csv",
          "speech_hotword_model_out_rank1.csv");
}

TEST(SpeechHotword, OkGoogleTestRank2) {
  constexpr int kModelInputTensor = 17;
  constexpr int kSvdfLayerStateTensor = 1;
  constexpr int kModelOutputTensor = 18;
  RunTest(kModelInputTensor, kSvdfLayerStateTensor, kModelOutputTensor,
          "speech_hotword_model_rank2.tflite", "speech_hotword_model_in.csv",
          "speech_hotword_model_out_rank2.csv");
}

}  // namespace models
}  // namespace tflite
