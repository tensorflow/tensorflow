/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

// An example app that uses the sample stable delegate.

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include <memory>

#include "tensorflow/lite/acceleration/configuration/c/stable_delegate.h"
#include "tensorflow/lite/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/c/c_api.h"  // For TfLiteTensorByteSize.
#include "tensorflow/lite/c/c_api_types.h"  // For kTfLiteOk
#include "tensorflow/lite/delegates/utils/experimental/stable_delegate/delegate_loader.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"

#ifdef NDEBUG
#define CHECK(x) ((void)(x))  // Avoid warnings for otherwise unused variables.
#else
#define CHECK(x) assert(x)
#endif

using tflite::TFLiteSettings;
using tflite::TFLiteSettingsBuilder;
using tflite::delegates::utils::LoadDelegateFromSharedLibrary;

bool EndsWith(const char* whole, const char* suffix) {
  size_t whole_length = strlen(whole);
  size_t suffix_length = strlen(suffix);
  return whole_length >= suffix_length &&
         strcmp(whole + whole_length - suffix_length, suffix) == 0;
}

int main(int argc, char* argv[]) {
  // It might be nicer style to use absl command-line flags, but here we're
  // trying to minimize dependencies to keep the example as simple as possible.
  if (argc != 2) {
    fprintf(stderr,
            "Usage: sample_app_using_stable_delegate <tflite model>\n"
            "\n"
            "This program runs the model using the sample stable delegate,\n"
            "passing in some arbitrary data as input to the model.\n"
            "This sample app assumes that the model's inputs and outputs are\n"
            "float tensors.\n");
    return 1;
  }
  const char* filename = argv[1];

  // Load a model from a file (the filename would typically end with ".tflite").
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(filename);
  CHECK(model != nullptr);

  // Load the example stable delegate plugin from a shared library file.
  const TfLiteStableDelegate* stable_delegate_handle =
      LoadDelegateFromSharedLibrary(
          "tensorflow/lite/delegates/utils/experimental/"
          "sample_stable_delegate/libtensorflowlite_sample_stable_delegate.so");
  CHECK(stable_delegate_handle != nullptr);

  // Build a TFLiteSettings flatbuffer.
  // The one in this example is an empty flatbuffer,
  // but additional delegate-specific parameters could
  // be passed to the delegate via this flatbuffer.
  flatbuffers::FlatBufferBuilder flatbuffer_builder;
  TFLiteSettingsBuilder tflite_settings_builder(flatbuffer_builder);
  // Additional delegate-specific parameters can be set here.
  // The example stable delegate currently doesn't take any additional
  // delegate-specific parameters, but here's an example of what it
  // would like like if we wanted to pass additional parameters to the
  // Google Edge TPU delegate, which does have additional parameters.
#if 0
  tflite::GoogleEdgeTpuSettingsBuilder edgetpu_settings_builder(
      flatbuffer_builder);
  edgetpu_settings_builder.add_log_verbosity(10);
  tflite_settings_builder.add_google_edgetpu_settings(
      edgetpu_settings_builder.Finish());
#endif
  flatbuffers::Offset<TFLiteSettings> tflite_settings =
      tflite_settings_builder.Finish();
  flatbuffer_builder.Finish(tflite_settings);
  const TFLiteSettings* settings = flatbuffers::GetRoot<TFLiteSettings>(
      flatbuffer_builder.GetBufferPointer());
  CHECK(settings != nullptr);

  // Construct the delegate instance, using the settings flatbuffer.
  TfLiteOpaqueDelegate* opaque_delegate =
      stable_delegate_handle->delegate_plugin->create(settings);
  CHECK(opaque_delegate != nullptr);

  // Construct the model interpreter, using the model and the delegate.
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  builder.AddDelegate(opaque_delegate);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  CHECK(interpreter != nullptr);

  // Allocate tensor buffers.
  CHECK(interpreter->AllocateTensors() == kTfLiteOk);

  // Fill input buffer.
  // The only input to the test model is a single tensor of floats.
  // We fill it with some arbitrary data.
  float* input = interpreter->typed_input_tensor<float>(0);
  int64_t num_input_elements =
      TfLiteTensorByteSize(interpreter->input_tensor(0)) / sizeof(float);
  for (int i = 0; i < num_input_elements; i++) {
    input[i] = 111.222 * i;  // Some arbitrary input data.
  }

  // Run inference.
  CHECK(interpreter->Invoke() == kTfLiteOk);

  // Get output buffer.
  // The only ouput to the test model is a single tensor of floats.
  float* output = interpreter->typed_output_tensor<float>(0);
  int64_t num_output_elements =
      TfLiteTensorByteSize(interpreter->output_tensor(0)) / sizeof(float);

  // Print inputs and results of computation.
  for (int i = 0; i < num_input_elements; i++) {
    printf("input[%d] = %.3f\n", i, input[i]);
  }
  printf("\n");
  for (int i = 0; i < num_output_elements; i++) {
    printf("output[%d] = %.3f\n", i, output[i]);
  }

  // Verify results, if we're using a specific known model.
  if (EndsWith(filename, "lite/testdata/add.bin")) {
    CHECK(num_input_elements == num_output_elements);
    for (int i = 0; i < num_output_elements; i++) {
      // The add.bin model computes f(X) = (X + X) + X.
      float expected_output_i = input[i] * 3.0;
      CHECK(fabs(output[i] - expected_output_i) <= 0.0000001 * fabs(output[i]));
    }
    printf("SUCCEEDED\n");
  }

  return 0;
}
