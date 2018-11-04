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

extern "C" {
  #define ARM_MATH_CM4
  #define IFFT_FLAG_R 0
  #define BIT_REVERSE_FLAG 1
  #define FFT_SIZE 512
  #include <arm_math.h>
  #include "tensorflow/contrib/lite/experimental/micro/examples/micro_speech/CMSIS/hann.h"
}

  #include "tensorflow/contrib/lite/experimental/micro/examples/micro_speech/CMSIS/preprocessor.h"

q15_t bufA[FFT_SIZE];
q15_t bufB[FFT_SIZE];
arm_rfft_instance_q15 S_arm_fft;
arm_status arm_math_status;

namespace {
// These constants allow us to allocate fixed-sized arrays on the stack for our
// working memory.
constexpr int kInputSize = 512;
constexpr int kAverageWindowSize = 6;
constexpr int kOutputSize =
    ((kInputSize / 2) + (kAverageWindowSize - 1)) / kAverageWindowSize;
} //namespace

TfLiteStatus Preprocess(tflite::ErrorReporter* error_reporter,
                        const int16_t* input, int input_size, int output_size,
                        uint8_t* output) {
  if (input_size > kInputSize) {
    error_reporter->Report("Input size %d larger than %d", input_size,
                           kInputSize);
    return kTfLiteError;
  }
  if (output_size != kOutputSize) {
    error_reporter->Report("Requested output size %d doesn't match %d",
                           output_size, kOutputSize);
    return kTfLiteError;
  }

  arm_mult_q15((q15_t *) input, hann, bufB, 512);

  // Should move init code outside of Preprocess() function
  arm_math_status = arm_rfft_init_q15(&S_arm_fft, FFT_SIZE, IFFT_FLAG_R, BIT_REVERSE_FLAG); 
  arm_rfft_q15(&S_arm_fft, bufB, bufA);
  arm_shift_q15(bufA, 5, bufB, FFT_SIZE);

  arm_cmplx_mag_squared_q15(bufB, bufA, 256);
  arm_shift_q15(bufA, 1, bufB, 256);

  int i;
  for (i=0; i<42; i++) {
    arm_mean_q15(bufB+6*i, 6, bufA+i);
  }
  arm_mean_q15(bufB+252, 4, bufA+42);

  for (i=0; i<43; i++) {
    output[i] = (uint8_t) (bufA[i] >> 5);
  }

  return kTfLiteOk;
}
