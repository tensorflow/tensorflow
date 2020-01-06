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

#include "tensorflow/lite/micro/examples/micro_speech/simple_features/simple_features_generator.h"

extern "C" {
#define IFFT_FLAG_R 0
#define BIT_REVERSE_FLAG 1
#define FFT_SIZE 512
#define FFT_SIZE_DIV2 256
#include <arm_math.h>

#include "arm_cmplx_mag_squared_q10p6.h"
#include "tensorflow/lite/micro/examples/micro_speech/CMSIS/hanning.h"
}

void quantize(q15_t* bufA, q15_t* bufB, uint8_t* output);

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
}  // namespace

TfLiteStatus GenerateSimpleFeatures(tflite::ErrorReporter* error_reporter,
                                    const int16_t* input, int input_size,
                                    int output_size, uint8_t* output) {
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

  // 30ms at 16 kHz = 480 samples
  // We want to pad the rest of the 512-sample buffer with zeros
  arm_mult_q15((q15_t*)input, g_hanning, bufB, 480);
  int i;
  for (i = 480; i < 512; i++) {
    bufB[i] = 0;
  }

  // Should move init code outside of Preprocess() function
  arm_math_status =
      arm_rfft_init_q15(&S_arm_fft, FFT_SIZE, IFFT_FLAG_R, BIT_REVERSE_FLAG);
  arm_rfft_q15(&S_arm_fft, bufB, bufA);

  // The rfft function packs data as follows:
  // {real[0], real[N/2], real[1], imag[1], ..., real[N/2-1], imag[N/2-1]}
  // Below we pack as follows:
  // {real[0], 0, real[1], imag[1], ..., real[N/2-1], imag[N/2-1, real[N/2], 0}
  bufA[FFT_SIZE_DIV2] = bufA[1];
  bufA[FFT_SIZE_DIV2 + 1] = 0;
  bufA[1] = 0;
  arm_cmplx_mag_squared_q10p6(bufA, bufB, FFT_SIZE_DIV2 + 1);

  quantize(bufA, bufB, output);

  return kTfLiteOk;
}

void quantize(q15_t* bufA, q15_t* bufB, uint8_t* output) {
  int i;
  for (i = 0; i < 42; i++) {
    arm_mean_q15(bufB + 6 * i, 6, bufA + i);
  }
  arm_mean_q15(bufB + 252, 5, bufA + 42);

  for (i = 0; i < 43; i++) {
    output[i] = (uint8_t)(bufA[i] >> 5);
  }
}
