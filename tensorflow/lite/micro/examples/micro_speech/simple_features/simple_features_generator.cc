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

// Reference implementation of the preprocessing pipeline, with the same
// results as the audio tutorial at
// https://www.tensorflow.org/tutorials/sequences/audio_recognition
// This module takes 30ms of PCM-encoded signed 16-bit audio samples (at 16KHz,
// so 480 values), and extracts a power spectrum of frequencies. There are 43
// frequency bands in the result, derived from the original 256 output from the
// discrete Fourier transform, and averaged together in groups of 6.
// It's expected that most platforms will have optimized versions of the
// functions used here, for example replacing the DFT with an FFT, so this
// version shouldn't be used where performance is critical.

#include "tensorflow/lite/micro/examples/micro_speech/simple_features/simple_features_generator.h"

#include <cmath>

#include "tensorflow/lite/micro/examples/micro_speech/simple_features/simple_model_settings.h"

namespace {

// Needed because some platforms don't have M_PI defined.
constexpr float kPi = 3.14159265358979323846f;

// Performs a discrete Fourier transform on the real inputs. This corresponds to
// rdft() in the FFT package at http://www.kurims.kyoto-u.ac.jp/~ooura/fft.html,
// and to kiss_fftr() in KISSFFT at https://github.com/mborgerding/kissfft.
// It takes in an array of float real values, and returns a result of the same
// length with float real and imaginary components interleaved, so
// fourier_output[0] is the first real value, fourier_output[1] is the first
// imaginary, fourier_output[2] is the second real, and so on.
// The calling function should ensure that the array passed in as fourier_output
// is at least time_series_size in length. Most optimized FFT implementations
// require the length to be a power of two as well, but this version doesn't
// enforce that.
void CalculateDiscreteFourierTransform(float* time_series, int time_series_size,
                                       float* fourier_output) {
  for (int i = 0; i < time_series_size / 2; ++i) {
    float real = 0;
    for (int j = 0; j < time_series_size; ++j) {
      real += time_series[j] * std::cos(j * i * kPi * 2 / time_series_size);
    }
    float imaginary = 0;
    for (int j = 0; j < time_series_size; ++j) {
      imaginary -=
          time_series[j] * std::sin(j * i * kPi * 2 / time_series_size);
    }
    fourier_output[(i * 2) + 0] = real;
    fourier_output[(i * 2) + 1] = imaginary;
  }
}

// Produces a simple sine curve that is used to ensure frequencies at the center
// of the current sample window are weighted more heavily than those at the end.
void CalculatePeriodicHann(int window_length, float* window_function) {
  for (int i = 0; i < window_length; ++i) {
    window_function[i] = 0.5 - 0.5 * std::cos((2 * kPi * i) / window_length);
  }
}

}  // namespace

TfLiteStatus GenerateSimpleFeatures(tflite::ErrorReporter* error_reporter,
                                    const int16_t* input, int input_size,
                                    int output_size, uint8_t* output) {
  // Ensure our input and output data arrays are valid.
  if (input_size > kMaxAudioSampleSize) {
    TF_LITE_REPORT_ERROR(error_reporter, "Input size %d larger than %d",
                         input_size, kMaxAudioSampleSize);
    return kTfLiteError;
  }
  if (output_size != kFeatureSliceSize) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Requested output size %d doesn't match %d",
                         output_size, kFeatureSliceSize);
    return kTfLiteError;
  }

  // Pre-calculate the window function we'll be applying to the input data.
  // In a real application, we'd calculate this table once in an initialization
  // function and store it for repeated reuse.
  float window_function[kMaxAudioSampleSize];
  CalculatePeriodicHann(input_size, window_function);

  // Apply the window function to our time series input, and pad it with zeroes
  // to the next power of two.
  float float_input[kMaxAudioSampleSize];
  for (int i = 0; i < kMaxAudioSampleSize; ++i) {
    if (i < input_size) {
      float_input[i] =
          (input[i] * window_function[i]) / static_cast<float>(1 << 15);
    } else {
      float_input[i] = 0.0f;
    }
  }

  // Pull the frequency data from the time series sample.
  float fourier_values[kMaxAudioSampleSize];
  CalculateDiscreteFourierTransform(float_input, kMaxAudioSampleSize,
                                    fourier_values);

  // We have the complex numbers giving us information about each frequency
  // band, but all we want to know is how strong each frequency is, so calculate
  // the squared magnitude by adding together the squares of each component.
  float power_spectrum[kMaxAudioSampleSize / 2];
  for (int i = 0; i < (kMaxAudioSampleSize / 2); ++i) {
    const float real = fourier_values[(i * 2) + 0];
    const float imaginary = fourier_values[(i * 2) + 1];
    power_spectrum[i] = (real * real) + (imaginary * imaginary);
  }

  // Finally, reduce the size of the output by averaging together six adjacent
  // frequencies into each slot, producing an array of 43 values.
  for (int i = 0; i < kFeatureSliceSize; ++i) {
    float total = 0.0f;
    for (int j = 0; j < kAverageWindowSize; ++j) {
      const int index = (i * kAverageWindowSize) + j;
      if (index < (kMaxAudioSampleSize / 2)) {
        total += power_spectrum[index];
      }
    }
    const float average = total / kAverageWindowSize;
    // Quantize the result into eight bits, effectively multiplying by two.
    // The 127.5 constant here has to match the features_max value defined in
    // tensorflow/examples/speech_commands/input_data.py, and this also assumes
    // that features_min is zero. It it wasn't, we'd have to subtract it first.
    int quantized_average = roundf(average * (255.0f / 127.5f));
    if (quantized_average < 0) {
      quantized_average = 0;
    }
    if (quantized_average > 255) {
      quantized_average = 255;
    }
    output[i] = quantized_average;
  }
  return kTfLiteOk;
}
