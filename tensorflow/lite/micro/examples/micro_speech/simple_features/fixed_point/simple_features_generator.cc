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
// This implementation uses fixed point for any non-constant calculations,
// instead of floating point, to help show how this can work on platforms that
// don't have good float support.

#include "tensorflow/lite/micro/examples/micro_speech/simple_features/simple_features_generator.h"

#include <cmath>

#include "tensorflow/lite/micro/examples/micro_speech/simple_features/simple_model_settings.h"

namespace {

// q format notation: qx.y => 1 sign bit, x-1 integer bits, y fraction bits.
// Use standard (non-saturating) arithmetic with signed ints of size x+y bits.
// Sacrifice some precision to avoid use of 64-bit ints.

// q1.15 * q1.15 => q2.30
inline int32_t Q1_15_FixedMultiply_Q2_30(int16_t a, int16_t b) {
  int32_t big_a = a;
  int32_t big_b = b;
  return big_a * big_b;
}

// q2.30 * q2.30 => q10.22
inline int32_t Q2_30_FixedMultiply_Q10_22(int32_t a, int32_t b) {
  // q2.30 result
  int32_t tmp = (a >> 15) * (b >> 15);
  // q10.22 result
  return tmp >> 8;
}

// q10.22 * q10.22 => q10.22
// Will overflow if product is >= 512.
// Largest product in small test set is 465.25
inline int32_t Q10_22_FixedMultiply_Q10_22(int32_t a, int32_t b) {
  // q10.22 result
  return (a >> 11) * (b >> 11);
}

// float => q2.30
// No checking for saturation.  Only used for inputs in range [-1, 1].
inline int32_t FloatToFixed_Q2_30(float input) {
  return static_cast<int32_t>(roundf(input * (1 << 30)));
}

// Performs a discrete Fourier transform on the real inputs. This corresponds to
// rdft() in the FFT package at http://www.kurims.kyoto-u.ac.jp/~ooura/fft.html,
// and to kiss_fftr() in KISSFFT at https://github.com/mborgerding/kissfft.
// It takes in an array of float real values, and returns a result of the same
// length with q10.22 fixed point real and imaginary components interleaved, so
// fourier_output[0] is the first real value, fourier_output[1] is the first
// imaginary, fourier_output[2] is the second real, and so on.
// The calling function should ensure that the array passed in as fourier_output
// is at least time_series_size in length. Most optimized FFT implementations
// require the length to be a power of two as well, but this version doesn't
// enforce that.

// input: q2.30 fixed point.  output: q10.22 fixed point.
// Outputs interpreted as q10.22 fixed point are un-scaled.
void CalculateDiscreteFourierTransform(int32_t* time_series,
                                       int time_series_size,
                                       int32_t* fourier_output) {
  for (int i = 0; i < time_series_size / 2; ++i) {
    int32_t real = 0;
    for (int j = 0; j < time_series_size; ++j) {
      const int32_t real_scale =
          FloatToFixed_Q2_30(cos(j * i * M_PI * 2 / time_series_size));
      real += Q2_30_FixedMultiply_Q10_22(time_series[j], real_scale);
    }
    int32_t imaginary = 0;
    for (int j = 0; j < time_series_size; ++j) {
      const int32_t imaginary_scale =
          FloatToFixed_Q2_30(sin(j * i * M_PI * 2 / time_series_size));
      imaginary -= Q2_30_FixedMultiply_Q10_22(time_series[j], imaginary_scale);
    }
    fourier_output[(i * 2) + 0] = real;
    fourier_output[(i * 2) + 1] = imaginary;
  }
}

// Produces a simple sine curve that is used to ensure frequencies at the center
// of the current sample window are weighted more heavily than those at the end.
// q1.15 output format.
void CalculatePeriodicHann(int window_length, int16_t* window_function) {
  for (int i = 0; i < window_length; ++i) {
    const float real_value = (0.5 - 0.5 * cos((2 * M_PI * i) / window_length));
    int tmp = static_cast<int32_t>(roundf(real_value * (1 << 15)));
    // Saturate the 0x8000 value to 0x7fff
    if (tmp > 0x7fff) tmp = 0x7fff;
    window_function[i] = tmp;
  }
}

}  // namespace

TfLiteStatus GenerateSimpleFeatures(tflite::ErrorReporter* error_reporter,
                                    const int16_t* input, int input_size,
                                    int output_size, uint8_t* output) {
  // Ensure our input and output data arrays are valid.
  if (input_size > kMaxAudioSampleSize) {
    error_reporter->Report("Input size %d larger than %d", input_size,
                           kMaxAudioSampleSize);
    return kTfLiteError;
  }
  if (output_size != kFeatureSliceSize) {
    error_reporter->Report("Requested output size %d doesn't match %d",
                           output_size, kFeatureSliceSize);
    return kTfLiteError;
  }

  // Pre-calculate the window function we'll be applying to the input data.
  // In a real application, we'd calculate this table once in an initialization
  // function and store it for repeated reuse.
  // q1.15 format.
  int16_t window_function[kMaxAudioSampleSize];
  CalculatePeriodicHann(input_size, window_function);

  // Apply the window function to our time series input, and pad it with zeroes
  // to the next power of two.
  int32_t fixed_input[kMaxAudioSampleSize];
  for (int i = 0; i < kMaxAudioSampleSize; ++i) {
    if (i < input_size) {
      // input is int16_t.  Treat as q1.15 fixed point value in range [-1,1)
      // window_function is also q1.15 fixed point number
      fixed_input[i] = Q1_15_FixedMultiply_Q2_30(input[i], window_function[i]);
    } else {
      fixed_input[i] = 0;
    }
  }

  // Pull the frequency data from the time series sample.
  // Calculated in q10.22 format from q2.30 inputs.
  int32_t fourier_values[kMaxAudioSampleSize];
  CalculateDiscreteFourierTransform(fixed_input, kMaxAudioSampleSize,
                                    fourier_values);

  // We have the complex numbers giving us information about each frequency
  // band, but all we want to know is how strong each frequency is, so calculate
  // the squared magnitude by adding together the squares of each component.
  int32_t power_spectrum[kMaxAudioSampleSize / 2];
  for (int i = 0; i < (kMaxAudioSampleSize / 2); ++i) {
    const int32_t real = fourier_values[(i * 2) + 0];
    const int32_t imaginary = fourier_values[(i * 2) + 1];
    // q10.22 results
    power_spectrum[i] = Q10_22_FixedMultiply_Q10_22(real, real) +
                        Q10_22_FixedMultiply_Q10_22(imaginary, imaginary);
  }

  // Finally, reduce the size of the output by averaging together six adjacent
  // frequencies into each slot, producing an array of 43 values.
  // Power_spectrum numbers are q10.22.  Divide by kAverageWindowSize inside
  // loop to prevent overflow.
  for (int i = 0; i < kFeatureSliceSize; ++i) {
    int32_t average = 0;
    for (int j = 0; j < kAverageWindowSize; ++j) {
      const int index = (i * kAverageWindowSize) + j;
      if (index < (kMaxAudioSampleSize / 2)) {
        average += power_spectrum[index] / kAverageWindowSize;
      }
    }
    // Quantize the result into eight bits, effectively multiplying by two.
    // The 127.5 constant here has to match the features_max value defined in
    // tensorflow/examples/speech_commands/input_data.py, and this also assumes
    // that features_min is zero.
    //
    // q10.22 input
    // integer output
    //
    // output = (input - features_min) *
    //     (output_max - output_min) / (features_max - features_min)
    // == (input) * (255) / (127.5)
    // == input * 2
    // == input << 1
    // Also want to round to nearest integer and only keep integer bits
    // => ((input << 1) + 0x200000) >> 22
    // == (input + 0x100000) >> 21
    int32_t quantized_average = (average + 0x100000) >> 21;
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
