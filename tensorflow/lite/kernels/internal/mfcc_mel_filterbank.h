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

// Basic class for applying a mel-scale mapping to a power spectrum.

#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_MFCC_MEL_FILTERBANK_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_MFCC_MEL_FILTERBANK_H_

#include <vector>

namespace tflite {
namespace internal {

class MfccMelFilterbank {
 public:
  MfccMelFilterbank();
  bool Initialize(int input_length,  // Number of unique FFT bins fftsize/2+1.
                  double input_sample_rate, int output_channel_count,
                  double lower_frequency_limit, double upper_frequency_limit);

  // Takes a squared-magnitude spectrogram slice as input, computes a
  // triangular-mel-weighted linear-magnitude filterbank, and places the result
  // in output.
  void Compute(const std::vector<double>& input,
               std::vector<double>* output) const;

 private:
  double FreqToMel(double freq) const;
  bool initialized_;
  int num_channels_;
  double sample_rate_;
  int input_length_;
  std::vector<double> center_frequencies_;  // In mel, for each mel channel.

  // Each FFT bin b contributes to two triangular mel channels, with
  // proportion weights_[b] going into mel channel band_mapper_[b], and
  // proportion (1 - weights_[b]) going into channel band_mapper_[b] + 1.
  // Thus, weights_ contains the weighting applied to each FFT bin for the
  // upper-half of the triangular band.
  std::vector<double> weights_;  // Right-side weight for this fft  bin.

  // FFT bin i contributes to the upper side of mel channel band_mapper_[i]
  std::vector<int> band_mapper_;
  int start_index_;  // Lowest FFT bin used to calculate mel spectrum.
  int end_index_;    // Highest FFT bin used to calculate mel spectrum.
};

}  // namespace internal
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_MFCC_MEL_FILTERBANK_H_
