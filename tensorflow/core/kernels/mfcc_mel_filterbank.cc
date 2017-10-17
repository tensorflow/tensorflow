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

// This code resamples the FFT bins, and smooths then with triangle-shaped
// weights to create a mel-frequency filter bank. For filter i centered at f_i,
// there is a triangular weighting of the FFT bins that extends from
// filter f_i-1 (with a value of zero at the left edge of the triangle) to f_i
// (where the filter value is 1) to f_i+1 (where the filter values returns to
// zero).

// Note: this code fails if you ask for too many channels.  The algorithm used
// here assumes that each FFT bin contributes to at most two channels: the
// right side of a triangle for channel i, and the left side of the triangle
// for channel i+1.  If you ask for so many channels that some of the
// resulting mel triangle filters are smaller than a single FFT bin, these
// channels may end up with no contributing FFT bins.  The resulting mel
// spectrum output will have some channels that are always zero.

#include "tensorflow/core/kernels/mfcc_mel_filterbank.h"

#include <math.h>

#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

MfccMelFilterbank::MfccMelFilterbank() : initialized_(false) {}

bool MfccMelFilterbank::Initialize(int input_length,
                               double input_sample_rate,
                               int output_channel_count,
                               double lower_frequency_limit,
                               double upper_frequency_limit) {
  num_channels_ = output_channel_count;
  sample_rate_  = input_sample_rate;
  input_length_ = input_length;

  if (num_channels_ < 1) {
    LOG(ERROR) << "Number of filterbank channels must be positive.";
    return false;
  }

  if (sample_rate_ <= 0) {
    LOG(ERROR) << "Sample rate must be positive.";
    return false;
  }

  if (input_length < 2) {
    LOG(ERROR) << "Input length must greater than 1.";
    return false;
  }

  if (lower_frequency_limit < 0) {
    LOG(ERROR) << "Lower frequency limit must be nonnegative.";
    return false;
  }

  if (upper_frequency_limit <= lower_frequency_limit) {
    LOG(ERROR) << "Upper frequency limit must be greater than "
               << "lower frequency limit.";
    return false;
  }

  // An extra center frequency is computed at the top to get the upper
  // limit on the high side of the final triangular filter.
  center_frequencies_.resize(num_channels_ + 1);
  const double mel_low = FreqToMel(lower_frequency_limit);
  const double mel_hi = FreqToMel(upper_frequency_limit);
  const double mel_span = mel_hi - mel_low;
  const double mel_spacing = mel_span / static_cast<double>(num_channels_ + 1);
  for (int i = 0; i < num_channels_ + 1; ++i) {
    center_frequencies_[i] = mel_low + (mel_spacing * (i + 1));
  }

  // Always exclude DC; emulate HTK.
  const double hz_per_sbin = 0.5 * sample_rate_ /
      static_cast<double>(input_length_ - 1);
  start_index_ = static_cast<int>(1.5 + (lower_frequency_limit /
                                           hz_per_sbin));
  end_index_ = static_cast<int>(upper_frequency_limit / hz_per_sbin);

  // Maps the input spectrum bin indices to filter bank channels/indices. For
  // each FFT bin, band_mapper tells us which channel this bin contributes to
  // on the right side of the triangle.  Thus this bin also contributes to the
  // left side of the next channel's triangle response.
  band_mapper_.resize(input_length_);
  int channel = 0;
  for (int i = 0; i < input_length_; ++i) {
    double melf = FreqToMel(i * hz_per_sbin);
    if ((i < start_index_) || (i > end_index_)) {
      band_mapper_[i] = -2;  // Indicate an unused Fourier coefficient.
    } else {
      while ((center_frequencies_[channel] < melf) &&
             (channel < num_channels_)) {
        ++channel;
      }
      band_mapper_[i] = channel - 1;  // Can be == -1
    }
  }

  // Create the weighting functions to taper the band edges.  The contribution
  // of any one FFT bin is based on its distance along the continuum between two
  // mel-channel center frequencies.  This bin contributes weights_[i] to the
  // current channel and 1-weights_[i] to the next channel.
  weights_.resize(input_length_);
  for (int i = 0; i < input_length_; ++i) {
    channel = band_mapper_[i];
    if ((i < start_index_) || (i > end_index_)) {
      weights_[i] = 0.0;
    } else {
      if (channel >= 0) {
        weights_[i] = (center_frequencies_[channel + 1] -
                       FreqToMel(i * hz_per_sbin)) /
            (center_frequencies_[channel + 1] - center_frequencies_[channel]);
      } else {
        weights_[i] = (center_frequencies_[0] - FreqToMel(i * hz_per_sbin)) /
            (center_frequencies_[0] - mel_low);
      }
    }
  }
  // Check the sum of FFT bin weights for every mel band to identify
  // situations where the mel bands are so narrow that they don't get
  // significant weight on enough (or any) FFT bins -- i.e., too many
  // mel bands have been requested for the given FFT size.
  std::vector<int> bad_channels;
  for (int c = 0; c < num_channels_; ++c) {
    float band_weights_sum = 0.0;
    for (int i = 0; i < input_length_; ++i) {
      if (band_mapper_[i] == c - 1) {
        band_weights_sum += (1.0 - weights_[i]);
      } else if (band_mapper_[i] == c) {
        band_weights_sum += weights_[i];
      }
    }
    // The lowest mel channels have the fewest FFT bins and the lowest
    // weights sum.  But given that the target gain at the center frequency
    // is 1.0, if the total sum of weights is 0.5, we're in bad shape.
    if (band_weights_sum < 0.5) {
      bad_channels.push_back(c);
    }
  }
  if (!bad_channels.empty()) {
    LOG(ERROR) << "Missing " << bad_channels.size() << " bands " <<
        " starting at " << bad_channels[0] <<
        " in mel-frequency design. " <<
        "Perhaps too many channels or " <<
        "not enough frequency resolution in spectrum. (" <<
        "input_length: " << input_length <<
        " input_sample_rate: " << input_sample_rate <<
        " output_channel_count: " << output_channel_count <<
        " lower_frequency_limit: " << lower_frequency_limit <<
        " upper_frequency_limit: " << upper_frequency_limit;
  }
  initialized_ = true;
  return true;
}

// Compute the mel spectrum from the squared-magnitude FFT input by taking the
// square root, then summing FFT magnitudes under triangular integration windows
// whose widths increase with frequency.
void MfccMelFilterbank::Compute(const std::vector<double> &input,
                            std::vector<double> *output) const {
  if (!initialized_) {
    LOG(ERROR) << "Mel Filterbank not initialized.";
    return;
  }

  if (input.size() <= end_index_) {
    LOG(ERROR) << "Input too short to compute filterbank";
    return;
  }

  // Ensure output is right length and reset all values.
  output->assign(num_channels_, 0.0);

  for (int i = start_index_; i <= end_index_; i++) {  // For each FFT bin
    double spec_val = sqrt(input[i]);
    double weighted = spec_val * weights_[i];
    int channel = band_mapper_[i];
    if (channel >= 0)
      (*output)[channel] += weighted;  // Right side of triangle, downward slope
    channel++;
    if (channel < num_channels_)
      (*output)[channel] += spec_val - weighted;  // Left side of triangle
  }
}

double MfccMelFilterbank::FreqToMel(double freq) const {
  return 1127.0 * log(1.0 + (freq / 700.0));
}

}  // namespace tensorflow
