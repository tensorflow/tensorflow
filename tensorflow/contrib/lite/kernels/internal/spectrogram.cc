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

#include "tensorflow/contrib/lite/kernels/internal/spectrogram.h"

#include <math.h>

#include "third_party/fft2d/fft.h"

namespace tflite {
namespace internal {

using std::complex;

namespace {
// Returns the default Hann window function for the spectrogram.
void GetPeriodicHann(int window_length, std::vector<double>* window) {
  // Some platforms don't have M_PI, so define a local constant here.
  const double pi = std::atan(1) * 4;
  window->resize(window_length);
  for (int i = 0; i < window_length; ++i) {
    (*window)[i] = 0.5 - 0.5 * cos((2 * pi * i) / window_length);
  }
}
}  // namespace

bool Spectrogram::Initialize(int window_length, int step_length) {
  std::vector<double> window;
  GetPeriodicHann(window_length, &window);
  return Initialize(window, step_length);
}

inline int Log2Floor(uint n) {
  if (n == 0) return -1;
  int log = 0;
  uint value = n;
  for (int i = 4; i >= 0; --i) {
    int shift = (1 << i);
    uint x = value >> shift;
    if (x != 0) {
      value = x;
      log += shift;
    }
  }
  assert(value == 1);
  return log;
}

inline int Log2Ceiling(uint n) {
  int floor = Log2Floor(n);
  if (n == (n & ~(n - 1)))  // zero or a power of two
    return floor;
  else
    return floor + 1;
}

inline uint NextPowerOfTwo(uint value) {
  int exponent = Log2Ceiling(value);
  // DCHECK_LT(exponent, std::numeric_limits<uint32>::digits);
  return 1 << exponent;
}

bool Spectrogram::Initialize(const std::vector<double>& window,
                             int step_length) {
  window_length_ = window.size();
  window_ = window;  // Copy window.
  if (window_length_ < 2) {
    // LOG(ERROR) << "Window length too short.";
    initialized_ = false;
    return false;
  }

  step_length_ = step_length;
  if (step_length_ < 1) {
    // LOG(ERROR) << "Step length must be positive.";
    initialized_ = false;
    return false;
  }

  fft_length_ = NextPowerOfTwo(window_length_);
  // CHECK(fft_length_ >= window_length_);
  output_frequency_channels_ = 1 + fft_length_ / 2;

  // Allocate 2 more than what rdft needs, so we can rationalize the layout.
  fft_input_output_.assign(fft_length_ + 2, 0.0);

  int half_fft_length = fft_length_ / 2;
  fft_double_working_area_.assign(half_fft_length, 0.0);
  fft_integer_working_area_.assign(2 + static_cast<int>(sqrt(half_fft_length)),
                                   0);
  // Set flag element to ensure that the working areas are initialized
  // on the first call to cdft.  It's redundant given the assign above,
  // but keep it as a reminder.
  fft_integer_working_area_[0] = 0;
  input_queue_.clear();
  samples_to_next_step_ = window_length_;
  initialized_ = true;
  return true;
}

template <class InputSample, class OutputSample>
bool Spectrogram::ComputeComplexSpectrogram(
    const std::vector<InputSample>& input,
    std::vector<std::vector<complex<OutputSample>>>* output) {
  if (!initialized_) {
    // LOG(ERROR) << "ComputeComplexSpectrogram() called before successful call
    // "
    //           << "to Initialize().";
    return false;
  }
  // CHECK(output);
  output->clear();
  int input_start = 0;
  while (GetNextWindowOfSamples(input, &input_start)) {
    // DCHECK_EQ(input_queue_.size(), window_length_);
    ProcessCoreFFT();  // Processes input_queue_ to fft_input_output_.
    // Add a new slice vector onto the output, to save new result to.
    output->resize(output->size() + 1);
    // Get a reference to the newly added slice to fill in.
    auto& spectrogram_slice = output->back();
    spectrogram_slice.resize(output_frequency_channels_);
    for (int i = 0; i < output_frequency_channels_; ++i) {
      // This will convert double to float if it needs to.
      spectrogram_slice[i] = complex<OutputSample>(
          fft_input_output_[2 * i], fft_input_output_[2 * i + 1]);
    }
  }
  return true;
}
// Instantiate it four ways:
template bool Spectrogram::ComputeComplexSpectrogram(
    const std::vector<float>& input, std::vector<std::vector<complex<float>>>*);
template bool Spectrogram::ComputeComplexSpectrogram(
    const std::vector<double>& input,
    std::vector<std::vector<complex<float>>>*);
template bool Spectrogram::ComputeComplexSpectrogram(
    const std::vector<float>& input,
    std::vector<std::vector<complex<double>>>*);
template bool Spectrogram::ComputeComplexSpectrogram(
    const std::vector<double>& input,
    std::vector<std::vector<complex<double>>>*);

template <class InputSample, class OutputSample>
bool Spectrogram::ComputeSquaredMagnitudeSpectrogram(
    const std::vector<InputSample>& input,
    std::vector<std::vector<OutputSample>>* output) {
  if (!initialized_) {
    // LOG(ERROR) << "ComputeSquaredMagnitudeSpectrogram() called before "
    //           << "successful call to Initialize().";
    return false;
  }
  // CHECK(output);
  output->clear();
  int input_start = 0;
  while (GetNextWindowOfSamples(input, &input_start)) {
    // DCHECK_EQ(input_queue_.size(), window_length_);
    ProcessCoreFFT();  // Processes input_queue_ to fft_input_output_.
    // Add a new slice vector onto the output, to save new result to.
    output->resize(output->size() + 1);
    // Get a reference to the newly added slice to fill in.
    auto& spectrogram_slice = output->back();
    spectrogram_slice.resize(output_frequency_channels_);
    for (int i = 0; i < output_frequency_channels_; ++i) {
      // Similar to the Complex case, except storing the norm.
      // But the norm function is known to be a performance killer,
      // so do it this way with explicit real and imagninary temps.
      const double re = fft_input_output_[2 * i];
      const double im = fft_input_output_[2 * i + 1];
      // Which finally converts double to float if it needs to.
      spectrogram_slice[i] = re * re + im * im;
    }
  }
  return true;
}
// Instantiate it four ways:
template bool Spectrogram::ComputeSquaredMagnitudeSpectrogram(
    const std::vector<float>& input, std::vector<std::vector<float>>*);
template bool Spectrogram::ComputeSquaredMagnitudeSpectrogram(
    const std::vector<double>& input, std::vector<std::vector<float>>*);
template bool Spectrogram::ComputeSquaredMagnitudeSpectrogram(
    const std::vector<float>& input, std::vector<std::vector<double>>*);
template bool Spectrogram::ComputeSquaredMagnitudeSpectrogram(
    const std::vector<double>& input, std::vector<std::vector<double>>*);

// Return true if a full window of samples is prepared; manage the queue.
template <class InputSample>
bool Spectrogram::GetNextWindowOfSamples(const std::vector<InputSample>& input,
                                         int* input_start) {
  auto input_it = input.begin() + *input_start;
  int input_remaining = input.end() - input_it;
  if (samples_to_next_step_ > input_remaining) {
    // Copy in as many samples are left and return false, no full window.
    input_queue_.insert(input_queue_.end(), input_it, input.end());
    *input_start += input_remaining;  // Increases it to input.size().
    samples_to_next_step_ -= input_remaining;
    return false;  // Not enough for a full window.
  } else {
    // Copy just enough into queue to make a new window, then trim the
    // front off the queue to make it window-sized.
    input_queue_.insert(input_queue_.end(), input_it,
                        input_it + samples_to_next_step_);
    *input_start += samples_to_next_step_;
    input_queue_.erase(
        input_queue_.begin(),
        input_queue_.begin() + input_queue_.size() - window_length_);
    // DCHECK_EQ(window_length_, input_queue_.size());
    samples_to_next_step_ = step_length_;  // Be ready for next time.
    return true;  // Yes, input_queue_ now contains exactly a window-full.
  }
}

void Spectrogram::ProcessCoreFFT() {
  for (int j = 0; j < window_length_; ++j) {
    fft_input_output_[j] = input_queue_[j] * window_[j];
  }
  // Zero-pad the rest of the input buffer.
  for (int j = window_length_; j < fft_length_; ++j) {
    fft_input_output_[j] = 0.0;
  }
  const int kForwardFFT = 1;  // 1 means forward; -1 reverse.
  // This real FFT is a fair amount faster than using cdft here.
  rdft(fft_length_, kForwardFFT, &fft_input_output_[0],
       &fft_integer_working_area_[0], &fft_double_working_area_[0]);
  // Make rdft result look like cdft result;
  // unpack the last real value from the first position's imag slot.
  fft_input_output_[fft_length_] = fft_input_output_[1];
  fft_input_output_[fft_length_ + 1] = 0;
  fft_input_output_[1] = 0;
}

}  // namespace internal
}  // namespace tflite
