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

// Class for generating spectrogram slices from a waveform.
// Initialize() should be called before calls to other functions.  Once
// Initialize() has been called and returned true, The Compute*() functions can
// be called repeatedly with sequential input data (ie. the first element of the
// next input vector directly follows the last element of the previous input
// vector). Whenever enough audio samples are buffered to produce a
// new frame, it will be placed in output. Output is cleared on each
// call to Compute*(). This class is thread-unsafe, and should only be
// called from one thread at a time.
// With the default parameters, the output of this class should be very
// close to the results of the following MATLAB code:
// overlap_samples = window_length_samples - step_samples;
// window = hann(window_length_samples, 'periodic');
// S = abs(spectrogram(audio, window, overlap_samples)).^2;

#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_SPECTROGRAM_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_SPECTROGRAM_H_

#include <complex>
#include <deque>
#include <vector>

#include "third_party/fft2d/fft.h"

namespace tflite {
namespace internal {

class Spectrogram {
 public:
  Spectrogram() : initialized_(false) {}
  ~Spectrogram() {}

  // Initializes the class with a given window length and step length
  // (both in samples). Internally a Hann window is used as the window
  // function. Returns true on success, after which calls to Process()
  // are possible. window_length must be greater than 1 and step
  // length must be greater than 0.
  bool Initialize(int window_length, int step_length);

  // Initialize with an explicit window instead of a length.
  bool Initialize(const std::vector<double>& window, int step_length);

  // Processes an arbitrary amount of audio data (contained in input)
  // to yield complex spectrogram frames. After a successful call to
  // Initialize(), Process() may be called repeatedly with new input data
  // each time.  The audio input is buffered internally, and the output
  // vector is populated with as many temporally-ordered spectral slices
  // as it is possible to generate from the input.  The output is cleared
  // on each call before the new frames (if any) are added.
  //
  // The template parameters can be float or double.
  template <class InputSample, class OutputSample>
  bool ComputeComplexSpectrogram(
      const std::vector<InputSample>& input,
      std::vector<std::vector<std::complex<OutputSample>>>* output);

  // This function works as the one above, but returns the power
  // (the L2 norm, or the squared magnitude) of each complex value.
  template <class InputSample, class OutputSample>
  bool ComputeSquaredMagnitudeSpectrogram(
      const std::vector<InputSample>& input,
      std::vector<std::vector<OutputSample>>* output);

  // Return reference to the window function used internally.
  const std::vector<double>& GetWindow() const { return window_; }

  // Return the number of frequency channels in the spectrogram.
  int output_frequency_channels() const { return output_frequency_channels_; }

 private:
  template <class InputSample>
  bool GetNextWindowOfSamples(const std::vector<InputSample>& input,
                              int* input_start);
  void ProcessCoreFFT();

  int fft_length_;
  int output_frequency_channels_;
  int window_length_;
  int step_length_;
  bool initialized_;
  int samples_to_next_step_;

  std::vector<double> window_;
  std::vector<double> fft_input_output_;
  std::deque<double> input_queue_;

  // Working data areas for the FFT routines.
  std::vector<int> fft_integer_working_area_;
  std::vector<double> fft_double_working_area_;
};

// Explicit instantiations in spectrogram.cc.

extern template bool Spectrogram::ComputeComplexSpectrogram(
    const std::vector<float>& input,
    std::vector<std::vector<std::complex<float>>>*);
extern template bool Spectrogram::ComputeComplexSpectrogram(
    const std::vector<double>& input,
    std::vector<std::vector<std::complex<float>>>*);
extern template bool Spectrogram::ComputeComplexSpectrogram(
    const std::vector<float>& input,
    std::vector<std::vector<std::complex<double>>>*);
extern template bool Spectrogram::ComputeComplexSpectrogram(
    const std::vector<double>& input,
    std::vector<std::vector<std::complex<double>>>*);

extern template bool Spectrogram::ComputeSquaredMagnitudeSpectrogram(
    const std::vector<float>& input, std::vector<std::vector<float>>*);
extern template bool Spectrogram::ComputeSquaredMagnitudeSpectrogram(
    const std::vector<double>& input, std::vector<std::vector<float>>*);
extern template bool Spectrogram::ComputeSquaredMagnitudeSpectrogram(
    const std::vector<float>& input, std::vector<std::vector<double>>*);
extern template bool Spectrogram::ComputeSquaredMagnitudeSpectrogram(
    const std::vector<double>& input, std::vector<std::vector<double>>*);

}  // namespace internal
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_SPECTROGRAM_H_
