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

#ifndef TENSORFLOW_CORE_KERNELS_SPECTROGRAM_TEST_UTILS_H_
#define TENSORFLOW_CORE_KERNELS_SPECTROGRAM_TEST_UTILS_H_

#include <complex>
#include <string>
#include <vector>

#include "tensorflow/core/framework/types.h"

namespace tensorflow {

// Reads a wav format file into a vector of floating-point values with range
// -1.0 to 1.0.
bool ReadWaveFileToVector(const string& file_name, std::vector<double>* data);

// Reads a binary file containing 32-bit floating point values in the
// form [real_1, imag_1, real_2, imag_2, ...] into a rectangular array
// of complex values where row_length is the length of each inner vector.
bool ReadRawFloatFileToComplexVector(
    const string& file_name, int row_length,
    std::vector<std::vector<std::complex<double> > >* data);

// Reads a CSV file of numbers in the format 1.1+2.2i,1.1,2.2i,3.3j into data.
void ReadCSVFileToComplexVectorOrDie(
    const string& file_name,
    std::vector<std::vector<std::complex<double> > >* data);

// Reads a 2D array of floats from an ASCII text file, where each line is a row
// of the array, and elements are separated by commas.
void ReadCSVFileToArrayOrDie(const string& filename,
                             std::vector<std::vector<float> >* array);

// Write a binary file containing 64-bit floating-point values for
// reading by, for example, MATLAB.
bool WriteDoubleVectorToFile(const string& file_name,
                             const std::vector<double>& data);

// Write a binary file containing 32-bit floating-point values for
// reading by, for example, MATLAB.
bool WriteFloatVectorToFile(const string& file_name,
                            const std::vector<float>& data);

// Write a binary file containing 64-bit floating-point values for
// reading by, for example, MATLAB.
bool WriteDoubleArrayToFile(const string& file_name, int size,
                            const double* data);

// Write a binary file containing 32-bit floating-point values for
// reading by, for example, MATLAB.
bool WriteFloatArrayToFile(const string& file_name, int size,
                           const float* data);

// Write a binary file in the format read by
// ReadRawDoubleFileToComplexVector above.
bool WriteComplexVectorToRawFloatFile(
    const string& file_name,
    const std::vector<std::vector<std::complex<double> > >& data);

// Generate a sine wave with the provided parameters, and populate
// data with the samples.
void SineWave(int sample_rate, float frequency, float duration_seconds,
              std::vector<double>* data);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_SPECTROGRAM_TEST_UTILS_H_
