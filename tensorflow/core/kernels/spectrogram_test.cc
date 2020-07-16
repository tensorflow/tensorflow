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

// The MATLAB test data were generated using GenerateTestData.m.

#include "tensorflow/core/kernels/spectrogram.h"

#include <complex>
#include <vector>

#include "tensorflow/core/kernels/spectrogram_test_utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/resource_loader.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

using ::std::complex;

string InputFilename() {
  return io::JoinPath("tensorflow", "core", "kernels", "spectrogram_test_data",
                      "short_test_segment.wav");
}

string ExpectedFilename() {
  return io::JoinPath("tensorflow", "core", "kernels", "spectrogram_test_data",
                      "short_test_segment_spectrogram.csv.bin");
}

const int kDataVectorLength = 257;
const int kNumberOfFramesInTestData = 178;

string ExpectedNonPowerOfTwoFilename() {
  return io::JoinPath("tensorflow", "core", "kernels", "spectrogram_test_data",
                      "short_test_segment_spectrogram_400_200.csv.bin");
}

const int kNonPowerOfTwoDataVectorLength = 257;
const int kNumberOfFramesInNonPowerOfTwoTestData = 228;

TEST(SpectrogramTest, TooLittleDataYieldsNoFrames) {
  Spectrogram sgram;
  sgram.Initialize(400, 200);
  std::vector<double> input;
  // Generate 44 samples of audio.
  SineWave(44100, 1000.0, 0.001, &input);
  EXPECT_EQ(44, input.size());
  std::vector<std::vector<complex<double>>> output;
  sgram.ComputeComplexSpectrogram(input, &output);
  EXPECT_EQ(0, output.size());
}

TEST(SpectrogramTest, StepSizeSmallerThanWindow) {
  Spectrogram sgram;
  EXPECT_TRUE(sgram.Initialize(400, 200));
  std::vector<double> input;
  // Generate 661 samples of audio.
  SineWave(44100, 1000.0, 0.015, &input);
  EXPECT_EQ(661, input.size());
  std::vector<std::vector<complex<double>>> output;
  sgram.ComputeComplexSpectrogram(input, &output);
  EXPECT_EQ(2, output.size());
}

TEST(SpectrogramTest, StepSizeBiggerThanWindow) {
  Spectrogram sgram;
  EXPECT_TRUE(sgram.Initialize(200, 400));
  std::vector<double> input;
  // Generate 882 samples of audio.
  SineWave(44100, 1000.0, 0.02, &input);
  EXPECT_EQ(882, input.size());
  std::vector<std::vector<complex<double>>> output;
  sgram.ComputeComplexSpectrogram(input, &output);
  EXPECT_EQ(2, output.size());
}

TEST(SpectrogramTest, StepSizeBiggerThanWindow2) {
  Spectrogram sgram;
  EXPECT_TRUE(sgram.Initialize(200, 400));
  std::vector<double> input;
  // Generate more than 600 but fewer than 800 samples of audio.
  SineWave(44100, 1000.0, 0.016, &input);
  EXPECT_GT(input.size(), 600);
  EXPECT_LT(input.size(), 800);
  std::vector<std::vector<complex<double>>> output;
  sgram.ComputeComplexSpectrogram(input, &output);
  EXPECT_EQ(2, output.size());
}

TEST(SpectrogramTest,
     MultipleCallsToComputeComplexSpectrogramMayYieldDifferentNumbersOfFrames) {
  // Repeatedly pass inputs with "extra" samples beyond complete windows
  // and check that the excess points cumulate to eventually cause an
  // extra output frame.
  Spectrogram sgram;
  sgram.Initialize(200, 400);
  std::vector<double> input;
  // Generate 882 samples of audio.
  SineWave(44100, 1000.0, 0.02, &input);
  EXPECT_EQ(882, input.size());
  std::vector<std::vector<complex<double>>> output;
  const std::vector<int> expected_output_sizes = {
      2,  // One pass of input leaves 82 samples buffered after two steps of
          // 400.
      2,  // Passing in 882 samples again will now leave 164 samples buffered.
      3,  // Third time gives 246 extra samples, triggering an extra output
          // frame.
  };
  for (int expected_output_size : expected_output_sizes) {
    sgram.ComputeComplexSpectrogram(input, &output);
    EXPECT_EQ(expected_output_size, output.size());
  }
}

TEST(SpectrogramTest, CumulatingExcessInputsForOverlappingFrames) {
  // Input frames that don't fit into whole windows are cumulated even when
  // the windows have overlap (similar to
  // MultipleCallsToComputeComplexSpectrogramMayYieldDifferentNumbersOfFrames
  // but with window size/hop size swapped).
  Spectrogram sgram;
  sgram.Initialize(400, 200);
  std::vector<double> input;
  // Generate 882 samples of audio.
  SineWave(44100, 1000.0, 0.02, &input);
  EXPECT_EQ(882, input.size());
  std::vector<std::vector<complex<double>>> output;
  const std::vector<int> expected_output_sizes = {
      3,  // Windows 0..400, 200..600, 400..800 with 82 samples buffered.
      4,  // 1764 frames input; outputs from 600, 800, 1000, 1200..1600.
      5,  // 2646 frames in; outputs from 1400, 1600, 1800, 2000, 2200..2600.
  };
  for (int expected_output_size : expected_output_sizes) {
    sgram.ComputeComplexSpectrogram(input, &output);
    EXPECT_EQ(expected_output_size, output.size());
  }
}

TEST(SpectrogramTest, StepSizeEqualToWindowWorks) {
  Spectrogram sgram;
  sgram.Initialize(200, 200);
  std::vector<double> input;
  // Generate 2205 samples of audio.
  SineWave(44100, 1000.0, 0.05, &input);
  EXPECT_EQ(2205, input.size());
  std::vector<std::vector<complex<double>>> output;
  sgram.ComputeComplexSpectrogram(input, &output);
  EXPECT_EQ(11, output.size());
}

template <class ExpectedSample, class ActualSample>
void CompareComplexData(
    const std::vector<std::vector<complex<ExpectedSample>>>& expected,
    const std::vector<std::vector<complex<ActualSample>>>& actual,
    double tolerance) {
  ASSERT_EQ(actual.size(), expected.size());
  for (int i = 0; i < expected.size(); ++i) {
    ASSERT_EQ(expected[i].size(), actual[i].size());
    for (int j = 0; j < expected[i].size(); ++j) {
      ASSERT_NEAR(real(expected[i][j]), real(actual[i][j]), tolerance)
          << ": where i=" << i << " and j=" << j << ".";
      ASSERT_NEAR(imag(expected[i][j]), imag(actual[i][j]), tolerance)
          << ": where i=" << i << " and j=" << j << ".";
    }
  }
}

template <class Sample>
double GetMaximumAbsolute(const std::vector<std::vector<Sample>>& spectrogram) {
  double max_absolute = 0.0;
  for (int i = 0; i < spectrogram.size(); ++i) {
    for (int j = 0; j < spectrogram[i].size(); ++j) {
      double absolute_value = std::abs(spectrogram[i][j]);
      if (absolute_value > max_absolute) {
        max_absolute = absolute_value;
      }
    }
  }
  return max_absolute;
}

template <class ExpectedSample, class ActualSample>
void CompareMagnitudeData(
    const std::vector<std::vector<complex<ExpectedSample>>>&
        expected_complex_output,
    const std::vector<std::vector<ActualSample>>& actual_squared_magnitude,
    double tolerance) {
  ASSERT_EQ(actual_squared_magnitude.size(), expected_complex_output.size());
  for (int i = 0; i < expected_complex_output.size(); ++i) {
    ASSERT_EQ(expected_complex_output[i].size(),
              actual_squared_magnitude[i].size());
    for (int j = 0; j < expected_complex_output[i].size(); ++j) {
      ASSERT_NEAR(norm(expected_complex_output[i][j]),
                  actual_squared_magnitude[i][j], tolerance)
          << ": where i=" << i << " and j=" << j << ".";
    }
  }
}

TEST(SpectrogramTest, ReInitializationWorks) {
  Spectrogram sgram;
  sgram.Initialize(512, 256);
  std::vector<double> input;
  CHECK(
      ReadWaveFileToVector(GetDataDependencyFilepath(InputFilename()), &input));
  std::vector<std::vector<complex<double>>> first_output;
  std::vector<std::vector<complex<double>>> second_output;
  sgram.Initialize(512, 256);
  sgram.ComputeComplexSpectrogram(input, &first_output);
  // Re-Initialize it.
  sgram.Initialize(512, 256);
  sgram.ComputeComplexSpectrogram(input, &second_output);
  // Verify identical outputs.
  ASSERT_EQ(first_output.size(), second_output.size());
  int slice_size = first_output[0].size();
  for (int i = 0; i < first_output.size(); ++i) {
    ASSERT_EQ(slice_size, first_output[i].size());
    ASSERT_EQ(slice_size, second_output[i].size());
    for (int j = 0; j < slice_size; ++j) {
      ASSERT_EQ(first_output[i][j], second_output[i][j]);
    }
  }
}

TEST(SpectrogramTest, ComputedComplexDataAgreeWithMatlab) {
  const int kInputDataLength = 45870;
  Spectrogram sgram;
  sgram.Initialize(512, 256);
  std::vector<double> input;
  CHECK(
      ReadWaveFileToVector(GetDataDependencyFilepath(InputFilename()), &input));
  EXPECT_EQ(kInputDataLength, input.size());
  std::vector<std::vector<complex<double>>> expected_output;
  ASSERT_TRUE(ReadRawFloatFileToComplexVector(
      GetDataDependencyFilepath(ExpectedFilename()), kDataVectorLength,
      &expected_output));
  EXPECT_EQ(kNumberOfFramesInTestData, expected_output.size());
  EXPECT_EQ(kDataVectorLength, expected_output[0].size());
  std::vector<std::vector<complex<double>>> output;
  sgram.ComputeComplexSpectrogram(input, &output);
  CompareComplexData(expected_output, output, 1e-5);
}

TEST(SpectrogramTest, ComputedFloatComplexDataAgreeWithMatlab) {
  const int kInputDataLength = 45870;
  Spectrogram sgram;
  sgram.Initialize(512, 256);
  std::vector<double> double_input;
  CHECK(ReadWaveFileToVector(GetDataDependencyFilepath(InputFilename()),
                             &double_input));
  std::vector<float> input;
  input.assign(double_input.begin(), double_input.end());
  EXPECT_EQ(kInputDataLength, input.size());
  std::vector<std::vector<complex<double>>> expected_output;
  ASSERT_TRUE(ReadRawFloatFileToComplexVector(
      GetDataDependencyFilepath(ExpectedFilename()), kDataVectorLength,
      &expected_output));
  EXPECT_EQ(kNumberOfFramesInTestData, expected_output.size());
  EXPECT_EQ(kDataVectorLength, expected_output[0].size());
  std::vector<std::vector<complex<float>>> output;
  sgram.ComputeComplexSpectrogram(input, &output);
  CompareComplexData(expected_output, output, 1e-4);
}

TEST(SpectrogramTest, ComputedSquaredMagnitudeDataAgreeWithMatlab) {
  const int kInputDataLength = 45870;
  Spectrogram sgram;
  sgram.Initialize(512, 256);
  std::vector<double> input;
  CHECK(
      ReadWaveFileToVector(GetDataDependencyFilepath(InputFilename()), &input));
  EXPECT_EQ(kInputDataLength, input.size());
  std::vector<std::vector<complex<double>>> expected_output;
  ASSERT_TRUE(ReadRawFloatFileToComplexVector(
      GetDataDependencyFilepath(ExpectedFilename()), kDataVectorLength,
      &expected_output));
  EXPECT_EQ(kNumberOfFramesInTestData, expected_output.size());
  EXPECT_EQ(kDataVectorLength, expected_output[0].size());
  std::vector<std::vector<double>> output;
  sgram.ComputeSquaredMagnitudeSpectrogram(input, &output);
  CompareMagnitudeData(expected_output, output, 1e-3);
}

TEST(SpectrogramTest, ComputedFloatSquaredMagnitudeDataAgreeWithMatlab) {
  const int kInputDataLength = 45870;
  Spectrogram sgram;
  sgram.Initialize(512, 256);
  std::vector<double> double_input;
  CHECK(ReadWaveFileToVector(GetDataDependencyFilepath(InputFilename()),
                             &double_input));
  EXPECT_EQ(kInputDataLength, double_input.size());
  std::vector<float> input;
  input.assign(double_input.begin(), double_input.end());
  std::vector<std::vector<complex<double>>> expected_output;
  ASSERT_TRUE(ReadRawFloatFileToComplexVector(
      GetDataDependencyFilepath(ExpectedFilename()), kDataVectorLength,
      &expected_output));
  EXPECT_EQ(kNumberOfFramesInTestData, expected_output.size());
  EXPECT_EQ(kDataVectorLength, expected_output[0].size());
  std::vector<std::vector<float>> output;
  sgram.ComputeSquaredMagnitudeSpectrogram(input, &output);
  double max_absolute = GetMaximumAbsolute(output);
  EXPECT_GT(max_absolute, 2300.0);  // Verify that we have some big numbers.
  // Squaring increases dynamic range; max square is about 2300,
  // so 2e-4 is about 7 decimal digits; not bad for a float.
  CompareMagnitudeData(expected_output, output, 2e-4);
}

TEST(SpectrogramTest, ComputedNonPowerOfTwoComplexDataAgreeWithMatlab) {
  const int kInputDataLength = 45870;
  Spectrogram sgram;
  sgram.Initialize(400, 200);
  std::vector<double> input;
  CHECK(
      ReadWaveFileToVector(GetDataDependencyFilepath(InputFilename()), &input));
  EXPECT_EQ(kInputDataLength, input.size());
  std::vector<std::vector<complex<double>>> expected_output;
  ASSERT_TRUE(ReadRawFloatFileToComplexVector(
      GetDataDependencyFilepath(ExpectedNonPowerOfTwoFilename()),
      kNonPowerOfTwoDataVectorLength, &expected_output));
  EXPECT_EQ(kNumberOfFramesInNonPowerOfTwoTestData, expected_output.size());
  EXPECT_EQ(kNonPowerOfTwoDataVectorLength, expected_output[0].size());
  std::vector<std::vector<complex<double>>> output;
  sgram.ComputeComplexSpectrogram(input, &output);
  CompareComplexData(expected_output, output, 1e-5);
}

}  // namespace tensorflow
