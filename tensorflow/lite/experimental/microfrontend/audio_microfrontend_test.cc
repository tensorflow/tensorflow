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
// Unit test for TFLite Micro Frontend op.

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flexbuffers.h"  // TF:flatbuffers
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"

namespace tflite {
namespace ops {
namespace custom {

TfLiteRegistration* Register_AUDIO_MICROFRONTEND();

namespace {

using ::testing::ElementsAreArray;

class MicroFrontendOpModel : public SingleOpModel {
 public:
  MicroFrontendOpModel(int n_input, int n_frame, int n_frequency_per_frame,
                       int n_left_context, int n_right_context,
                       int n_frame_stride,
                       const std::vector<std::vector<int>>& input_shapes)
      : n_input_(n_input),
        n_frame_(n_frame),
        n_frequency_per_frame_(n_frequency_per_frame),
        n_left_context_(n_left_context),
        n_right_context_(n_right_context),
        n_frame_stride_(n_frame_stride) {
    input_ = AddInput(TensorType_INT16);
    output_ = AddOutput(TensorType_INT32);

    // Set up and pass in custom options using flexbuffer.
    flexbuffers::Builder fbb;
    fbb.Map([&]() {
      // Parameters to initialize FFT state.
      fbb.Int("sample_rate", 1000);
      fbb.Int("window_size", 25);
      fbb.Int("window_step", 10);
      fbb.Int("num_channels", 2);
      fbb.Float("upper_band_limit", 450.0);
      fbb.Float("lower_band_limit", 8.0);
      fbb.Int("smoothing_bits", 10);
      fbb.Float("even_smoothing", 0.025);
      fbb.Float("odd_smoothing", 0.06);
      fbb.Float("min_signal_remaining", 0.05);
      fbb.Bool("enable_pcan", true);
      fbb.Float("pcan_strength", 0.95);
      fbb.Float("pcan_offset", 80.0);
      fbb.Int("gain_bits", 21);
      fbb.Bool("enable_log", true);
      fbb.Int("scale_shift", 6);

      // Parameters for micro frontend.
      fbb.Int("left_context", n_left_context);
      fbb.Int("right_context", n_right_context);
      fbb.Int("frame_stride", n_frame_stride);
      fbb.Bool("zero_padding", true);
      fbb.Int("out_scale", 1);
      fbb.Bool("out_float", false);
    });
    fbb.Finish();
    SetCustomOp("MICRO_FRONTEND", fbb.GetBuffer(),
                Register_AUDIO_MICROFRONTEND);
    BuildInterpreter(input_shapes);
  }

  void SetInput(const std::vector<int16_t>& data) {
    PopulateTensor(input_, data);
  }

  std::vector<int> GetOutput() { return ExtractVector<int>(output_); }

  int num_inputs() { return n_input_; }
  int num_frmes() { return n_frame_; }
  int num_frequency_per_frame() { return n_frequency_per_frame_; }
  int num_left_context() { return n_left_context_; }
  int num_right_context() { return n_right_context_; }
  int num_frame_stride() { return n_frame_stride_; }

 protected:
  int input_;
  int output_;
  int n_input_;
  int n_frame_;
  int n_frequency_per_frame_;
  int n_left_context_;
  int n_right_context_;
  int n_frame_stride_;
};

class BaseMicroFrontendTest : public ::testing::Test {
 protected:
  // Micro frontend input.
  std::vector<int16_t> micro_frontend_input_;

  // Compares output up to tolerance to the result of the micro_frontend given
  // the input.
  void VerifyGoldens(const std::vector<int16_t>& input,
                     const std::vector<std::vector<int>>& output,
                     MicroFrontendOpModel* micro_frontend,
                     float tolerance = 1e-5) {
    // Dimensionality check.
    const int num_inputs = micro_frontend->num_inputs();
    EXPECT_GT(num_inputs, 0);

    const int num_frames = micro_frontend->num_frmes();
    EXPECT_GT(num_frames, 0);
    EXPECT_EQ(num_frames, output.size());

    const int num_frequency_per_frame =
        micro_frontend->num_frequency_per_frame();
    EXPECT_GT(num_frequency_per_frame, 0);
    EXPECT_EQ(num_frequency_per_frame, output[0].size());

    // Set up input.
    micro_frontend->SetInput(input);

    // Call Invoke.
    micro_frontend->Invoke();

    // Mimic padding behaviour with zero_padding = true.
    std::vector<int> output_flattened;
    int anchor;
    for (anchor = 0; anchor < output.size();
         anchor += micro_frontend->num_frame_stride()) {
      int frame;
      for (frame = anchor - micro_frontend->num_left_context();
           frame <= anchor + micro_frontend->num_right_context(); ++frame) {
        if (frame < 0 || frame >= output.size()) {
          // Padding with zeros.
          int j;
          for (j = 0; j < num_frequency_per_frame; ++j) {
            output_flattened.push_back(0.0);
          }
        } else {
          // Copy real output.
          for (auto data_point : output[frame]) {
            output_flattened.push_back(data_point);
          }
        }
      }
    }

    // Validate result.
    EXPECT_THAT(micro_frontend->GetOutput(),
                ElementsAreArray(output_flattened));
  }
};  // namespace

class TwoConsecutive36InputsMicroFrontendTest : public BaseMicroFrontendTest {
  void SetUp() override {
    micro_frontend_input_ = {
        0, 32767, 0, -32768, 0, 32767, 0, -32768, 0, 32767, 0, -32768,
        0, 32767, 0, -32768, 0, 32767, 0, -32768, 0, 32767, 0, -32768,
        0, 32767, 0, -32768, 0, 32767, 0, -32768, 0, 32767, 0, -32768};
  }
};

TEST_F(TwoConsecutive36InputsMicroFrontendTest, MicroFrontendBlackBoxTest) {
  const int n_input = 36;
  const int n_frame = 2;
  const int n_frequency_per_frame = 2;

  MicroFrontendOpModel micro_frontend(n_input, n_frame, n_frequency_per_frame,
                                      1, 1, 1,
                                      {
                                          {n_input},
                                      });

  // Verify the final output.
  const std::vector<std::vector<int>> micro_frontend_golden_output = {
      {479, 425}, {436, 378}};
  VerifyGoldens(micro_frontend_input_, micro_frontend_golden_output,
                &micro_frontend);
}

}  // namespace
}  // namespace custom
}  // namespace ops
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
