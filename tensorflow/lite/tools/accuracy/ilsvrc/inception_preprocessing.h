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

#ifndef TENSORFLOW_LITE_TOOLS_ACCURACY_ILSVRC_INCEPTION_PREPROCESSING_H_
#define TENSORFLOW_LITE_TOOLS_ACCURACY_ILSVRC_INCEPTION_PREPROCESSING_H_

#include <utility>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/lite/tools/accuracy/stage.h"

namespace tensorflow {
namespace metrics {

// A stage that does inception preprocessing.
// Inputs: A tensor containing bytes of a JPEG image.
// Outputs: A tensor containing rescaled and preprocessed image that has
// shape {1, image_height, image_width, 3}, where 3 is the number of channels.
class InceptionPreprocessingStage : public Stage {
 public:
  // Preprocessing params that govern scaling and normalization of channels of
  // the image.
  struct Params {
    // Input means are subtracted from each channel.
    // In case of an empty vector this is skipped.
    std::vector<float> input_means;
    // Scale is used to divide the input.
    // A scale of 0 means divison is skipped.
    float scale;
    double cropping_fraction;
  };

  // Default preprocessing for inception stage based on |output_type|
  static Params DefaultParamsForType(DataType output_type) {
    const float kCroppingFraction = 0.875;
    Params params = {};
    params.cropping_fraction = kCroppingFraction;
    if (output_type == DT_UINT8) {
    } else if (output_type == DT_INT8) {
      params.input_means = {128.0, 128.0, 128.0};
    } else {
      // Assume floating point preprocessing.
      params.input_means = {127.5, 127.5, 127.5};
      params.scale = 127.5;
    }
    return params;
  }

  // Creates a new preprocessing stage object with provided |image_width|
  // |image_height| as the size of output image.
  // |output_datatype| is the datatype of output of the stage.
  InceptionPreprocessingStage(int image_width, int image_height,
                              DataType output_datatype)
      : output_datatype_(output_datatype),
        image_width_(image_width),
        image_height_(image_height) {
    params_ = DefaultParamsForType(output_datatype);
  }

  // Creates a new preprocessing stage object with provided |image_width|
  // |image_height| as the size of output image.
  // |output_datatype| is the datatype of output of the stage.
  InceptionPreprocessingStage(int image_width, int image_height,
                              DataType output_datatype, Params params)
      : output_datatype_(output_datatype),
        image_width_(image_width),
        image_height_(image_height),
        params_(std::move(params)) {}

  string name() const override { return "stage_inception_preprocess"; }
  string output_name() const override {
    return "stage_inception_preprocess_output";
  }

  void AddToGraph(const Scope& scope, const Input& input) override;

 private:
  DataType output_datatype_;
  int image_width_;
  int image_height_;
  bool is_quantized_;
  Params params_;
};

}  // namespace metrics
}  // namespace tensorflow

#endif  // TENSORFLOW_LITE_TOOLS_ACCURACY_INCEPTION_PREPROCESSING_H_
