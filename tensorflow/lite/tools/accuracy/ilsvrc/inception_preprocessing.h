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

#ifndef TENSORFLOW_LITE_TOOLS_ACCURACY_INCEPTION_PREPROCESSING_H_
#define TENSORFLOW_LITE_TOOLS_ACCURACY_INCEPTION_PREPROCESSING_H_

#include <utility>

#include "tensorflow/lite/tools/accuracy/stage.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace metrics {

// A stage that does inception preprocessing.
// Inputs: A tensor containing bytes of a JPEG image.
// Outputs: A tensor containing rescaled and preprocessed image that has
// shape {1, image_height, image_width, 3}, where 3 is the number of channels.
class InceptionPreprocessingStage : public Stage {
 public:
  struct Params {
    std::vector<float> input_means;
    float scale;
    double cropping_fraction;
  };

  static Params DefaultParams() {
    return {.input_means = {127.5, 127.5, 127.5},
            .scale = 127.5,
            .cropping_fraction = 0.875};
  }

  // Creates a new preprocessing stage object with provided |image_width|
  // |image_height| as the size of output image.
  // If |is_quantized| is set to true then |params| is ignored since quantized
  // images don't go through any preprocessing.
  InceptionPreprocessingStage(int image_width, int image_height,
                              bool is_quantized,
                              Params params = DefaultParams())
      : image_width_(image_width),
        image_height_(image_height),
        is_quantized_(is_quantized),
        params_(std::move(params)) {}

  string name() const override { return "stage_inception_preprocess"; }
  string output_name() const override {
    return "stage_inception_preprocess_output";
  }

  void AddToGraph(const Scope& scope, const Input& input) override;

 private:
  int image_width_;
  int image_height_;
  bool is_quantized_;
  Params params_;
};

}  // namespace metrics
}  // namespace tensorflow

#endif  // TENSORFLOW_LITE_TOOLS_ACCURACY_INCEPTION_PREPROCESSING_H_
