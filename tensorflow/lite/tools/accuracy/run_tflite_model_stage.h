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

#ifndef TENSORFLOW_LITE_TOOLS_ACCURACY_RUN_TFLITE_MODEL_STAGE_H_
#define TENSORFLOW_LITE_TOOLS_ACCURACY_RUN_TFLITE_MODEL_STAGE_H_

#include <string>

#include "tensorflow/lite/tools/accuracy/stage.h"

namespace tensorflow {
namespace metrics {
// Stage that loads and runs a TFLite model.
// Inputs: The input to TFLite model.
// Outputs: The output of running the TFLite model.
class RunTFLiteModelStage : public Stage {
 public:
  // The parameters for the stage.
  struct Params {
    string model_file_path;
    std::vector<TensorShape> output_shape;
    std::vector<DataType> input_type;
    std::vector<DataType> output_type;
  };

  explicit RunTFLiteModelStage(const Params& params) : params_(params) {}

  string name() const override { return "stage_run_tfl_model"; }
  // TODO(shashishekhar): This stage can have multiple inputs and
  // outputs, perhaps change the definition of stage.
  string output_name() const override { return "stage_run_tfl_model_output"; }

  void AddToGraph(const Scope& scope, const Input& input) override;

 private:
  Params params_;
};

}  //  namespace metrics
}  //  namespace tensorflow
#endif  // TENSORFLOW_LITE_TOOLS_ACCURACY_RUN_TFLITE_MODEL_STAGE_H_
