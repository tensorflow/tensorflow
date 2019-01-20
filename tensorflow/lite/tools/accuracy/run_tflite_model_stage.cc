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

#include "tensorflow/lite/tools/accuracy/run_tflite_model_stage.h"

#include <vector>

#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/standard_ops.h"

namespace tensorflow {
namespace metrics {
void RunTFLiteModelStage::AddToGraph(const Scope& scope, const Input& input) {
  if (!scope.ok()) return;
  Scope s = scope.WithOpName(name());

  std::vector<NodeBuilder::NodeOut> _data = {ops::AsNodeOut(s, input)};
  ::tensorflow::Node* ret;
  auto builder = NodeBuilder(output_name(), "RunTFLiteModel")
                     .Input(_data)
                     .Attr("model_file_path", params_.model_file_path)
                     .Attr("input_type", params_.input_type)
                     .Attr("output_type", params_.output_type);

  s.UpdateBuilder(&builder);
  s.UpdateStatus(builder.Finalize(s.graph(), &ret));
  if (!s.ok()) return;
  s.UpdateStatus(s.DoShapeInference(ret));
  this->stage_output_ = ::tensorflow::Output(ret, 0);
}

}  //  namespace metrics
}  //  namespace tensorflow
