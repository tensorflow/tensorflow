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

#include "tensorflow/contrib/lite/tools/accuracy/file_reader_stage.h"

#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/standard_ops.h"

namespace tensorflow {
namespace metrics {
void FileReaderStage::AddToGraph(const Scope& scope, const Input& input) {
  if (!scope.ok()) return;
  Scope s = scope.WithOpName(name());
  this->stage_output_ = ops::ReadFile(s.WithOpName(output_name()), input);
}
}  //  namespace metrics
}  //  namespace tensorflow
