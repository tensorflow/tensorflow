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
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

::tensorflow::Status DropIm2colArrays::Run(Model* model, std::size_t op_index,
                                           bool* modified) {
  *modified = false;
  auto conv_it = model->operators.begin() + op_index;
  if (conv_it->get()->type != OperatorType::kConv) {
    return ::tensorflow::OkStatus();
  }
  auto* conv_op = static_cast<ConvOperator*>(conv_it->get());
  if (conv_op->outputs.size() < 2) {
    // Conv op does not have im2col.
    return ::tensorflow::OkStatus();
  }

  // Drop the im2col array.
  CHECK_EQ(conv_op->outputs.size(), 2);
  model->EraseArray(conv_op->outputs[1]);
  conv_op->outputs.resize(1);
  AddMessageF("Dropped an im2col array for %s", LogName(*conv_op));

  *modified = true;
  return ::tensorflow::OkStatus();
}

}  // namespace toco
