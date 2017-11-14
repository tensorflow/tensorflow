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
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/strings/str_cat.h"
#include "tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/contrib/lite/toco/model.h"
#include "tensorflow/contrib/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

bool CreateIm2colArrays::Run(Model* model, std::size_t op_index) {
  auto conv_it = model->operators.begin() + op_index;
  if (conv_it->get()->type != OperatorType::kConv) {
    return false;
  }
  auto* conv_op = static_cast<ConvOperator*>(conv_it->get());
  if (conv_op->outputs.size() == 2) {
    // We already have an im2col array
    return false;
  }
  const auto& weights_array = *model->arrays[conv_op->inputs[1]];
  if (!weights_array.has_shape()) {
    // We need to yield until weights dims have been resolved, because
    // from the weights dims we determine whether an im2col array is
    // needed.
    return false;
  }
  const auto& weights_shape = weights_array.shape();
  const int kheight = weights_shape.dims(1);
  const int kwidth = weights_shape.dims(2);
  if (kwidth == 1 && kheight == 1 && conv_op->stride_width == 1 &&
      conv_op->stride_height == 1) {
    // 1x1 unstrided conv does not need an im2col array.
    return false;
  }

  // Create the im2col array.
  CHECK_EQ(conv_op->outputs.size(), 1);
  const string& im2col_array_name =
      AvailableArrayName(*model, conv_op->inputs[0] + "_im2col");
  model->GetOrCreateArray(im2col_array_name);
  conv_op->outputs.push_back(im2col_array_name);
  AddMessageF(
      "Created an im2col array for %s, with %dx%d kernel and stride_width=%d, "
      "stride_height=%d",
      LogName(*conv_op), kwidth, kheight, conv_op->stride_width,
      conv_op->stride_height);

  return true;
}

}  // namespace toco
