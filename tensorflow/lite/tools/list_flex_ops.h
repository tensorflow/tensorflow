/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_TOOLS_LIST_FLEX_OPS_H_
#define TENSORFLOW_LITE_TOOLS_LIST_FLEX_OPS_H_

#include <set>
#include <string>

#include "tensorflow/lite/model.h"

namespace tflite {
namespace flex {

// Store the Op and Kernel name of an op as the key of a set or map.
struct OpKernel {
  std::string op_name;
  std::string kernel_name;
};

// The comparison function for OpKernel.
struct OpKernelCompare {
  bool operator()(const OpKernel& lhs, const OpKernel& rhs) const {
    if (lhs.op_name == rhs.op_name) {
      return lhs.kernel_name < rhs.kernel_name;
    }
    return lhs.op_name < rhs.op_name;
  }
};

using OpKernelSet = std::set<OpKernel, OpKernelCompare>;

// Find flex ops and its kernel classes inside a TFLite model and add them to
// the map flex_ops.
void AddFlexOpsFromModel(const tflite::Model* model, OpKernelSet* flex_ops);

// Serialize the list op of to a json string. If flex_ops is empty, return an
// empty string.
std::string OpListToJSONString(const OpKernelSet& flex_ops);

}  // namespace flex
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_LIST_FLEX_OPS_H_
