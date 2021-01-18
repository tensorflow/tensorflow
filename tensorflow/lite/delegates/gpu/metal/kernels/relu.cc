/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/metal/kernels/relu.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"

namespace tflite {
namespace gpu {
namespace metal {

ComputeTaskDescriptor ReLU(const OperationDef& definition,
                           const ReLUAttributes& attr) {
  ComputeTaskDescriptor desc(definition);
  desc.is_linkable = true;
  const std::string min_func =
      attr.alpha == 0 ? "FLT4(0.0f)" : "min(in_out_value * args.alpha, 0.0f)";
  if (attr.clip != 0.0) {
    desc.shader_source = "in_out_value = FLT4(clamp(in_out_value, " + min_func +
                         ", FLT4(args.clip)));";
  } else {
    desc.shader_source =
        "in_out_value = FLT4(max(in_out_value, " + min_func + "));";
  }
  desc.args.AddFloat("alpha", attr.alpha);
  desc.args.AddFloat("clip", attr.clip);
  return desc;
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
