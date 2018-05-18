// Copyright 2017 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
#include "tensorflow/contrib/tensor_forest/kernels/v4/params.h"
#include <math.h>
#include <stdlib.h>
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace tensorforest {

float ResolveParam(const DepthDependentParam& param, int32 depth) {
  float val;
  switch (param.ParamType_case()) {
    case DepthDependentParam::kConstantValue:
      return param.constant_value();

    case DepthDependentParam::kLinear:
      val = depth * param.linear().slope() + param.linear().y_intercept();
      return std::min(std::max(val, param.linear().min_val()),
                      param.linear().max_val());

    case DepthDependentParam::kExponential:
      return param.exponential().bias() +
             param.exponential().multiplier() *
                 static_cast<float>(
                     pow(param.exponential().base(),
                         param.exponential().depth_multiplier() * depth));

    case DepthDependentParam::kThreshold:
      if (depth >= param.threshold().threshold()) {
        return param.threshold().on_value();
      } else {
        return param.threshold().off_value();
      }

    default:
      LOG(FATAL) << "unknown parameter type";
  }
}

}  // namespace tensorforest
}  // namespace tensorflow
