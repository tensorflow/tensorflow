/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2tensorrt/utils/trt_testutils.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT

#include <map>
#include <string>
#include <vector>

#include <gmock/gmock.h>

namespace tensorflow {

namespace tensorrt {
namespace convert {

::testing::Matcher<std::vector<float>> ArrayFloatNear(
    const std::vector<float>& values, float max_abs_error, bool nan_sensitive) {
  std::vector<::testing::Matcher<float>> matchers;
  matchers.reserve(values.size());
  for (const float& v : values) {
    if (nan_sensitive) {
      matchers.emplace_back(::testing::NanSensitiveFloatNear(v, max_abs_error));
    } else if (max_abs_error == 0) {
      matchers.emplace_back(::testing::FloatEq(v));
    } else {
      EXPECT_GE(max_abs_error, 0);
      matchers.emplace_back(::testing::FloatNear(v, max_abs_error));
    }
  }
  return ::testing::ElementsAreArray(matchers);
}

nvinfer1::Dims CreateDims(const std::vector<int>& d) {
  nvinfer1::Dims dims;
  dims.nbDims = d.size();
  for (int i = 0; i < d.size(); ++i) {
    dims.d[i] = d[i];
  }
  return dims;
}

NodeDef MakeNodeDef(const std::string& name, const std::string& op,
                    const std::vector<std::string>& inputs,
                    const std::map<std::string, AttrValue> attrs) {
  NodeDef node_def;
  node_def.set_name(name);
  node_def.set_op(op);
  for (const auto& input : inputs) {
    node_def.add_input(input);
  }
  for (const auto& attr : attrs) {
    (*node_def.mutable_attr())[attr.first] = attr.second;
  }
  return node_def;
}

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
