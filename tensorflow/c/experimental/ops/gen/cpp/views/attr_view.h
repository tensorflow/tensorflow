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
#ifndef TENSORFLOW_C_EXPERIMENTAL_OPS_GEN_CPP_VIEWS_ATTR_VIEW_H_
#define TENSORFLOW_C_EXPERIMENTAL_OPS_GEN_CPP_VIEWS_ATTR_VIEW_H_

#include "tensorflow/c/experimental/ops/gen/model/attr_spec.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace generator {
namespace cpp {

class AttrView {
 public:
  explicit AttrView(AttrSpec attr) : attr_(attr) {}

  string VariableName() const;
  string VariableType() const;
  string AttrNameString() const;
  string VariableStrLen() const;
  string VariableSpanData() const;
  string VariableSpanLen() const;
  string DefaultValue() const;
  string InputArg(bool with_default_value) const;
  string SetterMethod() const;
  std::vector<string> SetterArgs() const;

 private:
  AttrSpec attr_;
};

}  // namespace cpp
}  // namespace generator
}  // namespace tensorflow

#endif  // TENSORFLOW_C_EXPERIMENTAL_OPS_GEN_CPP_VIEWS_ATTR_VIEW_H_
