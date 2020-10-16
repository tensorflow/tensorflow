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

#ifndef TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_SIGNATURE_DEF_FUNCTION_METADATA_H_
#define TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_SIGNATURE_DEF_FUNCTION_METADATA_H_

#include <string>
#include <vector>

#include "tensorflow/c/experimental/saved_model/core/tensor_spec.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/struct.pb.h"

namespace tensorflow {

// SignatureDefParam represents a named Tensor input or output to a
// SignatureDefFunction.
class SignatureDefParam {
 public:
  SignatureDefParam(std::string name, TensorSpec spec);

  const std::string& name() const;

  const TensorSpec& spec() const;

 private:
  std::string name_;
  TensorSpec spec_;
};

class SignatureDefFunctionMetadata {
 public:
  SignatureDefFunctionMetadata() = default;
  SignatureDefFunctionMetadata(std::vector<SignatureDefParam> arguments,
                               std::vector<SignatureDefParam> returns);

  const std::vector<SignatureDefParam>& arguments() const;
  const std::vector<SignatureDefParam>& returns() const;

 private:
  std::vector<SignatureDefParam> arguments_;
  std::vector<SignatureDefParam> returns_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_SIGNATURE_DEF_FUNCTION_METADATA_H_
