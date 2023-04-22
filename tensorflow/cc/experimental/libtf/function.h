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
#ifndef TENSORFLOW_CC_EXPERIMENTAL_CORE_FUNCTION_H_
#define TENSORFLOW_CC_EXPERIMENTAL_CORE_FUNCTION_H_

#include <vector>

#include "tensorflow/c/eager/abstract_context.h"
#include "tensorflow/c/eager/abstract_function.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/cc/experimental/libtf/object.h"
#include "tensorflow/core/platform/statusor.h"

namespace tf {
namespace libtf {

class Function {
 public:
  tensorflow::Status RegisterTrace(tensorflow::AbstractFunctionPtr,
                                   TaggedValue input_signature,
                                   TaggedValue output_signature);

  // Executes this function under the execution context.
  //
  // Raises an error is no matching signature is found for TaggedValue.
  tensorflow::StatusOr<TaggedValue> Execute(tensorflow::AbstractContext*,
                                            TaggedValue) const;

 private:
  struct ConcreteFunction {
    tensorflow::AbstractFunctionPtr trace;
    TaggedValue input_signature;
    TaggedValue output_signature;
  };
  tensorflow::StatusOr<ConcreteFunction> GetConcreteFunction(TaggedValue) const;
  std::vector<ConcreteFunction> concrete_fns_;
};

}  // namespace libtf
}  // namespace tf

#endif  // TENSORFLOW_CC_EXPERIMENTAL_CORE_FUNCTION_H_
