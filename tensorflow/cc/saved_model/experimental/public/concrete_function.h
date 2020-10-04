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

#ifndef TENSORFLOW_CC_SAVED_MODEL_EXPERIMENTAL_PUBLIC_CONCRETE_FUNCTION_H_
#define TENSORFLOW_CC_SAVED_MODEL_EXPERIMENTAL_PUBLIC_CONCRETE_FUNCTION_H_

#include <vector>

#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/experimental/saved_model/public/concrete_function.h"
#include "tensorflow/cc/experimental/base/public/status.h"
#include "tensorflow/cc/saved_model/experimental/public/function_metadata.h"

namespace tensorflow {
namespace experimental {
namespace cc {

// ConcreteFunction is an executable "function" loaded from a SavedModelAPI.
class ConcreteFunction final {
 public:
  // TODO(bmzhao): Adding ConcreteFunction::Run in subsequent CL, since
  // it depends on tensorflow::cc::Tensor and tensorflow::cc::TensorHandle

  // Returns FunctionMetadata associated with this ConcreteFunction.
  const FunctionMetadata* GetFunctionMetadata();

 private:
  friend class SavedModelAPI;
  friend class ConcreteFunctionList;

  // TODO(bmzhao): Consider adding a macro for wrapping/unwrapping
  // when moving out of experimental.
  static ConcreteFunction* wrap(TF_ConcreteFunction* p) {
    return reinterpret_cast<ConcreteFunction*>(p);
  }
  static TF_ConcreteFunction* unwrap(ConcreteFunction* p) {
    return reinterpret_cast<TF_ConcreteFunction*>(p);
  }
};

inline const FunctionMetadata* ConcreteFunction::GetFunctionMetadata() {
  return FunctionMetadata::wrap(TF_ConcreteFunctionGetMetadata(unwrap(this)));
}

}  // namespace cc
}  // namespace experimental
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_SAVED_MODEL_EXPERIMENTAL_PUBLIC_CONCRETE_FUNCTION_H_
