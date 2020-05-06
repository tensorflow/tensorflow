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

#ifndef TENSORFLOW_CC_SAVED_MODEL_EXPERIMENTAL_PUBLIC_FUNCTION_METADATA_H_
#define TENSORFLOW_CC_SAVED_MODEL_EXPERIMENTAL_PUBLIC_FUNCTION_METADATA_H_

#include <memory>

#include "tensorflow/c/experimental/saved_model/public/function_metadata.h"

namespace tensorflow {
namespace cc {

// FunctionMetadata stores additional function information, including
// optional signaturedef feeds/fetches (for TF1-based ConcreteFunctions),
// a valid function path (for TF2-based ConcreteFunctions), and
// the types + number of inputs and outputs.
class FunctionMetadata final {
  // TODO(bmzhao): Add getters here as necessary.
 private:
  friend class ConcreteFunction;
  static FunctionMetadata* wrap(TF_FunctionMetadata* p) {
    return reinterpret_cast<FunctionMetadata*>(p);
  }
  static TF_FunctionMetadata* unwrap(FunctionMetadata* p) {
    return reinterpret_cast<TF_FunctionMetadata*>(p);
  }
};

}  // namespace cc
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_SAVED_MODEL_EXPERIMENTAL_PUBLIC_FUNCTION_METADATA_H_
