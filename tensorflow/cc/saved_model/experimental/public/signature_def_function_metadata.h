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

#ifndef TENSORFLOW_CC_SAVED_MODEL_EXPERIMENTAL_PUBLIC_SIGNATURE_DEF_FUNCTION_METADATA_H_
#define TENSORFLOW_CC_SAVED_MODEL_EXPERIMENTAL_PUBLIC_SIGNATURE_DEF_FUNCTION_METADATA_H_

#include <memory>

#include "tensorflow/c/experimental/saved_model/public/signature_def_function_metadata.h"

namespace tensorflow {
namespace experimental {
namespace cc {

// SignatureDefFunctionMetadata stores additional information on each input
// and output's names, dtypes, and shape.
class SignatureDefFunctionMetadata final {
  // TODO(bmzhao): Add getters here as necessary.
 private:
  friend class SignatureDefFunction;
  static SignatureDefFunctionMetadata* wrap(
      TF_SignatureDefFunctionMetadata* p) {
    return reinterpret_cast<SignatureDefFunctionMetadata*>(p);
  }
  static TF_SignatureDefFunctionMetadata* unwrap(
      SignatureDefFunctionMetadata* p) {
    return reinterpret_cast<TF_SignatureDefFunctionMetadata*>(p);
  }
};

}  // namespace cc
}  // namespace experimental
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_SAVED_MODEL_EXPERIMENTAL_PUBLIC_SIGNATURE_DEF_FUNCTION_METADATA_H_
