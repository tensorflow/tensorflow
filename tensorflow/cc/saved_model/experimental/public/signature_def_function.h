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

#ifndef TENSORFLOW_CC_SAVED_MODEL_EXPERIMENTAL_PUBLIC_SIGNATURE_DEF_FUNCTION_H_
#define TENSORFLOW_CC_SAVED_MODEL_EXPERIMENTAL_PUBLIC_SIGNATURE_DEF_FUNCTION_H_

#include <vector>

#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/experimental/saved_model/public/signature_def_function.h"
#include "tensorflow/cc/experimental/base/public/status.h"
#include "tensorflow/cc/saved_model/experimental/public/signature_def_function_metadata.h"

namespace tensorflow {
namespace experimental {
namespace cc {

// SignatureDefFunctions are functions that correspond to either:
// "signatures" saved from a TF2 SavedModel APIs:
// https://github.com/tensorflow/tensorflow/blob/8ce0600f58ed84a8c84a7bbdb014d1f09e44f4c8/tensorflow/python/saved_model/save.py#L830-L854
// Or the "SignatureDefMap" saved from TF1 SavedModel APIs:
// https://github.com/tensorflow/tensorflow/blob/8ce0600f58ed84a8c84a7bbdb014d1f09e44f4c8/tensorflow/python/saved_model/load_v1_in_v2_test.py#L170-L174
// In both cases, a SignatureDef is serialized as a SignatureDef protobuf:
// https://github.com/tensorflow/tensorflow/blob/8ce0600f58ed84a8c84a7bbdb014d1f09e44f4c8/tensorflow/core/protobuf/meta_graph.proto#L260-L330
// and represents a computation defined by a TF subgraph.
// These Signatures were primarily designed to be interoperable with the legacy
// TF 1 Session-based C++ SavedModelBundle loading APIs:
// https://github.com/tensorflow/tensorflow/blob/26c4ee0c833e74f94d0102d8b005c41a28b44445/tensorflow/cc/saved_model/loader.h#L96-L108
// SignatureDefFunctions have different semantics from regular TF2
// ConcreteFunctions, and are mainly intended provide a serving-friendly
// transition point from the TF1 Session API.
// First, SignatureDefFunctions have different calling conventions.
// SignatureDefFunctions' inputs and outputs are constrained to **flattened
// lists of TensorHandles only**. They do not support more exotic input/output
// types (like optionals, generators, etc). Additionally, this flattening means
// they will not preserve the exact interface of the original tf.function they
// were traced from, as things like composite tensors decay into their
// internal dense tensor representation.
// Second, all inputs and outputs are "named", and these names are load bearing
// (eg: they are part of the interface of tensorflow_serving):
// https://github.com/tensorflow/serving/blob/e0d247b2e4050713194b8fad0be24a0636df7209/tensorflow_serving/apis/predict.proto#L21
// https://github.com/tensorflow/serving/blob/e0d247b2e4050713194b8fad0be24a0636df7209/tensorflow_serving/apis/predict.proto#L39
// The name of each input/output is stored in the corresponding tf::Argument in
// SignatureDefFunctionMetadata::arguments(). Users must ensure the order of
// TensorHandles passed to the function matches with the order of named
// arguments. Similarly the name of the outputs is stored in
// SignatureDefFunctionMetadata::returns().
class SignatureDefFunction final {
 public:
  // Returns FunctionMetadata associated with this ConcreteFunction.
  const SignatureDefFunctionMetadata* GetFunctionMetadata();

 private:
  friend class SavedModelAPI;
  friend class ConcreteFunctionList;

  // TODO(bmzhao): Consider adding a macro for wrapping/unwrapping
  // when moving out of experimental.
  static SignatureDefFunction* wrap(TF_SignatureDefFunction* p) {
    return reinterpret_cast<SignatureDefFunction*>(p);
  }
  static TF_SignatureDefFunction* unwrap(SignatureDefFunction* p) {
    return reinterpret_cast<TF_SignatureDefFunction*>(p);
  }
};

inline const SignatureDefFunctionMetadata*
SignatureDefFunction::GetFunctionMetadata() {
  return SignatureDefFunctionMetadata::wrap(
      TF_SignatureDefFunctionGetMetadata(unwrap(this)));
}

}  // namespace cc
}  // namespace experimental
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_SAVED_MODEL_EXPERIMENTAL_PUBLIC_SIGNATURE_DEF_FUNCTION_H_
