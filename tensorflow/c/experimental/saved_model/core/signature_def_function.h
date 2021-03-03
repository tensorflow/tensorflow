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

#ifndef TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_SIGNATURE_DEF_FUNCTION_H_
#define TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_SIGNATURE_DEF_FUNCTION_H_

#include <memory>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/c/eager/immediate_execution_operation.h"
#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/c/experimental/saved_model/core/signature_def_function_metadata.h"

namespace tensorflow {

// See tensorflow/cc/experimental/saved_model/public/signature_def_function.h
// for SignatureDefFunction's intended user-facing semantics.
// This class is the "implementation" C++ part of the C++/C/C++ sandwich for
// a SignatureDefFunction.
// Note(bmzhao): Implementation-wise, SignatureDefFunctions are always saved as
// a "BareConcreteFunction", w/o a FunctionSpec, rather than a SavedFunction:
// https://github.com/tensorflow/tensorflow/blob/9bcefa44cd335c1db4a703a13da09f29ae1bbdb2/tensorflow/core/protobuf/saved_object_graph.proto#L60
// Additionally they are guaranteed to be children of the .signatures attribute
// of the root object, where the child object "name" is the signature_def key:
// https://github.com/tensorflow/tensorflow/blob/9bcefa44cd335c1db4a703a13da09f29ae1bbdb2/tensorflow/python/saved_model/signature_serialization.py#L181-L230
// One of the critical requirements of SignatureDef functions is that their
// inputs and outputs are "named". For example, a `.signatures` function:
// a. Requires users to pass: kwargs of all inputs:
// https://github.com/tensorflow/tensorflow/blob/26c4ee0c833e74f94d0102d8b005c41a28b44445/tensorflow/python/saved_model/signature_serialization.py#L119-L126
// b. Returns a dictionary of named outputs.
// https://github.com/tensorflow/tensorflow/blob/26c4ee0c833e74f94d0102d8b005c41a28b44445/tensorflow/python/saved_model/signature_serialization.py#L153-L161
// Since SignatureDefFunctions do not have FunctionSpecs, but guarantee the
// dictionary of inputs/outputs, we can parse these dictionaries' keys to obtain
// the input/output names of the SignatureDef:
// https://github.com/tensorflow/tensorflow/blob/9bcefa44cd335c1db4a703a13da09f29ae1bbdb2/tensorflow/core/protobuf/meta_graph.proto#L318-L321
class SignatureDefFunction {
 public:
  virtual ~SignatureDefFunction() = default;

  // Creates a "Call" Op used to execute the function.
  virtual Status MakeCallOp(absl::Span<AbstractTensorHandle* const> inputs,
                            ImmediateOpPtr* out) const = 0;

  virtual const SignatureDefFunctionMetadata& GetFunctionMetadata() const = 0;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_SIGNATURE_DEF_FUNCTION_H_
