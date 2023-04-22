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

#ifndef TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_TF_CONCRETE_FUNCTION_TEST_PROTOS_H_
#define TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_TF_CONCRETE_FUNCTION_TEST_PROTOS_H_

#include "tensorflow/core/protobuf/struct.pb.h"

namespace tensorflow {
namespace testing {

// Returns a StructuredValue corresponding to the serialized InputSignature of a
// tf.function with 0 inputs
StructuredValue ZeroArgInputSignature();

// Returns a StructuredValue corresponding to the serialized InputSignature of a
// tf.function with 1 input
StructuredValue SingleArgInputSignature();

// Returns a StructuredValue corresponding to the serialized InputSignature of a
// tf.function with 3 inputs
StructuredValue ThreeArgInputSignature();

// Returns a StructuredValue corresponding to the serialized OutputSignature of
// a tf.function with no return values
StructuredValue ZeroReturnOutputSignature();

// Returns a StructuredValue corresponding to the serialized OutputSignature of
// a tf.function with a single tensor output
StructuredValue SingleReturnOutputSignature();

// Returns a StructuredValue corresponding to the serialized OutputSignature of
// a tf.function with three tensor outputs
StructuredValue ThreeReturnOutputSignature();

}  // namespace testing
}  // namespace tensorflow
#endif  // TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_TF_CONCRETE_FUNCTION_TEST_PROTOS_H_
