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

#ifndef TENSORFLOW_C_EAGER_GRADIENTS_INTERNAL_H_
#define TENSORFLOW_C_EAGER_GRADIENTS_INTERNAL_H_

#include "tensorflow/c/eager/gradients.h"

namespace tensorflow {
namespace gradients {
namespace internal {

// Helper functions which delegate to `AbstractOperation`, update
// the state of the ForwardOperation and call the tape as appropriate.
// These APIs are mainly to facilitate testing and are subject to change.

// Records the op name in the `ForwardOperation`.
absl::Status Reset(AbstractOperation*, const char* op,
                   const char* raw_device_name, ForwardOperation*);

// Records the inputs in the `ForwardOperation`.
absl::Status AddInput(AbstractOperation*, AbstractTensorHandle*,
                      ForwardOperation*);
absl::Status AddInputList(AbstractOperation*,
                          absl::Span<AbstractTensorHandle* const> inputs,
                          ForwardOperation*);

// Sets the attrs in the `ForwardOperation`.
absl::Status SetAttrString(AbstractOperation*, const char* attr_name,
                           const char* data, size_t length, ForwardOperation*);
absl::Status SetAttrInt(AbstractOperation*, const char* attr_name,
                        int64_t value, ForwardOperation*);
absl::Status SetAttrFloat(AbstractOperation*, const char* attr_name,
                          float value, ForwardOperation*);
absl::Status SetAttrBool(AbstractOperation*, const char* attr_name, bool value,
                         ForwardOperation*);
absl::Status SetAttrType(AbstractOperation*, const char* attr_name,
                         DataType value, ForwardOperation*);
absl::Status SetAttrShape(AbstractOperation*, const char* attr_name,
                          const int64_t* dims, const int num_dims,
                          ForwardOperation*);
absl::Status SetAttrFunction(AbstractOperation*, const char* attr_name,
                             const AbstractOperation* value, ForwardOperation*);
absl::Status SetAttrFunctionName(AbstractOperation*, const char* attr_name,
                                 const char* value, size_t length,
                                 ForwardOperation*);
absl::Status SetAttrTensor(AbstractOperation*, const char* attr_name,
                           AbstractTensorInterface* tensor, ForwardOperation*);
absl::Status SetAttrStringList(AbstractOperation*, const char* attr_name,
                               const void* const* values, const size_t* lengths,
                               int num_values, ForwardOperation*);
absl::Status SetAttrFloatList(AbstractOperation*, const char* attr_name,
                              const float* values, int num_values,
                              ForwardOperation*);
absl::Status SetAttrIntList(AbstractOperation*, const char* attr_name,
                            const int64_t* values, int num_values,
                            ForwardOperation*);
absl::Status SetAttrTypeList(AbstractOperation*, const char* attr_name,
                             const DataType* values, int num_values,
                             ForwardOperation*);
absl::Status SetAttrBoolList(AbstractOperation*, const char* attr_name,
                             const unsigned char* values, int num_values,
                             ForwardOperation*);
absl::Status SetAttrShapeList(AbstractOperation*, const char* attr_name,
                              const int64_t** dims, const int* num_dims,
                              int num_values, ForwardOperation*);
absl::Status SetAttrFunctionList(AbstractOperation*, const char* attr_name,
                                 absl::Span<const AbstractOperation*> values,
                                 ForwardOperation*);

// Make the call to `Tape::RecordOperation`.
absl::Status Execute(AbstractOperation*, AbstractContext*,
                     absl::Span<AbstractTensorHandle*> retvals,
                     int* num_retvals, ForwardOperation*, Tape*,
                     const GradientRegistry&);

}  // namespace internal
}  // namespace gradients
}  // namespace tensorflow

#endif  // TENSORFLOW_C_EAGER_GRADIENTS_INTERNAL_H_
