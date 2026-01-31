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

#ifndef TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_TENSOR_SPEC_H_
#define TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_TENSOR_SPEC_H_

#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/protobuf/struct.pb.h"

namespace tensorflow {

// Note(bmzhao): TensorSpec deliberately does not store the "name" from a
// TensorSpecProto. From edloper@, "Names should really be associated with
// parameters, not the tensors inside those parameters. This would be
// inconsistent with the corresponding Python class, but I don't think that's
// necessarily a problem. If it turns out later that we really need a name
// attribute here, we can always add it back in; but let's see how far we can
// get without it."
class TensorSpec {
 public:
  // Constructs a scalar, DT_FLOAT TensorSpec
  TensorSpec();

  TensorSpec(PartialTensorShape shape, DataType dtype);

  explicit TensorSpec(const TensorSpecProto& proto);

  const PartialTensorShape& shape() const;
  DataType dtype() const;

 private:
  PartialTensorShape shape_;
  DataType dtype_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_TENSOR_SPEC_H_
