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

#ifndef TENSORFLOW_CORE_KERNELS_EXTENSION_TYPE_VARIANT_H_
#define TENSORFLOW_CORE_KERNELS_EXTENSION_TYPE_VARIANT_H_

#include <vector>

#include "absl/types/span.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/variant_tensor_data.h"

namespace tensorflow {

class CompositeTensorVariantMetadata;

// Encoding for a `tf.ExtensionType` value, that can be saved as a Variant.
//
// `tf.ExtensionType` (also known as `CompositeTensor`) is a Python base class
// used to Python types that are supported by TensorFlow APIs.  Example
// ExtensionTypes include `tf.RaggedTensor` and `tf.SparseTensor`.
//
// `CompositeTensorVariant` decomposes the `ExtensionType` value into two
// parts:
//
//   * `components`: A list of Tensors, which encodes the value's dynamic
//     data -- i.e., data that may change for different executions of a graph.
//   * `metadata`: A serialized TypeSpec, which encodes the value's
//     static data -- i.e., data that is the same for all executions of a graph.
//
// CompositeTensorVariant can be stored in a Tensor with dtype=DT_VARIANT.
// Typically, extension type values are encoded with a scalar tensor containing
// a single CompositeTensorVariant value.
class CompositeTensorVariant {
 public:
  CompositeTensorVariant(const CompositeTensorVariantMetadata& metadata,
                         absl::Span<Tensor> flat_components);

  CompositeTensorVariant();
  CompositeTensorVariant(const CompositeTensorVariant& other);
  CompositeTensorVariant& operator=(CompositeTensorVariant&& other) = default;
  CompositeTensorVariant& operator=(const CompositeTensorVariant& other) =
      delete;

  // Returns the list of Tensor components that encode this value's dynamic
  // data.
  absl::Span<const Tensor> flat_components() const {
    return absl::MakeConstSpan(flat_components_);
  }

  // Returns the serialized TypeSpec that encodes the value's static data.
  const CompositeTensorVariantMetadata& metadata() const { return *metadata_; }

  // Variant methods.
  string TypeName() const { return kTypeName; }

  // Updates `VariantTensorData` with an encoding for this value.
  void Encode(VariantTensorData* data) const;

  // Updates this value to match the encoding in a given `VariantTensorData`.
  bool Decode(const VariantTensorData& data);

  // Returns a string summary for this value.
  string DebugString() const;

  // Name of this type (used for variant serialization).
  static constexpr const char kTypeName[] = "CompositeTensorVariant";

 private:
  // Tensor components for this value.
  std::vector<Tensor> flat_components_;

  // TypeSpec for this value.  CompositeTensorVariantMetadata is a thin wrapper
  // around a TypeSpecProto, which is used to retain flexibility to change the
  // variant encoding.
  //
  // Note: we use a unique_ptr, because header files in the kernels/ directory
  // are not allowed to import .pb.h files.
  std::unique_ptr<CompositeTensorVariantMetadata> metadata_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_EXTENSION_TYPE_VARIANT_H_
