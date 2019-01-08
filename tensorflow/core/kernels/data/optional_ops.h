/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_KERNELS_DATA_OPTIONAL_OPS_H_
#define TENSORFLOW_CORE_KERNELS_DATA_OPTIONAL_OPS_H_

#include <vector>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/util/tensor_ops_util.h"

namespace tensorflow {
namespace data {

const char kOptionalVariantTypeName[] = "tensorflow::data::Optional";

// Stores a DT_VARIANT value representing an Optional with the given value
// in the `output_index`^th output of the given kernel execution context.
Status WriteOptionalWithValueToOutput(OpKernelContext* ctx, int output_index,
                                      std::vector<Tensor> value);

// Stores a DT_VARIANT value representing an Optional with no value
// in the `output_index`^th output of the given kernel execution context.
Status WriteOptionalNoneToOutput(OpKernelContext* ctx, int output_index);

// An `OptionalVariant` can represent either an "actual value" (a tuple of
// tensors) or "none", and may be stored in a DT_VARIANT tensor.
class OptionalVariant {
 public:
  // Create an `OptionalVariant` with no actual value.
  OptionalVariant() : values_(nullptr) {}

  // Create an `OptionalVariant` with the actual value given by the tuple of
  // tensors in `values`.
  explicit OptionalVariant(std::vector<Tensor> values)
      : values_(new std::vector<Tensor>(std::move(values))) {}

  OptionalVariant(const OptionalVariant& other) : values_(other.values_) {}

  // Returns true if `this` represents an actual value.
  bool has_value() const { return values_ != nullptr; }

  // REQUIRES: `this->has_value()` must be true.
  const std::vector<Tensor>& get_values() const {
    DCHECK(values_) << "Tried to get values from an empty OptionalVariant";
    return *values_;
  }

  // Implementations of the necessary methods for using `OptionalVariant`
  // objects in DT_VARIANT tensors.
  string TypeName() const { return kOptionalVariantTypeName; }
  void Encode(VariantTensorData* data) const {
    data->set_metadata(values_ != nullptr);
    if (values_ != nullptr) {
      for (const auto& t : *values_) {
        *(data->add_tensors()) = t;
      }
    }
  }

  bool Decode(const VariantTensorData& data) {
    if (data.type_name() != TypeName()) {
      return false;
    }
    bool has_value = false;
    if (!data.get_metadata(&has_value)) {
      return false;
    }
    if (has_value) {
      values_.reset(new std::vector<Tensor>(data.tensors()));
    } else {
      values_.reset();
    }
    return true;
  }

  string DebugString() const {
    if (values_) {
      return strings::StrCat("OptionalVariant<", "values: (",
                             str_util::Join(*values_, ", ",
                                            [](string* s, const Tensor& elem) {
                                              *s = elem.DebugString();
                                            }),
                             ")>");
    } else {
      return strings::StrCat("OptionalVariant<None>");
    }
  }

 private:
  std::shared_ptr<const std::vector<Tensor>> values_;
};

template <typename Device>
Status OptionalZerosLike(OpKernelContext* ctx, const OptionalVariant& x,
                         OptionalVariant* y) {
  if (!x.has_value()) {
    *y = x;
    return Status::OK();
  }
  std::vector<Tensor> zero_tensors;
  for (const Tensor& tensor : x.get_values()) {
    Tensor zero_t;
    TF_RETURN_IF_ERROR(ZerosLikeTensor<Device>(ctx, tensor, &zero_t));
    zero_tensors.push_back(std::move(zero_t));
  }
  *y = OptionalVariant(zero_tensors);
  return Status::OK();
}

template <typename Device>
Status OptionalBinaryAdd(OpKernelContext* ctx, const OptionalVariant& a,
                         const OptionalVariant& b, OptionalVariant* out) {
  // TODO(skyewm): should adding a value to a non-value be a no-op instead?
  if (a.has_value() != b.has_value()) {
    return errors::InvalidArgument(
        "Cannot add optionals because one has a value and the other doesn't.");
  }
  if (!a.has_value()) {
    *out = a;
    return Status::OK();
  }
  if (a.get_values().size() != b.get_values().size()) {
    return errors::InvalidArgument(
        "Cannot add optionals because they have different numbers of "
        "components (",
        a.get_values().size(), " vs. ", b.get_values().size(), ").");
  }
  std::vector<Tensor> out_tensors;
  for (int i = 0; i < a.get_values().size(); ++i) {
    const Tensor& a_tensor = a.get_values()[i];
    const Tensor& b_tensor = b.get_values()[i];
    Tensor out_tensor;
    TF_RETURN_IF_ERROR(
        BinaryAddTensors<Device>(ctx, a_tensor, b_tensor, &out_tensor));
    out_tensors.push_back(std::move(out_tensor));
  }
  *out = OptionalVariant(out_tensors);
  return Status::OK();
}

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_OPTIONAL_OPS_H_
