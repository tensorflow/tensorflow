/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_KERNELS_DATA_OPTIONAL_OPS_UTIL_H_
#define TENSORFLOW_CORE_KERNELS_DATA_OPTIONAL_OPS_UTIL_H_

#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/util/tensor_ops_util.h"

namespace tensorflow {
namespace data {

const char kOptionalVariantTypeName[] = "tensorflow::data::Optional";

// An `OptionalVariant` can represent either an "actual value" (a tuple of
// tensors) or "none", and may be stored in a DT_VARIANT tensor.
class OptionalVariant {
 public:
  // Create an `OptionalVariant` with no actual value.
  OptionalVariant() : values_(nullptr) {}

  // Create an `OptionalVariant` with the actual value given by the tuple of
  // tensors in `values`.
  explicit OptionalVariant(std::vector<Tensor> values) {
    values_ = std::make_shared<std::vector<Tensor>>(std::move(values));
  }

  OptionalVariant(const OptionalVariant& other) = default;

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
      values_ = std::make_shared<std::vector<Tensor>>(data.tensors());
    } else {
      values_.reset();
    }
    return true;
  }

  string DebugString() const {
    if (values_) {
      return absl::StrCat("OptionalVariant<", "values: (",
                          absl::StrJoin(*values_, ", ",
                                        [](string* s, const Tensor& elem) {
                                          *s = elem.DebugString();
                                        }),
                          ")>");
    } else {
      return absl::StrCat("OptionalVariant<None>");
    }
  }

 private:
  std::shared_ptr<const std::vector<Tensor>> values_;
};

absl::Status OptionalZerosLike(
    OpKernelContext* ctx, const OptionalVariant& x, OptionalVariant* y,
    std::function<absl::Status(OpKernelContext* ctx, const Tensor& input,
                               Tensor* out)>
        zeros_like_func);

absl::Status OptionalBinaryAdd(
    OpKernelContext* ctx, const OptionalVariant& a, const OptionalVariant& b,
    OptionalVariant* out,
    std::function<absl::Status(OpKernelContext* ctx, const Tensor& a,
                               const Tensor& b, Tensor* out)>
        binary_add_func);

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_OPTIONAL_OPS_UTIL_H_
