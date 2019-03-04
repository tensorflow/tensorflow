/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_STRING_VIEW_VARIANT_WRAPPER_H_
#define TENSORFLOW_CORE_KERNELS_STRING_VIEW_VARIANT_WRAPPER_H_

#include <string>

#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/variant_tensor_data.h"

namespace tensorflow {

// A wrapper class for storing an `absl::string_view` instance in a DT_VARIANT
// tensor.
class StringViewVariantWrapper {
 public:
  static constexpr const char kTypeName[] =
      "tensorflow::StringViewVariantWrapper";

  using value_type = absl::string_view;

  StringViewVariantWrapper() = default;

  explicit StringViewVariantWrapper(absl::string_view str_view)
      : str_view_(str_view) {}

  StringViewVariantWrapper(const StringViewVariantWrapper& other)
      : str_view_(other.str_view_) {}

  const absl::string_view* get() const { return &str_view_; }

  static string TypeName() { return kTypeName; }

  string DebugString() const { return string(str_view_); }

  void Encode(VariantTensorData* data) const {
    data->add_tensor(string(str_view_));
  }

  // Decode assumes that the source VariantTensorData will have a longer
  // lifetime than this StringViewVariantWrapper.
  bool Decode(const VariantTensorData& data) {
    if (data.tensors_size() != 1 || data.tensors(0).dtype() != DT_STRING) {
      return false;
    }
    str_view_ = data.tensors(0).scalar<string>()();
    return true;
  }

 private:
  absl::string_view str_view_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_STRING_VIEW_VARIANT_WRAPPER_H_
