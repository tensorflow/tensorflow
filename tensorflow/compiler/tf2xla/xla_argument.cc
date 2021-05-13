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

#include "tensorflow/compiler/tf2xla/xla_argument.h"

#include "llvm/ADT/STLExtras.h"

namespace tensorflow {

bool XlaArgument::operator==(const XlaArgument& other) const {
  if (std::tie(kind, resource_kind, type, name, initialized, max_array_size,
               tensor_array_gradients) !=
      std::tie(other.kind, other.resource_kind, other.type, other.name,
               other.initialized, other.max_array_size,
               other.tensor_array_gradients)) {
    return false;
  }
  if (absl::holds_alternative<xla::Shape>(shape)) {
    if (!absl::holds_alternative<xla::Shape>(other.shape)) {
      return false;
    }
    if (!xla::Shape::Equal()(absl::get<xla::Shape>(shape),
                             absl::get<xla::Shape>(other.shape))) {
      return false;
    }
  } else {
    if (!absl::holds_alternative<TensorShape>(other.shape)) {
      return false;
    }
    if (absl::get<TensorShape>(shape) != absl::get<TensorShape>(other.shape)) {
      return false;
    }
  }
  if (constant_value.shape() != other.constant_value.shape()) {
    return false;
  }
  if (is_same_data_across_replicas != other.is_same_data_across_replicas) {
    return false;
  }
  return constant_value.tensor_data() == other.constant_value.tensor_data();
}

bool AnyUninitializedResourceArg(absl::Span<const XlaArgument> args) {
  return llvm::any_of(args, [](const XlaArgument& arg) {
    return arg.kind == XlaArgument::kResource && arg.type == DT_INVALID;
  });
}

}  // end namespace tensorflow
