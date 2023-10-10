/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/c/experimental/ops/gen/cpp/views/arg_type_view.h"

#include "tensorflow/core/framework/types.pb.h"

namespace tensorflow {
namespace generator {
namespace cpp {

ArgTypeView::ArgTypeView(ArgType arg_type) : arg_type_(arg_type) {}

string ArgTypeView::TypeName() const {
  if (arg_type_.is_read_only()) {
    if (arg_type_.is_list()) {
      return "absl::Span<AbstractTensorHandle* const>";
    } else {
      return "AbstractTensorHandle* const";
    }
  } else {
    if (arg_type_.is_list()) {
      return "absl::Span<AbstractTensorHandle*>";
    } else {
      return "AbstractTensorHandle**";
    }
  }
}

}  // namespace cpp
}  // namespace generator
}  // namespace tensorflow
