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
#ifndef TENSORFLOW_COMPILER_TF2XLA_FRONTEND_ATTRIBUTES_UTIL_H_
#define TENSORFLOW_COMPILER_TF2XLA_FRONTEND_ATTRIBUTES_UTIL_H_

#include <string>

#include "absl/types/optional.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/platform/statusor.h"

namespace tensorflow {

// Return the FrontendAttributes stored in the AttrSlice if there are some.
//
// Return an InvalidArgument error if some attributes are present but
// cannot be parsed.
absl::StatusOr<std::optional<xla::FrontendAttributes>>
GetFrontendAttributesFromAttrSlice(const AttrSlice& attrs);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_FRONTEND_ATTRIBUTES_UTIL_H_
