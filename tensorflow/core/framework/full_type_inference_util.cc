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

#include "tensorflow/core/framework/full_type_util.h"

#include "tensorflow/core/framework/full_type.pb.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {

namespace full_type {

ForwardTypeInferenceFn ReplicateInputs(int n) {
  return [n](const std::vector<std::reference_wrapper<const FullTypeDef>>&
                 input_types) {
    FullTypeDef ret_type = input_types[0].get();
    if (ret_type.type_id() != TFT_UNSET) {
      for (int i = 1; i < n; i++) {
        *(ret_type.add_args()) = ret_type.args(0);
      }
    }
    return ret_type;
  };
}

ForwardTypeInferenceFn ReplicateIdenticalInputs() {
  return [](const std::vector<std::reference_wrapper<const FullTypeDef>>&
                input_types) -> StatusOr<FullTypeDef> {
    FullTypeDef ret_type = input_types[0].get();
    if (ret_type.type_id() != TFT_UNSET) {
      for (int i = 1; i < input_types.size(); i++) {
        if (ret_type.args(0).type_id() !=
               input_types[i].get().args(0).type_id()) {
          return Status(
              error::INVALID_ARGUMENT,
              absl::StrCat("expected identical input types, but input ", i,
                           " differed from 0:\n",
                           input_types[i].get().DebugString(), "\nvs.\n",
                           input_types[i].get().DebugString()));
        }
      }
    }
    return ret_type;
  };
}

}  // namespace full_type

}  // namespace tensorflow
