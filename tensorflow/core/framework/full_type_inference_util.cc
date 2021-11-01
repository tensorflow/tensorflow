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

// TODO(mdan): Rename to ReplicateInput (singular).
ForwardTypeInferenceFn ReplicateInputs(int n) {
  return [n](const std::vector<std::reference_wrapper<const FullTypeDef>>&
                 input_types) {
    DCHECK(input_types.size() == 1)
        << "expected exactly one input, got " << input_types.size();

    const auto& in_type = input_types[0].get();
    FullTypeDef ret_type;
    if (in_type.type_id() != TFT_UNSET) {
      ret_type.set_type_id(TFT_PRODUCT);
      for (int i = 0; i < n; i++) {
        *(ret_type.add_args()) = in_type;
      }
    }
    return ret_type;
  };
}

// TODO(mdan): Rename to MergeIdenticalInputs.
ForwardTypeInferenceFn ReplicateIdenticalInputs() {
  return [](const std::vector<std::reference_wrapper<const FullTypeDef>>&
                input_types) -> StatusOr<FullTypeDef> {
    DCHECK(!input_types.empty());

    FullTypeDef ret_type;
    int first_known = -1;
    FullTypeDef const* first_known_t = nullptr;
    for (int i = 0; i < input_types.size(); i++) {
      const auto& t = input_types[i].get();

      if (t.type_id() == TFT_UNSET) {
        continue;
      }

      if (first_known < 0) {
        first_known = i;
        first_known_t = &t;
        *(ret_type.add_args()) = t;
        continue;
      }

      // TODO(mdan): Make a deep comparison.
      if (first_known_t->type_id() != t.type_id()) {
        return Status(
            error::INVALID_ARGUMENT,
            absl::StrCat("expected identical input types, but input ", i,
                         " differed from ", first_known, ":\n", t.DebugString(),
                         "\nvs.\n", first_known_t->DebugString()));
      }
    }

    if (first_known >= 0) {
      ret_type.set_type_id(TFT_PRODUCT);
    }
    return ret_type;
  };
}

}  // namespace full_type

}  // namespace tensorflow
