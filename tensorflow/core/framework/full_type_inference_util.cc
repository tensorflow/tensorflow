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

#include "tensorflow/core/framework/full_type_inference_util.h"

#include <functional>

#include "tensorflow/core/framework/full_type.pb.h"
#include "tensorflow/core/framework/full_type_util.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {

namespace full_type {

ForwardTypeInferenceFn ReplicateInput(int i, int n) {
  return [i, n](const std::vector<std::reference_wrapper<const FullTypeDef>>&
                    input_types) {
    const FullTypeDef& in_type = input_types.at(i).get();
    FullTypeDef ret_type;
    if (in_type.type_id() != TFT_UNSET) {
      ret_type.set_type_id(TFT_PRODUCT);
      for (int k = 0; k < n; k++) {
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

ForwardTypeInferenceFn UnaryContainerCreate(FullTypeId t, int container_idx) {
  return [t, container_idx](
             const std::vector<std::reference_wrapper<const FullTypeDef>>&
                 input_types) -> StatusOr<FullTypeDef> {
    DCHECK(input_types.size() >= container_idx);

    FullTypeDef ret_type;
    ret_type.set_type_id(TFT_PRODUCT);
    FullTypeDef* arg_t = ret_type.add_args();
    arg_t->set_type_id(t);
    *(arg_t->add_args()) = input_types[container_idx].get();

    return ret_type;
  };
}

ForwardTypeInferenceFn UnaryContainerAdd(FullTypeId t, int container_idx,
                                         int element_idx, bool homogeneous) {
  return [t, container_idx, element_idx, homogeneous](
             const std::vector<std::reference_wrapper<const FullTypeDef>>&
                 input_types) -> StatusOr<FullTypeDef> {
    DCHECK(input_types.size() >= container_idx);
    DCHECK(input_types.size() >= element_idx);

    FullTypeDef ret_type;
    ret_type.set_type_id(TFT_PRODUCT);
    FullTypeDef* cont_t = ret_type.add_args();
    cont_t->set_type_id(t);

    const FullTypeDef& in_cont_t = input_types[container_idx].get();
    const FullTypeDef& in_el_t = input_types[element_idx].get();

    if (in_cont_t.type_id() != TFT_UNSET) {
      if (in_cont_t.type_id() != t) {
        return Status(
            error::INVALID_ARGUMENT,
            absl::StrCat("expected container type ", t, " for input ",
                         container_idx, ", got ", in_cont_t.DebugString()));
      }
      *cont_t = in_cont_t;
    }

    VLOG(1) << "ContainerAddUnary: " << cont_t->DebugString() << ", "
            << in_el_t.DebugString() << ", " << container_idx << "; "
            << element_idx;
    for (const auto& tmp : input_types) {
      VLOG(1) << "  input: " << tmp.get().DebugString();
    }

    if (in_el_t.type_id() == TFT_UNSET) {
      return ret_type;
    }

    const FullTypeDef& el_t = GetArgDefaultUnset(*cont_t, 0);

    if (el_t.type_id() == TFT_UNSET) {
      cont_t->clear_args();
      *(cont_t->add_args()) = in_el_t;
      return ret_type;
    }

    if (IsSubtype(in_el_t, el_t)) {
      // Nothing to do, will not refine the container type based on a single
      // addition.
      return ret_type;
    }

    if (homogeneous) {
      return Status(error::INVALID_ARGUMENT,
                    absl::StrCat("expected a subtype of ", el_t.DebugString(),
                                 " for input ", element_idx,
                                 " of a homogeneous container ", t, ", got ",
                                 in_el_t.DebugString()));
    } else {
      // TODO(mdan): Implement if needed.
      return Status(
          error::UNIMPLEMENTED,
          absl::StrCat("need union types for heterogeneous containers.\n"
                       "A homogeneous container would expect a subtype of ",
                       el_t.DebugString(), " for input ", element_idx,
                       ", but got ", in_el_t.DebugString()));
    }
  };
}

}  // namespace full_type

}  // namespace tensorflow
