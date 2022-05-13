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
#include <string>

#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/full_type.pb.h"
#include "tensorflow/core/framework/full_type_util.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {

namespace full_type {

// Note about error handling:
// For inputs which depend on the correctness of the op definition
// (i.e. if the op has three inputs, don't set an `i` that exceeds that),
// use DCHECK - an incorrect op def is considered a bug.
// Whereas for inputs that depend on the correctness of the graph (i.e. user
// used the correct ops), use Status - an incorrect graph is considered a user
// error.

ForwardTypeInferenceFn KeepExisting() { return nullptr; }

ForwardTypeInferenceFn ReplicateInput(int i, int n) {
  return [i, n](const TypeRefVector& input_types, const TypeRefMap& type_vars) {
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

ForwardTypeInferenceFn Merge() {
  return [](const TypeRefVector& input_types,
            const TypeRefMap& type_vars) -> StatusOr<FullTypeDef> {
    DCHECK(!input_types.empty());

    FullTypeDef merged;
    for (int i = 0; i < input_types.size(); i++) {
      const auto& t = input_types[i].get();

      if (t.type_id() == TFT_UNSET) {
        continue;
      }

      if (IsSubtype(t, merged)) {
        merged = t;
        continue;
      }
      if (IsSubtype(merged, t)) {
        continue;
      }

      return Status(error::INVALID_ARGUMENT,
                    absl::StrCat("expected compatible input types, but input ",
                                 i, ":\n", t.DebugString(),
                                 " is neither a subtype nor a supertype of the "
                                 "combined inputs preceding it:\n",
                                 merged.DebugString()));
    }

    FullTypeDef ret_type;
    if (merged.type_id() != TFT_UNSET) {
      ret_type.set_type_id(TFT_PRODUCT);
      *(ret_type.add_args()) = merged;
    }
    return ret_type;
  };
}

ForwardTypeInferenceFn Encode(FullTypeId t, int i) {
  return [t, i](const TypeRefVector& input_types,
                const TypeRefMap& type_vars) -> StatusOr<FullTypeDef> {
    DCHECK(input_types.size() >= i);

    FullTypeDef ret_type;
    const FullTypeDef& in_t = input_types[i].get();
    if (in_t.type_id() == TFT_UNSET) {
      return ret_type;
    }

    ret_type.set_type_id(TFT_PRODUCT);

    auto* enc_type = ret_type.add_args();
    enc_type->set_type_id(TFT_ENCODED);
    *enc_type->add_args() = in_t;
    enc_type->add_args()->set_type_id(t);
    return ret_type;
  };
}

ForwardTypeInferenceFn Decode(FullTypeId t, int i) {
  return [t, i](const TypeRefVector& input_types,
                const TypeRefMap& type_vars) -> StatusOr<FullTypeDef> {
    DCHECK(input_types.size() >= i);

    const FullTypeDef& in_t = input_types[i].get();

    const FullTypeId enc_tid = GetArgDefaultUnset(in_t, 1).type_id();
    if ((enc_tid != TFT_UNSET) && (enc_tid != t)) {
      return Status(error::INVALID_ARGUMENT,
                    absl::StrCat("expected encoded type ", t, " for input ", i,
                                 ", got ", in_t.DebugString()));
    }

    FullTypeDef ret_type;

    const FullTypeDef& out_t = GetArgDefaultUnset(in_t, 0);
    if (in_t.type_id() == TFT_UNSET) {
      return ret_type;
    }

    ret_type.set_type_id(TFT_PRODUCT);
    *ret_type.add_args() = out_t;
    return ret_type;
  };
}

ForwardTypeInferenceFn UnaryContainerCreate(FullTypeId t, int element_idx) {
  return
      [t, element_idx](const TypeRefVector& input_types,
                       const TypeRefMap& type_vars) -> StatusOr<FullTypeDef> {
        DCHECK(input_types.size() >= element_idx);

        FullTypeDef ret_type;
        ret_type.set_type_id(TFT_PRODUCT);
        FullTypeDef* arg_t = ret_type.add_args();
        arg_t->set_type_id(t);
        *(arg_t->add_args()) = input_types[element_idx].get();

        return ret_type;
      };
}

ForwardTypeInferenceFn UnaryContainerAdd(FullTypeId t, int container_idx,
                                         int element_idx, bool homogeneous) {
  return [t, container_idx, element_idx, homogeneous](
             const TypeRefVector& input_types,
             const TypeRefMap& type_vars) -> StatusOr<FullTypeDef> {
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

ForwardTypeInferenceFn MultiaryUnstack(
    FullTypeId t, std::function<FullTypeDef(const FullTypeDef&)> unstack) {
  return [t, unstack](const TypeRefVector& input_types,
                      const TypeRefMap& type_vars) -> StatusOr<FullTypeDef> {
    FullTypeDef ret_type;
    ret_type.set_type_id(TFT_PRODUCT);
    FullTypeDef* cont_t = ret_type.add_args();
    cont_t->set_type_id(t);
    FullTypeDef* el_t = cont_t->add_args();
    el_t->set_type_id(TFT_PRODUCT);
    for (int element_idx = 0; element_idx < input_types.size(); ++element_idx) {
      *(el_t->add_args()) = unstack(input_types[element_idx].get());
    }
    return ret_type;
  };
}

FullTypeDef UnstackTensor(const FullTypeDef& t) {
  // For now, only TFT_TENSOR and TFT_RAGGED are supported and
  // only if they have a single argument (i.e. they don't specify a shape).
  // If these have a shape in the future, this function needs to changed
  // so that the output shape is computed based on the input shape and the
  // effect of the unstack operation (e.g. a dimension is removed).
  // TFT_UNSET is also allowed to support weak type inference where
  // not having a fulltype is allowed.
  DCHECK((t.type_id() == TFT_TENSOR) || (t.type_id() == TFT_RAGGED) ||
         (t.type_id() == TFT_UNSET));
  DCHECK_LE(t.args_size(), 1);
  return t;
}

ForwardTypeInferenceFn ContainerMap(
    FullTypeId t, int input_idx,
    std::function<FullTypeDef(const FullTypeDef&)> map) {
  return [t, input_idx, map](
             const TypeRefVector& input_types,
             const TypeRefMap& type_vars) -> StatusOr<FullTypeDef> {
    DCHECK_GE(input_types.size(), input_idx);
    const FullTypeDef& in_cont_t = input_types.at(input_idx).get();
    FullTypeDef ret_type;
    if (in_cont_t.type_id() == TFT_UNSET) {
      return ret_type;
    }
    if (in_cont_t.type_id() != t) {
      return Status(error::INVALID_ARGUMENT,
                    absl::StrCat("expected type ", t, " for input ", input_idx,
                                 ", got ", in_cont_t.DebugString()));
    }
    ret_type.set_type_id(TFT_PRODUCT);
    FullTypeDef* out_cont_t = ret_type.add_args();
    out_cont_t->set_type_id(t);
    const FullTypeDef& in_el_t = GetArgDefaultUnset(in_cont_t, 0);
    if (in_el_t.type_id() == TFT_UNSET) {
      return ret_type;
    }
    if (in_el_t.type_id() != TFT_PRODUCT) {
      return Status(error::INVALID_ARGUMENT,
                    absl::StrCat("expected PRODUCT element type for input ",
                                 input_idx, ", got ", in_el_t.DebugString()));
    }
    FullTypeDef* out_el_t = out_cont_t->add_args();
    out_el_t->set_type_id(TFT_PRODUCT);
    for (int k = 0; k < in_el_t.args_size(); k++) {
      *(out_el_t->add_args()) = map(in_el_t.args(k));
    }
    return ret_type;
  };
}

ForwardTypeInferenceFn MapCovariant(FullTypeId t, FullTypeId u, int input_idx) {
  return
      [t, u, input_idx](const TypeRefVector& input_types,
                        const TypeRefMap& type_vars) -> StatusOr<FullTypeDef> {
        DCHECK_GE(input_types.size(), input_idx);
        const FullTypeDef& in_t = input_types.at(input_idx).get();
        FullTypeDef ret_type;
        if (in_t.type_id() == TFT_UNSET) {
          return ret_type;
        }
        if (in_t.type_id() != t) {
          return Status(error::INVALID_ARGUMENT,
                        absl::StrCat("expected type ", t, " for input ",
                                     input_idx, ", got ", in_t.DebugString()));
        }
        ret_type.set_type_id(TFT_PRODUCT);
        FullTypeDef* t = ret_type.add_args();
        t->set_type_id(u);
        *t->mutable_args() = in_t.args();
        return ret_type;
      };
}

FullTypeDef BatchTensor(const FullTypeDef& t) {
  // For now, just return the input type.
  // If the input type has a shape in the future, this function needs to be
  // changed so that the output shape is computed based on the input shape and
  // the effect of the op that changes the batch size (and this function would
  // require more information to do this computation).
  return t;
}

FullTypeDef ShardTensor(const FullTypeDef& t) {
  // For now, just return the input type.
  // If the input type has a shape in the future, this function needs to be
  // changed so that the output shape is computed based on the input shape and
  // the effect of the op that shards the input into multiple tensors (and this
  // function would require more information to do this computation).
  return t;
}

}  // namespace full_type

}  // namespace tensorflow
