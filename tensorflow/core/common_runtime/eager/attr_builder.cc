/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/eager/attr_builder.h"

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/eager/kernel_and_device.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/tensor_slice_reader_cache.h"

namespace tensorflow {
namespace {

mutex g_op_name_to_attr_type_map_lock(LINKER_INITIALIZED);

std::unordered_map<string, const AttrTypeMap*>* OpNameToAttrTypeMap() {
  static auto* const m = new std::unordered_map<string, const AttrTypeMap*>;
  return m;
}

const uint32 kIsList = 1U << 31;

}  // namespace

Status OpDefForOp(const char* op_name, const OpDef** op_def) {
  const OpRegistrationData* op_reg_data = nullptr;
  Status s = OpRegistry::Global()->LookUp(op_name, &op_reg_data);
  if (s.ok()) {
    *op_def = &op_reg_data->op_def;
  }
  return s;
}

Status AttrTypeMapForOp(const char* op_name, const AttrTypeMap** out) {
  mutex_lock l(g_op_name_to_attr_type_map_lock);
  *out = gtl::FindPtrOrNull(*OpNameToAttrTypeMap(), op_name);
  if (*out != nullptr) return Status::OK();
  const OpDef* op_def = nullptr;
  Status s = OpDefForOp(op_name, &op_def);
  if (!s.ok()) return s;
  std::unique_ptr<AttrTypeMap> m(new AttrTypeMap);
  // TODO(agarwal): Avoid having to create this "registry" at runtime,
  // perhaps can be done at op registration time?
  for (const auto& attr : op_def->attr()) {
    string type = attr.type();
    const bool is_list = (type.length() > 6 && type.compare(0, 4, "list") == 0);
    if (is_list) {
      type = type.substr(5, type.length() - 6);
    }
    uint32 t = is_list ? kIsList : 0;
    if (type == "string") {
      t |= TF_ATTR_STRING;
    } else if (type == "int") {
      t |= TF_ATTR_INT;
    } else if (type == "float") {
      t |= TF_ATTR_FLOAT;
    } else if (type == "bool") {
      t |= TF_ATTR_BOOL;
    } else if (type == "type") {
      t |= TF_ATTR_TYPE;
    } else if (type == "shape") {
      t |= TF_ATTR_SHAPE;
    } else if (type == "tensor") {
      t |= TF_ATTR_TENSOR;
    } else if (type == "func") {
      t |= TF_ATTR_FUNC;
    } else {
      return errors::Unimplemented(
          "TODO(agarwal): Enable support for ops with attributes of type '",
          type, "'");
    }
    gtl::InsertIfNotPresent(m.get(), attr.name(), t);
  }
  *out = m.get();
  (*OpNameToAttrTypeMap())[op_name] = m.release();
  return Status::OK();
}

#define DEFINE_SET_ATTR(value_type, value_field)                             \
  template <>                                                                \
  AttrBuilder& AttrBuilder::Set(StringPiece attr_name, value_type&& value) { \
    value_field.push_back(std::make_pair(attr_name, value));                 \
    return *this;                                                            \
  }

DEFINE_SET_ATTR(StringPiece, string_attrs_);
DEFINE_SET_ATTR(float, float_attrs_);
DEFINE_SET_ATTR(int, int_attrs_);
DEFINE_SET_ATTR(bool, bool_attrs_);
DEFINE_SET_ATTR(tensorflow::DataType, type_attrs_);

#undef DEFINE_SET_ATTR

AttrBuilder& AttrBuilder::NumInputs(int n) {
  DCHECK(!node_def_finalized_) << "Calling NumInputs after BuildNodeDef.";
  num_inputs_ = n;
  return *this;
}

void AttrBuilder::FillAttrValueMap(AttrValueMap* m,
                                   bool include_those_in_node_def) const {
  for (const auto& p : string_attrs_) {
    SetInAttrValueMap(m, p.first, p.second);
  }
  for (const auto& p : int_attrs_) {
    SetInAttrValueMap(m, p.first, p.second);
  }
  for (const auto& p : float_attrs_) {
    SetInAttrValueMap(m, p.first, p.second);
  }
  for (const auto& p : bool_attrs_) {
    SetInAttrValueMap(m, p.first, p.second);
  }
  for (const auto& p : type_attrs_) {
    SetInAttrValueMap(m, p.first, p.second);
  }
  if (include_those_in_node_def && node_def_ != nullptr) {
    for (AttrValueMap::const_iterator it = node_def_->attr().begin();
         it != node_def_->attr().end(); ++it) {
      m->insert(*it);
    }
  }
}

const NodeDef& AttrBuilder::BuildNodeDef() {
  if (node_def_finalized_) return *node_def_;
  MayBeInitializeNodeDef();
  for (int i = 0; i < num_inputs_; ++i) {
    node_def_->add_input("dummy_input");
  }
  FillAttrValueMap(node_def_->mutable_attr(), false);
  node_def_finalized_ = true;
  return *node_def_;
}

Status AttrTypeByName(const AttrTypeMap& m, const string& attr_name,
                      TF_AttrType* out, unsigned char* is_list) {
  auto* t = gtl::FindOrNull(m, attr_name);
  if (t == nullptr) {
    return errors::InvalidArgument("Attribute '", attr_name,
                                   "' does not exist for this operation");
  }
  *out = static_cast<TF_AttrType>(*t & ~kIsList);
  if (*t & kIsList) {
    *is_list = 1;
  } else {
    *is_list = 0;
  }
  return Status::OK();
}

namespace {
inline tensorflow::Fprint128 FingerprintCat128(const tensorflow::Fprint128& a,
                                               const tensorflow::Fprint128& b) {
  return {tensorflow::FingerprintCat64(a.low64, b.low64),
          tensorflow::FingerprintCat64(a.low64, b.low64)};
}

void CombineUnordered(const tensorflow::Fprint128& a,
                      tensorflow::Fprint128* b) {
  b->low64 += a.low64;
  b->high64 += a.high64;
}

inline tensorflow::Fprint128 CacheKeyHelper(StringPiece s,
                                            const tensorflow::Fprint128& b) {
  tensorflow::Fprint128 a = tensorflow::Fingerprint128(s);
  return FingerprintCat128(a, b);
}

inline tensorflow::Fprint128 CacheKeyHelper(StringPiece s, uint64 b) {
  return CacheKeyHelper(s, {b, b});
}

}  // namespace

tensorflow::Fprint128 AttrBuilder::CacheKey(const string& device) const {
  tensorflow::Fprint128 f = tensorflow::Fingerprint128(op_name_);
  f = tensorflow::FingerprintCat128(f, tensorflow::Fingerprint128(device));
  if (node_def_ != nullptr) {
    // Some attributes are directly written to node_def_ instead of being
    // stored explicitly.
    string value;
    for (const auto& attr : node_def_->attr()) {
      attr.second.SerializeToString(&value);
      CombineUnordered(
          CacheKeyHelper(attr.first, tensorflow::Fingerprint128(value)), &f);
    }
    // Note that node_def_ may be created but not finalized. This can happen
    // when the creation was triggered by a call to Set, but BuildNodeDef has
    // not been called.
    if (node_def_finalized_) return f;
  }
  for (const auto& p : string_attrs_) {
    CombineUnordered(
        CacheKeyHelper(p.first, tensorflow::Fingerprint128(p.second)), &f);
  }
  for (const auto& p : int_attrs_) {
    CombineUnordered(CacheKeyHelper(p.first, static_cast<uint64>(p.second)),
                     &f);
  }
  static std::hash<float> float_hasher;
  for (const auto& p : float_attrs_) {
    CombineUnordered(
        CacheKeyHelper(p.first, static_cast<uint64>(float_hasher(p.second))),
        &f);
  }
  for (const auto& p : bool_attrs_) {
    CombineUnordered(CacheKeyHelper(p.first, p.second ? 1u : 0u), &f);
  }
  for (const auto& p : type_attrs_) {
    CombineUnordered(CacheKeyHelper(p.first, static_cast<uint64>(p.second)),
                     &f);
  }
  return f;
}

void AttrBuilder::MayBeInitializeNodeDef() {
  if (node_def_ == nullptr) {
    node_def_.reset(new NodeDef());
    node_def_->set_name(op_name_);
    node_def_->set_op(op_name_);
  }
}

}  // namespace tensorflow
