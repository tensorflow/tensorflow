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
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/tensor_slice_reader_cache.h"

namespace tensorflow {
namespace {

mutex g_op_name_to_attr_type_map_lock(LINKER_INITIALIZED);

tensorflow::gtl::FlatMap<string, const AttrTypeMap*>* OpNameToAttrTypeMap() {
  static auto* const m =
      new tensorflow::gtl::FlatMap<string, const AttrTypeMap*>;
  return m;
}

const uint32 kIsList = 1U << 31;

AttrTypeMap* DefaultFunctionAttrTypeMap() {
  AttrTypeMap* map = new AttrTypeMap();
  (*map)["executor_type"] = TF_ATTR_STRING;
  (*map)["config_proto"] = TF_ATTR_STRING;
  return map;
}

const AttrTypeMap* GetDefaultFunctionAttrTypeMap() {
  static const AttrTypeMap* map = DefaultFunctionAttrTypeMap();
  return map;
}

}  // namespace

Status OpDefForOp(const string& op_name, const OpDef** op_def) {
  const OpRegistrationData* op_reg_data = nullptr;
  Status s = OpRegistry::Global()->LookUp(op_name, &op_reg_data);
  if (s.ok()) {
    *op_def = &op_reg_data->op_def;
  }
  return s;
}

Status AttrTypeMapForOp(const char* op_name, const AttrTypeMap** out,
                        bool* is_function) {
  {
    tf_shared_lock l(g_op_name_to_attr_type_map_lock);
    *is_function = false;
    *out = gtl::FindPtrOrNull(*OpNameToAttrTypeMap(), op_name);
    if (*out != nullptr) return Status::OK();
  }

  mutex_lock l(g_op_name_to_attr_type_map_lock);

  // Check the existence of AttrTypeMap for op_name again because another thread
  // may insert this map after the tf_shared_lock is released but before the
  // mutex_lock is acquired.
  *out = gtl::FindPtrOrNull(*OpNameToAttrTypeMap(), op_name);
  if (*out != nullptr) return Status::OK();

  const OpDef* op_def = nullptr;
  Status s = OpDefForOp(op_name, &op_def);
  if (errors::IsNotFound(s)) {
    // If we did not find the op def, we assume `op_name` is a function.
    // If it is actually a misspelled op, user will get another error when
    // trying to run it.
    // TODO(iga): If we ever have a use case for different attribute specs
    // in different functions, we will need to look at the OpDef in the
    // function def to retrieve their types.
    *out = GetDefaultFunctionAttrTypeMap();
    *is_function = true;
    return Status::OK();
  } else if (!s.ok()) {
    return s;
  }
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
  auto r = OpNameToAttrTypeMap()->emplace(op_name, m.release());
  DCHECK(r.second) << "AttrTypeMap already exists for " << op_name;

  return Status::OK();
}

#define DEFINE_GET_ATTR(TYPE, FIELD, ATTR_TYPE)                         \
  template <>                                                           \
  Status AttrBuilder::Get(StringPiece attr_name, TYPE* value) const {   \
    auto it = encoded_attrs_.find(string(attr_name));                   \
    if (it == encoded_attrs_.end()) {                                   \
      return errors::NotFound("No attr named'", attr_name,              \
                              "' found in AttrBuilder for ", op_name_); \
    }                                                                   \
    attr_tmp_.ParseFromString(it->second);                              \
    TF_RETURN_IF_ERROR(AttrValueHasType(attr_tmp_, ATTR_TYPE));         \
    *value = attr_tmp_.FIELD();                                         \
    return Status::OK();                                                \
  }

DEFINE_GET_ATTR(float, f, "float");
DEFINE_GET_ATTR(int, i, "int");
DEFINE_GET_ATTR(bool, b, "bool");
DEFINE_GET_ATTR(tensorflow::DataType, type, "type");

#undef DEFINE_GET_ATTR

AttrBuilder& AttrBuilder::NumInputs(int n) {
  DCHECK(!node_def_finalized_) << "Calling NumInputs after BuildNodeDef.";
  num_inputs_ = n;
  return *this;
}

void AttrBuilder::FillAttrValueMap(AttrValueMap* m) const {
  for (auto& entry : encoded_attrs_) {
    attr_tmp_.ParseFromString(entry.second);
    m->insert(AttrValueMap::value_type(entry.first, attr_tmp_));
  }
  // For any attr-value pairs that exist in the op def (from op registry) but
  // not `m`, fill them into `m`, so that we can run a TFE_Op without having to
  // specify all the default attr values (e.g. for matmul, the `transpose_a`
  // attr defaults to false).
  const OpDef* op_def = nullptr;
  Status s = OpDefForOp(op_name().c_str(), &op_def);
  // This is expected, if this op is a custom function, and is therefore not
  // present in the op registry.
  if (!s.ok()) return;

  DCHECK(op_def);
  for (const auto& attr_def : op_def->attr()) {
    if (attr_def.has_default_value() && !m->count(attr_def.name())) {
      SetInAttrValueMap(m, attr_def.name(), attr_def.default_value());
    }
  }
}

namespace {

bool ValueMatchesDefault(const OpDef* op_def, const string& attr_name,
                         const AttrValue& attr_value) {
  // TODO(iga): It might make sense to augment OpRegistrationData with a
  // {attr_name -> default_attr_value} FlatMap to avoid the loop here.
  for (const OpDef::AttrDef& attr_def : op_def->attr()) {
    if (attr_def.name() == attr_name && attr_def.has_default_value() &&
        AreAttrValuesEqual(attr_def.default_value(), attr_value)) {
      return true;
    }
  }
  return false;
}

}  // namespace

void AttrBuilder::FillAttrValueMapWithoutDefaults(AttrValueMap* m) const {
  const OpDef* op_def = nullptr;
  Status s = OpDefForOp(op_name().c_str(), &op_def);

  for (auto& entry : encoded_attrs_) {
    attr_tmp_.ParseFromString(entry.second);
    // Insert the attr-value pair if we did not find the OpDef or if the value
    // is different from default.
    if (!s.ok() || !ValueMatchesDefault(op_def, entry.first, attr_tmp_)) {
      m->insert(AttrValueMap::value_type(entry.first, attr_tmp_));
    }
  }
}

void AttrBuilder::AddAttrIfNotPresent(StringPiece attr_name,
                                      const AttrValue& value) {
  encoded_attrs_.emplace(string(attr_name), value.SerializeAsString());
}

const NodeDef& AttrBuilder::BuildNodeDef() {
  if (node_def_finalized_) return node_def_;
  if (!node_def_initialized_) {
    InitializeNodeDef();
  }
  for (int i = 0; i < num_inputs_; ++i) {
    node_def_.add_input("dummy_input");
  }
  FillAttrValueMap(node_def_.mutable_attr());
  node_def_finalized_ = true;
  return node_def_;
}

void AttrBuilder::CopyAttributes(const AttrBuilder& other) {
  encoded_attrs_.insert(other.encoded_attrs_.begin(),
                        other.encoded_attrs_.end());
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
          tensorflow::FingerprintCat64(a.high64, b.high64)};
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

tensorflow::Fprint128 AttrBuilder::CacheKey(const StringPiece device) {
  if (!cached_cache_key_ || device != device_for_cached_cache_key_) {
    cached_cache_key_ = BuildCacheKeyForDevice(device);
    device_for_cached_cache_key_ = string(device);
  }

  return *cached_cache_key_;
}

tensorflow::Fprint128 AttrBuilder::BuildCacheKeyForDevice(
    const StringPiece device) const {
  tensorflow::Fprint128 f = tensorflow::Fingerprint128(op_name());
  f = tensorflow::FingerprintCat128(f, tensorflow::Fingerprint128(device));
  for (const auto& p : encoded_attrs_) {
    CombineUnordered(
        CacheKeyHelper(p.first, tensorflow::Fingerprint128(p.second)), &f);
  }
  return f;
}

void AttrBuilder::InitializeNodeDef() {
  DCHECK(!node_def_initialized_);
  node_def_.Clear();
  node_def_.set_name(op_name_);
  node_def_.set_op(op_name_);
  node_def_initialized_ = true;
}

void AttrBuilder::GetNameAttrList(
    tensorflow::NameAttrList* name_and_attrs) const {
  FillAttrValueMap(name_and_attrs->mutable_attr());
  name_and_attrs->set_name(op_name());
}

bool AttrBuilder::GetInt(absl::string_view attr_name, int64_t* result) const {
  Status s = Get(attr_name, result);
  return s.ok();
}
bool AttrBuilder::GetFloat(absl::string_view attr_name, float* result) const {
  Status s = Get(attr_name, result);
  return s.ok();
}
bool AttrBuilder::GetBool(absl::string_view attr_name, bool* result) const {
  Status s = Get(attr_name, result);
  return s.ok();
}

bool AttrBuilder::GetType(absl::string_view attr_name,
                          tensorflow::DataType* result) const {
  Status s = Get(attr_name, result);
  return s.ok();
}

}  // namespace tensorflow
