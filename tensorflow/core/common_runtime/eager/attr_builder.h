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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_ATTR_BUILDER_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_ATTR_BUILDER_H_

// Support for eager execution of TensorFlow kernels.

#include <memory>
#include <unordered_map>

#include "tensorflow/c/tf_attrtype.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/gtl/optional.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/util/tensor_slice_reader_cache.h"

namespace tensorflow {

// Maps attribute name to an encoding of the type of the attribute value.
// If the type is not a list type, the value is the same as the TF_AttrType type
// of the value. Else, the highest order bit is on, and the rest of the bits
// represent the TF_AttrType type of the values in the list.
typedef std::unordered_map<string, uint32> AttrTypeMap;

// Look up OpDef for `op_name`.
Status OpDefForOp(const string& op_name, const OpDef** op_def);

// Returns the AttrTypeMap for the TensorFlow operation named op_name.
// If op_name is not registered in global op registry, AttrTypeMapForOp assumes
// the op to be a function and returns the default attributes for a function.
// `is_function` is set to true in this case.
Status AttrTypeMapForOp(const char* op_name, const AttrTypeMap** out,
                        bool* is_function);

// Looks for 'attr_name' in 'm' and sets 'out' and 'is_list'.
Status AttrTypeByName(const AttrTypeMap& m, const string& attr_name,
                      TF_AttrType* out, unsigned char* is_list);

// KernelAndDevice::Init needs a NodeDef only to pass the attribute map through.
// An AttrBuilder is a convenience class to help with that - providing a smaller
// interface than NodeDefBuilder and avoiding expensive (unnecessary?) sanity
// checks (like number of inputs matching the OpDef - we only care about
// attributes here).
//
// TODO(ashankar): Take a closer look at checks in NodeDefBuilder and see which
// ones make sense to replicate.

// This is a helper class for creating a NodeDef. Additionally, this class
// allows computing a cache key based on fingerprinting the attributes of this
// NodeDef.
//
// Example usage:
// AttrBuilder a;
// a.NumInputs(2);
// a.Set("T", TF_FLOAT);
// tensorflow::Fprint128 cache_key = a.CacheKey("cpu:0");
// const NodeDef& n = a.BuildNodeDef();
//
// Note that all calls to Set and NumInputs should happen before calling
// BuildNodeDef. Also, calls to NumInputs or Set between multiple invocations
// to CacheKey may cause different values to be returned by CacheKey.
//
// For performance reasons, the class internally delays the actual construction
// of the NodeDef till BuildNodeDef is called, or Set is called with certain
// uncommon types (see template specializations of Set to see which types
// trigger a NodeDef creation).
class AttrBuilder {
 public:
  AttrBuilder() {}
  explicit AttrBuilder(const char* op) { Reset(op); }

  void Reset(const char* op) {
    op_name_ = op;
    num_inputs_ = 0;
    encoded_attrs_.clear();
    node_def_initialized_ = false;
    node_def_finalized_ = false;
    cached_cache_key_ = absl::nullopt;
    device_for_cached_cache_key_.clear();
  }

  const string& op_name() const { return op_name_; }

  // Needed to work around call to ValidateNodeDef in CreateOpKernel.
  AttrBuilder& NumInputs(int n);

  template <class T>
  AttrBuilder& Set(StringPiece attr_name, T&& value) {
    SetAttrValue(value, &attr_tmp_);
    AddAttrIfNotPresent(attr_name, attr_tmp_);
    cached_cache_key_ = absl::nullopt;
    return *this;
  }

  size_t NumAttributes() const { return encoded_attrs_.size(); }

  AttrBuilder& Set(StringPiece attr_name, const AttrValue& value) {
    AddAttrIfNotPresent(attr_name, value);
    cached_cache_key_ = absl::nullopt;
    return *this;
  }

  // Retrieves the attribute value.
  // Note that Get() can involve a linear scan of all attributes with the same
  // value type in this Node. This is not an issue, because Get is used rarely
  // and nodes have a small number of attributes.
  template <class T>
  Status Get(StringPiece attr_name, T* value) const {
    // Common attributes are stored in AttrVecs. This Get() template
    // is specialized for them below. If we end up here, the type must be
    // among those that we store in the node_def_.
    if (!node_def_initialized_) {
      return errors::NotFound("No attr named'", attr_name,
                              "' found in AttrBuilder for ", op_name_);
    }
    return GetNodeAttr(AttrSlice(node_def_), attr_name, value);
  }

  tensorflow::Fprint128 CacheKey(const StringPiece device);

  // Fill `m` with the attr-value pairs set via AttrBuilder::Set() so far, as
  // well as any default attr-value pairs from the associated op_def, if there
  // is one.
  void FillAttrValueMap(AttrValueMap* m) const;

  // Fill `m` with the attr-value pairs set via AttrBuilder::Set() so far except
  // when the value matches the default for this attr.
  // More precisely, if the global op registry contains an OpDef for this op
  // and if an attribute value is the same as the default (according to the
  // OpDef), this attr-value pair is not added to `m`.
  void FillAttrValueMapWithoutDefaults(AttrValueMap* m) const;
  const NodeDef& BuildNodeDef();

 private:
  tensorflow::Fprint128 BuildCacheKeyForDevice(const StringPiece device) const;

  // Initialize the node_def_ object.
  // REQUIRES: node_def_initialized_ = false
  void InitializeNodeDef();

  template <class T>
  void SetInAttrValueMap(AttrValueMap* m, const string& attr_name,
                         T&& value) const {
    DCHECK(!node_def_finalized_)
        << "Calling SetInAttrValueMap after BuildNodeDef.";
    // If attribute is set more than once, its first value prevails
    m->insert({attr_name, value});
  }

  void AddAttrIfNotPresent(StringPiece attr_name, const AttrValue& value);

  gtl::FlatMap<string, string> encoded_attrs_;
  mutable AttrValue attr_tmp_;  // For encoding

  string op_name_;  // Conceptually const, but can't be because of Reset(...)
  int num_inputs_;
  NodeDef node_def_;
  bool node_def_initialized_;
  bool node_def_finalized_;

  absl::optional<tensorflow::Fprint128> cached_cache_key_;
  string device_for_cached_cache_key_;
};

template <>
Status AttrBuilder::Get(StringPiece attr_name, int* value) const;
template <>
Status AttrBuilder::Get(StringPiece attr_name, float* value) const;
template <>
Status AttrBuilder::Get(StringPiece attr_name, bool* value) const;
template <>
Status AttrBuilder::Get(StringPiece attr_name,
                        tensorflow::DataType* value) const;

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_ATTR_BUILDER_H_
