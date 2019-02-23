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

#include "tensorflow/c/c_api.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
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
Status OpDefForOp(const char* op_name, const OpDef** op_def);

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
  explicit AttrBuilder(const char* op)
      : op_name_(op),
        num_inputs_(0),
        node_def_(nullptr),
        node_def_finalized_(false) {}

  // Needed to work around call to ValidateNodeDef in CreateOpKernel.
  AttrBuilder& NumInputs(int n);

  template <class T>
  AttrBuilder& Set(StringPiece attr_name, T&& value) {
    MayBeInitializeNodeDef();
    SetInAttrValueMap(node_def_->mutable_attr(), string(attr_name), value);
    cached_cache_key_ = absl::nullopt;
    return *this;
  }

  tensorflow::Fprint128 CacheKey(const string& device);

  void FillAttrValueMap(AttrValueMap* m) const { FillAttrValueMap(m, true); }
  const NodeDef& BuildNodeDef();

 private:
  template <class T>
  using AttrVec = tensorflow::gtl::InlinedVector<std::pair<string, T>, 2>;

  tensorflow::Fprint128 BuildCacheKeyForDevice(const string& device) const;

  void MayBeInitializeNodeDef();
  // Fill `m` with the attr-value pairs set via AttrBuilder::Set() so far, as
  // well as any default attr-value pairs from the associated op_def, if there
  // is one.
  //
  // If `include_those_in_node_def` is true, also include any attr-value pairs
  // from `node_def_`.
  void FillAttrValueMap(AttrValueMap* m, bool include_those_in_node_def) const;

  template <class T>
  void SetInAttrValueMap(AttrValueMap* m, const string& attr_name,
                         T&& value) const {
    DCHECK(!node_def_finalized_)
        << "Calling SetInAttrValueMap after BuildNodeDef.";
    // Copied from NodeDefBuilder::Attr
    const AttrValue* found = AttrSlice(m).Find(attr_name);
    AttrValue attr_value;
    if (found == nullptr) {
      SetAttrValue(value, &attr_value);
      m->insert(AttrValueMap::value_type(attr_name, attr_value));
    } else {
      // TODO(ashankar): Do what is done in
      // NodeDefBuilder::CheckInconsistency(attr_name, *found, attr_value);
      SetAttrValue(std::forward<T>(value), &attr_value);
      (*m)[attr_name] = attr_value;
    }
  }

  AttrVec<int> int_attrs_;
  AttrVec<float> float_attrs_;
  AttrVec<bool> bool_attrs_;
  AttrVec<tensorflow::DataType> type_attrs_;
  const string op_name_;
  int num_inputs_;
  std::unique_ptr<NodeDef> node_def_;
  bool node_def_finalized_;

  absl::optional<tensorflow::Fprint128> cached_cache_key_;
  string device_for_cached_cache_key_;
};  // namespace tensorflow

template <>
AttrBuilder& AttrBuilder::Set(StringPiece attr_name, int&& value);
template <>
AttrBuilder& AttrBuilder::Set(StringPiece attr_name, float&& value);
template <>
AttrBuilder& AttrBuilder::Set(StringPiece attr_name, bool&& value);
template <>
AttrBuilder& AttrBuilder::Set(StringPiece attr_name,
                              tensorflow::DataType&& value);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_ATTR_BUILDER_H_
