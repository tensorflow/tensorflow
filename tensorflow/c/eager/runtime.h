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

#ifndef TENSORFLOW_C_EAGER_RUNTIME_H_
#define TENSORFLOW_C_EAGER_RUNTIME_H_

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
Status AttrTypeMapForOp(const char* op_name, const AttrTypeMap** out);

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
// uint64 cache_key = a.CacheKey("cpu:0");
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
    SetInAttrValueMap(node_def_->mutable_attr(), attr_name, value);
    return *this;
  }

  tensorflow::Fprint128 CacheKey(const string& device) const;

  void FillAttrValueMap(AttrValueMap* m) const { FillAttrValueMap(m, true); }
  const NodeDef& BuildNodeDef();

 private:
  template <class T>
  using AttrVec = tensorflow::gtl::InlinedVector<std::pair<StringPiece, T>, 2>;

  void MayBeInitializeNodeDef();
  void FillAttrValueMap(AttrValueMap* m, bool include_those_in_node_def) const;

  template <class T>
  void SetInAttrValueMap(AttrValueMap* m, StringPiece attr_name,
                         T&& value) const {
    DCHECK(!node_def_finalized_)
        << "Calling SetInAttrValueMap after BuildNodeDef.";
    // Copied from NodeDefBuilder::Attr
    const AttrValue* found = AttrSlice(m).Find(attr_name);
    AttrValue attr_value;
    if (found == nullptr) {
      SetAttrValue(value, &attr_value);
      m->insert(AttrValueMap::value_type(attr_name.ToString(), attr_value));
    } else {
      // TODO(ashankar): Do what is done in
      // NodeDefBuilder::CheckInconsistency(attr_name, *found, attr_value);
      SetAttrValue(std::forward<T>(value), &attr_value);
      (*m)[attr_name.ToString()] = attr_value;
    }
  }

  AttrVec<StringPiece> string_attrs_;
  AttrVec<int> int_attrs_;
  AttrVec<float> float_attrs_;
  AttrVec<bool> bool_attrs_;
  AttrVec<tensorflow::DataType> type_attrs_;
  const string op_name_;
  int num_inputs_;
  std::unique_ptr<NodeDef> node_def_;
  bool node_def_finalized_;
};  // namespace tensorflow

template <>
AttrBuilder& AttrBuilder::Set(StringPiece attr_name, StringPiece&& value);
template <>
AttrBuilder& AttrBuilder::Set(StringPiece attr_name, int&& value);
template <>
AttrBuilder& AttrBuilder::Set(StringPiece attr_name, float&& value);
template <>
AttrBuilder& AttrBuilder::Set(StringPiece attr_name, bool&& value);
template <>
AttrBuilder& AttrBuilder::Set(StringPiece attr_name,
                              tensorflow::DataType&& value);

// KernelAndDevice encapsulates an instantiated kernel and the device it is on.
//
// Also see:
// https://www.tensorflow.org/code/tensorflow/core/common_runtime/kernel_benchmark_testlib.h
// and
// https://www.tensorflow.org/code/tensorflow/core/kernels/ops_testutil.h
class KernelAndDevice {
 public:
  // Populates 'out' with a kernel appropriate for 'ndef'.
  //
  // The provided FunctionLibraryRuntime MUST outlive all calls to
  // Run() on the returned KernelAndDevice.
  //
  // TODO(ashankar): Figure out thread-safety concerns around
  // FunctionLibraryRuntime (in particular, how the underlying
  // FunctionLibraryDefinition might be mutated by another thread as new
  // functions are registered with it).  Conservatively, thread-safe usage of
  // the FunctionLibraryRuntime is pushed on to the caller (see locking in
  // c_api.cc).
  static Status Init(const NodeDef& ndef, FunctionLibraryRuntime* flib,
                     KernelAndDevice* out);
  // TODO(ashankar): Remove this
  static Status InitOp(Device* device, const NodeDef& ndef,
                       KernelAndDevice* out);

  KernelAndDevice(tensorflow::Rendezvous* rendez)
      : device_(nullptr), flib_(nullptr), rendez_(rendez) {}

  // TODO(ashankar): Handle list-valued inputs.
  Status Run(std::vector<Tensor>* inputs, std::vector<Tensor>* outputs,
             NodeExecStats* stats);

  const OpKernel* kernel() const { return kernel_.get(); }

  DataTypeVector* output_dtypes() { return &output_dtypes_; }

 private:
  std::unique_ptr<OpKernel> kernel_;
  Device* device_;
  FunctionLibraryRuntime* flib_;
  checkpoint::TensorSliceReaderCacheWrapper slice_reader_cache_;
  Rendezvous* rendez_;
  DataTypeVector output_dtypes_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_C_EAGER_RUNTIME_H_
