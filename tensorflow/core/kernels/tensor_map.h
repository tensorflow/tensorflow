/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_KERNELS_TENSOR_MAP_H_
#define TENSORFLOW_CORE_KERNELS_TENSOR_MAP_H_

#include <utility>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_key.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/lib/core/refcount.h"

namespace tensorflow {

// Variant compatible type for a map of tensors. This is mutable but instances
// should never be mutated after stored in a variant tensor.
//
// **NOTE**: TensorMap stores a refcounted container of tf::Tensor objects,
// which are accessible via TensorMap::tensors().  Because it is refcounted,
// straight copies of the form:
//
//    TensorMap b = a;
//    b.tensors().insert(k,v);  // WARNING: This modifies a.tensors().
//
// Do not create a true copy of the underlying container - but instead increment
// a reference count.  Modifying b.tensors() modifies a.tensors().  In this way,
// TensorMap should be considered similar to the tf::Tensor object.
//
// In order to get a copy of the underlying map, use the Copy method:
//
//    TensorMap b = a.Copy();
//    b.tensors().insert(k, v);  // This does not modify a.tensors().
//
// Note that this is not a deep copy: the memory locations of the underlying
// tensors will still point to the same locations of the corresponding tensors
// in the original.  To truly perform a deep copy, Device and Type-specific
// code needs to be applied to the underlying tensors as usual.
//
// The most important implication of RefCounted TensorMaps is that OpKernels
// wishing to reuse TensorMap inputs as outputs via context->forward_input()
// need to perform an additional check on the refcount of the TensorList,
// to ensure aliasing can be performed safely.  For example:
//
//     bool can_alias = false;
//     auto fw = c->forward_input(..., DT_VARIANT, {}, ...);
//     if (fw && fw->dtype() == DT_VARIANT && fw->NumElements() == 1) {
//       auto* tl = fw->scalar<Variant>()().get<TensorMap>();
//       if (tl && tl->RefCountIsOne()) {
//         can_alias = true;
//       }
//     }
//
class TensorMap {
 public:
  TensorMap() : tensors_(new Tensors) {}
  ~TensorMap();

  TensorMap(const TensorMap& other) : tensors_(other.tensors_) {
    tensors_->Ref();
  }

  TensorMap(TensorMap&& rhs) : tensors_(rhs.tensors_) {
    rhs.tensors_ = nullptr;
  }

  TensorMap& operator=(const TensorMap& rhs) {
    if (this == &rhs) return *this;
    tensors_->Unref();
    tensors_ = rhs.tensors_;
    tensors_->Ref();
    return *this;
  }

  TensorMap& operator=(TensorMap&& rhs) {
    if (this == &rhs) return *this;
    std::swap(tensors_, rhs.tensors_);
    return *this;
  }

  static const char kTypeName[];

  string TypeName() const { return kTypeName; }

  void Encode(VariantTensorData* data) const;

  bool Decode(const VariantTensorData& data);

  // TODO(apassos) fill this out
  string DebugString() const { return "TensorMap"; }

  // Access to the underlying tensor container.
  absl::flat_hash_map<TensorKey, Tensor>& tensors() {
    return tensors_->values_;
  }

  const absl::flat_hash_map<TensorKey, Tensor>& tensors() const {
    return tensors_->values_;
  }

  // Get a new TensorMap containing a copy of the underlying tensor container.
  TensorMap Copy() const {
    TensorMap out;
    // This performs a copy of the absl::hashmap.
    out.tensors_->values_ = tensors_->values_;
    return out;
  }

  // Insert key and value if the key does not already exist.
  // Returns true if the insertion happens.
  bool insert(const TensorKey& key, const Tensor& value) {
    auto r = tensors_->values_.try_emplace(key, value);
    return r.second;
  }

  // Lookup given key. Returns iterator to found key or end.
  absl::flat_hash_map<TensorKey, Tensor>::iterator find(TensorKey key) {
    return tensors_->values_.find(key);
  }

  Tensor& lookup(TensorKey key) { return tensors_->values_.find(key)->second; }

  Tensor& operator[](TensorKey& k) { return tensors_->values_[k]; }

  bool replace(const TensorKey& k, const Tensor& v) {
    tensors_->values_[k] = v;
    return true;
  }

  // Removes element with given key. Return size of removed element.
  size_t erase(TensorKey key) { return tensors_->values_.erase(key); }

  // Size returns the number of elements in the map
  size_t size() const { return tensors_->values_.size(); }

  std::vector<Tensor> keys() const {
    std::vector<Tensor> keys;
    keys.reserve(tensors_->values_.size());
    absl::flat_hash_map<TensorKey, Tensor>::iterator it =
        tensors_->values_.begin();
    while (it != tensors_->values_.end()) {
      keys.push_back(it->first);
      it++;
    }
    return keys;
  }

  // Is this TensorMap the only one with a reference to the underlying
  // container?
  bool RefCountIsOne() const { return tensors_->RefCountIsOne(); }

 private:
  class Tensors : public core::RefCounted {
   public:
    absl::flat_hash_map<TensorKey, Tensor> values_;
  };
  Tensors* tensors_;
};

#if defined(PLATFORM_GOOGLE)
// TODO(ebrevdo): Identify why Variant inline size is smaller on mobile devices.
// For 32-bit devices, it's acceptable not to inline.
static_assert(Variant::CanInlineType<TensorMap>() || sizeof(void*) < 8,
              "Must be able to inline TensorMap into a Variant");
#endif
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_TENSOR_MAP_H_
