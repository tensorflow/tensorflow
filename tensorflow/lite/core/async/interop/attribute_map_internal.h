/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_CORE_ASYNC_INTEROP_ATTRIBUTE_MAP_INTERNAL_H_
#define TENSORFLOW_LITE_CORE_ASYNC_INTEROP_ATTRIBUTE_MAP_INTERNAL_H_

#include <cstdint>
#include <map>
#include <string>

#include "tensorflow/lite/core/async/interop/attribute_keys.h"
#include "tensorflow/lite/core/async/interop/c/types.h"
#include "tensorflow/lite/core/async/interop/variant.h"

namespace tflite {
namespace interop {

// A value type pruned map, containing the attributes describing the properties
// of a backend buffer or synchronization object.
class AttributeMap {
 public:
  explicit AttributeMap(TfLiteAttrMapType type) : type_(type) {}
  using KeyT = uint32_t;
  using CustomKeyT = std::string;
  // TODO(b/191883048): Benchmark std::variant vs. tagged union.
  using ValueT = tflite::interop::Variant;
  // TODO(b/191883048): Currently the number of attributes is small enough.
  // So it's possible to optimize with a flat map.
  using ContainerT = std::map<KeyT, ValueT>;
  using CustomContainerT = std::map<CustomKeyT, ValueT>;

  bool IsBufferAttributeMap() const { return type_ == kTfLiteBufferAttrMap; }
  bool IsSyncAttributeMap() const { return type_ == kTfLiteSyncAttrMap; }

  // Reconciles and merges the attribute values from other.
  // After reconciliation, the merged value is compatible with both *this and
  // `other`. e.g. a merged buffer size will be the maximum of two operands.
  // If there's any attributes that cannot be reconciled, it will be filled to
  // `conflict` if provided.
  // `other` and `merged` should not be nullptr.
  // Returns true if there's no conflicting attributes.
  bool ReconcileAttributes(const AttributeMap* other, AttributeMap* merged,
                           AttributeMap* conflict) const;

  // Checks if the attributes fully covers requirements.
  // An attribute covers if the values are compatible or it only appears
  // in *this.
  // `other` should not be nullptr otherwise will return false.
  // Returns true if attrs completely covers requirements.
  bool CheckAttributeCoverage(const AttributeMap* other,
                              AttributeMap* conflict) const;

  // Retrieves attribute value by key.
  // Returns true if corresponding attribute exists, otherwise returns false.
  template <typename ValueT>
  bool GetAttr(TfLiteBufferAttributeKey key, ValueT* value) const {
    if (auto it = attrs_.find(static_cast<uint32_t>(key)); it != attrs_.end()) {
      *value = it->second.Get<ValueT>();
      return true;
    }
    return false;
  }

  // Sets attribute value by key.
  template <typename ValueT>
  void SetAttr(TfLiteBufferAttributeKey key, ValueT value) {
    attrs_.insert_or_assign(static_cast<uint32_t>(key), value);
  }

  // Retrieves custom attribute value by key.
  // Returns true if corresponding attribute exists, otherwise returns false.
  template <typename ValueT>
  bool GetCustomAttr(CustomKeyT key, ValueT* value) const {
    if (auto it = custom_attrs_.find(key); it != custom_attrs_.end()) {
      *value = it->second.Get<ValueT>();
      return true;
    }
    return false;
  }

  // Sets custom attribute value by key.
  template <typename ValueT>
  void SetCustomAttr(CustomKeyT key, ValueT value) {
    custom_attrs_.insert_or_assign(key, value);
  }

 private:
  TfLiteAttrMapType type_;
  ContainerT attrs_;
  CustomContainerT custom_attrs_;
};

}  // namespace interop
}  // namespace tflite

struct TfLiteAttributeMap {
  explicit TfLiteAttributeMap(TfLiteAttrMapType type) : impl(type) {}

  tflite::interop::AttributeMap impl;
};

#endif  // TENSORFLOW_LITE_CORE_ASYNC_INTEROP_ATTRIBUTE_MAP_INTERNAL_H_
