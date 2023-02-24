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
#include "tensorflow/lite/core/async/interop/reconcile_fns.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <set>

#include "tensorflow/lite/core/async/interop/attribute_keys.h"
#include "tensorflow/lite/core/async/interop/attribute_map_internal.h"

namespace tflite {
namespace interop {
namespace {

// TODO(b/191883048): Check binary size impact with <numeric> header and replace
// with std::lcm if possible.
template <typename T>
T gcd(T x, T y) {
  while (y) {
    auto m = x % y;
    x = y;
    y = m;
  }
  return x;
}

template <typename T>
T lcm(T x, T y) {
  return x / gcd(x, y) * y;
}

// Reconciled alignment is LCM of l and r.
void ReconcileAlignment(size_t l, size_t r, AttributeMap::ContainerT* merged) {
  merged->insert_or_assign(
      static_cast<size_t>(TfLiteBufferAttributeKey::kAlignment), lcm(l, r));
}

// Reconciled padding is LCM of l and r.
void ReconcilePadding(size_t l, size_t r, AttributeMap::ContainerT* merged) {
  merged->insert_or_assign(
      static_cast<size_t>(TfLiteBufferAttributeKey::kPadding), lcm(l, r));
}

// For alignment and padding, if l is multiples of r, it's covering r.
bool CheckMultiples(size_t l, size_t r) { return l % r == 0; }

// Reconciled size is max(l, r).
void ReconcileSize(size_t l, size_t r, AttributeMap::ContainerT* merged) {
  merged->insert_or_assign(static_cast<size_t>(TfLiteBufferAttributeKey::kSize),
                           std::max(l, r));
}

// Checks if l >= r.
bool CheckSize(size_t l, size_t r) { return l >= r; }

}  // namespace

bool ReconcileGeneralAttributeKeys(TfLiteAttrMapType type,
                                   const AttributeMap::ContainerT* lhs,
                                   const AttributeMap::ContainerT* rhs,
                                   AttributeMap::ContainerT* merged,
                                   AttributeMap::ContainerT* conflict) {
  if (lhs == nullptr || rhs == nullptr || merged == nullptr) return false;
  bool ret = true;
  std::set<uint32_t> keys;
  std::transform(lhs->begin(), lhs->end(), std::inserter(keys, keys.end()),
                 [](auto pair) { return pair.first; });
  std::transform(rhs->begin(), rhs->end(), std::inserter(keys, keys.end()),
                 [](auto pair) { return pair.first; });
  for (auto k : keys) {
    const auto l = lhs->find(k);
    const auto r = rhs->find(k);
    if (l == lhs->end()) {
      merged->insert_or_assign(k, r->second);
      continue;
    }
    if (r == rhs->end()) {
      merged->insert_or_assign(k, l->second);
      continue;
    }
    if (type == kTfLiteBufferAttrMap) {
      switch (static_cast<TfLiteBufferAttributeKey>(k)) {
        case TfLiteBufferAttributeKey::kSize:
          ReconcileSize(l->second.Get<size_t>(), r->second.Get<size_t>(),
                        merged);
          break;
        case TfLiteBufferAttributeKey::kAlignment:
          ReconcileAlignment(l->second.Get<size_t>(), r->second.Get<size_t>(),
                             merged);
          break;
        case TfLiteBufferAttributeKey::kPadding:
          ReconcilePadding(l->second.Get<size_t>(), r->second.Get<size_t>(),
                           merged);
          break;
        default:
          // For other keys, check equality.
          if (l->second == r->second) {
            merged->insert_or_assign(k, l->second);
          } else {
            ret = false;
            if (conflict) conflict->insert_or_assign(k, r->second);
          }
      }
    } else {
      // Check equality.
      if (l->second == r->second) {
        merged->insert_or_assign(k, l->second);
      } else {
        ret = false;
        if (conflict) conflict->insert_or_assign(k, r->second);
      }
    }
  }
  return ret;
}

bool CheckGeneralAttributeKeysCoverage(TfLiteAttrMapType type,
                                       const AttributeMap::ContainerT* lhs,
                                       const AttributeMap::ContainerT* rhs,
                                       AttributeMap::ContainerT* conflict) {
  if (lhs == nullptr || rhs == nullptr) return false;
  bool ret = true;
  std::set<uint32_t> keys;
  std::transform(lhs->begin(), lhs->end(), std::inserter(keys, keys.end()),
                 [](auto pair) { return pair.first; });
  std::transform(rhs->begin(), rhs->end(), std::inserter(keys, keys.end()),
                 [](auto pair) { return pair.first; });
  for (auto k : keys) {
    bool has_conflict = false;
    const auto l = lhs->find(k);
    const auto r = rhs->find(k);
    if (r == rhs->end()) {
      continue;
    } else if (l == lhs->end()) {
      has_conflict = true;
    } else {
      if (type == kTfLiteBufferAttrMap) {
        switch (static_cast<TfLiteBufferAttributeKey>(k)) {
          case TfLiteBufferAttributeKey::kSize:
            has_conflict |=
                !CheckSize(l->second.Get<size_t>(), r->second.Get<size_t>());
            break;
          case TfLiteBufferAttributeKey::kAlignment:
            has_conflict |= !CheckMultiples(l->second.Get<size_t>(),
                                            r->second.Get<size_t>());
            break;
          case TfLiteBufferAttributeKey::kPadding:
            has_conflict |=
                !CheckSize(l->second.Get<size_t>(), r->second.Get<size_t>());
            break;
          default:
            // For other keys, check equality.
            if (l->second != r->second) {
              has_conflict = true;
            }
        }
      } else {
        if (l->second != r->second) {
          has_conflict = true;
        }
      }
    }
    if (has_conflict) {
      if (conflict != nullptr) conflict->insert_or_assign(k, r->second);
      ret = false;
    }
  }
  return ret;
}

}  // namespace interop
}  // namespace tflite
