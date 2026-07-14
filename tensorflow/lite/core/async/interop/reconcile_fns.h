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
#ifndef TENSORFLOW_LITE_CORE_ASYNC_INTEROP_RECONCILE_FNS_H_
#define TENSORFLOW_LITE_CORE_ASYNC_INTEROP_RECONCILE_FNS_H_

// Reconciliation functions for merging and examinate buffer / synchronization
// attributes.

#include "tensorflow/lite/core/async/interop/attribute_map_internal.h"
#include "tensorflow/lite/core/async/interop/c/types.h"

namespace tflite {
namespace interop {

// Reconciles general attributes.
// `lhs`, `rhs`, `merged` are required to be not null, otherwise return false.
// The merged attribute will be set in `merged`. If there's any attribute that
// can not be reconciled, it will be set in `conflict` and return false.
bool ReconcileGeneralAttributeKeys(TfLiteAttrMapType type,
                                   const AttributeMap::ContainerT* lhs,
                                   const AttributeMap::ContainerT* rhs,
                                   AttributeMap::ContainerT* merged,
                                   AttributeMap::ContainerT* conflict);

// Check if `lhs` covers all attribute in `rhs`.
// `lhs` and `rhs` are required to be not null, otherwise return false.
// If there's any attribute that is not covered (i.e. missing from `lhs` or
// values are incompatible), it will be set in `conflict` and return false.
bool CheckGeneralAttributeKeysCoverage(TfLiteAttrMapType type,
                                       const AttributeMap::ContainerT* lhs,
                                       const AttributeMap::ContainerT* rhs,
                                       AttributeMap::ContainerT* conflict);

}  // namespace interop
}  // namespace tflite

#endif  // TENSORFLOW_LITE_CORE_ASYNC_INTEROP_RECONCILE_FNS_H_
