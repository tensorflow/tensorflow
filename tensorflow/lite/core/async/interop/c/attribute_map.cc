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
#include "tensorflow/lite/core/async/interop/c/attribute_map.h"

#include <cstddef>
#include <cstdint>

#include "tensorflow/lite/core/async/interop/attribute_map_internal.h"
#include "tensorflow/lite/core/async/interop/c/types.h"

extern "C" {

TfLiteAttributeMap* TfLiteAttributeMapCreate(int32_t type) {
  return new TfLiteAttributeMap(static_cast<TfLiteAttrMapType>(type));
}

void TfLiteAttributeMapDelete(TfLiteAttributeMap* attrs) { delete attrs; }

bool TfLiteAttributeMapIsBufferAttributeMap(const TfLiteAttributeMap* attrs) {
  if (attrs) return attrs->impl.IsBufferAttributeMap();
  return false;
}

bool TfLiteAttributeMapIsSyncAttributeMap(const TfLiteAttributeMap* attrs) {
  if (attrs) return attrs->impl.IsSyncAttributeMap();
  return false;
}

void TfLiteAttributeMapCopy(const TfLiteAttributeMap* src,
                            TfLiteAttributeMap* dst) {
  if (src && dst) {
    dst->impl = src->impl;
  }
}

#define DEFINE_ATTR_MAP_ACCESSOR(type, type_name)                              \
  bool TfLiteAttributeMapGet##type_name##Attr(const TfLiteAttributeMap* attrs, \
                                              uint32_t key, type* val) {       \
    return attrs ? attrs->impl.GetAttr(static_cast<TfLiteBufferAttrKey>(key),  \
                                       val)                                    \
                 : false;                                                      \
  }                                                                            \
  void TfLiteAttributeMapSet##type_name##Attr(TfLiteAttributeMap* attrs,       \
                                              uint32_t key, type val) {        \
    if (attrs) {                                                               \
      attrs->impl.SetAttr(static_cast<TfLiteBufferAttrKey>(key), val);         \
    }                                                                          \
  }                                                                            \
  bool TfLiteAttributeMapGetCustom##type_name##Attr(                           \
      const TfLiteAttributeMap* attrs, const char* key, type* val) {           \
    return attrs ? attrs->impl.GetCustomAttr(key, val) : false;                \
  }                                                                            \
  void TfLiteAttributeMapSetCustom##type_name##Attr(                           \
      TfLiteAttributeMap* attrs, const char* key, type val) {                  \
    if (attrs) {                                                               \
      attrs->impl.SetCustomAttr(key, val);                                     \
    }                                                                          \
  }

DEFINE_ATTR_MAP_ACCESSOR(int, Int);
DEFINE_ATTR_MAP_ACCESSOR(size_t, SizeT);
DEFINE_ATTR_MAP_ACCESSOR(const char*, String);

#undef DEFINE_ATTR_MAP_ACCESSOR

}  // extern "C"
