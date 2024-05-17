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
#ifndef TENSORFLOW_LITE_CORE_ASYNC_INTEROP_C_ATTRIBUTE_MAP_H_
#define TENSORFLOW_LITE_CORE_ASYNC_INTEROP_C_ATTRIBUTE_MAP_H_

#include <stddef.h>
#include <stdint.h>

#include "tensorflow/lite/core/async/interop/c/types.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// --------------------------------------------------------------------------
/// TfLiteAttributeMap API.
///
/// TfLiteAttributeMap stores buffer or sync attributes and keeps those
/// intelligible across different backends and applications.
/// Backend delegates can define a set of attribute keys to describe what's
/// the attribute and defines the type of the value.
/// Different components (application, TfLite runtime, backends) can use
/// TfLiteAttributeMap to negotiate the requirements of the buffer / sync
/// and establish the contract on specifications of a particular input / output.
/// WARNING: This is an experimental type and subject to change.

/// Opaque type for TfLiteAttributeMap.
typedef struct TfLiteAttributeMap TfLiteAttributeMap;

/// Creates an attribute map.
/// `type` argument determines what the attribute map is describing
/// (e.g. buffer, or sync object).
/// Returned object is owned by the caller.
TfLiteAttributeMap* TfLiteAttributeMapCreate(TfLiteAttrMapType type);

/// Destroys the attribute map.
/// Do nothing if `attrs` is nullptr.
void TfLiteAttributeMapDelete(TfLiteAttributeMap* attrs);

/// Returns true if `attrs` is a buffer attribute map.
/// If `attrs` is nullptr, returns false.
bool TfLiteAttributeMapIsBufferAttributeMap(const TfLiteAttributeMap* attrs);

/// Returns true if `attrs` is a sync object attribute map.
/// If `attrs` is nullptr, returns false.
bool TfLiteAttributeMapIsSyncAttributeMap(const TfLiteAttributeMap* attrs);

/// Copies all attributes from `src` to `dst`. Any existing attributes in `dst`
/// will be cleared.
/// If `src` or `dst` is null, does nothing.
void TfLiteAttributeMapCopy(const TfLiteAttributeMap* src,
                            TfLiteAttributeMap* dst);

// --------------------------------------------------------------------------
/// Accessor methods.
///
/// For getters, returns false if the key is not set, or the requested type
/// does not match.
/// For setters, if the value type is a pointer (e.g. c string literals),
/// caller needs to ensure the lifetime of value exceeds the attribute map.
/// If the key is set in previous calls, old value will be overriden by
/// successive setter calls.

/// Gets the int buffer attribute value for the given `key`.
/// Returns false if the key is not set, `attrs` is not a buffer attribute map,
/// or the value is not of type `size_t`.
bool TfLiteAttributeMapGetSizeTBufferAttr(const TfLiteAttributeMap* attrs,
                                          TfLiteBufferAttrKey key, size_t* val);

/// Sets the `key` buffer attribute as `val`.
/// Returns false if `attrs` is not a buffer attribute map.
bool TfLiteAttributeMapSetSizeTBufferAttr(TfLiteAttributeMap* attrs,
                                          TfLiteBufferAttrKey key, size_t val);

/// Gets the C string buffer attribute value for the given `key`.
/// Returns false if the key is not set, `attrs` is not a buffer attribute map,
/// or the value is not of type `size_t`.
/// Returned C string's lifespan is determined by the setter of that value.
/// Neither `attrs` nor the caller maintains the lifespan of the string.
bool TfLiteAttributeMapGetStringBufferAttr(const TfLiteAttributeMap* attrs,
                                           TfLiteBufferAttrKey key,
                                           const char** val);

/// Sets the `key` buffer attribute as `val`.
/// Returns false if `attrs` is not a buffer attribute map.
/// `attrs` does not own the `val` C string.
bool TfLiteAttributeMapSetStringBufferAttr(TfLiteAttributeMap* attrs,
                                           TfLiteBufferAttrKey key,
                                           const char* val);

/// Gets the bool buffer attribute value for the given `key`.
/// Returns false if the key is not set, `attrs` is not a buffer attribute map,
/// or the value is not of type `bool`.
bool TfLiteAttributeMapGetBoolBufferAttr(const TfLiteAttributeMap* attrs,
                                         TfLiteBufferAttrKey key, bool* val);

/// Sets the `key` buffer attribute as `val`.
/// Returns false if `attrs` is not a sync attribute map.
/// `attrs` does not own the `val` C string.
bool TfLiteAttributeMapSetBoolBufferAttr(TfLiteAttributeMap* attrs,
                                         TfLiteBufferAttrKey key, bool val);

/// Gets the C string synchronization attribute value for the given `key`.
/// Returns false if the key is not set, `attrs` is not a sync attribute map,
/// or the value is not of type `size_t`.
/// Returned C string's lifespan is determined by the setter of that value.
/// Neither `attrs` nor the caller maintains the lifespan of the string.
bool TfLiteAttributeMapGetStringSyncAttr(const TfLiteAttributeMap* attrs,
                                         TfLiteSynchronizationAttrKey key,
                                         const char** val);

/// Sets the `key` buffer attribute as `val`.
/// Returns false if `attrs` is not a sync attribute map.
/// `attrs` does not own the `val` C string.
bool TfLiteAttributeMapSetStringSyncAttr(TfLiteAttributeMap* attrs,
                                         TfLiteSynchronizationAttrKey key,
                                         const char* val);

/// \privatesection
/// Attribute map accessor methods that does not check the map type.
/// It's recommended to use methods above for setting / getting attribute values
/// as those will also check whether the attribute key matches the attribute
/// map type.
#define DECLARE_ATTR_MAP_ACCESSOR(type, type_name)                             \
  bool TfLiteAttributeMapGet##type_name##Attr(const TfLiteAttributeMap* attrs, \
                                              uint32_t key, type* val);        \
  void TfLiteAttributeMapSet##type_name##Attr(TfLiteAttributeMap* attrs,       \
                                              uint32_t key, type val);         \
  bool TfLiteAttributeMapGetCustom##type_name##Attr(                           \
      const TfLiteAttributeMap* attrs, const char* key, type* val);            \
  void TfLiteAttributeMapSetCustom##type_name##Attr(                           \
      TfLiteAttributeMap* attrs, const char* key, type val);

DECLARE_ATTR_MAP_ACCESSOR(int, Int);
DECLARE_ATTR_MAP_ACCESSOR(size_t, SizeT);
DECLARE_ATTR_MAP_ACCESSOR(const char*, String);
DECLARE_ATTR_MAP_ACCESSOR(bool, Bool);

#undef DECLARE_ATTR_MAP_ACCESSOR

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_CORE_ASYNC_INTEROP_C_ATTRIBUTE_MAP_H_
