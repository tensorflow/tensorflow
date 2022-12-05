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
/// `type` argument determines what's the attribute map is describing
/// (e.g. buffer, or sync object).
/// Returned object is owned by the caller.
TfLiteAttributeMap* TfLiteAttributeMapCreate(int32_t type);

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
/// For getters, returns false if the key is not set. TfLite does not check
/// the type of values, callers needs to make sure the requested type matches
/// the value set in the map.
/// For setters, if the value type is a pointer (e.g. c string literals),
/// caller needs to ensure the lifetime of value exceeds the attribute map.
/// If the key is set in previous calls, old value will be overriden by
/// successive setter calls.
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

#undef DECLARE_ATTR_MAP_ACCESSOR

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_CORE_ASYNC_INTEROP_C_ATTRIBUTE_MAP_H_
