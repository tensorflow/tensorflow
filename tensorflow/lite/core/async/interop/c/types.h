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
#ifndef TENSORFLOW_LITE_CORE_ASYNC_INTEROP_C_TYPES_H_
#define TENSORFLOW_LITE_CORE_ASYNC_INTEROP_C_TYPES_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// --------------------------------------------------------------------------
/// Types for hardware buffer object / synchronization object interoperability.
/// WARNING: This is an experimental type and subject to change.

/// TfLiteBackendBuffer is a an opaque type that abstracts platform specific
/// implementations of hardware buffer objects (e.g. AHardwareBuffer).
/// It's used for carrying the platform-specific hardware buffer object across
/// applications, TFLite runtime and backends.
typedef struct TfLiteBackendBuffer TfLiteBackendBuffer;

/// Creates an empty TfLiteBackendBuffer that does not contain any hardware
/// buffers object.
/// Returned object is owned by the caller.
TfLiteBackendBuffer* TfLiteBackendBufferCreate();

/// Destroys a TfLiteBackendBuffer.
/// Calling this function will not release the buffer object stored underneath.
void TfLiteBackendBufferDelete(TfLiteBackendBuffer* buf);

/// Stores a type puned buffer object to TfLiteBackendBuffer.
/// `buf` will not own or control the lifecycle of `ptr`.
/// Callers needs to ensure lifetime of *ptr exceeds `buf`.
void TfLiteBackendBufferSetPtr(TfLiteBackendBuffer* buf, void* ptr);

/// Retrieves the buffer object from TfLiteBackendBuffer.
/// Callers can use TfLiteAttributeMap buffer type name to interpret returned
/// pointer.
void* TfLiteBackendBufferGetPtr(const TfLiteBackendBuffer* buf);

/// TfLiteSynchronization is an opaque type that abstracts platform specific
/// implementations of synchronization objects. It's used for carrying the
/// synchronization object across applications, TFLite runtime and backends.
typedef struct TfLiteSynchronization TfLiteSynchronization;

/// Creates an empty TfLiteSynchronization.
/// Returned object is owned by the caller.
TfLiteSynchronization* TfLiteSynchronizationCreate();

/// Destroys a TfLiteSynchronization.
/// Calling this function will not release the synchronization object stored.
void TfLiteSynchronizationDelete(TfLiteSynchronization* sync);

/// Stores a type-punned pointer to a platform-specific synchronization object.
/// `sync` will not own or control the lifecycle of `ptr`.
/// Callers needs to ensure lifetime of *ptr exceeds `sync`.
void TfLiteSynchronizationSetPtr(TfLiteSynchronization* sync, void* ptr);

/// Retrieves the sync object from TfLiteSynchronization.
/// Callers can use TfLiteAttributeMap sync type name to interpret returned
/// pointer.
void* TfLiteSynchronizationGetPtr(const TfLiteSynchronization* sync);

/// Type of the attribute map.
/// An attribute map can either describe the properties of backend buffers
/// or synchronizations.
/// The value of the TfLiteAttrMapType determines the interpretation of
/// attribute keys. See comments below.
typedef enum TfLiteAttrMapType {
  /// Unknown type.
  kTfLiteAttrMapTypeUnknown = 0,

  /// The attributes describes a platform-specific hardware buffer object (e.g.
  /// AHardwareBuffer for Android).
  /// Keys are of TfLiteBufferAttrKey type.
  kTfLiteAttrMapTypeBuffer = 1,

  /// The attributes describes a sync object (e.g. a file descriptor as sync
  /// fence).
  /// Keys are of TfLiteSynchronizationAttrKey type.
  kTfLiteAttrMapTypeSync = 2,
} TfLiteAttrMapType;

/// General hardware buffer attribute keys that are recognizable by TFLite.
typedef enum TfLiteBufferAttrKey {
  kTfLiteBufferAttrKeyUnknown = 0,
  /// Backing buffer resource. const char*
  /// e.g. "AHardwareBuffer".
  kTfLiteBufferAttrKeyResourceTypeName = 1,
  /// Buffer alignment, size_t
  kTfLiteBufferAttrKeyAlignment = 2,
  /// Buffer padding, size_t
  kTfLiteBufferAttrKeyPadding = 3,
  /// Buffer offset, size_t
  kTfLiteBufferAttrKeyOffset = 4,
  /// Buffer size (padded size if applicable), size_t
  kTfLiteBufferAttrKeySize = 5,
  /// Buffer current host coherency state, bool
  kTfLiteBufferAttrKeyCurrentHostCoherencyState = 6,
  /// Buffer preferred host coherency state, bool
  kTfLiteBufferAttrKeyPreferredHostCoherencyState = 7,
  /// Buffer current host cache state, bool
  kTfLiteBufferAttrKeyCurrentHostCacheState = 8,
  /// Buffer preferred cache state, bool
  kTfLiteBufferAttrKeyPreferredHostCacheState = 9,
} TfLiteBufferAttrKey;

/// General synchronization attribute keys that are recognizable by TFLite.
typedef enum TfLiteSynchronizationAttrKey {
  kTfLiteSynchronizationAttrKeyUnknown = 0,
  /// Synchronization type name. const char*
  /// e.g. "sync_fence_fd"
  kTfLiteSynchronizationAttrKeyObjectTypeName = 1,
} TfLiteSynchronizationAttrKey;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_CORE_ASYNC_INTEROP_C_TYPES_H_
