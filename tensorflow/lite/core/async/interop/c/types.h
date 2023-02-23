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
/// Opaque types for buffer / synchronization object interoperability.
/// WARNING: This is an experimental type and subject to change.

/// Type of the attribute map.
/// An attribute map can either describe the propoerties of backend buffers
/// or sychronizations.
/// The value of the TfLiteAttrMapType determines the interpretation of
/// attribute keys. See comments below.
typedef enum TfLiteAttrMapType {
  kTfLiteAttrMapUnknown = 0,

  // The attributes describes a backend buffer.
  // Keys are of TfLiteBufferAttributeKey type.
  kTfLiteBufferAttrMap = 1,

  // The attributes describes a sync object.
  // Keys are of TfLiteSyncAttributeKey type.
  kTfLiteSyncAttrMap = 2,
} TfLiteAttrMapType;

/// TfLiteBackendBuffer is a an opaque type that abstracts platform specific
/// implementations of buffer objects. It's used for carrying the actual buffer
/// across applications, TFLite runtime and backends.
typedef struct TfLiteBackendBuffer TfLiteBackendBuffer;

/// Creates an empty TfLiteBackendBuffer.
/// Returned object is owned by the caller.
TfLiteBackendBuffer* TfLiteBackendBufferCreate();

/// Destroys a TfLiteBackendBuffer.
/// Calling this function will not release the actual buffer stored underneath.
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
/// actual sync object across applications, TFLite runtime and backends.
typedef struct TfLiteSynchronization TfLiteSynchronization;

/// Creates an empty TfLiteSynchronization.
/// Returned object is owned by the caller.
TfLiteSynchronization* TfLiteSynchronizationCreate();

/// Destroys a TfLiteSynchronization.
/// Calling this function will not release the actual sync object stored.
void TfLiteSynchronizationDelete(TfLiteSynchronization* sync);

/// Stores a type-punned pointer for actual synchronization object.
/// `sync` will not own or control the lifecycle of `ptr`.
/// Callers needs to ensure lifetime of *ptr exceeds `sync`.
void TfLiteSynchronizationSetPtr(TfLiteSynchronization* sync, void* ptr);

/// Retrieves the sync object from TfLiteSynchronization.
/// Callers can use TfLiteAttributeMap sync type name to interpret returned
/// pointer.
void* TfLiteSynchronizationGetPtr(const TfLiteSynchronization* sync);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_CORE_ASYNC_INTEROP_C_TYPES_H_
