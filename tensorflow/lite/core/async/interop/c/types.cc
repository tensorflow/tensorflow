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
#include "tensorflow/lite/core/async/interop/c/types.h"

struct TfLiteBackendBuffer {
  void* ptr = nullptr;
};

struct TfLiteSynchronization {
  void* ptr = nullptr;
};

extern "C" {

TfLiteBackendBuffer* TfLiteBackendBufferCreate() {
  return new TfLiteBackendBuffer;
}
void TfLiteBackendBufferDelete(TfLiteBackendBuffer* buf) {
  if (buf) delete buf;
}
void TfLiteBackendBufferSetPtr(TfLiteBackendBuffer* buf, void* ptr) {
  buf->ptr = ptr;
}

void* TfLiteBackendBufferGetPtr(const TfLiteBackendBuffer* buf) {
  return buf->ptr;
}

TfLiteSynchronization* TfLiteSynchronizationCreate() {
  return new TfLiteSynchronization;
}
void TfLiteSynchronizationDelete(TfLiteSynchronization* sync) {
  if (sync) delete sync;
}
void TfLiteSynchronizationSetPtr(TfLiteSynchronization* sync, void* ptr) {
  sync->ptr = ptr;
}

void* TfLiteSynchronizationGetPtr(const TfLiteSynchronization* sync) {
  return sync->ptr;
}

}  // extern "C"
