/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/c/eager/c_api_experimental.h"

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/eager/c_api_internal.h"

using tensorflow::string;

void TFE_OpConsumeInput(TFE_Op* op, TFE_TensorHandle* h, TF_Status* status) {
  op->operation.ConsumeInput(h->handle);
}

TFE_Profiler* TFE_NewProfiler(TFE_Context* ctx) {
  return new TFE_Profiler(ctx);
}

void TFE_DeleteProfiler(TFE_Profiler* profiler) { delete profiler; }

void TFE_ProfilerSerializeToString(TFE_Context* ctx, TFE_Profiler* profiler,
                                   TF_Buffer* buf, TF_Status* status) {
  TFE_ContextAsyncWait(ctx, status);
  if (!status->status.ok()) return;
  string content;
  status->status = profiler->profiler->SerializeToString(&content);
  void* data = tensorflow::port::Malloc(content.length());
  content.copy(static_cast<char*>(data), content.length(), 0);
  buf->data = data;
  buf->length = content.length();
  buf->data_deallocator = [](void* data, size_t length) {
    tensorflow::port::Free(data);
  };
}
