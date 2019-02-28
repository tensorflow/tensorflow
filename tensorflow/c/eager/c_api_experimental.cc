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
#include "tensorflow/core/profiler/rpc/client/capture_profile.h"
#include "tensorflow/core/profiler/rpc/profiler_server.h"

using tensorflow::string;

void TFE_OpConsumeInput(TFE_Op* op, TFE_TensorHandle* h, TF_Status* status) {
  op->operation.ConsumeInput(h->handle);
}

TFE_Profiler* TFE_NewProfiler(TFE_ProfilerContext* ctx) {
  return new TFE_Profiler(ctx);
}

bool TFE_ProfilerIsOk(TFE_Profiler* profiler) {
  return profiler->profiler->Status().ok();
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

TFE_ProfilerContext* TFE_NewProfilerContext() {
  return new TFE_ProfilerContext;
}

void TFE_ProfilerContextSetEagerContext(TFE_ProfilerContext* profiler_context,
                                        TFE_Context* eager_context) {
  profiler_context->profiler_context.eager_context = &eager_context->context;
}

void TFE_DeleteProfilerContext(TFE_ProfilerContext* profiler_context) {
  delete profiler_context;
}

void TFE_StartProfilerServer(TFE_ProfilerContext* context, int port) {
  // Release child thread intentionally. The child thread can be terminate by
  // terminating the main thread.
  tensorflow::StartProfilerServer(&context->profiler_context, port).release();
}

void TFE_ContextEnableGraphCollection(TFE_Context* ctx) {
  ctx->context.SetShouldStoreGraphs(true);
}

void TFE_ContextDisableGraphCollection(TFE_Context* ctx) {
  ctx->context.SetShouldStoreGraphs(false);
}

bool TFE_ProfilerClientStartTracing(const char* service_addr,
                                    const char* logdir, const char* worker_list,
                                    bool include_dataset_ops, int duration_ms,
                                    int num_tracing_attempts) {
  tensorflow::Status s =
      tensorflow::profiler::client::ValidateHostPortPair(service_addr);
  if (!s.ok()) {
    return false;
  }
  s = tensorflow::profiler::client::StartTracing(
      service_addr, logdir, worker_list, include_dataset_ops, duration_ms,
      num_tracing_attempts);
  return s.ok();
}
