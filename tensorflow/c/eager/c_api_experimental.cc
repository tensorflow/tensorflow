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
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/lib/monitoring/gauge.h"
#include "tensorflow/core/lib/monitoring/sampler.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/mutex.h"
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
  if (TF_GetCode(status) != TF_OK) return;
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

static tensorflow::mutex gauges_map_lock(tensorflow::LINKER_INITIALIZED);

static std::unordered_map<string,
                          tensorflow::monitoring::Gauge<tensorflow::int64, 1>*>*
get_gauges_map() EXCLUSIVE_LOCKS_REQUIRED(gauges_map_lock) {
  static std::unordered_map<
      string, tensorflow::monitoring::Gauge<tensorflow::int64, 1>*>*
      gauges_map = new std::unordered_map<
          string, tensorflow::monitoring::Gauge<tensorflow::int64, 1>*>;
  return gauges_map;
}

static tensorflow::mutex counters_map_lock(tensorflow::LINKER_INITIALIZED);

static std::unordered_map<string, tensorflow::monitoring::Counter<1>*>*
get_counters_map() EXCLUSIVE_LOCKS_REQUIRED(counters_map_lock) {
  static std::unordered_map<string, tensorflow::monitoring::Counter<1>*>*
      counters_map =
          new std::unordered_map<string, tensorflow::monitoring::Counter<1>*>;
  return counters_map;
}

static tensorflow::mutex samplers_map_lock(tensorflow::LINKER_INITIALIZED);

static std::unordered_map<string, tensorflow::monitoring::Sampler<1>*>*
get_samplers_map() EXCLUSIVE_LOCKS_REQUIRED(samplers_map_lock) {
  static std::unordered_map<string, tensorflow::monitoring::Sampler<1>*>*
      samplers_map =
          new std::unordered_map<string, tensorflow::monitoring::Sampler<1>*>;
  return samplers_map;
}

void TFE_MonitoringSetGauge(const char* name, const char* label,
                            int64_t value) {
  tensorflow::mutex_lock l(gauges_map_lock);
  auto gauges_map = get_gauges_map();
  if (gauges_map->find(name) == gauges_map->end()) {
    gauges_map->emplace(
        name, tensorflow::monitoring::Gauge<tensorflow::int64, 1>::New(
                  name,
                  tensorflow::strings::StrCat(
                      name, " :Gauge metric collected from Python API."),
                  "metric_descriptor"));
  }
  gauges_map->at(name)->GetCell(label)->Set(value);
}

void TFE_MonitoringAddCounter(const char* name, const char* label,
                              int64_t value) {
  tensorflow::mutex_lock l(counters_map_lock);
  auto counters_map = get_counters_map();
  if (counters_map->find(name) == counters_map->end()) {
    counters_map->emplace(
        name, tensorflow::monitoring::Counter<1>::New(
                  name,
                  tensorflow::strings::StrCat(
                      name, " :Counter metric collected from Python API."),
                  "metric_descriptor"));
  }
  counters_map->at(name)->GetCell(label)->IncrementBy(value);
}

void TFE_MonitoringAddSampler(const char* name, const char* label,
                              double value) {
  tensorflow::mutex_lock l(samplers_map_lock);
  auto samplers_map = get_samplers_map();
  if (samplers_map->find(name) == samplers_map->end()) {
    samplers_map->emplace(
        name, tensorflow::monitoring::Sampler<1>::New(
                  {name,
                   tensorflow::strings::StrCat(
                       name, " :Counter metric collected from Python API."),
                   "metric_descriptor"},
                  {tensorflow::monitoring::Buckets::Exponential(1, 2, 30)}));
  }
  samplers_map->at(name)->GetCell(label)->Add(value);
}
