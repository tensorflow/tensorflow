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
#ifndef TENSORFLOW_C_EAGER_C_API_EXPERIMENTAL_H_
#define TENSORFLOW_C_EAGER_C_API_EXPERIMENTAL_H_

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/eager/c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

TF_CAPI_EXPORT extern void TFE_OpConsumeInput(TFE_Op* op, TFE_TensorHandle* h,
                                              TF_Status* status);

typedef struct TFE_ProfilerContext TFE_ProfilerContext;

// A profiler which will start profiling when creating the object and will stop
// when the object is destroyed. It will profile all operations run under the
// given TFE_Context. Multiple instance of it can be created, but at most one
// of them will profile for each TFE_Context.
// Thread-safety: TFE_Profiler is thread-safe.
typedef struct TFE_Profiler TFE_Profiler;

TF_CAPI_EXPORT extern TFE_Profiler* TFE_NewProfiler(TFE_ProfilerContext* ctx);
TF_CAPI_EXPORT extern bool TFE_ProfilerIsOk(TFE_Profiler* profiler);
TF_CAPI_EXPORT extern void TFE_DeleteProfiler(TFE_Profiler* profiler);

// The output string is a binary string of tensorflow.tpu.Trace. User can write
// the string to file for offline analysis by tensorboard.
TF_CAPI_EXPORT extern void TFE_ProfilerSerializeToString(TFE_Context* ctx,
                                                         TFE_Profiler* profiler,
                                                         TF_Buffer* buf,
                                                         TF_Status* status);

// Return a new profiler context object.
TF_CAPI_EXPORT extern TFE_ProfilerContext* TFE_NewProfilerContext(void);

// Set the eager context in TFE_ProfilerServerOptions
TF_CAPI_EXPORT extern void TFE_ProfilerContextSetEagerContext(
    TFE_ProfilerContext* profiler_context, TFE_Context* eager_context);

// Destroy a profiler context object.
TF_CAPI_EXPORT extern void TFE_DeleteProfilerContext(
    TFE_ProfilerContext* profiler_context);

// Start a profiler grpc server which listens to specified port. It will start
// the server on its own thread. It can be shutdown by terminating tensorflow.
// It can be used in both Eager mode and graph mode. Creating multiple profiler
// server is allowed. The service defined in
// tensorflow/contrib/tpu/profiler/tpu_profiler.proto. Please use
// tensorflow/contrib/tpu/profiler/capture_tpu_profile to capture tracable
// file following
// https://cloud.google.com/tpu/docs/cloud-tpu-tools#capture_trace.
TF_CAPI_EXPORT extern void TFE_StartProfilerServer(TFE_ProfilerContext* context,
                                                   int port);

// Enables only graph collection in RunMetadata on the functions executed from
// this context.
TF_CAPI_EXPORT extern void TFE_ContextEnableGraphCollection(TFE_Context* ctx);

// Disables only graph collection in RunMetadata on the functions executed from
// this context.
TF_CAPI_EXPORT extern void TFE_ContextDisableGraphCollection(TFE_Context* ctx);

// Send a grpc request to profiler server (service_addr) to perform on-demand
// profiling and save the result into logdir which can be visualized by
// TensorBoard. worker_list is the list of worker TPUs separated by ','. Set
// include_dataset_opts to false to profile longer traces. It will block the
// caller thread until receives tracing result.
// This API is designed for TensorBoard, for end user, please use
// tensorflow/contrib/tpu/profiler/capture_tpu_profile instead following
// https://cloud.google.com/tpu/docs/cloud-tpu-tools#capture_trace.
TF_CAPI_EXPORT extern bool TFE_ProfilerClientStartTracing(
    const char* service_addr, const char* logdir, const char* worker_list,
    bool include_dataset_ops, int duration_ms, int num_tracing_attempts);

#ifdef __cplusplus
} /* end extern "C" */
#endif

#endif  // TENSORFLOW_C_EAGER_C_API_EXPERIMENTAL_H_
