/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_C_EAGER_C_API_INTERNAL_H_
#define TENSORFLOW_C_EAGER_C_API_INTERNAL_H_

#include <algorithm>
#include <cstddef>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <vector>

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/tensor_handle_interface.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/eager/attr_builder.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/eager_executor.h"
#include "tensorflow/core/common_runtime/eager/eager_operation.h"
#include "tensorflow/core/common_runtime/eager/kernel_and_device.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/lib/monitoring/gauge.h"
#include "tensorflow/core/lib/monitoring/sampler.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/profiler/lib/profiler_session.h"
#include "tensorflow/core/public/version.h"

struct TFE_ContextOptions {
  TF_SessionOptions session_options;
  // true if async execution is enabled.
  bool async = false;
  TFE_ContextDevicePlacementPolicy device_placement_policy{
      TFE_DEVICE_PLACEMENT_SILENT};
  TFE_ContextMirroringPolicy mirroring_policy{TFE_MIRRORING_NONE};
  // If true, lazily copy the remote inputs of a function to the target devices.
  bool lazy_remote_inputs_copy = true;
};

struct TFE_Context {
  TFE_Context(const tensorflow::SessionOptions& opts,
              TFE_ContextDevicePlacementPolicy default_device_placement_policy,
              TFE_ContextMirroringPolicy default_mirroring_policy, bool async,
              const bool lazy_remote_inputs_copy,
              const tensorflow::DeviceMgr* device_mgr, bool device_mgr_owned,
              tensorflow::Rendezvous* rendezvous,
              const tensorflow::CustomKernelCreator* custom_kernel_creator)
      : context(new tensorflow::EagerContext(
            opts,
            static_cast<tensorflow::ContextDevicePlacementPolicy>(
                default_device_placement_policy),
            static_cast<tensorflow::ContextMirroringPolicy>(
                default_mirroring_policy),
            async, lazy_remote_inputs_copy, device_mgr, device_mgr_owned,
            rendezvous, custom_kernel_creator)) {}

  ~TFE_Context() {
    // TODO(iga): Add a separate API method to shutdown TFE_Context so that we
    // don't send RPCs and block in destructor.
    context->WaitForAndCloseRemoteContexts();
    // context->RefCountIsOne() should be true here.
    // TODO(iga): Remove EagerContext refcounting.
    context->Unref();
  }

  tensorflow::EagerContext* context;
};

struct TFE_TensorHandle {
  static TFE_TensorHandle* CreateLocalHandle(const class tensorflow::Tensor& t,
                                             TF_Status* s) {
    tensorflow::TensorHandle* handle;
    s->status = tensorflow::TensorHandle::CreateLocalHandle(t, &handle);
    if (!s->status.ok()) {
      return nullptr;
    }
    return new TFE_TensorHandle{tensorflow::TensorHandleInterface(handle)};
  }

  tensorflow::TensorHandleInterface handle;
};

struct TFE_TensorDebugInfo {
  explicit TFE_TensorDebugInfo(const std::vector<tensorflow::int64>& dims)
      : dev_dims(dims) {}

  // Fully-padded, minor-to-major.
  std::vector<tensorflow::int64> dev_dims;
};

struct TFE_OpInferenceContext {
  explicit TFE_OpInferenceContext(const tensorflow::OpDef* op_def)
      : op_def(op_def) {}

  const tensorflow::OpDef* op_def;  // op definition from protobuf
  int input_arg_idx = 0;  // arg definition index for the next input to be added
  tensorflow::gtl::FlatSet<std::string> attrs;  // attributes inferred so far
};

struct TFE_Op {
  TFE_Op(TFE_Context* ctx, const char* op, bool is_function,
         const tensorflow::AttrTypeMap* t,
         std::unique_ptr<TFE_OpInferenceContext> inference_ctx)
      : ctx(ctx),
        operation(ctx->context, op, is_function, t),
        inference_ctx(std::move(inference_ctx)) {}

  void Clear() {
    operation.Clear();
    inference_ctx.reset();
  }

  tensorflow::Status Reset(const char* op, bool is_function,
                           const tensorflow::AttrTypeMap* t,
                           const char* raw_device_name,
                           std::unique_ptr<TFE_OpInferenceContext> infer_ctx) {
    inference_ctx = std::move(infer_ctx);
    return operation.Reset(ctx->context, op, is_function, t, raw_device_name,
                           nullptr);
  }

  void AddInput(TFE_TensorHandle* input, TF_Status* status);
  void Execute(TFE_TensorHandle** retvals, int* num_retvals, TF_Status* status);

  TFE_Context* ctx;
  tensorflow::EagerOperation operation;
  std::unique_ptr<TFE_OpInferenceContext> inference_ctx;
};

TFE_Op* NewOrResetOp(TFE_Context* ctx, const char* op_or_function_name,
                     const char* raw_device_name, TF_Status* status,
                     TFE_Op* op_to_reset = nullptr);

struct TFE_Profiler {
  explicit TFE_Profiler() { profiler = tensorflow::ProfilerSession::Create(); }

  std::unique_ptr<tensorflow::ProfilerSession> profiler;
};

struct TFE_MonitoringCounterCell {
  tensorflow::monitoring::CounterCell cell;
};

template <int NumLabels>
struct TFE_MonitoringCounter {
  template <typename... LabelDesc>
  TFE_MonitoringCounter(const char* name, const char* description,
                        LabelDesc&&... label) {
    counter = absl::WrapUnique(tensorflow::monitoring::Counter<NumLabels>::New(
        name, description, label...));
  }

  std::unique_ptr<tensorflow::monitoring::Counter<NumLabels>> counter;
};

struct TFE_MonitoringCounter0 : TFE_MonitoringCounter<0> {
  using TFE_MonitoringCounter::TFE_MonitoringCounter;
};
struct TFE_MonitoringCounter1 : TFE_MonitoringCounter<1> {
  using TFE_MonitoringCounter::TFE_MonitoringCounter;
};
struct TFE_MonitoringCounter2 : TFE_MonitoringCounter<2> {
  using TFE_MonitoringCounter::TFE_MonitoringCounter;
};

struct TFE_MonitoringIntGaugeCell {
  tensorflow::monitoring::GaugeCell<tensorflow::int64> cell;
};
struct TFE_MonitoringStringGaugeCell {
  tensorflow::monitoring::GaugeCell<tensorflow::string> cell;
};
struct TFE_MonitoringBoolGaugeCell {
  tensorflow::monitoring::GaugeCell<bool> cell;
};

template <typename ValueType, int NumLabels>
struct TFE_MonitoringGauge {
  template <typename... LabelDesc>
  TFE_MonitoringGauge(const char* name, const char* description,
                      LabelDesc&&... label) {
    gauge = absl::WrapUnique(
        tensorflow::monitoring::Gauge<ValueType, NumLabels>::New(
            name, description, label...));
  }

  std::unique_ptr<tensorflow::monitoring::Gauge<ValueType, NumLabels>> gauge;
};

struct TFE_MonitoringIntGauge0 : TFE_MonitoringGauge<tensorflow::int64, 0> {
  using TFE_MonitoringGauge::TFE_MonitoringGauge;
};
struct TFE_MonitoringIntGauge1 : TFE_MonitoringGauge<tensorflow::int64, 1> {
  using TFE_MonitoringGauge::TFE_MonitoringGauge;
};
struct TFE_MonitoringIntGauge2 : TFE_MonitoringGauge<tensorflow::int64, 2> {
  using TFE_MonitoringGauge::TFE_MonitoringGauge;
};

struct TFE_MonitoringStringGauge0 : TFE_MonitoringGauge<tensorflow::string, 0> {
  using TFE_MonitoringGauge::TFE_MonitoringGauge;
};
struct TFE_MonitoringStringGauge1 : TFE_MonitoringGauge<tensorflow::string, 1> {
  using TFE_MonitoringGauge::TFE_MonitoringGauge;
};
struct TFE_MonitoringStringGauge2 : TFE_MonitoringGauge<tensorflow::string, 2> {
  using TFE_MonitoringGauge::TFE_MonitoringGauge;
};

struct TFE_MonitoringBoolGauge0 : TFE_MonitoringGauge<bool, 0> {
  using TFE_MonitoringGauge::TFE_MonitoringGauge;
};
struct TFE_MonitoringBoolGauge1 : TFE_MonitoringGauge<bool, 1> {
  using TFE_MonitoringGauge::TFE_MonitoringGauge;
};
struct TFE_MonitoringBoolGauge2 : TFE_MonitoringGauge<bool, 2> {
  using TFE_MonitoringGauge::TFE_MonitoringGauge;
};

struct TFE_MonitoringBuckets {
  explicit TFE_MonitoringBuckets(
      std::function<std::unique_ptr<tensorflow::monitoring::Buckets>(void)>
          fn) {
    create_buckets = fn;
  }

  std::function<std::unique_ptr<tensorflow::monitoring::Buckets>(void)>
      create_buckets;
};

struct TFE_MonitoringSamplerCell {
  tensorflow::monitoring::SamplerCell cell;
};

template <int NumLabels>
struct TFE_MonitoringSampler {
  template <typename... LabelDesc>
  TFE_MonitoringSampler(
      const char* name,
      std::unique_ptr<tensorflow::monitoring::Buckets> buckets,
      const char* description, LabelDesc&&... label) {
    sampler = absl::WrapUnique(tensorflow::monitoring::Sampler<NumLabels>::New(
        {name, description, label...}, std::move(buckets)));
  }

  std::unique_ptr<tensorflow::monitoring::Sampler<NumLabels>> sampler;
};

struct TFE_MonitoringSampler0 : TFE_MonitoringSampler<0> {
  using TFE_MonitoringSampler::TFE_MonitoringSampler;
};
struct TFE_MonitoringSampler1 : TFE_MonitoringSampler<1> {
  using TFE_MonitoringSampler::TFE_MonitoringSampler;
};
struct TFE_MonitoringSampler2 : TFE_MonitoringSampler<2> {
  using TFE_MonitoringSampler::TFE_MonitoringSampler;
};

namespace tensorflow {
// Set an AttrValue on the op. Doesn't handle the list types.
void SetOpAttrValueScalar(TFE_Context* ctx, TFE_Op* op,
                          const tensorflow::AttrValue& default_value,
                          const char* attr_name, TF_Status* status);
}  // namespace tensorflow

struct TFE_CancellationManager {
  tensorflow::CancellationManager cancellation_manager;
};

struct TFE_Executor {
  explicit TFE_Executor(bool async)
      : owned_executor(new tensorflow::EagerExecutor(async)) {}

  explicit TFE_Executor(tensorflow::EagerExecutor* executor)
      : owned_executor(nullptr), unowned_executor(executor) {}

  tensorflow::EagerExecutor* executor() {
    return owned_executor == nullptr ? unowned_executor : owned_executor.get();
  }

  std::unique_ptr<tensorflow::EagerExecutor> owned_executor;
  tensorflow::EagerExecutor* unowned_executor;
};

#endif  // TENSORFLOW_C_EAGER_C_API_INTERNAL_H_
