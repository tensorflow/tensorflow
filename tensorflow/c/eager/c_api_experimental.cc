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

#include <vector>

#include "absl/strings/match.h"
#include "absl/time/time.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/eager/c_api_internal.h"
#include "tensorflow/c/eager/tfe_context_internal.h"
#include "tensorflow/c/eager/tfe_op_internal.h"
#include "tensorflow/c/eager/tfe_tensorhandle_internal.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/common_runtime/composite_device.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/eager/eager_operation.h"
#include "tensorflow/core/distributed_runtime/coordination/coordination_service_error_util.h"
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/lib/monitoring/gauge.h"
#include "tensorflow/core/lib/monitoring/sampler.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/tsl/distributed_runtime/coordination/coordination_service_agent.h"

using tensorflow::string;

void TFE_OpReset(TFE_Op* op_to_reset, const char* op_or_function_name,
                 const char* raw_device_name, TF_Status* status) {
  if (op_to_reset) {
    tensorflow::ImmediateExecutionOperation* op =
        tensorflow::unwrap(op_to_reset);
    op->Clear();
    status->status = op->Reset(op_or_function_name, raw_device_name);
  } else {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "op_to_reset should not be nullptr");
  }
}

void TFE_ContextEnableGraphCollection(TFE_Context* ctx) {
  tensorflow::unwrap(ctx)->SetShouldStoreGraphs(true);
}

void TFE_ContextDisableGraphCollection(TFE_Context* ctx) {
  tensorflow::unwrap(ctx)->SetShouldStoreGraphs(false);
}

uint64_t TFE_GetContextId(TFE_Context* ctx) {
  tensorflow::EagerContext* context =
      tensorflow::ContextFromInterface(tensorflow::unwrap(ctx));
  return context->GetContextId();
}

void TFE_MonitoringCounterCellIncrementBy(TFE_MonitoringCounterCell* cell,
                                          int64_t value) {
  cell->cell.IncrementBy(value);
}

int64_t TFE_MonitoringCounterCellValue(TFE_MonitoringCounterCell* cell) {
  return cell->cell.value();
}

TFE_MonitoringCounter0* TFE_MonitoringNewCounter0(const char* name,
                                                  TF_Status* status,
                                                  const char* description) {
  auto* result = new TFE_MonitoringCounter0({name, description});
  tsl::Set_TF_Status_from_Status(status, result->counter->GetStatus());
  if (!result->counter->GetStatus().ok()) {
    delete result;
    return nullptr;
  }
  return result;
}

void TFE_MonitoringDeleteCounter0(TFE_MonitoringCounter0* counter) {
  delete counter;
}

TFE_MonitoringCounterCell* TFE_MonitoringGetCellCounter0(
    TFE_MonitoringCounter0* counter) {
  return static_cast<TFE_MonitoringCounterCell*>(
      static_cast<void*>(counter->counter->GetCell()));
}

TFE_MonitoringCounter1* TFE_MonitoringNewCounter1(const char* name,
                                                  TF_Status* status,
                                                  const char* description,
                                                  const char* label1) {
  auto* result = new TFE_MonitoringCounter1({name, description, label1});
  tsl::Set_TF_Status_from_Status(status, result->counter->GetStatus());
  if (!result->counter->GetStatus().ok()) {
    delete result;
    return nullptr;
  }
  return result;
}

void TFE_MonitoringDeleteCounter1(TFE_MonitoringCounter1* counter) {
  delete counter;
}

TFE_MonitoringCounterCell* TFE_MonitoringGetCellCounter1(
    TFE_MonitoringCounter1* counter, const char* label1) {
  return static_cast<TFE_MonitoringCounterCell*>(
      static_cast<void*>(counter->counter->GetCell(label1)));
}

TFE_MonitoringCounter2* TFE_MonitoringNewCounter2(const char* name,
                                                  TF_Status* status,
                                                  const char* description,
                                                  const char* label1,
                                                  const char* label2) {
  auto* result =
      new TFE_MonitoringCounter2({name, description, label1, label2});
  tsl::Set_TF_Status_from_Status(status, result->counter->GetStatus());
  if (!result->counter->GetStatus().ok()) {
    delete result;
    return nullptr;
  }
  return result;
}

void TFE_MonitoringDeleteCounter2(TFE_MonitoringCounter2* counter) {
  delete counter;
}

TFE_MonitoringCounterCell* TFE_MonitoringGetCellCounter2(
    TFE_MonitoringCounter2* counter, const char* label1, const char* label2) {
  return static_cast<TFE_MonitoringCounterCell*>(
      static_cast<void*>(counter->counter->GetCell(label1, label2)));
}

void TFE_MonitoringIntGaugeCellSet(TFE_MonitoringIntGaugeCell* cell,
                                   int64_t value) {
  cell->cell.Set(value);
}

int64_t TFE_MonitoringIntGaugeCellValue(TFE_MonitoringIntGaugeCell* cell) {
  return cell->cell.value();
}

TFE_MonitoringIntGauge0* TFE_MonitoringNewIntGauge0(const char* name,
                                                    TF_Status* status,
                                                    const char* description) {
  auto* result = new TFE_MonitoringIntGauge0({name, description});
  tsl::Set_TF_Status_from_Status(status, result->gauge->GetStatus());
  if (!result->gauge->GetStatus().ok()) {
    delete result;
    return nullptr;
  }
  return result;
}

void TFE_MonitoringDeleteIntGauge0(TFE_MonitoringIntGauge0* gauge) {
  delete gauge;
}

TFE_MonitoringIntGaugeCell* TFE_MonitoringGetCellIntGauge0(
    TFE_MonitoringIntGauge0* gauge) {
  return static_cast<TFE_MonitoringIntGaugeCell*>(
      static_cast<void*>(gauge->gauge->GetCell()));
}

TFE_MonitoringIntGauge1* TFE_MonitoringNewIntGauge1(const char* name,
                                                    TF_Status* status,
                                                    const char* description,
                                                    const char* label1) {
  auto* result = new TFE_MonitoringIntGauge1({name, description, label1});
  tsl::Set_TF_Status_from_Status(status, result->gauge->GetStatus());
  if (!result->gauge->GetStatus().ok()) {
    delete result;
    return nullptr;
  }
  return result;
}

void TFE_MonitoringDeleteIntGauge1(TFE_MonitoringIntGauge1* gauge) {
  delete gauge;
}

TFE_MonitoringIntGaugeCell* TFE_MonitoringGetCellIntGauge1(
    TFE_MonitoringIntGauge1* gauge, const char* label1) {
  return static_cast<TFE_MonitoringIntGaugeCell*>(
      static_cast<void*>(gauge->gauge->GetCell(label1)));
}

TFE_MonitoringIntGauge2* TFE_MonitoringNewIntGauge2(const char* name,
                                                    TF_Status* status,
                                                    const char* description,
                                                    const char* label1,
                                                    const char* label2) {
  auto* result =
      new TFE_MonitoringIntGauge2({name, description, label1, label2});
  tsl::Set_TF_Status_from_Status(status, result->gauge->GetStatus());
  if (!result->gauge->GetStatus().ok()) {
    delete result;
    return nullptr;
  }
  return result;
}

void TFE_MonitoringDeleteIntGauge2(TFE_MonitoringIntGauge2* gauge) {
  delete gauge;
}

TFE_MonitoringIntGaugeCell* TFE_MonitoringGetCellIntGauge2(
    TFE_MonitoringIntGauge2* gauge, const char* label1, const char* label2) {
  return static_cast<TFE_MonitoringIntGaugeCell*>(
      static_cast<void*>(gauge->gauge->GetCell(label1, label2)));
}

void TFE_MonitoringStringGaugeCellSet(TFE_MonitoringStringGaugeCell* cell,
                                      const char* value) {
  cell->cell.Set({value});
}

const void TFE_MonitoringStringGaugeCellValue(
    TFE_MonitoringStringGaugeCell* cell, TF_Buffer* buf) {
  tensorflow::string value = cell->cell.value();
  void* data = tensorflow::port::Malloc(value.length());
  value.copy(static_cast<char*>(data), value.length(), 0);
  buf->data = data;
  buf->length = value.length();
  buf->data_deallocator = [](void* data, size_t length) {
    tensorflow::port::Free(data);
  };
}

TFE_MonitoringStringGauge0* TFE_MonitoringNewStringGauge0(
    const char* name, TF_Status* status, const char* description) {
  auto* result = new TFE_MonitoringStringGauge0({name, description});
  tsl::Set_TF_Status_from_Status(status, result->gauge->GetStatus());
  if (!result->gauge->GetStatus().ok()) {
    delete result;
    return nullptr;
  }
  return result;
}

void TFE_MonitoringDeleteStringGauge0(TFE_MonitoringStringGauge0* gauge) {
  delete gauge;
}

TFE_MonitoringStringGaugeCell* TFE_MonitoringGetCellStringGauge0(
    TFE_MonitoringStringGauge0* gauge) {
  return static_cast<TFE_MonitoringStringGaugeCell*>(
      static_cast<void*>(gauge->gauge->GetCell()));
}

TFE_MonitoringStringGauge1* TFE_MonitoringNewStringGauge1(
    const char* name, TF_Status* status, const char* description,
    const char* label1) {
  auto* result = new TFE_MonitoringStringGauge1({name, description, label1});
  tsl::Set_TF_Status_from_Status(status, result->gauge->GetStatus());
  if (!result->gauge->GetStatus().ok()) {
    delete result;
    return nullptr;
  }
  return result;
}

void TFE_MonitoringDeleteStringGauge1(TFE_MonitoringStringGauge1* gauge) {
  delete gauge;
}

TFE_MonitoringStringGaugeCell* TFE_MonitoringGetCellStringGauge1(
    TFE_MonitoringStringGauge1* gauge, const char* label1) {
  return static_cast<TFE_MonitoringStringGaugeCell*>(
      static_cast<void*>(gauge->gauge->GetCell(label1)));
}

TFE_MonitoringStringGauge2* TFE_MonitoringNewStringGauge2(
    const char* name, TF_Status* status, const char* description,
    const char* label1, const char* label2) {
  auto* result =
      new TFE_MonitoringStringGauge2({name, description, label1, label2});
  tsl::Set_TF_Status_from_Status(status, result->gauge->GetStatus());
  if (!result->gauge->GetStatus().ok()) {
    delete result;
    return nullptr;
  }
  return result;
}

void TFE_MonitoringDeleteStringGauge2(TFE_MonitoringStringGauge2* gauge) {
  delete gauge;
}

TFE_MonitoringStringGaugeCell* TFE_MonitoringGetCellStringGauge2(
    TFE_MonitoringStringGauge2* gauge, const char* label1, const char* label2) {
  return static_cast<TFE_MonitoringStringGaugeCell*>(
      static_cast<void*>(gauge->gauge->GetCell(label1, label2)));
}

TFE_MonitoringStringGauge3* TFE_MonitoringNewStringGauge3(
    const char* name, TF_Status* status, const char* description,
    const char* label1, const char* label2, const char* label3) {
  auto* result = new TFE_MonitoringStringGauge3(
      {name, description, label1, label2, label3});
  tsl::Set_TF_Status_from_Status(status, result->gauge->GetStatus());
  if (!result->gauge->GetStatus().ok()) {
    delete result;
    return nullptr;
  }
  return result;
}

void TFE_MonitoringDeleteStringGauge3(TFE_MonitoringStringGauge3* gauge) {
  delete gauge;
}

TFE_MonitoringStringGaugeCell* TFE_MonitoringGetCellStringGauge3(
    TFE_MonitoringStringGauge3* gauge, const char* label1, const char* label2,
    const char* label3) {
  return static_cast<TFE_MonitoringStringGaugeCell*>(
      static_cast<void*>(gauge->gauge->GetCell(label1, label2, label3)));
}

TFE_MonitoringStringGauge4* TFE_MonitoringNewStringGauge4(
    const char* name, TF_Status* status, const char* description,
    const char* label1, const char* label2, const char* label3,
    const char* label4) {
  auto* result = new TFE_MonitoringStringGauge4(
      {name, description, label1, label2, label3, label4});
  tsl::Set_TF_Status_from_Status(status, result->gauge->GetStatus());
  if (!result->gauge->GetStatus().ok()) {
    delete result;
    return nullptr;
  }
  return result;
}

void TFE_MonitoringDeleteStringGauge4(TFE_MonitoringStringGauge4* gauge) {
  delete gauge;
}

TFE_MonitoringStringGaugeCell* TFE_MonitoringGetCellStringGauge4(
    TFE_MonitoringStringGauge4* gauge, const char* label1, const char* label2,
    const char* label3, const char* label4) {
  return static_cast<TFE_MonitoringStringGaugeCell*>(static_cast<void*>(
      gauge->gauge->GetCell(label1, label2, label3, label4)));
}

void TFE_MonitoringBoolGaugeCellSet(TFE_MonitoringBoolGaugeCell* cell,
                                    bool value) {
  cell->cell.Set(value);
}

bool TFE_MonitoringBoolGaugeCellValue(TFE_MonitoringBoolGaugeCell* cell) {
  return cell->cell.value();
}

TFE_MonitoringBoolGauge0* TFE_MonitoringNewBoolGauge0(const char* name,
                                                      TF_Status* status,
                                                      const char* description) {
  auto* result = new TFE_MonitoringBoolGauge0({name, description});
  tsl::Set_TF_Status_from_Status(status, result->gauge->GetStatus());
  if (!result->gauge->GetStatus().ok()) {
    delete result;
    return nullptr;
  }
  return result;
}

void TFE_MonitoringDeleteBoolGauge0(TFE_MonitoringBoolGauge0* gauge) {
  delete gauge;
}

TFE_MonitoringBoolGaugeCell* TFE_MonitoringGetCellBoolGauge0(
    TFE_MonitoringBoolGauge0* gauge) {
  return static_cast<TFE_MonitoringBoolGaugeCell*>(
      static_cast<void*>(gauge->gauge->GetCell()));
}

TFE_MonitoringBoolGauge1* TFE_MonitoringNewBoolGauge1(const char* name,
                                                      TF_Status* status,
                                                      const char* description,
                                                      const char* label1) {
  auto* result = new TFE_MonitoringBoolGauge1({name, description, label1});
  tsl::Set_TF_Status_from_Status(status, result->gauge->GetStatus());
  if (!result->gauge->GetStatus().ok()) {
    delete result;
    return nullptr;
  }
  return result;
}

void TFE_MonitoringDeleteBoolGauge1(TFE_MonitoringBoolGauge1* gauge) {
  delete gauge;
}

TFE_MonitoringBoolGaugeCell* TFE_MonitoringGetCellBoolGauge1(
    TFE_MonitoringBoolGauge1* gauge, const char* label1) {
  return static_cast<TFE_MonitoringBoolGaugeCell*>(
      static_cast<void*>(gauge->gauge->GetCell(label1)));
}

TFE_MonitoringBoolGauge2* TFE_MonitoringNewBoolGauge2(const char* name,
                                                      TF_Status* status,
                                                      const char* description,
                                                      const char* label1,
                                                      const char* label2) {
  auto* result =
      new TFE_MonitoringBoolGauge2({name, description, label1, label2});
  tsl::Set_TF_Status_from_Status(status, result->gauge->GetStatus());
  if (!result->gauge->GetStatus().ok()) {
    delete result;
    return nullptr;
  }
  return result;
}

void TFE_MonitoringDeleteBoolGauge2(TFE_MonitoringBoolGauge2* gauge) {
  delete gauge;
}

TFE_MonitoringBoolGaugeCell* TFE_MonitoringGetCellBoolGauge2(
    TFE_MonitoringBoolGauge2* gauge, const char* label1, const char* label2) {
  return static_cast<TFE_MonitoringBoolGaugeCell*>(
      static_cast<void*>(gauge->gauge->GetCell(label1, label2)));
}

void TFE_MonitoringSamplerCellAdd(TFE_MonitoringSamplerCell* cell,
                                  double value) {
  cell->cell.Add(value);
}

void TFE_MonitoringSamplerCellValue(TFE_MonitoringSamplerCell* cell,
                                    TF_Buffer* buf) {
  string content;
  cell->cell.value().SerializeToString(&content);
  void* data = tensorflow::port::Malloc(content.length());
  content.copy(static_cast<char*>(data), content.length(), 0);
  buf->data = data;
  buf->length = content.length();
  buf->data_deallocator = [](void* data, size_t length) {
    tensorflow::port::Free(data);
  };
}

TFE_MonitoringBuckets* TFE_MonitoringNewExponentialBuckets(double scale,
                                                           double growth_factor,
                                                           int bucket_count) {
  return new TFE_MonitoringBuckets([scale, growth_factor, bucket_count]() {
    return tensorflow::monitoring::Buckets::Exponential(scale, growth_factor,
                                                        bucket_count);
  });
}

void TFE_MonitoringDeleteBuckets(TFE_MonitoringBuckets* buckets) {
  delete buckets;
}

TFE_MonitoringSampler0* TFE_MonitoringNewSampler0(
    const char* name, TFE_MonitoringBuckets* buckets, TF_Status* status,
    const char* description) {
  auto* result = new TFE_MonitoringSampler0(
      {name, buckets->create_buckets(), description});
  tsl::Set_TF_Status_from_Status(status, result->sampler->GetStatus());
  if (!result->sampler->GetStatus().ok()) {
    delete result;
    return nullptr;
  }
  return result;
}

void TFE_MonitoringDeleteSampler0(TFE_MonitoringSampler0* sampler) {
  delete sampler;
}

TFE_MonitoringSamplerCell* TFE_MonitoringGetCellSampler0(
    TFE_MonitoringSampler0* sampler) {
  return static_cast<TFE_MonitoringSamplerCell*>(
      static_cast<void*>(sampler->sampler->GetCell()));
}

TFE_MonitoringSampler1* TFE_MonitoringNewSampler1(
    const char* name, TFE_MonitoringBuckets* buckets, TF_Status* status,
    const char* description, const char* label1) {
  auto* result = new TFE_MonitoringSampler1(
      {name, buckets->create_buckets(), description, label1});
  tsl::Set_TF_Status_from_Status(status, result->sampler->GetStatus());
  if (!result->sampler->GetStatus().ok()) {
    delete result;
    return nullptr;
  }
  return result;
}

void TFE_MonitoringDeleteSampler1(TFE_MonitoringSampler1* sampler) {
  delete sampler;
}

TFE_MonitoringSamplerCell* TFE_MonitoringGetCellSampler1(
    TFE_MonitoringSampler1* sampler, const char* label1) {
  return static_cast<TFE_MonitoringSamplerCell*>(
      static_cast<void*>(sampler->sampler->GetCell(label1)));
}

TFE_MonitoringSampler2* TFE_MonitoringNewSampler2(
    const char* name, TFE_MonitoringBuckets* buckets, TF_Status* status,
    const char* description, const char* label1, const char* label2) {
  auto* result = new TFE_MonitoringSampler2(
      {name, buckets->create_buckets(), description, label1, label2});
  tsl::Set_TF_Status_from_Status(status, result->sampler->GetStatus());
  if (!result->sampler->GetStatus().ok()) {
    delete result;
    return nullptr;
  }
  return result;
}

void TFE_MonitoringDeleteSampler2(TFE_MonitoringSampler2* sampler) {
  delete sampler;
}

TFE_MonitoringSamplerCell* TFE_MonitoringGetCellSampler2(
    TFE_MonitoringSampler2* sampler, const char* label1, const char* label2) {
  return static_cast<TFE_MonitoringSamplerCell*>(
      static_cast<void*>(sampler->sampler->GetCell(label1, label2)));
}

void TFE_ContextOptionsSetTfrt(TFE_ContextOptions* options, bool use_tfrt) {
  options->use_tfrt = use_tfrt;
}

TFE_CancellationManager* TFE_NewCancellationManager() {
  return tensorflow::wrap(new tensorflow::CancellationManager);
}

void TFE_CancellationManagerStartCancel(
    TFE_CancellationManager* cancellation_manager) {
  tensorflow::unwrap(cancellation_manager)->StartCancel();
}

bool TFE_CancellationManagerIsCancelled(
    TFE_CancellationManager* cancellation_manager) {
  return tensorflow::unwrap(cancellation_manager)->IsCancelled();
}

void TFE_DeleteCancellationManager(
    TFE_CancellationManager* cancellation_manager) {
  delete tensorflow::unwrap(cancellation_manager);
}

void TFE_OpSetCancellationManager(TFE_Op* op,
                                  TFE_CancellationManager* cancellation_manager,
                                  TF_Status* status) {
  tensorflow::unwrap(op)->SetCancellationManager(
      tensorflow::unwrap(cancellation_manager));
  status->status = ::tensorflow::OkStatus();
}

TFE_Executor* TFE_NewExecutor(bool is_async, bool enable_streaming_enqueue,
                              int in_flight_nodes_limit) {
  return new TFE_Executor(is_async, enable_streaming_enqueue,
                          in_flight_nodes_limit);
}

void TFE_DeleteExecutor(TFE_Executor* executor) { delete executor; }

bool TFE_ExecutorIsAsync(TFE_Executor* executor) {
  return executor->executor()->Async();
}

void TFE_ExecutorWaitForAllPendingNodes(TFE_Executor* executor,
                                        TF_Status* status) {
  status->status = executor->executor()->WaitForAllPendingNodes();
}

void TFE_ExecutorClearError(TFE_Executor* executor) {
  executor->executor()->ClearError();
}

void TFE_ContextSetExecutorForThread(TFE_Context* ctx, TFE_Executor* executor) {
  tensorflow::unwrap(ctx)->SetExecutorForThread(executor->executor());
}

TFE_Executor* TFE_ContextGetExecutorForThread(TFE_Context* ctx) {
  return new TFE_Executor(&tensorflow::unwrap(ctx)->Executor());
}

void TFE_HostAddressSpace(TFE_Context* ctx, TF_Buffer* buf) {
  auto address_space = tensorflow::DeviceNameUtils::AddressSpace(
      tensorflow::unwrap(ctx)->HostCPUParsedName());
  auto str = tensorflow::DeviceNameUtils::ParsedNameToString(address_space);
  void* data = tensorflow::port::Malloc(str.length());
  str.copy(static_cast<char*>(data), str.length(), 0);
  buf->data = data;
  buf->length = str.length();
  buf->data_deallocator = [](void* data, size_t length) {
    tensorflow::port::Free(data);
  };
}

void TFE_ContextGetFunctionDef(TFE_Context* ctx, const char* function_name,
                               TF_Buffer* buf, TF_Status* status) {
  auto* function_def = tensorflow::unwrap(ctx)->FindFunctionDef(function_name);
  if (function_def == nullptr) {
    status->status = tensorflow::errors::NotFound(
        "Unable to find FunctionDef with name: ", function_name);
    return;
  }
  string str = function_def->SerializeAsString();
  void* data = tensorflow::port::Malloc(str.length());
  str.copy(static_cast<char*>(data), str.length(), 0);
  buf->data = data;
  buf->length = str.length();
  buf->data_deallocator = [](void* data, size_t length) {
    tensorflow::port::Free(data);
  };
  status->status = ::tensorflow::OkStatus();
}

TF_Tensor* TFE_AllocateHostTensor(TFE_Context* ctx, TF_DataType dtype,
                                  const int64_t* dims, int num_dims,
                                  TF_Status* status) {
  std::vector<int64_t> dimvec(num_dims);
  for (int i = 0; i < num_dims; ++i) {
    dimvec[i] = static_cast<int64_t>(dims[i]);
  }

  if (ctx == nullptr) {
    status->status = tensorflow::errors::InvalidArgument("Invalid Context");
    return nullptr;
  }

  tensorflow::AbstractTensorInterface* t =
      tensorflow::unwrap(ctx)->CreateTensor(
          static_cast<tensorflow::DataType>(dtype), dimvec);

  if (t == nullptr) {
    status->status =
        tensorflow::errors::InvalidArgument("Unsupported dtype: ", dtype);
    return nullptr;
  }

  return new TF_Tensor{t};
}

TFE_TensorHandle* TFE_NewTensorHandleFromTensor(TFE_Context* ctx, TF_Tensor* t,
                                                TF_Status* status) {
  return tensorflow::wrap(
      tensorflow::unwrap(ctx)->CreateLocalHandle(t->tensor));
}

TFE_TensorHandle* TFE_CreatePackedTensorHandle(TFE_Context* ctx,
                                               TFE_TensorHandle** handles,
                                               int* num_handles,
                                               TF_Status* status) {
  std::vector<tensorflow::TensorHandle*> tensor_handles;
  tensor_handles.reserve(*num_handles);
  for (int i = 0; i < *num_handles; ++i) {
    tensorflow::ImmediateExecutionTensorHandle* unwrapped_handle =
        tensorflow::unwrap(handles[i]);
    if (tensorflow::CustomDeviceTensorHandle::classof(unwrapped_handle)) {
      // One of the inputs we're trying to pack is on a custom device. We'll let
      // the first custom device we see handle all of the packing.
      auto* custom_device_handle =
          tensorflow::down_cast<tensorflow::CustomDeviceTensorHandle*>(
              unwrapped_handle);
      tensorflow::ImmediateExecutionTensorHandle* result;
      status->status = custom_device_handle->device()->Pack(
          absl::Span<tensorflow::ImmediateExecutionTensorHandle*>(
              tensorflow::unwrap(handles), *num_handles),
          &result);
      return tensorflow::wrap(result);
    }
    tensor_handles.push_back(
        tensorflow::TensorHandleFromInterface(unwrapped_handle));
  }
  tensorflow::EagerContext* context =
      tensorflow::ContextFromInterface(tensorflow::unwrap(ctx));
  tensorflow::TensorHandle* handle = nullptr;
  status->status = tensorflow::TensorHandle::CreatePackedHandle(
      std::move(tensor_handles), context, &handle);
  return tensorflow::wrap(handle);
}

void TFE_ContextSetSoftDevicePlacement(TFE_Context* ctx, unsigned char enable,
                                       TF_Status* status) {
  tensorflow::unwrap(ctx)->SetAllowSoftPlacement(enable);
}

void TFE_ContextSetLogDevicePlacement(TFE_Context* ctx, unsigned char enable,
                                      TF_Status* status) {
  tensorflow::unwrap(ctx)->SetLogDevicePlacement(enable);
}

void TFE_ContextSetRunEagerOpAsFunction(TFE_Context* ctx, unsigned char enable,
                                        TF_Status* status) {
  tensorflow::unwrap(ctx)->SetRunEagerOpAsFunction(enable);
}

void TFE_ContextSetJitCompileRewrite(TFE_Context* ctx, unsigned char enable,
                                     TF_Status* status) {
  tensorflow::unwrap(ctx)->SetJitCompileRewrite(enable);
}

const char* TFE_TensorHandleDeviceType(TFE_TensorHandle* h, TF_Status* status) {
  if (h == nullptr) {
    status->status = tensorflow::errors::InvalidArgument("Invalid handle");
    return nullptr;
  }
  return tensorflow::unwrap(h)->DeviceType(&status->status);
}

int TFE_TensorHandleDeviceID(TFE_TensorHandle* h, TF_Status* status) {
  if (h == nullptr) {
    status->status = tensorflow::errors::InvalidArgument("Invalid handle");
    return -1;
  }
  return tensorflow::unwrap(h)->DeviceId(&status->status);
}

TF_CAPI_EXPORT extern void TFE_TensorHandleGetStatus(TFE_TensorHandle* h,
                                                     TF_Status* status) {
  status->status = tensorflow::unwrap(h)->TensorHandleStatus();
}

void TFE_GetExecutedOpNames(TFE_Context* ctx, TF_Buffer* buf,
                            TF_Status* status) {
  const std::vector<std::string>& op_names =
      tensorflow::unwrap(ctx)->GetLoggedOpsTestonly();

  std::ostringstream op_names_oss;
  for (const auto& op : op_names) {
    op_names_oss << op << ", ";
  }
  const std::string& op_names_str = op_names_oss.str();
  void* data = tensorflow::port::Malloc(op_names_str.length());
  op_names_str.copy(static_cast<char*>(data), op_names_str.length(), 0);
  buf->data = data;
  buf->length = op_names_str.length();
  buf->data_deallocator = [](void* data, size_t length) {
    tensorflow::port::Free(data);
  };
  status->status = ::tensorflow::OkStatus();
}

void TFE_SetLogicalCpuDevices(TFE_Context* ctx, int num_cpus,
                              const char* prefix, TF_Status* status) {
  std::vector<std::unique_ptr<tensorflow::Device>> devices;

  if (prefix == nullptr || strlen(prefix) == 0)
    prefix = "/job:localhost/replica:0/task:0";

  tensorflow::SessionOptions sess_options;
  (*sess_options.config.mutable_device_count())["CPU"] = num_cpus;
  status->status =
      tensorflow::DeviceFactory::AddCpuDevices(sess_options, prefix, &devices);

  // Remove the device that has the host device name since host device is alreay
  // in an initialized context.
  for (auto d = devices.begin(); d != devices.end();) {
    if (absl::StrContains(d->get()->name(), "CPU:0")) {
      d = devices.erase(d);
    } else {
      ++d;
    }
  }

  status->status = tensorflow::unwrap(ctx)->AddDevices(std::move(devices));
}

void TFE_InsertConfigKeyValue(TFE_Context* ctx, const char* key,
                              const char* value, TF_Status* status) {
  tensorflow::ImmediateExecutionDistributedManager* dist_mgr =
      tensorflow::unwrap(ctx)->GetDistributedManager();
  tsl::CoordinationServiceAgent* coord_agent =
      dist_mgr->GetCoordinationServiceAgent();
  if (coord_agent == nullptr) {
    status->status = tensorflow::errors::FailedPrecondition(
        "Coordination service agent is not enabled.");
    return;
  }
  status->status = coord_agent->InsertKeyValue(key, value);
}

void TFE_GetConfigKeyValue(TFE_Context* ctx, const char* key,
                           int64_t timeout_in_ms, TF_Buffer* value_buf,
                           TF_Status* status) {
  tensorflow::ImmediateExecutionDistributedManager* dist_mgr =
      tensorflow::unwrap(ctx)->GetDistributedManager();
  tsl::CoordinationServiceAgent* coord_agent =
      dist_mgr->GetCoordinationServiceAgent();
  if (coord_agent == nullptr) {
    status->status = tensorflow::errors::FailedPrecondition(
        "Coordination service is not enabled.");
    return;
  }
  absl::Duration timeout;
  if (timeout_in_ms > 0) {
    timeout = absl::Milliseconds(timeout_in_ms);
  } else {
    // Block until the key-value is set or the worker shuts down.
    timeout = absl::InfiniteDuration();
  }
  auto status_or_value = coord_agent->GetKeyValue(key, timeout);
  status->status = status_or_value.status();
  if (!status_or_value.ok()) return;

  const std::string& value_string = status_or_value.value();
  void* data = tensorflow::port::Malloc(value_string.length());
  value_string.copy(static_cast<char*>(data), value_string.length(), 0);
  value_buf->data = data;
  value_buf->length = value_string.length();
  value_buf->data_deallocator = [](void* data, size_t length) {
    tensorflow::port::Free(data);
  };
}

void TFE_DeleteConfigKeyValue(TFE_Context* ctx, const char* key,
                              TF_Status* status) {
  tensorflow::ImmediateExecutionDistributedManager* dist_mgr =
      tensorflow::unwrap(ctx)->GetDistributedManager();
  tsl::CoordinationServiceAgent* coord_agent =
      dist_mgr->GetCoordinationServiceAgent();
  if (coord_agent == nullptr) {
    status->status = tensorflow::errors::FailedPrecondition(
        "Coordination service is not enabled.");
    return;
  }
  status->status = coord_agent->DeleteKeyValue(key);
}

void TFE_ReportErrorToCluster(TFE_Context* ctx, int error_code,
                              const char* error_message, TF_Status* status) {
  tensorflow::ImmediateExecutionDistributedManager* dist_mgr =
      tensorflow::unwrap(ctx)->GetDistributedManager();
  tsl::CoordinationServiceAgent* coord_agent =
      dist_mgr->GetCoordinationServiceAgent();
  if (coord_agent == nullptr) {
    status->status = tensorflow::errors::FailedPrecondition(
        "Coordination service is not enabled.");
    return;
  }
  tensorflow::Status s(static_cast<absl::StatusCode>(error_code),
                       error_message);
  status->status = coord_agent->ReportError(s);
}

void TFE_GetTaskStates(TFE_Context* ctx, const TF_Buffer& tasks, void* states,
                       TF_Status* status) {
  tensorflow::ImmediateExecutionDistributedManager* dist_mgr =
      tensorflow::unwrap(ctx)->GetDistributedManager();
  tsl::CoordinationServiceAgent* coord_agent =
      dist_mgr->GetCoordinationServiceAgent();
  if (coord_agent == nullptr) {
    status->status = tensorflow::errors::FailedPrecondition(
        "Coordination service is not enabled.");
    return;
  }
  std::vector<tensorflow::CoordinatedTask> task_vec(tasks.length);
  auto* task_iter = static_cast<const tensorflow::CoordinatedTask*>(tasks.data);
  for (size_t i = 0; i < tasks.length; ++i) {
    task_vec[i].set_job_name(task_iter->job_name());
    task_vec[i].set_task_id(task_iter->task_id());
    ++task_iter;
  }
  auto results = coord_agent->GetTaskState(task_vec);
  if (!results.ok()) {
    status->status = results.status();
    return;
  }
  auto* state_iter = static_cast<TF_Status*>(states);
  for (size_t i = 0; i < tasks.length; ++i) {
    const auto& result = (*results)[i];
    TF_Status s;
    TF_SetStatus(&s, static_cast<TF_Code>(result.error_code()),
                 std::string(result.error_message()).data());
    if (TF_GetCode(&s) != TF_Code::TF_OK) {
      tensorflow::CoordinationServiceError error;
      *error.mutable_source_task() = result.error_payload().source_task();
      TF_SetPayload(&s, tensorflow::CoordinationErrorPayloadKey().data(),
                    error.SerializeAsString().c_str());
    }
    *state_iter = std::move(s);
    ++state_iter;
  }
  status->status = tensorflow::OkStatus();
}

void TFE_WaitAtBarrier(TFE_Context* ctx, const char* barrier_id,
                       int64_t barrier_timeout_in_ms, TF_Status* status) {
  tensorflow::ImmediateExecutionDistributedManager* dist_mgr =
      tensorflow::unwrap(ctx)->GetDistributedManager();
  tsl::CoordinationServiceAgent* coord_agent =
      dist_mgr->GetCoordinationServiceAgent();
  if (coord_agent == nullptr) {
    status->status = tensorflow::errors::FailedPrecondition(
        "Coordination service is not enabled.");
    return;
  }
  status->status = coord_agent->WaitAtBarrier(
      barrier_id, absl::Milliseconds(barrier_timeout_in_ms), {});
}
