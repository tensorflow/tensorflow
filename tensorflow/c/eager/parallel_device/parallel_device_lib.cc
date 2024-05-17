/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/c/eager/parallel_device/parallel_device_lib.h"

#include <string>
#include <utility>

#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/tfe_cancellation_manager_internal.h"
#include "tensorflow/c/eager/tfe_tensorhandle_internal.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_status_internal.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace parallel_device {
namespace {

class OpDeleter {
 public:
  void operator()(TFE_Op* to_delete) const { TFE_DeleteOp(to_delete); }
};

using OpPtr = std::unique_ptr<TFE_Op, OpDeleter>;

class StatusDeleter {
 public:
  void operator()(TF_Status* to_delete) const { TF_DeleteStatus(to_delete); }
};

using StatusPtr = std::unique_ptr<TF_Status, StatusDeleter>;

class ExecutorDeleter {
 public:
  void operator()(TFE_Executor* to_delete) const {
    TFE_DeleteExecutor(to_delete);
  }
};

using ExecutorPtr = std::unique_ptr<TFE_Executor, ExecutorDeleter>;

}  // namespace

// Allows a single op at a time to be launched without blocking.
//
// DeviceThread itself is thread-safe, in that StartExecute will block if there
// is a pending execution. Since StartExecute is equivalent to grabbing a lock,
// multiple DeviceThreads should always be accessed in the same order to avoid
// deadlocks.
class DeviceThread {
 public:
  // Starts a background thread waiting for `StartExecute`.
  explicit DeviceThread(const std::string& device, const bool is_async,
                        const int in_flight_nodes_limit)
      : status_(TF_NewStatus()),
        // If the context's default exector is set to async, re-using that in
        // each thread would cause collectives to deadlock. For consistency we
        // create a new sync executor for every thread.
        //
        // TODO(allenl): We should have an async API that works with the
        // parallel device.
        device_(device),
        executor_(
            TFE_NewExecutor(is_async, /*enable_streaming_enqueue=*/true,
                            /*in_flight_nodes_limit=*/in_flight_nodes_limit)),
        op_(nullptr),
        thread_(tensorflow::Env::Default()->StartThread(
            tensorflow::ThreadOptions(), "parallel_device_execute",
            std::bind(&DeviceThread::Run, this))) {}
  ~DeviceThread();

  // Requests that the worker thread execute the specified operation. Blocks
  // until the previously pending operation (a StartExecute without a Join) has
  // finished, if any.
  //
  // `cancellation_manager` must live until after `Join` finishes and pending
  // `is_async` operations finish. In addition to allowing the caller to cancel
  // the operation, its `StartCancel` method will be called if op execution
  // fails on any device in order to cancel the others.
  void StartExecute(TFE_Context* context, const char* operation_name,
                    std::vector<TFE_TensorHandle*> inputs,
                    const TFE_OpAttrs* attributes, int expected_max_outputs,
                    CancellationManager& cancellation_manager,
                    absl::optional<int64_t> step_id = absl::nullopt);
  // Block until the previous `StartExecute` operation has executed. Forwards
  // the status from `TFE_Execute` and returns outputs if the status is OK.
  std::vector<TensorHandlePtr> Join(TF_Status* status);

  // Block until all Ops finished running on the thread.
  void AsyncWait(TF_Status* status);

 private:
  void Run();

  void Execute(TFE_Context* context, const char* operation_name,
               std::vector<TFE_TensorHandle*> inputs,
               const TFE_OpAttrs* attributes, int expected_max_outputs,
               std::vector<TensorHandlePtr>* outputs, TF_Status* status) const
      TF_EXCLUSIVE_LOCKS_REQUIRED(execution_mutex_);

  enum class ExecutionState {
    kReadyToExecute,
    kHasResult,
    kIdle,
    kShuttingDown,
  };

  tensorflow::mutex execution_mutex_;
  ExecutionState execution_state_ TF_GUARDED_BY(execution_mutex_) =
      ExecutionState::kIdle;
  // Tells the worker thread that there is new work.
  tensorflow::condition_variable start_execute_;
  // The worker thread notifies that work has finished.
  tensorflow::condition_variable finished_execute_;
  // Notifies a StartExecute that the previous Join has finished.
  tensorflow::condition_variable finished_join_;

  // Temporary state between `StartExecute` and `Join`.
  //
  //   Inputs; pointers are to objects not owned by the DeviceThread, but which
  //   are expected to live at least until `Join` finishes:
  TFE_Context* context_ TF_GUARDED_BY(execution_mutex_);
  const char* operation_name_ TF_GUARDED_BY(execution_mutex_);
  absl::optional<int64_t> step_id_ TF_GUARDED_BY(execution_mutex_) =
      absl::nullopt;
  std::vector<TFE_TensorHandle*> op_inputs_ TF_GUARDED_BY(execution_mutex_);
  const TFE_OpAttrs* attributes_ TF_GUARDED_BY(execution_mutex_);
  int expected_max_outputs_ TF_GUARDED_BY(execution_mutex_);
  CancellationManager* cancellation_manager_ TF_GUARDED_BY(execution_mutex_);
  //   Outputs:
  std::vector<TensorHandlePtr> op_outputs_ TF_GUARDED_BY(execution_mutex_);
  // TF_Status is an incomplete type and so can't be stack allocated. To avoid
  // unnecessary allocations each Execute call, we keep one heap-allocated
  // version for the thread.
  StatusPtr status_ TF_GUARDED_BY(execution_mutex_);

  const std::string device_;
  ExecutorPtr executor_ TF_GUARDED_BY(execution_mutex_);
  mutable OpPtr op_ TF_GUARDED_BY(execution_mutex_);
  std::unique_ptr<Thread> thread_;
};

DeviceThread::~DeviceThread() {
  {
    tensorflow::mutex_lock l(execution_mutex_);
    execution_state_ = ExecutionState::kShuttingDown;
  }
  start_execute_.notify_one();
}

void DeviceThread::AsyncWait(TF_Status* status) {
  tensorflow::mutex_lock l(execution_mutex_);
  TFE_ExecutorWaitForAllPendingNodes(executor_.get(), status);
  TFE_ExecutorClearError(executor_.get());
}

void DeviceThread::Run() {
  while (true) {
    {
      tensorflow::mutex_lock l(execution_mutex_);
      while (execution_state_ == ExecutionState::kIdle ||
             execution_state_ == ExecutionState::kHasResult) {
        start_execute_.wait(l);
      }
      if (execution_state_ == ExecutionState::kShuttingDown) {
        return;
      } else if (execution_state_ == ExecutionState::kReadyToExecute) {
        // op_outputs_ may have been std::moved
        op_outputs_ = std::vector<TensorHandlePtr>();
        Execute(context_, operation_name_, std::move(op_inputs_), attributes_,
                expected_max_outputs_, &op_outputs_, status_.get());
        execution_state_ = ExecutionState::kHasResult;
      }
    }
    finished_execute_.notify_one();
  }
}

void DeviceThread::StartExecute(TFE_Context* context,
                                const char* operation_name,
                                std::vector<TFE_TensorHandle*> inputs,
                                const TFE_OpAttrs* attributes,
                                int expected_max_outputs,
                                CancellationManager& cancellation_manager,
                                absl::optional<int64_t> step_id) {
  {
    tensorflow::mutex_lock l(execution_mutex_);
    while (execution_state_ != ExecutionState::kIdle) {
      // If there's already a pending execution, wait until Join finishes before
      // starting on the next operation.
      finished_join_.wait(l);
    }
    context_ = context;
    operation_name_ = operation_name;
    step_id_ = step_id;
    op_inputs_ = inputs;
    attributes_ = attributes;
    expected_max_outputs_ = expected_max_outputs;
    cancellation_manager_ = &cancellation_manager;
    execution_state_ = ExecutionState::kReadyToExecute;
  }
  start_execute_.notify_one();
}

std::vector<TensorHandlePtr> DeviceThread::Join(TF_Status* status) {
  std::vector<TensorHandlePtr> result;
  {
    tensorflow::mutex_lock l(execution_mutex_);
    while (execution_state_ != ExecutionState::kHasResult) {
      finished_execute_.wait(l);
    }
    if (TF_GetCode(status_.get()) != TF_OK) {
      TF_SetStatus(status, TF_GetCode(status_.get()),
                   TF_Message(status_.get()));
      // Reset the member `status_` so future op executions (after recovery from
      // the bad `status`) start with an OK status.
      TF_SetStatus(status_.get(), TF_OK, "");
    }
    cancellation_manager_ = nullptr;
    execution_state_ = ExecutionState::kIdle;
    result = std::move(op_outputs_);
  }
  finished_join_.notify_one();
  return result;
}

void DeviceThread::Execute(TFE_Context* context, const char* operation_name,
                           std::vector<TFE_TensorHandle*> inputs,
                           const TFE_OpAttrs* attributes,
                           int expected_max_outputs,
                           std::vector<TensorHandlePtr>* outputs,
                           TF_Status* status) const {
  if (op_ == nullptr) {
    TFE_ContextSetExecutorForThread(context, executor_.get());
    op_.reset(TFE_NewOp(context, operation_name, status));
    if (TF_GetCode(status) != TF_OK) return;
    TFE_OpSetDevice(op_.get(), device_.c_str(), status);
    if (TF_GetCode(status) != TF_OK) return;
  } else {
    TFE_OpReset(op_.get(), operation_name, device_.c_str(), status);
    if (TF_GetCode(status) != TF_OK) return;
  }
  TFE_OpAddAttrs(op_.get(), attributes);
  for (int input_index = 0; input_index < inputs.size(); ++input_index) {
    TFE_OpAddInput(op_.get(), inputs[input_index], status);
    if (TF_GetCode(status) != TF_OK) return;
  }
  std::vector<TFE_TensorHandle*> unwrapped_results(expected_max_outputs);
  int real_num_outputs = expected_max_outputs;
  TFE_OpSetCancellationManager(op_.get(), wrap(cancellation_manager_), status);
  if (TF_GetCode(status) != TF_OK) return;

  // unwrap op_ and set step_id only if valid step id value was set.
  // Currently only required for non-TFRT use cases, e.g., EagerOp.
  if (step_id_.has_value()) {
    tensorflow::unwrap(op_.get())->SetStepId(step_id_.value());
  }

  TFE_Execute(op_.get(), unwrapped_results.data(), &real_num_outputs, status);
  if (TF_GetCode(status) != TF_OK) {
    cancellation_manager_->StartCancel();
    return;
  }
  unwrapped_results.resize(real_num_outputs);
  outputs->reserve(real_num_outputs);
  for (TFE_TensorHandle* unwrapped_result : unwrapped_results) {
    outputs->emplace_back(unwrapped_result);
  }
}

ParallelDevice::ParallelDevice(const std::vector<std::string>& devices,
                               bool is_async, int in_flight_nodes_limit)
    : underlying_devices_(devices),
      default_cancellation_manager_(absl::make_unique<CancellationManager>()) {
  device_threads_.reserve(devices.size());
  for (int device_index = 0; device_index < devices.size(); ++device_index) {
    device_threads_.emplace_back(new DeviceThread(
        devices[device_index].c_str(), is_async, in_flight_nodes_limit));
  }
}

// Necessary for a unique_ptr to a forward-declared type.
ParallelDevice::~ParallelDevice() = default;

std::unique_ptr<ParallelTensor> ParallelDevice::CopyToParallelDevice(
    TFE_Context* context, TFE_TensorHandle* tensor, TF_Status* status) const {
  std::vector<TensorHandlePtr> components;
  components.reserve(underlying_devices_.size());
  for (const std::string& underlying_device_name : underlying_devices_) {
    TFE_TensorHandle* t = TFE_TensorHandleCopyToDevice(
        tensor, context, underlying_device_name.c_str(), status);
    if (TF_GetCode(status) != TF_OK) return nullptr;
    components.emplace_back(t);
  }
  return ParallelTensor::FromTensorHandles(*this, std::move(components),
                                           status);
}

std::unique_ptr<ParallelTensor> ParallelDevice::DeviceIDs(
    TFE_Context* context, TF_Status* status) const {
  std::vector<int32_t> ids;
  ids.reserve(num_underlying_devices());
  for (int i = 0; i < num_underlying_devices(); ++i) {
    ids.push_back(i);
  }
  return ScalarsFromSequence<int32_t>(ids, context, status);
}

absl::optional<std::vector<std::unique_ptr<ParallelTensor>>>
ParallelDevice::Execute(TFE_Context* context,
                        const std::vector<ParallelTensor*>& inputs,
                        const char* operation_name,
                        const TFE_OpAttrs* attributes, int expected_max_outputs,
                        TF_Status* status) const {
  std::vector<PartialTensorShape> expected_output_shapes(expected_max_outputs);
  StartExecute(context, inputs, operation_name, attributes,
               expected_max_outputs, *default_cancellation_manager_);
  auto result = Join(expected_output_shapes, status);
  if (TF_GetCode(status) != TF_OK) {
    std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> await_status(
        TF_NewStatus(), TF_DeleteStatus);
    // Wait until all pending nodes have completed since they may have a
    // reference to default_cancellation_manager_. We ignore the status return
    // since we already have a bad status to propagate.
    TFE_ContextAsyncWait(context, await_status.get());
    // Reset the cancellation manager on a bad status. Otherwise we'll cancel
    // all future operations.
    default_cancellation_manager_ = absl::make_unique<CancellationManager>();
  }
  return result;
}

void ParallelDevice::StartExecute(TFE_Context* context,
                                  const std::vector<ParallelTensor*>& inputs,
                                  const char* operation_name,
                                  const TFE_OpAttrs* attributes,
                                  int expected_max_outputs,
                                  CancellationManager& cancellation_manager,
                                  absl::optional<int64_t> step_id) const {
  for (int device_index = 0; device_index < underlying_devices_.size();
       ++device_index) {
    DeviceThread* device_thread = device_threads_[device_index].get();
    std::vector<TFE_TensorHandle*> device_inputs;
    device_inputs.reserve(inputs.size());
    for (int input_index = 0; input_index < inputs.size(); ++input_index) {
      // Parallel tensors are divided between operations by device.
      device_inputs.push_back(inputs[input_index]->tensor(device_index));
    }
    device_thread->StartExecute(
        context, operation_name, std::move(device_inputs), attributes,
        expected_max_outputs, cancellation_manager, step_id);
  }
}

void ParallelDevice::StartExecute(
    TFE_Context* context,
    const std::vector<std::vector<TFE_TensorHandle*>>& inputs,
    const char* operation_name, const TFE_OpAttrs* attributes,
    int expected_max_outputs, CancellationManager& cancellation_manager,
    absl::optional<int64_t> step_id) const {
  for (int device_index = 0; device_index < underlying_devices_.size();
       ++device_index) {
    DeviceThread* device_thread = device_threads_[device_index].get();
    std::vector<TFE_TensorHandle*> device_inputs;
    device_inputs.reserve(inputs.size());
    for (int input_index = 0; input_index < inputs.size(); ++input_index) {
      // Parallel tensors are divided between operations by device.
      device_inputs.push_back(inputs[input_index][device_index]);
    }
    device_thread->StartExecute(
        context, operation_name, std::move(device_inputs), attributes,
        expected_max_outputs, cancellation_manager, step_id);
  }
}

void ParallelDevice::AsyncWait(TFE_Context* context, TF_Status* status) const {
  StatusPtr first_bad_status(nullptr);

  for (const auto& dt : device_threads_) {
    StatusPtr async_wait_status(TF_NewStatus());
    dt->AsyncWait(async_wait_status.get());
    // Prefer non cancelled errors to uncover real failures.
    if (TF_GetCode(async_wait_status.get()) != TF_OK &&
        (first_bad_status == nullptr ||
         TF_GetCode(first_bad_status.get()) == TF_CANCELLED)) {
      first_bad_status.reset(TF_NewStatus());
      TF_SetStatus(first_bad_status.get(), TF_GetCode(async_wait_status.get()),
                   TF_Message(async_wait_status.get()));
    }
  }

  if (first_bad_status != nullptr) {
    TF_SetStatus(status, TF_GetCode(first_bad_status.get()),
                 TF_Message(first_bad_status.get()));
  }
}

absl::optional<std::vector<std::unique_ptr<ParallelTensor>>>
ParallelDevice::Join(
    const std::vector<PartialTensorShape>& expected_output_shapes,
    TF_Status* status) const {
  absl::optional<std::vector<std::unique_ptr<ParallelTensor>>> result;
  // Compute per-device per-output tensors
  std::vector<std::vector<TensorHandlePtr>> per_device_output_tensors;
  per_device_output_tensors.reserve(underlying_devices_.size());
  int first_op_output_count = 0;
  StatusPtr first_bad_status(nullptr);
  for (int device_index = 0; device_index < underlying_devices_.size();
       ++device_index) {
    DeviceThread* device_thread = device_threads_[device_index].get();
    per_device_output_tensors.push_back(device_thread->Join(status));
    // We will run every Join even if there are bad statuses in case the user
    // wants to recover and continue running ops on the parallel device (which
    // would otherwise deadlock).
    if (TF_GetCode(status) != TF_OK &&
        (first_bad_status == nullptr
         // Prefer propagating non-cancellation related statuses to avoid
         // shadowing the original failure.
         || TF_GetCode(first_bad_status.get()) == TF_CANCELLED)) {
      first_bad_status.reset(TF_NewStatus());
      TF_SetStatus(first_bad_status.get(), TF_GetCode(status),
                   TF_Message(status));
    }

    if (device_index == 0) {
      first_op_output_count = per_device_output_tensors.rbegin()->size();
    } else {
      if (first_bad_status == nullptr &&
          per_device_output_tensors.rbegin()->size() != first_op_output_count) {
        first_bad_status.reset(TF_NewStatus());
        TF_SetStatus(first_bad_status.get(), TF_INTERNAL,
                     "Parallel ops produced different numbers of tensors.");
      }
    }
  }
  if (first_bad_status != nullptr) {
    TF_SetStatus(status, TF_GetCode(first_bad_status.get()),
                 TF_Message(first_bad_status.get()));
    return result;
  }
  // For each output of the original operation, pack the per-device
  // TensorHandles we've computed into a single parallel TensorHandle.
  std::vector<std::unique_ptr<ParallelTensor>> per_device_outputs;
  per_device_outputs.reserve(first_op_output_count);
  for (int i = 0; i < first_op_output_count; ++i) {
    std::vector<TensorHandlePtr> components;
    components.reserve(underlying_devices_.size());
    for (int j = 0; j < underlying_devices_.size(); ++j) {
      components.push_back(std::move(per_device_output_tensors[j][i]));
    }
    if (expected_output_shapes[i].IsFullyDefined()) {
      per_device_outputs.push_back(ParallelTensor::FromTensorHandles(
          *this, std::move(components),
          absl::Span<const int64_t>(expected_output_shapes[i].dim_sizes()),
          status));
    } else {
      per_device_outputs.push_back(ParallelTensor::FromTensorHandles(
          *this, std::move(components), status));
    }
    if (TF_GetCode(status) != TF_OK) return result;
  }
  result.emplace(std::move(per_device_outputs));
  return result;
}

std::vector<std::string> ParallelDevice::SummarizeDeviceNames() const {
  std::vector<DeviceNameUtils::ParsedName> parsed_components(
      underlying_devices_.size());
  for (int component_index = 0; component_index < underlying_devices_.size();
       ++component_index) {
    if (!DeviceNameUtils::ParseFullName(underlying_devices_[component_index],
                                        &parsed_components[component_index]) ||
        !DeviceNameUtils::IsSameAddressSpace(
            underlying_devices_[component_index], underlying_devices_[0])) {
      // Device names are from different address spaces, or we can't figure out
      // whether they are, so we'll fully-qualify everything.
      return underlying_devices_;
    }
  }
  std::vector<std::string> local_names;
  local_names.reserve(underlying_devices_.size());
  for (const DeviceNameUtils::ParsedName& parsed_component :
       parsed_components) {
    local_names.push_back(
        absl::StrCat(parsed_component.type, ":", parsed_component.id));
  }
  return local_names;
}

std::unique_ptr<ParallelTensor> ParallelTensor::FromTensorHandles(
    const ParallelDevice& parallel_device,
    std::vector<TensorHandlePtr> components, absl::Span<const int64_t> shape,
    TF_Status* status) {
  if (components.empty()) {
    TF_SetStatus(status, TF_INTERNAL,
                 "No components are provide for creating a ParallelTensor");
    return nullptr;
  }
  TFE_TensorHandleGetStatus(components[0].get(), status);
  if (!status->status.ok()) {
    return nullptr;
  }

  TF_DataType dtype = TFE_TensorHandleDataType(components[0].get());
  // Verify that the TensorHandle's shape and dtype match all of the component
  // shapes and dtypes.
  for (TensorHandlePtr& component : components) {
    TFE_TensorHandleGetStatus(component.get(), status);
    if (!status->status.ok()) {
      return nullptr;
    }
    if (TFE_TensorHandleDataType(component.get()) != dtype) {
      TF_SetStatus(status, TF_INTERNAL,
                   "Components of a ParallelTensor must all have "
                   "the same dtype");
      return nullptr;
    }
  }
  return std::unique_ptr<ParallelTensor>(
      new ParallelTensor(parallel_device, std::move(components), shape, dtype));
}

std::unique_ptr<ParallelTensor> ParallelTensor::FromTensorHandles(
    const ParallelDevice& parallel_device,
    std::vector<TensorHandlePtr> components, TF_Status* status) {
  if (components.empty()) {
    TF_SetStatus(status, TF_INTERNAL,
                 "No components are provided for creating a ParallelTensor");
    return nullptr;
  }
  TFE_TensorHandleGetStatus(components[0].get(), status);
  if (!status->status.ok()) {
    return nullptr;
  }

  TF_DataType dtype = TFE_TensorHandleDataType(components[0].get());
  // Verify that the combined TensorHandle's dtype matches all of the component
  // dtypes.
  for (TensorHandlePtr& component : components) {
    TFE_TensorHandleGetStatus(component.get(), status);
    if (!status->status.ok()) {
      return nullptr;
    }
    if (TFE_TensorHandleDataType(component.get()) != dtype) {
      TF_SetStatus(status, TF_INTERNAL,
                   "Components of a ParallelTensor must all have "
                   "the same dtype");
      return nullptr;
    }
  }
  return std::unique_ptr<ParallelTensor>(
      new ParallelTensor(parallel_device, std::move(components), dtype));
}

Status ParallelTensor::Shape(const std::vector<int64_t>** shape) const {
  if (!shape_.has_value()) {
    TF_Status status;
    PartialTensorShape combined_shape;
    TF_RETURN_IF_ERROR(unwrap(tensors_[0].get())->Shape(&combined_shape));

    for (const TensorHandlePtr& component : tensors_) {
      PartialTensorShape component_shape;
      TF_RETURN_IF_ERROR(unwrap(component.get())->Shape(&component_shape));
      if (combined_shape.dims() < 0 ||
          combined_shape.dims() != component_shape.dims()) {
        PartialTensorShape first_shape;
        TF_RETURN_IF_ERROR(unwrap(tensors_[0].get())->Shape(&first_shape));
        return errors::Unimplemented(absl::StrCat(
            "Computing the shape of a ParallelTensor when the components do "
            "not all have the same rank is not supported. One tensor had "
            "shape ",
            first_shape.DebugString(), " and another had shape ",
            component_shape.DebugString()));
      } else {
        // Generalize differing axis lengths to "variable"/"unknown".
        for (int axis_index = 0; axis_index < combined_shape.dims();
             ++axis_index) {
          int64_t axis_length = combined_shape.dim_size(axis_index);
          if (axis_length != component_shape.dim_size(axis_index)) {
            axis_length = -1;
          }
          TF_RETURN_IF_ERROR(
              combined_shape.SetDimWithStatus(axis_index, axis_length));
        }
      }
    }
    auto dim_sizes = combined_shape.dim_sizes();
    shape_ = std::vector<int64_t>(dim_sizes.begin(), dim_sizes.end());
  }
  *shape = &*shape_;
  return absl::OkStatus();
}

Status ParallelTensor::SummarizeValue(std::string& summary) {
  summary = "{";
  std::vector<std::string> summarized_devices = device_.SummarizeDeviceNames();
  for (int component_index = 0; component_index < tensors_.size();
       ++component_index) {
    // TODO(allenl): Add a C API for summarizing tensors. Currently custom
    // devices limiting themselves to a C API (for ABI compatibility) would need
    // to implement summarization for component tensors themselves.
    ImmediateExecutionTensorHandle* component =
        tensorflow::unwrap(tensors_[component_index].get());
    std::string component_summary;
    TF_RETURN_IF_ERROR(component->SummarizeValue(component_summary));
    absl::StrAppend(&summary, component_index == 0 ? "" : ", ", "\"",
                    summarized_devices[component_index],
                    "\": ", component_summary);
  }
  summary += "}";
  return absl::OkStatus();
}

}  // namespace parallel_device
}  // namespace tensorflow
