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

#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"

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
  explicit DeviceThread(const std::string& device)
      : status_(TF_NewStatus()),
        device_(device),
        op_(nullptr),
        thread_(tensorflow::Env::Default()->StartThread(
            tensorflow::ThreadOptions(), "parallel_device_execute",
            std::bind(&DeviceThread::Run, this))) {}
  ~DeviceThread();

  // Requests that the worker thread execute the specified operation. Blocks
  // until the previously pending operation (a StartExecute without a Join) has
  // finished, if any.
  void StartExecute(TFE_Context* context, const char* operation_name,
                    std::vector<TFE_TensorHandle*> inputs,
                    const TFE_OpAttrs* attributes, int expected_max_outputs);
  // Block until the previous `StartExecute` operation has executed. Forwards
  // the status from `TFE_Execute` and returns outputs if the status is OK.
  std::vector<TensorHandlePtr> Join(TF_Status* status);

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
  //   Inputs
  TFE_Context* context_ TF_GUARDED_BY(execution_mutex_);
  const char* operation_name_ TF_GUARDED_BY(execution_mutex_);
  std::vector<TFE_TensorHandle*> op_inputs_ TF_GUARDED_BY(execution_mutex_);
  const TFE_OpAttrs* attributes_ TF_GUARDED_BY(execution_mutex_);
  int expected_max_outputs_ TF_GUARDED_BY(execution_mutex_);
  //   Outputs
  std::vector<TensorHandlePtr> op_outputs_ TF_GUARDED_BY(execution_mutex_);
  StatusPtr status_ TF_GUARDED_BY(execution_mutex_);

  const std::string device_;
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
                                int expected_max_outputs) {
  {
    tensorflow::mutex_lock l(execution_mutex_);
    while (execution_state_ != ExecutionState::kIdle) {
      // If there's already a pending execution, wait until Join finishes before
      // starting on the next operation.
      finished_join_.wait(l);
    }
    context_ = context;
    operation_name_ = operation_name;
    op_inputs_ = inputs;
    attributes_ = attributes;
    expected_max_outputs_ = expected_max_outputs;
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
    }
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
  if (TF_GetCode(status) != TF_OK) return;
  TFE_Execute(op_.get(), unwrapped_results.data(), &real_num_outputs, status);
  if (TF_GetCode(status) != TF_OK) return;
  unwrapped_results.resize(real_num_outputs);
  outputs->reserve(real_num_outputs);
  for (TFE_TensorHandle* unwrapped_result : unwrapped_results) {
    outputs->emplace_back(unwrapped_result);
  }
}

ParallelDevice::ParallelDevice(const std::vector<std::string>& devices)
    : underlying_devices_(devices) {
  device_threads_.reserve(devices.size());
  for (int device_index = 0; device_index < devices.size(); ++device_index) {
    device_threads_.emplace_back(
        new DeviceThread(devices[device_index].c_str()));
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
  // TODO(allenl): We could cache DeviceIDs (keyed by context).
  std::vector<TensorHandlePtr> components;
  components.reserve(underlying_devices_.size());
  for (int device_index = 0; device_index < underlying_devices_.size();
       ++device_index) {
    int64_t* device_id = new int64_t;
    *device_id = device_index;
    std::unique_ptr<TF_Tensor, decltype(&TF_DeleteTensor)> tensor(
        TF_NewTensor(
            TF_INT64, /*dims=*/nullptr, /*num_dims=*/0, device_id,
            sizeof(int64_t),
            [](void* data, size_t, void* arg) {
              delete reinterpret_cast<int64_t*>(data);
            },
            nullptr),
        TF_DeleteTensor);
    // TODO(allenl): Here and when executing regular operations, we could hold
    // on to one TFE_Op per device and just call TFE_ResetOp to avoid parsing
    // device names repeatedly.
    OpPtr const_op(TFE_NewOp(context, "Const", status));
    if (TF_GetCode(status) != TF_OK) return nullptr;
    TFE_OpSetDevice(const_op.get(), underlying_devices_[device_index].c_str(),
                    status);
    if (TF_GetCode(status) != TF_OK) return nullptr;
    TFE_OpSetAttrTensor(const_op.get(), "value", tensor.get(), status);
    if (TF_GetCode(status) != TF_OK) return nullptr;
    TFE_OpSetAttrType(const_op.get(), "dtype", TF_INT64);
    TFE_TensorHandle* device_handle;
    int num_outputs = 1;
    TFE_Execute(const_op.get(), &device_handle, &num_outputs, status);
    if (TF_GetCode(status) != TF_OK) return nullptr;
    components.emplace_back(device_handle);
    if (TF_GetCode(status) != TF_OK) return nullptr;
  }
  return ParallelTensor::FromTensorHandles(*this, std::move(components),
                                           status);
}

absl::optional<std::vector<std::unique_ptr<ParallelTensor>>>
ParallelDevice::Execute(TFE_Context* context,
                        const std::vector<ParallelTensor*>& inputs,
                        const char* operation_name,
                        const TFE_OpAttrs* attributes, int expected_max_outputs,
                        TF_Status* status) const {
  absl::optional<std::vector<std::unique_ptr<ParallelTensor>>> result;
  // Compute per-device per-output tensors
  std::vector<std::vector<TensorHandlePtr>> per_device_output_tensors;
  per_device_output_tensors.reserve(underlying_devices_.size());
  int first_op_output_count = 0;
  for (int device_index = 0; device_index < underlying_devices_.size();
       ++device_index) {
    DeviceThread* device_thread = device_threads_[device_index].get();
    std::vector<TFE_TensorHandle*> device_inputs;
    device_inputs.reserve(device_inputs.size());
    for (int input_index = 0; input_index < inputs.size(); ++input_index) {
      // Parallel tensors are divided between operations by device.
      device_inputs.push_back(inputs[input_index]->tensor(device_index));
    }
    device_thread->StartExecute(context, operation_name,
                                std::move(device_inputs), attributes,
                                expected_max_outputs);
  }
  for (int device_index = 0; device_index < underlying_devices_.size();
       ++device_index) {
    DeviceThread* device_thread = device_threads_[device_index].get();
    per_device_output_tensors.push_back(device_thread->Join(status));
    if (TF_GetCode(status) != TF_OK) return result;
    if (device_index == 0) {
      first_op_output_count = per_device_output_tensors.rbegin()->size();
    } else {
      if (per_device_output_tensors.rbegin()->size() != first_op_output_count) {
        TF_SetStatus(status, TF_INTERNAL,
                     "Parallel ops produced different numbers of tensors.");
        return result;
      }
    }
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
    per_device_outputs.push_back(ParallelTensor::FromTensorHandles(
        *this, std::move(components), status));
    if (TF_GetCode(status) != TF_OK) return result;
  }
  result.emplace(std::move(per_device_outputs));
  return result;
}

std::unique_ptr<ParallelTensor> ParallelTensor::FromTensorHandles(
    const ParallelDevice& parallel_device,
    std::vector<TensorHandlePtr> components, TF_Status* status) {
  TF_DataType dtype = TFE_TensorHandleDataType(components[0].get());
  std::vector<int64_t> shape(
      TFE_TensorHandleNumDims(components[0].get(), status));
  if (TF_GetCode(status) != TF_OK) return nullptr;
  for (int i = 0; i < shape.size(); ++i) {
    shape[i] = TFE_TensorHandleDim(components[0].get(), i, status);
    if (TF_GetCode(status) != TF_OK) return nullptr;
  }

  // Verify that the TensorHandle's shape and dtype match all of the component
  // shapes and dtypes.
  for (TensorHandlePtr& component : components) {
    for (int i = 0; i < shape.size(); ++i) {
      int64_t tensor_dim = TFE_TensorHandleDim(component.get(), i, status);
      if (TF_GetCode(status) != TF_OK) return nullptr;
      if (tensor_dim != shape[i]) {
        // TODO(allenl): Allow shapes to differ.
        TF_SetStatus(status, TF_UNIMPLEMENTED,
                     "Components of a ParallelTensor must currently all have "
                     "the same shape");
        return nullptr;
      }
      if (TFE_TensorHandleDataType(component.get()) != dtype) {
        TF_SetStatus(status, TF_INTERNAL,
                     "Components of a ParallelTensor must all have "
                     "the same dtype");
        return nullptr;
      }
    }
  }

  return std::unique_ptr<ParallelTensor>(new ParallelTensor(
      parallel_device, std::move(components), std::move(shape), dtype));
}

}  // namespace parallel_device
}  // namespace tensorflow
