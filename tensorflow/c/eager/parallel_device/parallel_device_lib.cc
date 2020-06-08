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

namespace tensorflow {
namespace parallel_device {
namespace {

class OpDeleter {
 public:
  void operator()(TFE_Op* to_delete) const { TFE_DeleteOp(to_delete); }
};

using OpPtr = std::unique_ptr<TFE_Op, OpDeleter>;

// Creates a vector of `count` new executors (threads).
std::vector<ExecutorPtr> MakeExecutors(size_t count) {
  std::vector<ExecutorPtr> executors;
  executors.reserve(count);
  for (int i = 0; i < count; ++i) {
    executors.emplace_back(TFE_NewExecutor(true /* is_async */));
  }
  return executors;
}

}  // namespace

ParallelDevice::ParallelDevice(const std::vector<std::string>& devices)
    : underlying_devices_(devices),
      executors_(MakeExecutors(underlying_devices_.size())) {}

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
                        std::vector<MaybeParallelTensorUnowned> inputs,
                        const char* operation_name,
                        const TFE_OpAttrs* attributes, int expected_max_outputs,
                        TF_Status* status) const {
  absl::optional<std::vector<std::unique_ptr<ParallelTensor>>> result;
  // Compute per-device per-output tensors
  std::vector<std::vector<TensorHandlePtr>> per_device_output_tensors;
  per_device_output_tensors.reserve(underlying_devices_.size());
  // TODO(allenl): Add a TFE_ExecuteWithExecutor API so we don't have to keep
  // setting the thread-local executor like this.
  TFE_Executor* previous_executor(TFE_ContextGetExecutorForThread(context));
  auto reset_executor =
      tensorflow::gtl::MakeCleanup([context, previous_executor]() {
        TFE_ContextSetExecutorForThread(context, previous_executor);
        TFE_DeleteExecutor(previous_executor);
      });
  int first_op_output_count;
  for (int device_index = 0; device_index < underlying_devices_.size();
       ++device_index) {
    TFE_Executor* executor = executors_[device_index].get();
    // Note that the `reset_executor` cleanup sets the thread's executor back to
    // the value before this function ran.
    TFE_ContextSetExecutorForThread(context, executor);
    OpPtr op(TFE_NewOp(context, operation_name, status));
    if (TF_GetCode(status) != TF_OK) return result;
    TFE_OpSetDevice(op.get(), underlying_devices_[device_index].c_str(),
                    status);
    TFE_OpAddAttrs(op.get(), attributes);
    for (int input_index = 0; input_index < inputs.size(); ++input_index) {
      if (absl::holds_alternative<TFE_TensorHandle*>(inputs[input_index])) {
        // Non-parallel tensors are implicitly broadcast, i.e. set as the input
        // to each parallel operation.
        //
        // TODO(allenl): There may be smarter ways to do this copy in some
        // cases, i.e. with a collective broadcast. We'll need to be careful
        // about things that are taken as inputs on the host or on their
        // existing device (for multi-device functions).
        TFE_OpAddInput(op.get(),
                       absl::get<TFE_TensorHandle*>(inputs[input_index]),
                       status);
        if (TF_GetCode(status) != TF_OK) return result;
      } else {
        // Parallel tensors are divided between operations by device.
        TFE_OpAddInput(op.get(),
                       absl::get<ParallelTensor*>(inputs[input_index])
                           ->tensor(device_index),
                       status);
        if (TF_GetCode(status) != TF_OK) return result;
      }
    }
    std::vector<TFE_TensorHandle*> op_outputs(expected_max_outputs);
    int real_num_outputs = expected_max_outputs;
    // For nested devices, the inner device sees the async executor we've
    // set. Inner parallel devices will just overwrite this with their own and
    // then set it back to ours before returning. This means parallel devices
    // which consist of several aliased parallel devices would hypothetically
    // deadlock if the outer parallel device ran one collective with a group
    // size equal to the total number of aliased physical devices. Currently
    // physical devices cannot participate in a single collective reduction
    // multiple times, so this would fail earlier.
    //
    // TODO(allenl): Keep a map from outer executor to list of inner executors
    // rather than a single list of executors so aliased nested parallel devices
    // don't re-use an executor.
    TFE_Execute(op.get(), op_outputs.data(), &real_num_outputs, status);
    if (device_index == 0) {
      first_op_output_count = real_num_outputs;
    } else {
      if (real_num_outputs != first_op_output_count) {
        TF_SetStatus(status, TF_INTERNAL,
                     "Parallel ops produced different numbers of tensors.");
        return result;
      }
    }
    if (TF_GetCode(status) != TF_OK) return result;
    std::vector<TensorHandlePtr> this_outputs;
    this_outputs.reserve(real_num_outputs);
    for (int output_num = 0; output_num < real_num_outputs; ++output_num) {
      this_outputs.emplace_back(op_outputs[output_num]);
    }
    per_device_output_tensors.push_back(std::move(this_outputs));
  }
  for (int device_index = 0; device_index < underlying_devices_.size();
       ++device_index) {
    TFE_Executor* executor = executors_[device_index].get();
    // TODO(b/157523095): Syncing the executor here shouldn't be
    // necessary. Currently async+remote is missing cross-executor
    // coordination.
    TFE_ExecutorWaitForAllPendingNodes(executor, status);
    if (TF_GetCode(status) != TF_OK) return result;
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
