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

#include "tensorflow/c/eager/parallel_device/parallel_device.h"

#include <memory>

#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "absl/types/variant.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/core/lib/gtl/cleanup.h"

namespace tensorflow {
namespace eager {
namespace {

// Functor for making unique_ptrs slightly more ergonomic. Using
// decltype(delete_fn) in the unique_ptr's second template argument requires
// passing a function pointer to delete_fn when constructing the unique_ptr.
class TensorHandleDeleter {
 public:
  void operator()(TFE_TensorHandle* to_delete) const {
    TFE_DeleteTensorHandle(to_delete);
  }
};

using TensorHandlePtr = std::unique_ptr<TFE_TensorHandle, TensorHandleDeleter>;

class OpDeleter {
 public:
  void operator()(TFE_Op* to_delete) const { TFE_DeleteOp(to_delete); }
};

using OpPtr = std::unique_ptr<TFE_Op, OpDeleter>;

class ExecutorDeleter {
 public:
  void operator()(TFE_Executor* to_delete) const {
    TFE_DeleteExecutor(to_delete);
  }
};

using ExecutorPtr = std::unique_ptr<TFE_Executor, ExecutorDeleter>;

class ParallelTensor;

using MaybeParallelTensorOwned =
    absl::variant<std::unique_ptr<ParallelTensor>, TensorHandlePtr>;
using MaybeParallelTensorUnowned =
    absl::variant<ParallelTensor*, TFE_TensorHandle*>;

// Creates a vector of `count` new executors (threads).
std::vector<ExecutorPtr> MakeExecutors(size_t count) {
  std::vector<ExecutorPtr> executors;
  executors.reserve(count);
  for (int i = 0; i < count; ++i) {
    executors.emplace_back(TFE_NewExecutor(true /* is_async */));
  }
  return executors;
}

// A representation of the custom device passed in and out of the TFE custom
// device APIs, providing context about the parallel device to
// ParallelDeviceExecute.
class ParallelDevice {
 public:
  ParallelDevice(const std::string& name,
                 const std::vector<std::string>& devices);

  // Helper to copy a tensor handle from another device once for each component
  // of the ParallelDevice.
  //
  // Sets a bad status and returns a nullptr if `tensor` is already on the
  // ParallelDevice, or if the individual copies fail.
  std::unique_ptr<ParallelTensor> CopyToParallelDevice(TFE_Context* context,
                                                       TFE_TensorHandle* tensor,
                                                       TF_Status* status) const;

  // A parallel tensor with scalar integers numbering component devices.
  std::unique_ptr<ParallelTensor> DeviceIDs(TFE_Context* context,
                                            TF_Status* status) const;

  // Takes a description of a single operation being executed on the
  // ParallelDevice, and in turn runs one operation per component device with
  // its corresponding inputs from the input ParallelTensors (or
  // implicitly-mirrored tensors on other devices). Wraps the resulting
  // per-device and per-output TFE_TensorHandles into one ParallelTensor per
  // output of the original operation.
  //
  // `inputs` are either ParallelTensors, i.e. already on the ParallelDevice, or
  // un-replicated TFE_TensorHandles on other devices. TPUReplicatedInput
  // requires non-parallel tensors, and TPUReplicatedOutput requires a parallel
  // tensor, but other operations will implicitly broadcast non-parallel input
  // tensors across the ParallelDevice's component devices.
  //
  // Two special-cased operations, TPUReplicatedInput and TPUReplicatedOutput,
  // pack and un-pack parallel tensors respectively. Only TPUReplicatedOutput
  // causes `Execute` to return non-parallel tensors.
  //
  // Attributes are forwarded to executed operations unmodified.
  //
  // The returned optional has a value if and only if `status` evaluates to
  // TF_OK.
  absl::optional<std::vector<MaybeParallelTensorOwned>> Execute(
      TFE_Context* context, std::vector<MaybeParallelTensorUnowned> inputs,
      const char* operation_name, const TFE_OpAttrs* attributes,
      int expected_max_outputs, TF_Status* status) const;

  // Implements the parallel case for `Execute`, where all of the outputs of the
  // operation are ParallelTensors, and all inputs are either ParallelTensors or
  // should be implicitly broadcast. This means the operation is not
  // TPUReplicatedInput or TPUReplicatedOutput.
  //
  // The returned optional has a value if and only if `status` evaluates to
  // TF_OK. Bad statuses are forwarded from underlying `TFE_Execute` calls, or
  // if sanity checks on dtypes/metadata fail.
  absl::optional<std::vector<std::unique_ptr<ParallelTensor>>>
  ExecuteParallelOperation(TFE_Context* context,
                           std::vector<MaybeParallelTensorUnowned> inputs,
                           const char* operation_name,
                           const TFE_OpAttrs* attributes,
                           int expected_max_outputs, TF_Status* status) const;

  const std::string& device_name() const { return device_name_; }

 private:
  // The name of the parallel device
  // (e.g. "/job:localhost/replica:0/task:0/device:CUSTOM:0")
  const std::string device_name_;
  // A sequence of device names, indicating which devices replicated operations
  // are forwarded to.
  const std::vector<std::string> underlying_devices_;
  // A sequence of TFE_Executors, one per device, for executing operations in
  // parallel.
  const std::vector<ExecutorPtr> executors_;
};

// The internal representation of a TFE_TensorHandle placed on a
// ParallelDevice. Contains a tuple of tensors, one on each of the
// `underlying_devices_` of the ParallelDevice.
class ParallelTensor {
 public:
  // Construct a ParallelTensor from TensorHandles placed on the component
  // devices of a ParallelDevice.
  static std::unique_ptr<ParallelTensor> FromTensorHandles(
      const ParallelDevice& parallel_device,
      std::vector<TensorHandlePtr> components, TF_Status* status);

  // Helper to wrap a ParallelTensor into a TFE_TensorHandle which contains it.
  static TensorHandlePtr AsTensorHandle(TFE_Context* context,
                                        std::unique_ptr<ParallelTensor> t,
                                        TF_Status* status);

  size_t num_tensors() const { return tensors_.size(); }
  TFE_TensorHandle* tensor(size_t index) const { return tensors_[index].get(); }

 private:
  ParallelTensor(const ParallelDevice& device,
                 std::vector<TensorHandlePtr> tensors,
                 std::vector<int64_t> shape, const TF_DataType dtype)
      : device_(device),
        tensors_(std::move(tensors)),
        shape_(std::move(shape)),
        dtype_(dtype) {}

  const ParallelDevice& device_;
  const std::vector<TensorHandlePtr> tensors_;
  const std::vector<int64_t> shape_;
  const TF_DataType dtype_;
};

ParallelDevice::ParallelDevice(const std::string& name,
                               const std::vector<std::string>& devices)
    : device_name_(name),
      underlying_devices_(devices),
      executors_(MakeExecutors(underlying_devices_.size())) {}

std::unique_ptr<ParallelTensor> ParallelDevice::CopyToParallelDevice(
    TFE_Context* context, TFE_TensorHandle* tensor, TF_Status* status) const {
  const char* current_device = TFE_TensorHandleDeviceName(tensor, status);
  if (device_name_ == current_device) {
    std::string message(absl::StrCat(
        "Tried to copy a TensorHandle to its existing device: ", device_name_));
    TF_SetStatus(status, TF_INTERNAL, message.c_str());
    return nullptr;
  }
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

absl::optional<std::vector<MaybeParallelTensorOwned>> ParallelDevice::Execute(
    TFE_Context* context, std::vector<MaybeParallelTensorUnowned> inputs,
    const char* operation_name, const TFE_OpAttrs* attributes,
    int expected_max_outputs, TF_Status* status) const {
  absl::optional<std::vector<MaybeParallelTensorOwned>> result;
  // TODO(allenl): We should remove "TPU" from these op names at the very least,
  // or consider other ways of packing/unpacking parallel tensors.
  if (operation_name == std::string("TPUReplicatedInput")) {
    // Special-cased operation for packing per-device tensors into one parallel
    // tensor.
    if (inputs.size() != underlying_devices_.size()) {
      std::string message(absl::StrCat(
          "The parallel device ", device_name_, " expected ",
          underlying_devices_.size(), " inputs to TPUReplicatedInput, but got ",
          inputs.size()));
      TF_SetStatus(status, TF_INVALID_ARGUMENT, message.c_str());
      return result;
    }
    std::vector<TensorHandlePtr> components;
    components.reserve(inputs.size());
    for (int i = 0; i < inputs.size(); ++i) {
      if (absl::holds_alternative<ParallelTensor*>(inputs[i])) {
        std::string message(absl::StrCat(
            "Expected all inputs to TPUReplicatedInput to be non-parallel "
            "TensorHandles. The input ",
            i,
            " was a parallel tensor (already "
            "placed on the parallel device)."));
        TF_SetStatus(status, TF_INVALID_ARGUMENT, message.c_str());
        return result;
      }
      components.emplace_back(TFE_TensorHandleCopySharingTensor(
          absl::get<TFE_TensorHandle*>(inputs[i]), status));
    }
    std::vector<MaybeParallelTensorOwned> result_content;
    result_content.reserve(1);
    result_content.push_back(ParallelTensor::FromTensorHandles(
        *this, std::move(components), status));
    if (TF_GetCode(status) != TF_OK) return result;
    result.emplace(std::move(result_content));
    return result;
  } else if (operation_name == std::string("TPUReplicatedOutput")) {
    // Special-cased operation for un-packing one parallel tensor into
    // per-device tensors.
    OpPtr op(TFE_NewOp(context, operation_name, status));
    TFE_OpAddAttrs(op.get(), attributes);
    int expected_outputs = TFE_OpGetOutputLength(op.get(), "outputs", status);
    if (TF_GetCode(status) != TF_OK) return result;
    if (expected_outputs != underlying_devices_.size()) {
      std::string message(absl::StrCat(
          "The parallel device ", device_name_, " expected ",
          underlying_devices_.size(),
          " outputs for TPUReplicatedOutput, but got ", expected_outputs));
      TF_SetStatus(status, TF_INVALID_ARGUMENT, message.c_str());
      return result;
    }
    if (absl::holds_alternative<TFE_TensorHandle*>(inputs[0])) {
      TF_SetStatus(status, TF_INVALID_ARGUMENT,
                   "Expected the input to "
                   "TPUReplicatedOutput to be a parallel tensor (placed on the "
                   "parallel device).");
      return result;
    }
    ParallelTensor* t = absl::get<ParallelTensor*>(inputs[0]);
    std::vector<MaybeParallelTensorOwned> outputs;
    outputs.reserve(t->num_tensors());
    for (int i = 0; i < t->num_tensors(); ++i) {
      // TODO(b/157523095): Syncing the executor here shouldn't be
      // necessary. Currently async+remote is missing cross-executor
      // coordination.
      TFE_ExecutorWaitForAllPendingNodes(executors_[i].get(), status);
      if (TF_GetCode(status) != TF_OK) return result;
      TensorHandlePtr this_output(
          TFE_TensorHandleCopySharingTensor(t->tensor(i), status));
      outputs.emplace_back(std::move(this_output));
      if (TF_GetCode(status) != TF_OK) return result;
    }
    result.emplace(std::move(outputs));
    return result;
  } else if (operation_name == std::string("DeviceID")) {
    std::vector<MaybeParallelTensorOwned> result_content;
    result_content.reserve(1);
    result_content.push_back(DeviceIDs(context, status));
    if (TF_GetCode(status) != TF_OK) return result;
    result.emplace(std::move(result_content));
    return result;
  }
  absl::optional<std::vector<std::unique_ptr<ParallelTensor>>>
      maybe_parallel_results(
          ExecuteParallelOperation(context, std::move(inputs), operation_name,
                                   attributes, expected_max_outputs, status));
  if (!maybe_parallel_results.has_value()) return result;
  std::vector<std::unique_ptr<ParallelTensor>> parallel_results(
      std::move(maybe_parallel_results.value()));
  std::vector<MaybeParallelTensorOwned> result_content;
  result_content.reserve(parallel_results.size());
  for (std::unique_ptr<ParallelTensor>& parallel_result : parallel_results) {
    result_content.push_back(
        MaybeParallelTensorOwned(std::move(parallel_result)));
  }
  result.emplace(std::move(result_content));
  return result;
}

absl::optional<std::vector<std::unique_ptr<ParallelTensor>>>
ParallelDevice::ExecuteParallelOperation(
    TFE_Context* context, std::vector<MaybeParallelTensorUnowned> inputs,
    const char* operation_name, const TFE_OpAttrs* attributes,
    int expected_max_outputs, TF_Status* status) const {
  absl::optional<std::vector<std::unique_ptr<ParallelTensor>>> result;
  // Compute per-device per-output tensors
  std::vector<std::vector<TensorHandlePtr>> per_device_output_tensors;
  per_device_output_tensors.reserve(underlying_devices_.size());
  // TODO(allenl): Add a TFE_ExecuteWithExecutor API so we don't have to keep
  // setting the thread-local executor like this.
  TFE_Executor* previous_executor(TFE_ContextGetExecutorForThread(context));
  auto reset_executor = gtl::MakeCleanup([context, previous_executor]() {
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

// Used as an argument to TFE_NewTensorHandleFromDeviceMemory, indicating how
// ParallelTensors wrapped in TFE_TensorHandles should be cleaned up once their
// reference counts drop to zero.
void ParallelTensorDeallocator(void* data, size_t len, void* arg) {
  delete reinterpret_cast<ParallelTensor*>(data);
}

TensorHandlePtr ParallelTensor::AsTensorHandle(
    TFE_Context* context, std::unique_ptr<ParallelTensor> t,
    TF_Status* status) {
  // The resulting TensorHandle owns an opaque pointer to "device memory", which
  // for a ParallelDevice is really a ParallelTensor. When the TensorHandle is
  // deleted, it will call ParallelTensorDeallocator to free the struct.
  ParallelTensor* t_released = t.release();
  return TensorHandlePtr(TFE_NewTensorHandleFromDeviceMemory(
      context, t_released->device_.device_name().c_str(), t_released->dtype_,
      t_released->shape_.data(), t_released->shape_.size(), t_released, 1,
      &ParallelTensorDeallocator, nullptr, status));
}

// For TFE_CustomDevice::copy_tensor_to_device in the parallel device
// registration.
//
// Replicates a single TFE_TensorHandle, producing a TFE_TensorHandle containing
// a ParallelTensor with one copy of `tensor` for each device in the
// ParallelDevice.
//
// Since this function is used to satisfy the TFE_CustomDevice C API,
// device_info is passed in using a C-style generic. It must always be a
// ParallelDevice.
TFE_TensorHandle* CopyToParallelDevice(TFE_Context* context,
                                       TFE_TensorHandle* tensor,
                                       TF_Status* status, void* device_info) {
  ParallelDevice* dev = reinterpret_cast<ParallelDevice*>(device_info);
  std::unique_ptr<ParallelTensor> parallel_tensor(
      dev->CopyToParallelDevice(context, tensor, status));
  if (TF_GetCode(status) != TF_OK) return nullptr;
  return ParallelTensor::AsTensorHandle(context, std::move(parallel_tensor),
                                        status)
      .release();
}

// For TFE_CustomDevice::copy_tensor_from_device in the parallel device
// registration.
//
// Currently this is an error, and un-packing ParallelTensors must be performed
// explicitly by running a TPUReplicatedOutput operation on the parallel device.
//
// TODO(allenl): There are some use-cases that are only supported by copying to
// host at the moment (e.g. debug print on a tensor, .numpy(), etc.). We either
// need to return something here or address these use-cases one by one.
TFE_TensorHandle* CopyTensorFromParallelDevice(TFE_Context* context,
                                               TFE_TensorHandle* tensor,
                                               const char* target_device_name,
                                               TF_Status* status,
                                               void* device_info) {
  TF_SetStatus(status, TF_INTERNAL,
               "Trying to copy a tensor out of a parallel device.");
  return nullptr;
}

// For TFE_CustomDevice::execute in the parallel device registration.
//
// Since this function is used to satisfy the TFE_CustomDevice C API,
// device_info is passed in using a C-style generic. It must always be a
// ParallelDevice.
void ParallelDeviceExecute(TFE_Context* context, int num_inputs,
                           TFE_TensorHandle** inputs,
                           const char* operation_name,
                           const TFE_OpAttrs* attributes, int* num_outputs,
                           TFE_TensorHandle** outputs, TF_Status* status,
                           void* device_info) {
  ParallelDevice* dev = reinterpret_cast<ParallelDevice*>(device_info);
  std::vector<MaybeParallelTensorUnowned> typed_inputs;
  typed_inputs.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    const char* tensor_handle_device =
        TFE_TensorHandleDeviceName(inputs[i], status);
    if (TF_GetCode(status) != TF_OK) return;
    if (dev->device_name() == tensor_handle_device) {
      // We assume that any tensors already placed on this device are
      // ParallelTensors.
      typed_inputs.emplace_back(reinterpret_cast<ParallelTensor*>(
          TFE_TensorHandleDevicePointer(inputs[i], status)));
      if (TF_GetCode(status) != TF_OK) return;
    } else {
      typed_inputs.emplace_back(inputs[i]);
    }
  }

  absl::optional<std::vector<MaybeParallelTensorOwned>> maybe_typed_outputs(
      dev->Execute(context, std::move(typed_inputs), operation_name, attributes,
                   *num_outputs, status));
  if (TF_GetCode(status) != TF_OK) return;
  if (!maybe_typed_outputs.has_value()) {
    TF_SetStatus(status, TF_INTERNAL, "OK status but no value was returned.");
    return;
  }

  std::vector<MaybeParallelTensorOwned> typed_outputs(
      std::move(maybe_typed_outputs.value()));

  if (typed_outputs.size() > *num_outputs) {
    TF_SetStatus(status, TF_INTERNAL,
                 "The allocated output buffer was too small.");
    return;
  }

  for (int i = 0; i < typed_outputs.size(); ++i) {
    MaybeParallelTensorOwned typed_output(std::move(typed_outputs[i]));
    if (absl::holds_alternative<TensorHandlePtr>(typed_output)) {
      outputs[i] = absl::get<TensorHandlePtr>(typed_output).release();
    } else {
      outputs[i] = ParallelTensor::AsTensorHandle(
                       context,
                       std::move(absl::get<std::unique_ptr<ParallelTensor>>(
                           typed_output)),
                       status)
                       .release();
      if (TF_GetCode(status) != TF_OK) return;
    }
  }
  *num_outputs = typed_outputs.size();
}

// For TFE_CustomDevice::delete_device in the parallel device registration.
//
// Since this function is used to satisfy the TFE_CustomDevice C API,
// device_info is passed in using a C-style generic. It must always be a
// ParallelDevice.
void DeleteParallelDevice(void* device_info) {
  delete reinterpret_cast<ParallelDevice*>(device_info);
}

}  // namespace

void AllocateParallelDevice(const char* device_name,
                            const char* const* underlying_devices,
                            int num_underlying_devices,
                            TFE_CustomDevice* device, void** device_info) {
  device->copy_tensor_to_device = &CopyToParallelDevice;
  device->copy_tensor_from_device = &CopyTensorFromParallelDevice;
  device->delete_device = &DeleteParallelDevice;
  device->execute = &ParallelDeviceExecute;
  std::vector<std::string> underlying_devices_vector;
  underlying_devices_vector.reserve(num_underlying_devices);
  for (int device_index = 0; device_index < num_underlying_devices;
       ++device_index) {
    underlying_devices_vector.push_back(underlying_devices[device_index]);
  }
  *device_info = new ParallelDevice(device_name, underlying_devices_vector);
}

}  // namespace eager
}  // namespace tensorflow
