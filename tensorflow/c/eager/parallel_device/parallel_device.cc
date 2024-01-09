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

#include <cstring>
#include <memory>

#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "absl/types/variant.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/parallel_device/parallel_device_lib.h"
#include "tensorflow/c/eager/tfe_tensorhandle_internal.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_status_helper.h"

namespace tensorflow {
namespace parallel_device {
namespace {

class OpDeleter {
 public:
  void operator()(TFE_Op* to_delete) const { TFE_DeleteOp(to_delete); }
};

using OpPtr = std::unique_ptr<TFE_Op, OpDeleter>;

using MaybeParallelTensorOwned =
    absl::variant<std::unique_ptr<ParallelTensor>, TensorHandlePtr>;

using MaybeParallelTensorUnowned =
    absl::variant<ParallelTensor*, TFE_TensorHandle*>;

// A ParallelDevice on its own is not registered with a TFE_Context, and so has
// no device name (e.g. for `tf.device`). `NamedParallelDevice` associates a
// name with it, which lets us pack its `ParallelTensor`s into TFE_TensorHandles
// placed on the parallel device.
class NamedParallelDevice {
 public:
  NamedParallelDevice(const std::string& name,
                      std::unique_ptr<ParallelDevice> parallel_device)
      : device_name_(name), parallel_device_(std::move(parallel_device)) {}
  const std::string& name() const { return device_name_; }
  const ParallelDevice& device() const { return *parallel_device_; }

 private:
  std::string device_name_;
  std::unique_ptr<ParallelDevice> parallel_device_;
};

absl::optional<std::vector<MaybeParallelTensorOwned>> ExecuteWithSpecialOps(
    const ParallelDevice& parallel_device,
    const std::string& parallel_device_name, TFE_Context* context,
    std::vector<MaybeParallelTensorUnowned> inputs, const char* operation_name,
    const TFE_OpAttrs* attributes, int expected_max_outputs,
    TF_Status* status) {
  absl::optional<std::vector<MaybeParallelTensorOwned>> result;
  // TODO(allenl): We should remove "TPU" from these op names at the very least,
  // or consider other ways of packing/unpacking parallel tensors.
  if (operation_name == std::string("TPUReplicatedInput")) {
    // Special-cased operation for packing per-device tensors into one parallel
    // tensor.
    if (inputs.size() != parallel_device.num_underlying_devices()) {
      std::string message(absl::StrCat(
          "The parallel device ", parallel_device_name, " expected ",
          parallel_device.num_underlying_devices(),
          " inputs to TPUReplicatedInput, but got ", inputs.size()));
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
        parallel_device, std::move(components), status));
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
    if (expected_outputs != parallel_device.num_underlying_devices()) {
      std::string message(absl::StrCat(
          "The parallel device ", parallel_device_name, " expected ",
          parallel_device.num_underlying_devices(),
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
      TensorHandlePtr this_output(
          TFE_TensorHandleCopySharingTensor(t->tensor(i), status));
      outputs.emplace_back(std::move(this_output));
      if (TF_GetCode(status) != TF_OK) return result;
    }
    result.emplace(std::move(outputs));
    return result;
  }
  std::vector<ParallelTensor*> parallel_inputs;
  std::vector<std::unique_ptr<ParallelTensor>> implicitly_broadcast_tensors;
  parallel_inputs.reserve(inputs.size());
  implicitly_broadcast_tensors.reserve(inputs.size());  // not tight
  for (const auto& input : inputs) {
    if (absl::holds_alternative<TFE_TensorHandle*>(input)) {
      if (operation_name == std::string("_EagerConst")) {
        // Non-parallel tensors from _EagerConst/tf.constant are implicitly
        // broadcast, i.e. set as the input to each parallel operation. This
        // allows code like "tf.constant(1.)" or "tf.reduce_sum(..., axis=1)"
        // (where the value starts on the host), without allowing other implicit
        // copies/broadcasts. Other implicit copies may be supported eventually,
        // but need special handling for gradients (gradient of copy-on is not
        // just copy-off but includes a sum) and consideration of performance.
        //
        // TODO(allenl): There may be smarter ways to do this copy in some
        // cases, i.e. with a collective broadcast. We'll need to be careful
        // about things that are taken as inputs on the host or on their
        // existing device (for multi-device functions).
        std::unique_ptr<ParallelTensor> parallel_tensor(
            parallel_device.CopyToParallelDevice(
                context, absl::get<TFE_TensorHandle*>(input), status));
        if (TF_GetCode(status) != TF_OK) return absl::nullopt;
        parallel_inputs.push_back(parallel_tensor.get());
        implicitly_broadcast_tensors.emplace_back(std::move(parallel_tensor));
      } else {
        TF_SetStatus(
            status, TF_INVALID_ARGUMENT,
            absl::StrCat(
                "Got a non-parallel tensor ",
                tensorflow::unwrap(absl::get<TFE_TensorHandle*>(input))
                    ->DebugString(),
                " as input to a parallel operation. First pack non-parallel "
                "tensors for each device into a parallel tensor explicitly.")
                .c_str());
        return absl::nullopt;
      }
    } else {
      parallel_inputs.push_back(absl::get<ParallelTensor*>(input));
    }
  }
  absl::optional<std::vector<std::unique_ptr<ParallelTensor>>>
      maybe_parallel_results(
          parallel_device.Execute(context, parallel_inputs, operation_name,
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

// Used as an argument to TFE_NewCustomDeviceTensorHandle, indicating how
// ParallelTensors wrapped in TFE_TensorHandles should be cleaned up once their
// reference counts drop to zero.
void ParallelTensorDeallocator(void* data) {
  delete reinterpret_cast<ParallelTensor*>(data);
}

// Used as an argument to TFE_NewCustomDeviceTensorHandle, for computing the
// number of dimensions of a parallel tensor.
int ParallelTensorNumDims(void* data, TF_Status* status) {
  const std::vector<int64_t>* shape;
  Status s = reinterpret_cast<ParallelTensor*>(data)->Shape(&shape);
  if (!s.ok()) {
    tsl::Set_TF_Status_from_Status(status, s);
    return -1;
  }
  return shape->size();
}

// Used as an argument to TFE_NewCustomDeviceTensorHandle, for computing a
// dimension of a parallel tensor.
int64_t ParallelTensorDim(void* data, int dim_index, TF_Status* status) {
  const std::vector<int64_t>* shape;
  Status s = reinterpret_cast<ParallelTensor*>(data)->Shape(&shape);
  if (!s.ok()) {
    tsl::Set_TF_Status_from_Status(status, s);
    return -1;
  }
  return (*shape)[dim_index];
}

TF_Buffer* ParallelTensorSummarize(void* data, TF_Status* status) {
  ParallelTensor* parallel_tensor = reinterpret_cast<ParallelTensor*>(data);
  std::string summary;
  Status cpp_status = parallel_tensor->SummarizeValue(summary);
  if (!cpp_status.ok()) {
    tsl::Set_TF_Status_from_Status(status, cpp_status);
    return nullptr;
  }
  return TF_NewBufferFromString(summary.data(), summary.size());
}

TensorHandlePtr ParallelTensorToTensorHandle(
    const std::string& parallel_device_name, TFE_Context* context,
    std::unique_ptr<ParallelTensor> t, TF_Status* status) {
  // The resulting TensorHandle owns an opaque pointer to "device memory", which
  // for a ParallelDevice is really a ParallelTensor. When the TensorHandle is
  // deleted, it will call ParallelTensorDeallocator to free the struct.
  ParallelTensor* t_released = t.release();
  TFE_CustomDeviceTensorHandleMethods handle_methods;
  handle_methods.num_dims = &ParallelTensorNumDims;
  handle_methods.dim = &ParallelTensorDim;
  handle_methods.deallocator = &ParallelTensorDeallocator;
  handle_methods.summarize = &ParallelTensorSummarize;
  return TensorHandlePtr(TFE_NewCustomDeviceTensorHandle(
      context, parallel_device_name.c_str(), t_released->dtype(), t_released,
      handle_methods, status));
}

// For TFE_CustomDevice::copy_tensor_to_device in the parallel device
// registration.
//
// Since this function is used to satisfy the TFE_CustomDevice C API,
// device_info is passed in using a C-style generic. It must always be a
// ParallelDevice.
TFE_TensorHandle* CopyToParallelDevice(TFE_Context* context,
                                       TFE_TensorHandle* tensor,
                                       TF_Status* status, void* device_info) {
  TF_SetStatus(
      status, TF_UNIMPLEMENTED,
      absl::StrCat("Trying to copy a tensor ",
                   tensorflow::unwrap(tensor)->DebugString(),
                   " on to a parallel device. Pack non-parallel "
                   "tensors for each device into a parallel tensor explicitly.")
          .c_str());
  return nullptr;
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
  ParallelTensor* parallel_tensor = reinterpret_cast<ParallelTensor*>(
      TFE_TensorHandleDevicePointer(tensor, status));
  if (TF_GetCode(status) != TF_OK) return nullptr;
  if (parallel_tensor->num_tensors() == 1) {
    // Copy-off for single-device tensors is allowed to make debugging dynamic
    // control flow easier.
    return TFE_TensorHandleCopySharingTensor(parallel_tensor->tensor(0),
                                             status);
  } else {
    TF_SetStatus(
        status, TF_UNIMPLEMENTED,
        absl::StrCat(
            "Trying to copy a tensor out of a parallel device. Since there "
            "are multiple components to parallel tensors, they must be "
            "unpacked explicitly.\n",
            tensorflow::unwrap(tensor)->DebugString())
            .c_str());
    return nullptr;
  }
}

// For TFE_CustomDevice::execute in the parallel device registration.
//
// Since this function is used to satisfy the TFE_CustomDevice C API,
// device_info is passed in using a C-style generic. It must always be a
// ParallelDevice.
void ParallelDeviceExecute(const TFE_Op* original_op, int* num_outputs,
                           TFE_TensorHandle** outputs, TF_Status* status,
                           void* device_info) {
  const char* requested_placement = TFE_OpGetDevice(original_op, status);
  if (*requested_placement == '\0') {
    TF_SetStatus(
        status, TF_INTERNAL,
        "Ops must be placed on the parallel device explicitly, or their inputs "
        "first un-packed. Got an un-placed op with an input placed on the "
        "parallel device.");
    return;
  }
  TFE_Context* context = TFE_OpGetContext(original_op, status);
  if (TF_GetCode(status) != TF_OK) return;
  const char* operation_name = TFE_OpGetName(original_op, status);
  if (TF_GetCode(status) != TF_OK) return;
  const TFE_OpAttrs* attributes = TFE_OpGetAttrs(original_op);

  NamedParallelDevice* named_device =
      reinterpret_cast<NamedParallelDevice*>(device_info);
  std::vector<MaybeParallelTensorUnowned> typed_inputs;
  int num_inputs = TFE_OpGetFlatInputCount(original_op, status);
  if (TF_GetCode(status) != TF_OK) return;
  typed_inputs.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    TFE_TensorHandle* input = TFE_OpGetFlatInput(original_op, i, status);
    if (TF_GetCode(status) != TF_OK) return;
    const char* tensor_handle_device =
        TFE_TensorHandleDeviceName(input, status);
    if (TF_GetCode(status) != TF_OK) return;
    if (named_device->name() == tensor_handle_device) {
      // We assume that any tensors already placed on this device are
      // ParallelTensors.
      typed_inputs.emplace_back(reinterpret_cast<ParallelTensor*>(
          TFE_TensorHandleDevicePointer(input, status)));
      if (TF_GetCode(status) != TF_OK) return;
    } else {
      typed_inputs.emplace_back(input);
    }
  }

  absl::optional<std::vector<MaybeParallelTensorOwned>> maybe_typed_outputs(
      ExecuteWithSpecialOps(named_device->device(), named_device->name(),
                            context, std::move(typed_inputs), operation_name,
                            attributes, *num_outputs, status));
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
      outputs[i] = ParallelTensorToTensorHandle(
                       named_device->name(), context,
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
  delete reinterpret_cast<NamedParallelDevice*>(device_info);
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
  std::unique_ptr<ParallelDevice> parallel_device(
      new ParallelDevice(underlying_devices_vector));
  *device_info =
      new NamedParallelDevice{device_name, std::move(parallel_device)};
}
}  // namespace parallel_device
}  // namespace tensorflow
