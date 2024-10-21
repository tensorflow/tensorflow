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

#ifndef TENSORFLOW_C_EAGER_PARALLEL_DEVICE_PARALLEL_DEVICE_LIB_H_
#define TENSORFLOW_C_EAGER_PARALLEL_DEVICE_PARALLEL_DEVICE_LIB_H_

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/tfe_op_internal.h"
#include "tensorflow/c/safe_ptr.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {
namespace parallel_device {

using TensorHandlePtr = tensorflow::Safe_TFE_TensorHandlePtr;

class ParallelTensor;
class DeviceThread;

// Forwards operations to `devices`, maintaining ParallelTensor with components
// placed on each underlying device.
class ParallelDevice {
 public:
  // Eager async execution is only supported when remote eager is not in use
  // (b/157523095).
  explicit ParallelDevice(const std::vector<std::string>& devices,
                          bool is_async = false, int in_flight_nodes_limit = 0);

  ~ParallelDevice();

  // Helper to copy a tensor handle from another device once for each component
  // of the ParallelDevice.
  //
  // Sets a bad status and returns a nullptr if `tensor` is already on the
  // ParallelDevice, or if the individual copies fail.
  std::unique_ptr<ParallelTensor> CopyToParallelDevice(TFE_Context* context,
                                                       TFE_TensorHandle* tensor,
                                                       TF_Status* status) const;

  // Construct a parallel tensor consisting of the scalar values from `values`.
  template <typename DataType>
  std::unique_ptr<ParallelTensor> ScalarsFromSequence(
      absl::Span<const DataType> values, TFE_Context* context,
      TF_Status* status) const;

  // A parallel tensor with scalar integers numbering component devices.
  std::unique_ptr<ParallelTensor> DeviceIDs(TFE_Context* context,
                                            TF_Status* status) const;

  // The number of devices operations run on.
  size_t num_underlying_devices() const { return underlying_devices_.size(); }

  // The devices operations run on.
  const std::vector<std::string>& underlying_devices() const {
    return underlying_devices_;
  }

  // Takes a description of a single operation being executed on the
  // ParallelDevice, and in turn runs one operation per component device with
  // its corresponding inputs from the input ParallelTensors. Wraps the
  // resulting per-device and per-output TFE_TensorHandles into one
  // ParallelTensor per output of the original operation.
  //
  // Attributes are forwarded to executed operations unmodified.
  //
  // The returned optional has a value if and only if `status` evaluates to
  // TF_OK. Bad statuses are forwarded from underlying `TFE_Execute` calls, or
  // if sanity checks on dtypes/metadata fail.
  absl::optional<std::vector<std::unique_ptr<ParallelTensor>>> Execute(
      TFE_Context* context, const std::vector<ParallelTensor*>& inputs,
      const char* operation_name, const TFE_OpAttrs* attributes,
      int expected_max_outputs, TF_Status* status) const;

  // A non-blocking version of `Execute`. After each call, `Join` must be called
  // before `StartExecute` is called again. Using `StartExecute` with `Join`
  // allows the caller to schedule computation on multiple ParallelDevices
  // without sequencing those operations (first call `StartExecute` on each
  // parallel device, then call `Join` on each; even if some of the `Join`s
  // return a bad status the caller must run all of the `Join`s or any future
  // `StartExecute`s will deadlock).
  //
  // If `is_async=false` (constructor argument), `cancellation_manager` must
  // live until `Join` finishes. If `is_async=true` it must live until `Join` is
  // followed by `TFE_ContextAsyncWait` to clear pending operations. It will be
  // used to cancel all other operations if any fails.
  //
  // Set step_id to configure the step id used for rendezvous creation. step id
  // of value -1 is reserved for global rendezvous and should not be set here.
  void StartExecute(TFE_Context* context,
                    const std::vector<ParallelTensor*>& inputs,
                    const char* operation_name, const TFE_OpAttrs* attributes,
                    int expected_max_outputs,
                    CancellationManager& cancellation_manager,
                    std::optional<int64_t> step_id = std::nullopt) const;

  void StartExecute(TFE_Context* context,
                    const std::vector<std::vector<TFE_TensorHandle*>>& inputs,
                    const char* operation_name, const TFE_OpAttrs* attributes,
                    int expected_max_outputs,
                    CancellationManager& cancellation_manager,
                    std::optional<int64_t> step_id = std::nullopt) const;

  // Blocks until the previous `StartExecute` has run `TFE_Execute` on each
  // device. If is_async=false (constructor argument) this means the ops have
  // run and have results. If is_async=true it means that all of the
  // device-specific executors have scheduled the op.
  //
  // Accepts inferred shapes for outputs (`expected_output_shapes`), which if
  // fully defined will avoid querying the shapes of the underlying
  // TensorHandles when ParallelTensor::Shape is called. This allows async
  // computation to continue without blocking.
  //
  // The return status and value is the same as `Execute`.
  absl::optional<std::vector<std::unique_ptr<ParallelTensor>>> Join(
      const std::vector<PartialTensorShape>& expected_output_shapes,
      TF_Status* status) const;

  void AsyncWait(TFE_Context* context, TF_Status* status) const;

  // Device strings for component devices that only include a
  // worker/task/replica if any of those differ across components. Useful for
  // printing debug messages.
  std::vector<std::string> SummarizeDeviceNames() const;

 private:
  // A sequence of device names, indicating which devices replicated operations
  // are forwarded to.
  const std::vector<std::string> underlying_devices_;
  // A sequence of thread wrappers, one per device, for executing operations in
  // parallel.
  //
  // Conceptually this is a thread pool with one thread per device. It requires
  // less synchronization than a thread pool would for this task, since Execute
  // acquires each thread in order (and so only one Execute will schedule
  // blocking collective operations at a time), and avoids some dynamic
  // allocation/scheduling.
  //
  // TODO(allenl): Keep a map from outer thread to list of inner threads rather
  // than a single list of threads so aliased nested parallel devices don't
  // re-use a thread.
  std::vector<std::unique_ptr<DeviceThread>> device_threads_;
  // A cancellation manager to use if the caller does not provide one. When ops
  // are executed asynchronously this must outlive the queued op, so it can't be
  // function-local to Execute.
  mutable std::unique_ptr<CancellationManager> default_cancellation_manager_;
};

// Contains a tuple of tensors, one on each of the `underlying_devices_` of the
// ParallelDevice.
class ParallelTensor {
 public:
  // Construct a ParallelTensor from TensorHandles placed on the component
  // devices of a ParallelDevice. If called, ParallelTensor::Shape inspects
  // `components` to determine a shape.
  static std::unique_ptr<ParallelTensor> FromTensorHandles(
      const ParallelDevice& parallel_device,
      std::vector<TensorHandlePtr> components, TF_Status* status);
  // Uses the provided shape without additional checks, which avoids blocking
  // when ParallelTensor::Shape is called.
  static std::unique_ptr<ParallelTensor> FromTensorHandles(
      const ParallelDevice& parallel_device,
      std::vector<TensorHandlePtr> components, absl::Span<const int64_t> shape,
      TF_Status* status);

  size_t num_tensors() const { return tensors_.size(); }
  TFE_TensorHandle* tensor(size_t index) const { return tensors_[index].get(); }

  // If the `shape` argument to `FromTensorHandles` is specified, returns that.
  //
  // Otherwise if all of the tensors have the same shape, returns that via the
  // `shape` output argument. This blocks waiting for async tensors, may return
  // a delayed bad status encountered during async execution, and will return a
  // bad status unless all tensors have the same shape.
  absl::Status Shape(const std::vector<int64_t>** shape) const;
  TF_DataType dtype() const { return dtype_; }

  // Sets its output argument to a summary of the values of this tensor on every
  // component device.
  absl::Status SummarizeValue(std::string& summary);

  std::vector<TensorHandlePtr> release_tensors() { return std::move(tensors_); }

  std::vector<TFE_TensorHandle*> tensors() const {
    std::vector<TFE_TensorHandle*> result;
    result.reserve(tensors_.size());
    for (const TensorHandlePtr& tensor : tensors_) {
      result.emplace_back(tensor.get());
    }
    return result;
  }

 private:
  ParallelTensor(const ParallelDevice& device,
                 std::vector<TensorHandlePtr> tensors,
                 absl::Span<const int64_t> shape, const TF_DataType dtype)
      : device_(device),
        tensors_(std::move(tensors)),
        shape_(std::vector<int64_t>(shape.begin(), shape.end())),
        dtype_(dtype) {}
  ParallelTensor(const ParallelDevice& device,
                 std::vector<TensorHandlePtr> tensors, const TF_DataType dtype)
      : device_(device),
        tensors_(std::move(tensors)),
        shape_(absl::nullopt),
        dtype_(dtype) {}

  const ParallelDevice& device_;
  std::vector<TensorHandlePtr> tensors_;
  // Parallel tensors are immutable but compute their shape lazily unless it is
  // provided on construction. The optional has a value if the lazy computation
  // has been completed or the shape was provided on construction.
  mutable absl::optional<std::vector<int64_t>> shape_;
  const TF_DataType dtype_;
};

template <typename DataType>
std::unique_ptr<ParallelTensor> ParallelDevice::ScalarsFromSequence(
    absl::Span<DataType const> values, TFE_Context* context,
    TF_Status* status) const {
  std::vector<TensorHandlePtr> components;
  components.reserve(underlying_devices_.size());

  if (values.size() != num_underlying_devices()) {
    TF_SetStatus(
        status, TF_INVALID_ARGUMENT,
        "Number of values did not match number of underlying devices.");
    return nullptr;
  }
  TF_DataType datatype_enum(
      static_cast<TF_DataType>(DataTypeToEnum<DataType>().value));
  for (int device_index = 0; device_index < num_underlying_devices();
       ++device_index) {
    auto device_value = absl::make_unique<DataType>();
    *device_value = values[device_index];
    std::unique_ptr<TF_Tensor, decltype(&TF_DeleteTensor)> tensor(
        TF_NewTensor(
            datatype_enum, /*dims=*/nullptr, /*num_dims=*/0,
            device_value.release(), sizeof(DataType),
            [](void* data, size_t, void* arg) {
              delete reinterpret_cast<DataType*>(data);
            },
            nullptr),
        TF_DeleteTensor);
    // TODO(allenl): Here and when executing regular operations, we could hold
    // on to one TFE_Op per device and just call TFE_ResetOp to avoid parsing
    // device names repeatedly.
    std::unique_ptr<TFE_Op, decltype(&TFE_DeleteOp)> const_op(
        TFE_NewOp(context, "Const", status), TFE_DeleteOp);
    if (TF_GetCode(status) != TF_OK) return nullptr;
    TFE_OpSetDevice(const_op.get(), underlying_devices_[device_index].c_str(),
                    status);
    if (TF_GetCode(status) != TF_OK) return nullptr;
    TFE_OpSetAttrTensor(const_op.get(), "value", tensor.get(), status);
    if (TF_GetCode(status) != TF_OK) return nullptr;
    TFE_OpSetAttrType(const_op.get(), "dtype", datatype_enum);
    TFE_TensorHandle* device_handle;
    int num_outputs = 1;
    TFE_Execute(const_op.get(), &device_handle, &num_outputs, status);
    if (TF_GetCode(status) != TF_OK) return nullptr;
    components.emplace_back(device_handle);
  }
  return ParallelTensor::FromTensorHandles(*this, std::move(components),
                                           status);
}

}  // namespace parallel_device
}  // namespace tensorflow

#endif  // TENSORFLOW_C_EAGER_PARALLEL_DEVICE_PARALLEL_DEVICE_LIB_H_
