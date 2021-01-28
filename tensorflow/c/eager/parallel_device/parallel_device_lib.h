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
#include <string>
#include <vector>

#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {
namespace parallel_device {

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

class ParallelTensor;
class DeviceThread;

// Forwards operations to `devices`, maintaining ParallelTensor with components
// placed on each underlying device.
class ParallelDevice {
 public:
  // Eager async execution is only supported when remote eager is not in use
  // (b/157523095).
  explicit ParallelDevice(const std::vector<std::string>& devices,
                          const bool is_async = false);

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
  std::unique_ptr<ParallelTensor> Vector(
      TFE_Context* context, TF_Status* status,
      absl::Span<const int32_t> values) const;

  // A parallel tensor with scalar integers numbering component devices.
  std::unique_ptr<ParallelTensor> DeviceIDs(TFE_Context* context,
                                            TF_Status* status) const;

  // The number of devices operations run on.
  size_t num_underlying_devices() const { return underlying_devices_.size(); }

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
  void StartExecute(TFE_Context* context,
                    const std::vector<ParallelTensor*>& inputs,
                    const char* operation_name, const TFE_OpAttrs* attributes,
                    int expected_max_outputs) const;

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
      std::vector<TensorHandlePtr> components, absl::Span<const int64> shape,
      TF_Status* status);

  size_t num_tensors() const { return tensors_.size(); }
  TFE_TensorHandle* tensor(size_t index) const { return tensors_[index].get(); }

  // If the `shape` argument to `FromTensorHandles` is specified, returns that.
  //
  // Otherwise if all of the tensors have the same shape, returns that via the
  // `shape` output argument. This blocks waiting for async tensors, may return
  // a delayed bad status encountered during async execution, and will return a
  // bad status unless all tensors have the same shape.
  Status Shape(const std::vector<int64_t>** shape) const;
  TF_DataType dtype() const { return dtype_; }

 private:
  ParallelTensor(const ParallelDevice& device,
                 std::vector<TensorHandlePtr> tensors,
                 absl::Span<const int64> shape, const TF_DataType dtype)
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
  const std::vector<TensorHandlePtr> tensors_;
  // Parallel tensors are immutable but compute their shape lazily unless it is
  // provided on construction. The optional has a value if the lazy computation
  // has been completed or the shape was provided on construction.
  mutable absl::optional<std::vector<int64_t>> shape_;
  const TF_DataType dtype_;
};

}  // namespace parallel_device
}  // namespace tensorflow

#endif  // TENSORFLOW_C_EAGER_PARALLEL_DEVICE_PARALLEL_DEVICE_LIB_H_
