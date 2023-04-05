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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_KERNEL_AND_DEVICE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_KERNEL_AND_DEVICE_H_

// Support for eager execution of TensorFlow kernels.

#include <memory>
#include <unordered_map>

// clang-format off
// Required for IS_MOBILE_PLATFORM
#include "absl/memory/memory.h"
#include "tensorflow/core/platform/platform.h"
// clang-format on

#include "absl/container/flat_hash_map.h"
#include "absl/types/optional.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/util/managed_stack_trace.h"
#include "tensorflow/core/util/tensor_slice_reader_cache.h"
#if !defined(IS_MOBILE_PLATFORM)
#include "tensorflow/core/protobuf/remote_tensor_handle.pb.h"
#endif  // IS_MOBILE_PLATFORM

namespace tensorflow {

static constexpr const char* const kOutputsOnOpDevice = "_OutputsOnOpDevice";

class ProcessFunctionLibraryRuntime;
class FunctionLibraryRuntime;

const int64_t kInvalidOpId = -1;

// This struc is used for:
// 1. setting op_id and step_id, is_component_function for single-client
// remote function scenario,
// 2. setting step_id for multi-client parallel_device scenario.
struct EagerFunctionParams {
  int64_t op_id = kInvalidOpId;
  bool is_component_function;
  absl::optional<int64_t> step_id = absl::nullopt;
};

class EagerKernelArgs : public FunctionArgsInterface {
 public:
  EagerKernelArgs() {}

  explicit EagerKernelArgs(int count) : tensor_args_(count) {}

  explicit EagerKernelArgs(gtl::InlinedVector<TensorValue, 4>&& tensor_args)
      : tensor_args_(std::move(tensor_args)) {}

  ~EagerKernelArgs() override{};

  bool HasRemoteOrPackedInputs() const override { return false; };
  TensorValue* MutableInput(int i) { return &tensor_args_[i]; }

  Status GetLocalArg(const FunctionArgIndex& index, Tensor* val) const override;

  std::vector<Tensor> GetLocalTensors() const override;

  const gtl::InlinedVector<TensorValue, 4>* GetTensorValues() const {
    return &tensor_args_;
  }

 protected:
  gtl::InlinedVector<TensorValue, 4> tensor_args_;
};

typedef absl::variant<Tensor, TensorShape> EagerKernelRet;

// KernelAndDevice encapsulates the logic needed to run a computation eagerly.
// The computation can be a single instantiated kernel (implemented by
// KernelAndDeviceOp below) or a multi-device function (implemented by
// KernelAndDeviceFunc below).
//
// Also see:
// https://www.tensorflow.org/code/tensorflow/core/common_runtime/kernel_benchmark_testlib.h
// and
// https://www.tensorflow.org/code/tensorflow/core/kernels/ops_testutil.h
class KernelAndDevice : public core::RefCounted {
 public:
  // Populates this with a kernel appropriate for 'ndef'.
  //
  // The provided FunctionLibraryRuntime MUST outlive all calls to
  // Run() on the returned KernelAndDevice.
  virtual Status Init(const bool log_device_placement, const NodeDef& ndef,
                      GraphCollector* graph_collector) = 0;

  // Non-multi-device functions are run using regular CallOp and look like
  // primitive operations from KernelAndDevice perspective.
  // `flr` can be nullptr if the operation is not run on any specific device
  // (currently can happen only for multi-device functions).
  KernelAndDevice(
      FunctionLibraryRuntime* flr,
      std::function<void(std::function<void()>)>* runner,
      std::unique_ptr<CollectiveExecutor::Handle> collective_executor,
      Device* host_cpu_device)
      : device_(flr == nullptr ? nullptr : flr->device()),
        host_cpu_device_(host_cpu_device),
        flr_(flr),
        collective_executor_(std::move(collective_executor)),
        runner_(runner) {}

  // Not thread safe.
  ~KernelAndDevice() override {}

  virtual bool IsFunction() { return false; }

  virtual bool IsCrossProcess() { return false; }

  // TODO(ashankar): Handle list-valued inputs.
  virtual Status Run(
      ScopedStepContainer* step_container, const EagerKernelArgs& inputs,
      std::vector<EagerKernelRet>* outputs,
      CancellationManager* cancellation_manager,
      const absl::optional<EagerFunctionParams>& eager_func_params,
      const absl::optional<ManagedStackTrace>& stack_trace,
      tsl::CoordinationServiceAgent* coordination_service_agent) = 0;

  // Execute kernel asynchronously when applicable. Different from `Run` which
  // blocks the caller thread and waits for the execution of the op/function,
  // `RunAsync` could return before finishing the execution. The `done` callback
  // will be triggered once the op/function execution finishes.
  // Currently, calling RunAsync on ops might not honor the asynchronicity when
  // it is called on an instance with only sync implementation, execute the
  // kernel synchronously and then call the callback with the return status
  // from sync execution.
  virtual void RunAsync(
      ScopedStepContainer* step_container, const EagerKernelArgs& inputs,
      std::vector<EagerKernelRet>* outputs,
      CancellationManager* cancellation_manager,
      const absl::optional<EagerFunctionParams>& eager_func_params,
      tsl::CoordinationServiceAgent* coordination_service_agent,
      StatusCallback done) = 0;

  virtual Device* InputDevice(int i) const = 0;
  virtual Device* OutputDevice(int idx) const = 0;
  // If idx'th output is a resource, returns the device backing the resource.
  // Else, returns nullptr.
  virtual Device* OutputResourceDevice(int idx) const = 0;

  // Returns the kernel that will be used to run this.
  // Returns nullptr if this will be run using function library runtime.
  virtual const OpKernel* kernel() const = 0;

  // Returns the device on which this kernel will run. In the case of
  // multi-device functions, this is the default device that is passed to the
  // placer but actual computation can happen on a different set of devices.
  // Also, outputs can be produced on devices different from what this method
  // returns.
  Device* device() const { return device_; }

  virtual const DataTypeVector& input_dtypes() const = 0;
  virtual const DataTypeVector& output_dtypes() const = 0;

  virtual int num_inputs() const = 0;
  virtual int num_outputs() const = 0;
  virtual const string& name() const = 0;

 protected:
  std::function<void(std::function<void()>)>* get_runner() const;

  Device* const device_;               // can be null
  Device* const host_cpu_device_;      // non-null
  FunctionLibraryRuntime* const flr_;  // can be null
  const std::unique_ptr<CollectiveExecutor::Handle> collective_executor_;

 private:
  std::function<void(std::function<void()>)>* const runner_;  // can be null
};

// Represents an op kernel and the device it will be run on.
class KernelAndDeviceOp final : public KernelAndDevice {
 public:
  KernelAndDeviceOp(
      tensorflow::Rendezvous* rendezvous, bool log_memory,
      FunctionLibraryRuntime* flr,
      std::function<void(std::function<void()>)>* runner,
      std::unique_ptr<CollectiveExecutor::Handle> collective_executor,
      Device* host_cpu_device)
      : KernelAndDevice(flr, runner, std::move(collective_executor),
                        host_cpu_device),
        rendezvous_(rendezvous),
        log_memory_(log_memory) {}

  ~KernelAndDeviceOp() override {}

  Status Init(const bool log_device_placement, const NodeDef& ndef,
              GraphCollector* graph_collector) override;

  Status Run(
      ScopedStepContainer* step_container, const EagerKernelArgs& inputs,
      std::vector<EagerKernelRet>* outputs,
      CancellationManager* cancellation_manager,
      const absl::optional<EagerFunctionParams>& eager_func_params,
      const absl::optional<ManagedStackTrace>& stack_trace,
      tsl::CoordinationServiceAgent* coordination_service_agent) override;

  void RunAsync(ScopedStepContainer* step_container,
                const EagerKernelArgs& inputs,
                std::vector<EagerKernelRet>* outputs,
                CancellationManager* cancellation_manager,
                const absl::optional<EagerFunctionParams>& eager_func_params,
                tsl::CoordinationServiceAgent* coordination_service_agent,
                StatusCallback done) override {
    // Trivial async implementation on top of the sync version
    done(Run(step_container, inputs, outputs, cancellation_manager,
             eager_func_params, {}, coordination_service_agent));
  }

  const OpKernel* kernel() const override { return kernel_.get(); }

  Device* InputDevice(int i) const override;
  Device* OutputDevice(int idx) const override;
  Device* OutputResourceDevice(int idx) const override;

  const DataTypeVector& input_dtypes() const override {
    return kernel_->input_types();
  }
  const DataTypeVector& output_dtypes() const override {
    return kernel_->output_types();
  }
  int num_inputs() const override { return kernel_->num_inputs(); }
  int num_outputs() const override { return kernel_->num_outputs(); }
  const string& name() const override { return kernel_->name(); }

 private:
  std::unique_ptr<OpKernel> kernel_;
  bool is_distributed_communication_op_;
  gtl::InlinedVector<AllocatorAttributes, 4> input_alloc_attrs_;
  std::vector<Device*> input_devices_;
  gtl::InlinedVector<AllocatorAttributes, 1> output_alloc_attrs_;
  Rendezvous* const rendezvous_;
  checkpoint::TensorSliceReaderCacheWrapper slice_reader_cache_;
  const bool log_memory_;
};

// Represents a multi-device function. Functions can also be run using
// various function-calling kernels including CallOp and PartitionedCallOp.
// In such cases, KernelAndDeviceOp is used.
class KernelAndDeviceFunc : public KernelAndDevice {
 public:
  // `flr` can be nullptr.
  // `pflr` must not be nullptr.
  // `host_cpu_device` must not be nullptr.
  KernelAndDeviceFunc(
      FunctionLibraryRuntime* flr, ProcessFunctionLibraryRuntime* pflr,
      std::vector<Device*> input_devices,
      absl::flat_hash_map<string, const std::vector<string>*> composite_devices,
      std::unordered_map<int, DtypeAndPartialTensorShape>
          input_resource_dtypes_and_shapes,
      std::function<void(std::function<void()>)>* runner,
      std::unique_ptr<CollectiveExecutor::Handle> collective_executor,
      Device* host_cpu_device, const string& name,
      const bool outputs_on_op_device,
      const bool allow_small_function_optimizations,
      const bool allow_control_flow_sync_execution,
      const bool shape_inference_on_tfe_dialect_import,
      const bool int_args_and_retvals_on_device,
      absl::optional<string> xla_compile_device_type,
      Rendezvous::Factory rendezvous_factory,
      std::function<int64_t()> get_op_id)
      : KernelAndDevice(flr, runner, std::move(collective_executor),
                        host_cpu_device),
        pflr_(pflr),
        handle_(kInvalidHandle),
        outputs_on_op_device_(outputs_on_op_device),
        allow_small_function_optimizations_(allow_small_function_optimizations),
        allow_control_flow_sync_execution_(allow_control_flow_sync_execution),
        shape_inference_on_tfe_dialect_import_(
            shape_inference_on_tfe_dialect_import),
        int_args_and_retvals_on_device_(int_args_and_retvals_on_device),
        xla_compile_device_type_(xla_compile_device_type),
        input_devices_(std::move(input_devices)),
        composite_devices_(std::move(composite_devices)),
        input_resource_dtypes_and_shapes_(
            std::move(input_resource_dtypes_and_shapes)),
        name_(name),
        rendezvous_factory_(std::move(rendezvous_factory)),
        get_op_id_(std::move(get_op_id)) {}

  ~KernelAndDeviceFunc() override;

  bool IsFunction() override { return true; };

  bool IsCrossProcess() override { return is_cross_process_; }

  Status InstantiateFunc(const bool log_device_placement, const NodeDef& ndef,
                         GraphCollector* graph_collector);

  Status Init(const bool log_device_placement, const NodeDef& ndef,
              GraphCollector* graph_collector) override;

  Status Run(
      ScopedStepContainer* step_container, const EagerKernelArgs& inputs,
      std::vector<EagerKernelRet>* outputs,
      CancellationManager* cancellation_manager,
      const absl::optional<EagerFunctionParams>& eager_func_params,
      const absl::optional<ManagedStackTrace>& stack_trace,
      tsl::CoordinationServiceAgent* coordination_service_agent) override;

  void RunAsync(ScopedStepContainer* step_container,
                const EagerKernelArgs& inputs,
                std::vector<EagerKernelRet>* outputs,
                CancellationManager* cancellation_manager,
                const absl::optional<EagerFunctionParams>& eager_func_params,
                tsl::CoordinationServiceAgent* coordination_service_agent,
                StatusCallback done) override;

  const OpKernel* kernel() const override { return nullptr; }

  Device* InputDevice(int i) const override;
  Device* OutputDevice(int idx) const override;
  Device* OutputResourceDevice(int idx) const override;

  const DataTypeVector& input_dtypes() const override { return input_dtypes_; }
  const DataTypeVector& output_dtypes() const override {
    return output_dtypes_;
  }
  int num_inputs() const override { return input_dtypes_.size(); }
  int num_outputs() const override { return output_dtypes_.size(); }
  const string& name() const override { return name_; };

 private:
  std::shared_ptr<FunctionLibraryRuntime::Options> PrepareForRun(
      ScopedStepContainer* step_container, std::vector<EagerKernelRet>* outputs,
      CancellationManager* cancellation_manager,
      const absl::optional<EagerFunctionParams>& eager_func_params,
      const absl::optional<ManagedStackTrace>& stack_trace,
      tsl::CoordinationServiceAgent* coordination_service_agent);

  ProcessFunctionLibraryRuntime* const pflr_;  // non-null
  FunctionLibraryRuntime::Handle handle_;
  // Indicates whether the function needs to execute cross process.
  bool is_cross_process_;

  // If true, function outputs are explicitly assigned to the default device;
  // if false, the output devices are inferred by pflr_.
  bool outputs_on_op_device_;

  // If True, allow optimizations which should be targeted at a limited
  // set of small functions.  (For example, running kernels synchronously can
  // be faster under some conditions.)
  const bool allow_small_function_optimizations_;

  // If True, allows control nodes to run on the single threaded executor.
  const bool allow_control_flow_sync_execution_;

  // TODO(b/176491312): Remove this if shape inference on import flag is
  // removed. If True, allows mlir roundtrip to run shape inference on import.
  const bool shape_inference_on_tfe_dialect_import_;

  const bool int_args_and_retvals_on_device_;

  const absl::optional<string> xla_compile_device_type_;

  // CPU devices are null. Resource handles' devices are actual backing
  // devices.
  std::vector<Device*> output_devices_;
  // CPU devices are not null. Resource handles' devices are actual backing
  // devices.
  std::vector<Device*> input_devices_;
  // Maps from a CompositeDevice name to a list of physical device names.
  absl::flat_hash_map<string, const std::vector<string>*> composite_devices_;
  std::unordered_map<int, DtypeAndPartialTensorShape>
      input_resource_dtypes_and_shapes_;

  DataTypeVector input_dtypes_;
  DataTypeVector output_dtypes_;
  string name_;

  Rendezvous::Factory rendezvous_factory_;
  std::function<int64_t()> get_op_id_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_KERNEL_AND_DEVICE_H_
