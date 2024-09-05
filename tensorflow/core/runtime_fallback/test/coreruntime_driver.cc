/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/runtime_fallback/test/coreruntime_driver.h"

#include <memory>
#include <string>
#include <utility>

#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_op_handler.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_tensor.h"
#include "tensorflow/core/runtime_fallback/runtime/runtime_fallback_op_handler.h"
#include "tensorflow/core/runtime_fallback/runtime/runtime_fallback_tensor.h"
#include "tensorflow/core/runtime_fallback/util/fallback_test_util.h"
#include "tfrt/cpu/core_runtime/cpu_op_handler.h"  // from @tf_runtime
#include "tfrt/core_runtime/op_attrs.h"  // from @tf_runtime
#include "tfrt/core_runtime/op_handler.h"  // from @tf_runtime
#include "tfrt/core_runtime/tensor_handle.h"  // from @tf_runtime
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime
#include "tfrt/host_context/host_allocator.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/tensor/scalar_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/string_host_tensor.h"  // from @tf_runtime

namespace tfrt {

namespace {
constexpr const char* kCpuOpHandlerName = "cpu";

std::unique_ptr<tfrt::CoreRuntime> CreateCoreRuntime() {
  auto diag_handler = [](const tfrt::DecodedDiagnostic& diag) {
    llvm::errs() << "Encountered runtime error: " << diag.message() << "\n";
  };
  auto corert = tfrt::CoreRuntime::Create(
      diag_handler, tfrt::CreateMallocAllocator(),
      tfrt::CreateMultiThreadedWorkQueue(
          /*num_threads=*/4, /*num_blocking_threads=*/64));
  assert(corert && "error creating CoreRuntime");
  return std::move(*corert);
}
}  // namespace

CoreRuntimeDriver::CoreRuntimeDriver(std::unique_ptr<tfrt::CoreRuntime> corert)
    : corert_(std::move(corert)),
      chain_(MakeAvailableAsyncValueRef<tfrt::Chain>()) {}

CoreRuntimeDriver::CoreRuntimeDriver()
    : CoreRuntimeDriver(CreateCoreRuntime()) {}

void CoreRuntimeDriver::InitializeCpuRuntimeFallbackOpHandler() {
  auto fallback_op_handler = tensorflow::tfd::CreateRuntimeFallbackOpHandler(
      corert_.get(), /*tf_device_name=*/"");

  auto cpu_device = corert_->GetHostContext()->GetHostDeviceRef();
  auto cpu_op_handler = tfrt::CreateCpuOpHandler(
      corert_.get(), std::move(cpu_device), fallback_op_handler.get());

  cpu_op_handler.get()->AddImplicitConversion(
      tensorflow::tfd::RuntimeFallbackTensor::kTensorType,
      DenseHostTensor::kTensorType);
  cpu_op_handler.get()->AddImplicitConversion(
      tensorflow::tfd::RuntimeFallbackTensor::kTensorType,
      AnyScalarHostTensor::kTensorType);
  cpu_op_handler.get()->AddImplicitConversion(
      tensorflow::tfd::RuntimeFallbackTensor::kTensorType,
      StringHostTensor::kTensorType);

  corert_->RegisterOpHandler(kCpuOpHandlerName, cpu_op_handler.get());
  op_handler_ = corert_->GetOpHandler(kCpuOpHandlerName);
  assert(op_handler_);
}

void CoreRuntimeDriver::InitializeCpuKernelFallbackOpHandler() {
  auto fallback_op_handler = tensorflow::tfd::CreateKernelFallbackOpHandler(
      corert_.get(), corert_->GetHostContext()->GetHostDeviceRef());

  auto cpu_device = corert_->GetHostContext()->GetHostDeviceRef();
  auto cpu_op_handler = tfrt::CreateCpuOpHandler(
      corert_.get(), std::move(cpu_device), fallback_op_handler.get());

  cpu_op_handler.get()->AddImplicitConversion(
      tensorflow::KernelFallbackTensor::kTensorType,
      DenseHostTensor::kTensorType);
  cpu_op_handler.get()->AddImplicitConversion(
      tensorflow::KernelFallbackTensor::kTensorType,
      AnyScalarHostTensor::kTensorType);
  cpu_op_handler.get()->AddImplicitConversion(
      tensorflow::KernelFallbackTensor::kTensorType,
      StringHostTensor::kTensorType);

  corert_->RegisterOpHandler(kCpuOpHandlerName, cpu_op_handler.get());
  op_handler_ = corert_->GetOpHandler(kCpuOpHandlerName);
  assert(op_handler_);
}

ExecutionContext CoreRuntimeDriver::CreateExecutionContext(
    tfrt::string_view filename, int line) {
  auto exec_ctx = tensorflow::tfd::CreateFallbackTestExecutionContext(
      GetHost(), &resource_context_);

  std::pair<std::string, int> loc{filename, line};

  if (!location_map_.contains(loc)) {
    int index = locations_.size();
    location_map_[loc] = index;
    locations_.push_back(loc);
  }
  tfrt::Location location(this, location_map_[loc]);

  exec_ctx.set_location(location);

  return exec_ctx;
}
void CoreRuntimeDriver::Execute(
    tfrt::string_view op_name, tfrt::MutableArrayRef<tfrt::TensorHandle> args,
    const tfrt::OpAttrsRef& attrs,
    tfrt::MutableArrayRef<tfrt::TensorHandle> results,
    tfrt::string_view filename, int line) {
  corert_->Execute(CreateExecutionContext(filename, line), op_name, op_handler_,
                   args, attrs, results, &chain_);
}

CoreRuntimeOp CoreRuntimeDriver::MakeOp(string_view op_name) {
  auto handle = corert_->MakeOp(op_name, op_handler_);
  assert(handle);
  return std::move(handle.get());
}

HostContext* CoreRuntimeDriver::GetHost() const {
  return corert_->GetHostContext();
}

void CoreRuntimeDriver::WaitForHostContextQuiesce() {
  corert_->GetHostContext()->Quiesce();
}

DecodedLocation CoreRuntimeDriver::DecodeLocation(Location loc) const {
  DecodedLocation decoded = FileLineColLocation{locations_[loc.data].first,
                                                locations_[loc.data].second};
  return decoded;
}

}  // namespace tfrt
