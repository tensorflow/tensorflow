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
#ifndef TENSORFLOW_CORE_RUNTIME_FALLBACK_TEST_CORERUNTIME_DRIVER_H_
#define TENSORFLOW_CORE_RUNTIME_FALLBACK_TEST_CORERUNTIME_DRIVER_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tfrt/core_runtime/core_runtime.h"  // from @tf_runtime
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime
#include "tfrt/host_context/chain.h"  // from @tf_runtime
#include "tfrt/host_context/location.h"  // from @tf_runtime
#include "tfrt/host_context/resource_context.h"  // from @tf_runtime

namespace tfrt {

class OpHandle;
class OpHandler;
class OpAttrsRef;
class TensorHandle;

class CoreRuntimeDriver final : public tfrt::LocationHandler {
 public:
  explicit CoreRuntimeDriver();

  void Execute(string_view op_name,
               tfrt::MutableArrayRef<tfrt::TensorHandle> args,
               const tfrt::OpAttrsRef& attrs,
               tfrt::MutableArrayRef<tfrt::TensorHandle> results,
               tfrt::string_view filename, int line);

  ExecutionContext CreateExecutionContext(tfrt::string_view filename, int line);

  void InitializeCpuRuntimeFallbackOpHandler();

  void InitializeGpuRuntimeFallbackOpHandler(int gpu_ordinal);

  void InitializeCpuKernelFallbackOpHandler();

  HostContext* GetHost() const;

  CoreRuntimeOp MakeOp(string_view op_name);

  void WaitForHostContextQuiesce();

  DecodedLocation DecodeLocation(Location loc) const override;

 private:
  explicit CoreRuntimeDriver(std::unique_ptr<tfrt::CoreRuntime> corert);

  std::unique_ptr<tfrt::CoreRuntime> corert_;
  tfrt::OpHandler* op_handler_;
  tfrt::AsyncValueRef<tfrt::Chain> chain_;
  tfrt::ResourceContext resource_context_;

  // `location_map_` is a map from (filename, line) to the opaque location data,
  // which is the index in `locations_`.
  absl::flat_hash_map<std::pair<std::string, int>, int> location_map_;
  std::vector<std::pair<std::string, int>> locations_;
};

}  // namespace tfrt

#endif  // TENSORFLOW_CORE_RUNTIME_FALLBACK_TEST_CORERUNTIME_DRIVER_H_
