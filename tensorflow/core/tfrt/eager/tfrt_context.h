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
#ifndef TENSORFLOW_CORE_TFRT_EAGER_TFRT_CONTEXT_H_
#define TENSORFLOW_CORE_TFRT_EAGER_TFRT_CONTEXT_H_

#include "tensorflow/c/eager/immediate_execution_context.h"
#include "tensorflow/core/public/session_options.h"
#include "tfrt/host_context/resource_context.h"  // from @tf_runtime

namespace tensorflow {
class EagerContext;
class DynamicDeviceMgr;
}
namespace tfrt {
class HostContext;
class CoreRuntime;
class OpHandler;

namespace tf {

// This class defines a list of objects needed to support execution with TFRT.
class TfrtContext {
 public:
  TfrtContext(
      const tensorflow::SessionOptions& opts,
      tensorflow::ContextDevicePlacementPolicy default_device_placement_policy,
      bool is_async);
  ~TfrtContext();

  HostContext* GetHostContext() { return host_context_; }
  CoreRuntime* GetCoreRuntime() { return corert_.get(); }
  tensorflow::EagerContext* GetEagerContext() { return eager_context_; }
  const tensorflow::EagerContext* GetEagerContext() const {
    return eager_context_;
  }
  OpHandler* GetFallbackOpHandler() { return fallback_op_handler_; }

  ResourceContext* GetResourceContext() { return &resource_context_; }

  const tensorflow::DeviceNameUtils::ParsedName& HostCPUParsedName() const;

  bool IsAsync() const;

 private:
  std::unique_ptr<CoreRuntime> corert_;
  ::tfrt::HostContext* host_context_;
  OpHandler* fallback_op_handler_;
  ResourceContext resource_context_;
  tensorflow::EagerContext* eager_context_;
};

}  // namespace tf
}  // namespace tfrt

#endif  // TENSORFLOW_CORE_TFRT_EAGER_TFRT_CONTEXT_H_
