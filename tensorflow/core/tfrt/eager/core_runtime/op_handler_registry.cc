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

#include "tensorflow/core/tfrt/eager/core_runtime/op_handler_registry.h"

#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tfrt/core_runtime/core_runtime.h"  // from @tf_runtime
#include "tfrt/host_context/resource_context.h"  // from @tf_runtime

namespace tfrt {
namespace tf {

static std::vector<OpHandlerRegistrationFn>* GetStaticOpHandlerRegistrations() {
  static std::vector<OpHandlerRegistrationFn>* ret =
      new std::vector<OpHandlerRegistrationFn>;
  return ret;
}

void RegisterOpHandlers(CoreRuntime* core_runtime,
                        ResourceContext* resource_context,
                        const DeviceMgr* device_mgr) {
  for (auto fn : *GetStaticOpHandlerRegistrations()) {
    fn(core_runtime, resource_context, device_mgr);
  }
}

OpHandlerRegistration::OpHandlerRegistration(OpHandlerRegistrationFn fn) {
  GetStaticOpHandlerRegistrations()->emplace_back(fn);
}

}  // namespace tf
}  // namespace tfrt
