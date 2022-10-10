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
#ifndef TENSORFLOW_CORE_TFRT_EAGER_CORE_RUNTIME_OP_HANDLER_REGISTRY_H_
#define TENSORFLOW_CORE_TFRT_EAGER_CORE_RUNTIME_OP_HANDLER_REGISTRY_H_

#include "tfrt/support/forward_decls.h"  // from @tf_runtime

namespace tensorflow {
class DeviceMgr;
}  // namespace tensorflow

namespace tfrt {

class CoreRuntime;
class ResourceContext;

namespace tf {

using ::tensorflow::DeviceMgr;

// TODO(fishx): Change the second argument to tfrt::DeviceManager and move this
// file into TFRT.
using OpHandlerRegistrationFn = void (*)(CoreRuntime* core_runtime,
                                         ResourceContext* resource_context,
                                         const DeviceMgr* device_mgr);

// This is called to register all OpHandlers into the given core_runtime.
void RegisterOpHandlers(CoreRuntime* core_runtime,
                        ResourceContext* resource_context,
                        const DeviceMgr* device_mgr);

// A helper class for registering a new OpHandler registration function.
struct OpHandlerRegistration {
  explicit OpHandlerRegistration(OpHandlerRegistrationFn fn);
};

}  // namespace tf
}  // namespace tfrt

#endif  // TENSORFLOW_CORE_TFRT_EAGER_CORE_RUNTIME_OP_HANDLER_REGISTRY_H_
