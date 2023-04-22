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

#ifndef TENSORFLOW_COMPILER_XLA_CLIENT_GLOBAL_DATA_H_
#define TENSORFLOW_COMPILER_XLA_CLIENT_GLOBAL_DATA_H_

#include <memory>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/service_interface.h"
#include "tensorflow/compiler/xla/xla.pb.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/macros.h"

namespace xla {

// A GlobalData object represents a globally-accessible allocation of
// data in the associated XLA service.
class GlobalData {
 public:
  // Gives ownership of the global data handle to this object.
  GlobalData(ServiceInterface* parent, GlobalDataHandle handle);

  // Unregisters the wrapped handle, which causes the service to
  // deallocate the associated data.
  ~GlobalData();

  const GlobalDataHandle& handle() const { return handle_; }

  // Releases a set of GlobalData handles. A single RPC will be issued
  // per unique ServiceInterface of the given GlobalData objects.
  static void Release(std::vector<std::unique_ptr<GlobalData>> instances);

 private:
  // Detaches the global data handle from the object, such that the destructor
  // will not try to release it.
  GlobalDataHandle Release() {
    parent_ = nullptr;
    return handle_;
  }

  GlobalDataHandle handle_;   // Handle being wrapped.
  ServiceInterface* parent_;  // Service used to unregister handle_.

  TF_DISALLOW_COPY_AND_ASSIGN(GlobalData);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_CLIENT_GLOBAL_DATA_H_
