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

#include "tensorflow/compiler/xla/service_interface.h"
#include "tensorflow/compiler/xla/xla.pb.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/macros.h"

namespace xla {

// Wraps a GlobalDataHandle with a lifetime.
class GlobalData {
 public:
  // Gives ownership of the global data handle to this object.
  GlobalData(ServiceInterface* parent, GlobalDataHandle handle);

  // Unregisters the wrapped handle.
  ~GlobalData();

  const GlobalDataHandle& handle() const { return handle_; }

 private:
  GlobalDataHandle handle_;   // Handle being wrapped.
  ServiceInterface* parent_;  // Service used to unregister handle_.

  TF_DISALLOW_COPY_AND_ASSIGN(GlobalData);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_CLIENT_GLOBAL_DATA_H_
