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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_INTERPRETER_DEVICE_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_INTERPRETER_DEVICE_H_

#include <memory>

#include "tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

class InterpreterDevice : public PjRtStreamExecutorDevice {
 public:
  InterpreterDevice(int id,
                    std::unique_ptr<LocalDeviceState> local_device_state);
};

StatusOr<std::unique_ptr<PjRtClient>> GetInterpreterClient();

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_INTERPRETER_DEVICE_H_
