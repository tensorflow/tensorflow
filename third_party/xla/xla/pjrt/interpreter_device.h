/* Copyright 2020 The OpenXLA Authors.

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

#ifndef XLA_PJRT_INTERPRETER_DEVICE_H_
#define XLA_PJRT_INTERPRETER_DEVICE_H_

#include <memory>

#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/statusor.h"

namespace xla {

class InterpreterDevice : public PjRtStreamExecutorDevice {
 public:
  InterpreterDevice(int id,
                    std::unique_ptr<LocalDeviceState> local_device_state);
};

absl::StatusOr<std::unique_ptr<PjRtClient>> GetInterpreterClient();

}  // namespace xla

#endif  // XLA_PJRT_INTERPRETER_DEVICE_H_
