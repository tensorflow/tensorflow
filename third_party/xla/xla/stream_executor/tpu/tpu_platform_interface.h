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

#ifndef XLA_STREAM_EXECUTOR_TPU_TPU_PLATFORM_INTERFACE_H_
#define XLA_STREAM_EXECUTOR_TPU_TPU_PLATFORM_INTERFACE_H_

#include <cstdint>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/tpu/c_api_decl.h"
#include "xla/stream_executor/tpu/tpu_topology.h"

namespace tensorflow {
namespace tpu {

// TODO(skyewm): get rid of TpuTopologyPtr and either use SE_TpuTopology* or
// return a TpuTopologyExternal.
typedef SE_TpuTopology* TpuTopologyPtr;

class TpuPlatformInterface : public stream_executor::Platform {
 public:
  // Returns a TPU platform to be used by TPU ops. If multiple TPU platforms are
  // registered, finds the most suitable one. Returns nullptr if no TPU platform
  // is registered or an error occurred.
  //
  // 'initialize_platform' can be set to false to not initialize a platform if
  // not necessary. 'num_tries' specifies the number of tries if the TPU
  // platform isn't initialized yet, with a 1-second delay between each try
  // (num_tries == 1 means try once with no retries).
  static TpuPlatformInterface* GetRegisteredPlatform(
      bool initialize_platform = true, int num_tries = 5);

  virtual absl::Status Reset(bool only_tear_down, absl::string_view reason) = 0;

  absl::Status Reset(absl::string_view reason) { return Reset(false, reason); }

  absl::Status Reset() { return Reset(false, {}); }

  virtual bool ShouldRegisterTpuDeviceToDeviceCopy() = 0;

  virtual const TpuTopologyPtr GetTopologyPtr() = 0;

  virtual const TpuHostLocationExternal GetTpuHostLocation() const = 0;

  virtual TpuRuntimeVersion version() const = 0;

  virtual void EraseEvent(stream_executor::Event* key) {};

  TpuTopologyExternal topology() {
    return TpuTopologyExternal(GetTopologyPtr());
  }
};

}  // namespace tpu
}  // namespace tensorflow

#endif  // XLA_STREAM_EXECUTOR_TPU_TPU_PLATFORM_INTERFACE_H_
