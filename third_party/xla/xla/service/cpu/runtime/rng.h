// Copyright 2023 The OpenXLA Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef XLA_SERVICE_CPU_RUNTIME_RNG_H_
#define XLA_SERVICE_CPU_RUNTIME_RNG_H_

#include <cstdint>

#include "absl/status/status.h"
#include "xla/executable_run_options.h"
#include "xla/runtime/memref_view.h"

namespace xla {
namespace cpu {

struct XlaThreeFry {
  absl::Status operator()(const ExecutableRunOptions*,
                          xla::runtime::FlatMemrefView state_buffer,
                          xla::runtime::FlatMemrefView state_out_buffer,
                          xla::runtime::FlatMemrefView values_buffer) const;
  static XlaThreeFry Handler() { return XlaThreeFry(); }
};

struct XlaPhilox {
  absl::Status operator()(const ExecutableRunOptions*,
                          xla::runtime::FlatMemrefView state_buffer,
                          xla::runtime::FlatMemrefView state_out_buffer,
                          xla::runtime::FlatMemrefView values_buffer) const;
  static XlaPhilox Handler() { return XlaPhilox(); }
};

}  // namespace cpu
}  // namespace xla

#endif  // XLA_SERVICE_CPU_RUNTIME_RNG_H_
