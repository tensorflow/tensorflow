/* Copyright 2026 The OpenXLA Authors.
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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_MORI_KERNELS_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_MORI_KERNELS_H_

#include <cstddef>
#include <cstdint>

// The CollectivesFacade owns the per-device staging + Run* entry points, which
// are non-templated and take mori::collective::DataType / ReduceOpKind enums.
// Host includers (mori_communicator.cc) see decl-only Run* methods; the device
// TU (mori_kernels.cu.cc, compiled as HIP with MORI_KERNELS_IMPL) pulls in the
// full device path and emits the definitions that resolve the host's
// references.
#include "xla/backends/gpu/collectives/mori_stub.h"

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_MORI_KERNELS_H_
