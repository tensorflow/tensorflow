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

#ifndef XLA_BACKENDS_GPU_FFI_COLLECTIVES_H_
#define XLA_BACKENDS_GPU_FFI_COLLECTIVES_H_

#include "xla/backends/gpu/runtime/collective_clique_requests.h"
#include "xla/backends/gpu/runtime/collective_cliques.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/ffi/api/collectives_c_api.h"

namespace xla::gpu {

// Per-invocation collective state read by the collectives FFI extension
// callbacks via `XLA_FFI_Collectives_Extension::state`. Pointers are non-owning
// and only valid for the stage they belong to: `collective_clique_requests` is
// set in Prepare, `collective_cliques` once cliques are acquired.
struct GpuCollectivesState {
  const CollectiveParams* collective_params = nullptr;
  CollectiveCliqueRequests* collective_clique_requests = nullptr;
  const CollectiveCliques* collective_cliques = nullptr;
};

// Builds a collectives FFI extension whose callbacks read `state`. Borrows
// `state`, which must outlive the returned extension (both are typically stack
// locals for the duration of the invocation).
XLA_FFI_Collectives_Extension MakeCollectivesExtension(
    GpuCollectivesState* state);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_FFI_COLLECTIVES_H_
