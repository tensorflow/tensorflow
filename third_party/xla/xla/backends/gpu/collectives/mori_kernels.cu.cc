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

// This is the single device translation unit for the MORI XLA collectives. It
// is compiled as HIP and defines MORI_KERNELS_IMPL before including the facade,
// so the facade's device path (kernels + non-templated Run* definitions) is
// compiled here exactly once. The host mori_communicator.cc includes the same
// header without MORI_KERNELS_IMPL (decl-only) and links against these symbols.
#define MORI_KERNELS_IMPL
#include "xla/backends/gpu/collectives/mori_kernels.h"
