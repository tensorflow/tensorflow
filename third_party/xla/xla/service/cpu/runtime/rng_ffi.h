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
#ifndef XLA_SERVICE_CPU_RUNTIME_RNG_FFI_H_
#define XLA_SERVICE_CPU_RUNTIME_RNG_FFI_H_

extern "C" {
bool xla_cpu_rng_three_fry(void* execution_context, void** args, void** attrs,
                           void** rets);
bool xla_cpu_rng_philox(void* execution_context, void** args, void** attrs,
                        void** rets);
}  // extern "C"

#endif  // XLA_SERVICE_CPU_RUNTIME_RNG_FFI_H_
