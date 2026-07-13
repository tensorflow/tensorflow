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

#ifndef XLA_BACKENDS_CPU_RUNTIME_THUNK_SERDES_COLLECTIVE_THUNK_SERDES_H_
#define XLA_BACKENDS_CPU_RUNTIME_THUNK_SERDES_COLLECTIVE_THUNK_SERDES_H_

namespace xla::cpu {

// Registers the CollectiveThunk (and its specific variants)
// serialization/deserialization logic with the ThunkSerDesRegistry.
void RegisterCollectiveThunkSerDes();

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_THUNK_SERDES_COLLECTIVE_THUNK_SERDES_H_
