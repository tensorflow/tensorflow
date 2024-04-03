/* Copyright 2022 The OpenXLA Authors.

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

#ifndef XLA_RUNTIME_COMPILER_H_
#define XLA_RUNTIME_COMPILER_H_

namespace xla {
namespace runtime {

// Dialect registry is a container for registering dialects supported by the Xla
// runtime compilation pipeline.
class DialectRegistry;  // NOLINT

// The main Xla runtime pass manager and compilation pipeline builder.
class PassManager;  // NOLINT

}  // namespace runtime
}  // namespace xla

#endif  // XLA_RUNTIME_COMPILER_H_
