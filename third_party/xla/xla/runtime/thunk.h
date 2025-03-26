/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_RUNTIME_THUNK_H_
#define XLA_RUNTIME_THUNK_H_

#include <memory>
#include <vector>

namespace xla {

// Thunk is the unit of execution for the XLA CPU and GPU runtime.
//
// XLA programs compiled to a sequence of backends-specific thunks that run XLA
// operations on the target device. The most common thunk type on both backends
// is a kernel thunk, which launches an XLA fusion compiled to a backend
// specific machine code (i.e. PTX for GPU, x86 assembly for CPU). Operations
// that are not compiled to machine code are also represented as thunks, e.g.
// collective operations which are implemented as library calls.
class Thunk {
 public:
  virtual ~Thunk() = default;
};

// A sequence of owned thunks.
class ThunkSequence : public std::vector<std::unique_ptr<Thunk>> {
 public:
  static ThunkSequence Empty() { return ThunkSequence(); }
};

}  // namespace xla

#endif  // XLA_RUNTIME_THUNK_H_
