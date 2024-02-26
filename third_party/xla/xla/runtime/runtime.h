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

#ifndef XLA_RUNTIME_RUNTIME_H_
#define XLA_RUNTIME_RUNTIME_H_

#include <stdint.h>

namespace xla {
namespace runtime {

//===----------------------------------------------------------------------===//
// XLA Runtime <-> XLA Executable integration API.
//===----------------------------------------------------------------------===//

// This API enables compiled XLA executables to call back into the XLA runtime
// for:
//
//  - Returning results back to the caller by writing to result storage
//    allocated in the call frame.
//  - User friendly error reporting integrated with the high level execution
//    model (errors do not crash the compiled binary).
//  - Invoking custom calls registered with the XLA runtime.
//
// XLA compilation pipeline sets up passes to convert the regular functions
// to the so called "XLA entrypoint functions" integrated with the runtime using
// the API defined below, e.g. instead of conventional returns all results are
// returned via the `GetResultStorage` API. At MLIR level these operations
// correspond to the `rt` dialect, and converted to LLVM using the `rt-to-llvm`
// conversion pass.
//
// Runtime errors are reported back to the runtime via the `SetError` API.
// The compilation pipeline will automatically convert assertions in the
// entrypoint function into run-time errors.

// Opaque runtime execution context passed as the first argument to compiled
// executables and passed back to all runtime API methods.
using ExecutionContext = struct ExecutionContext;

// Returns a pointer to the memory location of the result at the given index.
void *GetResultStorage(ExecutionContext *, int64_t);

// Sets execution context to an error state.
void SetError(ExecutionContext *, const char *);

// Calls the custom call function registered with the runtime. Returns true
// if the custom call was successful.
bool CustomCall(ExecutionContext *, const char *target, void **args,
                void **attrs, void **rets);

}  // namespace runtime
}  // namespace xla

#endif  // XLA_RUNTIME_RUNTIME_H_
