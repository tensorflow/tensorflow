//===- ExecutionEngine.h - MLIR Execution engine and utils -----*- C++ -*--===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file provides a JIT-backed execution engine for MLIR modules.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_EXECUTIONENGINE_EXECUTIONENGINE_H_
#define MLIR_EXECUTIONENGINE_EXECUTIONENGINE_H_

#include "mlir/Support/LLVM.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/Error.h"

#include <functional>
#include <memory>

namespace llvm {
template <typename T> class Expected;
class Module;
}

namespace mlir {

class Module;

namespace impl {
class OrcJIT;
} // end namespace impl

/// JIT-backed execution engine for MLIR modules.  Assumes the module can be
/// converted to LLVM IR.  For each function, creates a wrapper function with
/// the fixed interface
///
///     void _mlir_funcName(void **)
///
/// where the only argument is interpreted as a list of pointers to the actual
/// arguments of the function, followed by a pointer to the result.  This allows
/// the engine to provide the caller with a generic function pointer that can
/// be used to invoke the JIT-compiled function.
class ExecutionEngine {
public:
  ~ExecutionEngine();

  /// Creates an execution engine for the given module.  If `transformer` is
  /// provided, it will be called on the LLVM module during JIT-compilation and
  /// can be used, e.g., for reporting or optimization.
  static llvm::Expected<std::unique_ptr<ExecutionEngine>>
  create(Module *m,
         std::function<llvm::Error(llvm::Module *)> transformer = {});

  /// Looks up a packed-argument function with the given name and returns a
  /// pointer to it.  Propagates errors in case of failure.
  llvm::Expected<void (*)(void **)> lookup(StringRef name) const;

  /// Invokes the function with the given name passing it the list of arguments.
  /// The arguments are accepted by lvalue-reference since the packed function
  /// interface expects a list of non-null pointers.
  template <typename... Args>
  llvm::Error invoke(StringRef name, Args &... args);

  /// Invokes the function with the given name passing it the list of arguments
  /// as a list of opaque pointers. This is the arity-agnostic equivalent of
  /// the templated `invoke`.
  llvm::Error invoke(StringRef name, MutableArrayRef<void *> args);

private:
  // Ordering of llvmContext and jit is important for destruction purposes: the
  // jit must be destroyed before the context.
  llvm::LLVMContext llvmContext;
  // Private implementation of the JIT (PIMPL)
  std::unique_ptr<impl::OrcJIT> jit;
};

template <typename... Args>
llvm::Error ExecutionEngine::invoke(StringRef name, Args &... args) {
  auto expectedFPtr = lookup(name);
  if (!expectedFPtr)
    return expectedFPtr.takeError();
  auto fptr = *expectedFPtr;

  llvm::SmallVector<void *, 8> packedArgs{static_cast<void *>(&args)...};
  (*fptr)(packedArgs.data());

  return llvm::Error::success();
}

} // end namespace mlir

#endif // MLIR_EXECUTIONENGINE_EXECUTIONENGINE_H_
