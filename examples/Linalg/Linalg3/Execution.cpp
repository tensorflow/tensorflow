//===- Conversion.cpp - Linalg to LLVM execution driver -------------------===//
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

#include "TestHarness.h"

#include "linalg1/Common.h"
#include "linalg1/Dialect.h"
#include "linalg2/Intrinsics.h"
#include "linalg3/ConvertToLLVMDialect.h"
#include "linalg3/Ops.h"
#include "linalg3/Transforms.h"

#include "llvm/Support/TargetSelect.h"

#include "mlir/ExecutionEngine/ExecutionEngine.h"

// RUN: %p/execution | FileCheck %s

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;
using namespace linalg;
using namespace linalg::common;
using namespace linalg::intrinsics;

Function *makeFunctionWithAMatmulOp(Module &module, StringRef name) {
  MLIRContext *context = module.getContext();
  auto dynamic2DMemRefType = floatMemRefType<2>(context);
  mlir::Function *f = linalg::common::makeFunction(
      module, name,
      {dynamic2DMemRefType, dynamic2DMemRefType, dynamic2DMemRefType}, {});

  ScopedContext scope(f);
  // clang-format off
  ValueHandle
    M = dim(f->getArgument(0), 0),
    N = dim(f->getArgument(2), 1),
    K = dim(f->getArgument(0), 1),
    rM = range(constant_index(0), M, constant_index(1)),
    rN = range(constant_index(0), N, constant_index(1)),
    rK = range(constant_index(0), K, constant_index(1)),
    vA = view(f->getArgument(0), {rM, rK}),
    vB = view(f->getArgument(1), {rK, rN}),
    vC = view(f->getArgument(2), {rM, rN});
  matmul(vA, vB, vC);
  ret();
  // clang-format on

  return f;
}

// Representation of a Memref descriptor for a 2D dynamically-sized Memref in C.
// This is equivalent to the structure that the conversion produces.
struct MemRefDescriptor2D {
  float *ptr;
  int64_t sz1;
  int64_t sz2;
};

// Alocate a 2D memref of the given size, store the sizes in the descriptor and
// initialize all values with 1.0f.
static MemRefDescriptor2D allocateInit2DMemref(int64_t sz1, int64_t sz2) {
  MemRefDescriptor2D descriptor;
  descriptor.ptr = static_cast<float *>(malloc(sizeof(float) * sz1 * sz2));
  descriptor.sz1 = sz1;
  descriptor.sz2 = sz2;
  for (int64_t i = 0, e = sz1 * sz2; i < e; ++i)
    descriptor.ptr[i] = 1.0f;
  return descriptor;
}

// Print the contents of the memref given its descriptor.
static void print2DMemref(const MemRefDescriptor2D &descriptor) {
  for (int64_t i = 0; i < descriptor.sz1; ++i) {
    llvm::outs() << '[';
    for (int64_t j = 0; j < descriptor.sz2; ++j) {
      if (j != 0)
        llvm::outs() << ", ";
      llvm::outs() << descriptor.ptr[i * descriptor.sz2 + j];
    }
    llvm::outs() << "]\n";
  }
}

// Free a 2D memref given its descriptor.  Resets the pointer in the descriptor
// to nullptr.
static void free2DMemref(MemRefDescriptor2D &descriptor) {
  free(descriptor.ptr);
  descriptor.ptr = nullptr;
}

TEST_FUNC(execution) {
  // Create an MLIR module, create a function "matmul_as_loops" containing a
  // linalg.matmul operation and lower it all the way down to the LLVM IR
  // dialect through partial conversions.
  MLIRContext context;
  Module module(&context);
  mlir::Function *f = makeFunctionWithAMatmulOp(module, "matmul_as_loops");
  lowerToLoops(f);
  convertLinalg3ToLLVM(module);

  // Create an MLIR execution engine.  Note that it takes a null pass manager
  // to make sure it won't run "default" passes on the MLIR that would trigger
  // a second conversion to LLVM IR.  The execution engine eagerly JIT-compiles
  // the module.
  auto maybeEngine = mlir::ExecutionEngine::create(&module, /*pm=*/nullptr);
  assert(maybeEngine && "failed to construct an execution engine");
  auto &engine = maybeEngine.get();

  // Prepare arguments for the function invocation: allocate input and output
  // buffers.
  auto A = allocateInit2DMemref(5, 3);
  auto B = allocateInit2DMemref(3, 2);
  auto C = allocateInit2DMemref(5, 2);
  llvm::SmallVector<void *, 4> args;
  args.push_back(&A);
  args.push_back(&B);
  args.push_back(&C);

  // Invoke the JIT-compiled function with the arguments.  Note that, for API
  // uniformity reasons, it takes a list of type-erased pointers to arguments.
  auto invocationResult =
      engine->invoke("matmul_as_loops", MutableArrayRef<void *>(args));
  assert(!invocationResult && "call failed");

  // clang-format off
  // CHECK:      [3.000000e+00, 3.000000e+00]
  // CHECK-NEXT: [3.000000e+00, 3.000000e+00]
  // CHECK-NEXT: [3.000000e+00, 3.000000e+00]
  // CHECK-NEXT: [3.000000e+00, 3.000000e+00]
  // CHECK-NEXT: [3.000000e+00, 3.000000e+00]
  // clang-format on
  print2DMemref(C);

  // Cleanup.
  free2DMemref(A);
  free2DMemref(B);
  free2DMemref(C);
}

int main() {
  mlir::registerDialect<linalg::LinalgDialect>();

  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  RUN_TESTS();
  return 0;
}
