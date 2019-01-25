//===- MemRefUtils.cpp - MLIR runtime utilities for memrefs ---------------===//
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
// This is a set of utilities to working with objects of memref type in an JIT
// context using the MLIR execution engine.
//
//===----------------------------------------------------------------------===//

#include "mlir/ExecutionEngine/MemRefUtils.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Support/LLVM.h"

#include "llvm/Support/Error.h"

using namespace mlir;

static inline llvm::Error make_string_error(const llvm::Twine &message) {
  return llvm::make_error<llvm::StringError>(message.str(),
                                             llvm::inconvertibleErrorCode());
}

static llvm::Expected<StaticFloatMemRef *>
allocMemRefDescriptor(Type type, bool allocateData = true,
                      float initialValue = 0.0) {
  auto memRefType = type.dyn_cast<MemRefType>();
  if (!memRefType)
    return make_string_error("non-memref argument not supported");
  if (memRefType.getNumDynamicDims() != 0)
    return make_string_error("memref with dynamic shapes not supported");

  auto elementType = memRefType.getElementType();
  if (!elementType.isF32())
    return make_string_error(
        "memref with element other than f32 not supported");

  auto *descriptor =
      reinterpret_cast<StaticFloatMemRef *>(malloc(sizeof(StaticFloatMemRef)));
  if (!allocateData) {
    descriptor->data = nullptr;
    return descriptor;
  }

  auto shape = memRefType.getShape();
  int64_t size = std::accumulate(shape.begin(), shape.end(), 1,
                                 std::multiplies<int64_t>());
  descriptor->data = reinterpret_cast<float *>(malloc(sizeof(float) * size));
  for (int64_t i = 0; i < size; ++i) {
    descriptor->data[i] = initialValue;
  }
  return descriptor;
}

llvm::Expected<SmallVector<void *, 8>>
mlir::allocateMemRefArguments(const Function *func, float initialValue) {
  SmallVector<void *, 8> args;
  args.reserve(func->getNumArguments());
  for (const auto &arg : func->getArguments()) {
    auto descriptor =
        allocMemRefDescriptor(arg->getType(),
                              /*allocateData=*/true, initialValue);
    if (!descriptor)
      return descriptor.takeError();
    args.push_back(*descriptor);
  }

  if (func->getType().getNumResults() > 1)
    return make_string_error("functions with more than 1 result not supported");

  for (Type resType : func->getType().getResults()) {
    auto descriptor = allocMemRefDescriptor(resType, /*allocateData=*/false);
    if (!descriptor)
      return descriptor.takeError();
    args.push_back(*descriptor);
  }

  return args;
}

// Because the function can return the same descriptor as passed in arguments,
// we check that we don't attempt to free the underlying data twice.
void mlir::freeMemRefArguments(ArrayRef<void *> args) {
  llvm::DenseSet<void *> dataPointers;
  for (void *arg : args) {
    float *dataPtr = reinterpret_cast<StaticFloatMemRef *>(arg)->data;
    if (dataPointers.count(dataPtr) == 0) {
      free(dataPtr);
      dataPointers.insert(dataPtr);
    }
    free(arg);
  }
}
