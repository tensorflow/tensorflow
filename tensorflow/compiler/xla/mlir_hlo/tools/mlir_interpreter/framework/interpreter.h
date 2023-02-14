/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef MLIR_HLO_TOOLS_MLIR_INTERPRETER_FRAMEWORK_INTERPRETER_H_
#define MLIR_HLO_TOOLS_MLIR_INTERPRETER_FRAMEWORK_INTERPRETER_H_

#include <algorithm>
#include <complex>
#include <cstddef>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LLVM.h"
#include "tools/mlir_interpreter/framework/interpreter_value.h"

namespace mlir {
namespace interpreter {

class InterpreterScope;

class InterpreterListener {
 public:
  virtual ~InterpreterListener() = default;
  virtual void beforeOp(ArrayRef<InterpreterValue> args, mlir::Operation*) {}
  virtual void afterOp(ArrayRef<InterpreterValue> results) {}
  virtual void enterRegion(ArrayRef<InterpreterValue> args,
                           mlir::Region& region) {}
  virtual void leaveRegion(ArrayRef<InterpreterValue> terminatorArgs) {}
};

struct InterpreterStats {
  // Memrefs only.
  int64_t heapSize = 0;
  int64_t peakHeapSize = 0;
  int64_t numAllocations = 0;
  int64_t numDeallocations = 0;
};

struct InterpreterOptions {
  InterpreterListener* listener = nullptr;
  std::optional<int64_t> maxSteps = std::nullopt;
  // If set, ignore deallocations. Normally, accessing a deallocated memref will
  // trigger an assertion. This flag disables all alloctions, which can be
  // useful when debugging IR that includes a use-after-free bug.
  bool disableDeallocations = false;
  std::function<void(llvm::StringRef)> errorHandler =
      [](llvm::StringRef failure) {
        llvm::errs() << "Interpreter failure: " << failure << "\n";
      };
  InterpreterStats* stats = nullptr;
};

class InterpreterState {
 public:
  InterpreterState(const mlir::SymbolTable& symbols,
                   InterpreterOptions options);

  void step() {
    if (remainingSteps == 0) {
      addFailure("maximum number of steps exceeded");
      return;
    }
    --remainingSteps;
  }
  void addFailure(llvm::StringRef failure);
  bool hasFailure() const { return failed; }
  void checkSuccess(LogicalResult result, llvm::StringRef failure) {
    if (!result.succeeded()) {
      addFailure(failure);
    }
  }

  InterpreterScope* getTopScope() { return topScope; }
  const mlir::SymbolTable& getSymbols() const { return symbols; }
  const InterpreterOptions& getOptions() { return options; }

 private:
  const mlir::SymbolTable& symbols;
  InterpreterScope* topScope = nullptr;
  bool failed = false;
  InterpreterOptions options;
  int64_t remainingSteps = std::numeric_limits<int64_t>::max();

  friend class InterpreterScope;
  friend class InterpreterScopeStash;
};

// Used for passing arbitrary data to ops in sub-regions.
class InterpreterSideChannel {
 public:
  virtual ~InterpreterSideChannel() = default;
};

// Holds a mapping from SSA values to InterpreterValues and registered side
// channels. There's typically one scope per region, but ops can add additional
// scopes if needed (for example, to register a side channel).
class InterpreterScope {
 public:
  InterpreterScope(InterpreterScope&&) = delete;
  explicit InterpreterScope(InterpreterState& state)
      : state(state), parentScope(state.topScope) {
    state.topScope = this;
  }
  ~InterpreterScope();

  void Set(Value v, InterpreterValue iv) { values[v] = std::move(iv); }

  const InterpreterValue& Get(Value v) {
    auto ret = values.find(v);
    if (ret == values.end()) {
      if (!parentScope) {
        v.dump();
      }

      assert(parentScope && "value not found");
      return parentScope->Get(v);
    }
    return ret->second;
  }

  void verify() const;

  // Retrieves the side channel of the given type in this scope or one of its
  // ancestor scopes. If `optional` is set, returns nullptr if not found,
  // otherwise asserts.
  template <typename T>
  T* getSideChannel(bool optional = false) {
    for (auto& sideChannel : sideChannels) {
      if (auto it = dynamic_cast<T*>(sideChannel.get())) {
        return it;
      }
    }
    if (!parentScope && optional) return nullptr;
    assert(parentScope && "side channel not found");
    return parentScope->getSideChannel<T>(optional);
  }

  // Registers the given side channel. Will shadow a side channel of the same
  // type if registered in an outer scope.
  // The behavior of registering two side channels of the same type in the same
  // scope is undefined.
  void setSideChannel(std::shared_ptr<InterpreterSideChannel> sideChannel) {
    sideChannels.push_back(std::move(sideChannel));
  }

  InterpreterScope* getParentScope() const { return parentScope; }

 private:
  DenseMap<Value, InterpreterValue> values;
  SmallVector<std::shared_ptr<InterpreterSideChannel>> sideChannels;

  InterpreterState& state;
  InterpreterScope* parentScope;

  friend class InterpreterScopeStash;
};

// Interprets the given region and returns the terminator's arguments. The
// region must have a single block.
SmallVector<InterpreterValue> interpret(InterpreterState& state, Region& region,
                                        ArrayRef<InterpreterValue> bbargs);

// Interprets the given function.
mlir::FailureOr<SmallVector<InterpreterValue>> runInterpreter(
    const mlir::SymbolTable& symbols, mlir::func::FuncOp function,
    ArrayRef<InterpreterValue> args, InterpreterOptions options = {});

}  // namespace interpreter
}  // namespace mlir

#endif  // MLIR_HLO_TOOLS_MLIR_INTERPRETER_FRAMEWORK_INTERPRETER_H_
