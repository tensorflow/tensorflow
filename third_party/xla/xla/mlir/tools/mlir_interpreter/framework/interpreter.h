/* Copyright 2022 The OpenXLA Authors. All Rights Reserved.

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

#ifndef XLA_MLIR_TOOLS_MLIR_INTERPRETER_FRAMEWORK_INTERPRETER_H_
#define XLA_MLIR_TOOLS_MLIR_INTERPRETER_FRAMEWORK_INTERPRETER_H_

#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>

#include "absl/status/statusor.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "xla/mlir/tools/mlir_interpreter/framework/interpreter_value.h"

namespace mlir {
namespace interpreter {

class InterpreterScope;

class InterpreterListener {
 public:
  virtual ~InterpreterListener() = default;
  virtual void BeforeOp(ArrayRef<InterpreterValue> args, mlir::Operation*) {}
  virtual void AfterOp(ArrayRef<InterpreterValue> results) {}
  virtual void EnterRegion(ArrayRef<InterpreterValue> args,
                           mlir::Region& region) {}
  virtual void LeaveRegion(ArrayRef<InterpreterValue> terminator_args) {}
};

struct InterpreterStats {
  // Memrefs only.
  int64_t heap_size = 0;
  int64_t peak_heap_size = 0;
  int64_t num_allocations = 0;
  int64_t num_deallocations = 0;
};

struct InterpreterOptions {
  InterpreterListener* listener = nullptr;
  std::optional<int64_t> max_steps = std::nullopt;
  // If set, ignore deallocations. Normally, accessing a deallocated memref will
  // trigger an assertion. This flag disables all allocations, which can be
  // useful when debugging IR that includes a use-after-free bug.
  bool disable_deallocations = false;
  std::function<void(llvm::StringRef)> error_handler =
      [](llvm::StringRef failure) {
        llvm::errs() << "Interpreter failure: " << failure << "\n";
      };
  InterpreterStats* stats = nullptr;
};

class InterpreterState {
 public:
  InterpreterState(const mlir::SymbolTable& symbols,
                   InterpreterOptions options);

  void Step() {
    if (remaining_steps_ == 0) {
      AddFailure("maximum number of steps exceeded");
      return;
    }
    --remaining_steps_;
  }
  void AddFailure(llvm::StringRef failure);
  bool HasFailure() const { return failed_; }
  void CheckSuccess(LogicalResult result, llvm::StringRef failure) {
    if (!result.succeeded()) {
      AddFailure(failure);
    }
  }

  InterpreterScope* GetTopScope() { return top_scope_; }
  const mlir::SymbolTable& GetSymbols() const { return symbols_; }
  const InterpreterOptions& GetOptions() { return options_; }

 private:
  const mlir::SymbolTable& symbols_;
  InterpreterScope* top_scope_ = nullptr;
  bool failed_ = false;
  InterpreterOptions options_;
  int64_t remaining_steps_ = std::numeric_limits<int64_t>::max();

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
      : state_(state), parent_scope_(state.top_scope_) {
    state.top_scope_ = this;
  }
  ~InterpreterScope();

  void Set(Value v, InterpreterValue iv) { values_[v] = std::move(iv); }

  const InterpreterValue& Get(Value v) {
    auto ret = values_.find(v);
    if (ret == values_.end()) {
      if (!parent_scope_) {
        v.dump();
      }

      assert(parent_scope_ && "value not found");
      return parent_scope_->Get(v);
    }
    return ret->second;
  }

  void Verify() const;

  // Retrieves the side channel of the given type in this scope or one of its
  // ancestor scopes. If `optional` is set, returns nullptr if not found,
  // otherwise asserts.
  template <typename T>
  T* GetSideChannel(bool optional = false) {
    for (auto& side_channel : side_channels_) {
      if (auto it = dynamic_cast<T*>(side_channel.get())) {
        return it;
      }
    }
    if (!parent_scope_ && optional) return nullptr;
    assert(parent_scope_ && "side channel not found");
    return parent_scope_->GetSideChannel<T>(optional);
  }

  // Registers the given side channel. Will shadow a side channel of the same
  // type if registered in an outer scope.
  // The behavior of registering two side channels of the same type in the same
  // scope is undefined.
  void SetSideChannel(std::shared_ptr<InterpreterSideChannel> side_channel) {
    side_channels_.push_back(std::move(side_channel));
  }

  InterpreterScope* GetParentScope() const { return parent_scope_; }

 private:
  DenseMap<Value, InterpreterValue> values_;
  SmallVector<std::shared_ptr<InterpreterSideChannel>> side_channels_;

  InterpreterState& state_;
  InterpreterScope* parent_scope_;

  friend class InterpreterScopeStash;
};

// Interprets the given region and returns the terminator's arguments. The
// region must have a single block.
SmallVector<InterpreterValue> Interpret(InterpreterState& state, Region& region,
                                        ArrayRef<InterpreterValue> bbargs);

// Interprets the given function.
absl::StatusOr<SmallVector<InterpreterValue>> RunInterpreter(
    const mlir::SymbolTable& symbols, mlir::func::FuncOp function,
    ArrayRef<InterpreterValue> args, InterpreterOptions options = {});

}  // namespace interpreter
}  // namespace mlir

#endif  // XLA_MLIR_TOOLS_MLIR_INTERPRETER_FRAMEWORK_INTERPRETER_H_
