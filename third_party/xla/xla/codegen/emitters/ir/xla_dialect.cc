/* Copyright 2024 The OpenXLA Authors.

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

#include <cstdint>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"  // IWYU pragma: keep
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"  // IWYU pragma: keep
#include "mlir/IR/OpImplementation.h"  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/InliningUtils.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/codegen/emitters/type_util.h"

// The order of these includes is important.
#define GET_ATTRDEF_CLASSES
#include "xla/codegen/emitters/ir/xla_attrs.cc.inc"
#include "xla/codegen/emitters/ir/xla_enums.cc.inc"

namespace xla {
namespace {

constexpr int64_t kMaxFuncSize = 4000;

int64_t GetNumOps(mlir::Block& block) { return block.getOperations().size(); }

struct XlaInlinerInterface : public mlir::DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;
  // Returns true if the given operation 'callable', that implements the
  // 'CallableOpInterface', can be inlined into the position given call
  // operation 'call', that is registered to the current dialect and implements
  // the `CallOpInterface`. 'wouldBeCloned' is set to true if the region of the
  // given 'callable' is set to be cloned during the inlining process, or false
  // if the region is set to be moved in-place (i.e. no duplicates would be
  // created).
  bool isLegalToInline(mlir::Operation* call, mlir::Operation* callable,
                       bool wouldBeCloned) const final {
    if (call->hasAttr("noinline")) return false;
    if (callable->hasAttr(emitters::kHasNoCompute)) return true;
    // Otherwise, inline only if the called function is small. We could
    // theoretically also inline if there is no other caller in the function
    // that contains the callee that has a call path to the callable, but that
    // is more expensive to check.
    auto func_op = mlir::dyn_cast<mlir::func::FuncOp>(callable);
    if (!func_op) {
      return false;
    }
    auto pure_call_op = mlir::dyn_cast<PureCallOp>(call);
    if (!pure_call_op) {
      return false;
    }
    auto callable_region = func_op.getCallableRegion();
    if (!callable_region) {
      return false;
    }

    llvm::SmallDenseSet<llvm::StringRef> callee_calls;
    for (auto callee_call : callable_region->getOps<PureCallOp>()) {
      callee_calls.insert(callee_call.getCallee());
    }

    // If true, then the callee and the caller call the same third function.
    bool contains_call_to_same_function = false;
    // The number of calls to the callee in the caller.
    int num_calls_in_caller = 0;
    if (!wouldBeCloned) {
      num_calls_in_caller = 1;
    } else {
      for (auto neighbor_call : call->getParentRegion()->getOps<PureCallOp>()) {
        contains_call_to_same_function |=
            callee_calls.contains(neighbor_call.getCallee());
        if (neighbor_call.getCallee() == pure_call_op.getCallee()) {
          ++num_calls_in_caller;
        }
      }
    }
    if (num_calls_in_caller > 1) return false;
    // Don't inline functions, if after inlining the size of the function
    // becomes too big.
    int num_ops = num_calls_in_caller * GetNumOps(callable_region->front()) +
                  GetNumOps(call->getParentRegion()->front());
    if (num_ops > kMaxFuncSize) return false;
    return !wouldBeCloned || contains_call_to_same_function;
  }

  // Returns true if the given operation 'op', that is registered to this
  // dialect, can be inlined into the given region, false otherwise.
  // 'wouldBeCloned' is set to true if the given 'op' is set to be cloned
  // during the inlining process, or false if the operation is set to be moved
  // in-place(i.e. no duplicates would be created). 'valueMapping' contains any
  // remapped values from within the 'src' region. This can be used to examine
  // what values may potentially replace the operands to 'op'.
  bool isLegalToInline(mlir::Operation* op, mlir::Region* dest,
                       bool wouldBeCloned,
                       mlir::IRMapping& valueMapping) const final {
    // We allow any op from the xla dialect to be inlined.
    return true;
  }
  // We don't have any special restrictions on what can be inlined into
  // destination regions (e.g. while/conditional bodies). Always allow it.
  bool isLegalToInline(mlir::Region* dest, mlir::Region* src,
                       bool wouldBeCloned,
                       mlir::IRMapping& valueMapping) const final {
    return true;
  }
};

struct XlaOpAsmDialectInterface : public mlir::OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;
  AliasResult getAlias(mlir::Attribute attr,
                       mlir::raw_ostream& os) const final {
    if (llvm::isa<IndexingMapAttr>(attr)) {
      os << "indexing_map";
      return AliasResult::FinalAlias;
    }
    return AliasResult::NoAlias;
  }
};

}  // namespace

void XlaDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "xla/codegen/emitters/ir/xla_ops.cc.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "xla/codegen/emitters/ir/xla_attrs.cc.inc"
      >();
  addInterfaces<XlaInlinerInterface, XlaOpAsmDialectInterface>();
}

}  // namespace xla
