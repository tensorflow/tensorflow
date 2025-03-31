#ifndef TRITON_ANALYSIS_AXISINFO_H
#define TRITON_ANALYSIS_AXISINFO_H

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Support/LLVM.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include <optional>
#include <type_traits>

namespace mlir::triton {

//===----------------------------------------------------------------------===//
// AxisInfo
//===----------------------------------------------------------------------===//

/// This lattice value represents known information on the axes of a lattice.
class AxisInfo {
public:
  typedef SmallVector<int64_t> DimVectorT;

public:
  AxisInfo() : AxisInfo({}, {}, {}) {}

  AxisInfo(ArrayRef<int64_t> contiguity, ArrayRef<int64_t> divisibility,
           ArrayRef<int64_t> constancy)
      : AxisInfo(contiguity, divisibility, constancy, std::nullopt) {}

  AxisInfo(ArrayRef<int64_t> contiguity, ArrayRef<int64_t> divisibility,
           ArrayRef<int64_t> constancy, std::optional<int64_t> constantValue)
      : contiguity(contiguity), divisibility(divisibility),
        constancy(constancy), constantValue(constantValue) {
    assert(divisibility.size() == contiguity.size());
    assert(constancy.size() == contiguity.size());
  }

  // contiguity[d] is the length of the shortest sequence of contiguous integers
  // along dimension d.
  //
  // If we have an array of N elements with a contiguity value C, then the array
  // can be divided into a list of N/C sequences of C contiguous elements.
  // Since we have N = 2^k, C must be a power of two.
  //
  // For example, the 2D array
  //
  //   [[10, 11, 12, 13, 18, 19, 20, 21],
  //    [20, 21, 22, 23, 28, 29, 30, 31]]
  //
  // has contiguity [1, 4], and
  //
  //   [[12, 16, 20, 24],
  //    [13, 17, 21, 25],
  //    [14, 18, 22, 26],
  //    [15, 19, 23, 27],
  //    [18, 22, 26, 30],
  //    [19, 23, 27, 31]]
  //
  // has contiguity [2, 1].
  int64_t getContiguity(size_t dim) const { return contiguity[dim]; }
  const DimVectorT &getContiguity() const { return contiguity; }

  // divisibility[d] is the largest power of two that divides the first element
  // of all groups of length contiguity[d] along dimension d.
  //
  // For example,
  //
  //   [[10, 11, 12, 13, 18, 19, 20, 21],
  //    [20, 21, 22, 23, 28, 29, 30, 31]]
  //
  //  has divisibility [1, 2], and
  //
  //    [[12, 16, 20, 24],
  //     [13, 17, 21, 25],
  //     [14, 18, 22, 26],
  //     [15, 19, 23, 27]]
  //
  // has divisibility [4, 1].
  //
  // On the other hand,
  //
  //   [0, 1, 2, 0, 4, 5, 6, 7]
  //
  // has divisibility 1 because its contiguity is 1.
  int64_t getDivisibility(size_t dim) const { return divisibility[dim]; }
  const DimVectorT &getDivisibility() const { return divisibility; }

  // constancy[d] is the length of the shortest sequence of repeating integers
  // along dimension d.
  //
  // This is particularly useful to infer the contiguity of operations (e.g.
  // add) involving a constant.
  //
  // If we have an array of N elements, with a constancy value C, then the array
  // can be divided into a list of N/C sequences of C elements with the same
  // value.  Since we have N = 2^k, C must be a power of two.
  //
  // For example
  //
  //   [[8, 8, 8, 8, 12, 12, 12, 12],
  //    [16, 16, 16, 16, 20, 20, 20, 20]]
  //
  // has constancy [1, 4].
  int64_t getConstancy(size_t dim) const { return constancy[dim]; }
  const DimVectorT &getConstancy() const { return constancy; }

  int getRank() const { return contiguity.size(); }

  std::optional<int64_t> getConstantValue() const { return constantValue; }

  template <class T>
  static void
  initPessimisticStateFromFunc(int argNumber, T funcOp, DimVectorT *contiguity,
                               DimVectorT *divisibility, DimVectorT *constancy);

  bool operator==(const AxisInfo &other) const {
    return contiguity == other.contiguity &&
           divisibility == other.divisibility && constancy == other.constancy &&
           constantValue == other.constantValue;
  }

  static AxisInfo getPessimisticValueState(Value value);

  // The gcd of both arguments for each dimension
  static AxisInfo join(const AxisInfo &lhs, const AxisInfo &rhs);

  void print(raw_ostream &os) const {
    auto print = [&](StringRef name, DimVectorT vec) {
      os << name << " = [";
      llvm::interleaveComma(vec, os);
      os << "]";
    };
    print("contiguity", contiguity);
    print(", divisibility", divisibility);
    print(", constancy", constancy);
    os << ", constant_value = ";
    if (constantValue)
      os << *constantValue;
    else
      os << "<none>";
  }

private:
  DimVectorT contiguity;
  DimVectorT divisibility;
  DimVectorT constancy;

  // The constant value of the lattice if we can infer it.
  std::optional<int64_t> constantValue;
};

// Module level axis info analysis based on the call graph, assuming that we do
// not have recursive functions.
//
// Since each function will be called multiple times, we need to calculate the
// axis info based on the axis info of all the callers.  In the future, we can
// perform optimization using function cloning so that each call site will have
// unique axis info.
using AxisInfoMapT = DenseMap<Value, AxisInfo>;
class ModuleAxisInfoAnalysis : public CallGraph<AxisInfoMapT> {
public:
  explicit ModuleAxisInfoAnalysis(ModuleOp moduleOp)
      : CallGraph<AxisInfoMapT>(moduleOp) {
    SmallVector<FunctionOpInterface> funcs;
    for (auto root : getRoots()) {
      walk<WalkOrder::PreOrder, WalkOrder::PostOrder>(
          // Pre-order edge walk callback
          [](CallOpInterface callOp, FunctionOpInterface funcOp) {},
          // Post-order node walk callback
          [&](FunctionOpInterface funcOp) {
            funcs.push_back(funcOp);
            funcMap.try_emplace(funcOp, AxisInfoMapT{});
          });
    }
    SetVector<FunctionOpInterface> sortedFuncs(funcs.begin(), funcs.end());
    SymbolTableCollection symbolTable;
    for (auto funcOp : llvm::reverse(sortedFuncs)) {
      initialize(funcOp);
      funcOp.walk([&](CallOpInterface callOp) {
        auto callee = dyn_cast<FunctionOpInterface>(
            callOp.resolveCallableInTable(&symbolTable));
        update(callOp, callee);
      });
    }
  }

  AxisInfo *getAxisInfo(Value value) {
    auto funcOp =
        value.getParentRegion()->getParentOfType<FunctionOpInterface>();
    auto *axisInfoMap = getFuncData(funcOp);
    if (!axisInfoMap) {
      return nullptr;
    }
    auto it = axisInfoMap->find(value);
    if (it == axisInfoMap->end()) {
      return nullptr;
    }
    return &(it->second);
  }

  unsigned getContiguity(Value value);
  unsigned getAlignment(Value value);

  // Overloads of the above methods but have separated elementBitWidth to
  // calculate the contiguity. These are useful for computing axis info when
  // lowering to hardware intrinsics that require a scalar/warp-uniform base ptr
  // with separate per lane offsets like AMD buffer operations.
  //
  // As a concrete example, instead of a single tensor<128x64x!tt.ptr<f16>>
  // value, now we have two separate values: !tt.ptr<f16> for the base pointer
  // and tensor<128x64xi32> for the offset. For such cases, we want to compute
  // the contiguity on the offsets but use the pointee element type bit width
  // instead of the offset element type bit width for alignment
  unsigned getContiguity(Value offsetsValue, unsigned elementBitWidth);
  unsigned getAlignment(Value offsetsValue, unsigned elementBitWidth);

  unsigned getMaskAlignment(Value mask);

private:
  void initialize(FunctionOpInterface funcOp);
  void update(CallOpInterface callOp, FunctionOpInterface funcOp);
};

} // namespace mlir::triton

#endif
