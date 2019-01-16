// RUN: mlir-tblgen -gen-reference-implementations -I %S/../../include %s | FileCheck %s

#ifdef OP_BASE
#else
include "mlir/IR/op_base.td"
#endif // OP_BASE

def X_AddOp : Op<"x.add">,
    Arguments<(ins Tensor:$lhs, Tensor:$rhs)>,
    Results<[Tensor]> {
    // TODO: extract referenceImplementation to Op.
    // TODO: shrink the reference implementation
  code referenceImplementation = [{
    auto *lhsMemRef = *(f->getArguments().begin());
    auto *rhsMemRef = *(f->getArguments().begin() + 1);
    auto *resultMemRef = *(f->getArguments().begin() + 2);

    Bindable lhs, rhs, result;
    auto lhsShape = emitter.makeBoundSizes(lhsMemRef);

    auto ivs = makeBindables(lhsShape.size());
    Bindable zero, one;
    // Same bindable, all equal to `zero`.
    SmallVector<Bindable, 8> zeros(ivs.size(), zero);
    // Same bindable, all equal to `one`.
    SmallVector<Bindable, 8> ones(ivs.size(), one);
    Indexed IA(lhs), IB(rhs), IC(result);
    block = edsc::Block({
      ForNest(ivs, zeros, lhsShape, ones, {
        IC[ivs] = IA[ivs] + IB[ivs]
      })
    });
  }];
}

// CHECK: printRefImplementation