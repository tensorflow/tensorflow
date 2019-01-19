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
    auto ivs = makeBindables(lhsShape.size());
    Bindable zero, one;
    // Same bindable, all equal to `zero`.
    SmallVector<Bindable, 8> zeros(ivs.size(), zero);
    // Same bindable, all equal to `one`.
    SmallVector<Bindable, 8> ones(ivs.size(), one);
    Indexed IA(lhs), IB(rhs), IC(result);
    block = edsc::Block({
      For(ivs, zeros, lhsShape, ones, {
        IC[ivs] = IA[ivs] + IB[ivs]
      })
    });
  }];
}

// CHECK: printRefImplementation
