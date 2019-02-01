// RUN: mlir-tblgen -gen-reference-implementations -I %S/../../include %s | FileCheck %s

#ifdef OP_BASE
#else
include "mlir/IR/op_base.td"
#endif // OP_BASE

def X_AddOp : Op<"x.add">,
    Arguments<(ins Tensor:$A, Tensor:$B)>,
    Results<(outs Tensor: $C)> {
    // TODO: extract referenceImplementation to Op.
    // TODO: shrink the reference implementation
  code referenceImplementation = [{
    auto ivs = makeNewExprs(view_A.rank());
    // TODO(jpienaar@): automate the positional/named extraction. Need to be a
    // bit careful about things memref (from which a "view" can be extracted)
    // and the rest (see ReferenceImplGen.cpp).
    Indexed A = args[0], B = args[1], C = args[2];
    if (ivs.size() > 0) {
      block.set(
        For(ivs, view_A.lbs, view_A.ubs, view_A.steps, {
          C(ivs) = A(ivs) + B(ivs)
        }));
    } else {
      // 0-D case is always important to get right for composability.
      block.set(C() = A() + B());
    }
  }];
}

// CHECK: printRefImplementation
