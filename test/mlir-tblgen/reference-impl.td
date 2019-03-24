// RUN: mlir-tblgen -gen-reference-implementations -I %S/../../include %s | FileCheck %s

#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

def X_AddOp : Op<"x.add">,
    Arguments<(ins Tensor:$A, Tensor:$B)>,
    Results<(outs Tensor: $C)> {
    // TODO: extract referenceImplementation to Op.
  code referenceImplementation = [{
    auto ivs = IndexHandle::makeIndexHandles(view_A.rank());
    auto pivs = IndexHandle::makeIndexHandlePointers(ivs);
    IndexedValue A(arg_A), B(arg_B), C(arg_C);
    LoopNestBuilder(pivs, view_A.getLbs(), view_A.getUbs(), view_A.getSteps())({
      C(ivs) = A(ivs) + B(ivs)
    });
  }];
}

// CHECK: printRefImplementation
