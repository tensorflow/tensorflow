// RUN: mlir-tblgen -gen-rewriters -I %S/../../include %s | FileCheck %s

include "mlir/IR/OpBase.td"

def ThreeResultOp : Op<"three_result_op", []> {
  let arguments = (ins I32:$input);
  let results = (outs I32:$r1, I32:$r2, I32:$r3);
}

def OneResultOp : Op<"one_result_op", []> {
  let arguments = (ins I32:$input);
  let results = (outs I32:$r1);
}

def : Pattern<(ThreeResultOp $input), [
        (OneResultOp $input),
        (OneResultOp $input),
        (OneResultOp $input)
      ]>;

// CHECK-LABEL: struct GeneratedConvert0

// CHECK: void rewrite(
// CHECK: 	auto vOneResultOp0 = rewriter.create<OneResultOp>(
// CHECK: 	auto vOneResultOp1 = rewriter.create<OneResultOp>(
// CHECK: 	auto vOneResultOp2 = rewriter.create<OneResultOp>(
// CHECK: 	rewriter.replaceOp(op, {vOneResultOp0, vOneResultOp1, vOneResultOp2});
