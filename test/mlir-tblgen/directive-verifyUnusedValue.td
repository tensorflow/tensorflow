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
        (verifyUnusedValue),
        (OneResultOp $input),
        (verifyUnusedValue)
      ]>;

// CHECK-LABEL: struct GeneratedConvert0

// CHECK: PatternMatchResult match(
// CHECK:   if (!op0->getResult(0)->use_empty()) return matchFailure();
// CHECK:   if (!op0->getResult(2)->use_empty()) return matchFailure();

// CHECK: void rewrite(
// CHECK:   auto vOneResultOp0 = rewriter.create<OneResultOp>(
// CHECK:   rewriter.replaceOp(op, {nullptr, vOneResultOp0, nullptr});
