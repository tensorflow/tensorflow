// RUN: mlir-tblgen -gen-rewriters -I %S/../../include %s | FileCheck %s

include "mlir/IR/OpBase.td"

def Test_Dialect : Dialect {
  let name = "test";
}
class NS_Op<string mnemonic, list<OpTrait> traits> :
    Op<Test_Dialect, mnemonic, traits>;

def ThreeResultOp : NS_Op<"three_result_op", []> {
  let arguments = (ins I32:$input);
  let results = (outs I32:$r1, I32:$r2, I32:$r3);
}

def OneResultOp : NS_Op<"one_result_op", []> {
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
