// RUN: mlir-tblgen -gen-rewriters -I %S/../../include %s | FileCheck %s

include "mlir/IR/OpBase.td"

def Test_Dialect : Dialect {
  let name = "test";
  let cppNamespace = "";
}
class NS_Op<string mnemonic, list<OpTrait> traits> :
    Op<Test_Dialect, mnemonic, traits>;

def OpA : NS_Op<"op_a", []> {
  let arguments = (ins I32:$operand, I32Attr:$attr);
  let results = (outs I32:$result);
}

def OpB : NS_Op<"op_b", []> {
  let arguments = (ins I32:$operand);
  let results = (outs I32:$result);
}

def OpC : NS_Op<"op_c", []> {
  let arguments = (ins I32:$operand);
  let results = (outs I32:$result);
}

def OpD : NS_Op<"op_d", []> {
  let arguments = (ins I32:$input1, I32:$input2, I32:$input3, I32Attr:$attr);
  let results = (outs I32:$result);
}

def hasOneUse: Constraint<CPred<"$0->hasOneUse()">, "has one use">;

def : Pattern<(OpA:$res_a $operand, $attr),
              [(OpC:$res_c (OpB:$res_b $operand)),
               (OpD $res_b, $res_c, $res_a, $attr)],
              [(hasOneUse $res_a)]>;

// CHECK-LABEL: GeneratedConvert0

// Test struct for bound arguments
// ---
// CHECK: struct MatchedState : public PatternState
// CHECK:   Value *operand;
// CHECK:   IntegerAttr attr;
// CHECK:   Operation *res_a;

// Test bound arguments/results in source pattern
// ---
// CHECK: PatternMatchResult match
// CHECK:   auto state = llvm::make_unique<MatchedState>();
// CHECK:   auto &s = *state;
// CHECK:   s.res_a = op0;
// CHECK:   s.operand = op0->getOperand(0);
// CHECK:   attr = op0->getAttrOfType<IntegerAttr>("attr");
// CHECK:   s.attr = attr;
// CHECK:   if (!((s.res_a->getResult(0)->hasOneUse()))) return matchFailure();

// Test bound results in result pattern
// ---
// CHECK: void rewrite
// CHECK:   auto& s = *static_cast<MatchedState *>(state.get());
// CHECK:   auto res_b = rewriter.create<OpB>
// CHECK:   auto res_c = rewriter.create<OpC>(
// CHECK:     /*operand=*/res_b
// CHECK:   auto vOpD0 = rewriter.create<OpD>(
// CHECK:     /*input1=*/res_b.getOperation()->getResult(0),
// CHECK:     /*input2=*/res_c.getOperation()->getResult(0),
// CHECK:     /*input3=*/s.res_a->getResult(0),
// CHECK:     /*attr=*/s.attr
