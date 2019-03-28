// RUN: mlir-tblgen -gen-rewriters -I %S/../../include %s | FileCheck %s

include "mlir/IR/OpBase.td"

// Create a Type and Attribute.
def T : BuildableType<"buildT">;
def T_Attr : TypeBasedAttr<T, "Attribute", "attribute of T type">;
def T_Const_Attr : ConstantAttr<T_Attr, "attrValue">;
def T_Compose_Attr : tAttr<"{0}.getArrayAttr({{{1}, {2}})">;

// Define ops to rewrite.
def U: Type<CPred<"true">, "U">;
def X_AddOp : Op<"x.add"> {
  let arguments = (ins U, U);
}
def Y_AddOp : Op<"y.add"> {
  let arguments = (ins U, U, T_Attr:$attrName);
}
def Z_AddOp : Op<"z.add"> {
  let arguments = (ins U, U, T_Attr:$attrName1, T_Attr:$attrName2);
}

// Define rewrite pattern.
def : Pat<(Y_AddOp $lhs, $rhs, $attr1), (Y_AddOp $lhs, $rhs, (T_Compose_Attr $attr1, T_Const_Attr:$attr2))>;
// CHECK: struct GeneratedConvert0 : public RewritePattern
// CHECK: RewritePattern("y.add", 1, context)
// CHECK: PatternMatchResult match(Operation *
// CHECK: void rewrite(Operation *op, std::unique_ptr<PatternState>
// CHECK-NEXT:              PatternRewriter &rewriter)
// CHECK:      auto vAddOp0 = rewriter.create<Y::AddOp>(loc, op->getResult(0)->getType(),
// CHECK-NEXT:     s.lhs,
// CHECK-NEXT:     s.rhs,
// CHECK-NEXT:     /*attrName=*/rewriter.getArrayAttr({s.attr1, rewriter.getAttribute(rewriter.buildT, attrValue)})
// CHECK-NEXT: );
// CHECK-NEXT: rewriter.replaceOp(op, {vAddOp0});

def : Pat<(Z_AddOp $lhs, $rhs, $attr1, $attr2), (Y_AddOp $lhs, $rhs, (T_Compose_Attr $attr1, $attr2))>;
// CHECK: struct GeneratedConvert1 : public RewritePattern
// CHECK: RewritePattern("z.add", 1, context)
// CHECK: PatternMatchResult match(Operation *
// CHECK: void rewrite(Operation *op, std::unique_ptr<PatternState>
// CHECK-NEXT:              PatternRewriter &rewriter)
// CHECK:      auto vAddOp0 = rewriter.create<Y::AddOp>(loc, op->getResult(0)->getType(),
// CHECK-NEXT:     s.lhs,
// CHECK-NEXT:     s.rhs,
// CHECK-NEXT:     /*attrName=*/rewriter.getArrayAttr({s.attr1, s.attr2})
// CHECK-NEXT: );
// CHECK-NEXT: rewriter.replaceOp(op, {vAddOp0});

// CHECK: void populateWithGenerated
// CHECK: patterns->push_back(llvm::make_unique<GeneratedConvert0>(context))
// CHECK: patterns->push_back(llvm::make_unique<GeneratedConvert1>(context))
