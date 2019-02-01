// RUN: mlir-tblgen -gen-rewriters -I %S/../../include %s | FileCheck %s

include "mlir/IR/op_base.td"

// Create a Type and Attribute.
def T : BuildableType<"buildT">;
def T_Attr : TypeBasedAttr<T, "Attribute", "attribute of T type">;
def T_Const_Attr : ConstantAttr<T_Attr, "attrValue">;

// Define ops to rewrite.
def U: Type<CPred<"true">, "U">;
def X_AddOp : Op<"x.add"> {
  let arguments = (ins U, U);
}
def Y_AddOp : Op<"y.add"> {
  let arguments = (ins U, U, T_Attr:$attrName);
}

// Define rewrite pattern.
def : Pat<(X_AddOp $lhs, $rhs), (Y_AddOp $lhs, U:$rhs, T_Const_Attr:$x)>;

// CHECK: struct GeneratedConvert0 : public RewritePattern
// CHECK: RewritePattern("x.add", 1, context)
// CHECK: PatternMatchResult match(OperationInst *
// CHECK: void rewrite(OperationInst *op, std::unique_ptr<PatternState>
// CHECK:              PatternRewriter &rewriter)
// CHECK: rewriter.replaceOpWithNewOp<AddOp>(op, op->getResult(0)->getType()
// CHECK: void populateWithGenerated
// CHECK: patterns->push_back(std::make_unique<GeneratedConvert0>(context))
