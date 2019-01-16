// RUN: mlir-tblgen -gen-rewriters -I %S/../../include %s | FileCheck %s

include "mlir/IR/op_base.td"

// Create a Type and Attribute.
def YT : BuildableType<"buildYT">;
def Y_Attr : TypeBasedAttr<YT, "Attribute">;
def Y_Const_Attr {
  Attr attr = Y_Attr;
  string value = "attrValue";
}

// Define ops to rewrite.
def T1: Type;
def X_AddOp : Op<"x.add"> {
  let arguments = (ins T1, T1);
}
def Y_AddOp : Op<"y.add"> {
  let arguments = (ins T1, T1, Y_Attr:$attrName);
}

// Define rewrite pattern.
def : Pat<(X_AddOp $lhs, $rhs), (Y_AddOp $lhs, T1:$rhs, Y_Const_Attr:$x)>;

// CHECK: struct GeneratedConvert0 : public RewritePattern
// CHECK: RewritePattern("x.add", 1, context)
// CHECK: PatternMatchResult match(OperationInst *
// CHECK: void rewrite(OperationInst *op, std::unique_ptr<PatternState>
// CHECK:              PatternRewriter &rewriter)
// CHECK: rewriter.replaceOpWithNewOp<AddOp>(op, op->getResult(0)->getType()
// CHECK: void populateWithGenerated
// CHECK: patterns->push_back(std::make_unique<GeneratedConvert0>(context))
