// RUN: mlir-rewriter-gen %s | FileCheck %s

// Extracted & simplified from op_base.td to do more directed testing.
class Type;
class Pattern<dag patternToMatch, list<dag> resultOps> {
  dag PatternToMatch = patternToMatch;
  list<dag> ResultOps = resultOps;
}
class Pat<dag pattern, dag result> : Pattern<pattern, [result]>;
def ins;
class Op<string mnemonic> {
  string name = mnemonic;
  dag operands = (ins);
}
class Attr<Type t> {
  Type type = t;
}

// Create a Type and Attribute.
def YT : Type;
def Y_Attr : Attr<YT>;
def Y_Const_Attr {
  Type type = YT;
  string value = "attrValue";
}

// Define ops to rewrite.
def T1: Type;
def X_AddOp : Op<"x.add">;
def Y_AddOp : Op<"y.add"> {
  let operands = (ins T1, T1, Y_Attr:$attrName);
}

// Define rewrite pattern.
def : Pat<(X_AddOp $lhs, $rhs), (Y_AddOp $lhs, T1:$rhs, Y_Const_Attr:$x)>;

// CHECK: struct GeneratedConvert0 : public RewritePattern
// CHECK: RewritePattern("x.add", 1, context)
// CHECK: PatternMatchResult match(OperationInst *op)
// CHECK: void rewrite(OperationInst *op, PatternRewriter &rewriter)
// CHECK: rewriter.replaceOpWithNewOp<Y::AddOp>(op, op->getResult(0)->getType()

// CHECK: void populateWithGenerated
// CHECK: patterns->push_back(std::make_unique<GeneratedConvert0>(context))