// RUN: mlir-tblgen -gen-rewriters %s | FileCheck %s

// Extracted & simplified from op_base.td to do more directed testing.
class Type {
  code builderCall = ?;
}
class Pattern<dag patternToMatch, list<dag> resultOps> {
  dag PatternToMatch = patternToMatch;
  list<dag> ResultOps = resultOps;
}
class Pat<dag pattern, dag result> : Pattern<pattern, [result]>;
def ins;
class Op<string mnemonic> {
  string opName = mnemonic;
  dag arguments = (ins);
}
class Attr<Type t> {
  Type type = t;
  code storageType = ?;
  code returnType = ?;
}

// Create a Type and Attribute.
def YT : Type {
  let builderCall = "buildYT()";
}
def Y_Attr : Attr<YT>;
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
