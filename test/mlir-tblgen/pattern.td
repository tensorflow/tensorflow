// RUN: mlir-tblgen -gen-rewriters -I %S/../../include %s | FileCheck %s

include "mlir/IR/OpBase.td"

def OpA : Op<"op_a", []> {
  let arguments = (ins I32:$operand, I32Attr:$attr);
  let results = (outs I32:$result);
}

def OpB : Op<"op_b", []> {
  let arguments = (ins I32:$operand, I32Attr:$attr);
  let results = (outs I32:$result);
}

def MyRule : Pat<(OpA $input, $attr), (OpB $input, $attr)>;

// Test rewrite rule naming
// ---

// CHECK: struct MyRule : public RewritePattern

def : Pat<(OpA $input, $attr), (OpB $input, $attr)>;

// Test basic structure generated from Pattern
// ---

// CHECK: struct GeneratedConvert0 : public RewritePattern

// CHECK: GeneratedConvert0(MLIRContext *context) : RewritePattern("op_a", 1, context) {}

// CHECK: struct MatchedState : public PatternState {
// CHECK:   Value *input;
// CHECK:   IntegerAttr attr;
// CHECK: };

// CHECK: PatternMatchResult match(Operation *op0) const override

// CHECK: void rewrite(Operation *op, std::unique_ptr<PatternState> state,
// CHECK:              PatternRewriter &rewriter) const override


// CHECK: void populateWithGenerated(MLIRContext *context, OwningRewritePatternList *patterns)
// CHECK:   patterns->push_back(llvm::make_unique<MyRule>(context));
// CHECK:   patterns->push_back(llvm::make_unique<GeneratedConvert0>(context));
