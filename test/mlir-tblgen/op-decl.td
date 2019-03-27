// RUN: mlir-tblgen -gen-op-decls -I %S/../../include %s | FileCheck %s

include "mlir/IR/OpBase.td"

def NS_AOp : Op<"a_op", [NoSideEffect]> {
  let arguments = (ins
    I32:$a,
    Variadic<F32>:$b,

    I32Attr:$attr1,
    OptionalAttr<F32Attr>:$attr2
  );

  let results = (outs
    I32:$r,
    Variadic<F32>:$s
  );

  let builders = [OpBuilder<"Value *val">];
  let parser = [{ foo }];
  let printer = [{ bar }];
  let verifier = [{ baz }];

  let hasCanonicalizer = 1;
  let hasConstantFolder = 1;
  let hasFolder = 1;
}

// CHECK-LABEL: NS::AOp declarations

// CHECK: class AOp : public Op<AOp, OpTrait::AtLeastNResults<1>::Impl, OpTrait::HasNoSideEffect, OpTrait::AtLeastNOperands<1>::Impl> {
// CHECK: public:
// CHECK:   using Op::Op;
// CHECK:   static StringRef getOperationName();
// CHECK:   Value *a();
// CHECK:   Instruction::operand_range b();
// CHECK:   Value *r();
// CHECK:   APInt attr1();
// CHECK:   Optional< APFloat > attr2();
// CHECK:   static void build(Value *val);
// CHECK:   static void build(Builder *, OperationState *tblgen_state, Type r, ArrayRef<Type> s, Value *a, ArrayRef<Value *> b, IntegerAttr attr1, /*optional*/FloatAttr attr2);
// CHECK:   static void build(Builder *, OperationState *tblgen_state, ArrayRef<Type> resultTypes, ArrayRef<Value *> operands, ArrayRef<NamedAttribute> attributes);
// CHECK:   static bool parse(OpAsmParser *parser, OperationState *result);
// CHECK:   void print(OpAsmPrinter *p);
// CHECK:   bool verify();
// CHECK:   static void getCanonicalizationPatterns(OwningRewritePatternList &results, MLIRContext *context);
// CHECK:   LogicalResult constantFold(ArrayRef<Attribute> operands, SmallVectorImpl<Attribute> &results, MLIRContext *context);
// CHECK:   bool fold(SmallVectorImpl<Value *> &results);
// CHECK: };
