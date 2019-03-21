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
// CHECK:   static StringRef getOperationName();
// CHECK:   Value *a();
// CHECK:   const Value *a() const;
// CHECK:   SmallVector<Value *, 4> b();
// CHECK:   const SmallVector<const Value *, 4> b() const;
// CHECK:   Value *r();
// CHECK:   const Value *r() const;
// CHECK:   APInt attr1() const;
// CHECK:   Optional< APFloat > attr2() const;
// CHECK:   static void build(Value *val);
// CHECK:   static void build(Builder *builder, OperationState *result, Type r, ArrayRef<Type> s, Value *a, ArrayRef<Value *> b, IntegerAttr attr1, /*optional*/FloatAttr attr2);
// CHECK:   static void build(Builder *builder, OperationState *result, ArrayRef<Type> resultTypes, ArrayRef<Value *> args, ArrayRef<NamedAttribute> attributes);
// CHECK:   static bool parse(OpAsmParser *parser, OperationState *result);
// CHECK:   void print(OpAsmPrinter *p) const;
// CHECK:   bool verify() const;
// CHECK:   static void getCanonicalizationPatterns(OwningRewritePatternList &results, MLIRContext *context);
// CHECK:   LogicalResult constantFold(ArrayRef<Attribute> operands, SmallVectorImpl<Attribute> &results, MLIRContext *context) const;
// CHECK:   bool fold(SmallVectorImpl<Value *> &results);
// CHECK: private:
// CHECK:   friend class ::mlir::Instruction;
// CHECK:   explicit AOp(const Instruction *state) : Op(state) {}
// CHECK: };
