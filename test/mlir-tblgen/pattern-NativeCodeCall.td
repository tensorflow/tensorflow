// RUN: mlir-tblgen -gen-rewriters -I %S/../../include %s | FileCheck %s

include "mlir/IR/OpBase.td"

def Test_Dialect : Dialect {
  let name = "test";
}
class NS_Op<string mnemonic, list<OpTrait> traits> :
    Op<Test_Dialect, mnemonic, traits>;

def CreateOperand : NativeCodeCall<"buildOperand($0, $1)">;
def CreateArrayAttr : NativeCodeCall<"$_builder.getArrayAttr({$0, $1})">;
def CreateOpResult : NativeCodeCall<"buildOp($0, $1)">;

def NS_AOp : NS_Op<"a_op", []> {
  let arguments = (ins I32:$input1, I32:$input2, I32Attr:$attr);
  let results = (outs I32:$output);
}

def NS_BOp : NS_Op<"b_op", []> {
  let arguments = (ins I32:$input, I32Attr:$attr);
  let results = (outs I32:$output);
}

def TestCreateOpResult : Pat<
  (NS_BOp $input, $attr),
  (CreateOpResult $input, $attr)>;

// CHECK-LABEL: TestCreateOpResult

// CHECK: rewriter.replaceOp(op, {buildOp(s.input, s.attr)});

def TestCreateOperandAndAttr : Pat<
  (NS_AOp $input1, $input2, $attr),
  (NS_BOp (CreateOperand $input1, $input2), (CreateArrayAttr $attr, $attr))>;

// CHECK-LABEL: TestCreateOperandAndAttr

// CHECK:      rewriter.create<NS::BOp>
// CHECK-NEXT: buildOperand(s.input1, s.input2),
// CHECK-NEXT: rewriter.getArrayAttr({s.attr, s.attr})
