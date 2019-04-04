// RUN: mlir-tblgen -gen-rewriters -I %S/../../include %s | FileCheck %s

include "mlir/IR/OpBase.td"

// Create a Type and Attribute.
def T : BuildableType<"buildT()">;
def T_Attr : TypedAttrBase<T, "Attribute",CPred<"true">, "attribute of T type">;
def T_Const_Attr : ConstantAttr<T_Attr, "attrValue">;
def T_Compose_Attr : tAttr<"{0}.getArrayAttr({{{1}, {2}})">;

// Define ops to rewrite.
def Y_Op : Op<"y.op"> {
  let arguments = (ins T_Attr:$attrName);
  let results = (outs I32:$result);
}
def Z_Op : Op<"z.op"> {
  let arguments = (ins T_Attr:$attrName1, T_Attr:$attrName2);
  let results = (outs I32:$result);
}

// Define rewrite pattern.
def : Pat<(Y_Op $attr1), (Y_Op (T_Compose_Attr $attr1, T_Const_Attr))>;
// CHECK-LABEL: struct GeneratedConvert0
// CHECK: void rewrite(
// CHECK:   /*attrName=*/rewriter.getArrayAttr({s.attr1, rewriter.getAttribute(rewriter.buildT(), attrValue)})

def : Pat<(Z_Op $attr1, $attr2), (Y_Op (T_Compose_Attr $attr1, $attr2))>;
// CHECK-LABEL: struct GeneratedConvert1
// CHECK: void rewrite(
// CHECK:   /*attrName=*/rewriter.getArrayAttr({s.attr1, s.attr2})
