// RUN: mlir-opt %s -split-input-file | FileCheck %s

// -----
// CHECK-LABEL: parseFullySpecified
// CHECK: !quant.any<i8<-8:7>:f32>
!qalias = type !quant.any<i8<-8:7>:f32>
func @parseFullySpecified() -> !qalias {
  %0 = "foo"() : () -> !qalias
  return %0 : !qalias
}

// -----
// CHECK-LABEL: parseNoExpressedType
// CHECK: !quant.any<i8<-8:7>>
!qalias = type !quant.any<i8<-8:7>>
func @parseNoExpressedType() -> !qalias {
  %0 = "foo"() : () -> !qalias
  return %0 : !qalias
}

// -----
// CHECK-LABEL: parseOnlyStorageType
// CHECK: !quant.any<i8>
!qalias = type !quant.any<i8>
func @parseOnlyStorageType() -> !qalias {
  %0 = "foo"() : () -> !qalias
  return %0 : !qalias
}
