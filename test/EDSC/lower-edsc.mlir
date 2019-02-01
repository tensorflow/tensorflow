// RUN: mlir-opt -lower-edsc-test %s | FileCheck %s

func @t1(%lhs: memref<3x4x5x6xvector<4xi8>>, %rhs: memref<3x4x5x6xvector<4xi8>>, %result: memref<3x4x5x6xvector<4xi8>>) -> () { return }
func @t2(%lhs: memref<3x4xf32>, %rhs: memref<3x4xf32>, %result: memref<3x4xf32>) -> () { return }
func @t3(%lhs: memref<f32>, %rhs: memref<f32>, %result: memref<f32>) -> () { return }

func @fn() {
  "print"() {op: "x.add", fn: @t1: (memref<3x4x5x6xvector<4xi8>>, memref<3x4x5x6xvector<4xi8>>, memref<3x4x5x6xvector<4xi8>>) -> ()} : () -> ()
  "print"() {op: "x.add", fn: @t2: (memref<3x4xf32>, memref<3x4xf32>, memref<3x4xf32>) -> ()} : () -> ()
  "print"() {op: "x.add", fn: @t3: (memref<f32>, memref<f32>, memref<f32>) -> ()} : () -> ()
  return
}

// CHECK:   for({{.*}}=[[zero1:.*]] to {{.*}} step [[step1:.*]]) {
// CHECK-NEXT:     for({{.*}}=[[zero1]] to {{.*}} step [[step1]]) {
// CHECK-NEXT:       for({{.*}}=[[zero1]] to {{.*}} step [[step1]]) {
// CHECK-NEXT:         for({{.*}}=[[zero1]] to {{.*}} step [[step1]]) {
// CHECK-NEXT:           {{.*}} = store((load($3[{{.*}}, {{.*}}, {{.*}}, {{.*}}]) + load($4[{{.*}}, {{.*}}, {{.*}}, {{.*}}])), $5[{{.*}}, {{.*}}, {{.*}}, {{.*}}])
// CHECK-NEXT:         };
// CHECK-NEXT:       };
// CHECK-NEXT:     };
// CHECK-NEXT:   }
// CHECK:   for({{.*}}=[[zero2:.*]] to {{.*}} step [[step2:.*]]) {
// CHECK-NEXT:     for({{.*}}=[[zero2]] to {{.*}} step [[step2]]) {
// CHECK-NEXT:           {{.*}} = store((load($3[{{.*}}, {{.*}}]) + load($4[{{.*}}, {{.*}}])), $5[{{.*}}, {{.*}}])
// CHECK-NEXT:     };
// CHECK-NEXT:   }
// CHECK: {{.*}} = store((load($3[]) + load($4[])), $5[])
