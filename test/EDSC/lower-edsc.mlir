// RUN: mlir-opt -lower-edsc-test %s | FileCheck %s

func @t1(%lhs: memref<3x4x5x6xf32>, %rhs: memref<3x4x5x6xf32>, %result: memref<3x4x5x6xf32>) -> () { return }
func @t2(%lhs: memref<3x4xf32>, %rhs: memref<3x4xf32>, %result: memref<3x4xf32>) -> () { return }

func @fn() {
  "print"() {op: "x.add", fn: @t1: (memref<3x4x5x6xf32>, memref<3x4x5x6xf32>, memref<3x4x5x6xf32>) -> ()} : () -> ()
  "print"() {op: "x.add", fn: @t2: (memref<3x4xf32>, memref<3x4xf32>, memref<3x4xf32>) -> ()} : () -> ()
  return
}

// CHECK: block {
// CHECK:   for(idx({{.*}})=[[zero1:.*]] to {{.*}} step [[step1:.*]]) {
// CHECK:     for(idx({{.*}})=[[zero1]] to {{.*}} step [[step1]]) {
// CHECK:       for(idx({{.*}})=[[zero1]] to {{.*}} step [[step1]]) {
// CHECK:         for(idx({{.*}})=[[zero1]] to {{.*}} step [[step1]]) {
// CHECK:           lhs({{.*}}) = store( ... );
// CHECK:         };
// CHECK:       };
// CHECK:     };
// CHECK:   }
// CHECK: }
// CHECK: block {
// CHECK:   for(idx({{.*}})=[[zero2:.*]] to {{.*}} step [[step2:.*]]) {
// CHECK:     for(idx({{.*}})=[[zero2]] to {{.*}} step [[step2]]) {
// CHECK:       lhs({{.*}}) = store( ... );
// CHECK:     };
// CHECK:   }
// CHECK: }
