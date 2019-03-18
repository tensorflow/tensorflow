// RUN: mlir-opt -lower-edsc-test %s | FileCheck %s

// CHECK-LABEL: func @t1(%arg0: memref<3x4x5x6xvector<4xi8>>, %arg1: memref<3x4x5x6xvector<4xi8>>, %arg2: memref<3x4x5x6xvector<4xi8>>) {
func @t1(%lhs: memref<3x4x5x6xvector<4xi8>>, %rhs: memref<3x4x5x6xvector<4xi8>>, %result: memref<3x4x5x6xvector<4xi8>>) -> () {
//       CHECK: for {{.*}} = 0 to 3 {
//       CHECK:   for {{.*}} = 0 to 4 {
//       CHECK:     for {{.*}} = 0 to 5 {
//       CHECK:       for {{.*}}= 0 to 6 {
//       CHECK:         {{.*}} = load %arg1[{{.*}}] : memref<3x4x5x6xvector<4xi8>>
//       CHECK:         {{.*}} = load %arg0[{{.*}}] : memref<3x4x5x6xvector<4xi8>>
//       CHECK:         {{.*}} = addi {{.*}} : vector<4xi8>
//       CHECK:         store {{.*}}, %arg2[{{.*}}] : memref<3x4x5x6xvector<4xi8>>
  return
}

// CHECK-LABEL: func @t2(%arg0: memref<3x4xf32>, %arg1: memref<3x4xf32>, %arg2: memref<3x4xf32>) {
func @t2(%lhs: memref<3x4xf32>, %rhs: memref<3x4xf32>, %result: memref<3x4xf32>) -> () {
//       CHECK: for {{.*}} = 0 to 3 {
//       CHECK:   for {{.*}} = 0 to 4 {
//       CHECK:     {{.*}} = load %arg1[{{.*}}, {{.*}}] : memref<3x4xf32>
//       CHECK:     {{.*}} = load %arg0[{{.*}}, {{.*}}] : memref<3x4xf32>
//       CHECK:     {{.*}} = addf {{.*}}, {{.*}} : f32
//       CHECK:     store {{.*}}, %arg2[{{.*}}, {{.*}}] : memref<3x4xf32>
  return
}


// CHECK-LABEL: func @t3(%arg0: memref<f32>, %arg1: memref<f32>, %arg2: memref<f32>) {
func @t3(%lhs: memref<f32>, %rhs: memref<f32>, %result: memref<f32>) -> () {
//       CHECK: {{.*}} = load %arg1[] : memref<f32>
//       CHECK: {{.*}} = load %arg0[] : memref<f32>
//       CHECK: {{.*}} = addf {{.*}}, {{.*}} : f32
//       CHECK: store {{.*}}, %arg2[] : memref<f32>
  return
}

func @fn() {
  "print"() {op: "x.add", fn: @t1: (memref<3x4x5x6xvector<4xi8>>, memref<3x4x5x6xvector<4xi8>>, memref<3x4x5x6xvector<4xi8>>) -> ()} : () -> ()
  "print"() {op: "x.add", fn: @t2: (memref<3x4xf32>, memref<3x4xf32>, memref<3x4xf32>) -> ()} : () -> ()
  "print"() {op: "x.add", fn: @t3: (memref<f32>, memref<f32>, memref<f32>) -> ()} : () -> ()
  return
}


