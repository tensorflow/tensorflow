// RUN: mlir-opt %s -vector-lower-to-llvm-dialect | FileCheck %s

func @vec_1d(%arg0: vector<4xf32>, %arg1: vector<8xf32>) -> vector<8xf32> {
  %2 = vector.outerproduct %arg0, %arg1 : vector<4xf32>, vector<8xf32>
  %3 = vector.extractelement %2[0 : i32]: vector<4x8xf32>
  return %3 : vector<8xf32>
}
// CHECK-LABEL: vec_1d
//       CHECK:   llvm.undef : !llvm<"[4 x <8 x float>]">
//     CHECK-5:   llvm.shufflevector {{.*}}, {{.*}} [0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32] : !llvm<"<4 x float>">, !llvm<"<4 x float>">
//       CHECK:   llvm.fmul {{.*}}, {{.*}} : !llvm<"<8 x float>">
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm<"[4 x <8 x float>]">
//       CHECK:   llvm.extractvalue {{.*}}[0 : i32] : !llvm<"[4 x <8 x float>]">
//       CHECK:   llvm.return {{.*}} : !llvm<"<8 x float>">

func @vec_2d(%arg0: vector<4xf32>, %arg1: vector<8xf32>) -> vector<4x8xf32> {
  %2 = vector.outerproduct %arg0, %arg1 : vector<4xf32>, vector<8xf32>
  return %2 : vector<4x8xf32>
}
// CHECK-LABEL: vec_2d
//       CHECK:   llvm.undef : !llvm<"[4 x <8 x float>]">
//     CHECK-4:   llvm.shufflevector {{.*}}, {{.*}} [0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32] : !llvm<"<4 x float>">, !llvm<"<4 x float>">
//       CHECK:   llvm.fmul {{.*}}, {{.*}} : !llvm<"<8 x float>">
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm<"[4 x <8 x float>]">
//       CHECK:   llvm.return {{.*}} : !llvm<"[4 x <8 x float>]">

func @vec_3d(%arg0: vector<4x8x16xf32>) -> vector<8x16xf32> {
  %0 = vector.extractelement %arg0[0 : i32]: vector<4x8x16xf32>
  return %0 : vector<8x16xf32>
}
// CHECK-LABEL: vec_3d
//       CHECK:   llvm.extractvalue %{{.*}}[0 : i32] : !llvm<"[4 x [8 x <16 x float>]]">
//       CHECK:   llvm.return %{{.*}} : !llvm<"[8 x <16 x float>]">