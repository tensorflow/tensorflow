// RUN: triton-opt %s -split-input-file -tritongpu-combine-tensor-select-and-if | FileCheck %s

// CHECK-LABEL: @select_if_combine
tt.func public @select_if_combine(%arg0: tensor<64xf32>, %dst_ptr: tensor<64x!tt.ptr<f32>>, %cnd: i1) {
  // CHECK: %[[CST0:.*]] = arith.constant dense<0.000000e+00>
  %cst = arith.constant dense<0.000000e+00> : tensor<64xf32>
  // CHECK: %[[CST1:.*]] = arith.constant dense<1.000000e+00>
  %cst_1 = arith.constant dense<1.000000e+00> : tensor<64xf32>
  // CHECK-NOT: arith.select
  %sel = arith.select %cnd, %cst, %cst_1 : tensor<64xf32>
  // CHECK: %[[R:.+]] = scf.if %{{.*}}
  // CHECK:   tt.store %{{.*}}, %{{.*}}
  // CHECK:   scf.yield %[[CST0]]
  // CHECK: } else {
  // CHECK:   scf.yield %[[CST1]]
  // CHECK: }
  scf.if %cnd {
    tt.store %dst_ptr, %arg0 : tensor<64x!tt.ptr<f32>>
  }
  // CHECK: tt.store %{{.*}}, %[[R]]
  tt.store %dst_ptr, %sel : tensor<64x!tt.ptr<f32>>
  tt.return
}

// -----
// CHECK-LABEL: @if_multiple_sel
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @if_multiple_sel(%arg0: i1, %arg1: tensor<64xi32, #blocked>, %arg2: tensor<64xi32, #blocked>, %arg3: tensor<64xf32, #blocked>, %arg4: tensor<64xf32, #blocked>) -> (tensor<64xi32, #blocked>, tensor<64xf32, #blocked>, tensor<64xi32, #blocked>){
  // CHECK-NOT: select
  // CHECK: %[[R:.+]]:3 = scf.if %{{.*}} -> (tensor<64xi32, #blocked>, tensor<64xi32, #blocked>, tensor<64xf32, #blocked>) {
  // CHECK:   scf.yield {{.*}} : tensor<64xi32, #blocked>, tensor<64xi32, #blocked>, tensor<64xf32, #blocked>
  // CHECK: } else {
  // CHECK:   scf.yield {{.*}} : tensor<64xi32, #blocked>, tensor<64xi32, #blocked>, tensor<64xf32, #blocked>
  // CHECK: }
  // CHECK: tt.return %[[R]]#1, %[[R]]#2, %[[R]]#0 : tensor<64xi32, #blocked>, tensor<64xf32, #blocked>, tensor<64xi32, #blocked>
    %0 = arith.select %arg0, %arg1, %arg2 : tensor<64xi32, #blocked>
    %1 = arith.select %arg0, %arg3, %arg4 : tensor<64xf32, #blocked>
    %2 = scf.if %arg0 -> (tensor<64xi32, #blocked>) {
      %3 = arith.subi %arg1, %arg2 : tensor<64xi32, #blocked>
      scf.yield %3 : tensor<64xi32, #blocked>
    } else {
      scf.yield %arg1 : tensor<64xi32, #blocked>
    }
    tt.return %0, %1, %2 : tensor<64xi32, #blocked>, tensor<64xf32, #blocked>, tensor<64xi32, #blocked>
  }
}

// -----

tt.func @if_multiple_sel(%arg0: i1, %arg1: tensor<64xi32>, %arg2: tensor<64xi32>, %arg3: tensor<64xi32>, %arg4: tensor<64xi32>) -> (tensor<64xi32>, tensor<64xi32>, tensor<64xi32>){
  // CHECK-NOT: arith.select
  %0 = arith.select %arg0, %arg1, %arg2 : tensor<64xi32>
  %1 = arith.select %arg0, %arg3, %arg4 : tensor<64xi32>
  // CHECK: %[[R:.+]]:3 = scf.if %{{.*}} -> (tensor<64xi32>, tensor<64xi32>, tensor<64xi32>) {
  // CHECK:   scf.yield {{.*}} : tensor<64xi32>, tensor<64xi32>, tensor<64xi32>
  // CHECK: } else {
  // CHECK:   scf.yield {{.*}} : tensor<64xi32>, tensor<64xi32>, tensor<64xi32>
  // CHECK: }
  %2 = scf.if %arg0 -> (tensor<64xi32>) {
    %3 = arith.subi %arg1, %arg2 : tensor<64xi32>
    scf.yield %3 : tensor<64xi32>
  } else {
    scf.yield %arg1 : tensor<64xi32>
  }
  // CHECK: tt.return %[[R]]#1, %[[R]]#2, %[[R]]#0 : tensor<64xi32>, tensor<64xi32>, tensor<64xi32>
  tt.return %0, %1, %2 : tensor<64xi32>, tensor<64xi32>, tensor<64xi32>
}

// -----
// CHECK-LABEL: tt.func @users_in_if(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9_]+]]: i1
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9_]+]]: tensor<64xi32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9_]+]]: tensor<64xi32>
// CHECK-SAME:     %[[ARG3:[a-zA-Z0-9_]+]]: tensor<64xf32>
// CHECK-SAME:     %[[ARG4:[a-zA-Z0-9_]+]]: tensor<64xf32>
tt.func @users_in_if(%arg0: i1, %arg1: tensor<64xi32>, %arg2: tensor<64xi32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>) -> (tensor<64xi32>, tensor<64xf32>, tensor<64xi32>, tensor<64xi32>) {
  // CHECK: %[[CST:.*]] = arith.constant dense<8> : tensor<64xi32>
  %c8_i32 = arith.constant dense<8> : tensor<64xi32>
  // CHECK-NOT: arith.select
  %0 = arith.select %arg0, %arg1, %arg2 : tensor<64xi32>
  %1 = arith.select %arg0, %arg3, %arg4 : tensor<64xf32>
  // CHECK: %[[R:.+]]:4 = scf.if %[[ARG0]] -> (tensor<64xi32>, tensor<64xi32>, tensor<64xi32>, tensor<64xf32>) {
  // CHECK:   %[[MULI:.*]] = arith.muli %[[ARG1]], %[[ARG2]] : tensor<64xi32>
  // CHECK:   %[[ADDI:.*]] = arith.addi %[[ARG1]], %[[CST]] : tensor<64xi32>
  // CHECK:   scf.yield %[[MULI]], %[[ADDI]], %[[ARG1]], %[[ARG3]] : tensor<64xi32>, tensor<64xi32>, tensor<64xi32>, tensor<64xf32>
  // CHECK: } else {
  // CHECK:   %[[ADDI:.*]] = arith.subi %[[ARG2]], %[[CST]] : tensor<64xi32>
  // CHECK:   scf.yield %[[ARG1]], %[[ADDI]], %[[ARG2]], %[[ARG4]] : tensor<64xi32>, tensor<64xi32>, tensor<64xi32>, tensor<64xf32>
  // CHECK: }
  %2:2 = scf.if %arg0 -> (tensor<64xi32>, tensor<64xi32>) {
    %3 = arith.muli %0, %arg2 : tensor<64xi32>
    %4 = arith.addi %0, %c8_i32 : tensor<64xi32>
    scf.yield %3, %4 : tensor<64xi32>, tensor<64xi32>
  } else {
    %3 = arith.subi %0, %c8_i32 : tensor<64xi32>
    scf.yield %arg1, %3 : tensor<64xi32>, tensor<64xi32>
  }
  // CHECK: tt.return %[[R]]#2, %[[R]]#3, %[[R]]#0, %[[R]]#1 : tensor<64xi32>, tensor<64xf32>, tensor<64xi32>, tensor<64xi32>
  tt.return %0, %1, %2#0, %2#1 : tensor<64xi32>, tensor<64xf32>, tensor<64xi32>, tensor<64xi32>
}
