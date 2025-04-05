// RUN: triton-opt %s -triton-rewrite-tensor-pointer -split-input-file | FileCheck %s

tt.func public @rewrite_load(%arg0: !tt.ptr<f16>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c32_i64 = arith.constant 32 : i64
  %c128_i64 = arith.constant 128 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<128x32xf16>
  %0 = tt.make_tensor_ptr %arg0, [%c128_i64, %c32_i64], [%c1_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : !tt.ptr<tensor<128x32xf16>>
  %load = tt.load %0 {boundaryCheck = array<i32: 1>, padding = 2 : i32} : !tt.ptr<tensor<128x32xf16>>
  tt.return
}

// CHECK-LABEL: tt.func public @rewrite_load(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9_]+]]: !tt.ptr<f16>
// CHECK-DAG: %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK-DAG: %[[C32_I64:.*]] = arith.constant 32 : i64
// CHECK-DAG: %[[C128_I64:.*]] = arith.constant 128 : i64
// CHECK: %[[CST:.*]] = arith.constant dense<0.000000e+00> : tensor<128x32xf16>
// CHECK: %[[EXTSI0:.*]] = arith.extsi %[[C0_I32]] : i32 to i64
// CHECK: %[[EXTSI1:.*]] = arith.extsi %[[C0_I32]] : i32 to i64
// CHECK: %[[SPLAT0:.*]] = tt.splat %[[ARG0]] : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>>
// CHECK: %[[SPLAT1:.*]] = tt.splat %[[EXTSI0]] : i64 -> tensor<128xi64>
// CHECK: %[[MAKE_RANGE0:.*]] = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
// CHECK: %[[EXTSI2:.*]] = arith.extsi %[[MAKE_RANGE0]] : tensor<128xi32> to tensor<128xi64>
// CHECK: %[[ADDI0:.*]] = arith.addi %[[SPLAT1]], %[[EXTSI2]] : tensor<128xi64>
// CHECK: %[[EXPAND_DIMS0:.*]] = tt.expand_dims %[[ADDI0]] {axis = 1 : i32} : tensor<128xi64> -> tensor<128x1xi64>
// CHECK: %[[SPLAT2:.*]] = tt.splat %[[C1_I64]] : i64 -> tensor<128x1xi64>
// CHECK: %[[MULI0:.*]] = arith.muli %[[EXPAND_DIMS0]], %[[SPLAT2]] : tensor<128x1xi64>
// CHECK: %[[BROADCAST0:.*]] = tt.broadcast %[[MULI0]] : tensor<128x1xi64> -> tensor<128x32xi64>
// CHECK: %[[ADDPTR0:.*]] = tt.addptr %[[SPLAT0]], %[[BROADCAST0]] : tensor<128x32x!tt.ptr<f16>>, tensor<128x32xi64>
// CHECK: %[[SPLAT3:.*]] = tt.splat %[[EXTSI1]] : i64 -> tensor<32xi64>
// CHECK: %[[MAKE_RANGE1:.*]] = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
// CHECK: %[[EXTSI3:.*]] = arith.extsi %[[MAKE_RANGE1]] : tensor<32xi32> to tensor<32xi64>
// CHECK: %[[ADDI1:.*]] = arith.addi %[[SPLAT3]], %[[EXTSI3]] : tensor<32xi64>
// CHECK: %[[EXPAND_DIMS1:.*]] = tt.expand_dims %[[ADDI1]] {axis = 0 : i32} : tensor<32xi64> -> tensor<1x32xi64>
// CHECK: %[[SPLAT4:.*]] = tt.splat %[[C1_I64]] : i64 -> tensor<1x32xi64>
// CHECK: %[[MULI1:.*]] = arith.muli %[[EXPAND_DIMS1]], %[[SPLAT4]] : tensor<1x32xi64>
// CHECK: %[[BROADCAST1:.*]] = tt.broadcast %[[MULI1]] : tensor<1x32xi64> -> tensor<128x32xi64>
// CHECK: %[[ADDPTR1:.*]] = tt.addptr %[[ADDPTR0]], %[[BROADCAST1]] : tensor<128x32x!tt.ptr<f16>>, tensor<128x32xi64>
// CHECK: %[[C0_I64:.*]] = arith.constant 0 : i64
// CHECK: %[[SPLAT5:.*]] = tt.splat %[[C0_I64]] : i64 -> tensor<1x32xi64>
// CHECK: %[[CMP0:.*]] = arith.cmpi sge, %[[EXPAND_DIMS1]], %[[SPLAT5]] : tensor<1x32xi64>
// CHECK: %[[SPLAT6:.*]] = tt.splat %[[C32_I64]] : i64 -> tensor<1x32xi64>
// CHECK: %[[CMPI:.*]] = arith.cmpi slt, %[[EXPAND_DIMS1]], %[[SPLAT6]] : tensor<1x32xi64>
// CHECK: %[[ANDI:.*]] = arith.andi %[[CMP0]], %[[CMPI]] : tensor<1x32xi1>
// CHECK: %[[BROADCAST2:.*]] = tt.broadcast %[[ANDI]] : tensor<1x32xi1> -> tensor<128x32xi1>
// CHECK: %[[OTHER:.*]] = arith.constant 0x7E00 : f16
// CHECK: %[[SPLAT7:.*]] = tt.splat %[[OTHER]] : f16 -> tensor<128x32xf16>
// CHECK: %[[LOAD:.*]] = tt.load %[[ADDPTR1]], %[[BROADCAST2]], %[[SPLAT7]] : tensor<128x32x!tt.ptr<f16>>
// CHECK: tt.return

// -----
tt.func public @rewrite_store(%arg0: !tt.ptr<f16>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c32_i64 = arith.constant 32 : i64
  %c128_i64 = arith.constant 128 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<128x32xf16>
  %0 = tt.make_tensor_ptr %arg0, [%c128_i64, %c32_i64], [%c1_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : !tt.ptr<tensor<128x32xf16>>
  tt.store %0, %cst: !tt.ptr<tensor<128x32xf16>>
  tt.return
}

// CHECK-LABEL: tt.func public @rewrite_store(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9_]+]]: !tt.ptr<f16>
// CHECK-DAG: %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK-DAG: %[[C32_I64:.*]] = arith.constant 32 : i64
// CHECK-DAG: %[[C128_I64:.*]] = arith.constant 128 : i64
// CHECK: %[[CST:.*]] = arith.constant dense<0.000000e+00> : tensor<128x32xf16>
// CHECK: %[[EXTSI0:.*]] = arith.extsi %[[C0_I32]] : i32 to i64
// CHECK: %[[EXTSI1:.*]] = arith.extsi %[[C0_I32]] : i32 to i64
// CHECK: %[[SPLAT0:.*]] = tt.splat %[[ARG0]] : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>>
// CHECK: %[[SPLAT1:.*]] = tt.splat %[[EXTSI0]] : i64 -> tensor<128xi64>
// CHECK: %[[MAKE_RANGE0:.*]] = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
// CHECK: %[[EXTSI2:.*]] = arith.extsi %[[MAKE_RANGE0]] : tensor<128xi32> to tensor<128xi64>
// CHECK: %[[ADDI0:.*]] = arith.addi %[[SPLAT1]], %[[EXTSI2]] : tensor<128xi64>
// CHECK: %[[EXPAND_DIMS0:.*]] = tt.expand_dims %[[ADDI0]] {axis = 1 : i32} : tensor<128xi64> -> tensor<128x1xi64>
// CHECK: %[[SPLAT2:.*]] = tt.splat %[[C1_I64]] : i64 -> tensor<128x1xi64>
// CHECK: %[[MULI0:.*]] = arith.muli %[[EXPAND_DIMS0]], %[[SPLAT2]] : tensor<128x1xi64>
// CHECK: %[[BROADCAST0:.*]] = tt.broadcast %[[MULI0]] : tensor<128x1xi64> -> tensor<128x32xi64>
// CHECK: %[[ADDPTR0:.*]] = tt.addptr %[[SPLAT0]], %[[BROADCAST0]] : tensor<128x32x!tt.ptr<f16>>, tensor<128x32xi64>
// CHECK: %[[SPLAT3:.*]] = tt.splat %[[EXTSI1]] : i64 -> tensor<32xi64>
// CHECK: %[[MAKE_RANGE1:.*]] = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
// CHECK: %[[EXTSI3:.*]] = arith.extsi %[[MAKE_RANGE1]] : tensor<32xi32> to tensor<32xi64>
// CHECK: %[[ADDI1:.*]] = arith.addi %[[SPLAT3]], %[[EXTSI3]] : tensor<32xi64>
// CHECK: %[[EXPAND_DIMS1:.*]] = tt.expand_dims %[[ADDI1]] {axis = 0 : i32} : tensor<32xi64> -> tensor<1x32xi64>
// CHECK: %[[SPLAT4:.*]] = tt.splat %[[C1_I64]] : i64 -> tensor<1x32xi64>
// CHECK: %[[MULI1:.*]] = arith.muli %[[EXPAND_DIMS1]], %[[SPLAT4]] : tensor<1x32xi64>
// CHECK: %[[BROADCAST1:.*]] = tt.broadcast %[[MULI1]] : tensor<1x32xi64> -> tensor<128x32xi64>
// CHECK: %[[ADDPTR1:.*]] = tt.addptr %[[ADDPTR0]], %[[BROADCAST1]] : tensor<128x32x!tt.ptr<f16>>, tensor<128x32xi64>
// CHECK: tt.store %[[ADDPTR1]], %[[CST]] : tensor<128x32x!tt.ptr<f16>>
// CHECK: tt.return

// -----
tt.func public @rewrite_for(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c0_i32 = arith.constant 0 : i32
  %c32_i32 = arith.constant 32 : i32
  %c1_i64 = arith.constant 1 : i64
  %c32_i64 = arith.constant 32 : i64
  %c128_i64 = arith.constant 128 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<128x32xf16>
  %0 = tt.make_tensor_ptr %arg0, [%c128_i64, %c32_i64], [%c1_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : !tt.ptr<tensor<128x32xf16>>
  %1:2 = scf.for %arg2 = %c0 to %c32 step %c1 iter_args(%arg3 = %cst, %arg4 = %0) -> (tensor<128x32xf16>, !tt.ptr<tensor<128x32xf16>>) {
    %3 = tt.load %arg4 {boundaryCheck = array<i32: 1>, padding = 2 : i32} : !tt.ptr<tensor<128x32xf16>>
    %4 = arith.addf %arg3, %3 : tensor<128x32xf16>
    %5 = tt.advance %arg4, [%c32_i32, %c0_i32] : !tt.ptr<tensor<128x32xf16>>
    scf.yield %4, %5 : tensor<128x32xf16>, !tt.ptr<tensor<128x32xf16>>
  } {tt.num_stages = 3 : i32}
  %2 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>>
  tt.store %2, %1#0 : tensor<128x32x!tt.ptr<f16>>
  tt.return
}

// CHECK-LABEL: tt.func public @rewrite_for(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9_]+]]: !tt.ptr<f16>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9_]+]]: !tt.ptr<f16>
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
// CHECK-DAG: %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[C32_I32:.*]] = arith.constant 32 : i32
// CHECK-DAG: %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK-DAG: %[[C32_I64:.*]] = arith.constant 32 : i64
// CHECK-DAG: %[[C128_I64:.*]] = arith.constant 128 : i64
// CHECK: %[[CST:.*]] = arith.constant dense<0.000000e+00> : tensor<128x32xf16>
// CHECK: %[[EXTSI0:.*]] = arith.extsi %[[C0_I32]] : i32 to i64
// CHECK: %[[EXTSI1:.*]] = arith.extsi %[[C0_I32]] : i32 to i64
// CHECK: %[[FOR:.*]] = scf.for %[[ARG2:.*]] = %[[C0]] to %[[C32]] step %[[C1]]
// CHECK-SAME: iter_args(%[[ARG3:.*]] = %[[CST]], %[[ARG4:.*]] = %[[EXTSI0]], %[[ARG5:.*]] = %[[EXTSI1]]) -> (tensor<128x32xf16>, i64, i64)
// CHECK: %[[EXTSI2:.*]] = arith.extsi %[[C32_I32]] : i32 to i64
// CHECK: %[[ADDI0:.*]] = arith.addi %[[ARG4]], %[[EXTSI2]] : i64
// CHECK: %[[EXTSI3:.*]] = arith.extsi %[[C0_I32]] : i32 to i64
// CHECK: %[[ADDI1:.*]] = arith.addi %[[ARG5]], %[[EXTSI3]] : i64
// CHECK: scf.yield %{{.*}}, %[[ADDI0]], %[[ADDI1]] : tensor<128x32xf16>, i64, i64
// CHECK: tt.num_stages = 3

// -----
tt.func public @rewrite_if(%arg0: !tt.ptr<f16>, %arg1: i1, %arg2: tensor<128x32xf32>) -> tensor<128x32xf16> {
  %c0_i32 = arith.constant 0 : i32
  %c32_i32 = arith.constant 32 : i32
  %c1_i64 = arith.constant 1 : i64
  %c32_i64 = arith.constant 32 : i64
  %c128_i64 = arith.constant 128 : i64
  %0 = tt.make_tensor_ptr %arg0, [%c128_i64, %c32_i64], [%c1_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : !tt.ptr<tensor<128x32xf16>>
  %1:2 = scf.if %arg1 -> (tensor<128x32xf16>, !tt.ptr<tensor<128x32xf16>>) {
    %2 = tt.advance %0, [%c32_i32, %c0_i32] : !tt.ptr<tensor<128x32xf16>>
    %3 = arith.truncf %arg2 : tensor<128x32xf32> to tensor<128x32xf16>
    scf.yield %3, %2 : tensor<128x32xf16>, !tt.ptr<tensor<128x32xf16>>
  } else {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x32xf16>
    scf.yield %cst, %0 : tensor<128x32xf16>, !tt.ptr<tensor<128x32xf16>>
  }
  %4 = tt.load %1#1 {boundaryCheck = array<i32: 1>, padding = 2 : i32} : !tt.ptr<tensor<128x32xf16>>
  %5 = arith.addf %1#0, %4 : tensor<128x32xf16>
  tt.return %5 : tensor<128x32xf16>
}

// CHECK-LABEL: tt.func public @rewrite_if(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9_]+]]: !tt.ptr<f16>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9_]+]]: i1
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9_]+]]: tensor<128x32xf32>
// CHECK-DAG: %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[C32_I32:.*]] = arith.constant 32 : i32
// CHECK-DAG: %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK-DAG: %[[C32_I64:.*]] = arith.constant 32 : i64
// CHECK-DAG: %[[C128_I64:.*]] = arith.constant 128 : i64
// CHECK: %[[EXTSI0:.*]] = arith.extsi %[[C0_I32]] : i32 to i64
// CHECK: %[[EXTSI1:.*]] = arith.extsi %[[C0_I32]] : i32 to i64
// CHECK: %[[IF:.*]]:3 = scf.if %[[ARG1]] -> (tensor<128x32xf16>, i64, i64) {
// CHECK:   %[[EXTSI2:.*]] = arith.extsi %[[C32_I32]] : i32 to i64
// CHECK:   %[[ADDI0:.*]] = arith.addi %[[EXTSI0]], %[[EXTSI2]] : i64
// CHECK:   %[[EXTSI3:.*]] = arith.extsi %[[C0_I32]] : i32 to i64
// CHECK:   %[[ADDI1:.*]] = arith.addi %[[EXTSI1]], %[[EXTSI3]] : i64
// CHECK:   %[[TRUNCF:.*]] = arith.truncf %[[ARG2]] : tensor<128x32xf32> to tensor<128x32xf16>
// CHECK:   scf.yield %[[TRUNCF]], %[[ADDI0]], %[[ADDI1]] : tensor<128x32xf16>, i64, i64
// CHECK: } else {
// CHECK:   %[[CST:.*]] = arith.constant dense<0.000000e+00> : tensor<128x32xf16>
// CHECK:   scf.yield %[[CST]], %[[EXTSI0]], %[[EXTSI1]] : tensor<128x32xf16>, i64, i64
// CHECK: }
// CHECK: %{{.*}} = tt.splat %[[IF]]#1 : i64 -> tensor<128xi64>
// CHECK: %{{.*}} = tt.splat %[[IF]]#2 : i64 -> tensor<32xi64>
// CHECK: %{{.*}} = arith.addf %[[IF]]#0, %{{.*}} : tensor<128x32xf16>


// -----
tt.func public @asm_in_loop(%arg0: !tt.ptr<bf16>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c0_i64 = arith.constant 0 : i64
  %c128_i64 = arith.constant 128 : i64
  %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %1 = tt.make_tensor_ptr %arg0, [%c128_i64, %c128_i64], [%c128_i64, %c0_i64], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : !tt.ptr<tensor<128x128xbf16>>
  %2:1 = scf.for %arg1 = %c0_i32 to %c1_i32 step %c1_i32 iter_args(%arg2 = %1) -> (!tt.ptr<tensor<128x128xbf16>>)  : i32 {
    %3:2 = tt.elementwise_inline_asm "asm_multiple_results" {constraints = "=r,=r,r", packed_element = 1 : i32, pure = true} %0 : tensor<16xi32> -> tensor<16xi16>, tensor<16xi16>
    %4 = tt.advance %arg2, [%c0_i32, %c0_i32] : !tt.ptr<tensor<128x128xbf16>>
    scf.yield %4 : !tt.ptr<tensor<128x128xbf16>>
  }
  tt.return
}

// CHECK-LABEL: tt.func public @asm_in_loop(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9_]+]]: !tt.ptr<bf16>
// CHECK-DAG: %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[C1_I32:.*]] = arith.constant 1 : i32
// CHECK-DAG: %[[C0_I64:.*]] = arith.constant 0 : i64
// CHECK-DAG: %[[C128_I64:.*]] = arith.constant 128 : i64
// CHECK: %[[RANGE:.*]] = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK: %[[EXTSI0:.*]] = arith.extsi %[[C0_I32]] : i32 to i64
// CHECK: %[[EXTSI1:.*]] = arith.extsi %[[C0_I32]] : i32 to i64
// CHECK: %[[FOR:.*]]:2 = scf.for %[[ARG1:.*]] = %[[C0_I32]] to %[[C1_I32]] step %[[C1_I32]]
// CHECK-SAME: iter_args(%[[ARG2:.*]] = %[[EXTSI0]], %[[ARG3:.*]] = %[[EXTSI1]]) -> (i64, i64)
// CHECK: %[[ASM:.*]]:2 = tt.elementwise_inline_asm "asm_multiple_results" {{.*}} %[[RANGE]] : tensor<16xi32> -> tensor<16xi16>, tensor<16xi16>
