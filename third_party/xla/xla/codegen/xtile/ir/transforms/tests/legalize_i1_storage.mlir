// RUN: emitters_opt %s -xtile-legalize-i1-storage -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: xtile.entry_func @i1_nd_extract_insert(
// CHECK-SAME:    %[[IN:[^:]+]]: memref<16x32xi8>,
// CHECK-SAME:    %[[OUT:[^:]+]]: memref<16x32xi8>,
// CHECK-SAME:    %[[UNUSED:[^:]+]]: memref<8xi8>,
// CHECK-SAME:    %[[PID:[^:]+]]: index)
xtile.entry_func @i1_nd_extract_insert(
    %in: memref<16x32xi1>,
    %out: memref<16x32xi1>,
    %unused: memref<8xi1>,
    %pid: index) {
  %c0 = arith.constant 0 : index
  // CHECK-NOT: builtin.unrealized_conversion_cast
  // CHECK-DAG: %[[ZERO:.*]] = arith.constant dense<0> : tensor<4x8xi8>
  // CHECK: %[[LOAD:.*]] = xtile.extract %[[IN]][%[[PID]], %{{.*}}] [4, 8] [1, 1] : memref<16x32xi8> -> tensor<4x8xi8>
  // CHECK: %[[CMP:.*]] = arith.cmpi ne, %[[LOAD]], %[[ZERO]] : tensor<4x8xi8>
  // CHECK: %[[EXT:.*]] = arith.extui %[[CMP]] : tensor<4x8xi1> to tensor<4x8xi8>
  // CHECK: xtile.insert %[[EXT]] into %[[OUT]][%[[PID]], %{{.*}}] [4, 8] [1, 1] : tensor<4x8xi8> -> memref<16x32xi8>
  %tile = xtile.extract %in[%pid, %c0] [4, 8] [1, 1] : memref<16x32xi1> -> tensor<4x8xi1>
  xtile.insert %tile into %out[%pid, %c0] [4, 8] [1, 1] : tensor<4x8xi1> -> memref<16x32xi1>
  xtile.return
}

// -----

// CHECK-LABEL: xtile.entry_func @i1_scalar_extract_insert(
// CHECK-SAME:    %[[IN:[^:]+]]: memref<i8>,
// CHECK-SAME:    %[[OUT:[^:]+]]: memref<i8>,
// CHECK-SAME:    %[[PID:[^:]+]]: index)
xtile.entry_func @i1_scalar_extract_insert(
    %in: memref<i1>,
    %out: memref<i1>,
    %pid: index) {
  // CHECK-DAG: %[[ZERO:.*]] = arith.constant dense<0> : tensor<i8>
  // CHECK: %[[LOAD:.*]] = xtile.extract %[[IN]][] [] [] : memref<i8> -> tensor<i8>
  // CHECK: %[[CMP:.*]] = arith.cmpi ne, %[[LOAD]], %[[ZERO]] : tensor<i8>
  // CHECK: %[[EXT:.*]] = arith.extui %[[CMP]] : tensor<i1> to tensor<i8>
  // CHECK: xtile.insert %[[EXT]] into %[[OUT]][] [] [] : tensor<i8> -> memref<i8>
  %tile = xtile.extract %in[] [] [] : memref<i1> -> tensor<i1>
  xtile.insert %tile into %out[] [] [] : tensor<i1> -> memref<i1>
  xtile.return
}

// -----

// CHECK-LABEL: xtile.entry_func @i1_select_buffer_with_and_without_offset(
// CHECK-SAME:    %[[OUT_TBL:[^:]+]]: memref<2xi64>,
// CHECK-SAME:    %[[SCRATCH_TBL:[^:]+]]: memref<2xi64>,
// CHECK-SAME:    %{{[^:]+}}: memref<2xi64>,
// CHECK-SAME:    %{{[^:]+}}: index,
// CHECK-SAME:    %[[PID:[^:]+]]: index) attributes {has_rank, has_signal_buffer, num_scratch_buffers = 1 : i32}
xtile.entry_func @i1_select_buffer_with_and_without_offset(
    %out_tbl: memref<2xi64>,
    %scratch_tbl: memref<2xi64>,
    %signal_tbl: memref<2xi64>,
    %rank: index,
    %pid: index) attributes {
      has_rank, has_signal_buffer, num_scratch_buffers = 1 : i32} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  // CHECK-NOT: builtin.unrealized_conversion_cast
  // CHECK-DAG: %[[ZERO:.*]] = arith.constant dense<0> : tensor<4xi8>
  // CHECK: %[[SCRATCH:.*]] = xtile.select_buffer %[[SCRATCH_TBL]][%{{.*}}] offset %{{.*}} : memref<2xi64> -> memref<64xi8>
  // CHECK: %[[RAW:.*]] = xtile.extract %[[SCRATCH]][%[[PID]]] [4] [1] : memref<64xi8> -> tensor<4xi8>
  // CHECK: %[[UNPACKED:.*]] = arith.cmpi ne, %[[RAW]], %[[ZERO]] : tensor<4xi8>
  // CHECK: %[[OUT:.*]] = xtile.select_buffer %[[OUT_TBL]][%{{.*}}] : memref<2xi64> -> memref<64xi8>
  // CHECK: %[[PACKED:.*]] = arith.extui %[[UNPACKED]] : tensor<4xi1> to tensor<4xi8>
  // CHECK: xtile.insert %[[PACKED]] into %[[OUT]][%[[PID]]] [4] [1] : tensor<4xi8> -> memref<64xi8>
  %unused_buf = xtile.select_buffer %out_tbl[%c0] : memref<2xi64> -> memref<64xi1>
  %scratch = xtile.select_buffer %scratch_tbl[%c1] offset %c128 : memref<2xi64> -> memref<64xi1>
  %tile = xtile.extract %scratch[%pid] [4] [1] : memref<64xi1> -> tensor<4xi1>
  %out = xtile.select_buffer %out_tbl[%c0] : memref<2xi64> -> memref<64xi1>
  xtile.insert %tile into %out[%pid] [4] [1] : tensor<4xi1> -> memref<64xi1>
  xtile.return
}

// -----

!layout_i1 = memref<16x8xi1, #xtile.layout<[0, 1]>, #nvvm.memory_space<global>>
// CHECK-LABEL: xtile.entry_func @i1_layout_memspace(
// CHECK-SAME:    %[[IN:[^:]+]]: memref<16x8xi8, #xtile.layout<[0, 1]>, #nvvm.memory_space<global>>,
// CHECK-SAME:    %[[OUT:[^:]+]]: memref<8xi8>,
// CHECK-SAME:    %[[PID:[^:]+]]: index)
xtile.entry_func @i1_layout_memspace(
    %in: !layout_i1,
    %out: memref<8xi1>,
    %pid: index) {
  %c0 = arith.constant 0 : index
  // CHECK: %[[LOAD:.*]] = xtile.extract %[[IN]][%[[PID]], %{{.*}}] [1, 8] [1, 1] : memref<16x8xi8, #xtile.layout<[0, 1]>, #nvvm.memory_space<global>> -> tensor<8xi8>
  // CHECK: %[[CMP:.*]] = arith.cmpi ne, %[[LOAD]], %{{.*}} : tensor<8xi8>
  // CHECK: %[[EXT:.*]] = arith.extui %[[CMP]] : tensor<8xi1> to tensor<8xi8>
  // CHECK: xtile.insert %[[EXT]] into %[[OUT]][%{{.*}}] [8] [1] : tensor<8xi8> -> memref<8xi8>
  %tile = xtile.extract %in[%pid, %c0] [1, 8] [1, 1] : !layout_i1 -> tensor<8xi1>
  xtile.insert %tile into %out[%c0] [8] [1] : tensor<8xi1> -> memref<8xi1>
  xtile.return
}

// -----

func.func @unlegalized_func_arg_i1_extract(%in: memref<4xi1>) -> tensor<4xi1> {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{'xtile.extract' op failed to legalize i1 tile extract}}
  %tile = xtile.extract %in[%c0] [4] [1] : memref<4xi1> -> tensor<4xi1>
  return %tile : tensor<4xi1>
}

// -----

func.func private @consume(memref<8x4xi1, strided<[1, 8]>>)

xtile.entry_func @unhandled_i1_memref_consumer(%in: memref<4x8xi1>, %pid: index) {
  // expected-error @-1 {{'builtin.unrealized_conversion_cast' op unhandled i1 memref user}}
  %t = memref.transpose %in (d0, d1) -> (d1, d0) : memref<4x8xi1> to memref<8x4xi1, strided<[1, 8]>>
  func.call @consume(%t) : (memref<8x4xi1, strided<[1, 8]>>) -> ()
  xtile.return
}
