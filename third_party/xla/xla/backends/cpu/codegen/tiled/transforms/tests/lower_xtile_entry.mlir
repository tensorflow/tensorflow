// RUN: fusion_compiler_opt %s --xtile-cpu-lower-xtile-entry -split-input-file | FileCheck %s

xtile.entry_func @simple_wrap(%input: memref<1024xf32> {xla.some_attr = 1},
                             %output: memref<32xf64>,
                             %tile_id: index) attributes {xtile.tiling_info = #xtile.tiling_info<tile_count:1012, tiles_per_workgroup:64>} {
  xtile.return
}

// CHECK: func.func @simple_wrap(%[[CALL_FRAME:.*]]: !xla_cpu.call_frame) -> !xla_cpu.error {

// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[STEP:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[TILES_PER_WORKGROUP:.*]] = arith.constant 64 : index
// CHECK-DAG: %[[TILE_COUNT:.*]] = arith.constant 1012 : index

// CHECK: %[[INPUT:.*]] = xla_cpu.load %[[CALL_FRAME]], 0 : memref<1024xf32>
// CHECK: %[[OUTPUT:.*]] = xla_cpu.load %[[CALL_FRAME]], 1 : memref<32xf64>
// CHECK: %[[WORKGROUP_ID:.*]] = xla_cpu.extract_workgroup_id %[[CALL_FRAME]], x

// CHECK: %[[BOUNDED_WORKGROUP_ID:.*]] = arith.maxsi %[[WORKGROUP_ID]], %[[C0]] : index
// CHECK: %[[START_IDX:.*]] = arith.muli %[[BOUNDED_WORKGROUP_ID]], %[[TILES_PER_WORKGROUP]] overflow<nsw, nuw> : index
// CHECK: %[[CLAMPED_START_IDX:.*]] = arith.minsi %[[START_IDX]], %[[TILE_COUNT]] : index
// CHECK: %[[END_IDX:.*]] = arith.addi %[[START_IDX]], %[[TILES_PER_WORKGROUP]] overflow<nsw, nuw> : index
// CHECK: %[[CLAMPED_END_IDX:.*]] = arith.minsi %[[END_IDX]], %[[TILE_COUNT]] : index
// CHECK: scf.for %[[IDX:.*]] = %[[CLAMPED_START_IDX]] to %[[CLAMPED_END_IDX]] step %[[STEP]] {
// CHECK:   func.call @[[IMPL_FUNC:.*]](%[[INPUT]], %[[OUTPUT]], %[[IDX]]) : (memref<1024xf32>, memref<32xf64>, index) -> ()
// CHECK: }

// CHECK: %[[SUCCESS:.*]] = xla_cpu.success : !xla_cpu.error
// CHECK: return %[[SUCCESS]] : !xla_cpu.error

// CHECK: func.func @[[IMPL_FUNC]](
// CHECK-SAME: %{{.*}}: memref<1024xf32> {xla.some_attr = 1 : i64},
// CHECK-SAME: %{{.*}}: memref<32xf64>,
// CHECK-SAME: %{{.*}}: index)
// CHECK-SAME: attributes {always_inline, llvm.linkage = #llvm.linkage<internal>
// CHECK: return


// -----
