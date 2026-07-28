// RUN: fusion_compiler_opt %s --xtile-cpu-lower-xtile-entry -split-input-file | FileCheck %s

xtile.entry_func @simple_wrap(%input: memref<1024xf32> {xla.some_attr = 1},
                             %output: memref<32xf64>,
                             %tile_id: index) attributes {xtile.tiling_info = #xtile.tiling_info<tile_count:1012, tiles_per_workgroup:64>} {
  xtile.return
}

// CHECK-DAG: #[[MAP:.*]] = #xla.indexing_map<"(workgroup_id) -> (min(workgroup_id * 64, 1012), min(workgroup_id * 64 + 64, 1012)), domain: workgroup_id in [0, 15]">

// CHECK: func.func @simple_wrap(%[[CALL_FRAME:.*]]: !xla_cpu.call_frame) -> !xla_cpu.error {

// CHECK-DAG: %[[STEP:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[INPUT:.*]] = xla_cpu.load %[[CALL_FRAME]], 0 : memref<1024xf32>
// CHECK-DAG: %[[OUTPUT:.*]] = xla_cpu.load %[[CALL_FRAME]], 1 : memref<32xf64>
// CHECK-DAG: %[[WORKGROUP_ID:.*]] = xla_cpu.extract_workgroup_id %[[CALL_FRAME]], x

// CHECK: %[[APPLY_INDEXING:.*]]:2 = xla.apply_indexing #[[MAP]](%[[WORKGROUP_ID]])
// CHECK: scf.for %[[IDX:.*]] = %[[APPLY_INDEXING]]#0 to %[[APPLY_INDEXING]]#1 step %[[STEP]] {
// CHECK:   func.call @[[IMPL_FUNC:.*]](%[[INPUT]], %[[OUTPUT]], %[[IDX]]) : (memref<1024xf32>, memref<32xf64>, index) -> ()
// CHECK: }

// CHECK: %[[SUCCESS:.*]] = xla_cpu.success : !xla_cpu.error
// CHECK: return %[[SUCCESS]] : !xla_cpu.error

// CHECK: func.func @[[IMPL_FUNC]](
// CHECK-SAME: %{{.*}}: memref<1024xf32> {xla.some_attr = 1 : i64},
// CHECK-SAME: %{{.*}}: memref<32xf64>,
// CHECK-SAME: %{{.*}}: index)
// CHECK-SAME: attributes {llvm.always_inline, llvm.linkage = #llvm.linkage<internal>
// CHECK: return


// -----
