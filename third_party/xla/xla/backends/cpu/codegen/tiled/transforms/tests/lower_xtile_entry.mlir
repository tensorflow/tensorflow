// RUN: fusion_compiler_opt %s --xtile-cpu-lower-xtile-entry -split-input-file \
// RUN:   | FileCheck %s
// RUN: fusion_compiler_opt %s -split-input-file \
// RUN:   --xtile-cpu-lower-xtile-entry="prefer_vector_width=256" \
// RUN:   | FileCheck %s --check-prefix=PREFER_VECTOR

xtile.entry_func @simple_wrap(%input: memref<1024xf32> {xla.some_attr = 1},
                             %output: memref<32xf64>,
                             %pid: index) {
  xtile.return
}

// CHECK: func.func @simple_wrap(%[[CALL_FRAME:.*]]: !xla_cpu.call_frame) -> !xla_cpu.error {

// CHECK-DAG: %[[INPUT:.*]] = xla_cpu.load %[[CALL_FRAME]], 0 : memref<1024xf32>
// CHECK-DAG: %[[OUTPUT:.*]] = xla_cpu.load %[[CALL_FRAME]], 1 : memref<32xf64>
// CHECK-DAG: %[[WORKGROUP_ID:.*]] = xla_cpu.extract_workgroup_id %[[CALL_FRAME]], x

// CHECK:   call @simple_wrap_impl(%[[INPUT]], %[[OUTPUT]], %[[WORKGROUP_ID]]) : (memref<1024xf32>, memref<32xf64>, index) -> ()

// CHECK: %[[SUCCESS:.*]] = xla_cpu.success : !xla_cpu.error
// CHECK: return %[[SUCCESS]] : !xla_cpu.error

// CHECK: func.func @simple_wrap_impl(
// CHECK-SAME: %{{.*}}: memref<1024xf32> {xla.some_attr = 1 : i64},
// CHECK-SAME: %{{.*}}: memref<32xf64>,
// CHECK-SAME: %{{.*}}: index)
// CHECK-SAME: attributes {llvm.always_inline, llvm.linkage = #llvm.linkage<internal>
// CHECK: return

// PREFER_VECTOR:      func.func @simple_wrap(%{{.*}}: !xla_cpu.call_frame) -> !xla_cpu.error
// PREFER_VECTOR-SAME:   attributes {llvm.passthrough =
// PREFER_VECTOR-SAME:   ["prefer-vector-width", "256"]]} {
// PREFER_VECTOR:      func.func @simple_wrap_impl(
// PREFER_VECTOR-NOT:    llvm.passthrough

// -----
