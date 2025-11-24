// RUN: emitters_opt %s -split-input-file -xla-cpu-peel-workgroup-loop | FileCheck %s

#map = #xla.indexing_map<"(d0)[s0] -> (d0 + s0, s0), domain:"
  "d0 in [0, 20], s0 in [0, 7], d0 + s0 in [0, 15]">
func.func @peel_loop(%input: tensor<16x8xf32>, %init: f32) -> f32 {
  %workgroup_id = xla.workgroup_id x
  %sum = xla.loop (%workgroup_id)[%i] -> (%r0, %r1) in #map
      iter_args(%sum = %init) -> f32 {
    %extracted = tensor.extract %input[%r0, %r1] : tensor<16x8xf32>
    %add = arith.addf %sum, %extracted : f32
    xla.yield %add : f32
  }
  func.return %sum : f32
}

// CHECK-DAG: #[[UNCONSTRAINED_MAP:.*]] = #xla.indexing_map<"(d0)[s0] -> (d0 + s0, s0), domain: d0 in [0, 8], s0 in [0, 7]">
// CHECK-DAG: #[[CONSTRAINED_MAP:.*]] = #xla.indexing_map<"(d0)[s0] -> (d0 + s0, s0), domain: d0 in [9, 20], s0 in [0, 7], d0 + s0 in [0, 15]">
// CHECK: @peel_loop(%[[TENSOR:.*]]: tensor<16x8xf32>, %[[INIT:.*]]: f32)
// CHECK-DAG: %[[WORK_GROUP_LIMIT:.*]] = arith.constant 8 : index
// CHECK-DAG: %[[WORKGROUP_ID:.*]] = xla.workgroup_id  x
// CHECK-DAG: %[[WORK_GROUP_COND:.*]] =  arith.cmpi sle, %[[WORKGROUP_ID]], %[[WORK_GROUP_LIMIT]]
// CHECK: %[[IF_RESULT:.*]] = scf.if %[[WORK_GROUP_COND]]
// CHECK: xla.loop (%[[WORKGROUP_ID]])
// CHECK-SAME: in #[[UNCONSTRAINED_MAP]]
// CHECK: else
// CHECK: xla.loop (%[[WORKGROUP_ID]])
// CHECK-SAME: in #[[CONSTRAINED_MAP]]
// CHECK: return %[[IF_RESULT]]
