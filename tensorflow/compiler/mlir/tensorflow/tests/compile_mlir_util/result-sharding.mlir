// RUN: tf-mlir-translate -mlir-tf-to-hlo-text %s -tf-input-shapes=128,10:10,1024:128,1024 -tf-xla-emit-use-tuple-args -tf-xla-emit-return-tuple | FileCheck %s

module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 351 : i32}} {
  func @main(%arg0: tensor<128x10xf32>, %arg1: tensor<10x1024xf32>, %arg2: tensor<128x1024xf32>) -> (tensor<128x10xf32> {mhlo.sharding = "\08\03\1A\02\01\02\22\02\00\01"}, tensor<10x1024xf32> {mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, tensor<128x1024xf32> {mhlo.sharding = ""}) {
    return %arg0, %arg1, %arg2 : tensor<128x10xf32>, tensor<10x1024xf32>, tensor<128x1024xf32>
  }
}

// The following xla::OpSharding protos are used:
//  Serialized string:
//   "\08\03\1A\02\01\02\22\02\00\01"
//  Proto debug string:
//   type: OTHER
//   tile_assignment_dimensions: 1
//   tile_assignment_dimensions: 2
//   tile_assignment_devices: 0
//   tile_assignment_devices: 1
//
//  Serialized string:
//   "\08\01\1A\01\01\22\01\00"
//  Proto debug string:
//   type: MAXIMAL
//   tile_assignment_dimensions: 1
//   tile_assignment_devices: 0
//
//  Serialized string:
//   ""
//  Proto debug string (empty but would equivalent to):
//   type: REPLICATED

// CHECK-LABEL: HloModule main
// CHECK:       ENTRY %main.{{[0-9]+}}
// CHECK-SAME:  (arg_tuple.{{[0-9]+}}: (f32[128,10], f32[10,1024], f32[128,1024])) -> (f32[128,10], f32[10,1024], f32[128,1024]) {
// CHECK:         ROOT %tuple.{{[0-9]+}}
// CHECK-SAME:    sharding={
// CHECK-SAME:    {devices=[1,2]0,1}
// CHECK-SAME:    {maximal device=0}
// CHECK-SAME:    {replicated}
// CHECK-SAME:    }
