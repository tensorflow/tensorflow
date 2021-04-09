// RUN: not tf-mlir-translate -mlir-tf-to-hlo-text %s -tf-input-shapes=128,10 -emit-use-tuple-args -emit-return-tuple 2>&1 | FileCheck %s

module attributes {tf.versions = {producer = 179 : i32}} {
  func @main(%arg0: tensor<128x8xf32> {mhlo.sharding = "bad_sharding"}) {
    return
  }
}

// CHECK: failed to parse argument sharding 0 'bad_sharding'
