// RUN: not tf-mlir-translate -mlir-tf-to-hlo-text %s -tf-input-shapes=128,10 -tf-xla-emit-use-tuple-args -tf-xla-emit-return-tuple 2>&1 | FileCheck %s

module attributes {tf.versions = {producer = 179 : i32}} {
  func.func @main(%arg0: tensor<128x8xf32> {mhlo.sharding = "bad_sharding"}) {
    func.return
  }
}

// CHECK: failed to parse sharding 'bad_sharding'
