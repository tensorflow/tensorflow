// RUN: tf-mlir-translate -mlir-tf-mlir-to-str-attr -mlir-print-local-scope %s | FileCheck %s

module attributes {tf.versions = {producer = 888 : i32}} {
  func.func @main(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.Identity"(%arg0) : (tensor<?xi32>) -> tensor<?xi32> loc(unknown)
    func.return %0 : tensor<?xi32> loc(unknown)
  } loc(unknown)
} loc(unknown)

// CHECK: "module attributes {tf.versions = {producer = 888 : i32}} {\0A func.func @main(%arg0: tensor<?xi32>) -> tensor<?xi32> {\0A %0 = \22tf.Identity\22(%arg0) : (tensor<?xi32>) -> tensor<?xi32>\0A return %0 : tensor<?xi32>\0A }\0A}"