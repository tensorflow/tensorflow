// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - 2>&1 | FileCheck %s; if [[ ${PIPESTATUS[0]} != 0  &&  ${PIPESTATUS[1]} == 0 ]]; then exit 0; else exit 1; fi

// CHECK: Graph export failed: Failed precondition: entry function `main` must be present

func @const() {
^bb0:
  %0:2 = "_tf.Const"() {device = "TPU:0", name = "const", dtype = "tfdtype$DT_INT32", value = dense<[1, 2]> : tensor<2xi32>} : () -> (tensor<2xi32>, !_tf.control)
  return
}
