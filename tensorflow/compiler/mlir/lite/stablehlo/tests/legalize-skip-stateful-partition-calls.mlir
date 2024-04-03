// RUN: odml-to-stablehlo-opt %s --tf-stablehlo=skip-stateful-partitioned-call=true | FileCheck %s --check-prefix=CHECK-SKIP
// RUN: odml-to-stablehlo-opt %s --tf-stablehlo=skip-stateful-partitioned-call=false | FileCheck %s --check-prefix=CHECK-NOSKIP

module {
  func.func @partitioned_call(%arg0: tensor<1x2x2x3xf32>) -> (tensor<1x2x2x3xf32>) {
    %0 = "tf.StatefulPartitionedCall"(%arg0) <{
      config = "", config_proto = "", executor_type = "", f = @some_func
    }> {
      _collective_manager_ids = [], device = ""
    } : (tensor<1x2x2x3xf32>) -> tensor<1x2x2x3xf32>
    // CHECK-SKIP: tf.StatefulPartitionedCall
    // CHECK-NOSKIP-NOT: tf.StatefulPartitionedCall
    func.return %0: tensor<1x2x2x3xf32>
  }

  func.func private @some_func(%arg0: tensor<1x2x2x3xf32>) -> tensor<1x2x2x3xf32> attributes {tf._noinline = true} {
    return %arg0 : tensor<1x2x2x3xf32>
  }
}
