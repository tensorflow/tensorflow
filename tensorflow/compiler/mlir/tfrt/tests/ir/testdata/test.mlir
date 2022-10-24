func.func @test(%ch: !tfrt.chain, %arg0: !corert.tensorhandle, %arg1_th: !corert.tensorhandle) {
  %cpu = corert.get_op_handler %ch "cpu"
  %0 = corert.executeop(%cpu) "tf.Relu"(%arg0) { T = f32 } : 1
  %arg1 = tfrt_fallback_async.corert_tensorhandle_to_fallback_tensor %arg1_th {_tfrt_cost = 1 : i64, device = "/CPU:0"} : (!corert.tensorhandle) -> (!tfrt_fallback.tf_tensor)
  %1 = tfrt_fallback_async.executeop key(0) cost(100) device("/CPU:0") "tf.Relu"(%arg1) { T = f32 } : 1
  tfrt.return
}
