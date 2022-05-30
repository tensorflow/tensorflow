// RUN: tf-tfrt-opt -tfrt-print-stream -verify-diagnostics %s

module @rsqrt_m attributes { tfrt.compiled } {
  func.func @compute(%arg0: tensor<512xf32>) -> tensor<512xf32> {
    %0 = "tf.Rsqrt"(%arg0): (tensor<512xf32>) -> tensor<512xf32>
    func.return %0 : tensor<512xf32>
  }
}

module @add_m attributes { tfrt.compiled } {
  func.func @compute(%arg0: tensor<512x512xf32>) -> tensor<512x512xf32> {
    %0 = "tf.Rsqrt"(%arg0): (tensor<512x512xf32>) -> tensor<512x512xf32>
    func.return %0 : tensor<512x512xf32>
  }
}

module @fusion_m attributes { tfrt.compiled } {
  func.func @compute(%arg0: tensor<?x512xf32>) -> tensor<?x512xf32> {
    %0 = "tf.Rsqrt"(%arg0): (tensor<?x512xf32>) -> tensor<?x512xf32>
    %1 = "tf.Rsqrt"(%0): (tensor<?x512xf32>) -> tensor<?x512xf32>
    %2 = "tf.Rsqrt"(%1): (tensor<?x512xf32>) -> tensor<?x512xf32>
    %3 = "tf.Rsqrt"(%2): (tensor<?x512xf32>) -> tensor<?x512xf32>
    %4 = "tf.Rsqrt"(%3): (tensor<?x512xf32>) -> tensor<?x512xf32>
    func.return %4 : tensor<?x512xf32>
  }
}

module @dyn_m attributes {tfrt.compiled}  {
  func.func @compute(%arg0: tensor<?x?xf32>,
                %arg1: tensor<?x128xf32>,
                %arg2: tensor<?x?xf32>,
                %arg3: tensor<*xf32> {jitrt.constraint = "rank"})
                  -> tensor<?x128xf32> {
    %cst = "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>} : ()
      -> tensor<f32>
    %0 = "tf.Mul"(%arg0, %arg1) : (tensor<?x?xf32>, tensor<?x128xf32>)
      -> tensor<?x128xf32>
    %1 = "tf.Sub"(%cst, %arg0) : (tensor<f32>, tensor<?x?xf32>)
      -> tensor<?x?xf32>
    %2 = "tf.BiasAdd"(%arg2, %arg3) {data_format = "NHWC"}
      : (tensor<?x?xf32>, tensor<*xf32>) -> tensor<?x?xf32>
    %3 = "tf.Tanh"(%2) : (tensor<?x?xf32>) -> tensor<?x?xf32>
    %4 = "tf.Mul"(%1, %3) : (tensor<?x?xf32>, tensor<?x?xf32>)
      -> tensor<?x?xf32>
    %5 = "tf.AddV2"(%0, %4) : (tensor<?x128xf32>, tensor<?x?xf32>)
      -> tensor<?x128xf32>
    func.return %5 : tensor<?x128xf32>
  }
}

module @dyn_override_m attributes {tfrt.compiled,
                                   "tfrt.max-arg-size" = 1024 : i64}  {
  func.func @compute(%arg0: tensor<?x?xf32>,
                %arg1: tensor<?x128xf32>,
                %arg2: tensor<?x?xf32>,
                %arg3: tensor<*xf32> {jitrt.constraint = "rank"})
                  -> tensor<?x128xf32> {
    %cst = "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>} : ()
      -> tensor<f32>
    %0 = "tf.Mul"(%arg0, %arg1) : (tensor<?x?xf32>, tensor<?x128xf32>)
      -> tensor<?x128xf32>
    %1 = "tf.Sub"(%cst, %arg0) : (tensor<f32>, tensor<?x?xf32>)
      -> tensor<?x?xf32>
    %2 = "tf.BiasAdd"(%arg2, %arg3) {data_format = "NHWC"}
      : (tensor<?x?xf32>, tensor<*xf32>) -> tensor<?x?xf32>
    %3 = "tf.Tanh"(%2) : (tensor<?x?xf32>) -> tensor<?x?xf32>
    %4 = "tf.Mul"(%1, %3) : (tensor<?x?xf32>, tensor<?x?xf32>)
      -> tensor<?x?xf32>
    %5 = "tf.AddV2"(%0, %4) : (tensor<?x128xf32>, tensor<?x?xf32>)
      -> tensor<?x128xf32>
    func.return %5 : tensor<?x128xf32>
  }
}

// expected-remark@+1 {{stream id: 0, stream cost: 515, parent stream: -1}}
func.func @rsqrt(%arg0: !tfrt_fallback.tf_tensor) -> !tfrt_fallback.tf_tensor {
  // stream 0 cost = 1 (root) + 1 (%arg0) + 513 (cost @rsqrt_m)
  //               = 515
  // expected-remark@+1 {{stream id: 0, stream cost: 515, parent stream: -1}}
  %res = tf_jitrt.fallback.execute @rsqrt_m::@compute (%arg0)
           device("/device:CPU:0")
           :  (!tfrt_fallback.tf_tensor)
           -> (!tfrt_fallback.tf_tensor)

  // expected-remark@+1 {{stream id: 0, stream cost: 515, parent stream: -1}}
  tfrt.return %res : !tfrt_fallback.tf_tensor
}

// expected-remark@+1 {{stream id: 0, stream cost: 262147, parent stream: -1}}
func.func @add(%arg0: !tfrt_fallback.tf_tensor) -> !tfrt_fallback.tf_tensor {
  // stream 0 cost = 1 (root) + 1 (%arg0) + (1 + 512 * 512) (cost @add_m)
  //               = 262147
  // expected-remark@+1 {{stream id: 0, stream cost: 262147, parent stream: -1}}
  %res = tf_jitrt.fallback.execute @add_m::@compute (%arg0, %arg0)
           device("/device:CPU:0")
           :  (!tfrt_fallback.tf_tensor, !tfrt_fallback.tf_tensor)
           -> (!tfrt_fallback.tf_tensor)

  // expected-remark@+1 {{stream id: 0, stream cost: 262147, parent stream: -1}}
  tfrt.return %res : !tfrt_fallback.tf_tensor
}

// expected-remark@+1 {{stream id: 0, stream cost: 2567, parent stream: -1}}
func.func @fusion(%arg0: !tfrt_fallback.tf_tensor) -> !tfrt_fallback.tf_tensor {
  // stream 0 cost = 1 (root) + 1 (%arg0) + 513 * 5 (cost @fusion_m)
  //               = 1 + 1 + 2565 = 2567
  // expected-remark@+1 {{stream id: 0, stream cost: 2567, parent stream: -1}}
  %res = tf_jitrt.fallback.execute @fusion_m::@compute (%arg0)
           device("/device:CPU:0")
           :  (!tfrt_fallback.tf_tensor)
           -> (!tfrt_fallback.tf_tensor)

  // expected-remark@+1 {{stream id: 0, stream cost: 2567, parent stream: -1}}
  tfrt.return %res : !tfrt_fallback.tf_tensor
}

// expected-remark@+1 {{stream id: 0, stream cost: 401, parent stream: -1}}
func.func @dyn(%arg0: !tfrt_fallback.tf_tensor) -> !tfrt_fallback.tf_tensor {
  // stream 0 cost = 1 (root) + 1 (%arg0) +
  //                 (1 [Const] + 130 [Mul] + 3 [Sub] + 130 [BiasAdd] + 2 [Tanh]
  //                  + 3 [Mul] + 130 [AddV2]) (cost @dyn_m)
  //               = 2 + (9 + 390) = 401
  // expected-remark@+1 {{stream id: 0, stream cost: 401, parent stream: -1}}
  %res = tf_jitrt.fallback.execute @dyn_m::@compute (%arg0)
           device("/device:CPU:0")
           :  (!tfrt_fallback.tf_tensor)
           -> (!tfrt_fallback.tf_tensor)

  // expected-remark@+1 {{stream id: 0, stream cost: 401, parent stream: -1}}
  tfrt.return %res : !tfrt_fallback.tf_tensor
}

// expected-remark@+1 {{stream id: 0, stream cost: 1297, parent stream: -1}}
func.func @dyn_override(%arg0: !tfrt_fallback.tf_tensor)
  -> !tfrt_fallback.tf_tensor {
  // stream 0 cost = 1 (root) + 1 (%arg0) +
  //                 (1 [Const] + 130 [Mul] + 3 [Sub] + 1026 [BiasAdd]
  //                  + 2 [Tanh] + 3 [Mul] + 130 [AddV2]) (cost @dyn_override_m)
  //               = 2 + 1295 = 1297
  // expected-remark@+1 {{stream id: 0, stream cost: 1297, parent stream: -1}}
  %res = tf_jitrt.fallback.execute @dyn_override_m::@compute (%arg0)
           device("/device:CPU:0")
           :  (!tfrt_fallback.tf_tensor)
           -> (!tfrt_fallback.tf_tensor)

  // expected-remark@+1 {{stream id: 0, stream cost: 1297, parent stream: -1}}
  tfrt.return %res : !tfrt_fallback.tf_tensor
}

