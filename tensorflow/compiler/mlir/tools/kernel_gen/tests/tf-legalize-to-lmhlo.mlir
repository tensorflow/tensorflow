// RUN: tf-opt %s --xla-legalize-tf='legalize-chlo=false' | \
// RUN: mlir-hlo-opt --mhlo-rank-specialization-cluster \
// RUN:   --mhlo-rank-specialization-to-scf --chlo-legalize-to-hlo \
// RUN:   --hlo-legalize-to-linalg | \
// RUN: kernel-gen-opt --computeop-and-func-bufferize --shape-to-descriptors \
// RUN:   --canonicalize --final-bufferize

func @acos(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "tf.Acos"(%arg0) { } : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

func @tan(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "tf.Tan"(%arg0) { } : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

func @tanh(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "tf.Tanh"(%arg0) { } : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

func @sin(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "tf.Sin"(%arg0) { } : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

func @sinh(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "tf.Sinh"(%arg0) { } : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

func @erf(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "tf.Erf"(%arg0) { } : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

