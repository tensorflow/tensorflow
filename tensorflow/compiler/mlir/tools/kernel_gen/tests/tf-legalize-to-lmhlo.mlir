// RUN: tf-opt %s --xla-legalize-tf='legalize-chlo=false' | \
// RUN: mlir-hlo-opt --mhlo-rank-specialization-cluster \
// RUN:   --mhlo-rank-specialization-to-scf --chlo-legalize-to-hlo \
// RUN:   --hlo-legalize-to-linalg --computeop-and-func-bufferize | \
// RUN: kernel-gen-opt --shape-to-descriptors \
// RUN:   --canonicalize --kernelgen-final-bufferize

func.func @acos(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "tf.Acos"(%arg0) { } : (tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

func.func @tan(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "tf.Tan"(%arg0) { } : (tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

func.func @tanh(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "tf.Tanh"(%arg0) { } : (tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

func.func @sin(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "tf.Sin"(%arg0) { } : (tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

func.func @sinh(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "tf.Sinh"(%arg0) { } : (tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

func.func @erf(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "tf.Erf"(%arg0) { } : (tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

