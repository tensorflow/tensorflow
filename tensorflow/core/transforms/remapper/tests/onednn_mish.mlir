// RUN: tfg-transforms-opt --tfg-remapper=enable-onednn-patterns %s | FileCheck %s

// -----

// CHECK-LABEL: tfg.func @fusedmish_test
tfg.func @fusedmish_test() {
  // CHECK: %[[PLACEHOLDER:.*]], {{.*}} name("input_tensor")
  %Placeholder, %ctl = Placeholder device("/device:CPU:0") name("input_tensor") {dtype = f32, shape = #tf_type.shape<64x64>} : () -> (tensor<64x64xf32>)
  // CHECK: %[[SOFTPLUS:.*]], {{.*}} name("Softplus")
  %Softplus, %ctl_0 = Softplus(%Placeholder) device("/device:CPU:0") name("Softplus") {T = f32} : (tensor<64x64xf32>) -> (tensor<64x64xf32>)
  // CHECK: %[[TANH:.*]], {{.*}} name("Tanh")
  %Tanh, %ctl_1 = Tanh(%Softplus) device("/device:CPU:0") name("Tanh") {T = f32} : (tensor<64x64xf32>) -> (tensor<64x64xf32>)
  // CHECK: _MklFusedMish(%[[PLACEHOLDER:.*]]) {{.*}} name("Mul")
  %Mul, %ctl_2 = Mul(%Placeholder, %Tanh) device("/device:CPU:0") name("Mul") {T = f32} : (tensor<64x64xf32>, tensor<64x64xf32>) -> (tensor<64x64xf32>)
  return
}
