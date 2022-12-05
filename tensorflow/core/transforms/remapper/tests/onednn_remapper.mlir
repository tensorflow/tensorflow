// RUN: tfg-transforms-opt -tfg-remapper=enable-onednn-patterns %s | FileCheck %s
// RUN: tfg-transforms-opt -tfg-remapper=verify-pdll-patterns-only %s | FileCheck %s

module {
  // CHECK: %[[ARG0:.*]]: tensor<64x64xf32> {tfg.name = "arg0"}
  tfg.func @_mlir_graph(%arg0: tensor<64x64xf32> {tfg.name = "arg0"}) -> (tensor<*xf32> {tfg.name = "mul4:0"})
  {
    // CHECK: %[[PLACEHOLDER:.*]], {{.*}} name("input")
    %Placeholder_0, %ctl_1 = Placeholder device("/device:CPU:0") name("input") {dtype = f32, shape = #tf_type.shape<64x64>} : () -> (tensor<64x64xf32>)
    %Sigmoid, %ctl_2 = Sigmoid(%Placeholder_0) device("/device:CPU:0") name("sigmoid1") {T = f32} : (tensor<64x64xf32>) -> (tensor<*xf32>)
    %Sigmoid_3, %ctl_4 = Sigmoid(%Placeholder_0) device("/device:CPU:0") name("sigmoid2") {T = f32} : (tensor<64x64xf32>) -> (tensor<*xf32>)
    // CHECK: %[[SIGMOID_3:.*]], {{.*}} name("sigmoid3_1")
    %Sigmoid_5, %ctl_6 = Sigmoid(%Placeholder_0) device("/device:CPU:0") name("sigmoid3_1") {T = f32} : (tensor<64x64xf32>) -> (tensor<*xf32>)
    %Sigmoid_7, %ctl_8 = Sigmoid(%Sigmoid_5) device("/device:CPU:0") name("sigmoid3_2") {T = f32} : (tensor<*xf32>) -> (tensor<*xf32>)
    // CHECK: %[[SIGMOID_9:.*]], {{.*}} name("sigmoid4_1")
    %Sigmoid_9, %ctl_10 = Sigmoid(%Placeholder_0) device("/device:CPU:0") name("sigmoid4_1") {T = f32} : (tensor<64x64xf32>) -> (tensor<*xf32>)
    // CHECK: %[[SIGMOID_11:.*]], {{.*}} name("sigmoid4_2")
    %Sigmoid_11, %ctl_12 = Sigmoid(%Sigmoid_9) device("/device:CPU:0") name("sigmoid4_2") {T = f32} : (tensor<*xf32>) -> (tensor<*xf32>)
    // CHECK: _MklSwish(%[[PLACEHOLDER]]) {{.*}} name("mul1")
    %Mul, %ctl_13 = Mul(%Placeholder_0, %Sigmoid) [%ctl_12, %ctl_6] device("/device:CPU:0") name("mul1") {T = f32} : (tensor<64x64xf32>, tensor<*xf32>) -> (tensor<*xf32>)
    // CHECK: _MklSwish(%[[PLACEHOLDER]]) {{.*}} name("mul2")
    %Mul_14, %ctl_15 = Mul(%Sigmoid_3, %Placeholder_0) device("/device:CPU:0") name("mul2") {T = f32} : (tensor<*xf32>, tensor<64x64xf32>) -> (tensor<*xf32>)
    // CHECK: _MklSwish(%[[SIGMOID_3]]) {{.*}} name("mul3")
    %Mul_16, %ctl_17 = Mul(%Sigmoid_5, %Sigmoid_7) device("/device:CPU:0") name("mul3") {T = f32} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
    // CHECK: _MklSwish(%[[SIGMOID_9]]) {{.*}} name("mul4")
    %Mul_18, %ctl_19 = Mul(%Sigmoid_11, %Sigmoid_9) device("/device:CPU:0") name("mul4") {T = f32} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
    // CHECK: Mul(%[[SIGMOID_11]], %[[SIGMOID_9]]) {{.*}} name("mul5")
    %Mul_19, %ctl_20 = Mul(%Sigmoid_11, %Sigmoid_9) device("/device:GPU:0") name("mul5") {T = f32} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
    // CHECK: Mul(%[[ARG0]], %[[ARG0]]) {{.*}} name("mul5")
    %Mul_21, %ctl_22 = Mul(%arg0, %arg0) device("/device:CPU:0") name("mul5") {T = f32} : (tensor<64x64xf32>, tensor<64x64xf32>) -> (tensor<*xf32>)
    return(%Mul_18) : tensor<*xf32>
  }
}
