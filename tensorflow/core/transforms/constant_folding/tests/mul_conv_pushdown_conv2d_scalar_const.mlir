// RUN: tfg-transforms-opt -tfg-constant-folding %s | FileCheck %s

module {
  tfg.func @test() {
    %Const, %ctl = Const name("filter") {dtype = f32, value = dense<[[[[0.000000e+00, 1.000000e+00, 1.41421354, 1.73205078, 2.000000e+00], [2.23606801, 2.44948983, 2.64575124, 2.82842708, 3.000000e+00], [3.1622777, 3.31662488, 3.46410155, 3.60555124, 3.7416575]], [[3.87298346, 4.000000e+00, 4.12310553, 4.2426405, 4.35889912], [4.47213602, 4.5825758, 4.69041586, 4.79583168, 4.89897966], [5.000000e+00, 5.09901953, 5.19615221, 5.29150248, 5.38516474]]], [[[5.47722578, 5.56776428, 5.65685415, 5.74456263, 5.83095169], [5.916080e+00, 6.000000e+00, 6.08276271, 6.16441393, 6.24499797], [6.32455539, 6.40312433, 6.48074054, 6.55743837, 6.63324976]], [[6.70820379, 6.782330e+00, 6.85565471, 6.92820311, 7.000000e+00], [7.07106781, 7.14142847, 7.21110248, 7.280110e+00, 7.34846926], [7.41619825, 7.48331499, 7.54983425, 7.6157732, 7.68114566]]]]> : tensor<2x2x3x5xf32>} : () -> (tensor<2x2x3x5xf32>)
    // CHECK: %[[PLACEHOLDER:.*]], {{.*}} = {{.*}} name("x")
    %Placeholder, %ctl_0 = Placeholder name("x") {dtype = f32, shape = #tf_type.shape<4x10x10x3>} : () -> (tensor<4x10x10x3xf32>)
    %Conv2D, %ctl_1 = Conv2D(%Placeholder, %Const) name("conv") {T = f32, data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true} : (tensor<4x10x10x3xf32>, tensor<2x2x3x5xf32>) -> (tensor<4x9x9x5xf32>)
    %Const_2, %ctl_3 = Const name("c") {dtype = f32, value = dense<3.000000e+00> : tensor<f32>} : () -> (tensor<f32>)
    // CHECK-DAG: Conv2D(%[[PLACEHOLDER]], %[[CONST:.*]]) name("mul")
    // CHECK: %[[CONST]], {{.*}} = Const {{.*}} name("conv/merged_input")
    %Mul, %ctl_4 = Mul(%Const_2, %Conv2D) name("mul") {T = f32} : (tensor<f32>, tensor<4x9x9x5xf32>) -> (tensor<4x9x9x5xf32>)
    return
  }
}
