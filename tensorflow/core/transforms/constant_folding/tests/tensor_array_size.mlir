// RUN: tfg-transforms-opt -tfg-constant-folding %s | FileCheck %s

module {
  tfg.graph #tf_type.version<producer = 1140, min_consumer = 0> {
    %Const, %ctl = Const name("size") {dtype = i32, value = dense<5> : tensor<i32>} : () -> (tensor<*xi32>)
    %Placeholder, %ctl_0 = Placeholder name("placeholder") {dtype = !tf_type.resource, shape = #tf_type.shape<2>} : () -> (tensor<*x!tf_type.resource>)
    %Const_1, %ctl_2 = Const name("foo") {dtype = f32, value = dense<5.000000e+00> : tensor<f32>} : () -> (tensor<*xf32>)
    // CHECK: TensorArrayV3{{.*}} name("dynamic")
    %TensorArrayV3:2, %ctl_3 = TensorArrayV3(%Const) name("dynamic") {clear_after_read = true, dtype = f32, dynamic_size = true, element_shape = #tf_type.shape<*>, identical_element_shapes = false, tensor_array_name = ""} : (tensor<*xi32>) -> (tensor<*x!tf_type.resource>, tensor<*xf32>)
    // CHECK: %[[TENSORARRAY:.*]], %[[CTRL:.*]] = TensorArrayV3{{.*}} name("static")
    %TensorArrayV3_4:2, %ctl_5 = TensorArrayV3(%Const) name("static") {clear_after_read = true, dtype = f32, dynamic_size = false, element_shape = #tf_type.shape<*>, identical_element_shapes = false, tensor_array_name = ""} : (tensor<*xi32>) -> (tensor<*x!tf_type.resource>, tensor<*xf32>)
    // CHECK: TensorArraySizeV3{{.*}} name("dynamic_sz")
    %TensorArraySizeV3, %ctl_6 = TensorArraySizeV3(%TensorArrayV3#0, %TensorArrayV3#1) name("dynamic_sz") : (tensor<*x!tf_type.resource>, tensor<*xf32>) -> (tensor<*xi32>)
    // CHECK: Const [%[[CTRL]], %[[CTRL]]] name("static_sz/const_folded") {{.*}} -> (tensor<i32>)
    %TensorArraySizeV3_7, %ctl_8 = TensorArraySizeV3(%TensorArrayV3_4#0, %TensorArrayV3_4#1) name("static_sz") : (tensor<*x!tf_type.resource>, tensor<*xf32>) -> (tensor<*xi32>)
    // CHECK: TensorArraySizeV3{{.*}} name("placeholder_sz")
    %TensorArraySizeV3_9, %ctl_10 = TensorArraySizeV3(%Placeholder, %Const_1) name("placeholder_sz") : (tensor<*x!tf_type.resource>, tensor<*xf32>) -> (tensor<*xi32>)
  }
}
