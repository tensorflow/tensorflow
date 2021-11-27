// RUN: tfg-translate -mlir-to-graphdef %s | FileCheck %s

tfg.graph #tf_type.version<producer = 0, min_consumer = 0> {
// CHECK: name: "float"
// CHECK: float_val: -0
  %Const, %ctl = Const name("float") {dtype = f32, value = dense<-0.000000e+00> : tensor<f32>} : () -> (tensor<f32>)
// CHECK: name: "half"
// CHECK: half_val: 32768
  %Const_0, %ctl_1 = Const name("half") {dtype = f16, value = dense<-0.000000e+00> : tensor<f16>} : () -> (tensor<f16>)
// CHECK: name: "complex"
// CHECK: scomplex_val: -0
// CHECK: scomplex_val: -0
  %Const_2, %ctl_3 = Const name("complex") {dtype = complex<f32>, value = dense<(-0.000000e+00,-0.000000e+00)> : tensor<complex<f32>>} : () -> (tensor<complex<f32>>)
}
