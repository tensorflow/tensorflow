// RUN: tf-opt -allow-unregistered-dialect -tfl-load-recipe %s | FileCheck %s

// CHECK-LABEL: testLstm
func @testLstm(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>, %arg2: tensor<*xf32>, %arg3: tensor<*xf32>, %arg4: tensor<*xf32>, %arg5: tensor<*xf32>, %arg6: tensor<*xf32>, %arg7: tensor<*xf32>, %arg8: tensor<*xf32>, %arg9: tensor<*xf32>, %arg10: tensor<*xf32>, %arg11: tensor<*xf32>, %arg12: tensor<*xf32>, %arg13: tensor<*xf32>, %arg14: tensor<*xf32>, %arg15: tensor<*xf32>, %arg16: tensor<*xf32>, %arg17: tensor<*xf32>, %arg18: tensor<*xf32>, %arg19: tensor<*xf32>, %arg20: tensor<*xf32>, %arg21: tensor<*xf32>, %arg22: tensor<*xf32>, %arg23: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "tfl.lstm"(%arg0, // input
    %arg1, %arg2, %arg3, %arg4, // weights
    %arg5, %arg6, %arg7, %arg8, // recurrent weights
    %arg9, %arg10, %arg11, // cell weights
    %arg12, %arg13, %arg14, %arg15, // bias
    %arg16, %arg17, // projection weight and bias
    %arg18, %arg19, // stateful
    %arg20, %arg21, %arg22, %arg23 // layer norm coefficients
    ) ({}) {fused_activation_function = "NONE", kernel_type = "FULL"} : (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>

// CHECK-NEXT:  "tfl.lstm"
// CHECK-NEXT:  %[[cst:.*]] = constant unit

// input gate
// CHECK-NEXT:  %[[in1:.*]] = "tfl.fully_connected"(%arg0, %arg1, %[[cst]])
// CHECK-SAME:    -> tensor<*x!quant.any<i16:f32>>
// CHECK-NEXT:  %[[in2:.*]] = "tfl.fully_connected"(%arg18, %arg5, %[[cst]])
// CHECK-SAME:    -> tensor<*x!quant.any<i16:f32>>
// CHECK-NEXT:  %[[in3:.*]] = "tfl.mul"(%arg19, %arg9)
// CHECK-SAME:    -> tensor<*x!quant.any<i16:f32>>
// CHECK-NEXT:  %[[in4:.*]] = "tfl.add_n"(%[[in1]], %[[in2]], %[[in3]])
// CHECK-SAME:    -> tensor<*x!quant.any<i16:f32>>
// CHECK-NEXT:  %[[in5:.*]] = "tfl.l2_normalization"(%[[in4]])
// CHECK-SAME:    -> tensor<*x!quant.any<i16:f32>>
// CHECK-NEXT:  %[[in6:.*]] = tfl.add %[[in4]], %[[in5]]
// CHECK-SAME:    tensor<*x!quant.any<i16:f32>>
// CHECK-NEXT:  %[[in7:.*]] = "tfl.fully_connected"(%[[in6]], %arg20, %arg12)
// CHECK-SAME:    -> tensor<*x!quant.any<i16:f32>>
// CHECK-NEXT:  %[[in8:.*]] = "tfl.logistic"(%[[in7]])
// CHECK-SAME:    -> tensor<*x!quant.any<i16:f32>>

// forget gate
// CHECK-NEXT:  %[[fo1:.*]] = "tfl.fully_connected"(%arg0, %arg2, %[[cst]])
// CHECK-SAME:    tensor<*x!quant.any<i16:f32>>
// CHECK-NEXT:  %[[fo2:.*]] = "tfl.fully_connected"(%arg18, %arg6, %[[cst]])
// CHECK-SAME:    -> tensor<*x!quant.any<i16:f32>>
// CHECK-NEXT:  %[[fo3:.*]] = "tfl.mul"(%arg19, %arg10)
// CHECK-SAME:    -> tensor<*x!quant.any<i16:f32>>
// CHECK-NEXT:  %[[fo4:.*]] = "tfl.add_n"(%[[fo1]], %[[fo2]], %[[fo3]])
// CHECK-SAME:    -> tensor<*x!quant.any<i16:f32>>
// CHECK-NEXT:  %[[fo5:.*]] = "tfl.l2_normalization"(%[[fo4]])
// CHECK-SAME:    -> tensor<*x!quant.any<i16:f32>>
// CHECK-NEXT:  %[[fo6:.*]] = tfl.add %[[fo4]], %[[fo5]]
// CHECK-SAME:    tensor<*x!quant.any<i16:f32>>
// CHECK-NEXT:  %[[fo7:.*]] = "tfl.fully_connected"(%[[fo6]], %arg21, %arg13)
// CHECK-SAME:    -> tensor<*x!quant.any<i16:f32>>
// CHECK-NEXT:  %[[fo8:.*]] = "tfl.logistic"(%[[fo7]])
// CHECK-SAME:    -> tensor<*x!quant.any<i16:f32>>

// cell gate
// CHECK-NEXT:  %[[ce1:.*]] = "tfl.fully_connected"(%arg0, %arg3, %[[cst]])
// CHECK-SAME:    -> tensor<*x!quant.any<i16:f32>>
// CHECK-NEXT:  %[[ce2:.*]] = "tfl.fully_connected"(%arg18, %arg7, %[[cst]])
// CHECK-SAME:    -> tensor<*x!quant.any<i16:f32>>
// CHECK-NEXT:  %[[ce3:.*]] = "tfl.add_n"(%[[ce1]], %[[ce2]])
// CHECK-SAME:    -> tensor<*x!quant.any<i16:f32>>
// CHECK-NEXT:  %[[ce4:.*]] = "tfl.l2_normalization"(%[[ce3]])
// CHECK-SAME:    -> tensor<*x!quant.any<i16:f32>>
// CHECK-NEXT:  %[[ce5:.*]] = tfl.add %[[ce3]], %[[ce4]]
// CHECK-SAME:    tensor<*x!quant.any<i16:f32>>
// CHECK-NEXT:  %[[ce6:.*]] = "tfl.fully_connected"(%[[ce5]], %arg22, %arg14)
// CHECK-SAME:    -> tensor<*x!quant.any<i16:f32>>
// CHECK-NEXT:  %[[ce7:.*]] = "tfl.tanh"(%[[ce6]])
// CHECK-SAME:    -> tensor<*x!quant.any<i16:f32>>

// CHECK-NEXT:  %[[ac1:.*]] = "tfl.mul"(%[[fo8]], %arg19)
// CHECK-SAME:    -> tensor<*x!quant.any<i16:f32>>
// CHECK-NEXT:  %[[ac2:.*]] = tfl.mul %[[in8]], %[[ce7]]
// CHECK-SAME:    tensor<*x!quant.any<i16:f32>>
// CHECK-NEXT:  %[[ac3:.*]] = tfl.add %[[ac1]], %[[ac2]]
// CHECK-SAME:    tensor<*x!quant.any<i16:f32>>

// output gate
// CHECK-NEXT:  %[[ou1:.*]] = "tfl.fully_connected"(%arg0, %arg4, %[[cst]])
// CHECK-SAME:    -> tensor<*x!quant.any<i16:f32>>
// CHECK-NEXT:  %[[ou2:.*]] = "tfl.fully_connected"(%arg18, %arg8, %[[cst]])
// CHECK-SAME:    -> tensor<*x!quant.any<i16:f32>>
// CHECK-NEXT:  %[[ou3:.*]] = "tfl.mul"(%[[ac3]], %arg11)
// CHECK-SAME:    -> tensor<*x!quant.any<i16:f32>>
// CHECK-NEXT:  %[[ou4:.*]] = "tfl.add_n"(%[[ou1]], %[[ou2]], %[[ou3]])
// CHECK-SAME:    -> tensor<*x!quant.any<i16:f32>>
// CHECK-NEXT:  %[[ou5:.*]] = "tfl.l2_normalization"(%[[ou4]])
// CHECK-SAME:    -> tensor<*x!quant.any<i16:f32>>
// CHECK-NEXT:  %[[ou6:.*]] = tfl.add %[[ou4]], %[[ou5]]
// CHECK-SAME:    tensor<*x!quant.any<i16:f32>>
// CHECK-NEXT:  %[[ou7:.*]] = "tfl.fully_connected"(%[[ou6]], %arg23, %arg15)
// CHECK-SAME:    -> tensor<*x!quant.any<i16:f32>>
// CHECK-NEXT:  %[[ou8:.*]] = "tfl.logistic"(%[[ou7]])
// CHECK-SAME:    -> tensor<*x!quant.any<i16:f32>>

// output activation
// CHECK-NEXT:  %[[ac4:.*]] = "tfl.tanh"(%[[ac3]])
// CHECK-SAME:    -> tensor<*x!quant.any<i16:f32>>
// CHECK-NEXT:  %[[ac5:.*]] = tfl.mul %[[ac4]], %[[ou8]]
// CHECK-SAME:    tensor<*x!quant.any<i16:f32>>
// CHECK-NEXT:  %[[ac6:.*]] = "tfl.fully_connected"(%[[ac5]], %arg16, %arg17)
// CHECK-SAME:    (tensor<*x!quant.any<i16:f32>>, tensor<*xf32>, tensor<*xf32>) -> tensor<*x!quant.any<i8:f32>>
// CHECK-NEXT:  %[[ac7:.*]] = "tf_quant.pseudo_return"(%[[ac6]]) : (tensor<*x!quant.any<i8:f32>>) -> tensor<*x!quant.any<i8:f32>>
// CHECK-NEXT:  })
// CHECK-NEXT:  return

  return %0 : tensor<*xf32>
}
