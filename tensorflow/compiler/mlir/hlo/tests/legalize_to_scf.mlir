// RUN: mlir-hlo-opt --mhlo-control-flow-to-scf %s | FileCheck %s

func @lt_loop(%arg0: tensor<4xf32>, %arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<4xf32>, %arg4: tensor<f32>, %arg5: tensor<f32>, %arg6: tensor<f32>, %arg7: tensor<f32>, %arg8: tensor<i32>) -> (tuple<tensor<i32>, tensor<i32>, tensor<i32>>) {
  %cst = constant dense<-1> : tensor<i32>
  %cst_0 = constant dense<1> : tensor<i32>
  %cst_1 = constant dense<0> : tensor<i32>
  %cst_2 = constant dense<1000> : tensor<i32>
  %0 = "mhlo.tuple"(%cst_1, %cst, %cst_2) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tuple<tensor<i32>, tensor<i32>, tensor<i32>>
  %1 = "mhlo.while"(%0) ( {
  ^bb0(%arg9: tuple<tensor<i32>, tensor<i32>, tensor<i32>>):  // no predecessors
    %2 = "mhlo.get_tuple_element"(%arg9) {index = 0 : i32} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>>) -> tensor<i32>
    %3 = "mhlo.get_tuple_element"(%arg9) {index = 2 : i32} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>>) -> tensor<i32>
    %4 = "mhlo.compare"(%2, %3) {comparison_direction = "LT"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "mhlo.return"(%4) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg9: tuple<tensor<i32>, tensor<i32>, tensor<i32>>):  // no predecessors
    %2 = "mhlo.get_tuple_element"(%arg9) {index = 0 : i32} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>>) -> tensor<i32>
    %3 = mhlo.add %2, %cst_0 : tensor<i32>
    %4 = "mhlo.get_tuple_element"(%arg9) {index = 1 : i32} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>>) -> tensor<i32>
    %5 = "mhlo.get_tuple_element"(%arg9) {index = 2 : i32} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>>) -> tensor<i32>
    %6 = "mhlo.tuple"(%3, %4, %5) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tuple<tensor<i32>, tensor<i32>, tensor<i32>>
    "mhlo.return"(%6) : (tuple<tensor<i32>, tensor<i32>, tensor<i32>>) -> ()
  }) : (tuple<tensor<i32>, tensor<i32>, tensor<i32>>) -> tuple<tensor<i32>, tensor<i32>, tensor<i32>>
  return %1 : tuple<tensor<i32>, tensor<i32>, tensor<i32>>
}

// CHECK-LABEL:   func @lt_loop(
// CHECK:  %[[VAL_9:.*]] = constant dense<-1> : tensor<i32>
// CHECK:  %[[VAL_10:.*]] = constant dense<1> : tensor<i32>
// CHECK:  %[[VAL_11:.*]] = constant dense<0> : tensor<i32>
// CHECK:  %[[VAL_12:.*]] = constant dense<1000> : tensor<i32>
// CHECK:  %[[VAL_14:.*]] = index_cast %[[VAL_11]] : tensor<i32> to tensor<index>
// CHECK:  %[[VAL_15:.*]] = tensor.extract %[[VAL_14]][] : tensor<index>
// CHECK:  %[[VAL_16:.*]] = index_cast %[[VAL_12]] : tensor<i32> to tensor<index>
// CHECK:  %[[VAL_17:.*]] = tensor.extract %[[VAL_16]][] : tensor<index>
// CHECK:  %[[VAL_18:.*]] = index_cast %[[VAL_10]] : tensor<i32> to tensor<index>
// CHECK:  %[[VAL_19:.*]] = tensor.extract %[[VAL_18]][] : tensor<index>
// CHECK:  scf.for %[[VAL_21:.*]] = %[[VAL_15]] to %[[VAL_17]] step %[[VAL_19]] iter_args(%[[VAL_22:.*]] = %[[VAL_9]], %[[VAL_23:.*]] = %[[VAL_12]])
