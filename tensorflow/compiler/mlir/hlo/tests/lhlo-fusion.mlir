// RUN: mlir-hlo-opt --lhlo-fusion -split-input-file %s -o - | FileCheck %s

// CHECK-LABEL: @simple_kloop_fusion
// CHECK-SAME: (%[[ARG0:.*]]: memref<?x?xf32>, %[[ARG1:.*]]: memref<?x?xf32>, %[[ARG2:.*]]: memref<?x?xf32>, %[[ARG3:.*]]: memref<?x?xf32>) -> memref<?x?xf32>
func @simple_kloop_fusion(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>,
                          %arg2: memref<?x?xf32>, %arg3: memref<?x?xf32>) -> memref<?x?xf32> {
  // CHECK: "lmhlo.fusion"() ( {
  // CHECK: "lmhlo.abs"(%[[ARG0]], %[[ARG1]]) : (memref<?x?xf32>, memref<?x?xf32>) -> ()
  // CHECK: "lmhlo.add"(%[[ARG1]], %[[ARG2]], %[[ARG3]]) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
  // CHECK: }) : () -> ()
  // CHECK: return %[[ARG3]] : memref<?x?xf32>
  "lmhlo.abs"(%arg0, %arg1) : (memref<?x?xf32>, memref<?x?xf32>) -> ()
  "lmhlo.add"(%arg1, %arg2, %arg3) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
  return %arg3 : memref<?x?xf32>
}

// -----

// CHECK-LABEL: @simple_multi_output_kloop_fusion
// CHECK-SAME: (%[[ARG0:.*]]: memref<?x?xf32>, %[[ARG1:.*]]: memref<?x?xf32>, %[[ARG2:.*]]: memref<?x?xf32>, %[[ARG3:.*]]: memref<?x?xf32>) -> (memref<?x?xf32>, memref<?x?xf32>)
func @simple_multi_output_kloop_fusion(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>,
                          %arg2: memref<?x?xf32>, %arg3: memref<?x?xf32>) -> (memref<?x?xf32>, memref<?x?xf32>) {
  // CHECK: "lmhlo.fusion"() ( {
  // CHECK: "lmhlo.abs"(%[[ARG0]], %[[ARG1]]) : (memref<?x?xf32>, memref<?x?xf32>) -> ()
  // CHECK: "lmhlo.add"(%[[ARG1]], %[[ARG2]], %[[ARG3]]) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
  // CHECK: }) : () -> ()
  // CHECK: return %[[ARG1]], %[[ARG3]] : memref<?x?xf32>, memref<?x?xf32>
  "lmhlo.abs"(%arg0, %arg1) : (memref<?x?xf32>, memref<?x?xf32>) -> ()
  "lmhlo.add"(%arg1, %arg2, %arg3) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
  return %arg1, %arg3 : memref<?x?xf32>, memref<?x?xf32>
}

// -----

// CHECK-LABEL: @simple_multi_output_kloop_fusion_with_reorder
// CHECK-SAME: (%[[ARG0:.*]]: memref<?x?xf32>, %[[ARG1:.*]]: memref<?x?xf32>, %[[ARG2:.*]]: memref<?x?xf32>, %[[ARG3:.*]]: memref<?x?xf32>, %[[ARG4:.*]]: memref<2xindex>, %[[ARG5:.*]]: memref<?x?xf32>)
func @simple_multi_output_kloop_fusion_with_reorder(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>,
                          %arg2: memref<?x?xf32>, %arg3: memref<?x?xf32>,
                          %arg4: memref<2xindex>, %arg5:  memref<?x?xf32>) -> (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) {
  // CHECK: "lmhlo.fusion"() ( {
  // CHECK: "lmhlo.abs"(%[[ARG0]], %[[ARG1]]) : (memref<?x?xf32>, memref<?x?xf32>) -> ()
  // CHECK: "lmhlo.add"(%[[ARG1]], %[[ARG2]], %[[ARG3]]) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
  // CHECK: }) : () -> ()
  // CHECK: "lmhlo.dynamic_broadcast_in_dim"(%[[ARG1]], %[[ARG4]], %[[ARG5]])
  // CHECK: return %[[ARG1]], %[[ARG3]], %[[ARG5]] : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>
  "lmhlo.abs"(%arg0, %arg1) : (memref<?x?xf32>, memref<?x?xf32>) -> ()
  "lmhlo.dynamic_broadcast_in_dim"(%arg1, %arg4, %arg5) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (memref<?x?xf32>, memref<2xindex>, memref<?x?xf32>) -> ()
  "lmhlo.add"(%arg1, %arg2, %arg3) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
  return %arg1, %arg3, %arg5 : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>
}

// -----

// CHECK-LABEL: @same_num_elements_multi_output_kloop_fusion
// CHECK-SAME: (%[[ARG0:.*]]: memref<?x?xf32>, %[[ARG1:.*]]: memref<?x?xf32>, %[[ARG2:.*]]: memref<2xi64>, %[[ARG3:.*]]: memref<?x?x?xf32>, %[[ARG4:.*]]: memref<?x?x?xf32>, %[[ARG5:.*]]: memref<?x?x?xf32>)
func @same_num_elements_multi_output_kloop_fusion(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>,
                          %arg2: memref<2xi64>, %arg3: memref<?x?x?xf32>,
                          %arg4: memref<?x?x?xf32>, %arg5:  memref<?x?x?xf32>) -> (memref<?x?xf32>, memref<?x?x?xf32>) {
  // CHECK: "lmhlo.fusion"() ( {
  // CHECK: "lmhlo.abs"(%[[ARG0]], %[[ARG1]]) : (memref<?x?xf32>, memref<?x?xf32>) -> ()
  // CHECK: "lmhlo.dynamic_reshape"(%[[ARG1]], %[[ARG2]], %[[ARG3]])
  // CHECK: "lmhlo.add"(%[[ARG3]], %[[ARG4]], %[[ARG5]]) : (memref<?x?x?xf32>, memref<?x?x?xf32>, memref<?x?x?xf32>) -> ()
  // CHECK: }) : () -> ()
  // CHECK: return %[[ARG1]], %[[ARG5]] : memref<?x?xf32>, memref<?x?x?xf32>
  "lmhlo.abs"(%arg0, %arg1) : (memref<?x?xf32>, memref<?x?xf32>) -> ()
  "lmhlo.dynamic_reshape"(%arg1, %arg2, %arg3) : (memref<?x?xf32>, memref<2xi64>, memref<?x?x?xf32>) -> ()
  "lmhlo.add"(%arg3, %arg4, %arg5) : (memref<?x?x?xf32>, memref<?x?x?xf32>, memref<?x?x?xf32>) -> ()
  return %arg1, %arg5 : memref<?x?xf32>, memref<?x?x?xf32>
}

// -----

// CHECK-LABEL: @check_not_kloop_fusion
func @check_not_kloop_fusion(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>, %arg3: memref<?x?xf32>) -> (memref<?x?xf32>, memref<?x?xf32>) {
  // CHECK-NOT: "lmhlo.fusion"
  "lmhlo.add"(%arg0, %arg0, %arg1) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
  "lmhlo.subtract"(%arg2, %arg2, %arg3) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
  return %arg1, %arg3: memref<?x?xf32>, memref<?x?xf32>
}

// -----

// CHECK-LABEL: @kloop_fusion_with_dealloc
// CHECK-SAME: (%[[ARG0:.*]]: memref<?x?xf32>, %[[ARG1:.*]]: memref<?x?xf32>)
func @kloop_fusion_with_dealloc(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>) -> (memref<?x?xf32>, memref<?x?xf32>) {
  // CHECK: %[[TMP3:.*]] = memref.alloc
  // CHECK: %[[TMP5:.*]] = memref.alloc
  // CHECK: %[[TMP9:.*]] = memref.alloc
  // CHECK: %[[TMP13:.*]] = memref.alloc
  // CHECK: %[[TMP16:.*]] = memref.alloc
  // CHECK: "lmhlo.fusion"() ( {
  // CHECK: "lmhlo.add"(%[[ARG0]], %[[ARG1]], %[[TMP3]]) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
  // CHECK: "lmhlo.multiply"(%[[ARG0]], %[[ARG1]], %[[TMP5]]) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
  // CHECK: "lmhlo.abs"(%[[TMP3]], %[[TMP9]]) : (memref<?x?xf32>, memref<?x?xf32>) -> ()
  // CHECK: "lmhlo.abs"(%[[TMP5]], %[[TMP13]]) : (memref<?x?xf32>, memref<?x?xf32>) -> ()
  // CHECK: "lmhlo.multiply"(%[[TMP9]], %[[TMP13]], %[[TMP16]]) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
  // CHECK: }) : () -> ()
  // CHECK: memref.dealloc %[[TMP3]] : memref<?x?xf32>
  // CHECK: memref.dealloc %[[TMP5]] : memref<?x?xf32>
  // CHECK: memref.dealloc %[[TMP13]] : memref<?x?xf32>
  // CHECK: return %[[TMP9]], %[[TMP16]] : memref<?x?xf32>, memref<?x?xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = shape.shape_of %arg0 : memref<?x?xf32> -> tensor<2xindex>
  %1 = tensor.extract %0[%c0] : tensor<2xindex>
  %2 = tensor.extract %0[%c1] : tensor<2xindex>
  %3 = memref.alloc(%1, %2) : memref<?x?xf32>
  "lmhlo.add"(%arg0, %arg1, %3) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
  %4 = memref.alloc(%1, %2) : memref<?x?xf32>
  "lmhlo.multiply"(%arg0, %arg1, %4) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
  %5 = shape.shape_of %3 : memref<?x?xf32> -> tensor<2xindex>
  %6 = tensor.extract %5[%c0] : tensor<2xindex>
  %7 = tensor.extract %5[%c1] : tensor<2xindex>
  %8 = memref.alloc(%6, %7) : memref<?x?xf32>
  "lmhlo.abs"(%3, %8) : (memref<?x?xf32>, memref<?x?xf32>) -> ()
  memref.dealloc %3 : memref<?x?xf32>
  %9 = shape.shape_of %4 : memref<?x?xf32> -> tensor<2xindex>
  %10 = tensor.extract %9[%c0] : tensor<2xindex>
  %11 = tensor.extract %9[%c1] : tensor<2xindex>
  %12 = memref.alloc(%10, %11) : memref<?x?xf32>
  "lmhlo.abs"(%4, %12) : (memref<?x?xf32>, memref<?x?xf32>) -> ()
  memref.dealloc %4 : memref<?x?xf32>
  %13 = shape.shape_of %8 : memref<?x?xf32> -> tensor<2xindex>
  %14 = tensor.extract %13[%c0] : tensor<2xindex>
  %15 = tensor.extract %13[%c1] : tensor<2xindex>
  %16 = memref.alloc(%14, %15) : memref<?x?xf32>
  "lmhlo.multiply"(%8, %12, %16) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
  memref.dealloc %12 : memref<?x?xf32>
  return %8, %16 : memref<?x?xf32>, memref<?x?xf32>
}

// -----

// CHECK-LABEL: @simple_kinput
// CHECK-SAME: %[[ARG0:.*]]: memref<?x?xf32>, %[[ARG1:.*]]: memref<?x?xf32>, %[[ARG2:.*]]: memref<?xf32>, %[[ARG3:.*]]: memref<f32>
func @simple_kinput(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?xf32>, %init: memref<f32>) -> memref<?xf32> {
  // CHECK: "lmhlo.fusion"() ( {
  // CHECK: "lmhlo.abs"(%[[ARG0]], %[[ARG1]]) : (memref<?x?xf32>, memref<?x?xf32>) -> ()
  // CHECK: "lmhlo.reduce"(%[[ARG1]], %[[ARG3]], %[[ARG2]]) ( {
  // CHECK: }) : () -> ()
  // CHECK: return %[[ARG2]] : memref<?xf32>
  "lmhlo.abs"(%arg0, %arg1) : (memref<?x?xf32>, memref<?x?xf32>) -> ()
  "lmhlo.reduce"(%arg1, %init, %arg2) ( {
  ^bb0(%targ1: memref<f32>, %targ2: memref<f32>, %tresult: memref<f32>):
    "lmhlo.add"(%targ1, %targ2, %tresult) : (memref<f32>, memref<f32>, memref<f32>) -> ()
    "lmhlo.terminator"() : () -> ()
  } ) {dimensions = dense<[0]> : tensor<1xi64>} : (memref<?x?xf32>, memref<f32>, memref<?xf32>) -> ()
  return %arg2: memref<?xf32>
}

// -----

// CHECK-LABEL: @multi_output_kinput
// CHECK-SAME: %[[ARG0:.*]]: memref<?x?xf32>, %[[ARG1:.*]]: memref<?x?xf32>, %[[ARG2:.*]]: memref<?xf32>, %[[ARG3:.*]]: memref<f32>
func @multi_output_kinput(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?xf32>, %init: memref<f32>) -> (memref<?x?xf32>, memref<?xf32>) {
  // CHECK: "lmhlo.fusion"() ( {
  // CHECK: "lmhlo.abs"(%[[ARG0]], %[[ARG1]]) : (memref<?x?xf32>, memref<?x?xf32>) -> ()
  // CHECK: "lmhlo.reduce"(%[[ARG1]], %[[ARG3]], %[[ARG2]]) ( {
  // CHECK: }) : () -> ()
  // CHECK: return %[[ARG1]], %[[ARG2]] : memref<?x?xf32>, memref<?xf32>
  "lmhlo.abs"(%arg0, %arg1) : (memref<?x?xf32>, memref<?x?xf32>) -> ()
  "lmhlo.reduce"(%arg1, %init, %arg2) ( {
  ^bb0(%targ1: memref<f32>, %targ2: memref<f32>, %tresult: memref<f32>):
    "lmhlo.add"(%targ1, %targ2, %tresult) : (memref<f32>, memref<f32>, memref<f32>) -> ()
    "lmhlo.terminator"() : () -> ()
  } ) {dimensions = dense<[0]> : tensor<1xi64>} : (memref<?x?xf32>, memref<f32>, memref<?xf32>) -> ()
  return %arg1, %arg2: memref<?x?xf32>, memref<?xf32>
}

// -----

// CHECK-LABEL: @row_red_and_row_red_kinput
// CHECK-SAME: %[[ARG0:.*]]: memref<?x?xf32>, %[[ARG1:.*]]: memref<?x?xf32>, %[[ARG2:.*]]: memref<?x?xf32>, %[[ARG3:.*]]: memref<?xf32>, %[[ARG4:.*]]: memref<?xf32>, %[[ARG5:.*]]: memref<?x?xf32>, %[[ARG6:.*]]: memref<f32>
func @row_red_and_row_red_kinput(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>, %arg5: memref<?x?xf32>, %init: memref<f32>) -> (memref<?xf32>, memref<?xf32>) {
  // CHECK: "lmhlo.fusion"() ( {
  // CHECK: "lmhlo.add"(%[[ARG0]], %[[ARG1]], %[[ARG2]]) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
  // CHECK: "lmhlo.abs"(%[[ARG2]], %[[ARG5]]) : (memref<?x?xf32>, memref<?x?xf32>) -> ()
  // CHECK: "lmhlo.reduce"(%[[ARG5]], %[[ARG6]], %[[ARG3]]) ( {
  // CHECK: "lmhlo.reduce"(%[[ARG2]], %[[ARG6]], %[[ARG4]]) ( {
  // CHECK: }) : () -> ()
  // CHECK: return %[[ARG3]], %[[ARG4]] : memref<?xf32>, memref<?xf32>
  "lmhlo.add"(%arg0, %arg1, %arg2) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
  "lmhlo.abs"(%arg2, %arg5) : (memref<?x?xf32>, memref<?x?xf32>) -> ()
  "lmhlo.reduce"(%arg5, %init, %arg3) ( {
  ^bb0(%targ1: memref<f32>, %targ2: memref<f32>, %tresult: memref<f32>):
    "lmhlo.add"(%targ1, %targ2, %tresult) : (memref<f32>, memref<f32>, memref<f32>) -> ()
    "lmhlo.terminator"() : () -> ()
  } ) {dimensions = dense<[1]> : tensor<1xi64>} : (memref<?x?xf32>, memref<f32>, memref<?xf32>) -> ()
  "lmhlo.reduce"(%arg2, %init, %arg4) ( {
  ^bb0(%targ1: memref<f32>, %targ2: memref<f32>, %tresult: memref<f32>):
    "lmhlo.add"(%targ1, %targ2, %tresult) : (memref<f32>, memref<f32>, memref<f32>) -> ()
    "lmhlo.terminator"() : () -> ()
  } ) {dimensions = dense<[1]> : tensor<1xi64>} : (memref<?x?xf32>, memref<f32>, memref<?xf32>) -> ()
  return %arg3, %arg4: memref<?xf32>, memref<?xf32>
}

// -----

// CHECK-LABEL: @row_red_and_col_red_kinput
// CHECK-SAME: %[[ARG0:.*]]: memref<?x?xf32>, %[[ARG1:.*]]: memref<?x?xf32>, %[[ARG2:.*]]: memref<?x?xf32>, %[[ARG3:.*]]: memref<?xf32>, %[[ARG4:.*]]: memref<?xf32>, %[[ARG5:.*]]: memref<?x?xf32>, %[[ARG6:.*]]: memref<f32>
func @row_red_and_col_red_kinput(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>, %arg5: memref<?x?xf32>, %init: memref<f32>) -> (memref<?xf32>, memref<?xf32>) {
  // CHECK: "lmhlo.fusion"() ( {
  // CHECK: "lmhlo.add"(%[[ARG0]], %[[ARG1]], %[[ARG2]]) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
  // CHECK: "lmhlo.abs"(%[[ARG2]], %[[ARG5]]) : (memref<?x?xf32>, memref<?x?xf32>) -> ()
  // CHECK: "lmhlo.reduce"(%[[ARG5]], %[[ARG6]], %[[ARG3]]) ( {
  // CHECK: "lmhlo.reduce"(%[[ARG2]], %[[ARG6]], %[[ARG4]]) ( {
  // CHECK: }) : () -> ()
  // CHECK: return %[[ARG3]], %[[ARG4]] : memref<?xf32>, memref<?xf32>
  "lmhlo.add"(%arg0, %arg1, %arg2) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
  "lmhlo.abs"(%arg2, %arg5) : (memref<?x?xf32>, memref<?x?xf32>) -> ()
  "lmhlo.reduce"(%arg5, %init, %arg3) ( {
  ^bb0(%targ1: memref<f32>, %targ2: memref<f32>, %tresult: memref<f32>):
    "lmhlo.add"(%targ1, %targ2, %tresult) : (memref<f32>, memref<f32>, memref<f32>) -> ()
    "lmhlo.terminator"() : () -> ()
  } ) {dimensions = dense<[1]> : tensor<1xi64>} : (memref<?x?xf32>, memref<f32>, memref<?xf32>) -> ()
  "lmhlo.reduce"(%arg2, %init, %arg4) ( {
  ^bb0(%targ1: memref<f32>, %targ2: memref<f32>, %tresult: memref<f32>):
    "lmhlo.add"(%targ1, %targ2, %tresult) : (memref<f32>, memref<f32>, memref<f32>) -> ()
    "lmhlo.terminator"() : () -> ()
  } ) {dimensions = dense<[0]> : tensor<1xi64>} : (memref<?x?xf32>, memref<f32>, memref<?xf32>) -> ()
  return %arg3, %arg4: memref<?xf32>, memref<?xf32>
}

// -----

// CHECK-LABEL: @reduce_should_not_have_consumer_in_the_fusion
// CHECK-SAME: %[[ARG0:.*]]: memref<?x?xf32>, %[[ARG1:.*]]: memref<?x?xf32>
func @reduce_should_not_have_consumer_in_the_fusion(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>)
-> (memref<?x?xf32>, memref<?xf32>) {
  // CHECK: %[[TMP4:.*]] = memref.alloc
  // CHECK: %[[TMP7:.*]] = memref.alloc
  // CHECK: %[[TMP8:.*]] = memref.alloc
  // CHECK: %[[TMP9:.*]] = memref.alloc
  // CHECK: "lmhlo.fusion"() ( {
  // CHECK: "lmhlo.add"(%[[ARG0]], %[[ARG1]], %[[TMP4]]) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
  // CHECK: "lmhlo.subtract"(%[[ARG0]], %[[TMP4]], %[[TMP7]]) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
  // CHECK: "lmhlo.constant"(%[[TMP8]]) {value = dense<0.000000e+00> : tensor<f32>} : (memref<f32>) -> ()
  // CHECK: "lmhlo.reduce"(%[[TMP7]], %[[TMP8]], %[[TMP9]]) ( {
  // CHECK: }) : () -> ()
  // CHECK: memref.dealloc %[[TMP4]] : memref<?x?xf32>
  // CHECK: memref.dealloc %[[TMP8]] : memref<f32>
  // CHECK: %[[TMP12:.*]] = memref.alloc
  // CHECK: "lmhlo.add"(%[[TMP9]], %[[TMP9]], %[[TMP12]]) : (memref<?xf32>, memref<?xf32>, memref<?xf32>) -> ()
  // CHECK: memref.dealloc %[[TMP9]] : memref<?xf32>
  // CHECK: return %[[TMP7]], %[[TMP12]] : memref<?x?xf32>, memref<?xf32>
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = shape.shape_of %arg0 : memref<?x?xf32> -> tensor<2xindex>
  %1 = tensor.extract %0[%c0] : tensor<2xindex>
  %2 = tensor.extract %0[%c1] : tensor<2xindex>
  %3 = memref.alloc(%1, %2) : memref<?x?xf32>
  "lmhlo.add"(%arg0, %arg1, %3) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
  %4 = shape.shape_of %arg0 : memref<?x?xf32> -> tensor<2xindex>
  %5 = tensor.extract %4[%c0] : tensor<2xindex>
  %6 = tensor.extract %4[%c1] : tensor<2xindex>
  %7 = memref.alloc(%5, %6) : memref<?x?xf32>
  "lmhlo.subtract"(%arg0, %3, %7) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
  memref.dealloc %3 : memref<?x?xf32>
  %8 = memref.alloc() : memref<f32>
  "lmhlo.constant"(%8) {value = dense<0.000000e+00> : tensor<f32>} : (memref<f32>) -> ()
  %9 = memref.alloc(%5) : memref<?xf32>
  "lmhlo.reduce"(%7, %8, %9) ( {
  ^bb0(%arg2: memref<f32>, %arg3: memref<f32>, %arg4: memref<f32>):  // no predecessors
    "lmhlo.add"(%arg2, %arg3, %arg4) : (memref<f32>, memref<f32>, memref<f32>) -> ()
    "lmhlo.terminator"() : () -> ()
  }) {dimensions = dense<1> : tensor<1xi64>} : (memref<?x?xf32>, memref<f32>, memref<?xf32>) -> ()
  memref.dealloc %8 : memref<f32>
  %10 = shape.shape_of %9 : memref<?xf32> -> tensor<1xindex>
  %11 = tensor.extract %10[%c0] : tensor<1xindex>
  %12 = memref.alloc(%11) : memref<?xf32>
  "lmhlo.add"(%9, %9, %12) : (memref<?xf32>, memref<?xf32>, memref<?xf32>) -> ()
  memref.dealloc %9 : memref<?xf32>
  return %7, %12 : memref<?x?xf32>, memref<?xf32>
}

// -----

// CHECK-LABEL: @const_should_not_be_output
func @const_should_not_be_output(%arg0: memref<f32>) -> (memref<f32>, memref<f32>) {
  // CHECK-NOT: lmhlo.fusion
  %0 = memref.alloc() : memref<f32>
  "lmhlo.constant"(%0) {value = dense<1.000000e+00> : tensor<f32>} : (memref<f32>) -> ()
  %1 = memref.alloc() : memref<f32>
  "lmhlo.add"(%arg0, %0, %1) : (memref<f32>, memref<f32>, memref<f32>) -> ()
  return %0, %1 : memref<f32>, memref<f32>
}
