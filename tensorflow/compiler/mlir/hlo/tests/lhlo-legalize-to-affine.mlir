// RUN: mlir-hlo-opt -lhlo-legalize-to-affine %s -o - | FileCheck %s

// Smoke test.
// CHECK-LABEL: func @min_op
func @min_op(%lhs: memref<4x3x2x1xf32>, %rhs: memref<4x3x2x1xf32>,
             %result: memref<4x3x2x1xf32>) -> () {
  // CHECK-NEXT: %[[NAN:.*]] = constant 0x7FC00000 : f32
  // CHECK-NEXT: affine.for %[[I:.*]] = 0 to 4 {
  // CHECK-NEXT:   affine.for %[[J:.*]] = 0 to 3 {
  // CHECK-NEXT:     affine.for %[[K:.*]] = 0 to 2 {
  // CHECK-NEXT:       affine.for %[[L:.*]] = 0 to 1 {
  // CHECK-NEXT:         %[[LHS:.*]] = affine.load %{{.*}}[%[[I]], %[[J]], %[[K]], %[[L]]] : memref<4x3x2x1xf32>
  // CHECK-NEXT:         %[[RHS:.*]] = affine.load %{{.*}}[%[[I]], %[[J]], %[[K]], %[[L]]] : memref<4x3x2x1xf32>
  // CHECK-NEXT:         %[[MIN_PREDICATE:.*]] = cmpf olt, %[[LHS]], %[[RHS]] : f32
  // CHECK-NEXT:         %[[MIN:.*]] = select %[[MIN_PREDICATE]], %[[LHS]], %[[RHS]] : f32
  // CHECK-NEXT:         %[[ISNAN:.*]] = cmpf uno, %[[LHS]], %[[RHS]] : f32
  // CHECK-NEXT:         %[[MIN_NONAN:.*]] = select %[[ISNAN]], %[[NAN]], %[[MIN]] : f32
  // CHECK-NEXT:         affine.store %[[MIN_NONAN]], %{{.*}}[%[[I]], %[[J]], %[[K]], %[[L]]] : memref<4x3x2x1xf32>
  // CHECK:      return
  "lmhlo.minimum"(%lhs, %rhs, %result) {name = "min.1"} :
      (memref<4x3x2x1xf32>, memref<4x3x2x1xf32>, memref<4x3x2x1xf32>) -> ()
  return
}

// Add tests.
// CHECK-LABEL: func @float_add_op
func @float_add_op(%lhs: memref<7xf32>, %rhs: memref<7xf32>,
                   %result: memref<7xf32>) -> () {
  // CHECK: addf %{{.*}}, %{{.*}} : f32
  "lmhlo.add"(%lhs, %rhs, %result) {name = "add.1"}
      : (memref<7xf32>, memref<7xf32>, memref<7xf32>) -> ()
  return
}
// CHECK-LABEL: func @int_add_op
func @int_add_op(%lhs: memref<7xi32>, %rhs: memref<7xi32>,
                 %result: memref<7xi32>) -> () {
  // CHECK: addi %{{.*}}, %{{.*}} : i32
  "lmhlo.add"(%lhs, %rhs, %result) {name = "add.1"}
      : (memref<7xi32>, memref<7xi32>, memref<7xi32>) -> ()
  return
}

// And test.
// CHECK-LABEL: func @int_and_op
func @int_and_op(%lhs: memref<7xi32>, %rhs: memref<7xi32>,
                 %result: memref<7xi32>) -> () {
  // CHECK: and %{{.*}}, %{{.*}} : i32
  "lmhlo.and"(%lhs, %rhs, %result) {name = "and.1"}
      : (memref<7xi32>, memref<7xi32>, memref<7xi32>) -> ()
  return
}

// Div tests.
// CHECK-LABEL: func @float_div_op
func @float_div_op(%lhs: memref<7xf32>, %rhs: memref<7xf32>,
                   %result: memref<7xf32>) -> () {
  // CHECK: divf %{{.*}}, %{{.*}} : f32
  "lmhlo.divide"(%lhs, %rhs, %result) {name = "div.1"}
      : (memref<7xf32>, memref<7xf32>, memref<7xf32>) -> ()
  return
}
// CHECK-LABEL: func @int_div_op
func @int_div_op(%lhs: memref<7xi32>, %rhs: memref<7xi32>,
                 %result: memref<7xi32>) -> () {
  // CHECK: divi_signed %{{.*}}, %{{.*}} : i32
  "lmhlo.divide"(%lhs, %rhs, %result) {name = "div.1"}
      : (memref<7xi32>, memref<7xi32>, memref<7xi32>) -> ()
  return
}

// Max tests.
// CHECK-LABEL: func @float_max_op
func @float_max_op(%lhs: memref<7xf32>, %rhs: memref<7xf32>,
                   %result: memref<7xf32>) -> () {
  // CHECK: %[[NAN:.*]] = constant 0x7FC00000 : f32
  // CHECK: %[[CMP:.*]] = cmpf ogt, %[[LHS_IN:.*]], %[[RHS_IN:.*]] : f32
  // CHECK: %[[MIN:.*]] = select %[[CMP]], %[[LHS_IN]], %[[RHS_IN]] : f32
  // CHECK: %[[ISNAN:.*]] = cmpf uno, %[[LHS_IN]], %[[RHS_IN]] : f32
  // CHECK: select %[[ISNAN]], %[[NAN]], %[[MIN]] : f32
  "lmhlo.maximum"(%lhs, %rhs, %result) {name = "max.1"}
      : (memref<7xf32>, memref<7xf32>, memref<7xf32>) -> ()
  return
}

// CHECK-LABEL: func @int_max_op
func @int_max_op(%lhs: memref<7xi32>, %rhs: memref<7xi32>,
                 %result: memref<7xi32>) -> () {
  // CHECK: %[[CHECK:.*]] = cmpi sgt, %[[ONE:.*]], %[[TWO:.*]] : i32
  // CHECK: select %[[CHECK]], %[[ONE]], %[[TWO]] : i32
  "lmhlo.maximum"(%lhs, %rhs, %result) {name = "max.1"}
      : (memref<7xi32>, memref<7xi32>, memref<7xi32>) -> ()
  return
}

// Min tests.
// CHECK-LABEL: func @float_min_op
func @float_min_op(%lhs: memref<7xf32>, %rhs: memref<7xf32>,
                   %result: memref<7xf32>) -> () {
  // CHECK: %[[NAN:.*]] = constant 0x7FC00000 : f32
  // CHECK: %[[CMP:.*]] = cmpf olt, %[[LHS_IN:.*]], %[[RHS_IN:.*]] : f32
  // CHECK: %[[MIN:.*]] = select %[[CMP]], %[[LHS_IN]], %[[RHS_IN]] : f32
  // CHECK: %[[ISNAN:.*]] = cmpf uno, %[[LHS_IN]], %[[RHS_IN]] : f32
  // CHECK: select %[[ISNAN]], %[[NAN]], %[[MIN]] : f32
  "lmhlo.minimum"(%lhs, %rhs, %result) {name = "min.1"}
      : (memref<7xf32>, memref<7xf32>, memref<7xf32>) -> ()
  return
}

// CHECK-LABEL: func @int_min_op
func @int_min_op(%lhs: memref<7xi32>, %rhs: memref<7xi32>,
                 %result: memref<7xi32>) -> () {
  // CHECK: %[[CHECK:.*]] = cmpi slt, %[[ONE:.*]], %[[TWO:.*]] : i32
  // CHECK: select %[[CHECK]], %[[ONE]], %[[TWO]] : i32
  "lmhlo.minimum"(%lhs, %rhs, %result) {name = "min.1"}
      : (memref<7xi32>, memref<7xi32>, memref<7xi32>) -> ()
  return
}

// Mul tests.
// CHECK-LABEL: func @float_mul_op
func @float_mul_op(%lhs: memref<7xf32>, %rhs: memref<7xf32>,
                   %result: memref<7xf32>) -> () {
  // CHECK: mulf %{{.*}}, %{{.*}} : f32
  "lmhlo.multiply"(%lhs, %rhs, %result) {name = "mul.1"}
      : (memref<7xf32>, memref<7xf32>, memref<7xf32>) -> ()
  return
}

// CHECK-LABEL: func @int_mul_op
func @int_mul_op(%lhs: memref<7xi32>, %rhs: memref<7xi32>,
                 %result: memref<7xi32>) -> () {
  // CHECK: muli %{{.*}}, %{{.*}} : i32
  "lmhlo.multiply"(%lhs, %rhs, %result) {name = "mul.1"}
      : (memref<7xi32>, memref<7xi32>, memref<7xi32>) -> ()
  return
}

// Sub tests.
// CHECK-LABEL: func @float_sub_op
func @float_sub_op(%lhs: memref<7xf32>, %rhs: memref<7xf32>,
                   %result: memref<7xf32>) -> () {
  // CHECK: subf %{{.*}}, %{{.*}} : f32
  "lmhlo.subtract"(%lhs, %rhs, %result) {name = "sub.1"}
      : (memref<7xf32>, memref<7xf32>, memref<7xf32>) -> ()
  return
}
// CHECK-LABEL: func @int_sub_op
func @int_sub_op(%lhs: memref<7xi32>, %rhs: memref<7xi32>,
                 %result: memref<7xi32>) -> () {
  // CHECK: subi %{{.*}}, %{{.*}} : i32
  "lmhlo.subtract"(%lhs, %rhs, %result) {name = "sub.1"}
      : (memref<7xi32>, memref<7xi32>, memref<7xi32>) -> ()
  return
}

// Dot tests.
// CHECK-LABEL: func @float_dot_op
func @float_dot_op(%lhs: memref<7x3xf32>, %rhs:
                  memref<3x4xf32>, %result: memref<7x4xf32> ) -> () {
    // CHECK-NEXT: affine.for %[[I:.*]] = 0 to 7 {
    // CHECK-NEXT:  affine.for %[[J:.*]] = 0 to 4 {
    // CHECK-NEXT:    affine.for %[[K:.*]] = 0 to 3 {
    // CHECK-NEXT:      %[[LHS:.*]] = affine.load %{{.*}}[%[[I]], %[[K]]] : memref<7x3xf32>
    // CHECK-NEXT:      %[[RHS:.*]] = affine.load %{{.*}}[%[[K]], %[[J]]] : memref<3x4xf32>
    // CHECK-NEXT:      %[[RESULT:.*]] = affine.load %{{.*}}[%[[I]], %[[J]]] : memref<7x4xf32>
    // CHECK-NEXT:      %[[MULT:.*]] = mulf %[[LHS]], %[[RHS]] : f32
    // CHECK-NEXT:      %[[ADD:.*]] =  addf %[[MULT]], %[[RESULT]] : f32
    // CHECK-NEXT:      affine.store %[[ADD]], %{{.*}}[%[[I]], %[[J]]] : memref<7x4xf32>
    // CHECK: return
  "lmhlo.dot"(%lhs, %rhs, %result) {
      dot_dimension_numbers = {
        lhs_batching_dimensions = dense<> : tensor<0xi64>,
        rhs_batching_dimensions = dense<> : tensor<0xi64>,
        lhs_contracting_dimensions = dense<1> : tensor<1xi64>,
        rhs_contracting_dimensions = dense<0> : tensor<1xi64>
      }
    } :
    (memref<7x3xf32>, memref<3x4xf32>, memref<7x4xf32>) -> ()
  return
}
// CHECK-LABEL: func @int_dot_op
func @int_dot_op(%lhs: memref<7x3xi32>, %rhs:
                  memref<3x4xi32>, %result: memref<7x4xi32> ) -> () {
    // CHECK-NEXT: affine.for %[[I:.*]] = 0 to 7 {
    // CHECK-NEXT:  affine.for %[[J:.*]] = 0 to 4 {
    // CHECK-NEXT:    affine.for %[[K:.*]] = 0 to 3 {
    // CHECK-NEXT:      %[[LHS:.*]] = affine.load %{{.*}}[%[[I]], %[[K]]] : memref<7x3xi32>
    // CHECK-NEXT:      %[[RHS:.*]] = affine.load %{{.*}}[%[[K]], %[[J]]] : memref<3x4xi32>
    // CHECK-NEXT:      %[[RESULT:.*]] = affine.load %{{.*}}[%[[I]], %[[J]]] : memref<7x4xi32>
    // CHECK-NEXT:      %[[MULT:.*]] = muli %[[LHS]], %[[RHS]] : i32
    // CHECK-NEXT:      %[[ADD:.*]] =  addi %[[MULT]], %[[RESULT]] : i32
    // CHECK-NEXT:      affine.store %[[ADD]], %{{.*}}[%[[I]], %[[J]]] : memref<7x4xi32>
    // CHECK: return
  "lmhlo.dot"(%lhs, %rhs, %result) {
      dot_dimension_numbers = {
        lhs_batching_dimensions = dense<> : tensor<0xi64>,
        rhs_batching_dimensions = dense<> : tensor<0xi64>,
        lhs_contracting_dimensions = dense<1> : tensor<1xi64>,
        rhs_contracting_dimensions = dense<0> : tensor<1xi64>
      }
     } :
    (memref<7x3xi32>, memref<3x4xi32>, memref<7x4xi32>) -> ()
  return
}

// CHECK-LABEL: func @concatenate
func @concatenate(%arg0: memref<1x1xf32>, %arg1: memref<1x100xf32>, %arg2: memref<1x200xf32>, %arg3: memref<1x301xf32>) {
    // CHECK-NEXT:    %[[RESULT:.*]] = memref.alloc() : memref<1x301xf32>
    // CHECK-NEXT:    affine.for %[[X:.*]] = 0 to 1 {
    // CHECK-NEXT:      affine.for %[[Y:.*]] = 0 to 1 {
    // CHECK-NEXT:        %[[LOAD:.*]] = affine.load %arg0[%[[X]], %[[Y]]] : memref<1x1xf32>
    // CHECK-NEXT:        affine.store %[[LOAD]], %[[RESULT]][%[[X]], %[[Y]]] : memref<1x301xf32>
    // CHECK-NEXT:      }
    // CHECK-NEXT:    }
    // CHECK-NEXT:    affine.for %[[X:.*]] = 0 to 1 {
    // CHECK-NEXT:      affine.for %[[Y:.*]] = 1 to 101 {
    // CHECK-NEXT:        %[[LOAD:.*]] = affine.load %arg1[%[[X]], %[[Y]] - 1] : memref<1x100xf32>
    // CHECK-NEXT:        affine.store %[[LOAD]], %[[RESULT]][%[[X]], %[[Y]]] : memref<1x301xf32>
    // CHECK-NEXT:      }
    // CHECK-NEXT:    }
    // CHECK-NEXT:    affine.for %[[X:.*]] = 0 to 1 {
    // CHECK-NEXT:      affine.for %[[Y:.*]] = 101 to 301 {
    // CHECK-NEXT:        %[[LOAD:.*]] = affine.load %arg2[%[[X]], %[[Y]] - 101] : memref<1x200xf32>
    // CHECK-NEXT:        affine.store %[[LOAD]], %[[RESULT]][%[[X]], %[[Y]]] : memref<1x301xf32>
    %0 = memref.alloc() : memref<1x301xf32>
    "lmhlo.concatenate"(%arg0, %arg1, %arg2, %0) {dimension = 1 : i64} : (memref<1x1xf32>, memref<1x100xf32>, memref<1x200xf32>, memref<1x301xf32>) -> ()
    "lmhlo.copy"(%0, %arg3) : (memref<1x301xf32>, memref<1x301xf32>) -> ()
    "lmhlo.terminator"() : () -> ()
}

// TODO(pashu123): Extend Support for dynamic dimensions.
// CHECK-LABEL: func @concatenate_dynamic
func @concatenate_dynamic(%arg0: memref<1x?xf32>, %arg1: memref<1x?xf32>, %arg2: memref<1x?xf32>) {
    // CHECK: "lmhlo.concatenate"
    %cst_1 = constant 1 : index
    %0 = memref.alloc(%cst_1) : memref<1x?xf32>
    "lmhlo.concatenate"(%arg0, %arg1, %0) {dimension = 1 : i64} : (memref<1x?xf32>,  memref<1x?xf32>, memref<1x?xf32>) -> ()
    "lmhlo.copy"(%0, %arg2) : (memref<1x?xf32>, memref<1x?xf32>) -> ()
    "lmhlo.terminator"() : () -> ()
}

// Gather op.
// Test case 1: A general GatherOp test case.
// CHECK-LABEL: func @gather_1
// CHECK-SAME: (%[[OPERAND:.*]]: memref<28996x512xf32>, %[[START_INDICES:.*]]: memref<1x128xi32>, %[[OUTPUT:.*]]: memref<1x128x512xf32>)
func @gather_1(%arg0: memref<28996x512xf32>, %arg1: memref<1x128xi32>, %arg2: memref<1x128x512xf32>) {
  %0 = memref.alloc() : memref<1x128x512xf32>
  "lmhlo.gather"(%arg0, %arg1, %0) {dimension_numbers = { collapsed_slice_dims = dense<0> : tensor<1xi64>,
							     index_vector_dim = 2 : i64,
							     offset_dims = dense<2> : tensor<1xi64>,
							     start_index_map = dense<0> : tensor<1xi64>},
				       indices_are_sorted = false, name = "gather.381", slice_sizes = dense<[1, 512]> : tensor<2xi64>} :
  (memref<28996x512xf32>, memref<1x128xi32>, memref<1x128x512xf32>) -> ()
  "lmhlo.copy"(%0, %arg2) : (memref<1x128x512xf32>, memref<1x128x512xf32>) -> ()
  "lmhlo.terminator"() : () -> ()
}
// CHECK-NEXT: %[[zero:.*]] = constant 0.000000e+00 : f32
// CHECK-NEXT: %[[temp_output:.*]] = memref.alloc() : memref<1x128x512xf32>
// CHECK-NEXT: affine.for %{{.*}} = 0 to 1 {
// CHECK-NEXT:   affine.for %{{.*}} = 0 to 128 {
// CHECK-NEXT:     affine.for %{{.*}} = 0 to 512 {
// CHECK-NEXT:       affine.store %[[zero]], %[[temp_output]][%{{.*}}, %{{.*}}, %{{.*}}] : memref<1x128x512xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: affine.for %[[batch0:.*]] = 0 to 1 {
// CHECK-NEXT:   affine.for %[[batch1:.*]] = 0 to 128 {
// CHECK-NEXT:     affine.for %[[offset0:.*]] = 0 to 512 {
// CHECK-NEXT:       affine.for %[[iv0:.*]] = 0 to 28996 {
// CHECK-NEXT:         %[[a:.*]] = affine.load %[[START_INDICES]][%[[batch0]], %[[batch1]]] : memref<1x128xi32>
// CHECK-NEXT:         %[[S_in0:.*]] = index_cast %[[a]] : i32 to index
// CHECK-NEXT:         %[[operand_val:.*]] = affine.load %[[OPERAND]][%[[iv0]], %[[offset0]]] : memref<28996x512xf32>
// CHECK-NEXT:         %[[pred:.*]] = cmpi eq, %[[S_in0]], %[[iv0]] : index
// CHECK-NEXT:         %[[selected_value:.*]] = select %[[pred]], %[[operand_val]], %[[zero]] : f32
// CHECK-NEXT:         %[[prev_value:.*]] = affine.load %[[temp_output]][%[[batch0]], %[[batch1]], %[[offset0]]] : memref<1x128x512xf32>
// CHECK-NEXT:         %[[final_value:.*]] = addf %[[selected_value]], %[[prev_value]] : f32
// CHECK-NEXT:         affine.store %[[final_value]], %[[temp_output]][%[[batch0]], %[[batch1]], %[[offset0]]] : memref<1x128x512xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }

// Test case 2: Checks for multi-dimensional starting indices.
// CHECK-LABEL: func @gather_2
// CHECK-SAME: (%[[OPERAND:.*]]: memref<16x11xf32>, %[[START_INDICES:.*]]: memref<5x2xi32>, %[[OUTPUT:.*]]: memref<5x8x6xf32>)
func @gather_2(%arg0: memref<16x11xf32>, %arg1: memref<5x2xi32>, %arg2: memref<5x8x6xf32>) {
  %0 = memref.alloc() : memref<5x8x6xf32>
  "lmhlo.gather"(%arg0, %arg1, %0) {dimension_numbers = { collapsed_slice_dims = dense<-1> : tensor<1xi64>,
                                                             index_vector_dim = 1 : i64,
                                                             offset_dims = dense<[1,2]> : tensor<2xi64>,
                                                             start_index_map = dense<[0,1]> : tensor<2xi64>},
                                       indices_are_sorted = false, name = "gather.381", slice_sizes = dense<[8, 6]> : tensor<2xi64>} :
  (memref<16x11xf32>, memref<5x2xi32>, memref<5x8x6xf32>) -> ()
  "lmhlo.copy"(%0, %arg2) : (memref<5x8x6xf32>, memref<5x8x6xf32>) -> ()
  "lmhlo.terminator"() : () -> ()
}
// CHECK-NEXT: %[[zero:.*]] = constant 0.000000e+00 : f32
// CHECK-NEXT: %c0 = constant 0 : index
// CHECK-NEXT: %c1 = constant 1 : index
// CHECK-NEXT: %[[temp_output:.*]] = memref.alloc() : memref<5x8x6xf32>
// CHECK-NEXT: affine.for %{{.*}} = 0 to 5 {
// CHECK-NEXT:   affine.for %{{.*}} = 0 to 8 {
// CHECK-NEXT:     affine.for %{{.*}} = 0 to 6 {
// CHECK-NEXT:       affine.store %[[zero]], %[[temp_output]][%{{.*}}, %{{.*}}, %{{.*}}] : memref<5x8x6xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: affine.for %[[batch0:.*]] = 0 to 5 {
// CHECK-NEXT:   affine.for %[[offset0:.*]] = 0 to 8 {
// CHECK-NEXT:     affine.for %[[offset1:.*]] = 0 to 6 {
// CHECK-NEXT:       affine.for %[[iv0:.*]] = 0 to 16 {
// CHECK-NEXT:         affine.for %[[iv1:.*]] = 0 to 11 {
// CHECK-NEXT:           %[[a:.*]] = affine.load %[[START_INDICES]][%[[batch0]], %c0] : memref<5x2xi32>
// CHECK-NEXT:           %[[S_in0:.*]] = index_cast %[[a]] : i32 to index
// CHECK-NEXT:           %[[b:.*]] = affine.load %[[START_INDICES]][%[[batch0]], %c1] : memref<5x2xi32>
// CHECK-NEXT:           %[[S_in1:.*]] = index_cast %[[b]] : i32 to index
// CHECK-NEXT:           %[[operand_val:.*]] = affine.load %[[OPERAND]][%[[iv0]], %[[iv1]]] : memref<16x11xf32>
// CHECK-NEXT:           %[[In0:.*]] = addi %[[S_in0]], %[[offset0]] : index
// CHECK-NEXT:           %[[pred1:.*]] = cmpi eq, %[[In0]], %[[iv0]] : index
// CHECK-NEXT:           %[[In1:.*]] = addi %[[S_in1]], %[[offset1]] : index
// CHECK-NEXT:           %[[pred2:.*]] = cmpi eq, %[[In1]], %[[iv1]] : index
// CHECK-NEXT:           %[[and1:.*]] = and %[[pred1]], %[[pred2]] : i1
// CHECK-NEXT:           %[[selected_value:.*]] = select %[[and1]], %[[operand_val]], %[[zero]] : f32
// CHECK-NEXT:           %[[prev_value:.*]] = affine.load %[[temp_output]][%[[batch0]], %[[offset0]], %[[offset1]]] : memref<5x8x6xf32>
// CHECK-NEXT:           %[[final_value:.*]] = addf %[[selected_value]], %[[prev_value]] : f32
// CHECK-NEXT:           affine.store %[[final_value]], %[[temp_output]][%[[batch0]], %[[offset0]], %[[offset1]]] : memref<5x8x6xf32>
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }

// Test case 3: Checks for multi-dimensional start_indices with multi-dimensional batch size. This also tests for f16 type.
// CHECK-LABEL: func @gather_3
// CHECK-SAME: (%[[OPERAND:.*]]: memref<16x11xf16>, %[[START_INDICES:.*]]: memref<4x2x5xi32>, %[[OUTPUT:.*]]: memref<4x5x8x6xf16>)
func @gather_3(%arg0: memref<16x11xf16>, %arg1: memref<4x2x5xi32>, %arg2: memref<4x5x8x6xf16>) {
  %0 = memref.alloc() : memref<4x5x8x6xf16>
  "lmhlo.gather"(%arg0, %arg1, %0) {dimension_numbers = { collapsed_slice_dims = dense<-1> : tensor<1xi64>,
                                                             index_vector_dim = 1 : i64,
                                                             offset_dims = dense<[2,3]> : tensor<2xi64>,
                                                             start_index_map = dense<[0,1]> : tensor<2xi64>},
                                       indices_are_sorted = false, name = "gather.381", slice_sizes = dense<[8, 6]> : tensor<2xi64>} :
  (memref<16x11xf16>, memref<4x2x5xi32>, memref<4x5x8x6xf16>) -> ()
  "lmhlo.copy"(%0, %arg2) : (memref<4x5x8x6xf16>, memref<4x5x8x6xf16>) -> ()
  "lmhlo.terminator"() : () -> ()
}
// CHECK-NEXT: %[[zero:.*]] = constant 0.000000e+00 : f16
// CHECK-NEXT: %c0 = constant 0 : index
// CHECK-NEXT: %c1 = constant 1 : index
// CHECK-NEXT: %[[temp_output:.*]] = memref.alloc() : memref<4x5x8x6xf16>
// CHECK-NEXT: affine.for %{{.*}} = 0 to 4 {
// CHECK-NEXT:   affine.for %{{.*}} = 0 to 5 {
// CHECK-NEXT:     affine.for %{{.*}} = 0 to 8 {
// CHECK-NEXT:       affine.for %{{.*}} = 0  to 6 {
// CHECK-NEXT:         affine.store %[[zero]], %[[temp_output]][%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] : memref<4x5x8x6xf16>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: affine.for %[[batch0:.*]] = 0 to 4 {
// CHECK-NEXT:   affine.for %[[batch1:.*]] = 0 to 5 {
// CHECK-NEXT:     affine.for %[[offset0:.*]] = 0 to 8 {
// CHECK-NEXT:       affine.for %[[offset1:.*]] = 0 to 6 {
// CHECK-NEXT:         affine.for %[[iv0:.*]] = 0 to 16 {
// CHECK-NEXT:           affine.for %[[iv1:.*]] = 0 to 11 {
// CHECK-NEXT:             %[[a:.*]] = affine.load %[[START_INDICES]][%[[batch0]], %c0, %[[batch1]]] : memref<4x2x5xi32>
// CHECK-NEXT:             %[[S_in0:.*]] = index_cast %[[a]] : i32 to index
// CHECK-NEXT:             %[[b:.*]] = affine.load %[[START_INDICES]][%[[batch0]], %c1, %[[batch1]]] : memref<4x2x5xi32>
// CHECK-NEXT:             %[[S_in1:.*]] = index_cast %[[b]] : i32 to index
// CHECK-NEXT:             %[[operand_val:.*]] = affine.load %[[OPERAND]][%[[iv0]], %[[iv1]]] : memref<16x11xf16>
// CHECK-NEXT:             %[[In0:.*]] = addi %[[S_in0]], %[[offset0]] : index
// CHECK-NEXT:             %[[pred1:.*]] = cmpi eq, %[[In0]], %[[iv0]] : index
// CHECK-NEXT:             %[[In1:.*]] = addi %[[S_in1]], %[[offset1]] : index
// CHECK-NEXT:             %[[pred2:.*]] = cmpi eq, %[[In1]], %[[iv1]] : index
// CHECK-NEXT:             %[[and1:.*]] = and %[[pred1]], %[[pred2]] : i1
// CHECK-NEXT:             %[[selected_value:.*]] = select %[[and1]], %[[operand_val]], %[[zero]] : f16
// CHECK-NEXT:             %[[prev_value:.*]] = affine.load %[[temp_output]][%[[batch0]], %[[batch1]], %[[offset0]], %[[offset1]]] : memref<4x5x8x6xf16>
// CHECK-NEXT:             %[[final_value:.*]] = addf %[[selected_value]], %[[prev_value]] : f16
// CHECK-NEXT:             affine.store %[[final_value]], %[[temp_output]][%[[batch0]], %[[batch1]], %[[offset0]], %[[offset1]]] : memref<4x5x8x6xf16>
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }

// Test case 4: Changing starting_index_map : X -> [0,X]
// CHECK-LABEL: func @gather_4
// CHECK-SAME: (%[[OPERAND:.*]]: memref<16x11xf32>, %[[START_INDICES:.*]]: memref<5x4xi32>, %[[OUTPUT:.*]]: memref<4x5x6xf32>)
func @gather_4(%arg0: memref<16x11xf32>, %arg1: memref<5x4xi32>, %arg2: memref<4x5x6xf32>) {
  %0 = memref.alloc() : memref<4x5x6xf32>
  "lmhlo.gather"(%arg0, %arg1, %0) {dimension_numbers = { collapsed_slice_dims = dense<0> : tensor<1xi64>,
                                                             index_vector_dim = 2 : i64,
                                                             offset_dims = dense<2> : tensor<1xi64>,
                                                             start_index_map = dense<0> : tensor<1xi64>},
                                       indices_are_sorted = false, name = "gather.381", slice_sizes = dense<[1, 6]> : tensor<2xi64>} :
  (memref<16x11xf32>, memref<5x4xi32>, memref<4x5x6xf32>) -> ()
  "lmhlo.copy"(%0, %arg2) : (memref<4x5x6xf32>, memref<4x5x6xf32>) -> ()
  "lmhlo.terminator"() : () -> ()
}
// CHECK-NEXT: %[[zero:.*]] = constant 0.000000e+00 : f32
// CHECK-NEXT: %[[temp_output:.*]] = memref.alloc() : memref<4x5x6xf32>
// CHECK-NEXT: affine.for %{{.*}} = 0 to 4 {
// CHECK-NEXT:   affine.for %{{.*}} = 0 to 5 {
// CHECK-NEXT:     affine.for %{{.*}} = 0 to 6 {
// CHECK-NEXT:       affine.store %[[zero]], %[[temp_output]][%{{.*}}, %{{.*}}, %{{.*}}] : memref<4x5x6xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: affine.for %[[batch0:.*]] = 0 to 5 {
// CHECK-NEXT:   affine.for %[[batch1:.*]] = 0 to 4 {
// CHECK-NEXT:     affine.for %[[offset0:.*]] = 0 to 6 {
// CHECK-NEXT:       affine.for %[[iv0:.*]] = 0 to 16 {
// CHECK-NEXT:         %[[a:.*]] = affine.load %[[START_INDICES]][%[[batch0]], %[[batch1]]] : memref<5x4xi32>
// CHECK-NEXT:         %[[S_in0:.*]] = index_cast %[[a]] : i32 to index
// CHECK-NEXT:         %[[operand_val:.*]] = affine.load %[[OPERAND]][%[[iv0]], %[[offset0]]] : memref<16x11xf32>
// CHECK-NEXT:         %[[pred:.*]] = cmpi eq, %[[S_in0]], %[[iv0]] : index
// CHECK-NEXT:         %[[selected_value:.*]] = select %[[pred]], %[[operand_val]], %[[zero]] : f32
// CHECK-NEXT:         %[[prev_value:.*]] = affine.load %[[temp_output]][%[[batch0]], %[[batch1]], %[[offset0]]] : memref<4x5x6xf32>
// CHECK-NEXT:         %[[final_value:.*]] = addf %[[selected_value]], %[[prev_value]] : f32
// CHECK-NEXT:         affine.store %[[final_value]], %[[temp_output]][%[[batch0]], %[[batch1]], %[[offset0]]] : memref<4x5x6xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }

// Test case 5: Testing for more than two equality checks.
// CHECK-LABEL: func @gather_5
func @gather_5(%arg0: memref<28996x512x256xf32>, %arg1: memref<10x3xi32>, %arg2: memref<10x20x10x5xf32>) {
  %0 = memref.alloc() : memref<10x20x10x5xf32>
  "lmhlo.gather"(%arg0, %arg1, %0) {dimension_numbers = { collapsed_slice_dims = dense<-1> : tensor<1xi64>,
                                                             index_vector_dim = 1 : i64,
                                                             offset_dims = dense<[1,2,3]> : tensor<3xi64>,
                                                             start_index_map = dense<[0,1,2]> : tensor<3xi64>},
                                       indices_are_sorted = false, name = "gather.381", slice_sizes = dense<[20, 10, 5]> : tensor<3xi64>} :
  (memref<28996x512x256xf32>, memref<10x3xi32>, memref<10x20x10x5xf32>) -> ()
  "lmhlo.copy"(%0, %arg2) : (memref<10x20x10x5xf32>, memref<10x20x10x5xf32>) -> ()
  "lmhlo.terminator"() : () -> ()
}
// CHECK: %[[and1:.*]] = and %{{.*}}, %{{.*}} : i1
// CHECK-NEXT: and %[[and1]], %{{.*}} : i1

// CHECK-LABEL: func @log
func @log(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>) {
  
// CHECK-NEXT: %[[RES:.*]] = memref.alloc() : memref<2x2xf32>
// CHECK-NEXT: affine.for %{{.*}} = 0 to 2 {
// CHECK-NEXT:   affine.for %{{.*}} = 0 to 2 {
// CHECK-NEXT:     %[[LOAD:.*]] = affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<2x2xf32>
// CHECK-NEXT:     %[[LOG:.*]] = math.log %[[LOAD]] : f32
// CHECK-NEXT:     affine.store %[[LOG]], %[[RES]][%{{.*}}, %{{.*}}] : memref<2x2xf32>

  %0 = memref.alloc() : memref<2x2xf32>
  "lmhlo.log"(%arg0, %0) : (memref<2x2xf32>, memref<2x2xf32>) -> ()
  "lmhlo.copy"(%0, %arg1) : (memref<2x2xf32>, memref<2x2xf32>) -> ()
  "lmhlo.terminator"() : () -> ()
}
