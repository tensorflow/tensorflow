// RUN: kernel-gen-opt %s --split-input-file \
// RUN:   --func-to-jit-invocation="tile-sizes=1,2,3 unroll-factors=3,2,1 \
// RUN:       enable-ftz=false cpu-codegen=false" | \
// RUN: FileCheck %s

// RUN: kernel-gen-opt %s --split-input-file \
// RUN:   --func-to-jit-invocation="tile-sizes=1,2,3 unroll-factors=3,2,1 \
// RUN:       enable-ftz=false cpu-codegen=false \
// RUN:       jit_i64_indexed_for_large_tensors=true" | \
// RUN: FileCheck %s --check-prefix=CHECK-JFLT

func.func @unary_tanh(%arg : tensor<?xf32>) -> tensor<?xf32> {
  %0 = mhlo.tanh %arg : tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// CHECK-LABEL: @unary_tanh
// CHECK-SAME:  %[[ARG:.*]]: tensor<?xf32>
// CHECK:       %[[CALLABLE:.*]] = tf_framework.jit_compile_from_str
// CHECK-SAME:      "
// CHECK-SAME:      module {
// CHECK-SAME:        func @main(%[[ARG_JIT:.*]]: tensor<?xf32>) -> tensor<?xf32>
// CHECK-SAME:          attributes {tf_entry}
// CHECK-SAME:        {
// CHECK-SAME:          %[[RES_JIT:.*]] = mhlo.tanh %[[ARG_JIT]]
// CHECK-SAME:          return %[[RES_JIT]]
// CHECK-SAME:        }
// CHECK-SAME:      }
// CHECK-SAME:      "
// CHECK-SAME:      {
// CHECK-SAME:        cpuCodegen = false
// CHECK-SAME:        enableFtz = false
// CHECK-SAME:        tileSizes = [1, 2, 3]
// CHECK-SAME:        unrollFactors = [3, 2, 1]
// CHECK-SAME:      }
// CHECK:       %[[RES:.*]] = tf_framework.jit_execute %[[CALLABLE]](%[[ARG]])
// CHECK:       return %[[RES]]

// CHECK-JFLT-LABEL: @unary_tanh
// CHECK-JFLT-SAME:  %[[ARG0:.*]]: tensor<?xf32>
// CHECK-JFLT:       %[[SHAPE:.*]] = shape.shape_of %[[ARG0]]
// CHECK-JFLT:       %[[NUM:.*]] = shape.num_elements %[[SHAPE]]
// CHECK-JFLT:       %[[LIMIT:.*]] = arith.constant 2147483647
// CHECK-JFLT:       %[[CMPI:.*]] = arith.cmpi sgt, %[[NUM]], %[[LIMIT]]
// CHECK-JFLT:       %[[IF:.*]] = scf.if %[[CMPI]]
// CHECK-JFLT:         %[[JIT:.*]] = tf_framework.jit_compile_from_str
// CHECK-JFLT-SAME:        "module
// CHECK-JFLT-SAME:        cpuCodegen = false
// CHECK-JFLT-SAME:        enableFtz = false
// CHECK-JFLT-SAME:        index64Bit = true
// CHECK-JFLT-SAME:        tileSizes = [1, 2, 3]
// CHECK-JFLT-SAME:        unrollFactors = [3, 2, 1]
// CHECK-JFLT:         %[[JIT_0:.*]] = tf_framework.jit_execute %[[JIT]](%[[ARG0]])
// CHECK-JFLT:         scf.yield %[[JIT_0]]
// CHECK-JFLT:       else
// CHECK-JFLT:         %[[VAL:.*]] = mhlo.tanh %[[ARG0]]
// CHECK-JFLT:         scf.yield %[[VAL]]
// CHECK-JFLT:       return %[[IF]]

// -----

func.func @binary_sub(%arg0 : tensor<*xf32>, %arg1 : tensor<*xf32>) -> tensor<*xf32> {
  %0 = chlo.broadcast_subtract %arg0, %arg1 : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// CHECK-LABEL: @binary_sub
// CHECK-SAME:  %[[ARG0:.*]]: tensor<*xf32>, %[[ARG1:.*]]: tensor<*xf32>
// CHECK:       %[[CALLABLE:.*]] = tf_framework.jit_compile_from_str
// CHECK-SAME:      "
// CHECK-SAME:      module {
// CHECK-SAME:        func @main(%[[ARG0_JIT:.*]]: tensor<*xf32>, %[[ARG1_JIT:.*]]: tensor<*xf32>) -> tensor<*xf32>
// CHECK-SAME:          attributes {tf_entry}
// CHECK-SAME:        {
// CHECK-SAME:          %[[RES_JIT:.*]] = chlo.broadcast_subtract %[[ARG0_JIT]], %[[ARG1_JIT]]
// CHECK-SAME:          return %[[RES_JIT]]
// CHECK-SAME:        }
// CHECK-SAME:      }
// CHECK-SAME:      "
// CHECK-SAME:      {
// CHECK-SAME:        cpuCodegen = false
// CHECK-SAME:        enableFtz = false
// CHECK-SAME:        tileSizes = [1, 2, 3]
// CHECK-SAME:        unrollFactors = [3, 2, 1]
// CHECK-SAME:      }
// CHECK:       %[[RES:.*]] = tf_framework.jit_execute %[[CALLABLE]](%[[ARG0]], %[[ARG1]])
// CHECK:       return %[[RES]]

// CHECK-JFLT-LABEL: @binary_sub
// CHECK-JFLT:  %[[ARG0:.*]]: tensor<*xf32>, %[[ARG1:.*]]: tensor<*xf32>
// CHECK-JFLT:  %[[SHAPE1:.*]] = shape.shape_of %[[ARG0]] : tensor<*xf32> -> tensor<?xindex>
// CHECK-JFLT:  %[[ELEMENTCOUNT1:.*]] = shape.num_elements %[[SHAPE1]] : tensor<?xindex> -> index
// CHECK-JFLT:  %[[LIMIT:.*]] = arith.constant 2147483647
// CHECK-JFLT:  %[[COMP1:.*]] = arith.cmpi sgt, %[[ELEMENTCOUNT1]], %[[LIMIT]] : index
// CHECK-JFLT:  %[[SHAPE2:.*]] = shape.shape_of %[[ARG1]] : tensor<*xf32> -> tensor<?xindex>
// CHECK-JFLT:  %[[ELEMENTCOUNT2:.*]] = shape.num_elements %[[SHAPE2]] : tensor<?xindex> -> index
// CHECK-JFLT:  %[[COMP2:.*]]  = arith.cmpi sgt, %[[ELEMENTCOUNT2]], %[[LIMIT]] : index
// CHECK-JFLT:  %[[COMPRES:.*]] = arith.ori %[[COMP1]], %[[COMP2]] : i1
// CHECK-JFLT:  %[[IFRES:.*]] = scf.if %[[COMPRES]] -> (tensor<*xf32>) {
// CHECK-JFLT:       %[[CALLABLE:.*]] = tf_framework.jit_compile_from_str
// CHECK-JFLT-SAME:      "
// CHECK-JFLT-SAME:      module {
// CHECK-JFLT-SAME:        func @main(%[[ARG0_JIT:.*]]: tensor<*xf32>, %[[ARG1_JIT:.*]]: tensor<*xf32>) -> tensor<*xf32>
// CHECK-JFLT-SAME:          attributes {tf_entry}
// CHECK-JFLT-SAME:        {
// CHECK-JFLT-SAME:          %[[RES_JIT:.*]] = chlo.broadcast_subtract %[[ARG0_JIT]], %[[ARG1_JIT]]
// CHECK-JFLT-SAME:          return %[[RES_JIT]]
// CHECK-JFLT-SAME:        }
// CHECK-JFLT-SAME:      }
// CHECK-JFLT-SAME:      "
// CHECK-JFLT-SAME:      {
// CHECK-JFLT-SAME:        cpuCodegen = false
// CHECK-JFLT-SAME:        enableFtz = false
// CHECK-JFLT-SAME:        tileSizes = [1, 2, 3]
// CHECK-JFLT-SAME:        unrollFactors = [3, 2, 1]
// CHECK-JFLT-SAME:      }
// CHECK-JFLT:       %[[RES:.*]] = tf_framework.jit_execute %[[CALLABLE]](%[[ARG0]], %[[ARG1]])
// CHECK-JFLT:       scf.yield %[[RES]] : tensor<*xf32>
// CHECK-JFLT:     } else {
// CHECK-JFLT:       %[[RES2:.*]] = chlo.broadcast_subtract %[[ARG0]], %[[ARG1]] : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
// CHECK-JFLT:       scf.yield %[[RES2]] : tensor<*xf32>
// CHECK-JFLT:     }
// CHECK-JFLT:     return %[[IFRES]]

// -----

func.func @reciprocal(%arg0: tensor<*xf32>)
    -> tensor<*xf32> attributes {tf_entry, llvm.emit_c_interface} {
  %0 = mhlo.constant dense<1.0> : tensor<f32>
  %1 = chlo.broadcast_divide %0, %arg0 : (tensor<f32>, tensor<*xf32>) -> tensor<*xf32>
  func.return %1 : tensor<*xf32>
}

// CHECK-LABEL: @reciprocal
// CHECK-SAME:  %[[ARG:.*]]: tensor<*xf32>
// CHECK:       %[[CALLABLE:.*]] = tf_framework.jit_compile_from_str
// CHECK-SAME:    module {
// CHECK-SAME:     func.func @main(%[[ARG0_JIT:.*]]: tensor<*xf32>) -> tensor<*xf32>
// CHECK-SAME:       %[[CST:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK-SAME:       %[[RES_JIT:.*]] = chlo.broadcast_divide %[[CST]], %[[ARG0_JIT]] : (tensor<f32>, tensor<*xf32>) -> tensor<*xf32>
// CHECK=SAME:       return %[[RES_JIT]] : tensor<*xf32>
// CHECK-SAME:     }
// CHECK-SAME:   }
// CHECK-SAME:   {
// CHECK-SAME:     cpuCodegen = false,
// CHECK-SAME:     enableFtz = false,
// CHECK-SAME:     index64Bit = false,
// CHECK-SAME:     tileSizes = [1, 2, 3],
// CHECK-SAME:     unrollFactors = [3, 2, 1]
// CHECK-SAME:   }
// CHECK:      %[[RES:.*]] = tf_framework.jit_execute %[[CALLABLE]](%[[ARG]]) : tensor<*xf32> -> tensor<*xf32>
// CHECK:      return %[[RES]] : tensor<*xf32>

// CHECK-JFLT-LABEL: @reciprocal
// CHECK-JFLT-SAME:  %[[ARG0:.*]]: tensor<*xf32>
// CHECK-JFLT:       %[[SHAPE:.*]] = shape.shape_of %[[ARG0]]
// CHECK-JFLT:       %[[NUM:.*]] = shape.num_elements %[[SHAPE]]
// CHECK-JFLT:       %[[LIMIT:.*]] = arith.constant 2147483647
// CHECK-JFLT:       %[[CMPI:.*]] = arith.cmpi sgt, %[[NUM]], %[[LIMIT]]
// CHECK-JFLT:       %[[IF:.*]] = scf.if %[[CMPI]]
// CHECK-JFLT:         %[[JIT:.*]] = tf_framework.jit_compile_from_str
// CHECK-JFLT-SAME:        "module
// CHECK-JFLT-SAME:        cpuCodegen = false
// CHECK-JFLT-SAME:        enableFtz = false
// CHECK-JFLT-SAME:        index64Bit = true
// CHECK-JFLT-SAME:        tileSizes = [1, 2, 3]
// CHECK-JFLT-SAME:        unrollFactors = [3, 2, 1]
// CHECK-JFLT:         %[[JIT_0:.*]] = tf_framework.jit_execute %[[JIT]](%[[ARG0]])
// CHECK-JFLT:         scf.yield %[[JIT_0]]
// CHECK-JFLT:       else
// CHECK-JFLT:         %[[CST:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK-JFLT:         %[[VAL:.*]] = chlo.broadcast_divide %[[CST]], %[[ARG0]] : (tensor<f32>, tensor<*xf32>) -> tensor<*xf32>
// CHECK-JFLT:         scf.yield %[[VAL]]
// CHECK-JFLT:       return %[[IF]]
