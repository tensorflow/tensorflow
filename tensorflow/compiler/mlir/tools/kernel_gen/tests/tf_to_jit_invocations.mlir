// RUN: kernel-gen-opt %s --split-input-file \
// RUN:   --tf-to-jit-invocation="tile-sizes=1,2,3 unroll-factors=3,2,1 \
// RUN:       max-supported-rank=32 enable-ftz=false cpu-codegen=false" | \
// RUN: FileCheck %s

// RUN: kernel-gen-opt %s --split-input-file \
// RUN:   --tf-to-jit-invocation="tile-sizes=1,2,3 unroll-factors=3,2,1 \
// RUN:       max-supported-rank=32 enable-ftz=false cpu-codegen=false \
// RUN:       jit_i64_indexed_for_large_tensors=true" | \
// RUN: FileCheck %s --check-prefix=CHECK-JFLT

func.func @unary_tanh(%arg : tensor<*xf32>) -> tensor<*xf32> {
  %0 = "tf.Tanh"(%arg) : (tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// CHECK-LABEL: @unary_tanh
// CHECK-SAME:  %[[ARG:.*]]: tensor<*xf32>
// CHECK:       %[[CALLABLE:.*]] = tf_framework.jit_compile_from_str
// CHECK-SAME:      "
// CHECK-SAME:      module {
// CHECK-SAME:        func @main(%[[ARG_JIT:.*]]: tensor<*xf32>) -> tensor<*xf32>
// CHECK-SAME:          attributes {tf_entry}
// CHECK-SAME:        {
// CHECK-SAME:          %[[RES_JIT:.*]] = \22tf.Tanh\22(%[[ARG_JIT]])
// CHECK-SAME:          return %[[RES_JIT]]
// CHECK-SAME:        }
// CHECK-SAME:      }
// CHECK-SAME:      "
// CHECK-SAME:      {
// CHECK-SAME:        cpuCodegen = false
// CHECK-SAME:        enableFtz = false
// CHECK-SAME:        maxSupportedRank = 32 : i64
// CHECK-SAME:        tileSizes = [1, 2, 3]
// CHECK-SAME:        unrollFactors = [3, 2, 1]
// CHECK-SAME:      }
// CHECK:       %[[RES:.*]] = tf_framework.jit_execute %[[CALLABLE]](%[[ARG]])
// CHECK:       return %[[RES]]

// CHECK-JFLT-LABEL: @unary_tanh
// CHECK-JFLT-SAME:  %[[ARG0:.*]]: tensor<*xf32>
// CHECK-JFLT-DAG:   %[[C4294967296:.*]] = arith.constant 4294967296
// CHECK-JFLT:       %[[SHAPE:.*]] = shape.shape_of %[[ARG0]]
// CHECK-JFLT:       %[[NUM:.*]] = shape.num_elements %[[SHAPE]]
// CHECK-JFLT:       %[[CMPI:.*]] = arith.cmpi sgt, %[[NUM]], %[[C4294967296]]
// CHECK-JFLT:       %[[IF:.*]] = scf.if %[[CMPI]]
// CHECK-JFLT:         %[[JIT:.*]] = tf_framework.jit_compile_from_str
// CHECK-JFLT-SAME:        "module
// CHECK-JFLT-SAME:        cpuCodegen = false
// CHECK-JFLT-SAME:        enableFtz = false
// CHECK-JFLT-SAME:        index64Bit = true
// CHECK-JFLT-SAME:        maxSupportedRank = 32
// CHECK-JFLT-SAME:        tileSizes = [1, 2, 3]
// CHECK-JFLT-SAME:        unrollFactors = [3, 2, 1]
// CHECK-JFLT:         %[[JIT_0:.*]] = tf_framework.jit_execute %[[JIT]](%[[ARG0]])
// CHECK-JFLT:         scf.yield %[[JIT_0]]
// CHECK-JFLT:       else
// CHECK-JFLT:         %[[VAL:.*]] = "tf.Tanh"(%[[ARG0]])
// CHECK-JFLT:         scf.yield %[[VAL]]
// CHECK-JFLT:       return %[[IF]]

// -----

func.func @binary_sub(%arg0 : tensor<*xf32>, %arg1 : tensor<*xf32>) -> tensor<*xf32> {
  %0 = "tf.Sub"(%arg0, %arg1) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
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
// CHECK-SAME:          %[[RES_JIT:.*]] = \22tf.Sub\22(%[[ARG0_JIT]], %[[ARG1_JIT]])
// CHECK-SAME:          return %[[RES_JIT]]
// CHECK-SAME:        }
// CHECK-SAME:      }
// CHECK-SAME:      "
// CHECK-SAME:      {
// CHECK-SAME:        cpuCodegen = false
// CHECK-SAME:        enableFtz = false
// CHECK-SAME:        maxSupportedRank = 32 : i64
// CHECK-SAME:        tileSizes = [1, 2, 3]
// CHECK-SAME:        unrollFactors = [3, 2, 1]
// CHECK-SAME:      }
// CHECK:       %[[RES:.*]] = tf_framework.jit_execute %[[CALLABLE]](%[[ARG0]], %[[ARG1]])
// CHECK:       return %[[RES]]
