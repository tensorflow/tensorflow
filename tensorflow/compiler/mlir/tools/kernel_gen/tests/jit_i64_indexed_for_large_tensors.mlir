// RUN: kernel-gen-opt %s --split-input-file \
// RUN:   --tf-to-jit-invocation="tile-sizes=1,2,3 \
// RUN:   unroll-factors=3,2,1 max-supported-rank=32 \
// RUN:   enable-ftz=false index_64bit=false cpu-codegen=false \
// RUN:   jit_i64_indexed_for_large_tensors=true" | \
// RUN: FileCheck %s

// CHECK-LABEL: @unary_tanh_rint
// CHECK-SAME: (%[[ARG:.*]]: tensor<*xf32>)
func.func @unary_tanh_rint(%arg : tensor<*xf32>) -> (tensor<*xf32>) {
  // CHECK:      %[[MAX_SIZE:.*]] = arith.constant 4294967296 : index
  // CHECK:      %[[SHAPE:.*]] = shape.shape_of %arg0
  // CHECK:      %[[ELEMENT_COUNT:.*]] = shape.num_elements %[[SHAPE:.*]] : tensor<?xindex> -> index
  // CHECK:      %[[CONDITION:.*]] = arith.cmpi sgt, %[[ELEMENT_COUNT:.*]], %[[MAX_SIZE:.*]] : index
  // CHECK:      %[[IF_RES:.*]] = scf.if %[[CONDITION:.*]] -> (tensor<*xf32>) {
  // CHECK:        %[[CALLABLE:.*]] = tf_framework.jit_compile_from_str
  // CHECK-SAME:   "
  // CHECK-SAME:   module  {
  // CHECK-SAME:     func @main(%arg0: tensor<*xf32>) -> tensor<*xf32>
  // CHECK-SAME:     attributes {tf_entry}
  // CHECK-SAME:     {
  // CHECK-SAME:      %0 = \22tf.Tanh\22(%arg0)
  // CHECK-SAME:      return %0
  // CHECK-SAME:     }
  // CHECK-SAME:   }
  // CHECK-SAME:   "
  // CHECK-SAME:   {
  // CHECK-SAME:     cpuCodegen = false
  // CHECK-SAME:     enableFtz = false
  // CHECK-SAME:     index64Bit = true
  // CHECK-SAME:     maxSupportedRank = 32
  // CHECK-SAME:     tileSizes = [1, 2, 3]
  // CHECK-SAME:     unrollFactors = [3, 2, 1]
  // CHECK-SAME:   }
  // CHECK:        %[[RES:.*]] = tf_framework.jit_execute %[[CALLABLE]](%[[ARG]])
  // CHECK:        scf.yield %[[RES:.*]]
  // CHECK:      } else {
  // CHECK:        %4 = "tf.Tanh"(%arg0)
  // CHECK:        scf.yield %4 : tensor<*xf32>
  // CHECK:      }
  // CHECK:      return %[[IF_RES]]
  %0 = "tf.Tanh"(%arg) : (tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}