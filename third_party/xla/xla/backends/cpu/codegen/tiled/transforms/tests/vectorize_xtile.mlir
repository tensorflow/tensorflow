// RUN: fusion_compiler_opt --xtile-cpu-vectorize-xtile %s | FileCheck %s

// CHECK-LABEL: @vectorize_add
// CHECK-SAME: %[[MEMREF:.*]]: memref<128xf32>
func.func @vectorize_add(%memref: memref<128xf32>, %offset: index) {
  // CHECK: %[[EXT0:.*]] = xtile.extract %[[MEMREF]][%[[OFFSET:.*]]] [8] [1] : memref<128xf32> -> tensor<8xf32>
  // CHECK: %[[CAST0:.*]] = builtin.unrealized_conversion_cast %[[EXT0]] : tensor<8xf32> to vector<8xf32>
  // CHECK: %[[EXT1:.*]] = xtile.extract %[[MEMREF]][%[[OFFSET]]] [8] [1] : memref<128xf32> -> tensor<8xf32>
  // CHECK: %[[CAST1:.*]] = builtin.unrealized_conversion_cast %[[EXT1]] : tensor<8xf32> to vector<8xf32>
  // CHECK: %[[ADD:.*]] = arith.addf %[[CAST0]], %[[CAST1]] : vector<8xf32>
  // CHECK: %[[CAST2:.*]] = builtin.unrealized_conversion_cast %[[ADD]] : vector<8xf32> to tensor<8xf32>
  // CHECK: xtile.insert %[[CAST2]] into %[[MEMREF]][%[[OFFSET]]] [8] [1] : tensor<8xf32> -> memref<128xf32>
  
  %0 = xtile.extract %memref[%offset] [8] [1] : memref<128xf32> -> tensor<8xf32>
  %1 = xtile.extract %memref[%offset] [8] [1] : memref<128xf32> -> tensor<8xf32>
  %2 = arith.addf %0, %1 : tensor<8xf32>
  xtile.insert %2 into %memref[%offset] [8] [1] : tensor<8xf32> -> memref<128xf32>
  return
}

// CHECK-LABEL: @vectorize_math
// CHECK-SAME: %[[MEMREF:.*]]: memref<128xf32>
func.func @vectorize_math(%memref: memref<128xf32>, %offset: index) {
  // CHECK: %[[EXT:.*]] = xtile.extract %[[MEMREF]][%[[OFFSET:.*]]] [8] [1] : memref<128xf32> -> tensor<8xf32>
  // CHECK: %[[CAST0:.*]] = builtin.unrealized_conversion_cast %[[EXT]] : tensor<8xf32> to vector<8xf32>
  // CHECK: %[[SIN:.*]] = math.sin %[[CAST0]] : vector<8xf32>
  // CHECK: %[[CAST1:.*]] = builtin.unrealized_conversion_cast %[[SIN]] : vector<8xf32> to tensor<8xf32>
  // CHECK: xtile.insert %[[CAST1]] into %[[MEMREF]][%[[OFFSET]]] [8] [1] : tensor<8xf32> -> memref<128xf32>
  %0 = xtile.extract %memref[%offset] [8] [1] : memref<128xf32> -> tensor<8xf32>
  %1 = math.sin %0 : tensor<8xf32>
  xtile.insert %1 into %memref[%offset] [8] [1] : tensor<8xf32> -> memref<128xf32>
  return
}

// CHECK-LABEL: @vectorize_cast
// CHECK-SAME: %[[MEMREF:.*]]: memref<128xf32>, %{{.*}}: index, %[[MEMREF_I32:.*]]: memref<128xi32>
func.func @vectorize_cast(%memref: memref<128xf32>, %offset: index, %memref_i32: memref<128xi32>) {
  // CHECK: %[[EXT:.*]] = xtile.extract %[[MEMREF]][%[[OFFSET:.*]]] [8] [1] : memref<128xf32> -> tensor<8xf32>
  // CHECK: %[[CAST0:.*]] = builtin.unrealized_conversion_cast %[[EXT]] : tensor<8xf32> to vector<8xf32>
  // CHECK: %[[CAST_OP:.*]] = arith.fptosi %[[CAST0]] : vector<8xf32> to vector<8xi32>
  // CHECK: %[[CAST1:.*]] = builtin.unrealized_conversion_cast %[[CAST_OP]] : vector<8xi32> to tensor<8xi32>
  // CHECK: xtile.insert %[[CAST1]] into %[[MEMREF_I32]][%[[OFFSET]]] [8] [1] : tensor<8xi32> -> memref<128xi32>
  %0 = xtile.extract %memref[%offset] [8] [1] : memref<128xf32> -> tensor<8xf32>
  %1 = arith.fptosi %0 : tensor<8xf32> to tensor<8xi32>
  xtile.insert %1 into %memref_i32[%offset] [8] [1] : tensor<8xi32> -> memref<128xi32>
  return
}
