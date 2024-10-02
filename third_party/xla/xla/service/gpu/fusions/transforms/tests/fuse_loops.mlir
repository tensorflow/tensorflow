// RUN: mlir_fusions_opt -split-input-file %s -xla-gpu-fuse-loops \
// RUN: | FileCheck %s

#indexing_map = #xla_gpu.indexing_map<"(d0, d1)[s0, s1] ->"
"   (d1 floordiv 30,"
"   ((d1 floordiv 6) mod 5) * 32 + s0 * 4 + d0 floordiv 32,"
"   (d1 mod 6) * 32 + d0 mod 32),"
" domain:"
"   d0 in [0, 127], d1 in [0, 599],"
"   s0 in [0, 7], s1 in [0, 0],"
"   (d1 mod 6) * 32 + d0 mod 32 in [0, 169]">
#indexing_map1 = #xla_gpu.indexing_map<"(th_x, d1)[s0, s1] ->"
"   (0,"
"   th_x mod 32,"
"   th_x floordiv 32 + s0 * 4),"
" domain:"
"   th_x in [0, 127], d1 in [0, 599],"
"   s0 in [0, 7], s1 in [0, 0],"
"   (d1 mod 6) * 32 + th_x mod 32 in [0, 169]">
func.func @fuse_loops(%arg0: tensor<20x160x170xf32>) -> tensor<1x32x33xf32> {
  %cst = arith.constant dense<0.000000e+00> : vector<8x1xf32>
  %c0 = arith.constant 0 : index
  %tid = gpu.thread_id  x {xla.range = [0 : index, 127 : index]}
  %bid = gpu.block_id  x {xla.range = [0 : index, 599 : index]}
  %shmem = xla_gpu.allocate_shared : tensor<1x32x33xf32>
  %xla_loop = xla_gpu.loop (%tid, %bid)[%i, %j]
    -> (%ra, %rb, %rc) in #indexing_map iter_args(%iter = %cst) -> (vector<8x1xf32>) {
    %extracted = tensor.extract %arg0[%ra, %rb, %rc] : tensor<20x160x170xf32>
    %0 = math.exp %extracted : f32
    %1 = vector.insert %0, %iter [%i, %j] : f32 into vector<8x1xf32>
    xla_gpu.yield %1 : vector<8x1xf32>
  }
  %xla_loop_0 = xla_gpu.loop (%tid, %bid)[%i, %j]
    -> (%ra, %rb, %rc) in #indexing_map1 iter_args(%iter = %shmem) -> (tensor<1x32x33xf32>) {
    %0 = vector.extract %xla_loop[%i, %j] : f32 from vector<8x1xf32>
    %inserted = tensor.insert %0 into %iter[%ra, %rb, %rc] : tensor<1x32x33xf32>
    xla_gpu.yield %inserted : tensor<1x32x33xf32>
  }
  %synced_tensor = xla_gpu.sync_threads %xla_loop_0 : tensor<1x32x33xf32>
  return %synced_tensor : tensor<1x32x33xf32>
}


// CHECK: #[[$FUSED_MAP:.*]] = #xla_gpu.indexing_map<"(d0, d1)[s0, s1] ->
// CHECK-SAME: (d1 floordiv 30, ((d1 floordiv 6) mod 5) * 32 + s0 * 4 + d0 floordiv 32,
// CHECK-SAME: (d1 mod 6) * 32 + d0 mod 32, 0, d0 mod 32, d0 floordiv 32 + s0 * 4),
// CHECK-SAME: domain: d0 in [0, 127], d1 in [0, 599],
// CHECK-SAME: s0 in [0, 7], s1 in [0, 0], (d1 mod 6) * 32 + d0 mod 32 in [0, 169]

// CHECK: %[[FUSED_LOOP:.*]] = xla_gpu.loop  {{.*}} in #[[$FUSED_MAP]]
// CHECK-NOT: vector.insert
// CHECK-NOT: vector.extract
// CHECK: %[[EXTRACTED:.*]] = tensor.extract
// CHECK: %[[EXP:.*]] = math.exp %[[EXTRACTED]]
// CHECK: tensor.insert %[[EXP]]

// CHECK: xla_gpu.sync_threads %[[FUSED_LOOP]]

// -----

#indexing_map = #xla_gpu.indexing_map<"(d0, d1)[s0, s1] ->"
"   (d1 floordiv 30,"
"   ((d1 floordiv 6) mod 5) * 32 + s0 * 4 + d0 floordiv 32,"
"   (d1 mod 6) * 32 + d0 mod 32),"
" domain:"
"   d0 in [0, 127], d1 in [0, 599],"
"   s0 in [0, 7], s1 in [0, 0],"
"   (d1 mod 6) * 32 + d0 mod 32 in [0, 169]">
#indexing_map1 = #xla_gpu.indexing_map<"(d0, d1)[s0, s1] ->"
"   (0,"
"   d0 mod 32,"
"   d0 floordiv 32 + s0 * 4),"
" domain:"
"   d0 in [0, 127], d1 in [0, 599],"
"   s0 in [0, 7], s1 in [0, 0],"
"   (d1 mod 6) * 32 + d0 mod 32 in [0, 169]">
func.func @do_not_fuse_index_mismatch(%arg0: tensor<20x160x170xf32>) -> tensor<1x32x33xf32> {
  %cst = arith.constant dense<0.000000e+00> : vector<8x1xf32>
  %c0 = arith.constant 0 : index
  %tid = gpu.thread_id  x {xla.range = [0 : index, 127 : index]}
  %bid = gpu.block_id  x {xla.range = [0 : index, 599 : index]}
  %shmem = xla_gpu.allocate_shared : tensor<1x32x33xf32>
  %xla_loop = xla_gpu.loop (%tid, %bid)[%i, %j]
    -> (%ra, %rb, %rc) in #indexing_map iter_args(%iter = %cst) -> (vector<8x1xf32>) {
    %extracted = tensor.extract %arg0[%ra, %rb, %rc] : tensor<20x160x170xf32>
    %0 = math.exp %extracted : f32
    %1 = vector.insert %0, %iter [%j, %i] : f32 into vector<8x1xf32>
    xla_gpu.yield %1 : vector<8x1xf32>
  }
  %xla_loop_0 = xla_gpu.loop (%tid, %bid)[%i, %j]
    -> (%ra, %rb, %rc) in #indexing_map1 iter_args(%iter = %shmem) -> (tensor<1x32x33xf32>) {
    %0 = vector.extract %xla_loop[%i, %j] : f32 from vector<8x1xf32>
    %inserted = tensor.insert %0 into %iter[%ra, %rb, %rc] : tensor<1x32x33xf32>
    xla_gpu.yield %inserted : tensor<1x32x33xf32>
  }
  return %xla_loop_0 : tensor<1x32x33xf32>
}

// CHECK-LABEL: @do_not_fuse_index_mismatch
// CHECK: xla_gpu.loop
// CHECK: vector.insert
// CHECK: xla_gpu.loop
// CHECK: vector.extract

// -----

#indexing_map = #xla_gpu.indexing_map<"(d0, d1)[s0, s1] ->"
"   (d1 floordiv 30,"
"   ((d1 floordiv 6) mod 5) * 32 + s0 * 4 + d0 floordiv 32,"
"   (d1 mod 6) * 32 + d0 mod 32),"
" domain:"
"   d0 in [0, 127], d1 in [0, 599],"
"   s0 in [0, 7], s1 in [0, 0],"
"   (d1 mod 6) * 32 + d0 mod 32 in [0, 169]">
#indexing_map1 = #xla_gpu.indexing_map<"(d0, d1)[s0, s1] ->"
"   (0,"
"   d0 mod 32,"
"   d0 floordiv 32 + s0 * 4),"
" domain:"
"   d0 in [0, 127], d1 in [0, 599],"
"   s0 in [0, 7], s1 in [0, 0],"
"   (d1 mod 6) * 32 + d0 mod 32 in [0, 169]">
func.func @do_not_fuse_multiple_uses(%arg0: tensor<20x160x170xf32>) -> tensor<1x32x33xf32> {
  %cst = arith.constant dense<0.000000e+00> : vector<8x1xf32>
  %c0 = arith.constant 0 : index
  %tid = gpu.thread_id  x {xla.range = [0 : index, 127 : index]}
  %bid = gpu.block_id  x {xla.range = [0 : index, 599 : index]}
  %shmem = xla_gpu.allocate_shared : tensor<1x32x33xf32>
  %xla_loop = xla_gpu.loop (%tid, %bid)[%i, %j]
    -> (%ra, %rb, %rc) in #indexing_map iter_args(%iter = %cst) -> (vector<8x1xf32>) {
    %extracted = tensor.extract %arg0[%ra, %rb, %rc] : tensor<20x160x170xf32>
    %0 = math.exp %extracted : f32
    %1 = vector.insert %0, %iter [%i, %j] : f32 into vector<8x1xf32>
    xla_gpu.yield %1 : vector<8x1xf32>
  }
  %xla_loop_0 = xla_gpu.loop (%tid, %bid)[%i, %j]
    -> (%ra, %rb, %rc) in #indexing_map1 iter_args(%iter = %shmem) -> (tensor<1x32x33xf32>) {
    %0 = vector.extract %xla_loop[%i, %j] : f32 from vector<8x1xf32>
    %inserted = tensor.insert %0 into %iter[%ra, %rb, %rc] : tensor<1x32x33xf32>
    xla_gpu.yield %inserted : tensor<1x32x33xf32>
  }
  %synced_tensor = xla_gpu.sync_threads %xla_loop_0 : tensor<1x32x33xf32>
  %0 = vector.extract %xla_loop [2, 0] : f32 from vector<8x1xf32>
  return %synced_tensor : tensor<1x32x33xf32>
}

// CHECK-LABEL: @do_not_fuse_multiple_uses
// CHECK: xla_gpu.loop
// CHECK: vector.insert
// CHECK: xla_gpu.loop
// CHECK: vector.extract

// -----

#indexing_map = #xla_gpu.indexing_map<"(d0, d1)[s0, s1] ->"
"   (d1 floordiv 30,"
"   ((d1 floordiv 6) mod 5) * 32 + s0 * 4 + d0 floordiv 32,"
"   (d1 mod 6) * 32 + d0 mod 32),"
" domain:"
"   d0 in [0, 127], d1 in [0, 599],"
"   s0 in [0, 7], s1 in [0, 0],"
"   (d1 mod 6) * 32 + d0 mod 32 in [0, 169]">
#indexing_map1 = #xla_gpu.indexing_map<"(d0, d1)[s0, s1] ->"
"   (0,"
"   d0 mod 32,"
"   d0 floordiv 32 + s0 * 4),"
" domain:"
"   d0 in [0, 127], d1 in [0, 599],"
"   s0 in [0, 5], s1 in [0, 0],"
"   (d1 mod 6) * 32 + d0 mod 32 in [0, 169]">
func.func @do_not_fuse_map_domain_mismatch(%arg0: tensor<20x160x170xf32>) -> tensor<1x32x33xf32> {
  %cst = arith.constant dense<0.000000e+00> : vector<8x1xf32>
  %c0 = arith.constant 0 : index
  %tid = gpu.thread_id  x {xla.range = [0 : index, 127 : index]}
  %bid = gpu.block_id  x {xla.range = [0 : index, 599 : index]}
  %shmem = xla_gpu.allocate_shared : tensor<1x32x33xf32>
  %xla_loop = xla_gpu.loop (%tid, %bid)[%i, %j]
    -> (%ra, %rb, %rc) in #indexing_map iter_args(%iter = %cst) -> (vector<8x1xf32>) {
    %extracted = tensor.extract %arg0[%ra, %rb, %rc] : tensor<20x160x170xf32>
    %0 = math.exp %extracted : f32
    %1 = vector.insert %0, %iter [%i, %j] : f32 into vector<8x1xf32>
    xla_gpu.yield %1 : vector<8x1xf32>
  }
  %xla_loop_0 = xla_gpu.loop (%tid, %bid)[%i, %j]
    -> (%ra, %rb, %rc) in #indexing_map1 iter_args(%iter = %shmem) -> (tensor<1x32x33xf32>) {
    %0 = vector.extract %xla_loop[%i, %j] : f32 from vector<8x1xf32>
    %inserted = tensor.insert %0 into %iter[%ra, %rb, %rc] : tensor<1x32x33xf32>
    xla_gpu.yield %inserted : tensor<1x32x33xf32>
  }
  %synced_tensor = xla_gpu.sync_threads %xla_loop_0 : tensor<1x32x33xf32>
  return %synced_tensor : tensor<1x32x33xf32>
}

// CHECK-LABEL: @do_not_fuse_map_domain_mismatch
// CHECK: xla_gpu.loop
// CHECK: vector.insert
// CHECK: xla_gpu.loop
// CHECK: vector.extract

// -----

#indexing_map = #xla_gpu.indexing_map<"(d0, d1)[s0, s1] ->"
"   (d1 floordiv 30,"
"   ((d1 floordiv 6) mod 5) * 32 + s0 * 4 + d0 floordiv 32,"
"   (d1 mod 6) * 32 + d0 mod 32),"
" domain:"
"   d0 in [0, 127], d1 in [0, 599],"
"   s0 in [0, 7], s1 in [0, 0],"
"   (d1 mod 6) * 32 + d0 mod 32 in [0, 169]">
#indexing_map1 = #xla_gpu.indexing_map<"(d0, d1)[s0, s1] ->"
"   (0,"
"   d0 mod 32,"
"   d0 floordiv 32 + s0 * 4),"
" domain:"
"   d0 in [0, 127], d1 in [0, 599],"
"   s0 in [0, 7], s1 in [0, 0],"
"   (d1 mod 5) * 32 + d0 mod 32 in [0, 169]">
func.func @do_not_fuse_map_constraint_mismatch(%arg0: tensor<20x160x170xf32>) -> tensor<1x32x33xf32> {
  %cst = arith.constant dense<0.000000e+00> : vector<8x1xf32>
  %c0 = arith.constant 0 : index
  %tid = gpu.thread_id  x {xla.range = [0 : index, 127 : index]}
  %bid = gpu.block_id  x {xla.range = [0 : index, 599 : index]}
  %shmem = xla_gpu.allocate_shared : tensor<1x32x33xf32>
  %xla_loop = xla_gpu.loop (%tid, %bid)[%i, %j]
    -> (%ra, %rb, %rc) in #indexing_map iter_args(%iter = %cst) -> (vector<8x1xf32>) {
    %extracted = tensor.extract %arg0[%ra, %rb, %rc] : tensor<20x160x170xf32>
    %0 = math.exp %extracted : f32
    %1 = vector.insert %0, %iter [%i, %j] : f32 into vector<8x1xf32>
    xla_gpu.yield %1 : vector<8x1xf32>
  }
  %xla_loop_0 = xla_gpu.loop (%tid, %bid)[%i, %j]
    -> (%ra, %rb, %rc) in #indexing_map1 iter_args(%iter = %shmem) -> (tensor<1x32x33xf32>) {
    %0 = vector.extract %xla_loop[%i, %j] : f32 from vector<8x1xf32>
    %inserted = tensor.insert %0 into %iter[%ra, %rb, %rc] : tensor<1x32x33xf32>
    xla_gpu.yield %inserted : tensor<1x32x33xf32>
  }
  %synced_tensor = xla_gpu.sync_threads %xla_loop_0 : tensor<1x32x33xf32>
  return %synced_tensor : tensor<1x32x33xf32>
}

// CHECK-LABEL: @do_not_fuse_map_constraint_mismatch
// CHECK: xla_gpu.loop
// CHECK: vector.insert
// CHECK: xla_gpu.loop
// CHECK: vector.extract

// -----

#indexing_map = #xla_gpu.indexing_map<"(d0, d1)[s0, s1, s2] ->"
"   (d1 floordiv 30,"
"   ((d1 floordiv 6) mod 5) * 32 + s0 * 4 + d0 floordiv 32,"
"   (d1 mod 6) * 32 + d0 mod 32),"
" domain:"
"   d0 in [0, 127], d1 in [0, 599],"
"   s0 in [0, 7], s1 in [0, 0], s2 in [0, 1],"
"   (d1 mod 6) * 32 + d0 mod 32 in [0, 169]">
#indexing_map1 = #xla_gpu.indexing_map<"(d0, d1)[s0, s1, s2] ->"
"   (0,"
"   d0 mod 32,"
"   d0 floordiv 32 + s0 * 4),"
" domain:"
"   d0 in [0, 127], d1 in [0, 599],"
"   s0 in [0, 7], s1 in [0, 0], s2 in [0, 1],"
"   (d1 mod 6) * 32 + d0 mod 32 in [0, 169]">
func.func @do_not_fuse_unused_loop_iv(%arg0: tensor<20x160x170xf32>) -> tensor<1x32x33xf32> {
  %cst = arith.constant dense<0.000000e+00> : vector<8x1xf32>
  %c0 = arith.constant 0 : index
  %tid = gpu.thread_id  x {xla.range = [0 : index, 127 : index]}
  %bid = gpu.block_id  x {xla.range = [0 : index, 599 : index]}
  %shmem = xla_gpu.allocate_shared : tensor<1x32x33xf32>
  %xla_loop = xla_gpu.loop (%tid, %bid)[%i, %j, %k]
    -> (%ra, %rb, %rc) in #indexing_map iter_args(%iter = %cst) -> (vector<8x1xf32>) {
    %extracted = tensor.extract %arg0[%ra, %rb, %rc] : tensor<20x160x170xf32>
    %0 = math.exp %extracted : f32
    %1 = vector.insert %0, %iter [%i, %j] : f32 into vector<8x1xf32>
    xla_gpu.yield %1 : vector<8x1xf32>
  }
  %xla_loop_0 = xla_gpu.loop (%tid, %bid)[%i, %j, %k]
    -> (%ra, %rb, %rc) in #indexing_map1 iter_args(%iter = %shmem) -> (tensor<1x32x33xf32>) {
    %0 = vector.extract %xla_loop[%i, %j] : f32 from vector<8x1xf32>
    %inserted = tensor.insert %0 into %iter[%ra, %rb, %rc] : tensor<1x32x33xf32>
    xla_gpu.yield %inserted : tensor<1x32x33xf32>
  }
  %synced_tensor = xla_gpu.sync_threads %xla_loop_0 : tensor<1x32x33xf32>
  return %synced_tensor : tensor<1x32x33xf32>
}

// CHECK-LABEL: @do_not_fuse_unused_loop_iv
// CHECK: xla_gpu.loop
// CHECK: vector.insert
// CHECK: xla_gpu.loop
// CHECK: vector.extract
