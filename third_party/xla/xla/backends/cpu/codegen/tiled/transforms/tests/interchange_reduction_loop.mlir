// RUN: fusion_compiler_opt %s --xtile-cpu-interchange-reduction-loop --canonicalize --cse --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @interchange_static_bounds
func.func @interchange_static_bounds(
    %arg0: memref<1152x13824xf32>,
    %arg1: memref<1x13824xf32>) {
  %cst = arith.constant dense<0.000000e+00> : vector<8xf32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c144 = arith.constant 144 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %tid = %c0 to %c144 step %c1 {
    %col = arith.muli %tid, %c8 : index
    %res = scf.for %k = %c0 to %c144 step %c1
        iter_args(%acc = %cst) -> (vector<8xf32>) {
      %val = vector.transfer_read %arg0[%k, %col], %cst_0
          {in_bounds = [true]} : memref<1152x13824xf32>, vector<8xf32>
      %add = arith.addf %acc, %val : vector<8xf32>
      scf.yield %add : vector<8xf32>
    }
    vector.transfer_write %res, %arg1[%c0, %col]
        {in_bounds = [true]} : vector<8xf32>, memref<1x13824xf32>
  }
  return
}
// CHECK: %[[SCRATCH:.*]] = memref.alloca() alignment = 64 : memref<144x8xf32>
// CHECK: scf.for %[[TID_INIT:.*]] = %c0 to %c144 step %c1 {
// CHECK:   vector.store %cst, %[[SCRATCH]][%[[TID_INIT]], %c0]
// CHECK: scf.for %[[K:.*]] = %c0 to %c144 step %c1 {
// CHECK:   scf.for %[[TID_IN:.*]] = %c0 to %c144 step %c1 {
// CHECK:     %[[ACC:.*]] = vector.load %[[SCRATCH]][%[[TID_IN]], %c0]
// CHECK:     %[[COL:.*]] = arith.muli %[[TID_IN]], %c8 : index
// CHECK:     vector.transfer_read %arg0[%[[K]], %[[COL]]]
// CHECK:     %[[ADD:.*]] = arith.addf %[[ACC]], %{{.*}} : vector<8xf32>
// CHECK:     vector.store %[[ADD]], %[[SCRATCH]][%[[TID_IN]], %c0]
// CHECK: scf.for %[[TID_OUT:.*]] = %c0 to %c144 step %c1 {
// CHECK:   %[[VAL:.*]] = vector.load %[[SCRATCH]][%[[TID_OUT]], %c0]
// CHECK:   %[[COL_OUT:.*]] = arith.muli %[[TID_OUT]], %c8 : index
// CHECK:   vector.transfer_write %[[VAL]], %arg1[%c0, %[[COL_OUT]]]

// -----

#indexing_map = #xla.indexing_map<
    "(pid) -> (min(pid * -2731 + 32768, 2731)), domain: pid in [0, 11]">

// CHECK-LABEL: func.func @interchange_dynamic_bounds
func.func @interchange_dynamic_bounds(
    %arg0: memref<1152x262144xf32>,
    %arg1: memref<1x262144xf32>,
    %arg2: index {xla.range = [0 : index, 11 : index]}) {
  %cst = arith.constant dense<0.000000e+00> : vector<8xf32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c144 = arith.constant 144 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %ub = xla.apply_indexing #indexing_map(%arg2)
  scf.for %tid = %c0 to %ub step %c1 {
    %res = scf.for %k = %c0 to %c144 step %c1
        iter_args(%acc = %cst) -> (vector<8xf32>) {
      %val = vector.transfer_read %arg0[%k, %tid], %cst_0
          {in_bounds = [true]} : memref<1152x262144xf32>, vector<8xf32>
      %add = arith.addf %acc, %val : vector<8xf32>
      scf.yield %add : vector<8xf32>
    }
    vector.transfer_write %res, %arg1[%c0, %tid]
        {in_bounds = [true]} : vector<8xf32>, memref<1x262144xf32>
  }
  return
}
// CHECK: %[[UB:.*]] = xla.apply_indexing
// CHECK: %[[SCRATCH:.*]] = memref.alloca() alignment = 64 : memref<2731x8xf32>
// CHECK: scf.for %[[TID_INIT:.*]] = %c0 to %[[UB]] step %c1 {
// CHECK:   vector.store %cst, %[[SCRATCH]][%[[TID_INIT]], %c0]
// CHECK: scf.for %[[K:.*]] = %c0 to %c144 step %c1 {
// CHECK:   scf.for %[[TID_IN:.*]] = %c0 to %[[UB]] step %c1 {
// CHECK:     %[[ACC:.*]] = vector.load %[[SCRATCH]][%[[TID_IN]], %c0]
// CHECK:     vector.transfer_read %arg0[%[[K]], %[[TID_IN]]]
// CHECK:     %[[ADD:.*]] = arith.addf %[[ACC]], %{{.*}} : vector<8xf32>
// CHECK:     vector.store %[[ADD]], %[[SCRATCH]][%[[TID_IN]], %c0]
// CHECK: scf.for %[[TID_OUT:.*]] = %c0 to %[[UB]] step %c1 {
// CHECK:   %[[VAL:.*]] = vector.load %[[SCRATCH]][%[[TID_OUT]], %c0]
// CHECK:   vector.transfer_write %[[VAL]], %arg1[%c0, %[[TID_OUT]]]

// -----

// K-minor matrix: outer loop is NOT interchanged.
// CHECK-LABEL: func.func @no_interchange_k_minor
func.func @no_interchange_k_minor(
    %arg0: memref<13824x1152xf32>,
    %arg1: memref<1x13824xf32>) {
  %cst = arith.constant dense<0.000000e+00> : vector<8xf32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c144 = arith.constant 144 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %tid = %c0 to %c144 step %c1 {
    %res = scf.for %k = %c0 to %c144 step %c1
        iter_args(%acc = %cst) -> (vector<8xf32>) {
      %val = vector.transfer_read %arg0[%tid, %k], %cst_0
          {in_bounds = [true]} : memref<13824x1152xf32>, vector<8xf32>
      %add = arith.addf %acc, %val : vector<8xf32>
      scf.yield %add : vector<8xf32>
    }
    vector.transfer_write %res, %arg1[%c0, %tid]
        {in_bounds = [true]} : vector<8xf32>, memref<1x13824xf32>
  }
  return
}
// CHECK-NOT: memref.alloca

// -----

// CHECK-LABEL: func.func @interchange_nonzero_lb
func.func @interchange_nonzero_lb(
    %arg0: memref<16x64xf32>,
    %arg1: memref<1x64xf32>) {
  %cst = arith.constant dense<0.000000e+00> : vector<8xf32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c12 = arith.constant 12 : index
  scf.for %tid = %c4 to %c12 step %c1 {
    %res = scf.for %k = %c0 to %c8 step %c1
        iter_args(%acc = %cst) -> (vector<8xf32>) {
      %val = vector.transfer_read %arg0[%k, %tid], %cst_0
          {in_bounds = [true]} : memref<16x64xf32>, vector<8xf32>
      %add = arith.addf %acc, %val : vector<8xf32>
      scf.yield %add : vector<8xf32>
    }
    vector.transfer_write %res, %arg1[%c0, %tid]
        {in_bounds = [true]} : vector<8xf32>, memref<1x64xf32>
  }
  return
}
// CHECK: %[[SCRATCH:.*]] = memref.alloca() alignment = 64 : memref<8x8xf32>
// CHECK: scf.for %[[TID:.*]] = %c4 to %c12 step %c1 {
// CHECK:   %[[IDX:.*]] = arith.subi %[[TID]], %c4 : index
// CHECK:   vector.store %cst, %[[SCRATCH]][%[[IDX]], %c0]

// -----

// CHECK-LABEL: func.func @no_interchange_aliased_write
func.func @no_interchange_aliased_write(%arg0: memref<16x16xf32>) {
  %cst = arith.constant dense<0.000000e+00> : vector<8xf32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %view = memref.cast %arg0 : memref<16x16xf32> to memref<?x16xf32>
  scf.for %tid = %c0 to %c8 step %c1 {
    %res = scf.for %k = %c0 to %c8 step %c1
        iter_args(%acc = %cst) -> (vector<8xf32>) {
      %val = vector.transfer_read %view[%k, %tid], %cst_0
          {in_bounds = [true]} : memref<?x16xf32>, vector<8xf32>
      %add = arith.addf %acc, %val : vector<8xf32>
      scf.yield %add : vector<8xf32>
    }
    vector.transfer_write %res, %arg0[%c0, %tid]
        {in_bounds = [true]} : vector<8xf32>, memref<16x16xf32>
  }
  return
}
// CHECK-NOT: memref.alloca

// -----

// CHECK-LABEL: func.func @no_interchange_outer_variant_bound
func.func @no_interchange_outer_variant_bound(
    %arg0: memref<16x16xf32>,
    %arg1: memref<1x16xf32>) {
  %cst = arith.constant dense<0.000000e+00> : vector<8xf32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  scf.for %tid = %c0 to %c8 step %c1 {
    %ub = arith.addi %tid, %c2 : index
    %res = scf.for %k = %c0 to %ub step %c1
        iter_args(%acc = %cst) -> (vector<8xf32>) {
      %val = vector.transfer_read %arg0[%k, %tid], %cst_0
          {in_bounds = [true]} : memref<16x16xf32>, vector<8xf32>
      %add = arith.addf %acc, %val : vector<8xf32>
      scf.yield %add : vector<8xf32>
    }
    vector.transfer_write %res, %arg1[%c0, %tid]
        {in_bounds = [true]} : vector<8xf32>, memref<1x16xf32>
  }
  return
}
// CHECK-NOT: memref.alloca

// -----

// CHECK-LABEL: func.func @no_interchange_large_scratch
func.func @no_interchange_large_scratch(
    %arg0: memref<16x8388608xf32>,
    %arg1: memref<1x8388608xf32>) {
  %cst = arith.constant dense<0.000000e+00> : vector<8xf32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c1048576 = arith.constant 1048576 : index
  scf.for %tid = %c0 to %c1048576 step %c1 {
    %res = scf.for %k = %c0 to %c16 step %c1
        iter_args(%acc = %cst) -> (vector<8xf32>) {
      %val = vector.transfer_read %arg0[%k, %tid], %cst_0
          {in_bounds = [true]} : memref<16x8388608xf32>, vector<8xf32>
      %add = arith.addf %acc, %val : vector<8xf32>
      scf.yield %add : vector<8xf32>
    }
    vector.transfer_write %res, %arg1[%c0, %tid]
        {in_bounds = [true]} : vector<8xf32>, memref<1x8388608xf32>
  }
  return
}
// CHECK-NOT: memref.alloca

// -----

// CHECK-LABEL: func.func @no_interchange_scalable_acc
func.func @no_interchange_scalable_acc(
    %arg0: memref<16x64xf32>,
    %arg1: memref<1x64xf32>) {
  %cst = arith.constant dense<0.000000e+00> : vector<[8]xf32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %tid = %c0 to %c8 step %c1 {
    %res = scf.for %k = %c0 to %c8 step %c1
        iter_args(%acc = %cst) -> (vector<[8]xf32>) {
      %val = vector.transfer_read %arg0[%k, %tid], %cst_0
          {in_bounds = [true]} : memref<16x64xf32>, vector<[8]xf32>
      %add = arith.addf %acc, %val : vector<[8]xf32>
      scf.yield %add : vector<[8]xf32>
    }
    vector.transfer_write %res, %arg1[%c0, %tid]
        {in_bounds = [true]} : vector<[8]xf32>, memref<1x64xf32>
  }
  return
}
// CHECK-NOT: memref.alloca


// -----

// CHECK-LABEL: func.func @no_interchange_post_op_alloc
func.func @no_interchange_post_op_alloc(
    %arg0: memref<16x64xf32>,
    %arg1: memref<1x64xf32>) {
  %cst = arith.constant dense<0.000000e+00> : vector<8xf32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %tid = %c0 to %c8 step %c1 {
    %res = scf.for %k = %c0 to %c8 step %c1
        iter_args(%acc = %cst) -> (vector<8xf32>) {
      %val = vector.transfer_read %arg0[%k, %tid], %cst_0
          {in_bounds = [true]} : memref<16x64xf32>, vector<8xf32>
      %add = arith.addf %acc, %val : vector<8xf32>
      scf.yield %add : vector<8xf32>
    }
    %tmp = memref.alloc() : memref<8xf32>
    memref.dealloc %tmp : memref<8xf32>
    vector.transfer_write %res, %arg1[%c0, %tid]
        {in_bounds = [true]} : vector<8xf32>, memref<1x64xf32>
  }
  return
}
// CHECK-NOT: memref.alloca
