// RUN: triton-opt %s -canonicalize -triton-combine | FileCheck %s

// We don't combine if the dot result is used by more than one op.
// CHECK-LABEL: @test_combine_dot_add_invalid_pattern
tt.func @test_combine_dot_add_invalid_pattern() -> (tensor<128x128xf32>, tensor<128x128xf32>) {
    // CHECK-DAG: %[[d:.*]] = arith.constant dense<3.000000e+00> : tensor<128x128xf32>
    // CHECK-DAG: %[[e:.*]] = arith.constant dense<4.000000e+00> : tensor<128x128xf32>
    %a = arith.constant dense<1.0> : tensor<128x128xf32>
    %b = arith.constant dense<2.0> : tensor<128x128xf32>
    %zero = arith.constant dense<0.0> : tensor<128x128xf32>
    %d = arith.constant dense<3.0> : tensor<128x128xf32>
    %e = arith.constant dense<4.0> : tensor<128x128xf32>

    %dot_out = tt.dot %a, %b, %zero : tensor<128x128xf32> * tensor<128x128xf32> -> tensor<128x128xf32>

    // CHECK: arith.addf %{{.*}}, %[[d]] : tensor<128x128xf32>
    %res0 = arith.addf %dot_out, %d : tensor<128x128xf32>

    // CHECK-NEXT: arith.addf %{{.*}}, %[[e]]  : tensor<128x128xf32>
    %res1 = arith.addf %dot_out, %e : tensor<128x128xf32>

    tt.return %res0, %res1 : tensor<128x128xf32>, tensor<128x128xf32>
}


// CHECK-LABEL: @test_combine_dot_add_pattern
tt.func @test_combine_dot_add_pattern() -> (tensor<128x128xf32>) {
    // CHECK-DAG: %[[d:.*]] = arith.constant dense<3.000000e+00> : tensor<128x128xf32>
    // CHECK-DAG: %[[b:.*]] = arith.constant dense<2.000000e+00> : tensor<128x128xf32>
    // CHECK-DAG: %[[a:.*]] = arith.constant dense<1.000000e+00> : tensor<128x128xf32>
    %a = arith.constant dense<1.0> : tensor<128x128xf32>
    %b = arith.constant dense<2.0> : tensor<128x128xf32>
    %zero = arith.constant dense<0.0> : tensor<128x128xf32>
    %d = arith.constant dense<3.0> : tensor<128x128xf32>

    %dot_out = tt.dot %a, %b, %zero : tensor<128x128xf32> * tensor<128x128xf32> -> tensor<128x128xf32>

    // CHECK-NEXT: %[[res:.*]] = tt.dot %[[a]], %[[b]], %[[d]] : tensor<128x128xf32> * tensor<128x128xf32> -> tensor<128x128xf32>
    // CHECK-NEXT: tt.return %[[res]] : tensor<128x128xf32>
    %res = arith.addf %dot_out, %d : tensor<128x128xf32>

    tt.return %res : tensor<128x128xf32>
}


// CHECK-LABEL: @test_combine_dot_add_rev_pattern
tt.func @test_combine_dot_add_rev_pattern() -> (tensor<128x128xf32>) {
    // CHECK-DAG: %[[d:.*]] = arith.constant dense<3.000000e+00> : tensor<128x128xf32>
    // CHECK-DAG: %[[b:.*]] = arith.constant dense<2.000000e+00> : tensor<128x128xf32>
    // CHECK-DAG: %[[a:.*]] = arith.constant dense<1.000000e+00> : tensor<128x128xf32>
    %a = arith.constant dense<1.0> : tensor<128x128xf32>
    %b = arith.constant dense<2.0> : tensor<128x128xf32>
    %zero = arith.constant dense<0.0> : tensor<128x128xf32>
    %d = arith.constant dense<3.0> : tensor<128x128xf32>

    %dot_out = tt.dot %a, %b, %zero : tensor<128x128xf32> * tensor<128x128xf32> -> tensor<128x128xf32>

    // CHECK-NEXT: %[[res:.*]] = tt.dot %[[a]], %[[b]], %[[d]] : tensor<128x128xf32> * tensor<128x128xf32> -> tensor<128x128xf32>
    // CHECK-NEXT: tt.return %[[res]] : tensor<128x128xf32>
    %res = arith.addf %d, %dot_out : tensor<128x128xf32>

    tt.return %res : tensor<128x128xf32>
}


// CHECK-LABEL: @test_combine_addptr_pattern
tt.func @test_combine_addptr_pattern(%base: !tt.ptr<f32>) -> tensor<8x!tt.ptr<f32>> {
    %off0 = arith.constant 10 : i32
    %off1 = arith.constant 15 : i32

    // CHECK-NEXT: %[[cst:.*]] = arith.constant dense<25> : tensor<8xi32>

    %base_ = tt.splat %base : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>

    // CHECK-NEXT: %[[tmp0:.*]] = tt.splat %{{.*}} : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>

    %idx0 = tt.splat %off0 : i32 -> tensor<8xi32>
    %idx1 = tt.splat %off1 : i32 -> tensor<8xi32>

    // CHECK-NEXT: %1 = tt.addptr %[[tmp0]], %[[cst]] : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %ptr0 = tt.addptr %base_, %idx0 : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %ptr1 = tt.addptr %ptr0, %idx1 : tensor<8x!tt.ptr<f32>>, tensor<8xi32>

    tt.return %ptr1 : tensor<8x!tt.ptr<f32>>
}

// CHECK-LABEL: @test_combine_addptr_pattern_i64
tt.func @test_combine_addptr_pattern_i64(%base: !tt.ptr<f32>) -> tensor<8x!tt.ptr<f32>> {
    %off0 = arith.constant 10 : i64
    %off1 = arith.constant dense<15> : tensor<8xi64>

    // CHECK-NEXT: %[[cst:.*]] = arith.constant dense<25> : tensor<8xi64>

    %base_ = tt.splat %base : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>

    // CHECK-NEXT: %[[tmp0:.*]] = tt.splat %{{.*}} : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>

    %idx0 = tt.splat %off0 : i64 -> tensor<8xi64>

    // CHECK-NEXT: %1 = tt.addptr %[[tmp0]], %[[cst]] : tensor<8x!tt.ptr<f32>>, tensor<8xi64>
    %ptr0 = tt.addptr %base_, %idx0 : tensor<8x!tt.ptr<f32>>, tensor<8xi64>
    %ptr1 = tt.addptr %ptr0, %off1 : tensor<8x!tt.ptr<f32>>, tensor<8xi64>

    tt.return %ptr1 : tensor<8x!tt.ptr<f32>>
}

// CHECK-LABEL: @test_combine_addptr_pattern_scalar
tt.func @test_combine_addptr_pattern_scalar(%base: !tt.ptr<f32>) -> !tt.ptr<f32> {
    %off0 = arith.constant 10 : i32
    %off1 = arith.constant 15 : i32

    // CHECK-NEXT: %[[cst:.*]] = arith.constant 25 : i32
    // CHECK-NEXT: %0 = tt.addptr %{{.*}}, %[[cst]] : !tt.ptr<f32>, i32
    %ptr0 = tt.addptr %base, %off0 : !tt.ptr<f32>, i32
    %ptr1 = tt.addptr %ptr0, %off1 : !tt.ptr<f32>, i32

    tt.return %ptr1 : !tt.ptr<f32>
}

// CHECK-LABEL: @test_not_combine_addptr_pattern_1
tt.func @test_not_combine_addptr_pattern_1(%base: !tt.ptr<f32>, %idx0: tensor<8xi32>) -> tensor<8x!tt.ptr<f32>> {
    %off1 = arith.constant 15 : i32

    %base_ = tt.splat %base : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %idx1 = tt.splat %off1 : i32 -> tensor<8xi32>

    // CHECK: tt.addptr
    // CHECK-NEXT: tt.addptr
    %ptr0 = tt.addptr %base_, %idx0 : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %ptr1 = tt.addptr %ptr0, %idx1 : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    tt.return %ptr1 : tensor<8x!tt.ptr<f32>>
}

// CHECK-LABEL: @test_not_combine_addptr_pattern
tt.func @test_not_combine_addptr_pattern(%base: !tt.ptr<f32>) -> tensor<8x!tt.ptr<f32>> {
    %off0 = arith.constant 10 : i16
    %off1 = arith.constant 15 : i32

    // CHECK-DAG: %[[cst:.*]] = arith.constant dense<10> : tensor<8xi16>
    // CHECK-DAG: %[[cst1:.*]] = arith.constant dense<15> : tensor<8xi32>

    %base_ = tt.splat %base : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>

    %idx0 = tt.splat %off0 : i16 -> tensor<8xi16>
    %idx1 = tt.splat %off1 : i32 -> tensor<8xi32>

    %ptr0 = tt.addptr %base_, %idx0 : tensor<8x!tt.ptr<f32>>, tensor<8xi16>
    %ptr1 = tt.addptr %ptr0, %idx1 : tensor<8x!tt.ptr<f32>>, tensor<8xi32>

    tt.return %ptr1 : tensor<8x!tt.ptr<f32>>
}

// CHECK-LABEL: @test_not_combine_addptr_pattern_overflow
tt.func @test_not_combine_addptr_pattern_overflow(%base: !tt.ptr<f32>) -> tensor<8x!tt.ptr<f32>> {
    %off0 = arith.constant 127 : i8
    %off1 = arith.constant 1 : i8

    // CHECK-DAG: %[[cst:.*]] = arith.constant dense<127> : tensor<8xi8>
    // CHECK-DAG: %[[cst1:.*]] = arith.constant dense<1> : tensor<8xi8>

    %base_ = tt.splat %base : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>

    %idx0 = tt.splat %off0 : i8 -> tensor<8xi8>
    %idx1 = tt.splat %off1 : i8 -> tensor<8xi8>

    %ptr0 = tt.addptr %base_, %idx0 : tensor<8x!tt.ptr<f32>>, tensor<8xi8>
    %ptr1 = tt.addptr %ptr0, %idx1 : tensor<8x!tt.ptr<f32>>, tensor<8xi8>

    tt.return %ptr1 : tensor<8x!tt.ptr<f32>>
}

// CHECK-LABEL: @test_combine_select_masked_load_pattern
tt.func @test_combine_select_masked_load_pattern(%ptr: tensor<8x!tt.ptr<f32>>, %cond: i1) -> (tensor<8xf32>, tensor<8xf32>) {
    %mask = tt.splat %cond : i1 -> tensor<8xi1>
    %false_val = arith.constant dense<0.0> : tensor<8xf32>

    // CHECK: %[[res1:.*]] = tt.load %{{.*}}, %{{.*}}, %{{.*}} : tensor<8x!tt.ptr<f32>>
    %x = tt.load %ptr, %mask, %false_val : tensor<8x!tt.ptr<f32>>
    %0 = arith.select %cond, %x, %false_val : tensor<8xf32>

    // CHECK: %[[res2:.*]] = tt.load %{{.*}}, %{{.*}}, %{{.*}} : tensor<8x!tt.ptr<f32>>
    %y = tt.load %ptr, %mask, %false_val : tensor<8x!tt.ptr<f32>>
    %1 = arith.select %cond, %y, %false_val : tensor<8xf32>

    // CHECK: tt.return %[[res1]], %[[res2]] : tensor<8xf32>, tensor<8xf32>
    tt.return %0, %1 : tensor<8xf32>, tensor<8xf32>
}

// CHECK-LABEL: @test_combine_select_masked_load_fail_pattern
tt.func @test_combine_select_masked_load_fail_pattern(%ptr: tensor<8x!tt.ptr<f32>>, %dummy_load: tensor<8xf32>, %dummy_broadcast: tensor<8xi1>, %cond0: i1, %cond1: i1) -> (tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) {
    %false_val = arith.constant dense<0.0> : tensor<8xf32>

    // Case 1: value at the "load" position is not an "op".  Select should not be canonicalized.
    // CHECK: %{{.*}} = arith.select %{{.*}}, %{{.*}}, %{{.*}} : tensor<8xf32>
    %0 = arith.select %cond0, %dummy_load, %false_val : tensor<8xf32>

    // Case 2: value at the "broadcast" position is not an "op".  Select should not be canonicalized.
    %real_load0 = tt.load %ptr, %dummy_broadcast, %false_val : tensor<8x!tt.ptr<f32>>
    // CHECK: %{{.*}} = arith.select %{{.*}}, %{{.*}}, %{{.*}} : tensor<8xf32>
    %1 = arith.select %cond0, %real_load0, %false_val : tensor<8xf32>

    // Case 3: condition of "broadcast" is not the same as the condition of "select".  Select should not be canonicalized.
    %cond0_ = tt.splat %cond0 : i1 -> tensor<8xi1>
    %real_load1 = tt.load %ptr, %cond0_, %false_val : tensor<8x!tt.ptr<f32>>
    // CHECK: %{{.*}} = arith.select %{{.*}}, %{{.*}}, %{{.*}} : tensor<8xf32>
    %2 = arith.select %cond1, %real_load1, %false_val : tensor<8xf32>

    tt.return %0, %1, %2 : tensor<8xf32>, tensor<8xf32>, tensor<8xf32>
}

// CHECK-LABEL: @test_canonicalize_masked_load_pattern
tt.func @test_canonicalize_masked_load_pattern(%ptr: tensor<8x!tt.ptr<f32>>) -> (tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) {
    %true_mask = arith.constant dense<true> : tensor<8xi1>
    %false_mask = arith.constant dense<false> : tensor<8xi1>
    %other_val = arith.constant dense<0.0> : tensor<8xf32>

    // true_mask with other
    // CHECK: %[[res1:.*]] = tt.load %{{.*}} : tensor<8x!tt.ptr<f32>>
    %x = tt.load %ptr, %true_mask : tensor<8x!tt.ptr<f32>>

    // true_mask without other
    // CHECK: %[[res2:.*]] = tt.load %{{.*}} : tensor<8x!tt.ptr<f32>>
    %y = tt.load %ptr, %true_mask, %other_val : tensor<8x!tt.ptr<f32>>

    // false_mask with other. It should become "other" (i.e., %y)
    %z = tt.load %ptr, %false_mask, %y : tensor<8x!tt.ptr<f32>>

    // CHECK: tt.return %[[res1]], %[[res2]], %[[res2]] : tensor<8xf32>, tensor<8xf32>, tensor<8xf32>
    tt.return %x, %y, %z: tensor<8xf32>, tensor<8xf32>, tensor<8xf32>
}

// CHECK-LABEL: @test_canonicalize_masked_load_fail_pattern
tt.func @test_canonicalize_masked_load_fail_pattern(%ptr: tensor<8x!tt.ptr<f32>>, %mask: tensor<8xi1>) -> (tensor<8xf32>, tensor<8xf32>) {
    %other_val = arith.constant dense<0.0> : tensor<8xf32>

    // Case: value at the "mask" position is not an "op".  Load should not be canonicalized.
    // CHECK: %[[res1:.*]] = tt.load %{{.*}}, %{{.*}} : tensor<8x!tt.ptr<f32>>
    %x = tt.load %ptr, %mask : tensor<8x!tt.ptr<f32>>
    // CHECK: %[[res1:.*]] = tt.load %{{.*}}, %{{.*}}, %{{.*}} : tensor<8x!tt.ptr<f32>>
    %y = tt.load %ptr, %mask, %other_val : tensor<8x!tt.ptr<f32>>

    tt.return %x, %y: tensor<8xf32>, tensor<8xf32>
}

// CHECK-LABEL: @test_canonicalize_masked_store_pattern
tt.func @test_canonicalize_masked_store_pattern(%ptr: tensor<8x!tt.ptr<f32>>, %val: tensor<8xf32>) {
    %true_mask = arith.constant dense<true> : tensor<8xi1>
    %false_mask = arith.constant dense<false> : tensor<8xi1>

    // CHECK: tt.store %{{.*}}, %{{.*}} : tensor<8x!tt.ptr<f32>>
    tt.store %ptr, %val, %true_mask : tensor<8x!tt.ptr<f32>>

    // The following store should disappear.
    // CHECK-NEXT: tt.return
    tt.store %ptr, %val, %false_mask : tensor<8x!tt.ptr<f32>>
    tt.return
}

// CHECK-LABEL: @test_canonicalize_masked_store_fail_pattern
tt.func @test_canonicalize_masked_store_fail_pattern(%ptr: tensor<8x!tt.ptr<f32>>, %val: tensor<8xf32>, %mask: tensor<8xi1>) {
    // Case: value at the "mask" position is not an "op".  Store should not be canonicalized.
    // CHECK: tt.store %{{.*}}, %{{.*}}, %{{.*}} : tensor<8x!tt.ptr<f32>>
    tt.store %ptr, %val, %mask : tensor<8x!tt.ptr<f32>>
    tt.return
}

// CHECK-LABEL: @test_canonicalize_expand_dims
tt.func @test_canonicalize_expand_dims(%arg0: tensor<f32>, %arg1: tensor<1xf32>) -> (tensor<1x8xf32>, tensor<8x8xf32>) {
    %splat = tt.splat %arg0 : tensor<f32> -> tensor<8xf32>
    // CHECK: %{{.*}} = tt.splat %arg0 : tensor<f32> -> tensor<1x8xf32>
    %ed = tt.expand_dims %splat {axis = 0 : i32} : tensor<8xf32> -> tensor<1x8xf32>

    // CHECK-NEXT: %[[ed2:.*]] = tt.expand_dims %arg1 {axis = 0 : i32} : tensor<1xf32> -> tensor<1x1xf32>
    // CHECK-NEXT: %{{.*}} = tt.broadcast %[[ed2]] : tensor<1x1xf32> -> tensor<8x8xf32>
    %bc = tt.broadcast %arg1 : tensor<1xf32> -> tensor<8xf32>
    %ed2 = tt.expand_dims %bc {axis = 0 : i32} : tensor<8xf32> -> tensor<1x8xf32>
    %bc2 = tt.broadcast %ed2 : tensor<1x8xf32> -> tensor<8x8xf32>

    tt.return %ed, %bc2 : tensor<1x8xf32>, tensor<8x8xf32>
}

// CHECK-LABEL: @test_canonicalize_view
tt.func @test_canonicalize_view(%arg0: tensor<8xf32>, %arg1: tensor<f32>) -> (tensor<4x2xf32>, tensor<2x2x2xf32>, tensor<8xf32>, tensor<2x2x2xf32>) {
    %view0 = tt.reshape %arg0 allow_reorder : tensor<8xf32> -> tensor<2x4xf32>
    // CHECK: %{{.*}} = tt.reshape %arg0 allow_reorder : tensor<8xf32> -> tensor<4x2xf32>
    %view1 = tt.reshape %view0 allow_reorder : tensor<2x4xf32> -> tensor<4x2xf32>

    %splat = tt.splat %arg1 : tensor<f32> -> tensor<8xf32>
    // CHECK: %{{.*}} = tt.splat %arg1 : tensor<f32> -> tensor<2x2x2xf32>
    %view2 = tt.reshape %splat allow_reorder : tensor<8xf32> -> tensor<2x2x2xf32>

    %view3 = tt.reshape %arg0 : tensor<8xf32> -> tensor<8xf32>
    // CHECK: %{{.*}} = arith.addf %arg0, %arg0 : tensor<8xf32>
    %add = arith.addf %view3, %arg0 : tensor<8xf32>

    // CHECK: %{{.*}} = tt.reshape %arg0 allow_reorder : tensor<8xf32> -> tensor<2x2x2xf32>
    %reshape = tt.reshape %view0 : tensor<2x4xf32> -> tensor<2x2x2xf32>

    tt.return %view1, %view2, %add, %reshape : tensor<4x2xf32>, tensor<2x2x2xf32>, tensor<8xf32>, tensor<2x2x2xf32>
}

// CHECK-LABEL: @test_canonicalize_reshape
tt.func @test_canonicalize_reshape(%arg0: tensor<8xf32>, %arg1: tensor<f32>) -> (tensor<4x2xf32>, tensor<2x2x2xf32>, tensor<8xf32>, tensor<2x2x2xf32>) {
    %reshape0 = tt.reshape %arg0 : tensor<8xf32> -> tensor<2x4xf32>
    // CHECK: %{{.*}} = tt.reshape %arg0 : tensor<8xf32> -> tensor<4x2xf32>
    %reshape1 = tt.reshape %reshape0 : tensor<2x4xf32> -> tensor<4x2xf32>

    %splat = tt.splat %arg1 : tensor<f32> -> tensor<8xf32>
    // CHECK: %{{.*}} = tt.splat %arg1 : tensor<f32> -> tensor<2x2x2xf32>
    %reshape2 = tt.reshape %splat : tensor<8xf32> -> tensor<2x2x2xf32>

    %reshape3 = tt.reshape %arg0 : tensor<8xf32> -> tensor<8xf32>
    // CHECK: %{{.*}} = arith.addf %arg0, %arg0 : tensor<8xf32>
    %add = arith.addf %reshape3, %arg0 : tensor<8xf32>

    // CHECK: %{{.*}} = tt.reshape %arg0 allow_reorder : tensor<8xf32> -> tensor<2x2x2xf32>
    %view = tt.reshape %reshape0 allow_reorder : tensor<2x4xf32> -> tensor<2x2x2xf32>

    tt.return %reshape1, %reshape2, %add, %view : tensor<4x2xf32>, tensor<2x2x2xf32>, tensor<8xf32>, tensor<2x2x2xf32>
}

// CHECK-LABEL: @test_canonicalize_broadcast
tt.func @test_canonicalize_broadcast(%arg0: tensor<1x1x8xf32>, %arg1: tensor<f32>) -> (tensor<4x2x8xf32>, tensor<8x8xf32>, tensor<1x1x8xf32>) {
    %broadcast0 = tt.broadcast %arg0 : tensor<1x1x8xf32> -> tensor<1x2x8xf32>
    // CHECK: %{{.*}} = tt.broadcast %arg0 : tensor<1x1x8xf32> -> tensor<4x2x8xf32>
    %broadcast1 = tt.broadcast %broadcast0 : tensor<1x2x8xf32> -> tensor<4x2x8xf32>

    %splat = tt.splat %arg1 : tensor<f32> -> tensor<1x8xf32>
    // CHECK: %{{.*}} = tt.splat %arg1 : tensor<f32> -> tensor<8x8xf32>
    %broadcast2 = tt.broadcast %splat : tensor<1x8xf32> -> tensor<8x8xf32>

    %broadcast3 = tt.broadcast %arg0 : tensor<1x1x8xf32> -> tensor<1x1x8xf32>
    // CHECK: %{{.*}} = arith.addf %arg0, %arg0 : tensor<1x1x8xf32>
    %add = arith.addf %broadcast3, %arg0 : tensor<1x1x8xf32>

    tt.return %broadcast1, %broadcast2, %add : tensor<4x2x8xf32>, tensor<8x8xf32>, tensor<1x1x8xf32>
}

// CHECK-LABEL: @test_fold_views
tt.func @test_fold_views() -> (tensor<16x8xf32>, tensor<16x128xf32>, tensor<1x1x128xf32>) {
    %a = arith.constant dense<1.0> : tensor<1x128xf32>

    // CHECK-DAG: %{{.*}} = arith.constant dense<1.{{.*}}> : tensor<16x8xf32>
    %b = tt.reshape %a allow_reorder : tensor<1x128xf32> -> tensor<16x8xf32>

    // CHECK-DAG: %{{.*}} = arith.constant dense<1.{{.*}}> : tensor<16x128xf32>
    %c = tt.broadcast %a : tensor<1x128xf32> -> tensor<16x128xf32>

    // CHECK-DAG: %{{.*}} = arith.constant dense<1.{{.*}}> : tensor<1x1x128xf32>
    %d = tt.expand_dims %a {axis = 0: i32} : tensor<1x128xf32> -> tensor<1x1x128xf32>

    tt.return %b, %c, %d : tensor<16x8xf32>, tensor<16x128xf32>, tensor<1x1x128xf32>
}

// CHECK-LABEL: @test_nop_transpose
tt.func @test_nop_transpose(%arg0: tensor<2x4xf32>) -> (tensor<2x4xf32>) {
    %a = tt.trans %arg0 {order = array<i32: 0, 1>} : tensor<2x4xf32> -> tensor<2x4xf32>
    // CHECK: tt.return %arg0
    tt.return %a : tensor<2x4xf32>
}

// CHECK-LABEL: @test_nested_transpose
tt.func @test_nested_transpose(%arg0: tensor<2x4x8xf32>) -> (tensor<8x2x4xf32>) {
    %a = tt.trans %arg0 {order = array<i32: 1, 0, 2>} : tensor<2x4x8xf32> -> tensor<4x2x8xf32>
    %b = tt.trans %a {order = array<i32: 2, 1, 0>} : tensor<4x2x8xf32> -> tensor<8x2x4xf32>
    // CHECK: %[[res:.*]] = tt.trans %arg0 {order = array<i32: 2, 0, 1>}
    // CHECK: tt.return %[[res]]
    tt.return %b : tensor<8x2x4xf32>
}

// CHECK-LABEL: test_reshape_reduce
tt.func @test_reshape_reduce(%0: tensor<32x4x2xi32>) -> (i32, tensor<16xi32>) {
  // CHECK: tt.reshape %{{.+}} allow_reorder : tensor<32x4x2xi32> -> tensor<256xi32>
  %1 = tt.reshape %0 : tensor<32x4x2xi32> -> tensor<256xi32>
  %2 = "tt.reduce" (%1) ({
    ^bb0(%arg7: i32, %arg8: i32):
      %add = arith.addi %arg7, %arg8 : i32
      tt.reduce.return %add : i32
    }) {axis = 0 : i32} : (tensor<256xi32>) -> i32
  %3 = tt.histogram %1 : tensor<256xi32> -> tensor<16xi32>
  tt.return %2, %3 : i32, tensor<16xi32>
}

// CHECK-LABEL: test_rank_reduce_desc_load
tt.func @test_rank_reduce_desc_load(%0: !tt.tensordesc<tensor<1x128x64xf16>>) -> (tensor<128x64xf16>) {
  %c0 = arith.constant 0 : i32
  // CHECK: %[[R:.+]] = tt.descriptor_load {{.*}} : !tt.tensordesc<tensor<1x128x64xf16>> -> tensor<128x64xf16>
  // CHECK: tt.return %[[R]]
  %l = tt.descriptor_load %0[%c0, %c0, %c0] : !tt.tensordesc<tensor<1x128x64xf16>> -> tensor<1x128x64xf16>
  %r = tt.reshape %l : tensor<1x128x64xf16> -> tensor<128x64xf16>
  tt.return %r :  tensor<128x64xf16>
}
