// RUN: triton-opt %s | FileCheck %s

// CHECK-LABEL: @cast_ops
tt.func @cast_ops(%scalar_ptr: !tt.ptr<f32>, %scalar_f32: f32, %scalar_i64: i64) {
  // scalar -> scalar
  // CHECK:  i64 -> !tt.ptr<f32>
  %0 = tt.int_to_ptr %scalar_i64 : i64 -> !tt.ptr<f32>
  // CHECK: !tt.ptr<f32> -> i64
  %1 = tt.ptr_to_int %scalar_ptr : !tt.ptr<f32> -> i64
  // CHECK: f32 to f16
  %2 = arith.truncf %scalar_f32 : f32 to f16

  // 0D tensor -> 0D tensor
  %tensor_ptr_0d = tt.splat %scalar_ptr : !tt.ptr<f32> -> tensor<!tt.ptr<f32>>
  %tensor_f32_0d = tt.splat %scalar_f32 : f32 -> tensor<f32>
  %tensor_i64_0d = tt.splat %scalar_i64 : i64 -> tensor<i64>

  // CHECK: tensor<i64> -> tensor<!tt.ptr<f32>>
  %3 = tt.int_to_ptr %tensor_i64_0d : tensor<i64> -> tensor<!tt.ptr<f32>>
  // CHECK: tensor<!tt.ptr<f32>> -> tensor<i64>
  %4 = tt.ptr_to_int %tensor_ptr_0d : tensor<!tt.ptr<f32>> -> tensor<i64>
  // CHECK: tensor<f32> to tensor<f16>
  %5 = arith.truncf %tensor_f32_0d : tensor<f32> to tensor<f16>

  // 1D tensor -> 1D tensor
  %tensor_ptr_1d = tt.splat %scalar_ptr : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
  %tensor_f32_1d = tt.splat %scalar_f32 : f32 -> tensor<16xf32>
  %tensor_i64_1d = tt.splat %scalar_i64 : i64 -> tensor<16xi64>

  // CHECK: tensor<16xi64> -> tensor<16x!tt.ptr<f32>>
  %6 = tt.int_to_ptr %tensor_i64_1d : tensor<16xi64> -> tensor<16x!tt.ptr<f32>>
  // CHECK: tensor<16x!tt.ptr<f32>> -> tensor<16xi64>
  %7 = tt.ptr_to_int %tensor_ptr_1d : tensor<16x!tt.ptr<f32>> -> tensor<16xi64>
  // CHECK: tensor<16xf32> to tensor<16xf16>
  %8 = arith.truncf %tensor_f32_1d : tensor<16xf32> to tensor<16xf16>
  tt.return
}

// CHECK-LABEL: @addptr_ops
tt.func @addptr_ops(%scalar_ptr: !tt.ptr<f32>, %scalar_i32: i32) {
  // scalar -> scalar
  // CHECK: !tt.ptr<f32>
  %0 = tt.addptr %scalar_ptr, %scalar_i32 : !tt.ptr<f32>, i32

  // 0D tensor -> 0D tensor
  %tensor_ptr_0d = tt.splat %scalar_ptr : !tt.ptr<f32> -> tensor<!tt.ptr<f32>>
  %tensor_i32_0d = tt.splat %scalar_i32 : i32 -> tensor<i32>
  // CHECK: tensor<!tt.ptr<f32>>
  %1 = tt.addptr %tensor_ptr_0d, %tensor_i32_0d : tensor<!tt.ptr<f32>>, tensor<i32>

  // 1D tensor -> 1D tensor
  %tensor_ptr_1d = tt.splat %scalar_ptr : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
  %tensor_i32_1d = tt.splat %scalar_i32 : i32 -> tensor<16xi32>
  // CHECK: tensor<16x!tt.ptr<f32>>
  %2 = tt.addptr %tensor_ptr_1d, %tensor_i32_1d : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
  tt.return
}

// CHECK-LABEL: @load_store_ops_scalar
tt.func @load_store_ops_scalar(%ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %mask : i1) {
  // Test if Load/Store ops can handle scalar values
  %other = arith.constant 0.0e+0 : f32

  // load scalar
  // CHECK: %[[L0:.*]] = tt.load %{{.*}} : !tt.ptr<f32>
  %a = tt.load %ptr : !tt.ptr<f32>
  // CHECK: %[[L1:.*]] = tt.load %{{.*}}, %{{.*}} : !tt.ptr<f32>
  %b = tt.load %ptr, %mask : !tt.ptr<f32>
  // CHECK: %[[L2:.*]] = tt.load %{{.*}}, %{{.*}}, %{{.*}} : !tt.ptr<f32>
  %c = tt.load %ptr, %mask, %other : !tt.ptr<f32>

  // store scalar
  // CHECK: tt.store %{{.*}}, %[[L0]] : !tt.ptr<f32>
  tt.store %ptr, %a : !tt.ptr<f32>
  // CHECK: tt.store %{{.*}}, %[[L1]], %{{.*}} : !tt.ptr<f32>
  tt.store %ptr, %b, %mask : !tt.ptr<f32>
  // CHECK: tt.store %{{.*}}, %[[L2]], %{{.*}} : !tt.ptr<f32>
  tt.store %ptr, %c, %mask : !tt.ptr<f32>
  tt.return
}

// CHECK-LABEL: reduce_ops_infer
tt.func @reduce_ops_infer(%ptr: !tt.ptr<f32>, %v : tensor<1x2x4xf32>) {
  // Test if reduce ops infer types correctly

  // CHECK: tt.reduce
  // CHECK-SAME: axis = 0
  // CHECK: tt.reduce.return
  // CHECK-NEXT: (tensor<1x2x4xf32>) -> tensor<2x4xf32>
  %a = "tt.reduce" (%v) ({
  ^bb0(%arg0: f32, %arg1: f32):
    %add = arith.addf %arg0, %arg1 : f32
    tt.reduce.return %add : f32
  }) {axis = 0 : i32}  : (tensor<1x2x4xf32>) -> tensor<2x4xf32>

  // CHECK: tt.reduce
  // CHECK-SAME: axis = 1
  // CHECK: tt.reduce.return
  // CHECK-NEXT: (tensor<1x2x4xf32>) -> tensor<1x4xf32>
  %b = "tt.reduce" (%v) ({
  ^bb0(%arg0: f32, %arg1: f32):
    %add = arith.addf %arg0, %arg1 : f32
    tt.reduce.return %add : f32
  }) {axis = 1 : i32}  : (tensor<1x2x4xf32>) -> tensor<1x4xf32>

  // CHECK: tt.reduce
  // CHECK-SAME: axis = 2
  // CHECK: tt.reduce.return
  // CHECK-NEXT: (tensor<1x2x4xf32>) -> tensor<1x2xf32>
  %c = "tt.reduce" (%v) ({
  ^bb0(%arg0: f32, %arg1: f32):
    %add = arith.addf %arg0, %arg1 : f32
    tt.reduce.return %add : f32
  }) {axis = 2 : i32}  : (tensor<1x2x4xf32>) -> tensor<1x2xf32>

  // CHECK: tt.reduce
  // CHECK-SAME: axis = 1
  // CHECK: tt.reduce.return
  // CHECK-NEXT: (tensor<1x4xf32>) -> tensor<1xf32>
  %e = "tt.reduce" (%b) ({
  ^bb0(%arg0: f32, %arg1: f32):
    %add = arith.addf %arg0, %arg1 : f32
    tt.reduce.return %add : f32
  }) {axis = 1 : i32}  : (tensor<1x4xf32>) -> tensor<1xf32>

  // CHECK: tt.reduce
  // CHECK-SAME: axis = 0
  // CHECK: tt.reduce.return
  // CHECK-NEXT: (tensor<2x4xf32>) -> tensor<4xf32>
  %f = "tt.reduce" (%a) ({
  ^bb0(%arg0: f32, %arg1: f32):
    %add = arith.addf %arg0, %arg1 : f32
    tt.reduce.return %add : f32
  }) {axis = 0 : i32}  : (tensor<2x4xf32>) -> tensor<4xf32>

  // CHECK: tt.reduce
  // CHECK-SAME: axis = 0
  // CHECK: tt.reduce.return
  // CHECK-NEXT: (tensor<4xf32>) -> f32
  %g = "tt.reduce" (%f) ({
  ^bb0(%arg0: f32, %arg1: f32):
    %add = arith.addf %arg0, %arg1 : f32
    tt.reduce.return %add : f32
  }) {axis = 0 : i32}  : (tensor<4xf32>) -> f32

  // Avoid optimizations for c, e, and g
  %ptr1x2 = tt.splat %ptr : !tt.ptr<f32> -> tensor<1x2x!tt.ptr<f32>>
  %ptr1 = tt.splat %ptr : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>>
  tt.store %ptr1x2, %c : tensor<1x2x!tt.ptr<f32>>
  tt.store %ptr1, %e : tensor<1x!tt.ptr<f32>>
  tt.store %ptr, %g : !tt.ptr<f32>
  tt.return
}

// CHECK-LABEL: @dot_ops_infer
tt.func @dot_ops_infer(%ptr: !tt.ptr<f32>, %v : f32) {
  // Test if reduce ops infer types correctly
  %v128x32 = tt.splat %v : f32 -> tensor<128x32xf32>
  %v32x128 = tt.splat %v : f32 -> tensor<32x128xf32>
  %v128x1 = tt.splat %v : f32 -> tensor<128x1xf32>
  %v1x128 = tt.splat %v : f32 -> tensor<1x128xf32>

  %zero128x128 = arith.constant dense<0.00e+00> : tensor<128x128xf32>
  %zero32x32 = arith.constant dense<0.00e+00> : tensor<32x32xf32>
  %zero1x1 = arith.constant dense<0.00e+00> : tensor<1x1xf32>

  // CHECK: %{{.*}} = tt.dot %{{.*}} -> tensor<128x128xf32>
  %r1 = tt.dot %v128x32, %v32x128, %zero128x128 : tensor<128x32xf32> * tensor<32x128xf32> -> tensor<128x128xf32>
  // CHECK: %{{.*}} = tt.dot %{{.*}} -> tensor<32x32xf32>
  %r2 = tt.dot %v32x128, %v128x32, %zero32x32 : tensor<32x128xf32> * tensor<128x32xf32> -> tensor<32x32xf32>
  // CHECK: %{{.*}} = tt.dot %{{.*}} -> tensor<128x128xf32>
  %r3 = tt.dot %v128x1, %v1x128, %zero128x128 : tensor<128x1xf32> * tensor<1x128xf32> -> tensor<128x128xf32>
  // CHECK: %{{.*}} = tt.dot %{{.*}} -> tensor<1x1xf32>
  %r4 = tt.dot %v1x128, %v128x1, %zero1x1 : tensor<1x128xf32> * tensor<128x1xf32> -> tensor<1x1xf32>

  %ptr128x128 = tt.splat %ptr : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>>
  %ptr32x32 = tt.splat %ptr : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>>
  %ptr1x1 = tt.splat %ptr : !tt.ptr<f32> -> tensor<1x1x!tt.ptr<f32>>
  tt.store %ptr128x128, %r1 : tensor<128x128x!tt.ptr<f32>>
  tt.store %ptr32x32, %r2 : tensor<32x32x!tt.ptr<f32>>
  tt.store %ptr128x128, %r3 : tensor<128x128x!tt.ptr<f32>>
  tt.store %ptr1x1, %r4 : tensor<1x1x!tt.ptr<f32>>
  tt.return
}

// CHECK-LABEL: @print_no_arg
tt.func @print_no_arg(%arg0: !tt.ptr<f32>) {
// CHECK: tt.print "test"
  tt.print "test" { hex = false, isSigned = array<i32: 0>}
  %0 = tt.load %arg0 : !tt.ptr<f32>
  tt.store %arg0, %0 : !tt.ptr<f32>
  tt.return
}

// CHECK-LABEL: scan_op
tt.func @scan_op(%ptr: tensor<1x2x4x!tt.ptr<f32>>, %v : tensor<1x2x4xf32>) {
  // CHECK: tt.scan
  // CHECK-SAME: axis = 1
  // CHECK: tt.scan.return
  // CHECK-NEXT: (tensor<1x2x4xf32>) -> tensor<1x2x4xf32>
  %a = "tt.scan"(%v) <{axis = 1 : i32, reverse = false}>({
  ^bb0(%arg0: f32, %arg1: f32):
    %add = arith.addf %arg0, %arg1 : f32
    tt.scan.return %add : f32
  }) : (tensor<1x2x4xf32>) -> tensor<1x2x4xf32>
  tt.store %ptr, %a : tensor<1x2x4x!tt.ptr<f32>>
  tt.return
}

// CHECK-LABEL: inline_asm
// CHECK: tt.elementwise_inline_asm "shl.b32 $0, $0, 3;"
tt.func @inline_asm(%0: tensor<512xi8>) {
  %1 = tt.elementwise_inline_asm "shl.b32 $0, $0, 3;"
    {constraints = "=r,r", packed_element = 4 : i32, pure = true} %0 : tensor<512xi8> -> tensor<512xi8>
  tt.return
}

// CHECK-LABEL: inline_asm_scalar
// CHECK: tt.elementwise_inline_asm "shl.b32 $0, $0, 3;" {{.*}} : i32 -> i32
tt.func @inline_asm_scalar(%0: i32) {
  %1 = tt.elementwise_inline_asm "shl.b32 $0, $0, 3;"
    {constraints = "=r,r", packed_element = 1 : i32, pure = true} %0 : i32 -> i32
  tt.return
}

// CHECK-LABEL: reshape
tt.func @reshape(%0: tensor<512xi32>) {
  // CHECK: tt.reshape %{{.+}} : tensor<512xi32> -> tensor<16x32xi32>
  %1 = tt.reshape %0 : tensor<512xi32> -> tensor<16x32xi32>
  // CHECK: tt.reshape %{{.+}} allow_reorder : tensor<512xi32> -> tensor<16x32xi32>
  %2 = tt.reshape %0 allow_reorder : tensor<512xi32> -> tensor<16x32xi32>
  // CHECK: tt.reshape %{{.+}} allow_reorder efficient_layout : tensor<512xi32> -> tensor<16x32xi32>
  %3 = tt.reshape %0 allow_reorder efficient_layout : tensor<512xi32> -> tensor<16x32xi32>
  // CHECK: tt.reshape %{{.+}} efficient_layout : tensor<512xi32> -> tensor<16x32xi32>
  %4 = tt.reshape %0 efficient_layout : tensor<512xi32> -> tensor<16x32xi32>
  tt.return
}

// CHECK-LABEL: histogram
tt.func @histogram(%0: tensor<512xi32>) {
  // CHECK: tt.histogram %{{.+}} : tensor<512xi32> -> tensor<16xi32>
  %1 = tt.histogram %0 : tensor<512xi32> -> tensor<16xi32>
  tt.return
}

// CHECK-LABEL: descriptor_load
tt.func @descriptor_load(%0: !tt.tensordesc<tensor<128xf32>>) {
  // CHECK: tt.descriptor_load %{{.+}}[%{{.+}}] : !tt.tensordesc<tensor<128xf32>> -> tensor<128xf32>
  %c0_i32 = arith.constant 0 : i32
  %1 = tt.descriptor_load %0[%c0_i32] : !tt.tensordesc<tensor<128xf32>> -> tensor<128xf32>
  tt.return
}

// CHECK-LABEL: @gather_op
tt.func @gather_op(%arg0: tensor<128x16xf32>, %arg1: tensor<512x16xi32>) -> tensor<512x16xf32> {
  // CHECK-NEXT: %0 = tt.gather %arg0[%arg1] {axis = 0 : i32} : (tensor<128x16xf32>, tensor<512x16xi32>) -> tensor<512x16xf32>
  %0 = tt.gather %arg0[%arg1] {axis = 0 : i32} : (tensor<128x16xf32>, tensor<512x16xi32>) -> tensor<512x16xf32>
  tt.return %0 : tensor<512x16xf32>
}

// CHECK-LABEL: @tma_gather
tt.func @tma_gather(%arg0: !tt.tensordesc<tensor<1x128xbf16>>, %arg1: tensor<32xi32>, %arg2: i32) {
  // CHECK-NEXT: %0 = tt.descriptor_gather %arg0[%arg1, %arg2] : (!tt.tensordesc<tensor<1x128xbf16>>, tensor<32xi32>, i32) -> tensor<32x128xbf16>
  %0 = tt.descriptor_gather %arg0[%arg1, %arg2] : (!tt.tensordesc<tensor<1x128xbf16>>, tensor<32xi32>, i32) -> tensor<32x128xbf16>
  tt.return
}

// CHECK-LABEL: @tma_scatter
tt.func @tma_scatter(%arg0: !tt.tensordesc<tensor<1x128xbf16>>, %arg1: tensor<32xi32>, %arg2: i32, %arg3: tensor<32x128xbf16>) {
  // CHECK-NEXT: tt.descriptor_scatter %arg0[%arg1, %arg2], %arg3 : !tt.tensordesc<tensor<1x128xbf16>>, tensor<32xi32>, i32, tensor<32x128xbf16>
  tt.descriptor_scatter %arg0[%arg1, %arg2], %arg3 : !tt.tensordesc<tensor<1x128xbf16>>, tensor<32xi32>, i32, tensor<32x128xbf16>
  tt.return
}
