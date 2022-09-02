// RUN: xla-runtime-opt %s  --split-input-file --xla-rt-to-llvm | FileCheck %s

// CHECK: func @pass_context(
// CHECK:   %[[CTX:.*]]: !llvm.ptr<i8>
// CHECK: )
func.func @pass_context(%arg0: !rt.kernel_context) {
  func.return
}

// -----

// CHECK: func @set_output(
// CHECK:   %[[CTX:.*]]: !llvm.ptr<i8>
// CHECK: )
func.func @set_output(%arg0: !rt.kernel_context) {
  // CHECK: %[[MEMREF:.*]] = memref.alloc
  // CHECK: %[[LLVM_MEMREF:.*]] = builtin.unrealized_conversion_cast %[[MEMREF]]
  %0 = memref.alloc() : memref<f32>
  // CHECK: %[[C0:.*]] = arith.constant 0 : i64
  // CHECK: %[[RES_PTR:.*]] = call @runtimeGetResultStorage(%[[CTX]], %[[C0]])
  // CHECK: %[[LLVM_PTR:.*]] = llvm.bitcast %[[RES_PTR]]
  // CHECK: llvm.store %[[LLVM_MEMREF]], %[[LLVM_PTR]]
  rt.set_output %arg0, 0, %0 : memref<f32>
  func.return
}

// -----

// CHECK-DAG: llvm.mlir.global {{.*}} @[[ERR0:.*]]("Failed precondition #0\00")
// CHECK-DAG: llvm.mlir.global {{.*}} @[[ERR1:.*]]("Failed precondition #1\00")

// CHECK: func @set_error(
// CHECK:   %[[CTX:.*]]: !llvm.ptr<i8>
// CHECK: )
func.func @set_error(%arg0: !rt.kernel_context) {
  // CHECK: %[[ADDR0:.*]] = llvm.mlir.addressof @[[ERR0]]
  // CHECK: %[[PTR0:.*]] = llvm.bitcast %[[ADDR0]] {{.*}} to !llvm.ptr<i8>
  // CHECK: call @runtimeSetError(%[[CTX]], %[[PTR0]])
  rt.set_error %arg0, "Failed precondition #0"
  // CHECK: %[[ADDR1:.*]] = llvm.mlir.addressof @[[ERR1]]
  // CHECK: %[[PTR1:.*]] = llvm.bitcast %[[ADDR1]] {{.*}} to !llvm.ptr<i8>
  // CHECK: call @runtimeSetError(%[[CTX]], %[[PTR1]])
  rt.set_error %arg0, "Failed precondition #1"
  func.return
}

// -----

// CHECK: llvm.mlir.global {{.*}} @[[ERR:.*]]("Failed precondition\00")
// CHECK-NOT: Failed precondition

// CHECK: func @dedup_error_message(
// CHECK:   %[[CTX:.*]]: !llvm.ptr<i8>
// CHECK: )
func.func @dedup_error_message(%arg0: !rt.kernel_context) {
  // CHECK: %[[ADDR:.*]] = llvm.mlir.addressof @[[ERR]]
  rt.set_error %arg0, "Failed precondition"
  // CHECK: %[[ADDR:.*]] = llvm.mlir.addressof @[[ERR]]
  rt.set_error %arg0, "Failed precondition"
  func.return
}

// -----

// CHECK: global internal constant @__rt_num_attrs(1 : i64) {{.*}}: i64

// CHECK: global internal constant @__rt_attr_value()
// CHECK-SAME: !llvm.array<3 x i64> {
// CHECK:   llvm.mlir.undef : !llvm.array<3 x i64>
// CHECK:   arith.constant 1 : i64
// CHECK:   llvm.insertvalue
// CHECK:   arith.constant 2 : i64
// CHECK:   llvm.insertvalue
// CHECK:   arith.constant 3 : i64
// CHECK:   llvm.insertvalue
// CHECK:   llvm.return
// CHECK: }

// CHECK: global internal constant @__rt_attr_value_0()
// CHECK-SAME: !llvm.struct<(i64, ptr<array<3 x i64>>)> {
// CHECK:   arith.constant 3 : i64
// CHECK:   llvm.mlir.addressof @__rt_attr_value : !llvm.ptr<array<3 x i64>>
// CHECK:   llvm.mlir.undef : !llvm.struct<(i64, ptr<array<3 x i64>>)>
// CHECK:   llvm.insertvalue
// CHECK:   llvm.insertvalue
// CHECK:   llvm.return
// CHECK: }

// CHECK: func @custom_call(
// CHECK:   %[[CTX:.*]]: !llvm.ptr<i8>
// CHECK: )
func.func @custom_call(%arg0: !rt.kernel_context) {
  // CHECK: call @runtimeCustomCall
  rt.custom_call %arg0["target"] () { arr = [1, 2, 3] } : () -> ()
  func.return
}

// -----

// CHECK: global internal constant @__rt_num_attrs(1 : i64) {{.*}}: i64

// CHECK: global internal constant @__rt_attr_value()
// CHECK-SAME: : !llvm.array<3 x i64> {
// CHECK:   llvm.mlir.undef : !llvm.array<3 x i64>
// CHECK:   arith.constant 1 : i64
// CHECK:   llvm.insertvalue
// CHECK:   arith.constant 2 : i64
// CHECK:   llvm.insertvalue
// CHECK:   arith.constant 3 : i64
// CHECK:   llvm.insertvalue
// CHECK: }

// CHECK: global internal constant @__rt_attr_value_0()
// CHECK-SAME: !llvm.struct<(i64, ptr<array<3 x i64>>)> {
// CHECK    arith.constant 3 : i64
// CHECK    llvm.mlir.addressof @__rt_attr_value
// CHECK    llvm.mlir.undef : !llvm.struct<(i64, ptr<array<3 x i64>>)>
// CHECK    llvm.mlir.insertvalue
// CHECK    llvm.mlir.insertvalue
// CHECK: }

// CHECK: func @custom_call(
// CHECK:   %[[CTX:.*]]: !llvm.ptr<i8>
// CHECK: )
func.func @custom_call(%arg0: !rt.kernel_context) {
  // CHECK: call @runtimeCustomCall
  rt.custom_call %arg0["target"] ()
    { attr_name = array<i64: 1, 2, 3> } : () -> ()
  func.return
}

// -----

// CHECK: global internal constant @__rt_num_attrs(1 : i64)

// CHECK: global internal constant @__rt_attr_value()
// CHECK-SAME: !llvm.struct<(i64, ptr<i8>)> {
// CHECK:    arith.constant 0 : i64
// CHECK:    llvm.mlir.null : !llvm.ptr<i8>
// CHECK:    llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
// CHECK:    llvm.insertvalue
// CHECK:    llvm.insertvalue
// CHECK: }

// CHECK: func @custom_call(
// CHECK:   %[[CTX:.*]]: !llvm.ptr<i8>
// CHECK: )
func.func @custom_call(%arg0: !rt.kernel_context) {
  // CHECK: call @runtimeCustomCall
  rt.custom_call %arg0["target"] () { arr = [] } : () -> ()
  func.return
}

// -----

// CHECK: global internal constant @__rt_custom_call_callee("target\00")
// CHECK: global internal constant @__rt_num_attrs(0 : i64)

// CHECK: global internal constant @__rt_custom_call_attrs()
// CHECK: {
// CHECK:   llvm.mlir.undef : !llvm.array<1 x ptr<i8>>
// CHECK:   llvm.mlir.addressof @__rt_num_attrs : !llvm.ptr<i64>
// CHECK: }

// CHECK: global internal constant @__rt_num_args(0 : i64)

// CHECK: func @custom_call(
// CHECK:   %[[CTX:.*]]: !llvm.ptr<i8>
// CHECK: )
func.func @custom_call(%arg0: !rt.kernel_context) {

  // CHECK: %[[C1:.*]] = arith.constant 1 : i32
  // CHECK: %[[RETS_ALLOCA:.*]] = llvm.alloca %[[C1]] x !llvm.array<1 x ptr<i8>>

  // CHECK: %[[C1_0:.*]] = arith.constant 1 : i32
  // CHECK: %[[ARGS_ALLOCA:.*]] = llvm.alloca %[[C1_0]] x !llvm.array<1 x ptr<i8>>
  // CHECK: %[[ARGS:.*]] = llvm.getelementptr %[[ARGS_ALLOCA]]

  // CHECK: %[[ATTRS_ADDR:.*]] = llvm.mlir.addressof @__rt_custom_call_attrs
  // CHECK: %[[ATTRS:.*]] = llvm.getelementptr %[[ATTRS_ADDR]]
  
  // CHECK: %[[RETS:.*]] = llvm.getelementptr %[[RETS_ALLOCA]]

  // CHECK: %[[CALLEE_ADDR:.*]] = llvm.mlir.addressof @__rt_custom_call_callee
  // CHECK: %[[CALLEE:.*]] = llvm.bitcast %[[CALLEE_ADDR]]

  // CHECK: %[[STATUS:.*]] = call @runtimeCustomCall(%[[CTX]], %[[CALLEE]],
  // CHECK-SAME:                                     %[[ARGS]], %[[ATTRS]], 
  // CHECK-SAME:                                     %[[RETS]])
  // CHECK: cf.assert %[[STATUS]], "oops"
  %status = rt.custom_call %arg0["target"] () : () -> ()
  %ok = rt.is_ok %status
  cf.assert %ok, "oops"
  func.return
}

// -----

// CHECK: global internal constant @__rt_num_attrs(1 : i64)
// CHECK: global internal constant @__rt_attr_value(1.230000e+02 : f32)
// CHECK: global internal constant @__rt_str("attr_name\00")

// CHECK: global internal constant @__rt_attr_name()
// CHECK-SAME: : !llvm.struct<(i64, ptr<array<10 x i8>>)> {
// CHECK:   arith.constant 9 : i64
// CHECK:   llvm.mlir.addressof @__rt_str : !llvm.ptr<array<10 x i8>>
// CHECK: }

// CHECK: global internal constant @__rt_custom_call_attrs()
// CHECK-SAME: : !llvm.array<4 x ptr<i8>> {
// CHECK:   llvm.mlir.addressof @__rt_attr_name
// CHECK:   llvm.mlir.addressof @__type_id_float
// CHECK:   llvm.mlir.addressof @__rt_attr_value : !llvm.ptr<f32>
// CHECK: }

// CHECK: func @custom_call(
// CHECK:   %[[CTX:.*]]: !llvm.ptr<i8>
// CHECK: )
func.func @custom_call(%arg0: !rt.kernel_context) {
  // CHECK: call @runtimeCustomCall
  rt.custom_call %arg0["target"] () { attr_name = 123.0 : f32 } : () -> ()
  func.return
}

// -----

// CHECK: global internal constant @__rt_num_attrs(1 : i64)

// CHECK:   llvm.mlir.global internal constant @__rt_attr_value
// CHECK-SAME: (dense<[1, 2, 3]> : tensor<3xi32>)

// CHECK:   llvm.mlir.global internal constant @__rt_attr_value_0()
// CHECK-SAME: : !llvm.struct
// CHECK-SAME: <(struct<(i64, ptr<array<3 x i32>>)>, i64, array<1 x i64>)> {
// CHECK:   arith.constant 3 : i64
// CHECK:   llvm.mlir.addressof
// CHECK:   llvm.mlir.undef : !llvm.struct<(i64, ptr<array<3 x i32>>)>
// CHECK:   llvm.insertvalue
// CHECK:   llvm.insertvalue
// CHECK:   arith.constant 1 : i64
// CHECK:   llvm.mlir.undef : !llvm.array<1 x i64>
// CHECK:   arith.constant 3 : i64
// CHECK:   llvm.insertvalue
// CHECK:   llvm.mlir.undef : !llvm.struct
// CHECK-SAME: <(struct<(i64, ptr<array<3 x i32>>)>, i64, array<1 x i64>)>
// CHECK:   llvm.insertvalue
// CHECK:   llvm.insertvalue
// CHECK:   llvm.insertvalue
// CHECK: }

// CHECK: func @custom_call(
// CHECK:   %[[CTX:.*]]: !llvm.ptr<i8>
// CHECK: )
func.func @custom_call(%arg0: !rt.kernel_context) {
  // CHECK: call @runtimeCustomCall
  rt.custom_call %arg0["target"] ()
    { attr_name = dense<[1, 2, 3]> : tensor<3xi32> } : () -> ()
  func.return
}

// -----

// CHECK: global internal constant @__rt_num_attrs(1 : i64)

// CHECK:   llvm.mlir.global internal constant @__rt_attr_value
// CHECK-SAME: (dense<[1, 2]> : tensor<2xi32>)

// CHECK:   llvm.mlir.global internal constant @__rt_attr_value_0()
// CHECK-SAME: : !llvm.struct
// CHECK-SAME: <(struct<(i64, ptr<array<2 x i32>>)>, i64, array<2 x i64>)> {
// CHECK:   arith.constant 2 : i64
// CHECK:   llvm.mlir.addressof
// CHECK:   llvm.mlir.undef : !llvm.struct<(i64, ptr<array<2 x i32>>)>
// CHECK:   llvm.insertvalue
// CHECK:   llvm.insertvalue
// CHECK:   arith.constant 2 : i64
// CHECK:   llvm.mlir.undef : !llvm.array<2 x i64>
// CHECK:   arith.constant 2 : i64
// CHECK:   llvm.insertvalue
// CHECK:   arith.constant 1 : i64
// CHECK:   llvm.insertvalue
// CHECK:   llvm.mlir.undef : !llvm.struct
// CHECK-SAME: <(struct<(i64, ptr<array<2 x i32>>)>, i64, array<2 x i64>)>
// CHECK:   llvm.insertvalue
// CHECK:   llvm.insertvalue
// CHECK:   llvm.insertvalue
// CHECK: }

// CHECK: func @custom_call(
// CHECK:   %[[CTX:.*]]: !llvm.ptr<i8>
// CHECK: )
func.func @custom_call(%arg0: !rt.kernel_context) {
  // CHECK: call @runtimeCustomCall
  rt.custom_call %arg0["target"] ()
    { attr_name = dense<[[1], [2]]> : tensor<2x1xi32> } : () -> ()
  func.return
}

// -----

// CHECK: global internal constant @__rt_num_attrs(1 : i64)
// CHECK: global internal constant @[[STR:.*]]("attr_value\00")

// CHECK: global internal constant @__rt_attr_value()
// CHECK-SAME: : !llvm.struct<(i64, ptr<array<11 x i8>>)> {
// CHECK:   arith.constant 10 : i64
// CHECK:   llvm.mlir.addressof @[[STR]] : !llvm.ptr<array<11 x i8>>
// CHECK: }

// CHECK: func @custom_call(
// CHECK:   %[[CTX:.*]]: !llvm.ptr<i8>
// CHECK: )
func.func @custom_call(%arg0: !rt.kernel_context) {
  // CHECK: call @runtimeCustomCall
  rt.custom_call %arg0["target"] () { attr_name = "attr_value" } : () -> ()
  func.return
}

// -----

// CHECK: func @custom_call(
// CHECK:   %[[CTX:.*]]: !llvm.ptr<i8>,
// CHECK:   %[[ARG:.*]]: f32
// CHECK: )
func.func @custom_call(%arg0: !rt.kernel_context, %arg1 : f32) {
  // CHECK-DAG: %[[MEM:.*]] = llvm.alloca {{.*}} x f32
  // CHECK-DAG: %[[ARGS:.*]] = llvm.alloca {{.*}} x !llvm.array<3 x ptr<i8>

  // CHECK-DAG: %[[TYPE_ID:.*]] = llvm.mlir.addressof @__type_id_float
  // CHECK-DAG: %[[N_ARGS:.*]] = llvm.mlir.addressof @__rt_num_args

  // CHECK-DAG: llvm.store %[[ARG]], %[[MEM]]
  // CHECK-DAG: llvm.store {{.*}}, %[[ARGS]] : !llvm.ptr<array<3 x ptr<i8>>>

  // CHECK: call @runtimeCustomCall
  rt.custom_call %arg0["target"] (%arg1) : (f32) -> ()
  func.return
}

// -----

// CHECK: func @custom_call(
// CHECK:   %[[CTX:.*]]: !llvm.ptr<i8>,
// CHECK:   %[[ARG:.*]]: memref<?x256xf32>
// CHECK: )
func.func @custom_call(%arg0: !rt.kernel_context, %arg1 : memref<?x256xf32>) {

  // CHECK: %[[DESC:.*]] = builtin.unrealized_conversion_cast %[[ARG]]
  // CHECK-SAME: to !llvm.struct

  // CHECK: %[[TYPE_ID:.*]] = llvm.mlir.addressof @__type_id_memref_view

  // CHECK: llvm.mlir.undef : !llvm.array<4 x i64>
  // CHECK-NEXT: llvm.extractvalue %[[DESC]][3, 0]
  // CHECK-NEXT: arith.constant 256 : i64
  // CHECK-NEXT: llvm.insertvalue
  // CHECK-NEXT: llvm.insertvalue
  // CHECK-NEXT: arith.constant 256 : i64
  // CHECK-NEXT: arith.constant 1 : i64
  // CHECK-NEXT: llvm.insertvalue
  // CHECK-NEXT: %[[SIZES:.*]] = llvm.insertvalue

  // llvm.mlir.undef : !llvm.struct<(i8, i8, ptr<i8>, array<2 x i64>)>
  // CHECK: llvm.insertvalue
  // CHECK: llvm.insertvalue
  // CHECK: llvm.insertvalue %[[SIZES]]
  // CHECK: llvm.insertvalue

  // CHECK: %[[N_ARGS:.*]] = llvm.mlir.addressof @__rt_num_args

  // CHECK: call @runtimeCustomCall
  rt.custom_call %arg0["target"] (%arg1) : (memref<?x256xf32>) -> ()
  func.return
}

// -----

// CHECK: func @custom_call(
// CHECK:   %[[CTX:.*]]: !llvm.ptr<i8>,
// CHECK:   %[[ARG:.*]]: f16
// CHECK: )
func.func @custom_call(%arg0: !rt.kernel_context, %arg1: f16) {
  // CHECK: call @target
  rt.custom_call direct %arg0["target"] (%arg1) : (f16) -> ()
  func.return
}

// -----

// CHECK: internal constant @__rt_custom_call_attrs() {{.*}}: !llvm.array<4 x ptr<i8>>
// CHECK-NOT: internal constant @__rt_custom_call_attrs

// CHECK: func @dedup_custom_call_attrs(
// CHECK:   %[[CTX:.*]]: !llvm.ptr<i8>
// CHECK: )
func.func @dedup_custom_call_attrs(%arg0: !rt.kernel_context) {
  // CHECK: call @runtimeCustomCall
  rt.custom_call %arg0["target"] () { arr = [1, 2, 3] } : () -> ()
  // CHECK: call @runtimeCustomCall
  rt.custom_call %arg0["target"] () { arr = [1, 2, 3] } : () -> ()
  func.return
}

// -----

// CHECK: func @direct_custom_call(
// CHECK:   %[[CTX:.*]]: !llvm.ptr<i8>
// CHECK: )
func.func @direct_custom_call(%arg0: !rt.kernel_context) {
  // CHECK: call @target
  // CHECK: call @target
  rt.custom_call direct %arg0["target"] () : () -> ()
  rt.custom_call direct %arg0["target"] () : () -> ()
  func.return
}

// CHECK: func private @target(!llvm.ptr<i8>, !llvm.ptr<ptr<i8>>,
// CHECK-SAME:                 !llvm.ptr<ptr<i8>>) -> i1

// -----

// CHECK: %[[C1:.*]] = arith.constant 1 : i32
// CHECK: %[[RETS_ALLOCA:.*]] = llvm.alloca %[[C1]] x !llvm.array<3 x ptr<i8>>

// CHECK: %[[C1_0:.*]] = arith.constant 1 : i32
// CHECK: %[[F32_ALLOCA:.*]] = llvm.alloca %[[C1_0]] x f32

// CHECK: %[[N_RETS:.*]]  = llvm.mlir.addressof @__rt_num_rets
// CHECK: %[[RETS:.*]] = llvm.getelementptr %[[RETS_ALLOCA]]


// CHECK: call @runtimeCustomCall
// CHECK: %[[LOAD2:.*]] = llvm.load %[[F32_ALLOCA]]
func.func @custom_call(%ctx: !rt.kernel_context,
                       %input: memref<?xf32>) -> (f32) {
  %status, %0 = rt.custom_call %ctx["f32_reduce"] (%input)
                : (memref<?xf32>) -> (f32)
  %ok = rt.is_ok %status
  return %0 : f32
}
