// RUN: xla-runtime-opt %s  --split-input-file --xla-rt-to-llvm | FileCheck %s

// CHECK: func @pass_context(
// CHECK:   %[[CTX:.*]]: !llvm.ptr
// CHECK: )
func.func @pass_context(%arg0: !rt.execution_context) {
  func.return
}

// -----

// CHECK: func @set_output(
// CHECK:   %[[CTX:.*]]: !llvm.ptr
// CHECK: )
func.func @set_output(%arg0: !rt.execution_context) {
  // CHECK: %[[MEMREF:.*]] = memref.alloc
  // CHECK: %[[LLVM_MEMREF:.*]] = builtin.unrealized_conversion_cast %[[MEMREF]]
  %0 = memref.alloc() : memref<f32>
  // CHECK: %[[C0:.*]] = arith.constant 0 : i64
  // CHECK: %[[RES_PTR:.*]] = call @runtimeGetResultStorage(%[[CTX]], %[[C0]])
  // CHECK: llvm.store %[[LLVM_MEMREF]], %[[RES_PTR]]
  rt.set_output %arg0, 0, %0 : memref<f32>
  func.return
}

// -----

// CHECK-DAG: llvm.mlir.global {{.*}} @[[ERR0:.*]]("Failed precondition #0\00")
// CHECK-DAG: llvm.mlir.global {{.*}} @[[ERR1:.*]]("Failed precondition #1\00")

// CHECK: func @set_error(
// CHECK:   %[[CTX:.*]]: !llvm.ptr
// CHECK: )
func.func @set_error(%arg0: !rt.execution_context) {
  // CHECK: %[[ADDR0:.*]] = llvm.mlir.addressof @[[ERR0]]
  // CHECK: call @runtimeSetError(%[[CTX]], %[[ADDR0]])
  rt.set_error %arg0, "Failed precondition #0"
  // CHECK: %[[ADDR1:.*]] = llvm.mlir.addressof @[[ERR1]]
  // CHECK: call @runtimeSetError(%[[CTX]], %[[ADDR1]])
  rt.set_error %arg0, "Failed precondition #1"
  func.return
}

// -----

// CHECK: llvm.mlir.global {{.*}} @[[ERR:.*]]("Failed precondition\00")
// CHECK-NOT: Failed precondition

// CHECK: func @dedup_error_message(
// CHECK:   %[[CTX:.*]]: !llvm.ptr
// CHECK: )
func.func @dedup_error_message(%arg0: !rt.execution_context) {
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
// CHECK-SAME: !llvm.struct<(i64, ptr)> {
// CHECK:   arith.constant 3 : i64
// CHECK:   llvm.mlir.addressof @__rt_attr_value : !llvm.ptr
// CHECK:   llvm.mlir.undef : !llvm.struct<(i64, ptr)>
// CHECK:   llvm.insertvalue
// CHECK:   llvm.insertvalue
// CHECK:   llvm.return
// CHECK: }

// CHECK: func @custom_call(
// CHECK:   %[[CTX:.*]]: !llvm.ptr
// CHECK: )
func.func @custom_call(%arg0: !rt.execution_context) {
  // CHECK: call @target
  rt.call %arg0["target"] () { arr = [1, 2, 3] } : () -> ()
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
// CHECK-SAME: !llvm.struct<(i64, ptr)> {
// CHECK    arith.constant 3 : i64
// CHECK    llvm.mlir.addressof @__rt_attr_value
// CHECK    llvm.mlir.undef : !llvm.struct<(i64, ptr)>
// CHECK    llvm.mlir.insertvalue
// CHECK    llvm.mlir.insertvalue
// CHECK: }

// CHECK: func @custom_call(
// CHECK:   %[[CTX:.*]]: !llvm.ptr
// CHECK: )
func.func @custom_call(%arg0: !rt.execution_context) {
  // CHECK: call @target
  rt.call %arg0["target"] ()
    { attr_name = array<i64: 1, 2, 3> } : () -> ()
  func.return
}

// -----

// CHECK: global internal constant @__rt_num_attrs(1 : i64)

// CHECK: global internal constant @__rt_attr_value()
// CHECK-SAME: !llvm.struct<(i64, ptr)> {
// CHECK:    arith.constant 0 : i64
// CHECK:    llvm.mlir.null : !llvm.ptr
// CHECK:    llvm.mlir.undef : !llvm.struct<(i64, ptr)>
// CHECK:    llvm.insertvalue
// CHECK:    llvm.insertvalue
// CHECK: }

// CHECK: func @custom_call(
// CHECK:   %[[CTX:.*]]: !llvm.ptr
// CHECK: )
func.func @custom_call(%arg0: !rt.execution_context) {
  // CHECK: call @target
  rt.call %arg0["target"] () { arr = [] } : () -> ()
  func.return
}

// -----

// CHECK: global internal constant @__rt_custom_call_name("target\00")
// CHECK: global internal constant @__rt_num_attrs(0 : i64)

// CHECK: global internal constant @__rt_custom_call_attrs()
// CHECK: {
// CHECK:   llvm.mlir.undef : !llvm.array<1 x ptr>
// CHECK:   llvm.mlir.addressof @__rt_num_attrs : !llvm.ptr
// CHECK: }

// CHECK: global internal constant @__rt_num_args(0 : i64)

// CHECK: func @dynamic_custom_call(
// CHECK:   %[[CTX:.*]]: !llvm.ptr
// CHECK: )
func.func @dynamic_custom_call(%arg0: !rt.execution_context) {

  // CHECK: %[[C1:.*]] = arith.constant 1 : i32
  // CHECK: %[[RETS:.*]] = llvm.alloca %[[C1]] x !llvm.array<1 x ptr>

  // CHECK: %[[C1_0:.*]] = arith.constant 1 : i32
  // CHECK: %[[ARGS:.*]] = llvm.alloca %[[C1_0]] x !llvm.array<1 x ptr>

  // CHECK: %[[CALLEE_ADDR:.*]] = llvm.mlir.addressof @__rt_custom_call_name
  // CHECK: %[[ATTRS:.*]] = llvm.mlir.addressof @__rt_custom_call_attrs

  // CHECK: %[[STATUS:.*]] = call @runtimeCustomCall(%[[CTX]], %[[CALLEE_ADDR]],
  // CHECK-SAME:                                     %[[ARGS]], %[[ATTRS]],
  // CHECK-SAME:                                     %[[RETS]])
  // CHECK: cf.assert %[[STATUS]], "oops"
  %status = rt.call dynamic %arg0["target"] () : () -> ()
  %ok = rt.is_ok %status
  cf.assert %ok, "oops"
  func.return
}

// -----

// CHECK: global internal constant @__rt_num_attrs(1 : i64)
// CHECK: global internal constant @__rt_attr_value(1.230000e+02 : f32)
// CHECK: global internal constant @__rt_str("attr_name\00")

// CHECK: global internal constant @__rt_attr_name()
// CHECK-SAME: : !llvm.struct<(i64, ptr)> {
// CHECK:   arith.constant 9 : i64
// CHECK:   llvm.mlir.addressof @__rt_str : !llvm.ptr
// CHECK: }

// CHECK: global internal constant @__rt_custom_call_attrs()
// CHECK-SAME: : !llvm.array<4 x ptr> {
// CHECK:   llvm.mlir.addressof @__rt_num_attrs
// CHECK:   llvm.mlir.addressof @__rt_attr_name
// CHECK:   llvm.mlir.addressof @__type_id_float
// CHECK:   llvm.mlir.addressof @__rt_attr_value
// CHECK: }

// CHECK: func @custom_call(
// CHECK:   %[[CTX:.*]]: !llvm.ptr
// CHECK: )
func.func @custom_call(%arg0: !rt.execution_context) {
  // CHECK: call @target
  rt.call %arg0["target"] () { attr_name = 123.0 : f32 } : () -> ()
  func.return
}

// -----

// CHECK: global internal constant @__rt_num_attrs(1 : i64)

// CHECK:   llvm.mlir.global internal constant @__rt_attr_value
// CHECK-SAME: (dense<[1, 2, 3]> : tensor<3xi32>)

// CHECK:   llvm.mlir.global internal constant @__rt_attr_value_0()
// CHECK-SAME: : !llvm.struct
// CHECK-SAME: <(struct<(i64, ptr)>, i64, array<1 x i64>)> {
// CHECK:   arith.constant 3 : i64
// CHECK:   llvm.mlir.addressof
// CHECK:   llvm.mlir.undef : !llvm.struct<(i64, ptr)>
// CHECK:   llvm.insertvalue
// CHECK:   llvm.insertvalue
// CHECK:   arith.constant 1 : i64
// CHECK:   llvm.mlir.undef : !llvm.array<1 x i64>
// CHECK:   arith.constant 3 : i64
// CHECK:   llvm.insertvalue
// CHECK:   llvm.mlir.undef : !llvm.struct
// CHECK-SAME: <(struct<(i64, ptr)>, i64, array<1 x i64>)>
// CHECK:   llvm.insertvalue
// CHECK:   llvm.insertvalue
// CHECK:   llvm.insertvalue
// CHECK: }

// CHECK: func @custom_call(
// CHECK:   %[[CTX:.*]]: !llvm.ptr
// CHECK: )
func.func @custom_call(%arg0: !rt.execution_context) {
  // CHECK: call @target
  rt.call %arg0["target"] ()
    { attr_name = dense<[1, 2, 3]> : tensor<3xi32> } : () -> ()
  func.return
}

// -----

// CHECK: global internal constant @__rt_num_attrs(1 : i64)

// CHECK:   llvm.mlir.global internal constant @__rt_attr_value
// CHECK-SAME: (dense<[1, 2]> : tensor<2xi32>)

// CHECK:   llvm.mlir.global internal constant @__rt_attr_value_0()
// CHECK-SAME: : !llvm.struct
// CHECK-SAME: <(struct<(i64, ptr)>, i64, array<2 x i64>)> {
// CHECK:   arith.constant 2 : i64
// CHECK:   llvm.mlir.addressof
// CHECK:   llvm.mlir.undef : !llvm.struct<(i64, ptr)>
// CHECK:   llvm.insertvalue
// CHECK:   llvm.insertvalue
// CHECK:   arith.constant 2 : i64
// CHECK:   llvm.mlir.undef : !llvm.array<2 x i64>
// CHECK:   arith.constant 2 : i64
// CHECK:   llvm.insertvalue
// CHECK:   arith.constant 1 : i64
// CHECK:   llvm.insertvalue
// CHECK:   llvm.mlir.undef : !llvm.struct
// CHECK-SAME: <(struct<(i64, ptr)>, i64, array<2 x i64>)>
// CHECK:   llvm.insertvalue
// CHECK:   llvm.insertvalue
// CHECK:   llvm.insertvalue
// CHECK: }

// CHECK: func @custom_call(
// CHECK:   %[[CTX:.*]]: !llvm.ptr
// CHECK: )
func.func @custom_call(%arg0: !rt.execution_context) {
  // CHECK: call @target
  rt.call %arg0["target"] ()
    { attr_name = dense<[[1], [2]]> : tensor<2x1xi32> } : () -> ()
  func.return
}

// -----

// CHECK: global internal constant @__rt_num_attrs(1 : i64)
// CHECK: global internal constant @[[STR:.*]]("attr_value\00")

// CHECK: global internal constant @__rt_attr_value()
// CHECK-SAME: : !llvm.struct<(i64, ptr)> {
// CHECK:   arith.constant 10 : i64
// CHECK:   llvm.mlir.addressof @[[STR]] : !llvm.ptr
// CHECK: }

// CHECK: func @custom_call(
// CHECK:   %[[CTX:.*]]: !llvm.ptr
// CHECK: )
func.func @custom_call(%arg0: !rt.execution_context) {
  // CHECK: call @target
  rt.call %arg0["target"] () { attr_name = "attr_value" } : () -> ()
  func.return
}

// -----

// CHECK: llvm.mlir.global internal constant @__rt_rets_type_table
// CHECK: llvm.mlir.undef : !llvm.array<0 x ptr>
// CHECK: llvm.return {{.*}} : !llvm.array<0 x ptr>
// CHECK: llvm.mlir.global internal constant @__rt_num_rets(0 : i64)

// CHECK: llvm.mlir.global internal constant @__rt_num_attrs(0 : i64)
// CHECK: llvm.mlir.global internal constant @__rt_custom_call_attrs

// CHECK: llvm.mlir.global internal constant @__rt_args_type_table
// CHECK: llvm.mlir.undef : !llvm.array<1 x ptr>
// CHECK: llvm.mlir.addressof @__type_id_float

// CHECK: func @custom_call(
// CHECK:   %[[CTX:.*]]: !llvm.ptr
// CHECK:   %[[ARG:.*]]: f32
// CHECK: )
func.func @custom_call(%arg0: !rt.execution_context, %arg1 : f32) {
  // CHECK-DAG: %[[MEM:.*]] = llvm.alloca {{.*}} x f32
  // CHECK-DAG: %[[ARGS:.*]] = llvm.alloca {{.*}} x !llvm.array<3 x ptr>
  // CHECK-DAG: %[[RETS:.*]] = llvm.alloca {{.*}} x !llvm.array<1 x ptr>

  // CHECK-DAG: %[[N_ARGS:.*]] = llvm.mlir.addressof @__rt_num_args
  // CHECK-DAG: llvm.store %[[ARG]], %[[MEM]]

  // CHECK: %[[ARGS_TYPES:.*]] = llvm.mlir.addressof @__rt_args_type_table
  // CHECK: llvm.insertvalue %[[ARGS_TYPES]], {{.*}}[1] : !llvm.array<3 x ptr>
  // CHECK: llvm.intr.lifetime.start -1, %[[ARGS]]
  // CHECK: llvm.store {{.*}}, %[[ARGS]] : !llvm.array<3 x ptr>, !llvm.ptr

  // CHECK: llvm.intr.lifetime.start -1, %[[RETS]]
  // CHECK: llvm.store {{.*}}, %[[RETS]] : !llvm.array<1 x ptr>, !llvm.ptr

  // CHECK: call @target
  // CHECK: llvm.intr.lifetime.end -1, %[[ARGS]]
  // CHECK: llvm.intr.lifetime.end -1, %[[RETS]]
  rt.call %arg0["target"] (%arg1) : (f32) -> ()
  func.return
}

// -----

// CHECK: llvm.mlir.global internal constant @__rt_args_type_table
// CHECK: llvm.mlir.addressof @__type_id_memref_view

// CHECK: func @custom_call(
// CHECK:   %[[CTX:.*]]: !llvm.ptr
// CHECK:   %[[ARG:.*]]: memref<?x256xf32>
// CHECK: )
func.func @custom_call(%arg0: !rt.execution_context, %arg1 : memref<?x256xf32>) {

  // CHECK: %[[DESC:.*]] = builtin.unrealized_conversion_cast %[[ARG]]
  // CHECK-SAME: to !llvm.struct

  // CHECK: llvm.mlir.undef : !llvm.array<4 x i64>
  // CHECK-NEXT: llvm.extractvalue %[[DESC]][3, 0]
  // CHECK-NEXT: arith.constant 256 : i64
  // CHECK-NEXT: llvm.insertvalue
  // CHECK-NEXT: llvm.insertvalue
  // CHECK-NEXT: arith.constant 256 : i64
  // CHECK-NEXT: arith.constant 1 : i64
  // CHECK-NEXT: llvm.insertvalue
  // CHECK-NEXT: %[[SIZES:.*]] = llvm.insertvalue

  // llvm.mlir.undef : !llvm.struct<(i8, i8, ptr, array<2 x i64>)>
  // CHECK: llvm.insertvalue
  // CHECK: llvm.insertvalue
  // CHECK: llvm.insertvalue %[[SIZES]]
  // CHECK: llvm.insertvalue

  // CHECK: %[[N_ARGS:.*]] = llvm.mlir.addressof @__rt_num_args
  // CHECK: %[[TYPES:.*]] = llvm.mlir.addressof @__rt_args_type_table

  // CHECK: call @target
  rt.call %arg0["target"] (%arg1) : (memref<?x256xf32>) -> ()
  func.return
}

// -----

// CHECK: internal constant @__rt_custom_call_attrs() {{.*}}: !llvm.array<4 x ptr>
// CHECK-NOT: internal constant @__rt_custom_call_attrs

// CHECK: func @dedup_custom_call_attrs(
// CHECK:   %[[CTX:.*]]: !llvm.ptr
// CHECK: )
func.func @dedup_custom_call_attrs(%arg0: !rt.execution_context) {
  // CHECK: call @target
  rt.call %arg0["target"] () { arr = [1, 2, 3] } : () -> ()
  // CHECK: call @target
  rt.call %arg0["target"] () { arr = [1, 2, 3] } : () -> ()
  func.return
}

// CHECK: func private @target(!llvm.ptr, !llvm.ptr,
// CHECK-SAME:                 !llvm.ptr) -> i1

// -----

// CHECK: func @dynamic_custom_call(
// CHECK:   %[[CTX:.*]]: !llvm.ptr
// CHECK: )
func.func @dynamic_custom_call(%arg0: !rt.execution_context) {
  // CHECK: call @runtimeCustomCall
  // CHECK: call @runtimeCustomCall
  rt.call dynamic %arg0["target"] () : () -> ()
  rt.call dynamic %arg0["target"] () : () -> ()
  func.return
}

// -----

func.func @custom_call(%ctx: !rt.execution_context) -> (f32) {
  // CHECK: %[[C1:.*]] = arith.constant 1 : i32
  // CHECK: %[[RETS:.*]] = llvm.alloca %[[C1]] x !llvm.array<3 x ptr>

  // CHECK: %[[C1_0:.*]] = arith.constant 1 : i32
  // CHECK: %[[F32_ALLOCA:.*]] = llvm.alloca %[[C1_0]] x f32

  // CHECK: %[[N_RETS:.*]]  = llvm.mlir.addressof @__rt_num_rets

  // CHECK: call @f32_reduce
  // CHECK: %[[LOAD2:.*]] = llvm.load %[[F32_ALLOCA]]
  // CHECK: llvm.intr.lifetime.end -1, %[[F32_ALLOCA]]
  %status, %0 = rt.call %ctx["f32_reduce"] () : () -> (f32)
  return %0 : f32
}

// -----

// CHECK: func @opaque_arg(
// CHECK-SAME:   %[[ARG0:.*]]: !llvm.ptr,
// CHECK-SAME:   %[[ARG1:.*]]: !llvm.ptr
// CHECK-SAME: )
func.func @opaque_arg(%ctx: !rt.execution_context, %arg: !rt.opaque) {
  return
}

// -----

// CHECK: llvm.mlir.global internal constant @__rt_args_type_table
// CHECK: llvm.mlir.addressof @__type_id_opaque : !llvm.ptr

// CHECK: func @opaque_custom_call_arg(
// CHECK-SAME:   %[[ARG0:.*]]: !llvm.ptr,
// CHECK-SAME:   %[[ARG1:.*]]: !llvm.ptr
// CHECK-SAME: )
func.func @opaque_custom_call_arg(%ctx: !rt.execution_context,
                                  %arg: !rt.opaque) {
  // CHECK: %[[ALLOCA:.*]] = llvm.alloca {{.*}} x !llvm.ptr
  // CHECK: llvm.store %[[ARG1]], %[[ALLOCA]] : !llvm.ptr
  // CHECK: call @target
  %status = rt.call %ctx["target"] (%arg) : (!rt.opaque) -> ()
  return
}

// -----

// CHECK: func @opaque_custom_call_res(
// CHECK-SAME:   %[[ARG0:.*]]: !llvm.ptr
// CHECK-SAME: )
func.func @opaque_custom_call_res(%ctx: !rt.execution_context) {
  // CHECK: %[[ALLOCA:.*]] = llvm.alloca {{.*}} x !llvm.ptr
  // CHECK: call @target
  %status, %res = rt.call %ctx["target"] () : () -> (!rt.opaque)
  // CHECK: llvm.load %[[ALLOCA]] : !llvm.ptr -> !llvm.ptr
  return
}

// -----

// CHECK: llvm.mlir.global internal constant @__rt_custom_call_attrs
// CHECK: llvm.mlir.addressof @__type_id_nullopt

// CHECK: func @custom_call_unit_attr(
// CHECK-SAME:   %[[ARG0:.*]]: !llvm.ptr
// CHECK-SAME: )
func.func @custom_call_unit_attr(%ctx: !rt.execution_context) {
  // CHECK: llvm.mlir.addressof @__rt_custom_call_attrs
  %status = rt.call %ctx["target"] () { attr } : () -> ()
  return
}

// -----

// CHECK: %[[C1:.*]] = arith.constant 1 : i32
// CHECK: %[[RETS_ALLOCA:.*]] = llvm.alloca %[[C1]] x !llvm.array<3 x ptr>

// CHECK: %[[C1_0:.*]] = arith.constant 1 : i32
// CHECK: %[[MEMREF_ALLOCA:.*]] = llvm.alloca %[[C1_0]] x !llvm.struct<(i8, i8, ptr, array<4 x i64>)>

// CHECK: call @f32_reduce

// CHECK: %[[DESC:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[DATA_GEP:.*]] = llvm.getelementptr %[[MEMREF_ALLOCA]]
// CHECK: %[[DATA_PTR:.*]] = llvm.load %[[DATA_GEP]]
// CHECK: %[[DATA:.*]] = llvm.bitcast %[[DATA_PTR]] : !llvm.ptr to !llvm.ptr<f32>

// CHECK: llvm.insertvalue %[[DATA]], {{.*}}[0]
// CHECK: llvm.insertvalue %[[DATA]], {{.*}}[1]

// CHECK: %[[OFFSET:.*]] = llvm.mlir.constant(0 : index)
// CHECK: llvm.insertvalue %[[OFFSET]], {{.*}}[2]

// CHECK: %[[DIM0:.*]] = llvm.mlir.constant(2 : index)
// CHECK: llvm.insertvalue %[[DIM0]], {{.*}}[3, 0]
// CHECK: %[[STRIDE0:.*]] = llvm.mlir.constant(2 : index)
// CHECK: llvm.insertvalue %[[STRIDE0]], {{.*}}[4, 0]

// CHECK: %[[DIM1:.*]] = llvm.mlir.constant(2 : index)
// CHECK: llvm.insertvalue %[[DIM1]], {{.*}}[3, 1]
// CHECK: %[[STRIDE1:.*]] = llvm.mlir.constant(1 : index)
// CHECK: llvm.insertvalue %[[STRIDE1]], {{.*}}[4, 1]
func.func @custom_call(%ctx: !rt.execution_context) -> (memref<2x2xf32>) {
  %status, %0 = rt.call %ctx["f32_reduce"] () : () -> (memref<2x2xf32>)
  return %0 : memref<2x2xf32>
}

// -----

// Test that custom call encoding can pass a reference to exported function as a
// custom call attribute.
func.func @init(%ctx: !rt.execution_context)
  attributes {rt.exported = 0: i32} { return }

// CHECK-DAG: mlir.global internal constant @__rt_num_attrs(1 : i64)
// CHECK-DAG: mlir.global external constant @__type_id_function_ordinal()
// CHECK-DAG: mlir.global internal constant @__rt_attr_value(0 : i32)

// CHECK: mlir.global internal constant @__rt_custom_call_attrs
// CHECK:  mlir.addressof @__type_id_function_ordinal
// CHECK:  mlir.addressof @__rt_attr_value
// CHECK:  llvm.return {{.*}} : !llvm.array<4 x ptr>

// CHECK: @custom_call_exported_function_ref
func.func @custom_call_exported_function_ref(%ctx: !rt.execution_context) {
  %status = rt.call %ctx["call_init"] () { init = @init } : () -> ()
  return
}

// -----

func.func private @compute() -> tensor<?xf32>

// CHECK: mlir.global internal constant @__rt_aggregate_hlo_trace
// CHECK: llvm.mlir.addressof @__rt_aggregate_hlo_trace

// CHECK: func @trace
func.func @trace(%ctx: !rt.execution_context) -> tensor<?xf32> {
  // CHECK: call @xla.trace.activity_start
  // CHECK: call @compute
  // CHECK: call @xla.trace.activity_end
  %0 = rt.trace #rt.hlo_trace<"foo", "bar", 0>, %ctx -> tensor<?xf32> {
    %1 = func.call @compute(): () -> tensor<?xf32>
    yield %1 : tensor<?xf32>
  }
  return %0 : tensor<?xf32>
}
