// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

//
// Declarations of the allocation functions to be linked against.
//

// CHECK: declare i8* @__mlir_alloc(i64)
// CHECK: declare void @__mlir_free(i8*)


//
// Basic functionality: function and block conversion, function calls,
// phi nodes, scalar type conversion, arithmetic operations.
//

// CHECK-LABEL: define void @empty() {
// CHECK-NEXT:    ret void
// CHECK-NEXT:  }
cfgfunc @empty() {
bb0:
  return
}

// CHECK-LABEL: declare void @body(i64)
extfunc @body(index)


// CHECK-LABEL: define void @simple_loop() {
cfgfunc @simple_loop() {
bb0:
// CHECK: br label %[[SIMPLE_BB1:[0-9]+]]
  br bb1

// Constants are inlined in LLVM rather than a separate instruction.
// CHECK: <label>:[[SIMPLE_BB1]]:
// CHECK-NEXT: br label %[[SIMPLE_BB2:[0-9]+]]
bb1:	// pred: bb0
  %c1 = constant 1 : index
  %c42 = constant 42 : index
  br bb2(%c1 : index)

// CHECK: <label>:[[SIMPLE_BB2]]:
// CHECK-NEXT:   %{{[0-9]+}} = phi i64 [ %{{[0-9]+}}, %[[SIMPLE_BB3:[0-9]+]] ], [ 1, %[[SIMPLE_BB1]] ]
// CHECK-NEXT:   %{{[0-9]+}} = icmp slt i64 %{{[0-9]+}}, 42
// CHECK-NEXT:   br i1 %{{[0-9]+}}, label %[[SIMPLE_BB3]], label %[[SIMPLE_BB4:[0-9]+]]
bb2(%0: index):	// 2 preds: bb1, bb3
  %1 = cmpi "slt", %0, %c42 : index
  cond_br %1, bb3, bb4

// CHECK: ; <label>:[[SIMPLE_BB3]]:
// CHECK-NEXT:   call void @body(i64 %{{[0-9]+}})
// CHECK-NEXT:   %{{[0-9]+}} = add i64 %{{[0-9]+}}, 1
// CHECK-NEXT:   br label %[[SIMPLE_BB2]]
bb3:	// pred: bb2
  call @body(%0) : (index) -> ()
  %c1_0 = constant 1 : index
  %2 = addi %0, %c1_0 : index
  br bb2(%2 : index)

// CHECK: ; <label>:[[SIMPLE_BB4]]:
// CHECK-NEXT:    ret void
bb4:	// pred: bb2
  return
}

// CHECK-LABEL: define void @simple_caller() {
// CHECK-NEXT:   call void @simple_loop()
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
cfgfunc @simple_caller() {
bb0:
  call @simple_loop() : () -> ()
  return
}

//cfgfunc @simple_indirect_caller() {
//bb0:
//  %f = constant @simple_loop : () -> ()
//  call_indirect %f() : () -> ()
//  return
//}

// CHECK-LABEL: define void @ml_caller() {
// CHECK-NEXT:   call void @simple_loop()
// CHECK-NEXT:   call void @more_imperfectly_nested_loops()
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
cfgfunc @ml_caller() {
bb0:
  call @simple_loop() : () -> ()
  call @more_imperfectly_nested_loops() : () -> ()
  return
}

// CHECK-LABEL: declare i64 @body_args(i64)
extfunc @body_args(index) -> index
// CHECK-LABEL: declare i32 @other(i64, i32)
extfunc @other(index, i32) -> i32

// CHECK-LABEL: define i32 @mlfunc_args(i32, i32) {
// CHECK-NEXT: br label %[[ARGS_BB1:[0-9]+]]
cfgfunc @mlfunc_args(i32, i32) -> i32 {
bb0(%arg0: i32, %arg1: i32):
  %c0_i32 = constant 0 : i32
  br bb1

// CHECK: <label>:[[ARGS_BB1]]:
// CHECK-NEXT: br label %[[ARGS_BB2:[0-9]+]]
bb1:	// pred: bb0
  %c0 = constant 0 : index
  %c42 = constant 42 : index
  br bb2(%c0 : index)

// CHECK: <label>:[[ARGS_BB2]]:
// CHECK-NEXT:   %5 = phi i64 [ %12, %[[ARGS_BB3:[0-9]+]] ], [ 0, %[[ARGS_BB1]] ]
// CHECK-NEXT:   %6 = icmp slt i64 %5, 42
// CHECK-NEXT:   br i1 %6, label %[[ARGS_BB3]], label %[[ARGS_BB4:[0-9]+]]
bb2(%0: index):	// 2 preds: bb1, bb3
  %1 = cmpi "slt", %0, %c42 : index
  cond_br %1, bb3, bb4

// CHECK: <label>:[[ARGS_BB3]]:
// CHECK-NEXT:   %8 = call i64 @body_args(i64 %5)
// CHECK-NEXT:   %9 = call i32 @other(i64 %8, i32 %0)
// CHECK-NEXT:   %10 = call i32 @other(i64 %8, i32 %9)
// CHECK-NEXT:   %11 = call i32 @other(i64 %8, i32 %1)
// CHECK-NEXT:   %12 = add i64 %5, 1
// CHECK-NEXT:   br label %[[ARGS_BB2]]
bb3:	// pred: bb2
  %2 = call @body_args(%0) : (index) -> index
  %3 = call @other(%2, %arg0) : (index, i32) -> i32
  %4 = call @other(%2, %3) : (index, i32) -> i32
  %5 = call @other(%2, %arg1) : (index, i32) -> i32
  %c1 = constant 1 : index
  %6 = addi %0, %c1 : index
  br bb2(%6 : index)

// CHECK: <label>:[[ARGS_BB4]]:
// CHECK-NEXT:   %14 = call i32 @other(i64 0, i32 0)
// CHECK-NEXT:   ret i32 %14
bb4:	// pred: bb2
  %c0_0 = constant 0 : index
  %7 = call @other(%c0_0, %c0_i32) : (index, i32) -> i32
  return %7 : i32
}

// CHECK: declare void @pre(i64)
extfunc @pre(index)

// CHECK: declare void @body2(i64, i64)
extfunc @body2(index, index)

// CHECK: declare void @post(i64)
extfunc @post(index)

// CHECK-LABEL: define void @imperfectly_nested_loops() {
// CHECK-NEXT:   br label %[[IMPER_BB1:[0-9]+]]
cfgfunc @imperfectly_nested_loops() {
bb0:
  br bb1

// CHECK: <label>:[[IMPER_BB1]]:
// CHECK-NEXT:   br label %[[IMPER_BB2:[0-9]+]]
bb1:	// pred: bb0
  %c0 = constant 0 : index
  %c42 = constant 42 : index
  br bb2(%c0 : index)

// CHECK: <label>:[[IMPER_BB2]]:
// CHECK-NEXT:   %3 = phi i64 [ %13, %[[IMPER_BB7:[0-9]+]] ], [ 0, %[[IMPER_BB1]] ]
// CHECK-NEXT:   %4 = icmp slt i64 %3, 42
// CHECK-NEXT:   br i1 %4, label %[[IMPER_BB3:[0-9]+]], label %[[IMPER_BB8:[0-9]+]]
bb2(%0: index):	// 2 preds: bb1, bb7
  %1 = cmpi "slt", %0, %c42 : index
  cond_br %1, bb3, bb8

// CHECK: <label>:[[IMPER_BB3]]:
// CHECK-NEXT:   call void @pre(i64 %3)
// CHECK-NEXT:   br label %[[IMPER_BB4:[0-9]+]]
bb3:	// pred: bb2
  call @pre(%0) : (index) -> ()
  br bb4

// CHECK: <label>:[[IMPER_BB4]]:
// CHECK-NEXT:   br label %[[IMPER_BB5:[0-9]+]]
bb4:	// pred: bb3
  %c7 = constant 7 : index
  %c56 = constant 56 : index
  br bb5(%c7 : index)

// CHECK: <label>:[[IMPER_BB5]]:
// CHECK-NEXT:   %8 = phi i64 [ %11, %[[IMPER_BB6:[0-9]+]] ], [ 7, %[[IMPER_BB4]] ]
// CHECK-NEXT:   %9 = icmp slt i64 %8, 56
// CHECK-NEXT:   br i1 %9, label %[[IMPER_BB6]], label %[[IMPER_BB7]]
bb5(%2: index):	// 2 preds: bb4, bb6
  %3 = cmpi "slt", %2, %c56 : index
  cond_br %3, bb6, bb7

// CHECK: <label>:[[IMPER_BB6]]:
// CHECK-NEXT:   call void @body2(i64 %3, i64 %8)
// CHECK-NEXT:   %11 = add i64 %8, 2
// CHECK-NEXT:   br label %[[IMPER_BB5]]
bb6:	// pred: bb5
  call @body2(%0, %2) : (index, index) -> ()
  %c2 = constant 2 : index
  %4 = addi %2, %c2 : index
  br bb5(%4 : index)

// CHECK: <label>:[[IMPER_BB7]]:
// CHECK-NEXT:   call void @post(i64 %3)
// CHECK-NEXT:   %13 = add i64 %3, 1
// CHECK-NEXT:   br label %[[IMPER_BB2]]
bb7:	// pred: bb5
  call @post(%0) : (index) -> ()
  %c1 = constant 1 : index
  %5 = addi %0, %c1 : index
  br bb2(%5 : index)

// CHECK: <label>:[[IMPER_BB8]]:
// CHECK-NEXT:   ret void
bb8:	// pred: bb2
  return
}

// CHECK: declare void @mid(i64)
extfunc @mid(index)

// CHECK: declare void @body3(i64, i64)
extfunc @body3(index, index)

// A complete function transformation check.
// CHECK-LABEL: define void @more_imperfectly_nested_loops() {
// CHECK-NEXT:   br label %1
// CHECK: ; <label>:1:                                      ; preds = %0
// CHECK-NEXT:   br label %2
// CHECK: ; <label>:2:                                      ; preds = %19, %1
// CHECK-NEXT:   %3 = phi i64 [ %20, %19 ], [ 0, %1 ]
// CHECK-NEXT:   %4 = icmp slt i64 %3, 42
// CHECK-NEXT:   br i1 %4, label %5, label %21
// CHECK: ; <label>:5:                                      ; preds = %2
// CHECK-NEXT:   call void @pre(i64 %3)
// CHECK-NEXT:   br label %6
// CHECK: ; <label>:6:                                      ; preds = %5
// CHECK-NEXT:   br label %7
// CHECK: ; <label>:7:                                      ; preds = %10, %6
// CHECK-NEXT:   %8 = phi i64 [ %11, %10 ], [ 7, %6 ]
// CHECK-NEXT:   %9 = icmp slt i64 %8, 56
// CHECK-NEXT:   br i1 %9, label %10, label %12
// CHECK: ; <label>:10:                                     ; preds = %7
// CHECK-NEXT:   call void @body2(i64 %3, i64 %8)
// CHECK-NEXT:   %11 = add i64 %8, 2
// CHECK-NEXT:   br label %7
// CHECK: ; <label>:12:                                     ; preds = %7
// CHECK-NEXT:   call void @mid(i64 %3)
// CHECK-NEXT:   br label %13
// CHECK: ; <label>:13:                                     ; preds = %12
// CHECK-NEXT:   br label %14
// CHECK: ; <label>:14:                                     ; preds = %17, %13
// CHECK-NEXT:   %15 = phi i64 [ %18, %17 ], [ 18, %13 ]
// CHECK-NEXT:   %16 = icmp slt i64 %15, 37
// CHECK-NEXT:   br i1 %16, label %17, label %19
// CHECK: ; <label>:17:                                     ; preds = %14
// CHECK-NEXT:   call void @body3(i64 %3, i64 %15)
// CHECK-NEXT:   %18 = add i64 %15, 3
// CHECK-NEXT:   br label %14
// CHECK: ; <label>:19:                                     ; preds = %14
// CHECK-NEXT:   call void @post(i64 %3)
// CHECK-NEXT:   %20 = add i64 %3, 1
// CHECK-NEXT:   br label %2
// CHECK: ; <label>:21:                                     ; preds = %2
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
cfgfunc @more_imperfectly_nested_loops() {
bb0:
  br bb1
bb1:	// pred: bb0
  %c0 = constant 0 : index
  %c42 = constant 42 : index
  br bb2(%c0 : index)
bb2(%0: index):	// 2 preds: bb1, bb11
  %1 = cmpi "slt", %0, %c42 : index
  cond_br %1, bb3, bb12
bb3:	// pred: bb2
  call @pre(%0) : (index) -> ()
  br bb4
bb4:	// pred: bb3
  %c7 = constant 7 : index
  %c56 = constant 56 : index
  br bb5(%c7 : index)
bb5(%2: index):	// 2 preds: bb4, bb6
  %3 = cmpi "slt", %2, %c56 : index
  cond_br %3, bb6, bb7
bb6:	// pred: bb5
  call @body2(%0, %2) : (index, index) -> ()
  %c2 = constant 2 : index
  %4 = addi %2, %c2 : index
  br bb5(%4 : index)
bb7:	// pred: bb5
  call @mid(%0) : (index) -> ()
  br bb8
bb8:	// pred: bb7
  %c18 = constant 18 : index
  %c37 = constant 37 : index
  br bb9(%c18 : index)
bb9(%5: index):	// 2 preds: bb8, bb10
  %6 = cmpi "slt", %5, %c37 : index
  cond_br %6, bb10, bb11
bb10:	// pred: bb9
  call @body3(%0, %5) : (index, index) -> ()
  %c3 = constant 3 : index
  %7 = addi %5, %c3 : index
  br bb9(%7 : index)
bb11:	// pred: bb9
  call @post(%0) : (index) -> ()
  %c1 = constant 1 : index
  %8 = addi %0, %c1 : index
  br bb2(%8 : index)
bb12:	// pred: bb2
  return
}

//
// MemRef type conversion, allocation and communication with functions.
//

// CHECK-LABEL: define void @memref_alloc()
cfgfunc @memref_alloc() {
bb0:
// CHECK-NEXT: %{{[0-9]+}} = call i8* @__mlir_alloc(i64 400)
// CHECK-NEXT: %{{[0-9]+}} = bitcast i8* %{{[0-9]+}} to float*
// CHECK-NEXT: %{{[0-9]+}} = insertvalue { float* } undef, float* %{{[0-9]+}}, 0
  %0 = alloc() : memref<10x10xf32>
// CHECK-NEXT: ret void
  return
}

// CHECK-LABEL: declare i64 @get_index()
extfunc @get_index() -> index

// CHECK-LABEL: define void @store_load_static()
cfgfunc @store_load_static() {
bb0:
// CHECK-NEXT: %{{[0-9]+}} = call i8* @__mlir_alloc(i64 40)
// CHECK-NEXT: %{{[0-9]+}} = bitcast i8* %{{[0-9]+}} to float*
// CHECK-NEXT: %{{[0-9]+}} = insertvalue { float* } undef, float* %{{[0-9]+}}, 0
  %0 = alloc() : memref<10xf32>
  %cst = constant 1.000000e+00 : f32
  br bb1
bb1:	// pred: bb0
  %c0 = constant 0 : index
  %c10 = constant 10 : index
  br bb2(%c0 : index)
// CHECK: %{{[0-9]+}} = phi i64 [ %{{[0-9]+}}, %{{[0-9]+}} ], [ 0, %{{[0-9]+}} ]
bb2(%1: index):	// 2 preds: bb1, bb3
// CHECK-NEXT: %{{[0-9]+}} = icmp slt i64 %{{[0-9]+}}, 10
  %2 = cmpi "slt", %1, %c10 : index
// CHECK-NEXT: br i1 %{{[0-9]+}}, label %{{[0-9]+}}, label %{{[0-9]+}}
  cond_br %2, bb3, bb4
bb3:	// pred: bb2
// CHECK: %{{[0-9]+}} = extractvalue { float* } %{{[0-9]+}}, 0
// CHECK-NEXT: %{{[0-9]+}} = getelementptr float, float* %{{[0-9]+}}, i64 %{{[0-9]+}}
// CHECK-NEXT: store float 1.000000e+00, float* %{{[0-9]+}}
  store %cst, %0[%1] : memref<10xf32>
  %c1 = constant 1 : index
// CHECK-NEXT: %{{[0-9]+}} = add i64 %{{[0-9]+}}, 1
  %3 = addi %1, %c1 : index
// CHECK-NEXT: br label %{{[0-9]+}}
  br bb2(%3 : index)
bb4:	// pred: bb2
  br bb5
bb5:	// pred: bb4
  %c0_0 = constant 0 : index
  %c10_1 = constant 10 : index
  br bb6(%c0_0 : index)
// CHECK: %{{[0-9]+}} = phi i64 [ %{{[0-9]+}}, %{{[0-9]+}} ], [ 0, %{{[0-9]+}} ]
bb6(%4: index):	// 2 preds: bb5, bb7
// CHECK-NEXT: %{{[0-9]+}} = icmp slt i64 %{{[0-9]+}}, 10
  %5 = cmpi "slt", %4, %c10_1 : index
// CHECK-NEXT: br i1 %{{[0-9]+}}, label %{{[0-9]+}}, label %{{[0-9]+}}
  cond_br %5, bb7, bb8
bb7:	// pred: bb6
// CHECK:      %{{[0-9]+}} = extractvalue { float* } %{{[0-9]+}}, 0
// CHECK-NEXT: %{{[0-9]+}} = getelementptr float, float* %{{[0-9]+}}, i64 %{{[0-9]+}}
// CHECK-NEXT: %{{[0-9]+}} = load float, float* %{{[0-9]+}}
  %6 = load %0[%4] : memref<10xf32>
  %c1_2 = constant 1 : index
// CHECK-NEXT: %{{[0-9]+}} = add i64 %{{[0-9]+}}, 1
  %7 = addi %4, %c1_2 : index
// CHECK-NEXT: br label %{{[0-9]+}}
  br bb6(%7 : index)
bb8:	// pred: bb6
// CHECK: ret void
  return
}

// CHECK-LABEL: define void @store_load_dynamic(i64)
cfgfunc @store_load_dynamic(index) {
bb0(%arg0: index):
// CHECK-NEXT: %{{[0-9]+}} = mul i64 %{{[0-9]+}}, 4
// CHECK-NEXT: %{{[0-9]+}} = call i8* @__mlir_alloc(i64 %{{[0-9]+}})
// CHECK-NEXT: %{{[0-9]+}} = bitcast i8* %{{[0-9]+}} to float*
// CHECK-NEXT: %{{[0-9]+}} = insertvalue { float*, i64 } undef, float* %{{[0-9]+}}, 0
// CHECK-NEXT: %{{[0-9]+}} = insertvalue { float*, i64 } %{{[0-9]+}}, i64 %{{[0-9]+}}, 1
  %0 = alloc(%arg0) : memref<?xf32>
  %cst = constant 1.000000e+00 : f32
// CHECK-NEXT: br label %{{[0-9]+}}
  br bb1
bb1:	// pred: bb0
  %c0 = constant 0 : index
  br bb2(%c0 : index)
// CHECK: %{{[0-9]+}} = phi i64 [ %{{[0-9]+}}, %{{[0-9]+}} ], [ 0, %{{[0-9]+}} ]
bb2(%1: index):	// 2 preds: bb1, bb3
// CHECK-NEXT: %{{[0-9]+}} = icmp slt i64 %{{[0-9]+}}, %{{[0-9]+}}
  %2 = cmpi "slt", %1, %arg0 : index
// CHECK-NEXT: br i1 %{{[0-9]+}}, label %{{[0-9]+}}, label %{{[0-9]+}}
  cond_br %2, bb3, bb4
bb3:	// pred: bb2
// CHECK:      %{{[0-9]+}} = extractvalue { float*, i64 } %{{[0-9]+}}, 1
// CHECK-NEXT: %{{[0-9]+}} = extractvalue { float*, i64 } %{{[0-9]+}}, 0
// CHECK-NEXT: %{{[0-9]+}} = getelementptr float, float* %{{[0-9]+}}, i64 %{{[0-9]+}}
// CHECK-NEXT: store float 1.000000e+00, float* %{{[0-9]+}}
  store %cst, %0[%1] : memref<?xf32>
  %c1 = constant 1 : index
// CHECK-NEXT: %{{[0-9]+}} = add i64 %{{[0-9]+}}, 1
  %3 = addi %1, %c1 : index
// CHECK-NEXT: br label %{{[0-9]+}}
  br bb2(%3 : index)
bb4:	// pred: bb3
  br bb5
bb5:	// pred: bb4
  %c0_0 = constant 0 : index
  br bb6(%c0_0 : index)
// CHECK: %{{[0-9]+}} = phi i64 [ %{{[0-9]+}}, %{{[0-9]+}} ], [ 0, %{{[0-9]+}} ]
bb6(%4: index):	// 2 preds: bb5, bb7
// CHECK-NEXT: %{{[0-9]+}} = icmp slt i64 %{{[0-9]+}}, %{{[0-9]+}}
  %5 = cmpi "slt", %4, %arg0 : index
// CHECK-NEXT: br i1 %{{[0-9]+}}, label %{{[0-9]+}}, label %{{[0-9]+}}
  cond_br %5, bb7, bb8
bb7:	// pred: bb6
// CHECK:      %{{[0-9]+}} = extractvalue { float*, i64 } %{{[0-9]+}}, 1
// CHECK-NEXT: %{{[0-9]+}} = extractvalue { float*, i64 } %{{[0-9]+}}, 0
// CHECK-NEXT: %{{[0-9]+}} = getelementptr float, float* %{{[0-9]+}}, i64 %{{[0-9]+}}
// CHECK-NEXT: %{{[0-9]+}} = load float, float* %{{[0-9]+}}
  %6 = load %0[%4] : memref<?xf32>
  %c1_1 = constant 1 : index
// CHECK-NEXT: %{{[0-9]+}} = add i64 %{{[0-9]+}}, 1
  %7 = addi %4, %c1_1 : index
// CHECK-NEXT: br label %{{[0-9]+}}
  br bb6(%7 : index)
bb8:	// pred: bb6
// CHECK: ret void
  return
}

// CHECK-LABEL: define void @store_load_mixed(i64)
cfgfunc @store_load_mixed(index) {
bb0(%arg0: index):
  %c10 = constant 10 : index
// CHECK-NEXT: %{{[0-9]+}} = mul i64 2, %{{[0-9]+}}
// CHECK-NEXT: %{{[0-9]+}} = mul i64 %{{[0-9]+}}, 4
// CHECK-NEXT: %{{[0-9]+}} = mul i64 %{{[0-9]+}}, 10
// CHECK-NEXT: %{{[0-9]+}} = mul i64 %{{[0-9]+}}, 4
// CHECK-NEXT: %{{[0-9]+}} = call i8* @__mlir_alloc(i64 %{{[0-9]+}})
// CHECK-NEXT: %{{[0-9]+}} = bitcast i8* %{{[0-9]+}} to float*
// CHECK-NEXT: %{{[0-9]+}} = insertvalue { float*, i64, i64 } undef, float* %{{[0-9]+}}, 0
// CHECK-NEXT: %{{[0-9]+}} = insertvalue { float*, i64, i64 } %{{[0-9]+}}, i64 %{{[0-9]+}}, 1
// CHECK-NEXT: %{{[0-9]+}} = insertvalue { float*, i64, i64 } %{{[0-9]+}}, i64 10, 2
  %0 = alloc(%arg0, %c10) : memref<2x?x4x?xf32>
  %c1 = constant 1 : index
  %c2 = constant 2 : index

// CHECK-NEXT: %{{[0-9]+}} = call i64 @get_index()
// CHECK-NEXT: %{{[0-9]+}} = call i64 @get_index()
  %1 = call @get_index() : () -> index
  %2 = call @get_index() : () -> index
  %cst = constant 4.200000e+01 : f32
// CHECK-NEXT: %{{[0-9]+}} = extractvalue { float*, i64, i64 } %{{[0-9]+}}, 1
// CHECK-NEXT: %{{[0-9]+}} = extractvalue { float*, i64, i64 } %{{[0-9]+}}, 2
// CHECK-NEXT: %{{[0-9]+}} = mul i64 1, %{{[0-9]+}}
// CHECK-NEXT: %{{[0-9]+}} = add i64 %{{[0-9]+}}, 2
// CHECK-NEXT: %{{[0-9]+}} = mul i64 %{{[0-9]+}}, 4
// CHECK-NEXT: %{{[0-9]+}} = add i64 %{{[0-9]+}}, %{{[0-9]+}}
// CHECK-NEXT: %{{[0-9]+}} = mul i64 %{{[0-9]+}}, %{{[0-9]+}}
// CHECK-NEXT: %{{[0-9]+}} = add i64 %{{[0-9]+}}, %{{[0-9]+}}
// CHECK-NEXT: %{{[0-9]+}} = extractvalue { float*, i64, i64 } %{{[0-9]+}}, 0
// CHECK-NEXT: %{{[0-9]+}} = getelementptr float, float* %{{[0-9]+}}, i64 %{{[0-9]+}}
// CHECK-NEXT: store float 4.200000e+01, float* %{{[0-9]+}}
  store %cst, %0[%c1, %c2, %1, %2] : memref<2x?x4x?xf32>
// CHECK-NEXT: %{{[0-9]+}} = extractvalue { float*, i64, i64 } %{{[0-9]+}}, 1
// CHECK-NEXT: %{{[0-9]+}} = extractvalue { float*, i64, i64 } %{{[0-9]+}}, 2
// CHECK-NEXT: %{{[0-9]+}} = mul i64 %{{[0-9]+}}, %{{[0-9]+}}
// CHECK-NEXT: %{{[0-9]+}} = add i64 %{{[0-9]+}}, %{{[0-9]+}}
// CHECK-NEXT: %{{[0-9]+}} = mul i64 %{{[0-9]+}}, 4
// CHECK-NEXT: %{{[0-9]+}} = add i64 %{{[0-9]+}}, 2
// CHECK-NEXT: %{{[0-9]+}} = mul i64 %{{[0-9]+}}, %{{[0-9]+}}
// CHECK-NEXT: %{{[0-9]+}} = add i64 %{{[0-9]+}}, 1
// CHECK-NEXT: %{{[0-9]+}} = extractvalue { float*, i64, i64 } %{{[0-9]+}}, 0
// CHECK-NEXT: %{{[0-9]+}} = getelementptr float, float* %{{[0-9]+}}, i64 %{{[0-9]+}}
// CHECK-NEXT: %{{[0-9]+}} = load float, float* %{{[0-9]+}}
  %3 = load %0[%2, %1, %c2, %c1] : memref<2x?x4x?xf32>
// CHECK-NEXT: ret void
  return
}

// CHECK-LABEL: define { float*, i64 } @memref_args_rets({ float* }, { float*, i64 }, { float*, i64 }) {
cfgfunc @memref_args_rets(memref<10xf32>, memref<?xf32>, memref<10x?xf32>) -> memref<10x?xf32> {
bb0(%arg0: memref<10xf32>, %arg1: memref<?xf32>, %arg2: memref<10x?xf32>):
  %c7 = constant 7 : index
// CHECK-NEXT: %{{[0-9]+}} = call i64 @get_index()
  %0 = call @get_index() : () -> index
  %cst = constant 4.200000e+01 : f32
// CHECK-NEXT: %{{[0-9]+}} = extractvalue { float* } %{{[0-9]+}}, 0
// CHECK-NEXT: %{{[0-9]+}} = getelementptr float, float* %{{[0-9]+}}, i64 7
// CHECK-NEXT: store float 4.200000e+01, float* %{{[0-9]+}}
  store %cst, %arg0[%c7] : memref<10xf32>
// CHECK-NEXT: %{{[0-9]+}} = extractvalue { float*, i64 } %{{[0-9]+}}, 1
// CHECK-NEXT: %{{[0-9]+}} = extractvalue { float*, i64 } %{{[0-9]+}}, 0
// CHECK-NEXT: %{{[0-9]+}} = getelementptr float, float* %{{[0-9]+}}, i64 7
// CHECK-NEXT: store float 4.200000e+01, float* %{{[0-9]+}}
  store %cst, %arg1[%c7] : memref<?xf32>
// CHECK-NEXT: %{{[0-9]+}} = extractvalue { float*, i64 } %{{[0-9]+}}, 1
// CHECK-NEXT: %{{[0-9]+}} = mul i64 7, %{{[0-9]+}}
// CHECK-NEXT: %{{[0-9]+}} = add i64 %{{[0-9]+}}, %{{[0-9]+}}
// CHECK-NEXT: %{{[0-9]+}} = extractvalue { float*, i64 } %{{[0-9]+}}, 0
// CHECK-NEXT: %{{[0-9]+}} = getelementptr float, float* %{{[0-9]+}}, i64 %{{[0-9]+}}
// CHECK-NEXT: store float 4.200000e+01, float* %{{[0-9]+}}
  store %cst, %arg2[%c7, %0] : memref<10x?xf32>
// CHECK-NEXT: %{{[0-9]+}} = mul i64 10, %{{[0-9]+}}
// CHECK-NEXT: %{{[0-9]+}} = mul i64 %{{[0-9]+}}, 4
// CHECK-NEXT: %{{[0-9]+}} = call i8* @__mlir_alloc(i64 %{{[0-9]+}})
// CHECK-NEXT: %{{[0-9]+}} = bitcast i8* %{{[0-9]+}} to float*
// CHECK-NEXT: %{{[0-9]+}} = insertvalue { float*, i64 } undef, float* %{{[0-9]+}}, 0
// CHECK-NEXT: %{{[0-9]+}} = insertvalue { float*, i64 } %{{[0-9]+}}, i64 %{{[0-9]+}}, 1
  %3 = alloc(%0) : memref<10x?xf32>
// CHECK-NEXT: ret { float*, i64 } %{{[0-9]+}}
  return %3 : memref<10x?xf32>
}


// CHECK-LABEL: define i64 @memref_dim({ float*, i64, i64 })
cfgfunc @memref_dim(memref<42x?x10x?xf32>) -> index {
bb0(%arg0: memref<42x?x10x?xf32>):
// Expecting this to create an LLVM constant.
  %d0 = dim %arg0, 0 : memref<42x?x10x?xf32>
// CHECK-NEXT: %2 = extractvalue { float*, i64, i64 } %0, 1
  %d1 = dim %arg0, 1 : memref<42x?x10x?xf32>
// Expecting this to create an LLVM constant.
  %d2 = dim %arg0, 2 : memref<42x?x10x?xf32>
// CHECK-NEXT: %3 = extractvalue { float*, i64, i64 } %0, 2
  %d3 = dim %arg0, 3 : memref<42x?x10x?xf32>
// Checking that the constant for d0 has been created.
// CHECK-NEXT: %4 = add i64 42, %2
  %d01 = addi %d0, %d1 : index
// Checking that the constant for d2 has been created.
// CHECK-NEXT: %5 = add i64 10, %3
  %d23 = addi %d2, %d3 : index
// CHECK-NEXT: %6 = add i64 %4, %5
  %d0123 = addi %d01, %d23 : index
// CHECK-NEXT: ret i64 %6
  return %d0123 : index
}

extfunc @get_i64() -> (i64)
extfunc @get_f32() -> (f32)
extfunc @get_memref() -> (memref<42x?x10x?xf32>)

// CHECK-LABEL: define { i64, float, { float*, i64, i64 } } @multireturn() {
cfgfunc @multireturn() -> (i64, f32, memref<42x?x10x?xf32>) {
bb0:
  %0 = call @get_i64() : () -> (i64)
  %1 = call @get_f32() : () -> (f32)
  %2 = call @get_memref() : () -> (memref<42x?x10x?xf32>)
// CHECK:        %{{[0-9]+}} = insertvalue { i64, float, { float*, i64, i64 } } undef, i64 %{{[0-9]+}}, 0
// CHECK-NEXT:   %{{[0-9]+}} = insertvalue { i64, float, { float*, i64, i64 } } %{{[0-9]+}}, float %{{[0-9]+}}, 1
// CHECK-NEXT:   %{{[0-9]+}} = insertvalue { i64, float, { float*, i64, i64 } } %{{[0-9]+}}, { float*, i64, i64 } %{{[0-9]+}}, 2
// CHECK-NEXT:   ret { i64, float, { float*, i64, i64 } } %{{[0-9]+}}
  return %0, %1, %2 : i64, f32, memref<42x?x10x?xf32>
}


// CHECK-LABEL: define void @multireturn_caller() {
cfgfunc @multireturn_caller() {
bb0:
// CHECK-NEXT:   %1 = call { i64, float, { float*, i64, i64 } } @multireturn()
// CHECK-NEXT:   [[ret0:%[0-9]+]] = extractvalue { i64, float, { float*, i64, i64 } } %1, 0
// CHECK-NEXT:   [[ret1:%[0-9]+]] = extractvalue { i64, float, { float*, i64, i64 } } %1, 1
// CHECK-NEXT:   [[ret2:%[0-9]+]] = extractvalue { i64, float, { float*, i64, i64 } } %1, 2
  %0 = call @multireturn() : () -> (i64, f32, memref<42x?x10x?xf32>)
  %1 = constant 42 : i64
// CHECK:   add i64 [[ret0]], 42
  %2 = addi %0#0, %1 : i64
  %3 = constant 42.0 : f32
// CHECK:   fadd float [[ret1]], 4.200000e+01
  %4 = addf %0#1, %3 : f32
  %5 = constant 0 : index
// CHECK:   extractvalue { float*, i64, i64 } [[ret2]], 0
  %6 = load %0#2 [%5, %5, %5, %5] : memref<42x?x10x?xf32>
  return
}

// CHECK-LABEL: define <4 x float> @vector_ops(<4 x float>) {
// CHECK-NEXT:    %2 = fadd <4 x float> %0, <float 4.200000e+01, float 4.200000e+01, float 4.200000e+01, float 4.200000e+01>
// CHECK-NEXT:    ret <4 x float> %2
// CHECK-NEXT:  }
cfgfunc @vector_ops(vector<4xf32>) -> vector<4xf32> {
bb0(%arg0 : vector<4xf32>):
  %0 = constant splat<vector<4xf32>, 42.> : vector<4xf32>
  %1 = addf %arg0, %0 : vector<4xf32>
  return %1 : vector<4xf32>
}

// CHECK-LABEL: @ops
cfgfunc @ops(f32, f32, i32, i32) -> (f32, i32) {
bb0(%arg0 : f32, %arg1 : f32, %arg2 : i32, %arg3 : i32):
// CHECK-NEXT: fsub float %0, %1
  %0 = subf %arg0, %arg1 : f32
// CHECK-NEXT: sub i32 %2, %3
  %1 = subi %arg2, %arg3 : i32
  return %0, %1 : f32, i32
}
