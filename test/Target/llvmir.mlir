// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

//
// Declarations of the allocation functions to be linked against.
//

// CHECK: declare i8* @malloc(i64)
func @malloc(!llvm.i64) -> !llvm<"i8*">
// CHECK: declare void @free(i8*)


//
// Basic functionality: function and block conversion, function calls,
// phi nodes, scalar type conversion, arithmetic operations.
//

// CHECK-LABEL: define void @empty() {
// CHECK-NEXT:    ret void
// CHECK-NEXT:  }
func @empty() {
  llvm.return
}

// CHECK-LABEL: declare void @body(i64)
func @body(!llvm.i64)


// CHECK-LABEL: define void @simple_loop() {
func @simple_loop() {
// CHECK: br label %[[SIMPLE_bb1:[0-9]+]]
  llvm.br ^bb1

// Constants are inlined in LLVM rather than a separate instruction.
// CHECK: [[SIMPLE_bb1]]:
// CHECK-NEXT: br label %[[SIMPLE_bb2:[0-9]+]]
^bb1:   // pred: ^bb0
  %0 = llvm.constant(1 : index) : !llvm.i64
  %1 = llvm.constant(42 : index) : !llvm.i64
  llvm.br ^bb2(%0 : !llvm.i64)

// CHECK: [[SIMPLE_bb2]]:
// CHECK-NEXT:   %{{[0-9]+}} = phi i64 [ %{{[0-9]+}}, %[[SIMPLE_bb3:[0-9]+]] ], [ 1, %[[SIMPLE_bb1]] ]
// CHECK-NEXT:   %{{[0-9]+}} = icmp slt i64 %{{[0-9]+}}, 42
// CHECK-NEXT:   br i1 %{{[0-9]+}}, label %[[SIMPLE_bb3]], label %[[SIMPLE_bb4:[0-9]+]]
^bb2(%2: !llvm.i64): // 2 preds: ^bb1, ^bb3
  %3 = llvm.icmp "slt" %2, %1 : !llvm.i64
  llvm.cond_br %3, ^bb3, ^bb4

// CHECK: [[SIMPLE_bb3]]:
// CHECK-NEXT:   call void @body(i64 %{{[0-9]+}})
// CHECK-NEXT:   %{{[0-9]+}} = add i64 %{{[0-9]+}}, 1
// CHECK-NEXT:   br label %[[SIMPLE_bb2]]
^bb3:   // pred: ^bb2
  llvm.call @body(%2) : (!llvm.i64) -> ()
  %4 = llvm.constant(1 : index) : !llvm.i64
  %5 = llvm.add %2, %4 : !llvm.i64
  llvm.br ^bb2(%5 : !llvm.i64)

// CHECK: [[SIMPLE_bb4]]:
// CHECK-NEXT:    ret void
^bb4:   // pred: ^bb2
  llvm.return
}

// CHECK-LABEL: define void @simple_caller() {
// CHECK-NEXT:   call void @simple_loop()
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
func @simple_caller() {
  llvm.call @simple_loop() : () -> ()
  llvm.return
}

//func @simple_indirect_caller() {
//^bb0:
//  %f = constant @simple_loop : () -> ()
//  call_indirect %f() : () -> ()
//  return
//}

// CHECK-LABEL: define void @ml_caller() {
// CHECK-NEXT:   call void @simple_loop()
// CHECK-NEXT:   call void @more_imperfectly_nested_loops()
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
func @ml_caller() {
  llvm.call @simple_loop() : () -> ()
  llvm.call @more_imperfectly_nested_loops() : () -> ()
  llvm.return
}

// CHECK-LABEL: declare i64 @body_args(i64)
func @body_args(!llvm.i64) -> !llvm.i64
// CHECK-LABEL: declare i32 @other(i64, i32)
func @other(!llvm.i64, !llvm.i32) -> !llvm.i32

// CHECK-LABEL: define i32 @func_args(i32 {{%.*}}, i32 {{%.*}}) {
// CHECK-NEXT: br label %[[ARGS_bb1:[0-9]+]]
func @func_args(%arg0: !llvm.i32, %arg1: !llvm.i32) -> !llvm.i32 {
  %0 = llvm.constant(0 : i32) : !llvm.i32
  llvm.br ^bb1

// CHECK: [[ARGS_bb1]]:
// CHECK-NEXT: br label %[[ARGS_bb2:[0-9]+]]
^bb1:   // pred: ^bb0
  %1 = llvm.constant(0 : index) : !llvm.i64
  %2 = llvm.constant(42 : index) : !llvm.i64
  llvm.br ^bb2(%1 : !llvm.i64)

// CHECK: [[ARGS_bb2]]:
// CHECK-NEXT:   %5 = phi i64 [ %12, %[[ARGS_bb3:[0-9]+]] ], [ 0, %[[ARGS_bb1]] ]
// CHECK-NEXT:   %6 = icmp slt i64 %5, 42
// CHECK-NEXT:   br i1 %6, label %[[ARGS_bb3]], label %[[ARGS_bb4:[0-9]+]]
^bb2(%3: !llvm.i64): // 2 preds: ^bb1, ^bb3
  %4 = llvm.icmp "slt" %3, %2 : !llvm.i64
  llvm.cond_br %4, ^bb3, ^bb4

// CHECK: [[ARGS_bb3]]:
// CHECK-NEXT:   %8 = call i64 @body_args(i64 %5)
// CHECK-NEXT:   %9 = call i32 @other(i64 %8, i32 %0)
// CHECK-NEXT:   %10 = call i32 @other(i64 %8, i32 %9)
// CHECK-NEXT:   %11 = call i32 @other(i64 %8, i32 %1)
// CHECK-NEXT:   %12 = add i64 %5, 1
// CHECK-NEXT:   br label %[[ARGS_bb2]]
^bb3:   // pred: ^bb2
  %5 = llvm.call @body_args(%3) : (!llvm.i64) -> !llvm.i64
  %6 = llvm.call @other(%5, %arg0) : (!llvm.i64, !llvm.i32) -> !llvm.i32
  %7 = llvm.call @other(%5, %6) : (!llvm.i64, !llvm.i32) -> !llvm.i32
  %8 = llvm.call @other(%5, %arg1) : (!llvm.i64, !llvm.i32) -> !llvm.i32
  %9 = llvm.constant(1 : index) : !llvm.i64
  %10 = llvm.add %3, %9 : !llvm.i64
  llvm.br ^bb2(%10 : !llvm.i64)

// CHECK: [[ARGS_bb4]]:
// CHECK-NEXT:   %14 = call i32 @other(i64 0, i32 0)
// CHECK-NEXT:   ret i32 %14
^bb4:   // pred: ^bb2
  %11 = llvm.constant(0 : index) : !llvm.i64
  %12 = llvm.call @other(%11, %0) : (!llvm.i64, !llvm.i32) -> !llvm.i32
  llvm.return %12 : !llvm.i32
}

// CHECK: declare void @pre(i64)
func @pre(!llvm.i64)

// CHECK: declare void @body2(i64, i64)
func @body2(!llvm.i64, !llvm.i64)

// CHECK: declare void @post(i64)
func @post(!llvm.i64)

// CHECK-LABEL: define void @imperfectly_nested_loops() {
// CHECK-NEXT:   br label %[[IMPER_bb1:[0-9]+]]
func @imperfectly_nested_loops() {
  llvm.br ^bb1

// CHECK: [[IMPER_bb1]]:
// CHECK-NEXT:   br label %[[IMPER_bb2:[0-9]+]]
^bb1:   // pred: ^bb0
  %0 = llvm.constant(0 : index) : !llvm.i64
  %1 = llvm.constant(42 : index) : !llvm.i64
  llvm.br ^bb2(%0 : !llvm.i64)

// CHECK: [[IMPER_bb2]]:
// CHECK-NEXT:   %3 = phi i64 [ %13, %[[IMPER_bb7:[0-9]+]] ], [ 0, %[[IMPER_bb1]] ]
// CHECK-NEXT:   %4 = icmp slt i64 %3, 42
// CHECK-NEXT:   br i1 %4, label %[[IMPER_bb3:[0-9]+]], label %[[IMPER_bb8:[0-9]+]]
^bb2(%2: !llvm.i64): // 2 preds: ^bb1, ^bb7
  %3 = llvm.icmp "slt" %2, %1 : !llvm.i64
  llvm.cond_br %3, ^bb3, ^bb8

// CHECK: [[IMPER_bb3]]:
// CHECK-NEXT:   call void @pre(i64 %3)
// CHECK-NEXT:   br label %[[IMPER_bb4:[0-9]+]]
^bb3:   // pred: ^bb2
  llvm.call @pre(%2) : (!llvm.i64) -> ()
  llvm.br ^bb4

// CHECK: [[IMPER_bb4]]:
// CHECK-NEXT:   br label %[[IMPER_bb5:[0-9]+]]
^bb4:   // pred: ^bb3
  %4 = llvm.constant(7 : index) : !llvm.i64
  %5 = llvm.constant(56 : index) : !llvm.i64
  llvm.br ^bb5(%4 : !llvm.i64)

// CHECK: [[IMPER_bb5]]:
// CHECK-NEXT:   %8 = phi i64 [ %11, %[[IMPER_bb6:[0-9]+]] ], [ 7, %[[IMPER_bb4]] ]
// CHECK-NEXT:   %9 = icmp slt i64 %8, 56
// CHECK-NEXT:   br i1 %9, label %[[IMPER_bb6]], label %[[IMPER_bb7]]
^bb5(%6: !llvm.i64): // 2 preds: ^bb4, ^bb6
  %7 = llvm.icmp "slt" %6, %5 : !llvm.i64
  llvm.cond_br %7, ^bb6, ^bb7

// CHECK: [[IMPER_bb6]]:
// CHECK-NEXT:   call void @body2(i64 %3, i64 %8)
// CHECK-NEXT:   %11 = add i64 %8, 2
// CHECK-NEXT:   br label %[[IMPER_bb5]]
^bb6:   // pred: ^bb5
  llvm.call @body2(%2, %6) : (!llvm.i64, !llvm.i64) -> ()
  %8 = llvm.constant(2 : index) : !llvm.i64
  %9 = llvm.add %6, %8 : !llvm.i64
  llvm.br ^bb5(%9 : !llvm.i64)

// CHECK: [[IMPER_bb7]]:
// CHECK-NEXT:   call void @post(i64 %3)
// CHECK-NEXT:   %13 = add i64 %3, 1
// CHECK-NEXT:   br label %[[IMPER_bb2]]
^bb7:   // pred: ^bb5
  llvm.call @post(%2) : (!llvm.i64) -> ()
  %10 = llvm.constant(1 : index) : !llvm.i64
  %11 = llvm.add %2, %10 : !llvm.i64
  llvm.br ^bb2(%11 : !llvm.i64)

// CHECK: [[IMPER_bb8]]:
// CHECK-NEXT:   ret void
^bb8:   // pred: ^bb2
  llvm.return
}

// CHECK: declare void @mid(i64)
func @mid(!llvm.i64)

// CHECK: declare void @body3(i64, i64)
func @body3(!llvm.i64, !llvm.i64)

// A complete function transformation check.
// CHECK-LABEL: define void @more_imperfectly_nested_loops() {
// CHECK-NEXT:   br label %1
// CHECK: 1:                                      ; preds = %0
// CHECK-NEXT:   br label %2
// CHECK: 2:                                      ; preds = %19, %1
// CHECK-NEXT:   %3 = phi i64 [ %20, %19 ], [ 0, %1 ]
// CHECK-NEXT:   %4 = icmp slt i64 %3, 42
// CHECK-NEXT:   br i1 %4, label %5, label %21
// CHECK: 5:                                      ; preds = %2
// CHECK-NEXT:   call void @pre(i64 %3)
// CHECK-NEXT:   br label %6
// CHECK: 6:                                      ; preds = %5
// CHECK-NEXT:   br label %7
// CHECK: 7:                                      ; preds = %10, %6
// CHECK-NEXT:   %8 = phi i64 [ %11, %10 ], [ 7, %6 ]
// CHECK-NEXT:   %9 = icmp slt i64 %8, 56
// CHECK-NEXT:   br i1 %9, label %10, label %12
// CHECK: 10:                                     ; preds = %7
// CHECK-NEXT:   call void @body2(i64 %3, i64 %8)
// CHECK-NEXT:   %11 = add i64 %8, 2
// CHECK-NEXT:   br label %7
// CHECK: 12:                                     ; preds = %7
// CHECK-NEXT:   call void @mid(i64 %3)
// CHECK-NEXT:   br label %13
// CHECK: 13:                                     ; preds = %12
// CHECK-NEXT:   br label %14
// CHECK: 14:                                     ; preds = %17, %13
// CHECK-NEXT:   %15 = phi i64 [ %18, %17 ], [ 18, %13 ]
// CHECK-NEXT:   %16 = icmp slt i64 %15, 37
// CHECK-NEXT:   br i1 %16, label %17, label %19
// CHECK: 17:                                     ; preds = %14
// CHECK-NEXT:   call void @body3(i64 %3, i64 %15)
// CHECK-NEXT:   %18 = add i64 %15, 3
// CHECK-NEXT:   br label %14
// CHECK: 19:                                     ; preds = %14
// CHECK-NEXT:   call void @post(i64 %3)
// CHECK-NEXT:   %20 = add i64 %3, 1
// CHECK-NEXT:   br label %2
// CHECK: 21:                                     ; preds = %2
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
func @more_imperfectly_nested_loops() {
  llvm.br ^bb1
^bb1:	// pred: ^bb0
  %0 = llvm.constant(0 : index) : !llvm.i64
  %1 = llvm.constant(42 : index) : !llvm.i64
  llvm.br ^bb2(%0 : !llvm.i64)
^bb2(%2: !llvm.i64):	// 2 preds: ^bb1, ^bb11
  %3 = llvm.icmp "slt" %2, %1 : !llvm.i64
  llvm.cond_br %3, ^bb3, ^bb12
^bb3:	// pred: ^bb2
  llvm.call @pre(%2) : (!llvm.i64) -> ()
  llvm.br ^bb4
^bb4:	// pred: ^bb3
  %4 = llvm.constant(7 : index) : !llvm.i64
  %5 = llvm.constant(56 : index) : !llvm.i64
  llvm.br ^bb5(%4 : !llvm.i64)
^bb5(%6: !llvm.i64):	// 2 preds: ^bb4, ^bb6
  %7 = llvm.icmp "slt" %6, %5 : !llvm.i64
  llvm.cond_br %7, ^bb6, ^bb7
^bb6:	// pred: ^bb5
  llvm.call @body2(%2, %6) : (!llvm.i64, !llvm.i64) -> ()
  %8 = llvm.constant(2 : index) : !llvm.i64
  %9 = llvm.add %6, %8 : !llvm.i64
  llvm.br ^bb5(%9 : !llvm.i64)
^bb7:	// pred: ^bb5
  llvm.call @mid(%2) : (!llvm.i64) -> ()
  llvm.br ^bb8
^bb8:	// pred: ^bb7
  %10 = llvm.constant(18 : index) : !llvm.i64
  %11 = llvm.constant(37 : index) : !llvm.i64
  llvm.br ^bb9(%10 : !llvm.i64)
^bb9(%12: !llvm.i64):	// 2 preds: ^bb8, ^bb10
  %13 = llvm.icmp "slt" %12, %11 : !llvm.i64
  llvm.cond_br %13, ^bb10, ^bb11
^bb10:	// pred: ^bb9
  llvm.call @body3(%2, %12) : (!llvm.i64, !llvm.i64) -> ()
  %14 = llvm.constant(3 : index) : !llvm.i64
  %15 = llvm.add %12, %14 : !llvm.i64
  llvm.br ^bb9(%15 : !llvm.i64)
^bb11:	// pred: ^bb9
  llvm.call @post(%2) : (!llvm.i64) -> ()
  %16 = llvm.constant(1 : index) : !llvm.i64
  %17 = llvm.add %2, %16 : !llvm.i64
  llvm.br ^bb2(%17 : !llvm.i64)
^bb12:	// pred: ^bb2
  llvm.return
}

//
// MemRef type conversion, allocation and communication with functions.
//

// CHECK-LABEL: define void @memref_alloc()
func @memref_alloc() {
// CHECK-NEXT: %{{[0-9]+}} = call i8* @malloc(i64 400)
// CHECK-NEXT: %{{[0-9]+}} = bitcast i8* %{{[0-9]+}} to float*
// CHECK-NEXT: %{{[0-9]+}} = insertvalue { float* } undef, float* %{{[0-9]+}}, 0
  %0 = llvm.constant(10 : index) : !llvm.i64
  %1 = llvm.constant(10 : index) : !llvm.i64
  %2 = llvm.mul %0, %1 : !llvm.i64
  %3 = llvm.undef : !llvm<"{ float* }">
  %4 = llvm.constant(4 : index) : !llvm.i64
  %5 = llvm.mul %2, %4 : !llvm.i64
  %6 = llvm.call @malloc(%5) : (!llvm.i64) -> !llvm<"i8*">
  %7 = llvm.bitcast %6 : !llvm<"i8*"> to !llvm<"float*">
  %8 = llvm.insertvalue %7, %3[0] : !llvm<"{ float* }">
// CHECK-NEXT: ret void
  llvm.return
}

// CHECK-LABEL: declare i64 @get_index()
func @get_index() -> !llvm.i64

// CHECK-LABEL: define void @store_load_static()
func @store_load_static() {
^bb0:
// CHECK-NEXT: %{{[0-9]+}} = call i8* @malloc(i64 40)
// CHECK-NEXT: %{{[0-9]+}} = bitcast i8* %{{[0-9]+}} to float*
// CHECK-NEXT: %{{[0-9]+}} = insertvalue { float* } undef, float* %{{[0-9]+}}, 0
  %0 = llvm.constant(10 : index) : !llvm.i64
  %1 = llvm.undef : !llvm<"{ float* }">
  %2 = llvm.constant(4 : index) : !llvm.i64
  %3 = llvm.mul %0, %2 : !llvm.i64
  %4 = llvm.call @malloc(%3) : (!llvm.i64) -> !llvm<"i8*">
  %5 = llvm.bitcast %4 : !llvm<"i8*"> to !llvm<"float*">
  %6 = llvm.insertvalue %5, %1[0] : !llvm<"{ float* }">
  %7 = llvm.constant(1.000000e+00 : f32) : !llvm.float
  llvm.br ^bb1
^bb1:   // pred: ^bb0
  %8 = llvm.constant(0 : index) : !llvm.i64
  %9 = llvm.constant(10 : index) : !llvm.i64
  llvm.br ^bb2(%8 : !llvm.i64)
// CHECK: %{{[0-9]+}} = phi i64 [ %{{[0-9]+}}, %{{[0-9]+}} ], [ 0, %{{[0-9]+}} ]
^bb2(%10: !llvm.i64):        // 2 preds: ^bb1, ^bb3
// CHECK-NEXT: %{{[0-9]+}} = icmp slt i64 %{{[0-9]+}}, 10
  %11 = llvm.icmp "slt" %10, %9 : !llvm.i64
// CHECK-NEXT: br i1 %{{[0-9]+}}, label %{{[0-9]+}}, label %{{[0-9]+}}
  llvm.cond_br %11, ^bb3, ^bb4
^bb3:   // pred: ^bb2
// CHECK: %{{[0-9]+}} = extractvalue { float* } %{{[0-9]+}}, 0
// CHECK-NEXT: %{{[0-9]+}} = getelementptr float, float* %{{[0-9]+}}, i64 %{{[0-9]+}}
// CHECK-NEXT: store float 1.000000e+00, float* %{{[0-9]+}}
  %12 = llvm.constant(10 : index) : !llvm.i64
  %13 = llvm.extractvalue %6[0] : !llvm<"{ float* }">
  %14 = llvm.getelementptr %13[%10] : (!llvm<"float*">, !llvm.i64) -> !llvm<"float*">
  llvm.store %7, %14 : !llvm<"float*">
  %15 = llvm.constant(1 : index) : !llvm.i64
// CHECK-NEXT: %{{[0-9]+}} = add i64 %{{[0-9]+}}, 1
  %16 = llvm.add %10, %15 : !llvm.i64
// CHECK-NEXT: br label %{{[0-9]+}}
  llvm.br ^bb2(%16 : !llvm.i64)
^bb4:   // pred: ^bb2
  llvm.br ^bb5
^bb5:   // pred: ^bb4
  %17 = llvm.constant(0 : index) : !llvm.i64
  %18 = llvm.constant(10 : index) : !llvm.i64
  llvm.br ^bb6(%17 : !llvm.i64)
// CHECK: %{{[0-9]+}} = phi i64 [ %{{[0-9]+}}, %{{[0-9]+}} ], [ 0, %{{[0-9]+}} ]
^bb6(%19: !llvm.i64):        // 2 preds: ^bb5, ^bb7
// CHECK-NEXT: %{{[0-9]+}} = icmp slt i64 %{{[0-9]+}}, 10
  %20 = llvm.icmp "slt" %19, %18 : !llvm.i64
// CHECK-NEXT: br i1 %{{[0-9]+}}, label %{{[0-9]+}}, label %{{[0-9]+}}
  llvm.cond_br %20, ^bb7, ^bb8
^bb7:   // pred: ^bb6
// CHECK:      %{{[0-9]+}} = extractvalue { float* } %{{[0-9]+}}, 0
// CHECK-NEXT: %{{[0-9]+}} = getelementptr float, float* %{{[0-9]+}}, i64 %{{[0-9]+}}
// CHECK-NEXT: %{{[0-9]+}} = load float, float* %{{[0-9]+}}
  %21 = llvm.constant(10 : index) : !llvm.i64
  %22 = llvm.extractvalue %6[0] : !llvm<"{ float* }">
  %23 = llvm.getelementptr %22[%19] : (!llvm<"float*">, !llvm.i64) -> !llvm<"float*">
  %24 = llvm.load %23 : !llvm<"float*">
  %25 = llvm.constant(1 : index) : !llvm.i64
// CHECK-NEXT: %{{[0-9]+}} = add i64 %{{[0-9]+}}, 1
  %26 = llvm.add %19, %25 : !llvm.i64
// CHECK-NEXT: br label %{{[0-9]+}}
  llvm.br ^bb6(%26 : !llvm.i64)
^bb8:   // pred: ^bb6
// CHECK: ret void
  llvm.return
}

// CHECK-LABEL: define void @store_load_dynamic(i64 {{%.*}})
func @store_load_dynamic(%arg0: !llvm.i64) {
// CHECK-NEXT: %{{[0-9]+}} = mul i64 %{{[0-9]+}}, 4
// CHECK-NEXT: %{{[0-9]+}} = call i8* @malloc(i64 %{{[0-9]+}})
// CHECK-NEXT: %{{[0-9]+}} = bitcast i8* %{{[0-9]+}} to float*
// CHECK-NEXT: %{{[0-9]+}} = insertvalue { float*, i64 } undef, float* %{{[0-9]+}}, 0
// CHECK-NEXT: %{{[0-9]+}} = insertvalue { float*, i64 } %{{[0-9]+}}, i64 %{{[0-9]+}}, 1
  %0 = llvm.undef : !llvm<"{ float*, i64 }">
  %1 = llvm.constant(4 : index) : !llvm.i64
  %2 = llvm.mul %arg0, %1 : !llvm.i64
  %3 = llvm.call @malloc(%2) : (!llvm.i64) -> !llvm<"i8*">
  %4 = llvm.bitcast %3 : !llvm<"i8*"> to !llvm<"float*">
  %5 = llvm.insertvalue %4, %0[0] : !llvm<"{ float*, i64 }">
  %6 = llvm.insertvalue %arg0, %5[1] : !llvm<"{ float*, i64 }">
  %7 = llvm.constant(1.000000e+00 : f32) : !llvm.float
// CHECK-NEXT: br label %{{[0-9]+}}
  llvm.br ^bb1
^bb1:   // pred: ^bb0
  %8 = llvm.constant(0 : index) : !llvm.i64
  llvm.br ^bb2(%8 : !llvm.i64)
// CHECK: %{{[0-9]+}} = phi i64 [ %{{[0-9]+}}, %{{[0-9]+}} ], [ 0, %{{[0-9]+}} ]
^bb2(%9: !llvm.i64): // 2 preds: ^bb1, ^bb3
// CHECK-NEXT: %{{[0-9]+}} = icmp slt i64 %{{[0-9]+}}, %{{[0-9]+}}
  %10 = llvm.icmp "slt" %9, %arg0 : !llvm.i64
// CHECK-NEXT: br i1 %{{[0-9]+}}, label %{{[0-9]+}}, label %{{[0-9]+}}
  llvm.cond_br %10, ^bb3, ^bb4
^bb3:   // pred: ^bb2
// CHECK:      %{{[0-9]+}} = extractvalue { float*, i64 } %{{[0-9]+}}, 1
// CHECK-NEXT: %{{[0-9]+}} = extractvalue { float*, i64 } %{{[0-9]+}}, 0
// CHECK-NEXT: %{{[0-9]+}} = getelementptr float, float* %{{[0-9]+}}, i64 %{{[0-9]+}}
// CHECK-NEXT: store float 1.000000e+00, float* %{{[0-9]+}}
  %11 = llvm.extractvalue %6[1] : !llvm<"{ float*, i64 }">
  %12 = llvm.extractvalue %6[0] : !llvm<"{ float*, i64 }">
  %13 = llvm.getelementptr %12[%9] : (!llvm<"float*">, !llvm.i64) -> !llvm<"float*">
  llvm.store %7, %13 : !llvm<"float*">
  %14 = llvm.constant(1 : index) : !llvm.i64
// CHECK-NEXT: %{{[0-9]+}} = add i64 %{{[0-9]+}}, 1
  %15 = llvm.add %9, %14 : !llvm.i64
// CHECK-NEXT: br label %{{[0-9]+}}
  llvm.br ^bb2(%15 : !llvm.i64)
^bb4:   // pred: ^bb3
  llvm.br ^bb5
^bb5:   // pred: ^bb4
  %16 = llvm.constant(0 : index) : !llvm.i64
  llvm.br ^bb6(%16 : !llvm.i64)
// CHECK: %{{[0-9]+}} = phi i64 [ %{{[0-9]+}}, %{{[0-9]+}} ], [ 0, %{{[0-9]+}} ]
^bb6(%17: !llvm.i64):        // 2 preds: ^bb5, ^bb7
// CHECK-NEXT: %{{[0-9]+}} = icmp slt i64 %{{[0-9]+}}, %{{[0-9]+}}
  %18 = llvm.icmp "slt" %17, %arg0 : !llvm.i64
// CHECK-NEXT: br i1 %{{[0-9]+}}, label %{{[0-9]+}}, label %{{[0-9]+}}
  llvm.cond_br %18, ^bb7, ^bb8
^bb7:   // pred: ^bb6
// CHECK:      %{{[0-9]+}} = extractvalue { float*, i64 } %{{[0-9]+}}, 1
// CHECK-NEXT: %{{[0-9]+}} = extractvalue { float*, i64 } %{{[0-9]+}}, 0
// CHECK-NEXT: %{{[0-9]+}} = getelementptr float, float* %{{[0-9]+}}, i64 %{{[0-9]+}}
// CHECK-NEXT: %{{[0-9]+}} = load float, float* %{{[0-9]+}}
  %19 = llvm.extractvalue %6[1] : !llvm<"{ float*, i64 }">
  %20 = llvm.extractvalue %6[0] : !llvm<"{ float*, i64 }">
  %21 = llvm.getelementptr %20[%17] : (!llvm<"float*">, !llvm.i64) -> !llvm<"float*">
  %22 = llvm.load %21 : !llvm<"float*">
  %23 = llvm.constant(1 : index) : !llvm.i64
// CHECK-NEXT: %{{[0-9]+}} = add i64 %{{[0-9]+}}, 1
  %24 = llvm.add %17, %23 : !llvm.i64
// CHECK-NEXT: br label %{{[0-9]+}}
  llvm.br ^bb6(%24 : !llvm.i64)
^bb8:   // pred: ^bb6
// CHECK: ret void
  llvm.return
}

// CHECK-LABEL: define void @store_load_mixed(i64 {{%.*}})
func @store_load_mixed(%arg0: !llvm.i64) {
  %0 = llvm.constant(10 : index) : !llvm.i64
// CHECK-NEXT: %{{[0-9]+}} = mul i64 2, %{{[0-9]+}}
// CHECK-NEXT: %{{[0-9]+}} = mul i64 %{{[0-9]+}}, 4
// CHECK-NEXT: %{{[0-9]+}} = mul i64 %{{[0-9]+}}, 10
// CHECK-NEXT: %{{[0-9]+}} = mul i64 %{{[0-9]+}}, 4
// CHECK-NEXT: %{{[0-9]+}} = call i8* @malloc(i64 %{{[0-9]+}})
// CHECK-NEXT: %{{[0-9]+}} = bitcast i8* %{{[0-9]+}} to float*
// CHECK-NEXT: %{{[0-9]+}} = insertvalue { float*, i64, i64 } undef, float* %{{[0-9]+}}, 0
// CHECK-NEXT: %{{[0-9]+}} = insertvalue { float*, i64, i64 } %{{[0-9]+}}, i64 %{{[0-9]+}}, 1
// CHECK-NEXT: %{{[0-9]+}} = insertvalue { float*, i64, i64 } %{{[0-9]+}}, i64 10, 2
  %1 = llvm.constant(2 : index) : !llvm.i64
  %2 = llvm.constant(4 : index) : !llvm.i64
  %3 = llvm.mul %1, %arg0 : !llvm.i64
  %4 = llvm.mul %3, %2 : !llvm.i64
  %5 = llvm.mul %4, %0 : !llvm.i64
  %6 = llvm.undef : !llvm<"{ float*, i64, i64 }">
  %7 = llvm.constant(4 : index) : !llvm.i64
  %8 = llvm.mul %5, %7 : !llvm.i64
  %9 = llvm.call @malloc(%8) : (!llvm.i64) -> !llvm<"i8*">
  %10 = llvm.bitcast %9 : !llvm<"i8*"> to !llvm<"float*">
  %11 = llvm.insertvalue %10, %6[0] : !llvm<"{ float*, i64, i64 }">
  %12 = llvm.insertvalue %arg0, %11[1] : !llvm<"{ float*, i64, i64 }">
  %13 = llvm.insertvalue %0, %12[2] : !llvm<"{ float*, i64, i64 }">

// CHECK-NEXT: %{{[0-9]+}} = call i64 @get_index()
// CHECK-NEXT: %{{[0-9]+}} = call i64 @get_index()
  %14 = llvm.constant(1 : index) : !llvm.i64
  %15 = llvm.constant(2 : index) : !llvm.i64
  %16 = llvm.call @get_index() : () -> !llvm.i64
  %17 = llvm.call @get_index() : () -> !llvm.i64
  %18 = llvm.constant(4.200000e+01 : f32) : !llvm.float
  %19 = llvm.constant(2 : index) : !llvm.i64
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
  %20 = llvm.extractvalue %13[1] : !llvm<"{ float*, i64, i64 }">
  %21 = llvm.constant(4 : index) : !llvm.i64
  %22 = llvm.extractvalue %13[2] : !llvm<"{ float*, i64, i64 }">
  %23 = llvm.mul %14, %20 : !llvm.i64
  %24 = llvm.add %23, %15 : !llvm.i64
  %25 = llvm.mul %24, %21 : !llvm.i64
  %26 = llvm.add %25, %16 : !llvm.i64
  %27 = llvm.mul %26, %22 : !llvm.i64
  %28 = llvm.add %27, %17 : !llvm.i64
  %29 = llvm.extractvalue %13[0] : !llvm<"{ float*, i64, i64 }">
  %30 = llvm.getelementptr %29[%28] : (!llvm<"float*">, !llvm.i64) -> !llvm<"float*">
  llvm.store %18, %30 : !llvm<"float*">
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
  %31 = llvm.constant(2 : index) : !llvm.i64
  %32 = llvm.extractvalue %13[1] : !llvm<"{ float*, i64, i64 }">
  %33 = llvm.constant(4 : index) : !llvm.i64
  %34 = llvm.extractvalue %13[2] : !llvm<"{ float*, i64, i64 }">
  %35 = llvm.mul %17, %32 : !llvm.i64
  %36 = llvm.add %35, %16 : !llvm.i64
  %37 = llvm.mul %36, %33 : !llvm.i64
  %38 = llvm.add %37, %15 : !llvm.i64
  %39 = llvm.mul %38, %34 : !llvm.i64
  %40 = llvm.add %39, %14 : !llvm.i64
  %41 = llvm.extractvalue %13[0] : !llvm<"{ float*, i64, i64 }">
  %42 = llvm.getelementptr %41[%40] : (!llvm<"float*">, !llvm.i64) -> !llvm<"float*">
  %43 = llvm.load %42 : !llvm<"float*">
// CHECK-NEXT: ret void
  llvm.return
}

// CHECK-LABEL: define { float*, i64 } @memref_args_rets({ float* } {{%.*}}, { float*, i64 } {{%.*}}, { float*, i64 } {{%.*}}) {
func @memref_args_rets(%arg0: !llvm<"{ float* }">, %arg1: !llvm<"{ float*, i64 }">, %arg2: !llvm<"{ float*, i64 }">) -> !llvm<"{ float*, i64 }"> {
  %0 = llvm.constant(7 : index) : !llvm.i64
// CHECK-NEXT: %{{[0-9]+}} = call i64 @get_index()
  %1 = llvm.call @get_index() : () -> !llvm.i64
  %2 = llvm.constant(4.200000e+01 : f32) : !llvm.float
// CHECK-NEXT: %{{[0-9]+}} = extractvalue { float* } %{{[0-9]+}}, 0
// CHECK-NEXT: %{{[0-9]+}} = getelementptr float, float* %{{[0-9]+}}, i64 7
// CHECK-NEXT: store float 4.200000e+01, float* %{{[0-9]+}}
  %3 = llvm.constant(10 : index) : !llvm.i64
  %4 = llvm.extractvalue %arg0[0] : !llvm<"{ float* }">
  %5 = llvm.getelementptr %4[%0] : (!llvm<"float*">, !llvm.i64) -> !llvm<"float*">
  llvm.store %2, %5 : !llvm<"float*">
// CHECK-NEXT: %{{[0-9]+}} = extractvalue { float*, i64 } %{{[0-9]+}}, 1
// CHECK-NEXT: %{{[0-9]+}} = extractvalue { float*, i64 } %{{[0-9]+}}, 0
// CHECK-NEXT: %{{[0-9]+}} = getelementptr float, float* %{{[0-9]+}}, i64 7
// CHECK-NEXT: store float 4.200000e+01, float* %{{[0-9]+}}
  %6 = llvm.extractvalue %arg1[1] : !llvm<"{ float*, i64 }">
  %7 = llvm.extractvalue %arg1[0] : !llvm<"{ float*, i64 }">
  %8 = llvm.getelementptr %7[%0] : (!llvm<"float*">, !llvm.i64) -> !llvm<"float*">
  llvm.store %2, %8 : !llvm<"float*">
// CHECK-NEXT: %{{[0-9]+}} = extractvalue { float*, i64 } %{{[0-9]+}}, 1
// CHECK-NEXT: %{{[0-9]+}} = mul i64 7, %{{[0-9]+}}
// CHECK-NEXT: %{{[0-9]+}} = add i64 %{{[0-9]+}}, %{{[0-9]+}}
// CHECK-NEXT: %{{[0-9]+}} = extractvalue { float*, i64 } %{{[0-9]+}}, 0
// CHECK-NEXT: %{{[0-9]+}} = getelementptr float, float* %{{[0-9]+}}, i64 %{{[0-9]+}}
// CHECK-NEXT: store float 4.200000e+01, float* %{{[0-9]+}}
  %9 = llvm.constant(10 : index) : !llvm.i64
  %10 = llvm.extractvalue %arg2[1] : !llvm<"{ float*, i64 }">
  %11 = llvm.mul %0, %10 : !llvm.i64
  %12 = llvm.add %11, %1 : !llvm.i64
  %13 = llvm.extractvalue %arg2[0] : !llvm<"{ float*, i64 }">
  %14 = llvm.getelementptr %13[%12] : (!llvm<"float*">, !llvm.i64) -> !llvm<"float*">
  llvm.store %2, %14 : !llvm<"float*">
// CHECK-NEXT: %{{[0-9]+}} = mul i64 10, %{{[0-9]+}}
// CHECK-NEXT: %{{[0-9]+}} = mul i64 %{{[0-9]+}}, 4
// CHECK-NEXT: %{{[0-9]+}} = call i8* @malloc(i64 %{{[0-9]+}})
// CHECK-NEXT: %{{[0-9]+}} = bitcast i8* %{{[0-9]+}} to float*
// CHECK-NEXT: %{{[0-9]+}} = insertvalue { float*, i64 } undef, float* %{{[0-9]+}}, 0
// CHECK-NEXT: %{{[0-9]+}} = insertvalue { float*, i64 } %{{[0-9]+}}, i64 %{{[0-9]+}}, 1
  %15 = llvm.constant(10 : index) : !llvm.i64
  %16 = llvm.mul %15, %1 : !llvm.i64
  %17 = llvm.undef : !llvm<"{ float*, i64 }">
  %18 = llvm.constant(4 : index) : !llvm.i64
  %19 = llvm.mul %16, %18 : !llvm.i64
  %20 = llvm.call @malloc(%19) : (!llvm.i64) -> !llvm<"i8*">
  %21 = llvm.bitcast %20 : !llvm<"i8*"> to !llvm<"float*">
  %22 = llvm.insertvalue %21, %17[0] : !llvm<"{ float*, i64 }">
  %23 = llvm.insertvalue %1, %22[1] : !llvm<"{ float*, i64 }">
// CHECK-NEXT: ret { float*, i64 } %{{[0-9]+}}
  llvm.return %23 : !llvm<"{ float*, i64 }">
}


// CHECK-LABEL: define i64 @memref_dim({ float*, i64, i64 } {{%.*}})
func @memref_dim(%arg0: !llvm<"{ float*, i64, i64 }">) -> !llvm.i64 {
// Expecting this to create an LLVM constant.
  %0 = llvm.constant(42 : index) : !llvm.i64
// CHECK-NEXT: %2 = extractvalue { float*, i64, i64 } %0, 1
  %1 = llvm.extractvalue %arg0[1] : !llvm<"{ float*, i64, i64 }">
// Expecting this to create an LLVM constant.
  %2 = llvm.constant(10 : index) : !llvm.i64
// CHECK-NEXT: %3 = extractvalue { float*, i64, i64 } %0, 2
  %3 = llvm.extractvalue %arg0[2] : !llvm<"{ float*, i64, i64 }">
// Checking that the constant for d0 has been created.
// CHECK-NEXT: %4 = add i64 42, %2
  %4 = llvm.add %0, %1 : !llvm.i64
// Checking that the constant for d2 has been created.
// CHECK-NEXT: %5 = add i64 10, %3
  %5 = llvm.add %2, %3 : !llvm.i64
// CHECK-NEXT: %6 = add i64 %4, %5
  %6 = llvm.add %4, %5 : !llvm.i64
// CHECK-NEXT: ret i64 %6
  llvm.return %6 : !llvm.i64
}

func @get_i64() -> !llvm.i64
func @get_f32() -> !llvm.float
func @get_memref() -> !llvm<"{ float*, i64, i64 }">

// CHECK-LABEL: define { i64, float, { float*, i64, i64 } } @multireturn() {
func @multireturn() -> !llvm<"{ i64, float, { float*, i64, i64 } }"> {
  %0 = llvm.call @get_i64() : () -> !llvm.i64
  %1 = llvm.call @get_f32() : () -> !llvm.float
  %2 = llvm.call @get_memref() : () -> !llvm<"{ float*, i64, i64 }">
// CHECK:        %{{[0-9]+}} = insertvalue { i64, float, { float*, i64, i64 } } undef, i64 %{{[0-9]+}}, 0
// CHECK-NEXT:   %{{[0-9]+}} = insertvalue { i64, float, { float*, i64, i64 } } %{{[0-9]+}}, float %{{[0-9]+}}, 1
// CHECK-NEXT:   %{{[0-9]+}} = insertvalue { i64, float, { float*, i64, i64 } } %{{[0-9]+}}, { float*, i64, i64 } %{{[0-9]+}}, 2
// CHECK-NEXT:   ret { i64, float, { float*, i64, i64 } } %{{[0-9]+}}
  %3 = llvm.undef : !llvm<"{ i64, float, { float*, i64, i64 } }">
  %4 = llvm.insertvalue %0, %3[0] : !llvm<"{ i64, float, { float*, i64, i64 } }">
  %5 = llvm.insertvalue %1, %4[1] : !llvm<"{ i64, float, { float*, i64, i64 } }">
  %6 = llvm.insertvalue %2, %5[2] : !llvm<"{ i64, float, { float*, i64, i64 } }">
  llvm.return %6 : !llvm<"{ i64, float, { float*, i64, i64 } }">
}


// CHECK-LABEL: define void @multireturn_caller() {
func @multireturn_caller() {
// CHECK-NEXT:   %1 = call { i64, float, { float*, i64, i64 } } @multireturn()
// CHECK-NEXT:   [[ret0:%[0-9]+]] = extractvalue { i64, float, { float*, i64, i64 } } %1, 0
// CHECK-NEXT:   [[ret1:%[0-9]+]] = extractvalue { i64, float, { float*, i64, i64 } } %1, 1
// CHECK-NEXT:   [[ret2:%[0-9]+]] = extractvalue { i64, float, { float*, i64, i64 } } %1, 2
  %0 = llvm.call @multireturn() : () -> !llvm<"{ i64, float, { float*, i64, i64 } }">
  %1 = llvm.extractvalue %0[0] : !llvm<"{ i64, float, { float*, i64, i64 } }">
  %2 = llvm.extractvalue %0[1] : !llvm<"{ i64, float, { float*, i64, i64 } }">
  %3 = llvm.extractvalue %0[2] : !llvm<"{ i64, float, { float*, i64, i64 } }">
  %4 = llvm.constant(42) : !llvm.i64
// CHECK:   add i64 [[ret0]], 42
  %5 = llvm.add %1, %4 : !llvm.i64
  %6 = llvm.constant(4.200000e+01 : f32) : !llvm.float
// CHECK:   fadd float [[ret1]], 4.200000e+01
  %7 = llvm.fadd %2, %6 : !llvm.float
  %8 = llvm.constant(0 : index) : !llvm.i64
  %9 = llvm.constant(42 : index) : !llvm.i64
// CHECK:   extractvalue { float*, i64, i64 } [[ret2]], 0
  %10 = llvm.extractvalue %3[1] : !llvm<"{ float*, i64, i64 }">
  %11 = llvm.constant(10 : index) : !llvm.i64
  %12 = llvm.extractvalue %3[2] : !llvm<"{ float*, i64, i64 }">
  %13 = llvm.mul %8, %10 : !llvm.i64
  %14 = llvm.add %13, %8 : !llvm.i64
  %15 = llvm.mul %14, %11 : !llvm.i64
  %16 = llvm.add %15, %8 : !llvm.i64
  %17 = llvm.mul %16, %12 : !llvm.i64
  %18 = llvm.add %17, %8 : !llvm.i64
  %19 = llvm.extractvalue %3[0] : !llvm<"{ float*, i64, i64 }">
  %20 = llvm.getelementptr %19[%18] : (!llvm<"float*">, !llvm.i64) -> !llvm<"float*">
  %21 = llvm.load %20 : !llvm<"float*">
  llvm.return
}

// CHECK-LABEL: define <4 x float> @vector_ops(<4 x float> {{%.*}}, <4 x i1> {{%.*}}, <4 x i64> {{%.*}}) {
func @vector_ops(%arg0: !llvm<"<4 x float>">, %arg1: !llvm<"<4 x i1>">, %arg2: !llvm<"<4 x i64>">) -> !llvm<"<4 x float>"> {
  %0 = llvm.constant(dense<4.200000e+01> : vector<4xf32>) : !llvm<"<4 x float>">
// CHECK-NEXT: %4 = fadd <4 x float> %0, <float 4.200000e+01, float 4.200000e+01, float 4.200000e+01, float 4.200000e+01>
  %1 = llvm.fadd %arg0, %0 : !llvm<"<4 x float>">
// CHECK-NEXT: %5 = select <4 x i1> %1, <4 x float> %4, <4 x float> %0
  %2 = llvm.select %arg1, %1, %arg0 : !llvm<"<4 x i1>">, !llvm<"<4 x float>">
// CHECK-NEXT: %6 = sdiv <4 x i64> %2, %2
  %3 = llvm.sdiv %arg2, %arg2 : !llvm<"<4 x i64>">
// CHECK-NEXT: %7 = udiv <4 x i64> %2, %2
  %4 = llvm.udiv %arg2, %arg2 : !llvm<"<4 x i64>">
// CHECK-NEXT: %8 = srem <4 x i64> %2, %2
  %5 = llvm.srem %arg2, %arg2 : !llvm<"<4 x i64>">
// CHECK-NEXT: %9 = urem <4 x i64> %2, %2
  %6 = llvm.urem %arg2, %arg2 : !llvm<"<4 x i64>">
// CHECK-NEXT: %10 = fdiv <4 x float> %0, <float 4.200000e+01, float 4.200000e+01, float 4.200000e+01, float 4.200000e+01>
  %7 = llvm.fdiv %arg0, %0 : !llvm<"<4 x float>">
// CHECK-NEXT: %11 = frem <4 x float> %0, <float 4.200000e+01, float 4.200000e+01, float 4.200000e+01, float 4.200000e+01>
  %8 = llvm.frem %arg0, %0 : !llvm<"<4 x float>">
// CHECK-NEXT: %12 = and <4 x i64> %2, %2
  %9 = llvm.and %arg2, %arg2 : !llvm<"<4 x i64>">
// CHECK-NEXT: %13 = or <4 x i64> %2, %2
  %10 = llvm.or %arg2, %arg2 : !llvm<"<4 x i64>">
// CHECK-NEXT: %14 = xor <4 x i64> %2, %2
  %11 = llvm.xor %arg2, %arg2 : !llvm<"<4 x i64>">
// CHECK-NEXT:    ret <4 x float> %4
  llvm.return %1 : !llvm<"<4 x float>">
}

// CHECK-LABEL: @ops
func @ops(%arg0: !llvm.float, %arg1: !llvm.float, %arg2: !llvm.i32, %arg3: !llvm.i32) -> !llvm<"{ float, i32 }"> {
// CHECK-NEXT: fsub float %0, %1
  %0 = llvm.fsub %arg0, %arg1 : !llvm.float
// CHECK-NEXT: %6 = sub i32 %2, %3
  %1 = llvm.sub %arg2, %arg3 : !llvm.i32
// CHECK-NEXT: %7 = icmp slt i32 %2, %6
  %2 = llvm.icmp "slt" %arg2, %1 : !llvm.i32
// CHECK-NEXT: %8 = select i1 %7, i32 %2, i32 %6
  %3 = llvm.select %2, %arg2, %1 : !llvm.i1, !llvm.i32
// CHECK-NEXT: %9 = sdiv i32 %2, %3
  %4 = llvm.sdiv %arg2, %arg3 : !llvm.i32
// CHECK-NEXT: %10 = udiv i32 %2, %3
  %5 = llvm.udiv %arg2, %arg3 : !llvm.i32
// CHECK-NEXT: %11 = srem i32 %2, %3
  %6 = llvm.srem %arg2, %arg3 : !llvm.i32
// CHECK-NEXT: %12 = urem i32 %2, %3
  %7 = llvm.urem %arg2, %arg3 : !llvm.i32

  %8 = llvm.undef : !llvm<"{ float, i32 }">
  %9 = llvm.insertvalue %0, %8[0] : !llvm<"{ float, i32 }">
  %10 = llvm.insertvalue %3, %9[1] : !llvm<"{ float, i32 }">

// CHECK: %15 = fdiv float %0, %1
  %11 = llvm.fdiv %arg0, %arg1 : !llvm.float
// CHECK-NEXT: %16 = frem float %0, %1
  %12 = llvm.frem %arg0, %arg1 : !llvm.float

// CHECK-NEXT: %17 = and i32 %2, %3
  %13 = llvm.and %arg2, %arg3 : !llvm.i32
// CHECK-NEXT: %18 = or i32 %2, %3
  %14 = llvm.or %arg2, %arg3 : !llvm.i32
// CHECK-NEXT: %19 = xor i32 %2, %3
  %15 = llvm.xor %arg2, %arg3 : !llvm.i32

  llvm.return %10 : !llvm<"{ float, i32 }">
}

//
// Indirect function calls
//

// CHECK-LABEL: define void @indirect_const_call(i64 {{%.*}}) {
func @indirect_const_call(%arg0: !llvm.i64) {
// CHECK-NEXT:  call void @body(i64 %0)
  %0 = llvm.constant(@body) : !llvm<"void (i64)*">
  llvm.call %0(%arg0) : (!llvm.i64) -> ()
// CHECK-NEXT:  ret void
  llvm.return
}

// CHECK-LABEL: define i32 @indirect_call(i32 (float)* {{%.*}}, float {{%.*}}) {
func @indirect_call(%arg0: !llvm<"i32 (float)*">, %arg1: !llvm.float) -> !llvm.i32 {
// CHECK-NEXT:  %3 = call i32 %0(float %1)
  %0 = llvm.call %arg0(%arg1) : (!llvm.float) -> !llvm.i32
// CHECK-NEXT:  ret i32 %3
  llvm.return %0 : !llvm.i32
}

//
// Check that we properly construct phi nodes in the blocks that have the same
// predecessor more than once.
//

// CHECK-LABEL: define void @cond_br_arguments(i1 {{%.*}}, i1 {{%.*}}) {
func @cond_br_arguments(%arg0: !llvm.i1, %arg1: !llvm.i1) {
// CHECK-NEXT:   br i1 %0, label %3, label %5
  llvm.cond_br %arg0, ^bb1(%arg0 : !llvm.i1), ^bb2

// CHECK:      3:
// CHECK-NEXT:   %4 = phi i1 [ %1, %5 ], [ %0, %2 ]
^bb1(%0 : !llvm.i1):
// CHECK-NEXT:   ret void
  llvm.return

// CHECK:      5:
^bb2:
// CHECK-NEXT:   br label %3
  llvm.br ^bb1(%arg1 : !llvm.i1)
}

// CHECK-LABEL: define void @llvm_noalias(float* noalias {{%*.}}) {
func @llvm_noalias(%arg0: !llvm<"float*"> {llvm.noalias = true}) {
  llvm.return
}

// CHECK-LABEL: @llvm_varargs(...) 
func @llvm_varargs()
  attributes {std.varargs = true}

func @intpointerconversion(%arg0 : !llvm.i32) -> !llvm.i32 {
// CHECK:      %2 = inttoptr i32 %0 to i32*
// CHECK-NEXT: %3 = ptrtoint i32* %2 to i32
  %1 = llvm.inttoptr %arg0 : !llvm.i32 to !llvm<"i32*">
  %2 = llvm.ptrtoint %1 : !llvm<"i32*"> to !llvm.i32
  llvm.return %2 : !llvm.i32
}

func @stringconstant() -> !llvm<"i8*"> {
  %1 = llvm.constant("Hello world!") : !llvm<"i8*">
  // CHECK: ret [12 x i8] c"Hello world!"
  llvm.return %1 : !llvm<"i8*">
}

func @noreach() {
// CHECK:    unreachable
  llvm.unreachable
}

// CHECK-LABEL: define void @fcmp
func @fcmp(%arg0: !llvm.float, %arg1: !llvm.float) {
  // CHECK: fcmp oeq float %0, %1
  // CHECK-NEXT: fcmp ogt float %0, %1
  // CHECK-NEXT: fcmp oge float %0, %1
  // CHECK-NEXT: fcmp olt float %0, %1
  // CHECK-NEXT: fcmp ole float %0, %1
  // CHECK-NEXT: fcmp one float %0, %1
  // CHECK-NEXT: fcmp ord float %0, %1
  // CHECK-NEXT: fcmp ueq float %0, %1
  // CHECK-NEXT: fcmp ugt float %0, %1
  // CHECK-NEXT: fcmp uge float %0, %1
  // CHECK-NEXT: fcmp ult float %0, %1
  // CHECK-NEXT: fcmp ule float %0, %1
  // CHECK-NEXT: fcmp une float %0, %1
  // CHECK-NEXT: fcmp uno float %0, %1
  %0 = llvm.fcmp "oeq" %arg0, %arg1 : !llvm.float
  %1 = llvm.fcmp "ogt" %arg0, %arg1 : !llvm.float
  %2 = llvm.fcmp "oge" %arg0, %arg1 : !llvm.float
  %3 = llvm.fcmp "olt" %arg0, %arg1 : !llvm.float
  %4 = llvm.fcmp "ole" %arg0, %arg1 : !llvm.float
  %5 = llvm.fcmp "one" %arg0, %arg1 : !llvm.float
  %6 = llvm.fcmp "ord" %arg0, %arg1 : !llvm.float
  %7 = llvm.fcmp "ueq" %arg0, %arg1 : !llvm.float
  %8 = llvm.fcmp "ugt" %arg0, %arg1 : !llvm.float
  %9 = llvm.fcmp "uge" %arg0, %arg1 : !llvm.float
  %10 = llvm.fcmp "ult" %arg0, %arg1 : !llvm.float
  %11 = llvm.fcmp "ule" %arg0, %arg1 : !llvm.float
  %12 = llvm.fcmp "une" %arg0, %arg1 : !llvm.float
  %13 = llvm.fcmp "uno" %arg0, %arg1 : !llvm.float
  llvm.return
}
