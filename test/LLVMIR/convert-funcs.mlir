// RUN: mlir-opt -convert-to-llvmir %s | FileCheck %s

//CHECK: func @second_order_arg(!llvm<"void ()*">)
func @second_order_arg(%arg0 : () -> ())

//CHECK: func @second_order_result() -> !llvm<"void ()*">
func @second_order_result() -> (() -> ())

//CHECK: func @second_order_multi_result() -> !llvm<"{ i32 ()*, i64 ()*, float ()* }">
func @second_order_multi_result() -> (() -> (i32), () -> (i64), () -> (f32))

//CHECK: func @third_order(!llvm<"void ()* (void ()*)*">) -> !llvm<"void ()* (void ()*)*">
func @third_order(%arg0 : (() -> ()) -> (() -> ())) -> ((() -> ()) -> (() -> ()))

//CHECK: func @fifth_order_left(!llvm<"void (void (void (void ()*)*)*)*">)
func @fifth_order_left(%arg0: (((() -> ()) -> ()) -> ()) -> ())

//CHECK: func @fifth_order_right(!llvm<"void ()* ()* ()* ()*">)
func @fifth_order_right(%arg0: () -> (() -> (() -> (() -> ()))))

//CHECK-LABEL: func @pass_through(%arg0: !llvm<"void ()*">) -> !llvm<"void ()*"> {
func @pass_through(%arg0: () -> ()) -> (() -> ()) {
//CHECK-NEXT:   "llvm.br"()[^bb1(%arg0 : !llvm<"void ()*">)] : () -> ()
  br ^bb1(%arg0 : () -> ())

//CHECK-NEXT: ^bb1(%0: !llvm<"void ()*">):	// pred: ^bb0
^bb1(%bbarg: () -> ()):
//CHECK-NEXT:   "llvm.return"(%0) : (!llvm<"void ()*">) -> ()
  return %bbarg : () -> ()
}
