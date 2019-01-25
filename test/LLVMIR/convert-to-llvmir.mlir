// RUN: mlir-opt -convert-to-llvmir %s | FileCheck %s

// CHECK-LABEL: func @empty() {
// CHECK-NEXT:   "llvm.return"() : () -> ()
// CHECK-NEXT: }
func @empty() {
^bb0:
  return
}

// CHECK-LABEL: func @body(!llvm<"i64">)
func @body(index)

// CHECK-LABEL: func @simple_loop() {
func @simple_loop() {
^bb0:
// CHECK-NEXT: "llvm.br"()[^bb1] : () -> ()
  br ^bb1

// CHECK-NEXT: ^bb1:	// pred: ^bb0
// CHECK-NEXT:   {{.*}} = "llvm.constant"() {value: 1} : () -> !llvm<"i64">
// CHECK-NEXT:   {{.*}} = "llvm.constant"() {value: 42} : () -> !llvm<"i64">
// CHECK-NEXT:   "llvm.br"()[^bb2({{.*}} : !llvm<"i64">)] : () -> ()
^bb1:	// pred: ^bb0
  %c1 = constant 1 : index
  %c42 = constant 42 : index
  br ^bb2(%c1 : index)

// CHECK:      ^bb2({{.*}}: !llvm<"i64">):	// 2 preds: ^bb1, ^bb3
// CHECK-NEXT:   {{.*}} = "llvm.icmp"({{.*}}, {{.*}}) {predicate: 2} : (!llvm<"i64">, !llvm<"i64">) -> !llvm<"i1">
// CHECK-NEXT:   "llvm.cond_br"({{.*}})[^bb3, ^bb4] : (!llvm<"i1">) -> ()
^bb2(%0: index):	// 2 preds: ^bb1, ^bb3
  %1 = cmpi "slt", %0, %c42 : index
  cond_br %1, ^bb3, ^bb4

// CHECK:      ^bb3:	// pred: ^bb2
// CHECK-NEXT:   "llvm.call0"({{.*}}) {callee: @body : (!llvm<"i64">) -> ()} : (!llvm<"i64">) -> ()
// CHECK-NEXT:   {{.*}} = "llvm.constant"() {value: 1} : () -> !llvm<"i64">
// CHECK-NEXT:   {{.*}} = "llvm.add"({{.*}}, {{.*}}) : (!llvm<"i64">, !llvm<"i64">) -> !llvm<"i64">
// CHECK-NEXT:   "llvm.br"()[^bb2({{.*}} : !llvm<"i64">)] : () -> ()
^bb3:	// pred: ^bb2
  call @body(%0) : (index) -> ()
  %c1_0 = constant 1 : index
  %2 = addi %0, %c1_0 : index
  br ^bb2(%2 : index)

// CHECK:      ^bb4:	// pred: ^bb2
// CHECK-NEXT:   "llvm.return"() : () -> ()
^bb4:	// pred: ^bb2
  return
}

// CHECK-LABEL: func @simple_caller() {
// CHECK-NEXT:   "llvm.call0"() {callee: @simple_loop : () -> ()} : () -> ()
// CHECK-NEXT:   "llvm.return"() : () -> ()
// CHECK-NEXT: }
func @simple_caller() {
^bb0:
  call @simple_loop() : () -> ()
  return
}

// CHECK-LABEL: func @ml_caller() {
// CHECK-NEXT:   "llvm.call0"() {callee: @simple_loop : () -> ()} : () -> ()
// CHECK-NEXT:   "llvm.call0"() {callee: @more_imperfectly_nested_loops : () -> ()} : () -> ()
// CHECK-NEXT:   "llvm.return"() : () -> ()
// CHECK-NEXT: }
func @ml_caller() {
^bb0:
  call @simple_loop() : () -> ()
  call @more_imperfectly_nested_loops() : () -> ()
  return
}

// CHECK-LABEL: func @body_args(!llvm<"i64">) -> !llvm<"i64">
func @body_args(index) -> index
// CHECK-LABEL: func @other(!llvm<"i64">, !llvm<"i32">) -> !llvm<"i32">
func @other(index, i32) -> i32

// CHECK-LABEL: func @func_args(%arg0: !llvm<"i32">, %arg1: !llvm<"i32">) -> !llvm<"i32"> {
// CHECK-NEXT:   {{.*}} = "llvm.constant"() {value: 0} : () -> !llvm<"i32">
// CHECK-NEXT:   "llvm.br"()[^bb1] : () -> ()
func @func_args(i32, i32) -> i32 {
^bb0(%arg0: i32, %arg1: i32):
  %c0_i32 = constant 0 : i32
  br ^bb1

// CHECK-NEXT: ^bb1:	// pred: ^bb0
// CHECK-NEXT:   {{.*}} = "llvm.constant"() {value: 0} : () -> !llvm<"i64">
// CHECK-NEXT:   {{.*}} = "llvm.constant"() {value: 42} : () -> !llvm<"i64">
// CHECK-NEXT:   "llvm.br"()[^bb2({{.*}} : !llvm<"i64">)] : () -> ()
^bb1:	// pred: ^bb0
  %c0 = constant 0 : index
  %c42 = constant 42 : index
  br ^bb2(%c0 : index)

// CHECK-NEXT: ^bb2({{.*}}: !llvm<"i64">):	// 2 preds: ^bb1, ^bb3
// CHECK-NEXT:   {{.*}} = "llvm.icmp"({{.*}}, {{.*}}) {predicate: 2} : (!llvm<"i64">, !llvm<"i64">) -> !llvm<"i1">
// CHECK-NEXT:   "llvm.cond_br"({{.*}})[^bb3, ^bb4] : (!llvm<"i1">) -> ()
^bb2(%0: index):	// 2 preds: ^bb1, ^bb3
  %1 = cmpi "slt", %0, %c42 : index
  cond_br %1, ^bb3, ^bb4

// CHECK-NEXT: ^bb3:	// pred: ^bb2
// CHECK-NEXT:   {{.*}} = "llvm.call"({{.*}}) {callee: @body_args : (!llvm<"i64">) -> !llvm<"i64">} : (!llvm<"i64">) -> !llvm<"i64">
// CHECK-NEXT:   {{.*}} = "llvm.call"({{.*}}, %arg0) {callee: @other : (!llvm<"i64">, !llvm<"i32">) -> !llvm<"i32">} : (!llvm<"i64">, !llvm<"i32">) -> !llvm<"i32">
// CHECK-NEXT:   {{.*}} = "llvm.call"({{.*}}, {{.*}}) {callee: @other : (!llvm<"i64">, !llvm<"i32">) -> !llvm<"i32">} : (!llvm<"i64">, !llvm<"i32">) -> !llvm<"i32">
// CHECK-NEXT:   {{.*}} = "llvm.call"({{.*}}, %arg1) {callee: @other : (!llvm<"i64">, !llvm<"i32">) -> !llvm<"i32">} : (!llvm<"i64">, !llvm<"i32">) -> !llvm<"i32">
// CHECK-NEXT:   {{.*}} = "llvm.constant"() {value: 1} : () -> !llvm<"i64">
// CHECK-NEXT:   {{.*}} = "llvm.add"({{.*}}, {{.*}}) : (!llvm<"i64">, !llvm<"i64">) -> !llvm<"i64">
// CHECK-NEXT:   "llvm.br"()[^bb2({{.*}} : !llvm<"i64">)] : () -> ()
^bb3:	// pred: ^bb2
  %2 = call @body_args(%0) : (index) -> index
  %3 = call @other(%2, %arg0) : (index, i32) -> i32
  %4 = call @other(%2, %3) : (index, i32) -> i32
  %5 = call @other(%2, %arg1) : (index, i32) -> i32
  %c1 = constant 1 : index
  %6 = addi %0, %c1 : index
  br ^bb2(%6 : index)

// CHECK-NEXT: ^bb4:	// pred: ^bb2
// CHECK-NEXT:   {{.*}} = "llvm.constant"() {value: 0} : () -> !llvm<"i64">
// CHECK-NEXT:   {{.*}} = "llvm.call"({{.*}}, {{.*}}) {callee: @other : (!llvm<"i64">, !llvm<"i32">) -> !llvm<"i32">} : (!llvm<"i64">, !llvm<"i32">) -> !llvm<"i32">
// CHECK-NEXT:   "llvm.return"({{.*}}) : (!llvm<"i32">) -> ()
^bb4:	// pred: ^bb2
  %c0_0 = constant 0 : index
  %7 = call @other(%c0_0, %c0_i32) : (index, i32) -> i32
  return %7 : i32
}

// CHECK-LABEL: func @pre(!llvm<"i64">)
func @pre(index)

// CHECK-LABEL: func @body2(!llvm<"i64">, !llvm<"i64">)
func @body2(index, index)

// CHECK-LABEL: func @post(!llvm<"i64">)
func @post(index)

// CHECK-LABEL: func @imperfectly_nested_loops() {
// CHECK-NEXT:   "llvm.br"()[^bb1] : () -> ()
func @imperfectly_nested_loops() {
^bb0:
  br ^bb1

// CHECK-NEXT: ^bb1:	// pred: ^bb0
// CHECK-NEXT:   {{.*}} = "llvm.constant"() {value: 0} : () -> !llvm<"i64">
// CHECK-NEXT:   {{.*}} = "llvm.constant"() {value: 42} : () -> !llvm<"i64">
// CHECK-NEXT:   "llvm.br"()[^bb2({{.*}} : !llvm<"i64">)] : () -> ()
^bb1:	// pred: ^bb0
  %c0 = constant 0 : index
  %c42 = constant 42 : index
  br ^bb2(%c0 : index)

// CHECK-NEXT: ^bb2({{.*}}: !llvm<"i64">):	// 2 preds: ^bb1, ^bb7
// CHECK-NEXT:   {{.*}} = "llvm.icmp"({{.*}}, {{.*}}) {predicate: 2} : (!llvm<"i64">, !llvm<"i64">) -> !llvm<"i1">
// CHECK-NEXT:   "llvm.cond_br"({{.*}})[^bb3, ^bb8] : (!llvm<"i1">) -> ()
^bb2(%0: index):	// 2 preds: ^bb1, ^bb7
  %1 = cmpi "slt", %0, %c42 : index
  cond_br %1, ^bb3, ^bb8

// CHECK-NEXT: ^bb3:
// CHECK-NEXT:   "llvm.call0"({{.*}}) {callee: @pre : (!llvm<"i64">) -> ()} : (!llvm<"i64">) -> ()
// CHECK-NEXT:   "llvm.br"()[^bb4] : () -> ()
^bb3:	// pred: ^bb2
  call @pre(%0) : (index) -> ()
  br ^bb4

// CHECK-NEXT: ^bb4:	// pred: ^bb3
// CHECK-NEXT:   {{.*}} = "llvm.constant"() {value: 7} : () -> !llvm<"i64">
// CHECK-NEXT:   {{.*}} = "llvm.constant"() {value: 56} : () -> !llvm<"i64">
// CHECK-NEXT:   "llvm.br"()[^bb5({{.*}} : !llvm<"i64">)] : () -> ()
^bb4:	// pred: ^bb3
  %c7 = constant 7 : index
  %c56 = constant 56 : index
  br ^bb5(%c7 : index)

// CHECK-NEXT: ^bb5({{.*}}: !llvm<"i64">):	// 2 preds: ^bb4, ^bb6
// CHECK-NEXT:   {{.*}} = "llvm.icmp"({{.*}}, {{.*}}) {predicate: 2} : (!llvm<"i64">, !llvm<"i64">) -> !llvm<"i1">
// CHECK-NEXT:   "llvm.cond_br"({{.*}})[^bb6, ^bb7] : (!llvm<"i1">) -> ()
^bb5(%2: index):	// 2 preds: ^bb4, ^bb6
  %3 = cmpi "slt", %2, %c56 : index
  cond_br %3, ^bb6, ^bb7

// CHECK-NEXT: ^bb6:	// pred: ^bb5
// CHECK-NEXT:   "llvm.call0"({{.*}}, {{.*}}) {callee: @body2 : (!llvm<"i64">, !llvm<"i64">) -> ()} : (!llvm<"i64">, !llvm<"i64">) -> ()
// CHECK-NEXT:   {{.*}} = "llvm.constant"() {value: 2} : () -> !llvm<"i64">
// CHECK-NEXT:   {{.*}} = "llvm.add"({{.*}}, {{.*}}) : (!llvm<"i64">, !llvm<"i64">) -> !llvm<"i64">
// CHECK-NEXT:   "llvm.br"()[^bb5({{.*}} : !llvm<"i64">)] : () -> ()
^bb6:	// pred: ^bb5
  call @body2(%0, %2) : (index, index) -> ()
  %c2 = constant 2 : index
  %4 = addi %2, %c2 : index
  br ^bb5(%4 : index)

// CHECK-NEXT: ^bb7:	// pred: ^bb5
// CHECK-NEXT:   "llvm.call0"({{.*}}) {callee: @post : (!llvm<"i64">) -> ()} : (!llvm<"i64">) -> ()
// CHECK-NEXT:   {{.*}} = "llvm.constant"() {value: 1} : () -> !llvm<"i64">
// CHECK-NEXT:   {{.*}} = "llvm.add"({{.*}}, {{.*}}) : (!llvm<"i64">, !llvm<"i64">) -> !llvm<"i64">
// CHECK-NEXT:   "llvm.br"()[^bb2({{.*}} : !llvm<"i64">)] : () -> ()
^bb7:	// pred: ^bb5
  call @post(%0) : (index) -> ()
  %c1 = constant 1 : index
  %5 = addi %0, %c1 : index
  br ^bb2(%5 : index)

// CHECK-NEXT: ^bb8:	// pred: ^bb2
// CHECK-NEXT:   "llvm.return"() : () -> ()
^bb8:	// pred: ^bb2
  return
}

// CHECK-LABEL: func @mid(!llvm<"i64">)
func @mid(index)

// CHECK-LABEL: func @body3(!llvm<"i64">, !llvm<"i64">)
func @body3(index, index)

// A complete function transformation check.
// CHECK-LABEL: func @more_imperfectly_nested_loops() {
// CHECK-NEXT:   "llvm.br"()[^bb1] : () -> ()
// CHECK-NEXT: ^bb1:	// pred: ^bb0
// CHECK-NEXT:   {{.*}} = "llvm.constant"() {value: 0} : () -> !llvm<"i64">
// CHECK-NEXT:   {{.*}} = "llvm.constant"() {value: 42} : () -> !llvm<"i64">
// CHECK-NEXT:   "llvm.br"()[^bb2({{.*}} : !llvm<"i64">)] : () -> ()
// CHECK-NEXT: ^bb2({{.*}}: !llvm<"i64">):	// 2 preds: ^bb1, ^bb11
// CHECK-NEXT:   {{.*}} = "llvm.icmp"({{.*}}, {{.*}}) {predicate: 2} : (!llvm<"i64">, !llvm<"i64">) -> !llvm<"i1">
// CHECK-NEXT:   "llvm.cond_br"({{.*}})[^bb3, ^bb12] : (!llvm<"i1">) -> ()
// CHECK-NEXT: ^bb3:	// pred: ^bb2
// CHECK-NEXT:   "llvm.call0"({{.*}}) {callee: @pre : (!llvm<"i64">) -> ()} : (!llvm<"i64">) -> ()
// CHECK-NEXT:   "llvm.br"()[^bb4] : () -> ()
// CHECK-NEXT: ^bb4:	// pred: ^bb3
// CHECK-NEXT:   {{.*}} = "llvm.constant"() {value: 7} : () -> !llvm<"i64">
// CHECK-NEXT:   {{.*}} = "llvm.constant"() {value: 56} : () -> !llvm<"i64">
// CHECK-NEXT:   "llvm.br"()[^bb5({{.*}} : !llvm<"i64">)] : () -> ()
// CHECK-NEXT: ^bb5({{.*}}: !llvm<"i64">):	// 2 preds: ^bb4, ^bb6
// CHECK-NEXT:   {{.*}} = "llvm.icmp"({{.*}}, {{.*}}) {predicate: 2} : (!llvm<"i64">, !llvm<"i64">) -> !llvm<"i1">
// CHECK-NEXT:   "llvm.cond_br"({{.*}})[^bb6, ^bb7] : (!llvm<"i1">) -> ()
// CHECK-NEXT: ^bb6:	// pred: ^bb5
// CHECK-NEXT:   "llvm.call0"({{.*}}, {{.*}}) {callee: @body2 : (!llvm<"i64">, !llvm<"i64">) -> ()} : (!llvm<"i64">, !llvm<"i64">) -> ()
// CHECK-NEXT:   {{.*}} = "llvm.constant"() {value: 2} : () -> !llvm<"i64">
// CHECK-NEXT:   {{.*}} = "llvm.add"({{.*}}, {{.*}}) : (!llvm<"i64">, !llvm<"i64">) -> !llvm<"i64">
// CHECK-NEXT:   "llvm.br"()[^bb5({{.*}} : !llvm<"i64">)] : () -> ()
// CHECK-NEXT: ^bb7:	// pred: ^bb5
// CHECK-NEXT:   "llvm.call0"({{.*}}) {callee: @mid : (!llvm<"i64">) -> ()} : (!llvm<"i64">) -> ()
// CHECK-NEXT:   "llvm.br"()[^bb8] : () -> ()
// CHECK-NEXT: ^bb8:	// pred: ^bb7
// CHECK-NEXT:   {{.*}} = "llvm.constant"() {value: 18} : () -> !llvm<"i64">
// CHECK-NEXT:   {{.*}} = "llvm.constant"() {value: 37} : () -> !llvm<"i64">
// CHECK-NEXT:   "llvm.br"()[^bb9({{.*}} : !llvm<"i64">)] : () -> ()
// CHECK-NEXT: ^bb9({{.*}}: !llvm<"i64">):	// 2 preds: ^bb8, ^bb10
// CHECK-NEXT:   {{.*}} = "llvm.icmp"({{.*}}, {{.*}}) {predicate: 2} : (!llvm<"i64">, !llvm<"i64">) -> !llvm<"i1">
// CHECK-NEXT:   "llvm.cond_br"({{.*}})[^bb10, ^bb11] : (!llvm<"i1">) -> ()
// CHECK-NEXT: ^bb10:	// pred: ^bb9
// CHECK-NEXT:   "llvm.call0"({{.*}}, {{.*}}) {callee: @body3 : (!llvm<"i64">, !llvm<"i64">) -> ()} : (!llvm<"i64">, !llvm<"i64">) -> ()
// CHECK-NEXT:   {{.*}} = "llvm.constant"() {value: 3} : () -> !llvm<"i64">
// CHECK-NEXT:   {{.*}} = "llvm.add"({{.*}}, {{.*}}) : (!llvm<"i64">, !llvm<"i64">) -> !llvm<"i64">
// CHECK-NEXT:   "llvm.br"()[^bb9({{.*}} : !llvm<"i64">)] : () -> ()
// CHECK-NEXT: ^bb11:	// pred: ^bb9
// CHECK-NEXT:   "llvm.call0"({{.*}}) {callee: @post : (!llvm<"i64">) -> ()} : (!llvm<"i64">) -> ()
// CHECK-NEXT:   {{.*}} = "llvm.constant"() {value: 1} : () -> !llvm<"i64">
// CHECK-NEXT:   {{.*}} = "llvm.add"({{.*}}, {{.*}}) : (!llvm<"i64">, !llvm<"i64">) -> !llvm<"i64">
// CHECK-NEXT:   "llvm.br"()[^bb2({{.*}} : !llvm<"i64">)] : () -> ()
// CHECK-NEXT: ^bb12:	// pred: ^bb2
// CHECK-NEXT:   "llvm.return"() : () -> ()
// CHECK-NEXT: }
func @more_imperfectly_nested_loops() {
^bb0:
  br ^bb1
^bb1:	// pred: ^bb0
  %c0 = constant 0 : index
  %c42 = constant 42 : index
  br ^bb2(%c0 : index)
^bb2(%0: index):	// 2 preds: ^bb1, ^bb11
  %1 = cmpi "slt", %0, %c42 : index
  cond_br %1, ^bb3, ^bb12
^bb3:	// pred: ^bb2
  call @pre(%0) : (index) -> ()
  br ^bb4
^bb4:	// pred: ^bb3
  %c7 = constant 7 : index
  %c56 = constant 56 : index
  br ^bb5(%c7 : index)
^bb5(%2: index):	// 2 preds: ^bb4, ^bb6
  %3 = cmpi "slt", %2, %c56 : index
  cond_br %3, ^bb6, ^bb7
^bb6:	// pred: ^bb5
  call @body2(%0, %2) : (index, index) -> ()
  %c2 = constant 2 : index
  %4 = addi %2, %c2 : index
  br ^bb5(%4 : index)
^bb7:	// pred: ^bb5
  call @mid(%0) : (index) -> ()
  br ^bb8
^bb8:	// pred: ^bb7
  %c18 = constant 18 : index
  %c37 = constant 37 : index
  br ^bb9(%c18 : index)
^bb9(%5: index):	// 2 preds: ^bb8, ^bb10
  %6 = cmpi "slt", %5, %c37 : index
  cond_br %6, ^bb10, ^bb11
^bb10:	// pred: ^bb9
  call @body3(%0, %5) : (index, index) -> ()
  %c3 = constant 3 : index
  %7 = addi %5, %c3 : index
  br ^bb9(%7 : index)
^bb11:	// pred: ^bb9
  call @post(%0) : (index) -> ()
  %c1 = constant 1 : index
  %8 = addi %0, %c1 : index
  br ^bb2(%8 : index)
^bb12:	// pred: ^bb2
  return
}

// CHECK-LABEL: func @get_i64() -> !llvm<"i64">
func @get_i64() -> (i64)
// CHECK-LABEL: func @get_f32() -> !llvm<"float">
func @get_f32() -> (f32)
// CHECK-LABEL: func @get_memref() -> !llvm<"{ float*, i64, i64 }">
func @get_memref() -> (memref<42x?x10x?xf32>)

// CHECK-LABEL: func @multireturn() -> !llvm<"{ i64, float, { float*, i64, i64 } }"> {
func @multireturn() -> (i64, f32, memref<42x?x10x?xf32>) {
^bb0:
// CHECK-NEXT:   {{.*}} = "llvm.call"() {callee: @get_i64 : () -> !llvm<"i64">} : () -> !llvm<"i64">
// CHECK-NEXT:   {{.*}} = "llvm.call"() {callee: @get_f32 : () -> !llvm<"float">} : () -> !llvm<"float">
// CHECK-NEXT:   {{.*}} = "llvm.call"() {callee: @get_memref : () -> !llvm<"{ float*, i64, i64 }">} : () -> !llvm<"{ float*, i64, i64 }">
  %0 = call @get_i64() : () -> (i64)
  %1 = call @get_f32() : () -> (f32)
  %2 = call @get_memref() : () -> (memref<42x?x10x?xf32>)
// CHECK-NEXT:   {{.*}} = "llvm.undef"() : () -> !llvm<"{ i64, float, { float*, i64, i64 } }">
// CHECK-NEXT:   {{.*}} = "llvm.insertvalue"({{.*}}, {{.*}}) {position: [0]} : (!llvm<"{ i64, float, { float*, i64, i64 } }">, !llvm<"i64">) -> !llvm<"{ i64, float, { float*, i64, i64 } }">
// CHECK-NEXT:   {{.*}} = "llvm.insertvalue"({{.*}}, {{.*}}) {position: [1]} : (!llvm<"{ i64, float, { float*, i64, i64 } }">, !llvm<"float">) -> !llvm<"{ i64, float, { float*, i64, i64 } }">
// CHECK-NEXT:   {{.*}} = "llvm.insertvalue"({{.*}}, {{.*}}) {position: [2]} : (!llvm<"{ i64, float, { float*, i64, i64 } }">, !llvm<"{ float*, i64, i64 }">) -> !llvm<"{ i64, float, { float*, i64, i64 } }">
// CHECK-NEXT:   "llvm.return"({{.*}}) : (!llvm<"{ i64, float, { float*, i64, i64 } }">) -> ()
  return %0, %1, %2 : i64, f32, memref<42x?x10x?xf32>
}


// CHECK-LABEL: func @multireturn_caller() {
func @multireturn_caller() {
^bb0:
// CHECK-NEXT:   {{.*}} = "llvm.call"() {callee: @multireturn : () -> !llvm<"{ i64, float, { float*, i64, i64 } }">} : () -> !llvm<"{ i64, float, { float*, i64, i64 } }">
// CHECK-NEXT:   {{.*}} = "llvm.extractvalue"({{.*}}) {position: [0]} : (!llvm<"{ i64, float, { float*, i64, i64 } }">) -> !llvm<"i64">
// CHECK-NEXT:   {{.*}} = "llvm.extractvalue"({{.*}}) {position: [1]} : (!llvm<"{ i64, float, { float*, i64, i64 } }">) -> !llvm<"float">
// CHECK-NEXT:   {{.*}} = "llvm.extractvalue"({{.*}}) {position: [2]} : (!llvm<"{ i64, float, { float*, i64, i64 } }">) -> !llvm<"{ float*, i64, i64 }">
  %0 = call @multireturn() : () -> (i64, f32, memref<42x?x10x?xf32>)
  %1 = constant 42 : i64
// CHECK:   {{.*}} = "llvm.add"({{.*}}, {{.*}}) : (!llvm<"i64">, !llvm<"i64">) -> !llvm<"i64">
  %2 = addi %0#0, %1 : i64
  %3 = constant 42.0 : f32
// CHECK:   {{.*}} = "llvm.fadd"({{.*}}, {{.*}}) : (!llvm<"float">, !llvm<"float">) -> !llvm<"float">
  %4 = addf %0#1, %3 : f32
  %5 = constant 0 : index
  return
}

// CHECK-LABEL: func @vector_ops(%arg0: !llvm<"<4 x float>">, %arg1: !llvm<"<4 x i1>">, %arg2: !llvm<"<4 x i64>">) -> !llvm<"<4 x float>"> {
func @vector_ops(vector<4xf32>, vector<4xi1>, vector<4xi64>) -> vector<4xf32> {
^bb0(%arg0: vector<4xf32>, %arg1: vector<4xi1>, %arg2: vector<4xi64>):
// CHECK-NEXT:   {{.*}} = "llvm.constant"() {value: splat<vector<4xf32>, 4.200000e+01>} : () -> !llvm<"<4 x float>">
  %0 = constant splat<vector<4xf32>, 42.> : vector<4xf32>
// CHECK-NEXT:   {{.*}} = "llvm.fadd"(%arg0, {{.*}}) : (!llvm<"<4 x float>">, !llvm<"<4 x float>">) -> !llvm<"<4 x float>">
  %1 = addf %arg0, %0 : vector<4xf32>
// CHECK-NEXT:   {{.*}} = "llvm.sdiv"(%arg2, %arg2) : (!llvm<"<4 x i64>">, !llvm<"<4 x i64>">) -> !llvm<"<4 x i64>">
  %3 = divis %arg2, %arg2 : vector<4xi64>
// CHECK-NEXT:   {{.*}} = "llvm.udiv"(%arg2, %arg2) : (!llvm<"<4 x i64>">, !llvm<"<4 x i64>">) -> !llvm<"<4 x i64>">
  %4 = diviu %arg2, %arg2 : vector<4xi64>
// CHECK-NEXT:   {{.*}} = "llvm.srem"(%arg2, %arg2) : (!llvm<"<4 x i64>">, !llvm<"<4 x i64>">) -> !llvm<"<4 x i64>">
  %5 = remis %arg2, %arg2 : vector<4xi64>
// CHECK-NEXT:   {{.*}} = "llvm.urem"(%arg2, %arg2) : (!llvm<"<4 x i64>">, !llvm<"<4 x i64>">) -> !llvm<"<4 x i64>">
  %6 = remiu %arg2, %arg2 : vector<4xi64>
  return %1 : vector<4xf32>
}

// CHECK-LABEL: @ops
func @ops(f32, f32, i32, i32) -> (f32, i32) {
^bb0(%arg0: f32, %arg1: f32, %arg2: i32, %arg3: i32):
// CHECK-NEXT:   {{.*}} = "llvm.fsub"(%arg0, %arg1) : (!llvm<"float">, !llvm<"float">) -> !llvm<"float">
  %0 = subf %arg0, %arg1: f32
// CHECK-NEXT:   {{.*}} = "llvm.sub"(%arg2, %arg3) : (!llvm<"i32">, !llvm<"i32">) -> !llvm<"i32">
  %1 = subi %arg2, %arg3: i32
// CHECK-NEXT:   {{.*}} = "llvm.icmp"(%arg2, {{.*}}) {predicate: 2} : (!llvm<"i32">, !llvm<"i32">) -> !llvm<"i1">
  %2 = cmpi "slt", %arg2, %1 : i32
// CHECK-NEXT:   {{.*}} = "llvm.sdiv"(%arg2, %arg3) : (!llvm<"i32">, !llvm<"i32">) -> !llvm<"i32">
  %4 = divis %arg2, %arg3 : i32
// CHECK-NEXT:   {{.*}} = "llvm.udiv"(%arg2, %arg3) : (!llvm<"i32">, !llvm<"i32">) -> !llvm<"i32">
  %5 = diviu %arg2, %arg3 : i32
// CHECK-NEXT:   {{.*}} = "llvm.srem"(%arg2, %arg3) : (!llvm<"i32">, !llvm<"i32">) -> !llvm<"i32">
  %6 = remis %arg2, %arg3 : i32
// CHECK-NEXT:   {{.*}} = "llvm.urem"(%arg2, %arg3) : (!llvm<"i32">, !llvm<"i32">) -> !llvm<"i32">
  %7 = remiu %arg2, %arg3 : i32

  return %0, %4 : f32, i32
}

// CHECK-LABEL: @dfs_block_order
func @dfs_block_order() -> (i32) {
// CHECK-NEXT:   {{.*}} = "llvm.constant"() {value: 42} : () -> !llvm<"i32">
  %0 = constant 42 : i32
// CHECK-NEXT:   "llvm.br"()[^bb2] : () -> ()
  br ^bb2

// CHECK-NEXT: ^bb1:
// CHECK-NEXT:   {{.*}} = "llvm.add"({{.*}}, {{.*}}) : (!llvm<"i32">, !llvm<"i32">) -> !llvm<"i32">
// CHECK-NEXT:   "llvm.return"({{.*}}) : (!llvm<"i32">) -> ()
^bb1:
  %2 = addi %0, %1 : i32
  return %2 : i32

// CHECK-NEXT: ^bb2:
^bb2:
// CHECK-NEXT:   {{.*}} = "llvm.constant"() {value: 55} : () -> !llvm<"i32">
  %1 = constant 55 : i32
// CHECK-NEXT:   "llvm.br"()[^bb1] : () -> ()
  br ^bb1
}

