// RUN: mlir-opt %s | FileCheck %s

// CHECK-LABEL: func @ops(%arg0: !llvm<"i32">, %arg1: !llvm<"float">)
func @ops(%arg0 : !llvm<"i32">, %arg1 : !llvm<"float">) {
// Integer artithmetics binary instructions.
//
// CHECK-NEXT:  %0 = "llvm.add"(%arg0, %arg0) : (!llvm<"i32">, !llvm<"i32">) -> !llvm<"i32">
// CHECK-NEXT:  %1 = "llvm.sub"(%arg0, %arg0) : (!llvm<"i32">, !llvm<"i32">) -> !llvm<"i32">
// CHECK-NEXT:  %2 = "llvm.mul"(%arg0, %arg0) : (!llvm<"i32">, !llvm<"i32">) -> !llvm<"i32">
// CHECK-NEXT:  %3 = "llvm.udiv"(%arg0, %arg0) : (!llvm<"i32">, !llvm<"i32">) -> !llvm<"i32">
// CHECK-NEXT:  %4 = "llvm.sdiv"(%arg0, %arg0) : (!llvm<"i32">, !llvm<"i32">) -> !llvm<"i32">
// CHECK-NEXT:  %5 = "llvm.urem"(%arg0, %arg0) : (!llvm<"i32">, !llvm<"i32">) -> !llvm<"i32">
// CHECK-NEXT:  %6 = "llvm.srem"(%arg0, %arg0) : (!llvm<"i32">, !llvm<"i32">) -> !llvm<"i32">
// CHECK-NEXT:  %7 = "llvm.icmp"(%arg0, %arg0) {predicate: 1} : (!llvm<"i32">, !llvm<"i32">) -> !llvm<"i1">
  %0 = "llvm.add"(%arg0, %arg0) : (!llvm<"i32">, !llvm<"i32">) -> !llvm<"i32">
  %1 = "llvm.sub"(%arg0, %arg0) : (!llvm<"i32">, !llvm<"i32">) -> !llvm<"i32">
  %2 = "llvm.mul"(%arg0, %arg0) : (!llvm<"i32">, !llvm<"i32">) -> !llvm<"i32">
  %3 = "llvm.udiv"(%arg0, %arg0) : (!llvm<"i32">, !llvm<"i32">) -> !llvm<"i32">
  %4 = "llvm.sdiv"(%arg0, %arg0) : (!llvm<"i32">, !llvm<"i32">) -> !llvm<"i32">
  %5 = "llvm.urem"(%arg0, %arg0) : (!llvm<"i32">, !llvm<"i32">) -> !llvm<"i32">
  %6 = "llvm.srem"(%arg0, %arg0) : (!llvm<"i32">, !llvm<"i32">) -> !llvm<"i32">
  %7 = "llvm.icmp"(%arg0, %arg0) {predicate: 1} : (!llvm<"i32">, !llvm<"i32">) -> !llvm<"i1">

// Floating point binary instructions.
//
// CHECK-NEXT:  %8 = "llvm.fadd"(%arg1, %arg1) : (!llvm<"float">, !llvm<"float">) -> !llvm<"float">
// CHECK-NEXT:  %9 = "llvm.fsub"(%arg1, %arg1) : (!llvm<"float">, !llvm<"float">) -> !llvm<"float">
// CHECK-NEXT:  %10 = "llvm.fmul"(%arg1, %arg1) : (!llvm<"float">, !llvm<"float">) -> !llvm<"float">
// CHECK-NEXT:  %11 = "llvm.fdiv"(%arg1, %arg1) : (!llvm<"float">, !llvm<"float">) -> !llvm<"float">
// CHECK-NEXT:  %12 = "llvm.frem"(%arg1, %arg1) : (!llvm<"float">, !llvm<"float">) -> !llvm<"float">
  %8 = "llvm.fadd"(%arg1, %arg1) : (!llvm<"float">, !llvm<"float">) -> !llvm<"float">
  %9 = "llvm.fsub"(%arg1, %arg1) : (!llvm<"float">, !llvm<"float">) -> !llvm<"float">
  %10 = "llvm.fmul"(%arg1, %arg1) : (!llvm<"float">, !llvm<"float">) -> !llvm<"float">
  %11 = "llvm.fdiv"(%arg1, %arg1) : (!llvm<"float">, !llvm<"float">) -> !llvm<"float">
  %12 = "llvm.frem"(%arg1, %arg1) : (!llvm<"float">, !llvm<"float">) -> !llvm<"float">

// Memory-related instructions.
//
// CHECK-NEXT:  %13 = "llvm.alloca"(%arg0) : (!llvm<"i32">) -> !llvm<"double*">
// CHECK-NEXT:  %14 = "llvm.getelementptr"(%13, %arg0, %arg0) : (!llvm<"double*">, !llvm<"i32">, !llvm<"i32">) -> !llvm<"double*">
// CHECK-NEXT:  %15 = "llvm.load"(%14) : (!llvm<"double*">) -> !llvm<"double">
// CHECK-NEXT:  "llvm.store"(%15, %13) : (!llvm<"double">, !llvm<"double*">) -> ()
// CHECK-NEXT:  %16 = "llvm.bitcast"(%13) : (!llvm<"double*">) -> !llvm<"i64*">
  %13 = "llvm.alloca"(%arg0) : (!llvm<"i32">) -> !llvm<"double*">
  %14 = "llvm.getelementptr"(%13, %arg0, %arg0) : (!llvm<"double*">, !llvm<"i32">, !llvm<"i32">) -> !llvm<"double*">
  %15 = "llvm.load"(%14) : (!llvm<"double*">) -> !llvm<"double">
  "llvm.store"(%15, %13) : (!llvm<"double">, !llvm<"double*">) -> ()
  %16 = "llvm.bitcast"(%13) : (!llvm<"double*">) -> !llvm<"i64*">

// Function call-related instructions.
//
// CHECK-NEXT:  %17 = "llvm.call"(%arg0) {callee: @foo : (!llvm<"i32">) -> !llvm<"{ i32, double, i32 }">} : (!llvm<"i32">) -> !llvm<"{ i32, double, i32 }">
// CHECK-NEXT:  %18 = "llvm.extractvalue"(%17) {position: [0]} : (!llvm<"{ i32, double, i32 }">) -> !llvm<"i32">
// CHECK-NEXT:  %19 = "llvm.insertvalue"(%17, %18) {position: [2]} : (!llvm<"{ i32, double, i32 }">, !llvm<"i32">) -> !llvm<"{ i32, double, i32 }">
  %17 = "llvm.call"(%arg0) {callee: @foo : (!llvm<"i32">) -> !llvm<"{ i32, double, i32 }">}
         : (!llvm<"i32">) -> !llvm<"{ i32, double, i32 }">
  %18 = "llvm.extractvalue"(%17) {position: [0]} : (!llvm<"{ i32, double, i32 }">) -> !llvm<"i32">
  %19 = "llvm.insertvalue"(%17, %18) {position: [2]} : (!llvm<"{ i32, double, i32 }">, !llvm<"i32">) -> !llvm<"{ i32, double, i32 }">

// Terminator instructions and their successors.
//
// CHECK:       "llvm.br"()[^bb1] : () -> ()
  "llvm.br"()[^bb1] : () -> ()

^bb1:
// CHECK:       "llvm.cond_br"(%7)[^bb2, ^bb1] : (!llvm<"i1">) -> ()
  "llvm.cond_br"(%7)[^bb2,^bb1] : (!llvm<"i1">) -> ()

^bb2:
// CHECK:       %20 = "llvm.pseudo.undef"() : () -> !llvm<"{ i32, double, i32 }">
// CHECK-NEXT:  %21 = "llvm.pseudo.constant"() {value: 42} : () -> !llvm<"i47">
// CHECK-NEXT:  "llvm.return"() : () -> ()
  %20 = "llvm.pseudo.undef"() : () -> !llvm<"{ i32, double, i32 }">
  %21 = "llvm.pseudo.constant"() {value: 42} : () -> !llvm<"i47">
  "llvm.return"() : () -> ()
}

// An larger self-contained function.
// CHECK-LABEL:func @foo(%arg0: !llvm<"i32">) -> !llvm<"{ i32, double, i32 }"> {
func @foo(%arg0: !llvm<"i32">) -> !llvm<"{ i32, double, i32 }"> {
// CHECK-NEXT:  %0 = "llvm.pseudo.constant"() {value: 3} : () -> !llvm<"i32">
// CHECK-NEXT:  %1 = "llvm.pseudo.constant"() {value: 3} : () -> !llvm<"i32">
// CHECK-NEXT:  %2 = "llvm.pseudo.constant"() {value: 4.200000e+01} : () -> !llvm<"double">
// CHECK-NEXT:  %3 = "llvm.pseudo.constant"() {value: 4.200000e+01} : () -> !llvm<"double">
// CHECK-NEXT:  %4 = "llvm.add"(%0, %1) : (!llvm<"i32">, !llvm<"i32">) -> !llvm<"i32">
// CHECK-NEXT:  %5 = "llvm.mul"(%4, %1) : (!llvm<"i32">, !llvm<"i32">) -> !llvm<"i32">
// CHECK-NEXT:  %6 = "llvm.fadd"(%2, %3) : (!llvm<"double">, !llvm<"double">) -> !llvm<"double">
// CHECK-NEXT:  %7 = "llvm.fsub"(%3, %6) : (!llvm<"double">, !llvm<"double">) -> !llvm<"double">
// CHECK-NEXT:  %8 = "llvm.pseudo.constant"() {value: 1} : () -> !llvm<"i1">
// CHECK-NEXT:  "llvm.cond_br"(%8)[^bb1(%4 : !llvm<"i32">), ^bb2(%4 : !llvm<"i32">)] : (!llvm<"i1">) -> ()
  %0 = "llvm.pseudo.constant"() {value: 3} : () -> !llvm<"i32">
  %1 = "llvm.pseudo.constant"() {value: 3} : () -> !llvm<"i32">
  %2 = "llvm.pseudo.constant"() {value: 4.200000e+01} : () -> !llvm<"double">
  %3 = "llvm.pseudo.constant"() {value: 4.200000e+01} : () -> !llvm<"double">
  %4 = "llvm.add"(%0, %1) : (!llvm<"i32">, !llvm<"i32">) -> !llvm<"i32">
  %5 = "llvm.mul"(%4, %1) : (!llvm<"i32">, !llvm<"i32">) -> !llvm<"i32">
  %6 = "llvm.fadd"(%2, %3) : (!llvm<"double">, !llvm<"double">) -> !llvm<"double">
  %7 = "llvm.fsub"(%3, %6) : (!llvm<"double">, !llvm<"double">) -> !llvm<"double">
  %8 = "llvm.pseudo.constant"() {value: 1} : () -> !llvm<"i1">
  "llvm.cond_br"(%8)[^bb1(%4 : !llvm<"i32">), ^bb2(%4 : !llvm<"i32">)] : (!llvm<"i1">) -> ()
// CHECK-NEXT:^bb1(%9: !llvm<"i32">):	// pred: ^bb0
// CHECK-NEXT:  %10 = "llvm.call"(%9) {callee: @foo : (!llvm<"i32">) -> !llvm<"{ i32, double, i32 }">} : (!llvm<"i32">) -> !llvm<"{ i32, double, i32 }">
// CHECK-NEXT:  %11 = "llvm.extractvalue"(%10) {position: [0]} : (!llvm<"{ i32, double, i32 }">) -> !llvm<"i32">
// CHECK-NEXT:  %12 = "llvm.extractvalue"(%10) {position: [1]} : (!llvm<"{ i32, double, i32 }">) -> !llvm<"double">
// CHECK-NEXT:  %13 = "llvm.extractvalue"(%10) {position: [2]} : (!llvm<"{ i32, double, i32 }">) -> !llvm<"i32">
// CHECK-NEXT:  %14 = "llvm.pseudo.undef"() : () -> !llvm<"{ i32, double, i32 }">
// CHECK-NEXT:  %15 = "llvm.insertvalue"(%14, %5) {position: [0]} : (!llvm<"{ i32, double, i32 }">, !llvm<"i32">) -> !llvm<"{ i32, double, i32 }">
// CHECK-NEXT:  %16 = "llvm.insertvalue"(%15, %7) {position: [1]} : (!llvm<"{ i32, double, i32 }">, !llvm<"double">) -> !llvm<"{ i32, double, i32 }">
// CHECK-NEXT:  %17 = "llvm.insertvalue"(%16, %11) {position: [2]} : (!llvm<"{ i32, double, i32 }">, !llvm<"i32">) -> !llvm<"{ i32, double, i32 }">
// CHECK-NEXT:  "llvm.return"(%17) : (!llvm<"{ i32, double, i32 }">) -> ()
^bb1(%9: !llvm<"i32">):	// pred: ^bb0
  %10 = "llvm.call"(%9) {callee: @foo : (!llvm<"i32">) -> !llvm<"{ i32, double, i32 }">} : (!llvm<"i32">) -> !llvm<"{ i32, double, i32 }">
  %11 = "llvm.extractvalue"(%10) {position: [0]} : (!llvm<"{ i32, double, i32 }">) -> !llvm<"i32">
  %12 = "llvm.extractvalue"(%10) {position: [1]} : (!llvm<"{ i32, double, i32 }">) -> !llvm<"double">
  %13 = "llvm.extractvalue"(%10) {position: [2]} : (!llvm<"{ i32, double, i32 }">) -> !llvm<"i32">
  %14 = "llvm.pseudo.undef"() : () -> !llvm<"{ i32, double, i32 }">
  %15 = "llvm.insertvalue"(%14, %5) {position: [0]} : (!llvm<"{ i32, double, i32 }">, !llvm<"i32">) -> !llvm<"{ i32, double, i32 }">
  %16 = "llvm.insertvalue"(%15, %7) {position: [1]} : (!llvm<"{ i32, double, i32 }">, !llvm<"double">) -> !llvm<"{ i32, double, i32 }">
  %17 = "llvm.insertvalue"(%16, %11) {position: [2]} : (!llvm<"{ i32, double, i32 }">, !llvm<"i32">) -> !llvm<"{ i32, double, i32 }">
  "llvm.return"(%17) : (!llvm<"{ i32, double, i32 }">) -> ()
// CHECK-NEXT:^bb2(%18: !llvm<"i32">):	// pred: ^bb0
// CHECK-NEXT:  %19 = "llvm.pseudo.undef"() : () -> !llvm<"{ i32, double, i32 }">
// CHECK-NEXT:  %20 = "llvm.insertvalue"(%19, %18) {position: [0]} : (!llvm<"{ i32, double, i32 }">, !llvm<"i32">) -> !llvm<"{ i32, double, i32 }">
// CHECK-NEXT:  %21 = "llvm.insertvalue"(%20, %7) {position: [1]} : (!llvm<"{ i32, double, i32 }">, !llvm<"double">) -> !llvm<"{ i32, double, i32 }">
// CHECK-NEXT:  %22 = "llvm.insertvalue"(%21, %5) {position: [2]} : (!llvm<"{ i32, double, i32 }">, !llvm<"i32">) -> !llvm<"{ i32, double, i32 }">
// CHECK-NEXT:  "llvm.return"(%22) : (!llvm<"{ i32, double, i32 }">) -> ()
^bb2(%18: !llvm<"i32">):	// pred: ^bb0
  %19 = "llvm.pseudo.undef"() : () -> !llvm<"{ i32, double, i32 }">
  %20 = "llvm.insertvalue"(%19, %18) {position: [0]} : (!llvm<"{ i32, double, i32 }">, !llvm<"i32">) -> !llvm<"{ i32, double, i32 }">
  %21 = "llvm.insertvalue"(%20, %7) {position: [1]} : (!llvm<"{ i32, double, i32 }">, !llvm<"double">) -> !llvm<"{ i32, double, i32 }">
  %22 = "llvm.insertvalue"(%21, %5) {position: [2]} : (!llvm<"{ i32, double, i32 }">, !llvm<"i32">) -> !llvm<"{ i32, double, i32 }">
  "llvm.return"(%22) : (!llvm<"{ i32, double, i32 }">) -> ()
}

