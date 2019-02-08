// RUN: mlir-opt -convert-to-llvmir %s | FileCheck %s

// CHECK-LABEL: func @alloc(%arg0: !llvm<"i64">, %arg1: !llvm<"i64">) -> !llvm<"{ float*, i64, i64 }"> {
func @alloc(%arg0: index, %arg1: index) -> memref<?x42x?xf32> {
// CHECK-NEXT: %0 = "llvm.constant"() {value: 42 : index} : () -> !llvm<"i64">
// CHECK-NEXT: %1 = "llvm.mul"(%arg0, %0) : (!llvm<"i64">, !llvm<"i64">) -> !llvm<"i64">
// CHECK-NEXT: %2 = "llvm.mul"(%1, %arg1) : (!llvm<"i64">, !llvm<"i64">) -> !llvm<"i64">
// CHECK-NEXT: %3 = "llvm.undef"() : () -> !llvm<"{ float*, i64, i64 }">
// CHECK-NEXT: %4 = "llvm.constant"() {value: 4 : index} : () -> !llvm<"i64">
// CHECK-NEXT: %5 = "llvm.mul"(%2, %4) : (!llvm<"i64">, !llvm<"i64">) -> !llvm<"i64">
// CHECK-NEXT: %6 = "llvm.call"(%5) {callee: @malloc : (!llvm<"i64">) -> !llvm<"i8*">} : (!llvm<"i64">) -> !llvm<"i8*">
// CHECK-NEXT: %7 = "llvm.bitcast"(%6) : (!llvm<"i8*">) -> !llvm<"float*">
// CHECK-NEXT: %8 = "llvm.insertvalue"(%3, %7) {position: [0]} : (!llvm<"{ float*, i64, i64 }">, !llvm<"float*">) -> !llvm<"{ float*, i64, i64 }">
// CHECK-NEXT: %9 = "llvm.insertvalue"(%8, %arg0) {position: [1]} : (!llvm<"{ float*, i64, i64 }">, !llvm<"i64">) -> !llvm<"{ float*, i64, i64 }">
// CHECK-NEXT: %10 = "llvm.insertvalue"(%9, %arg1) {position: [2]} : (!llvm<"{ float*, i64, i64 }">, !llvm<"i64">) -> !llvm<"{ float*, i64, i64 }">
  %0 = alloc(%arg0, %arg1) : memref<?x42x?xf32>
// CHECK-NEXT:  "llvm.return"(%10) : (!llvm<"{ float*, i64, i64 }">) -> ()
  return %0 : memref<?x42x?xf32>
}

// CHECK-LABEL: func @dealloc(%arg0: !llvm<"{ float*, i64, i64 }">) {
func @dealloc(%arg0: memref<?x42x?xf32>) {
// CHECK-NEXT:  %0 = "llvm.extractvalue"(%arg0) {position: [0]} : (!llvm<"{ float*, i64, i64 }">) -> !llvm<"float*">
// CHECK-NEXT:  %1 = "llvm.bitcast"(%0) : (!llvm<"float*">) -> !llvm<"i8*">
// CHECK-NEXT:  "llvm.call0"(%1) {callee: @free : (!llvm<"i8*">) -> ()} : (!llvm<"i8*">) -> ()
  dealloc %arg0 : memref<?x42x?xf32>
// CHECK-NEXT:  "llvm.return"() : () -> ()
  return
}

// CHECK-LABEL: func @load
func @load(%static : memref<10x42xf32>, %dynamic : memref<?x?xf32>,
           %mixed : memref<42x?xf32>, %i : index, %j : index) {
// CHECK-NEXT: %0 = "llvm.constant"() {value: 10 : index} : () -> !llvm<"i64">
// CHECK-NEXT: %1 = "llvm.constant"() {value: 42 : index} : () -> !llvm<"i64">
// CHECK-NEXT: %2 = "llvm.mul"(%arg3, %1) : (!llvm<"i64">, !llvm<"i64">) -> !llvm<"i64">
// CHECK-NEXT: %3 = "llvm.add"(%2, %arg4) : (!llvm<"i64">, !llvm<"i64">) -> !llvm<"i64">
// CHECK-NEXT: %4 = "llvm.extractvalue"(%arg0) {position: [0]} : (!llvm<"{ float* }">) -> !llvm<"float*">
// CHECK-NEXT: %5 = "llvm.getelementptr"(%4, %3) : (!llvm<"float*">, !llvm<"i64">) -> !llvm<"float*">
// CHECK-NEXT: %6 = "llvm.load"(%5) : (!llvm<"float*">) -> !llvm<"float">
  %0 = load %static[%i, %j] : memref<10x42xf32>

// CHECK-NEXT: %7 = "llvm.extractvalue"(%arg1) {position: [1]} : (!llvm<"{ float*, i64, i64 }">) -> !llvm<"i64">
// CHECK-NEXT: %8 = "llvm.extractvalue"(%arg1) {position: [2]} : (!llvm<"{ float*, i64, i64 }">) -> !llvm<"i64">
// CHECK-NEXT: %9 = "llvm.mul"(%arg3, %8) : (!llvm<"i64">, !llvm<"i64">) -> !llvm<"i64">
// CHECK-NEXT: %10 = "llvm.add"(%9, %arg4) : (!llvm<"i64">, !llvm<"i64">) -> !llvm<"i64">
// CHECK-NEXT: %11 = "llvm.extractvalue"(%arg1) {position: [0]} : (!llvm<"{ float*, i64, i64 }">) -> !llvm<"float*">
// CHECK-NEXT: %12 = "llvm.getelementptr"(%11, %10) : (!llvm<"float*">, !llvm<"i64">) -> !llvm<"float*">
// CHECK-NEXT: %13 = "llvm.load"(%12) : (!llvm<"float*">) -> !llvm<"float">
  %1 = load %dynamic[%i, %j] : memref<?x?xf32>

// CHECK-NEXT: %14 = "llvm.constant"() {value: 42 : index} : () -> !llvm<"i64">
// CHECK-NEXT: %15 = "llvm.extractvalue"(%arg2) {position: [1]} : (!llvm<"{ float*, i64 }">) -> !llvm<"i64">
// CHECK-NEXT: %16 = "llvm.mul"(%arg3, %15) : (!llvm<"i64">, !llvm<"i64">) -> !llvm<"i64">
// CHECK-NEXT: %17 = "llvm.add"(%16, %arg4) : (!llvm<"i64">, !llvm<"i64">) -> !llvm<"i64">
// CHECK-NEXT: %18 = "llvm.extractvalue"(%arg2) {position: [0]} : (!llvm<"{ float*, i64 }">) -> !llvm<"float*">
// CHECK-NEXT: %19 = "llvm.getelementptr"(%18, %17) : (!llvm<"float*">, !llvm<"i64">) -> !llvm<"float*">
// CHECK-NEXT: %20 = "llvm.load"(%19) : (!llvm<"float*">) -> !llvm<"float">
  %2 = load %mixed[%i, %j] : memref<42x?xf32>
  return
}

// CHECK-LABEL: func @store
func @store(%static : memref<10x42xf32>, %dynamic : memref<?x?xf32>,
            %mixed : memref<42x?xf32>, %i : index, %j : index, %val : f32) {
// CHECK-NEXT:  %0 = "llvm.constant"() {value: 10 : index} : () -> !llvm<"i64">
// CHECK-NEXT:  %1 = "llvm.constant"() {value: 42 : index} : () -> !llvm<"i64">
// CHECK-NEXT:  %2 = "llvm.mul"(%arg3, %1) : (!llvm<"i64">, !llvm<"i64">) -> !llvm<"i64">
// CHECK-NEXT:  %3 = "llvm.add"(%2, %arg4) : (!llvm<"i64">, !llvm<"i64">) -> !llvm<"i64">
// CHECK-NEXT:  %4 = "llvm.extractvalue"(%arg0) {position: [0]} : (!llvm<"{ float* }">) -> !llvm<"float*">
// CHECK-NEXT:  %5 = "llvm.getelementptr"(%4, %3) : (!llvm<"float*">, !llvm<"i64">) -> !llvm<"float*">
// CHECK-NEXT:  "llvm.store"(%arg5, %5) : (!llvm<"float">, !llvm<"float*">) -> ()
  store %val, %static[%i, %j] : memref<10x42xf32>

// CHECK-NEXT:  %6 = "llvm.extractvalue"(%arg1) {position: [1]} : (!llvm<"{ float*, i64, i64 }">) -> !llvm<"i64">
// CHECK-NEXT:  %7 = "llvm.extractvalue"(%arg1) {position: [2]} : (!llvm<"{ float*, i64, i64 }">) -> !llvm<"i64">
// CHECK-NEXT:  %8 = "llvm.mul"(%arg3, %7) : (!llvm<"i64">, !llvm<"i64">) -> !llvm<"i64">
// CHECK-NEXT:  %9 = "llvm.add"(%8, %arg4) : (!llvm<"i64">, !llvm<"i64">) -> !llvm<"i64">
// CHECK-NEXT:  %10 = "llvm.extractvalue"(%arg1) {position: [0]} : (!llvm<"{ float*, i64, i64 }">) -> !llvm<"float*">
// CHECK-NEXT:  %11 = "llvm.getelementptr"(%10, %9) : (!llvm<"float*">, !llvm<"i64">) -> !llvm<"float*">
// CHECK-NEXT:  "llvm.store"(%arg5, %11) : (!llvm<"float">, !llvm<"float*">) -> ()
  store %val, %dynamic[%i, %j] : memref<?x?xf32>

// CHECK-NEXT:  %12 = "llvm.constant"() {value: 42 : index} : () -> !llvm<"i64">
// CHECK-NEXT:  %13 = "llvm.extractvalue"(%arg2) {position: [1]} : (!llvm<"{ float*, i64 }">) -> !llvm<"i64">
// CHECK-NEXT:  %14 = "llvm.mul"(%arg3, %13) : (!llvm<"i64">, !llvm<"i64">) -> !llvm<"i64">
// CHECK-NEXT:  %15 = "llvm.add"(%14, %arg4) : (!llvm<"i64">, !llvm<"i64">) -> !llvm<"i64">
// CHECK-NEXT:  %16 = "llvm.extractvalue"(%arg2) {position: [0]} : (!llvm<"{ float*, i64 }">) -> !llvm<"float*">
// CHECK-NEXT:  %17 = "llvm.getelementptr"(%16, %15) : (!llvm<"float*">, !llvm<"i64">) -> !llvm<"float*">
// CHECK-NEXT:  "llvm.store"(%arg5, %17) : (!llvm<"float">, !llvm<"float*">) -> ()
  store %val, %mixed[%i, %j] : memref<42x?xf32>
  return
}
