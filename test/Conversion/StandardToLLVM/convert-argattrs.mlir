// RUN: mlir-opt -lower-to-llvm %s | FileCheck %s


// CHECK-LABEL: func @check_attributes(%arg0: !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }*"> {dialect.a = true, dialect.b = 4 : i64}) {
//  CHECK-NEXT:   llvm.load %arg0 : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }*">
func @check_attributes(%static: memref<10x20xf32> {dialect.a = true, dialect.b = 4 : i64 }) {
  %c0 = constant 0 : index
  %0 = load %static[%c0, %c0]: memref<10x20xf32>
  return
}

// CHECK-LABEL: func @external_func(!llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }*">)
//       CHECK: func @call_external(%[[arg:.*]]: !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }*">) {
//       CHECK:   %[[ld:.*]] = llvm.load %[[arg]] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }*">
//       CHECK:   %[[c1:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
//       CHECK:   %[[alloca:.*]] = llvm.alloca %[[c1]] x !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }"> : (!llvm.i64) -> !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }*">
//       CHECK:   llvm.store %[[ld]], %[[alloca]] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }*">
//       CHECK:   call @external_func(%[[alloca]]) : (!llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }*">) -> ()
func @external_func(memref<10x20xf32>)

func @call_external(%static: memref<10x20xf32>) {
  call @external_func(%static) : (memref<10x20xf32>) -> ()
  return
}

