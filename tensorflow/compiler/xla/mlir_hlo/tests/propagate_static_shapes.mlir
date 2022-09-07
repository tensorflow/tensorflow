// RUN: mlir-hlo-opt %s \
// RUN:   -propagate-static-shapes='convert_pointer_args=!llvm.ptr<i8>' \
// RUN: | FileCheck %s

module attributes {gpu.container_module} {

  gpu.module @gpu_module {
    // CHECK: llvm.func @kernel(%arg0: f32, %arg1: !llvm.ptr<i8>, %arg2: f32)
    llvm.func @kernel(
      %arg0: f32,
      %base: !llvm.ptr<f32>, %align: !llvm.ptr<f32>, %offset: i64,
      %size.x: i64, %size.y: i64, %stride.x: i64, %stride.y: i64,
      %argN: f32
    ) attributes {gpu.kernel} {
      // CHECK-DAG:  %[[base:.*]] = llvm.bitcast %arg1 : !llvm.ptr<i8> to !llvm.ptr<f32>
      // CHECK:      %[[ptr:.*]] = llvm.getelementptr %[[base]][4]
      // CHECK:      llvm.call @dummy(%[[ptr]]) : (!llvm.ptr<f32>) -> ()
      %ptr = llvm.getelementptr %align[%stride.x] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      llvm.call @dummy(%ptr) : (!llvm.ptr<f32>) -> ()
      llvm.return
    }
    // CHECK: llvm.func @dummy(%arg0: !llvm.ptr<f32>)
    llvm.func @dummy(%arg0: !llvm.ptr<f32>) attributes {gpu.kernel} {
      llvm.return
    }
  }

  func.func @func(%arg0: f32, %arg1: memref<2x4xf32>) {
    %c1 = arith.constant 1 : index
    gpu.launch_func  @gpu_module::@kernel
      blocks in (%c1, %c1, %c1)
      threads in (%c1, %c1, %c1)
      args(%arg0 : f32, %arg1 : memref<2x4xf32>, %arg0 : f32)
    func.return
  }

}
