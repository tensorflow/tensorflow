// RUN: kernel-gen-opt %s -allow-unregistered-dialect -propagate-tf-abi-knowledge-to-kernels | FileCheck %s --check-prefixes=CHECK,ABI
// RUN: kernel-gen-opt %s -allow-unregistered-dialect -propagate-shape-knowledge-to-kernels | FileCheck %s --check-prefixes=CHECK,SHAPE

// The input is taken from what is actually used in kernel generator lowering
// for unary operations. This could be minimized but then we would not be
// testing how this actually gets used.

// CHECK-LABEL: module attributes {gpu.container_module}
module attributes {gpu.container_module} {
  // CHECK-LABEL: func @abs
  func @abs(%ctx: !tf_framework.op_kernel_context, %arg0: memref<*xf32>)
      -> memref<*xf32> attributes {tf_entry} {
    %c256 = constant 256 : index
    %c1024 = constant 1024 : index
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %0 = rank %arg0 : memref<*xf32>
    %1 = alloca(%0) : memref<?xindex>
    br ^bb1(%c0 : index)
  ^bb1(%2: index):  // 2 preds: ^bb0, ^bb2
    %3 = cmpi "slt", %2, %0 : index
    cond_br %3, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %4 = dim %arg0, %2 : memref<*xf32>
    store %4, %1[%2] : memref<?xindex>
    %5 = addi %2, %c1 : index
    br ^bb1(%5 : index)
  ^bb3:  // pred: ^bb1
    %6 = dim %1, %c0 : memref<?xindex>
    br ^bb4(%c0, %c1 : index, index)
  ^bb4(%7: index, %8: index):  // 2 preds: ^bb3, ^bb5
    %9 = cmpi "slt", %7, %6 : index
    cond_br %9, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %10 = dim %arg0, %7 : memref<*xf32>
    %11 = muli %10, %8 : index
    %12 = addi %7, %c1 : index
    br ^bb4(%12, %11 : index, index)
  ^bb6:  // pred: ^bb4
    %13 = alloca() : memref<1xindex>
    store %8, %13[%c0] : memref<1xindex>
    %14 = memref_reshape %arg0(%13) : (memref<*xf32>, memref<1xindex>) -> memref<?xf32>
    %15 = dim %14, %c0 : memref<?xf32>
    %16 = tf_framework.alloc(%ctx, %15) : memref<?xf32>
    %17 = cmpi "sle", %15, %c0 : index
    %18 = subi %c0, %15 : index
    %19 = subi %15, %c1 : index
    %20 = select %17, %18, %19 : index
    %21 = divi_signed %20, %c1024 : index
    %22 = subi %c0, %21 : index
    %23 = addi %21, %c1 : index
    %24 = select %17, %22, %23 : index
    gpu.launch_func @abs_kernel::@abs_kernel
        blocks in (%24, %c1, %c1) threads in (%c256, %c1, %c1)
        args(%14 : memref<?xf32>, %16 : memref<?xf32>)
    %25 = memref_reshape %16(%1) : (memref<?xf32>, memref<?xindex>) -> memref<*xf32>
    return %25 : memref<*xf32>
  }

  // CHECK-LABEL: gpu.module @abs_kernel
  gpu.module @abs_kernel {
    // CHECK-LABEL: @__nv_fabsf
    llvm.func @__nv_fabsf(!llvm.float) -> !llvm.float
    // CHECK-LABEL: @abs_kernel
    // ABI-SAME: %[[ARG0:.*]]: !llvm.ptr<float>, %[[ARG1:.*]]: !llvm.ptr<float> {llvm.align = 16 : index},
    // ABI-SAME: %[[ARG2:.*]]: !llvm.i64, %[[ARG3:.*]]: !llvm.i64, %[[ARG4:.*]]: !llvm.i64, %[[ARG5:.*]]: !llvm.ptr<float>, %[[ARG6:.*]]: !llvm.ptr<float> {llvm.align = 16 : index, llvm.noalias = true},
    // ABI-SAME: %[[ARG7:.*]]: !llvm.i64, %[[ARG8:.*]]: !llvm.i64, %[[ARG9:.*]]: !llvm.i64
    // SHAPE-SAME: %[[ARG0:.*]]: !llvm.ptr<float>, %[[ARG1:.*]]: !llvm.ptr<float>, %[[ARG2:.*]]: !llvm.i64, %[[ARG3:.*]]: !llvm.i64, %[[ARG4:.*]]: !llvm.i64, %[[ARG5:.*]]: !llvm.ptr<float>, %[[ARG6:.*]]: !llvm.ptr<float>, %[[ARG7:.*]]: !llvm.i64, %[[ARG8:.*]]: !llvm.i64, %[[ARG9:.*]]: !llvm.i64
    llvm.func @abs_kernel(%arg0: !llvm.ptr<float>, %arg1: !llvm.ptr<float>, %arg2: !llvm.i64, %arg3: !llvm.i64, %arg4: !llvm.i64, %arg5: !llvm.ptr<float>, %arg6: !llvm.ptr<float>, %arg7: !llvm.i64, %arg8: !llvm.i64, %arg9: !llvm.i64) attributes {gpu.kernel} {
      // ABI: %[[ZERO:.*]] = llvm.mlir.constant(0 : index)
      // ABI: %[[ONE:.*]] = llvm.mlir.constant(1 : index)
      // CHECK: llvm.mlir.undef
      %0 = llvm.mlir.undef : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: llvm.insertvalue %[[ARG1]]
      // SHAPE-NEXT: llvm.insertvalue %[[ARG0]]
      %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
      // CHECK-NEXT: llvm.insertvalue %[[ARG1]]
      %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: llvm.insertvalue %[[ZERO]]
      // SHAPE-NEXT: llvm.insertvalue %[[ARG2]]
      %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
      // CHECK-NEXT: llvm.insertvalue %[[ARG3]]
      %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: llvm.insertvalue %[[ONE]]
      // SHAPE-NEXT: llvm.insertvalue %[[ARG4]]
      %5 = llvm.insertvalue %arg4, %4[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
      // CHECK-NEXT: llvm.mlir.undef
      %6 = llvm.mlir.undef : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: llvm.insertvalue %[[ARG6]]
      // SHAPE-NEXT: llvm.insertvalue %[[ARG5]]
      %7 = llvm.insertvalue %arg5, %6[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
      // CHECK-NEXT: llvm.insertvalue %[[ARG6]]
      %8 = llvm.insertvalue %arg6, %7[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: llvm.insertvalue %[[ZERO]]
      // SHAPE-NEXT: llvm.insertvalue %[[ARG7]]
      %9 = llvm.insertvalue %arg7, %8[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: llvm.insertvalue %[[ARG8]]
      // SHAPE-NEXT: llvm.insertvalue %[[ARG3]]
      %10 = llvm.insertvalue %arg8, %9[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: llvm.insertvalue %[[ONE]]
      // SHAPE-NEXT: llvm.insertvalue %[[ARG4]]
      %11 = llvm.insertvalue %arg9, %10[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
      %12 = nvvm.read.ptx.sreg.ctaid.x : !llvm.i32
      %13 = llvm.sext %12 : !llvm.i32 to !llvm.i64
      %14 = nvvm.read.ptx.sreg.ctaid.y : !llvm.i32
      %15 = llvm.sext %14 : !llvm.i32 to !llvm.i64
      %16 = nvvm.read.ptx.sreg.ctaid.z : !llvm.i32
      %17 = llvm.sext %16 : !llvm.i32 to !llvm.i64
      %18 = nvvm.read.ptx.sreg.tid.x : !llvm.i32
      %19 = llvm.sext %18 : !llvm.i32 to !llvm.i64
      %20 = nvvm.read.ptx.sreg.tid.y : !llvm.i32
      %21 = llvm.sext %20 : !llvm.i32 to !llvm.i64
      %22 = nvvm.read.ptx.sreg.tid.z : !llvm.i32
      %23 = llvm.sext %22 : !llvm.i32 to !llvm.i64
      %24 = nvvm.read.ptx.sreg.nctaid.x : !llvm.i32
      %25 = llvm.sext %24 : !llvm.i32 to !llvm.i64
      %26 = nvvm.read.ptx.sreg.nctaid.y : !llvm.i32
      %27 = llvm.sext %26 : !llvm.i32 to !llvm.i64
      %28 = nvvm.read.ptx.sreg.nctaid.z : !llvm.i32
      %29 = llvm.sext %28 : !llvm.i32 to !llvm.i64
      %30 = nvvm.read.ptx.sreg.ntid.x : !llvm.i32
      %31 = llvm.sext %30 : !llvm.i32 to !llvm.i64
      %32 = nvvm.read.ptx.sreg.ntid.y : !llvm.i32
      %33 = llvm.sext %32 : !llvm.i32 to !llvm.i64
      %34 = nvvm.read.ptx.sreg.ntid.z : !llvm.i32
      %35 = llvm.sext %34 : !llvm.i32 to !llvm.i64
      llvm.br ^bb1
    ^bb1:  // pred: ^bb0
      %36 = llvm.mlir.constant(0 : index) : !llvm.i64
      %37 = llvm.extractvalue %5[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
      %38 = llvm.mlir.constant(256 : index) : !llvm.i64
      %39 = llvm.mul %13, %38 : !llvm.i64
      %40 = llvm.mlir.constant(256 : index) : !llvm.i64
      %41 = llvm.mlir.constant(-256 : index) : !llvm.i64
      %42 = llvm.mul %13, %41 : !llvm.i64
      %43 = llvm.add %42, %37 : !llvm.i64
      %44 = llvm.icmp "slt" %40, %43 : !llvm.i64
      %45 = llvm.select %44, %40, %43 : !llvm.i1, !llvm.i64
      %46 = llvm.mlir.undef : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
      %47 = llvm.extractvalue %5[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
      %48 = llvm.bitcast %47 : !llvm.ptr<float> to !llvm.ptr<float>
      %49 = llvm.insertvalue %48, %46[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
      %50 = llvm.extractvalue %5[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
      %51 = llvm.bitcast %50 : !llvm.ptr<float> to !llvm.ptr<float>
      %52 = llvm.insertvalue %51, %49[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
      %53 = llvm.extractvalue %5[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
      %54 = llvm.extractvalue %5[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
      %55 = llvm.mul %39, %53 : !llvm.i64
      %56 = llvm.add %54, %55 : !llvm.i64
      %57 = llvm.insertvalue %56, %52[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
      %58 = llvm.insertvalue %45, %57[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
      %59 = llvm.mlir.constant(1 : i64) : !llvm.i64
      %60 = llvm.insertvalue %59, %58[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
      %61 = llvm.mlir.undef : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
      %62 = llvm.extractvalue %11[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
      %63 = llvm.bitcast %62 : !llvm.ptr<float> to !llvm.ptr<float>
      %64 = llvm.insertvalue %63, %61[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
      %65 = llvm.extractvalue %11[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
      %66 = llvm.bitcast %65 : !llvm.ptr<float> to !llvm.ptr<float>
      %67 = llvm.insertvalue %66, %64[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
      %68 = llvm.extractvalue %11[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
      %69 = llvm.extractvalue %11[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
      %70 = llvm.mul %39, %68 : !llvm.i64
      %71 = llvm.add %69, %70 : !llvm.i64
      %72 = llvm.insertvalue %71, %67[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
      %73 = llvm.insertvalue %45, %72[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
      %74 = llvm.mlir.constant(1 : i64) : !llvm.i64
      %75 = llvm.insertvalue %74, %73[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
      %76 = llvm.icmp "slt" %19, %45 : !llvm.i64
      llvm.cond_br %76, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %77 = llvm.extractvalue %60[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
      %78 = llvm.extractvalue %60[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
      %79 = llvm.mlir.constant(1 : index) : !llvm.i64
      %80 = llvm.mul %19, %79 : !llvm.i64
      %81 = llvm.add %78, %80 : !llvm.i64
      %82 = llvm.getelementptr %77[%81] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
      %83 = llvm.load %82 : !llvm.ptr<float>
      %84 = llvm.call @__nv_fabsf(%83) : (!llvm.float) -> !llvm.float
      %85 = llvm.extractvalue %75[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
      %86 = llvm.extractvalue %75[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
      %87 = llvm.mlir.constant(1 : index) : !llvm.i64
      %88 = llvm.mul %19, %87 : !llvm.i64
      %89 = llvm.add %86, %88 : !llvm.i64
      %90 = llvm.getelementptr %85[%89] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
      llvm.store %84, %90 : !llvm.ptr<float>
      llvm.br ^bb3
    ^bb3:  // 2 preds: ^bb1, ^bb2
      llvm.return
    }
  }
}

