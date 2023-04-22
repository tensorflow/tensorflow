// RUN: mlir-hlo-opt -disc-ral-to-llvm -split-input-file %s -o - | FileCheck %s

// CHECK: llvm.mlir.global internal constant @ral_recv_input___cpu___pvoid_i64___m2df32("ral_recv_input___cpu___pvoid_i64___m2df32\00")
// CHECK: llvm.func @disc_ral_call(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<ptr<i8>>)

// CHECK-LABEL: test_recv_input
// CHECK-SAME: (%[[CTX:.*]]: !llvm.ptr<i8>)
func @test_recv_input(%arg0: !disc_ral.context) {
  // CHECK-DAG: %[[i64_0:.*]] = llvm.mlir.constant(0 : index) : i64
  // CHECK-DAG: %[[i32_0:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-DAG: %[[i32_1:.*]] = llvm.mlir.constant(1 : i32) : i32

  // CHECK: %[[T0:.*]] = llvm.alloca %[[i32_1]] x !llvm.struct<"", (ptr<i8>, i64, struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>)>
  // CHECK: %[[T1:.*]] = llvm.mlir.constant(3 : i32) : i32
  // CHECK: %[[T2:.*]] = llvm.alloca %[[T1]] x !llvm.ptr<i8> : (i32) -> !llvm.ptr<ptr<i8>>

  // CHECK: %[[T3:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[T4:.*]] = llvm.getelementptr %[[T0]][%[[i32_0]], %[[T3]]]
  // CHECK: llvm.store %[[CTX]], %[[T4]] : !llvm.ptr<ptr<i8>>
  // CHECK: %[[T5:.*]] = llvm.getelementptr %[[T2]][%[[T3]]] : (!llvm.ptr<ptr<i8>>, i32) -> !llvm.ptr<ptr<i8>>
  // CHECK: %[[T6:.*]] = llvm.bitcast %[[T4]] : !llvm.ptr<ptr<i8>> to !llvm.ptr<i8>
  // CHECK: llvm.store %[[T6]], %[[T5]] : !llvm.ptr<ptr<i8>>

  // CHECK: %[[T7:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[T8:.*]] = llvm.getelementptr %[[T0]][%[[i32_0]], %[[T7]]]
  // CHECK: llvm.store %[[i64_0]], %[[T8]] : !llvm.ptr<i64>
  // CHECK: %[[T9:.*]] = llvm.getelementptr %[[T2]][%[[T7]]] : (!llvm.ptr<ptr<i8>>, i32) -> !llvm.ptr<ptr<i8>>
  // CHECK: %[[T10:.*]] = llvm.bitcast %[[T8]] : !llvm.ptr<i64> to !llvm.ptr<i8>
  // CHECK: llvm.store %[[T10]], %[[T9]] : !llvm.ptr<ptr<i8>>

  // CHECK: %[[T11:.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK: %[[T12:.*]] = llvm.getelementptr %[[T0]][%[[i32_0]], %[[T11]]]
  // CHECK: %[[T13:.*]] = llvm.getelementptr %[[T2]][%[[T11]]] : (!llvm.ptr<ptr<i8>>, i32) -> !llvm.ptr<ptr<i8>>
  // CHECK: %[[T14:.*]] = llvm.bitcast %[[T12]] : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>> to !llvm.ptr<i8>
  // CHECK: llvm.store %[[T14]], %[[T13]] : !llvm.ptr<ptr<i8>>

  // CHECK: %[[T15:.*]] = llvm.mlir.addressof @ral_recv_input___cpu___pvoid_i64___m2df32 : !llvm.ptr<array<42 x i8>>
  // CHECK: %[[T16:.*]] = llvm.mlir.constant(0 : index) : i64
  // CHECK: %[[T17:.*]] = llvm.getelementptr %[[T15]][%[[T16]], %[[T16]]] : (!llvm.ptr<array<42 x i8>>, i64, i64) -> !llvm.ptr<i8>
  // CHECK: llvm.call @disc_ral_call(%[[CTX]], %[[T17]], %[[T2]]) : (!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<ptr<i8>>) -> ()
  // CHECK: %[[OUT:.*]] = llvm.load %[[T12]] : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>>
  %c0 = constant 0 : index
  %0 = "disc_ral.dispatch"(%arg0, %c0) {backend_config = "cpu", call_target_name = "ral_recv_input", has_side_effect = false} : (!disc_ral.context, index) -> memref<?x?xf32>
  return
}

// -----

// CHECK: llvm.mlir.global internal constant @ral_send_output___cpu___pvoid_i64_m2df32___void("ral_send_output___cpu___pvoid_i64_m2df32___void\00")
// CHECK: llvm.func @disc_ral_call(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<ptr<i8>>)

// CHECK-LABEL: test_send_output
// CHECK-SAME: (%[[CTX:.*]]: !llvm.ptr<i8>

func @test_send_output(%arg0: !disc_ral.context, %arg1: memref<?x?xf32>) {
  // CHECK-DAG: %[[i64_0:.*]] = llvm.mlir.constant(0 : index) : i64
  // CHECK-DAG: %[[i32_0:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-DAG: %[[i32_1:.*]] = llvm.mlir.constant(1 : i32) : i32

  // CHECK: %[[T0:.*]] = llvm.alloca %10 x !llvm.struct<"", (ptr<i8>, i64, ptr<f32>, ptr<f32>, i64, i64, i64, i64, i64)>
  // CHECK: %[[T1:.*]] = llvm.mlir.constant(9 : i32) : i32
  // CHECK: %[[OARGS:.*]] = llvm.alloca %[[T1]] x !llvm.ptr<i8> : (i32) -> !llvm.ptr<ptr<i8>>
  // CHECK: %[[OP0:.*]] = llvm.getelementptr %[[OARGS]]{{.*}} : (!llvm.ptr<ptr<i8>>, i32) -> !llvm.ptr<ptr<i8>>
  // CHECK: llvm.store %{{.*}}, %[[OP0]] : !llvm.ptr<ptr<i8>>
  // CHECK: %[[OP1:.*]] = llvm.getelementptr %[[OARGS]]{{.*}} : (!llvm.ptr<ptr<i8>>, i32) -> !llvm.ptr<ptr<i8>>
  // CHECK: llvm.store %{{.*}}, %[[OP1]] : !llvm.ptr<ptr<i8>>
  // CHECK: %[[OP2:.*]] = llvm.getelementptr %[[OARGS]]{{.*}} : (!llvm.ptr<ptr<i8>>, i32) -> !llvm.ptr<ptr<i8>>
  // CHECK: llvm.store %{{.*}}, %[[OP2]] : !llvm.ptr<ptr<i8>>
  // CHECK: %[[OP3:.*]] = llvm.getelementptr %[[OARGS]]{{.*}} : (!llvm.ptr<ptr<i8>>, i32) -> !llvm.ptr<ptr<i8>>
  // CHECK: llvm.store %{{.*}}, %[[OP3]] : !llvm.ptr<ptr<i8>>
  // CHECK: %[[OP4:.*]] = llvm.getelementptr %[[OARGS]]{{.*}} : (!llvm.ptr<ptr<i8>>, i32) -> !llvm.ptr<ptr<i8>>
  // CHECK: llvm.store %{{.*}}, %[[OP4]] : !llvm.ptr<ptr<i8>>
  // CHECK: %[[OP5:.*]] = llvm.getelementptr %[[OARGS]]{{.*}} : (!llvm.ptr<ptr<i8>>, i32) -> !llvm.ptr<ptr<i8>>
  // CHECK: llvm.store %{{.*}}, %[[OP5]] : !llvm.ptr<ptr<i8>>
  // CHECK: %[[OP6:.*]] = llvm.getelementptr %[[OARGS]]{{.*}} : (!llvm.ptr<ptr<i8>>, i32) -> !llvm.ptr<ptr<i8>>
  // CHECK: llvm.store %{{.*}}, %[[OP6]] : !llvm.ptr<ptr<i8>>
  // CHECK: %[[OP7:.*]] = llvm.getelementptr %[[OARGS]]{{.*}} : (!llvm.ptr<ptr<i8>>, i32) -> !llvm.ptr<ptr<i8>>
  // CHECK: llvm.store %{{.*}}, %[[OP7]] : !llvm.ptr<ptr<i8>>
  // CHECK: %[[OP8:.*]] = llvm.getelementptr %[[OARGS]]{{.*}} : (!llvm.ptr<ptr<i8>>, i32) -> !llvm.ptr<ptr<i8>>
  // CHECK: llvm.store %{{.*}}, %[[OP8]] : !llvm.ptr<ptr<i8>>

  // CHECK: %[[T2:.*]] = llvm.mlir.addressof @ral_send_output___cpu___pvoid_i64_m2df32___void : !llvm.ptr<array<48 x i8>>
  // CHECK: %[[T3:.*]] = llvm.getelementptr %[[T2]]{{.*}} : (!llvm.ptr<array<48 x i8>>, i64, i64) -> !llvm.ptr<i8>
  // CHECK: llvm.call @disc_ral_call(%[[CTX]], %[[T3]], %[[OARGS]]) : (!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<ptr<i8>>) -> ()

  %c0 = constant 0 : index
  "disc_ral.dispatch"(%arg0, %c0, %arg1) {backend_config = "cpu", call_target_name = "ral_send_output", has_side_effect = false} : (!disc_ral.context, index, memref<?x?xf32>) -> ()
  return
}

// -----

// CHECK-DAG: llvm.mlir.global internal constant @ral_kernel_launch___cpu___pvoid_pvoid_pvoid_i64_i64_i64_i64_i64_i64_i32_i32_ppvoid___void("ral_kernel_launch___cpu___pvoid_pvoid_pvoid_i64_i64_i64_i64_i64_i64_i32_i32_ppvoid___void\00")
// CHECK-DAG: llvm.func @disc_ral_call(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<ptr<i8>>)
// CHECK-DAG: llvm.mlir.global internal constant @kernel_module_the_kernel_kernel_name("the_kernel\00")
// CHECK-DAG: llvm.mlir.global internal constant @kernel_module_blob("BLOB!")

module @main attributes {gpu.container_module}  {
  // CHECK-NOT: gpu.module
  gpu.module @kernel_module attributes {gpu.binary_blob = "BLOB!"} {
    llvm.func @the_kernel() attributes {gpu.kernel}
  }

  // CHECK: llvm.func @test_gpu_launch(%[[CTX:.*]]: !llvm.ptr<i8>
  func @test_gpu_launch(%arg0: !disc_ral.context, %arg1: memref<?x?xf32>) {
    %c1 = constant 1 : index

    // CHECK: %[[T18:.*]] = llvm.mlir.addressof @kernel_module_blob : !llvm.ptr<array<5 x i8>>
    // CHECK: %[[T19:.*]] = llvm.getelementptr %[[T18]]{{.*}}
    // CHECK: %[[T20:.*]] = llvm.mlir.addressof @kernel_module_the_kernel_kernel_name : !llvm.ptr<array<11 x i8>>
    // CHECK: %[[T21:.*]] = llvm.getelementptr %[[T20]]{{.*}}

    // prepare arguments for the kernel to launch
    // CHECK: %[[KARGS:.*]] = llvm.alloca %{{.*}} x !llvm.ptr<i8> : (i32) -> !llvm.ptr<ptr<i8>>
    // CHECK: %[[KF0:.*]] = llvm.getelementptr %[[KARGS]]{{.*}} : (!llvm.ptr<ptr<i8>>, i32) -> !llvm.ptr<ptr<i8>>
    // CHECK: llvm.store %{{.*}}, %[[KF0]] : !llvm.ptr<ptr<i8>>
    // CHECK: %[[KF1:.*]] = llvm.getelementptr %[[KARGS]]{{.*}} : (!llvm.ptr<ptr<i8>>, i32) -> !llvm.ptr<ptr<i8>>
    // CHECK: llvm.store %{{.*}}, %[[KF1]] : !llvm.ptr<ptr<i8>>
    // CHECK: %[[KF2:.*]] = llvm.getelementptr %[[KARGS]]{{.*}} : (!llvm.ptr<ptr<i8>>, i32) -> !llvm.ptr<ptr<i8>>
    // CHECK: llvm.store %{{.*}}, %[[KF2]] : !llvm.ptr<ptr<i8>>
    // CHECK: %[[KF3:.*]] = llvm.getelementptr %[[KARGS]]{{.*}} : (!llvm.ptr<ptr<i8>>, i32) -> !llvm.ptr<ptr<i8>>
    // CHECK: llvm.store %{{.*}}, %[[KF3]] : !llvm.ptr<ptr<i8>>
    // CHECK: %[[KF4:.*]] = llvm.getelementptr %[[KARGS]]{{.*}} : (!llvm.ptr<ptr<i8>>, i32) -> !llvm.ptr<ptr<i8>>
    // CHECK: llvm.store %{{.*}}, %[[KF4]] : !llvm.ptr<ptr<i8>>
    // CHECK: %[[KF5:.*]] = llvm.getelementptr %[[KARGS]]{{.*}} : (!llvm.ptr<ptr<i8>>, i32) -> !llvm.ptr<ptr<i8>>
    // CHECK: llvm.store %{{.*}}, %[[KF5]] : !llvm.ptr<ptr<i8>>
    // CHECK: %[[KF6:.*]] = llvm.getelementptr %[[KARGS]]{{.*}} : (!llvm.ptr<ptr<i8>>, i32) -> !llvm.ptr<ptr<i8>>
    // CHECK: llvm.store %{{.*}}, %[[KF6]] : !llvm.ptr<ptr<i8>>

    // prepare arguments for the packed ral launch functions
    // CHECK: %[[KPT0:.*]] = llvm.mlir.constant(12 : i32) : i32
    // CHECK: %[[KPACK:.*]] = llvm.alloca %[[KPT0]] x !llvm.ptr<i8> : (i32) -> !llvm.ptr<ptr<i8>>
    // CHECK: %[[KP0:.*]] = llvm.getelementptr %[[KPACK]]{{.*}} : (!llvm.ptr<ptr<i8>>, i32) -> !llvm.ptr<ptr<i8>>
    // CHECK: llvm.store %{{.*}}, %[[KP0]] : !llvm.ptr<ptr<i8>>
    // CHECK: %[[KP1:.*]] = llvm.getelementptr %[[KPACK]]{{.*}} : (!llvm.ptr<ptr<i8>>, i32) -> !llvm.ptr<ptr<i8>>
    // CHECK: llvm.store %{{.*}}, %[[KP1]] : !llvm.ptr<ptr<i8>>
    // CHECK: %[[KP2:.*]] = llvm.getelementptr %[[KPACK]]{{.*}} : (!llvm.ptr<ptr<i8>>, i32) -> !llvm.ptr<ptr<i8>>
    // CHECK: llvm.store %{{.*}}, %[[KP2]] : !llvm.ptr<ptr<i8>>
    // CHECK: %[[KP3:.*]] = llvm.getelementptr %[[KPACK]]{{.*}} : (!llvm.ptr<ptr<i8>>, i32) -> !llvm.ptr<ptr<i8>>
    // CHECK: llvm.store %{{.*}}, %[[KP3]] : !llvm.ptr<ptr<i8>>
    // CHECK: %[[KP4:.*]] = llvm.getelementptr %[[KPACK]]{{.*}} : (!llvm.ptr<ptr<i8>>, i32) -> !llvm.ptr<ptr<i8>>
    // CHECK: llvm.store %{{.*}}, %[[KP4]] : !llvm.ptr<ptr<i8>>
    // CHECK: %[[KP5:.*]] = llvm.getelementptr %[[KPACK]]{{.*}} : (!llvm.ptr<ptr<i8>>, i32) -> !llvm.ptr<ptr<i8>>
    // CHECK: llvm.store %{{.*}}, %[[KP5]] : !llvm.ptr<ptr<i8>>
    // CHECK: %[[KP6:.*]] = llvm.getelementptr %[[KPACK]]{{.*}} : (!llvm.ptr<ptr<i8>>, i32) -> !llvm.ptr<ptr<i8>>
    // CHECK: llvm.store %{{.*}}, %[[KP6]] : !llvm.ptr<ptr<i8>>
    // CHECK: %[[KP7:.*]] = llvm.getelementptr %[[KPACK]]{{.*}} : (!llvm.ptr<ptr<i8>>, i32) -> !llvm.ptr<ptr<i8>>
    // CHECK: llvm.store %{{.*}}, %[[KP7]] : !llvm.ptr<ptr<i8>>
    // CHECK: %[[KP8:.*]] = llvm.getelementptr %[[KPACK]]{{.*}} : (!llvm.ptr<ptr<i8>>, i32) -> !llvm.ptr<ptr<i8>>
    // CHECK: llvm.store %{{.*}}, %[[KP8]] : !llvm.ptr<ptr<i8>>
    // CHECK: %[[KP9:.*]] = llvm.getelementptr %[[KPACK]]{{.*}} : (!llvm.ptr<ptr<i8>>, i32) -> !llvm.ptr<ptr<i8>>
    // CHECK: llvm.store %{{.*}}, %[[KP9]] : !llvm.ptr<ptr<i8>>
    // CHECK: %[[KP10:.*]] = llvm.getelementptr %[[KPACK]]{{.*}} : (!llvm.ptr<ptr<i8>>, i32) -> !llvm.ptr<ptr<i8>>
    // CHECK: llvm.store %{{.*}}, %[[KP10]] : !llvm.ptr<ptr<i8>>
    // CHECK: %[[KP11:.*]] = llvm.getelementptr %[[KPACK]]{{.*}} : (!llvm.ptr<ptr<i8>>, i32) -> !llvm.ptr<ptr<i8>>
    // CHECK: llvm.store %{{.*}}, %[[KP11]] : !llvm.ptr<ptr<i8>>

    // CHECK: %[[T22:.*]] = llvm.mlir.addressof @ral_kernel_launch___cpu___pvoid_pvoid_pvoid_i64_i64_i64_i64_i64_i64_i32_i32_ppvoid___void : !llvm.ptr<array<90 x i8>>
    // CHECK: %[[T23:.*]] = llvm.getelementptr %[[T22]]{{.*}} : (!llvm.ptr<array<90 x i8>>, i64, i64) -> !llvm.ptr<i8>
    // CHECK: llvm.call @disc_ral_call(%[[CTX]], %[[T23]], %[[KPACK]]) : (!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<ptr<i8>>) -> ()

    gpu.launch_func  @kernel_module::@the_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%arg1 : memref<?x?xf32>)
    return
  }
}
