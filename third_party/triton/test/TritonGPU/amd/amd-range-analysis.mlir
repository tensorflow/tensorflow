// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -test-tritonamdgpu-range-analysis -verify-diagnostics | FileCheck %s

// CHECK-LABEL:   tt.func @conversion1
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @conversion1(%arg0: !tt.ptr<f32>) -> tensor<1024xf32> {
    // expected-remark@+2 {{unsigned : [1024, 1024] signed : [1024, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %c1024_i32 = arith.constant 1024 : i32
    // expected-remark@+2 {{unsigned : [0, 65535] signed : [0, 65535]}}
    // expected-remark@+1 {{non-neg}}
    %0 = tt.get_program_id x : i32
    // expected-remark@+2 {{unsigned : [0, 67107840] signed : [0, 67107840]}}
    // expected-remark@+1 {{non-neg}}
    %1 = arith.muli %0, %c1024_i32 : i32
    // expected-remark@+2 {{unsigned : [0, 65536] signed : [0, 65536]}}
    // expected-remark@+1 {{non-neg}}
    %numps = tt.get_num_programs x : i32
    %2 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    %3 = tt.splat %2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %4 = tt.load %3 : tensor<1024x!tt.ptr<f32>>
    tt.return %4 : tensor<1024xf32>
  }
}

// -----

// CHECK-LABEL:   tt.func @assumepid
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @assumepid(%arg0: !tt.ptr<f32>) -> tensor<1024xf32> {
    // expected-remark@+2 {{unsigned : [0, 0] signed : [0, 0]}}
    // expected-remark@+1 {{non-neg}}
    %c0 = arith.constant 0 : i32
    // expected-remark@+2 {{unsigned : [1024, 1024] signed : [1024, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %c1024_i32 = arith.constant 1024 : i32
    // expected-remark@+2 {{unsigned : [0, 2147483647] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %pid = tt.get_program_id x : i32
    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %cmpsle = arith.cmpi sle, %pid, %c1024_i32 : i32
    llvm.intr.assume %cmpsle : i1
    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %cmpsge = arith.cmpi sge, %pid, %c0 : i32
    llvm.intr.assume %cmpsge : i1
    // expected-remark@+2 {{unsigned : [0, 1048576] signed : [0, 1048576]}}
    // expected-remark@+1 {{non-neg}}
    %1 = arith.muli %pid, %c1024_i32 : i32
    %2 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    %3 = tt.splat %2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %4 = tt.load %3 : tensor<1024x!tt.ptr<f32>>
    tt.return %4 : tensor<1024xf32>
  }
}

// -----

// CHECK-LABEL:   tt.func @conversion2
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @conversion2(%arg0: !tt.ptr<f32>) -> tensor<1024xf32> {
    // expected-remark@+2 {{unsigned : [1024, 1024] signed : [1024, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %c1024_i32 = arith.constant 1024 : i32
    // expected-remark@+2 {{unsigned : [0, 65535] signed : [0, 65535]}}
    // expected-remark@+1 {{non-neg}}
    %0 = tt.get_program_id x : i32
    // expected-remark@+2 {{unsigned : [0, 67107840] signed : [0, 67107840]}}
    // expected-remark@+1 {{non-neg}}
    %1 = arith.muli %0, %c1024_i32 : i32
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    %4 = tt.splat %3 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %5 = tt.addptr %4, %2 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %6 = tt.load %5 : tensor<1024x!tt.ptr<f32>>
    tt.return %6 : tensor<1024xf32>
  }
}

// -----

// CHECK-LABEL:   tt.func @conversion3
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @conversion3(%arg0: !tt.ptr<f32>) -> tensor<1024xf32> {
    // expected-remark@+2 {{unsigned : [1024, 1024] signed : [1024, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %c1024_i32 = arith.constant 1024 : i32
    // expected-remark@+2 {{unsigned : [0, 65535] signed : [0, 65535]}}
    // expected-remark@+1 {{non-neg}}
    %0 = tt.get_program_id x : i32
    // expected-remark@+2 {{unsigned : [0, 67107840] signed : [0, 67107840]}}
    // expected-remark@+1 {{non-neg}}
    %1 = arith.muli %0, %c1024_i32 : i32
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %4 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
    %5 = tt.addptr %3, %1 : !tt.ptr<f32>, i32
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %6 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
    // expected-remark@+2 {{unsigned : [0, 2048] signed : [0, 2048]}}
    // expected-remark@+1 {{non-neg}}
    %7 = arith.addi %6, %4 : tensor<1024xi64>
    %8 = tt.splat %5 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %9 = tt.addptr %8, %7 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
    %10 = tt.load %9 : tensor<1024x!tt.ptr<f32>>
    tt.return %10 : tensor<1024xf32>
  }
}

// -----

// CHECK-LABEL:   tt.func @conversion4
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @conversion4(%arg0: !tt.ptr<f32> {tt.pointer_range = 32 : i32}) -> tensor<1024xf32> {
    // expected-remark@+2 {{unsigned : [1024, 1024] signed : [1024, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %c1024_i32 = arith.constant 1024 : i32
    // expected-remark@+2 {{unsigned : [0, 65535] signed : [0, 65535]}}
    // expected-remark@+1 {{non-neg}}
    %0 = tt.get_program_id x : i32
    // expected-remark@+2 {{unsigned : [0, 67107840] signed : [0, 67107840]}}
    // expected-remark@+1 {{non-neg}}
    %1 = arith.muli %0, %c1024_i32 : i32
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    %4 = tt.addptr %3, %1 : !tt.ptr<f32>, i32
    // expected-remark@+2 {{unsigned : [0, 2048] signed : [0, 2048]}}
    // expected-remark@+1 {{non-neg}}
    %5 = arith.addi %2, %2 : tensor<1024xi32>
    %6 = tt.splat %4 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %7 = tt.addptr %6, %5 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %8 = tt.load %7 : tensor<1024x!tt.ptr<f32>>
    tt.return %8 : tensor<1024xf32>
  }
}

// -----

// CHECK-LABEL:   tt.func @forOp
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @forOp(%arg0: !tt.ptr<f32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    // expected-remark@+2 {{unsigned : [1024, 1024] signed : [1024, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %c1024_i32 = arith.constant 1024 : i32
    // expected-remark@+2 {{unsigned : [0, 0] signed : [0, 0]}}
    // expected-remark@+1 {{non-neg}}
    %c0 = arith.constant 0 : index
    // expected-remark@+2 {{unsigned : [128, 128] signed : [128, 128]}}
    // expected-remark@+1 {{non-neg}}
    %c128 = arith.constant 128 : index
    // expected-remark@+2 {{unsigned : [1, 1] signed : [1, 1]}}
    // expected-remark@+1 {{non-neg}}
    %c1 = arith.constant 1 : index
    // expected-remark@+2 {{unsigned : [0, 65535] signed : [0, 65535]}}
    // expected-remark@+1 {{non-neg}}
    %0 = tt.get_program_id x : i32
    // expected-remark@+2 {{unsigned : [0, 67107840] signed : [0, 67107840]}}
    // expected-remark@+1 {{non-neg}}
    %1 = arith.muli %0, %c1024_i32 : i32
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %4 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
    // expected-remark@+1 {{result 1: non-neg}}
    %5:3 = scf.for %arg2 = %c0 to %c128 step %c1 iter_args(%arg3 = %3, %arg4 = %4, %arg5 = %arg1) -> (!tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>) {
      %12 = tt.addptr %arg3, %1 : !tt.ptr<f32>, i32
      // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
      // expected-remark@+1 {{non-neg}}
      %13 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
      // expected-remark@+2 {{unsigned : [0, 132096] signed : [0, 132096]}}
      // expected-remark@+1 {{non-neg}}
      %14 = arith.addi %13, %arg4 : tensor<1024xi64>
      %15 = tt.splat %12 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
      %16 = tt.addptr %15, %14 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
      %17 = tt.load %16 : tensor<1024x!tt.ptr<f32>>
      %18 = arith.addf %17, %arg5 : tensor<1024xf32>
      scf.yield %12, %14, %18 : !tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>
    }
    %6 = tt.addptr %5#0, %1 : !tt.ptr<f32>, i32
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %7 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
    // expected-remark@+2 {{unsigned : [0, 132096] signed : [0, 132096]}}
    // expected-remark@+1 {{non-neg}}
    %8 = arith.addi %7, %5#1 : tensor<1024xi64>
    %9 = tt.splat %6 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %10 = tt.addptr %9, %8 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
    %11 = tt.load %10 : tensor<1024x!tt.ptr<f32>>
    tt.return %11 : tensor<1024xf32>
  }
}

// -----

// CHECK-LABEL:   tt.func @forOp2
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @forOp2(%arg0: !tt.ptr<f32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    // expected-remark@+2 {{unsigned : [0, 0] signed : [0, 0]}}
    // expected-remark@+1 {{non-neg}}
    %cst = arith.constant dense<0> : tensor<1024xi64>
    // expected-remark@+2 {{unsigned : [1024, 1024] signed : [1024, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %c1024_i32 = arith.constant 1024 : i32
    // expected-remark@+2 {{unsigned : [0, 0] signed : [0, 0]}}
    // expected-remark@+1 {{non-neg}}
    %c0 = arith.constant 0 : index
    // expected-remark@+2 {{unsigned : [128, 128] signed : [128, 128]}}
    // expected-remark@+1 {{non-neg}}
    %c128 = arith.constant 128 : index
    // expected-remark@+2 {{unsigned : [1, 1] signed : [1, 1]}}
    // expected-remark@+1 {{non-neg}}
    %c1 = arith.constant 1 : index
    // expected-remark@+2 {{unsigned : [0, 65535] signed : [0, 65535]}}
    // expected-remark@+1 {{non-neg}}
    %0 = tt.get_program_id x : i32
    // expected-remark@+2 {{unsigned : [0, 67107840] signed : [0, 67107840]}}
    // expected-remark@+1 {{non-neg}}
    %1 = arith.muli %0, %c1024_i32 : i32
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    // expected-remark@+1 {{result 1: non-neg}}
    %3:3 = scf.for %arg2 = %c0 to %c128 step %c1 iter_args(%arg3 = %arg0, %arg4 = %cst, %arg5 = %arg1) -> (!tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>) {
      %10 = tt.addptr %arg3, %1 : !tt.ptr<f32>, i32
      // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
      // expected-remark@+1 {{non-neg}}
      %11 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
      // expected-remark@+2 {{unsigned : [0, 131072] signed : [0, 131072]}}
      // expected-remark@+1 {{non-neg}}
      %12 = arith.addi %11, %arg4 : tensor<1024xi64>
      %13 = tt.splat %10 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
      %14 = tt.addptr %13, %12 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
      %15 = tt.load %14 : tensor<1024x!tt.ptr<f32>>
      %16 = arith.addf %15, %arg5 : tensor<1024xf32>
      scf.yield %10, %12, %16 : !tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>
    }
    %4 = tt.addptr %3#0, %1 : !tt.ptr<f32>, i32
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %5 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
    // expected-remark@+2 {{unsigned : [0, 131072] signed : [0, 131072]}}
    // expected-remark@+1 {{non-neg}}
    %6 = arith.addi %5, %3#1 : tensor<1024xi64>
    %7 = tt.splat %4 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %8 = tt.addptr %7, %6 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
    %9 = tt.load %8 : tensor<1024x!tt.ptr<f32>>
    tt.return %9 : tensor<1024xf32>
  }
}

// -----

// CHECK-LABEL:   tt.func @forNested
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @forNested(%arg0: !tt.ptr<f32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    // expected-remark@+2 {{unsigned : [0, 0] signed : [0, 0]}}
    // expected-remark@+1 {{non-neg}}
    %cst = arith.constant dense<0> : tensor<1024xi64>
    // expected-remark@+2 {{unsigned : [1024, 1024] signed : [1024, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %c1024_i32 = arith.constant 1024 : i32
    // expected-remark@+2 {{unsigned : [0, 0] signed : [0, 0]}}
    // expected-remark@+1 {{non-neg}}
    %c0 = arith.constant 0 : index
    // expected-remark@+2 {{unsigned : [16, 16] signed : [16, 16]}}
    // expected-remark@+1 {{non-neg}}
    %c16 = arith.constant 16 : index
    // expected-remark@+2 {{unsigned : [1, 1] signed : [1, 1]}}
    // expected-remark@+1 {{non-neg}}
    %c1 = arith.constant 1 : index
    // expected-remark@+2 {{unsigned : [0, 65535] signed : [0, 65535]}}
    // expected-remark@+1 {{non-neg}}
    %0 = tt.get_program_id x : i32
    // expected-remark@+2 {{unsigned : [0, 67107840] signed : [0, 67107840]}}
    // expected-remark@+1 {{non-neg}}
    %1 = arith.muli %0, %c1024_i32 : i32
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    // expected-remark@+1 {{result 1: non-neg}}
    %3:3 = scf.for %arg2 = %c0 to %c16 step %c1 iter_args(%arg3 = %arg0, %arg4 = %cst, %arg5 = %arg1) -> (!tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>) {
      // expected-remark@+1 {{result 1: non-neg}}
      %10:3 = scf.for %arg6 = %c0 to %c16 step %c1 iter_args(%arg7 = %arg3, %arg8 = %arg4, %arg9 = %arg5) -> (!tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>) {
        %11 = tt.addptr %arg7, %1 : !tt.ptr<f32>, i32
        // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
        // expected-remark@+1 {{non-neg}}
        %12 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
        // expected-remark@+2 {{unsigned : [0, 262144] signed : [0, 262144]}}
        // expected-remark@+1 {{non-neg}}
        %13 = arith.addi %12, %arg8 : tensor<1024xi64>
        %14 = tt.splat %11 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
        %15 = tt.addptr %14, %13 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
        %16 = tt.load %15 : tensor<1024x!tt.ptr<f32>>
        %17 = arith.addf %16, %arg9 : tensor<1024xf32>
        scf.yield %11, %13, %17 : !tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>
      }
      scf.yield %10#0, %10#1, %10#2 : !tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>
    }
    %4 = tt.addptr %3#0, %1 : !tt.ptr<f32>, i32
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %5 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
    // expected-remark@+2 {{unsigned : [0, 16384] signed : [0, 16384]}}
    // expected-remark@+1 {{non-neg}}
    %6 = arith.addi %5, %3#1 : tensor<1024xi64>
    %7 = tt.splat %4 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %8 = tt.addptr %7, %6 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
    %9 = tt.load %8 : tensor<1024x!tt.ptr<f32>>
    tt.return %9 : tensor<1024xf32>
  }
}

// -----

// CHECK-LABEL:   tt.func @forNestedOverMaxTripCount
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @forNestedOverMaxTripCount(%arg0: !tt.ptr<f32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    // expected-remark@+2 {{unsigned : [0, 0] signed : [0, 0]}}
    // expected-remark@+1 {{non-neg}}
    %cst = arith.constant dense<0> : tensor<1024xi64>
    // expected-remark@+2 {{unsigned : [1024, 1024] signed : [1024, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %c1024_i32 = arith.constant 1024 : i32
    // expected-remark@+2 {{unsigned : [0, 0] signed : [0, 0]}}
    // expected-remark@+1 {{non-neg}}
    %c0 = arith.constant 0 : index
    // expected-remark@+2 {{unsigned : [128, 128] signed : [128, 128]}}
    // expected-remark@+1 {{non-neg}}
    %c128 = arith.constant 128 : index
    // expected-remark@+2 {{unsigned : [1, 1] signed : [1, 1]}}
    // expected-remark@+1 {{non-neg}}
    %c1 = arith.constant 1 : index
    // expected-remark@+2 {{unsigned : [0, 65535] signed : [0, 65535]}}
    // expected-remark@+1 {{non-neg}}
    %0 = tt.get_program_id x : i32
    // expected-remark@+2 {{unsigned : [0, 67107840] signed : [0, 67107840]}}
    // expected-remark@+1 {{non-neg}}
    %1 = arith.muli %0, %c1024_i32 : i32
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3:3 = scf.for %arg2 = %c0 to %c128 step %c1 iter_args(%arg3 = %arg0, %arg4 = %cst, %arg5 = %arg1) -> (!tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>) {
      %10:3 = scf.for %arg6 = %c0 to %c128 step %c1 iter_args(%arg7 = %arg3, %arg8 = %arg4, %arg9 = %arg5) -> (!tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>) {
        %11 = tt.addptr %arg7, %1 : !tt.ptr<f32>, i32
        // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
        // expected-remark@+1 {{non-neg}}
        %12 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
        // expected-remark@+1 {{unsigned : [0, 18446744073709551615] signed : [-9223372036854775808, 9223372036854775807]}}
        %13 = arith.addi %12, %arg8 : tensor<1024xi64>
        %14 = tt.splat %11 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
        %15 = tt.addptr %14, %13 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
        %16 = tt.load %15 : tensor<1024x!tt.ptr<f32>>
        %17 = arith.addf %16, %arg9 : tensor<1024xf32>
        scf.yield %11, %13, %17 : !tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>
      }
      scf.yield %10#0, %10#1, %10#2 : !tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>
    }
    %4 = tt.addptr %3#0, %1 : !tt.ptr<f32>, i32
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %5 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
    // expected-remark@+1 {{unsigned : [0, 18446744073709551615] signed : [-9223372036854775808, 9223372036854775807]}}
    %6 = arith.addi %5, %3#1 : tensor<1024xi64>
    %7 = tt.splat %4 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %8 = tt.addptr %7, %6 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
    %9 = tt.load %8 : tensor<1024x!tt.ptr<f32>>
    tt.return %9 : tensor<1024xf32>
  }
}

// -----

// CHECK-LABEL:   tt.func @ifOp
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @ifOp(%arg0: !tt.ptr<f32>, %arg1: tensor<1024xf32>, %arg2: i1) -> tensor<1024xf32> {
    // expected-remark@+2 {{unsigned : [0, 0] signed : [0, 0]}}
    // expected-remark@+1 {{non-neg}}
    %cst = arith.constant dense<0> : tensor<1024xi64>
    // expected-remark@+2 {{unsigned : [1024, 1024] signed : [1024, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %c1024_i32 = arith.constant 1024 : i32
    // expected-remark@+2 {{unsigned : [0, 65535] signed : [0, 65535]}}
    // expected-remark@+1 {{non-neg}}
    %0 = tt.get_program_id x : i32
    // expected-remark@+2 {{unsigned : [0, 67107840] signed : [0, 67107840]}}
    // expected-remark@+1 {{non-neg}}
    %1 = arith.muli %0, %c1024_i32 : i32
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    // expected-remark@+1 {{result 1: non-neg}}
    %3:2 = scf.if %arg2 -> (!tt.ptr<f32>, tensor<1024xi64>) {
      %8 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
      // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
      // expected-remark@+1 {{non-neg}}
      %9 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
      scf.yield %8, %9 : !tt.ptr<f32>, tensor<1024xi64>
    } else {
      %8 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
      scf.yield %8, %cst : !tt.ptr<f32>, tensor<1024xi64>
    }
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %4 = arith.trunci %3#1 : tensor<1024xi64> to tensor<1024xi32>
    %5 = tt.splat %3#0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %6 = tt.addptr %5, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %7 = tt.load %6 : tensor<1024x!tt.ptr<f32>>
    tt.return %7 : tensor<1024xf32>
  }
}

// -----

// CHECK-LABEL:   tt.func @condBranch
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @condBranch(%arg0: !tt.ptr<f32>, %arg1: i1) -> tensor<1024xf32> {
    // expected-remark@+2 {{unsigned : [0, 0] signed : [0, 0]}}
    // expected-remark@+1 {{non-neg}}
    %cst = arith.constant dense<0> : tensor<1024xi64>
    // expected-remark@+2 {{unsigned : [1024, 1024] signed : [1024, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %c1024_i32 = arith.constant 1024 : i32
    // expected-remark@+2 {{unsigned : [0, 65535] signed : [0, 65535]}}
    // expected-remark@+1 {{non-neg}}
    %0 = tt.get_program_id x : i32
    // expected-remark@+2 {{unsigned : [0, 67107840] signed : [0, 67107840]}}
    // expected-remark@+1 {{non-neg}}
    %1 = arith.muli %0, %c1024_i32 : i32
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %4 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
    cf.cond_br %arg1, ^bb1(%arg0, %cst : !tt.ptr<f32>, tensor<1024xi64>), ^bb2(%3, %4 : !tt.ptr<f32>, tensor<1024xi64>)
  ^bb1(%5: !tt.ptr<f32>, %6: tensor<1024xi64>):  // pred: ^bb0
    // expected-remark@+2 {{unsigned : [0, 0] signed : [0, 0]}}
    // expected-remark@+1 {{non-neg}}
    %7 = arith.trunci %6 : tensor<1024xi64> to tensor<1024xi32>
    %8 = tt.splat %5 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %9 = tt.addptr %8, %7 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %10 = tt.load %9 : tensor<1024x!tt.ptr<f32>>
    tt.return %10 : tensor<1024xf32>
  ^bb2(%11: !tt.ptr<f32>, %12: tensor<1024xi64>):  // pred: ^bb0
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %13 = arith.trunci %12 : tensor<1024xi64> to tensor<1024xi32>
    %14 = tt.splat %11 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %15 = tt.addptr %14, %13 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %16 = tt.load %15 : tensor<1024x!tt.ptr<f32>>
    tt.return %16 : tensor<1024xf32>
  }
}

// -----

// CHECK-LABEL:   tt.func @branch
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @branch(%arg0: !tt.ptr<f32>, %arg1: i1) -> tensor<1024xf32> {
    // expected-remark@+2 {{unsigned : [1024, 1024] signed : [1024, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %c1024_i32 = arith.constant 1024 : i32
    // expected-remark@+2 {{unsigned : [0, 65535] signed : [0, 65535]}}
    // expected-remark@+1 {{non-neg}}
    %0 = tt.get_program_id x : i32
    // expected-remark@+2 {{unsigned : [0, 67107840] signed : [0, 67107840]}}
    // expected-remark@+1 {{non-neg}}
    %1 = arith.muli %0, %c1024_i32 : i32
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    %4 = tt.splat %3 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %5 = tt.addptr %4, %2 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %6 = tt.load %5 : tensor<1024x!tt.ptr<f32>>
    tt.return %6 : tensor<1024xf32>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @tile_offset(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32) -> tensor<16x256xf16, #blocked> {
    // expected-remark@+2 {{unsigned : [256, 256] signed : [256, 256]}}
    // expected-remark@+1 {{non-neg}}
    %c256_i32 = arith.constant 256 : i32
    // expected-remark@+2 {{unsigned : [0, 65535] signed : [0, 65535]}}
    // expected-remark@+1 {{non-neg}}
    %0 = tt.get_program_id x : i32
    // expected-remark@+2 {{unsigned : [0, 16776960] signed : [0, 16776960]}}
    // expected-remark@+1 {{non-neg}}
    %1 = arith.muli %0, %c256_i32 : i32
    // expected-remark@+2 {{unsigned : [0, 256] signed : [0, 256]}}
    // expected-remark@+1 {{non-neg}}
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    // expected-remark@+2 {{unsigned : [0, 16] signed : [0, 16]}}
    // expected-remark@+1 {{non-neg}}
    %3 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    // expected-remark@+2 {{unsigned : [0, 16] signed : [0, 16]}}
    // expected-remark@+1 {{non-neg}}
    %4 = tt.expand_dims %3 {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi32, #blocked>
    // expected-remark@+1 {{unsigned : [0, 4294967295] signed : [-2147483648, 2147483647]}}
    %5 = tt.splat %arg2 : i32 -> tensor<16x1xi32, #blocked>
    // expected-remark@+1 {{unsigned : [0, 4294967295] signed : [-2147483648, 2147483647]}}
    %6 = arith.muli %4, %5 : tensor<16x1xi32, #blocked>
    // expected-remark@+1 {{unsigned : [0, 4294967295] signed : [-2147483648, 2147483647]}}
    %7 = tt.broadcast %6 : tensor<16x1xi32, #blocked> -> tensor<16x256xi32, #blocked>
    // expected-remark@+2 {{unsigned : [0, 256] signed : [0, 256]}}
    // expected-remark@+1 {{non-neg}}
    %8 = tt.expand_dims %2 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
    // expected-remark@+2 {{unsigned : [0, 256] signed : [0, 256]}}
    // expected-remark@+1 {{non-neg}}
    %9 = tt.broadcast %8 : tensor<1x256xi32, #blocked> -> tensor<16x256xi32, #blocked>
    // expected-remark@+1 {{unsigned : [0, 4294967295] signed : [-2147483648, 2147483647]}}
    %10 = arith.addi %7, %9 : tensor<16x256xi32, #blocked>
    %11 = tt.addptr %arg0, %1 : !tt.ptr<f16>, i32
    %12 = tt.splat %11 : !tt.ptr<f16> -> tensor<16x256x!tt.ptr<f16>, #blocked>
    %13 = tt.addptr %12, %10 : tensor<16x256x!tt.ptr<f16>, #blocked>, tensor<16x256xi32, #blocked>
    %14 = tt.load %13 : tensor<16x256x!tt.ptr<f16>, #blocked>
    tt.return %14 : tensor<16x256xf16, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: i32 {tt.divisibility = 16 : i32}) -> tensor<128x16xf16, #blocked> {
    // expected-remark@+2 {{unsigned : [128, 128] signed : [128, 128]}}
    // expected-remark@+1 {{non-neg}}
    %c128_i32 = arith.constant 128 : i32
    // expected-remark@+2 {{unsigned : [0, 65535] signed : [0, 65535]}}
    // expected-remark@+1 {{non-neg}}
    %0 = tt.get_program_id x : i32
    // expected-remark@+2 {{unsigned : [0, 8388480] signed : [0, 8388480]}}
    // expected-remark@+1 {{non-neg}}
    %1 = arith.muli %0, %c128_i32 : i32
    // expected-remark@+2 {{unsigned : [0, 128] signed : [0, 128]}}
    // expected-remark@+1 {{non-neg}}
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    // expected-remark@+2 {{unsigned : [0, 16] signed : [0, 16]}}
    // expected-remark@+1 {{non-neg}}
    %3 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    // expected-remark@+2 {{unsigned : [0, 128] signed : [0, 128]}}
    // expected-remark@+1 {{non-neg}}
    %4 = tt.expand_dims %2 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    // expected-remark@+1 {{unsigned : [0, 4294967295] signed : [-2147483648, 2147483647]}}
    %5 = arith.muli %1, %arg1 : i32
    // expected-remark@+1 {{unsigned : [0, 4294967295] signed : [-2147483648, 2147483647]}}
    %6 = tt.splat %arg1 : i32 -> tensor<128x1xi32, #blocked>
    // expected-remark@+1 {{unsigned : [0, 4294967295] signed : [-2147483648, 2147483647]}}
    %7 = arith.muli %4, %6 : tensor<128x1xi32, #blocked>
    // expected-remark@+1 {{unsigned : [0, 4294967295] signed : [-2147483648, 2147483647]}}
    %8 = tt.broadcast %7 : tensor<128x1xi32, #blocked> -> tensor<128x16xi32, #blocked>
    // expected-remark@+2 {{unsigned : [0, 16] signed : [0, 16]}}
    // expected-remark@+1 {{non-neg}}
    %9 = tt.expand_dims %3 {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x16xi32, #blocked>
    // expected-remark@+2 {{unsigned : [0, 16] signed : [0, 16]}}
    // expected-remark@+1 {{non-neg}}
    %10 = tt.broadcast %9 : tensor<1x16xi32, #blocked> -> tensor<128x16xi32, #blocked>
    // expected-remark@+1 {{unsigned : [0, 4294967295] signed : [-2147483648, 2147483647]}}
    %11 = arith.addi %8, %10 : tensor<128x16xi32, #blocked>
    %12 = tt.addptr %arg0, %5 : !tt.ptr<f16>, i32
    %13 = tt.splat %12 : !tt.ptr<f16> -> tensor<128x16x!tt.ptr<f16>, #blocked>
    %14 = tt.addptr %13, %11 : tensor<128x16x!tt.ptr<f16>, #blocked>, tensor<128x16xi32, #blocked>
    %15 = tt.load %14 : tensor<128x16x!tt.ptr<f16>, #blocked>
    tt.return %15 : tensor<128x16xf16, #blocked>
  }
}

// -----

// CHECK-LABEL:   tt.func @select
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @select(%arg0: !tt.ptr<f32>, %arg1: i1) -> tensor<1024xf32> {
    // expected-remark@+2 {{unsigned : [0, 0] signed : [0, 0]}}
    // expected-remark@+1 {{non-neg}}
    %cst = arith.constant dense<0> : tensor<1024xi64>
    // expected-remark@+2 {{unsigned : [1024, 1024] signed : [1024, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %c1024_i32 = arith.constant 1024 : i32
    // expected-remark@+2 {{unsigned : [0, 65535] signed : [0, 65535]}}
    // expected-remark@+1 {{non-neg}}
    %0 = tt.get_program_id x : i32
    // expected-remark@+2 {{unsigned : [0, 67107840] signed : [0, 67107840]}}
    // expected-remark@+1 {{non-neg}}
    %1 = arith.muli %0, %c1024_i32 : i32
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %4 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
    %5 = arith.select %arg1, %arg0, %3 : !tt.ptr<f32>
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %6 = arith.select %arg1, %cst, %4 : tensor<1024xi64>
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %7 = arith.trunci %6 : tensor<1024xi64> to tensor<1024xi32>
    %8 = tt.splat %5 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %9 = tt.addptr %8, %7 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %10 = tt.load %9 : tensor<1024x!tt.ptr<f32>>
    tt.return %10 : tensor<1024xf32>
  }
}

// -----

// CHECK-LABEL:   tt.func @where_kernel
module attributes {"ttg.num-ctas" = 1 : i32} {
  tt.func @where_kernel(%arg0: !tt.ptr<i64>, %arg1: !tt.ptr<i64>, %arg2: i8) -> tensor<1024xi64> {
    // expected-remark@+2 {{unsigned : [0, 0] signed : [0, 0]}}
    // expected-remark@+1 {{non-neg}}
    %c0_i8 = arith.constant 0 : i8
    // expected-remark@+2 {{unsigned : [1024, 1024] signed : [1024, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %c1024_i32 = arith.constant 1024 : i32
    // expected-remark@+2 {{unsigned : [0, 65535] signed : [0, 65535]}}
    // expected-remark@+1 {{non-neg}}
    %0 = tt.get_program_id x : i32
    // expected-remark@+2 {{unsigned : [0, 67107840] signed : [0, 67107840]}}
    // expected-remark@+1 {{non-neg}}
    %1 = arith.muli %0, %c1024_i32 : i32
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    // expected-remark@+1 {{unsigned : [0, 1] signed : [-1, 0]}}
    %3 = arith.cmpi ne, %arg2, %c0_i8 : i8
    %4 = arith.select %3, %arg0, %arg1 : !tt.ptr<i64>
    %5 = tt.addptr %4, %1 : !tt.ptr<i64>, i32
    %6 = tt.splat %5 : !tt.ptr<i64> -> tensor<1024x!tt.ptr<i64>>
    %7 = tt.addptr %6, %2 : tensor<1024x!tt.ptr<i64>>, tensor<1024xi32>
    // expected-remark@+1 {{unsigned : [0, 18446744073709551615] signed : [-9223372036854775808, 9223372036854775807]}}
    %8 = tt.load %7 : tensor<1024x!tt.ptr<i64>>
    tt.return %8 : tensor<1024xi64>
  }
}

// -----

// CHECK-LABEL:   tt.func @forOpWithHints
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @forOpWithHints(%arg0: !tt.ptr<f32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    // expected-remark@+2 {{unsigned : [0, 0] signed : [0, 0]}}
    // expected-remark@+1 {{non-neg}}
    %c0 = arith.constant 0 : index
    // expected-remark@+2 {{unsigned : [1, 1] signed : [1, 1]}}
    // expected-remark@+1 {{non-neg}}
    %c1 = arith.constant 1 : index
    // expected-remark@+2 {{unsigned : [128, 128] signed : [128, 128]}}
    // expected-remark@+1 {{non-neg}}
    %c128 = arith.constant 128 : index
    // expected-remark@+2 {{unsigned : [0, 65535] signed : [0, 65535]}}
    // expected-remark@+1 {{non-neg}}
    %0 = tt.get_program_id x : i32
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %1 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %2 = tt.addptr %arg0, %0 : !tt.ptr<f32>, i32
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %3 = arith.extsi %1 : tensor<1024xi32> to tensor<1024xi64>
    // expected-remark@+1 {{result 1: non-neg}}
    %4:3 = scf.for %arg2 = %c0 to %c128 step %c1 iter_args(%arg3 = %2, %arg4 = %3, %arg5 = %arg1) -> (!tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>) {
      // expected-remark@+2 {{unsigned : [0, 131072] signed : [0, 131072]}}
      // expected-remark@+1 {{non-neg}}
      %11 = arith.trunci %arg4 : tensor<1024xi64> to tensor<1024xi32>
      %12 = tt.splat %arg3 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
      %13 = tt.addptr %12, %11 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
      %14 = tt.load %13 : tensor<1024x!tt.ptr<f32>>
      %15 = tt.addptr %arg3, %0 : !tt.ptr<f32>, i32
      // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
      // expected-remark@+1 {{non-neg}}
      %16 = arith.extsi %1 : tensor<1024xi32> to tensor<1024xi64>
      // expected-remark@+2 {{unsigned : [0, 132096] signed : [0, 132096]}}
      // expected-remark@+1 {{non-neg}}
      %17 = arith.addi %16, %arg4 : tensor<1024xi64>
      %18 = tt.addptr %15, %0 : !tt.ptr<f32>, i32
      %19 = arith.addf %14, %arg5 : tensor<1024xf32>
      scf.yield %18, %17, %19 : !tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>
    } {tt.divisibility_arg1 = dense<16> : tensor<1xi32>, tt.divisibility_arg2 = dense<16> : tensor<1xi32>}
    %5 = tt.addptr %4#0, %0 : !tt.ptr<f32>, i32
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %6 = arith.extsi %1 : tensor<1024xi32> to tensor<1024xi64>
    // expected-remark@+2 {{unsigned : [0, 132096] signed : [0, 132096]}}
    // expected-remark@+1 {{non-neg}}
    %7 = arith.addi %6, %4#1 : tensor<1024xi64>
    %8 = tt.splat %5 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %9 = tt.addptr %8, %7 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
    %10 = tt.load %9 : tensor<1024x!tt.ptr<f32>>
    tt.return %10 : tensor<1024xf32>
  }
}

// -----

// CHECK-LABEL:   tt.func public @scalar_pointers
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func public @scalar_pointers(%arg0: !tt.ptr<i64> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    // expected-remark@+2 {{unsigned : [0, 0] signed : [0, 0]}}
    // expected-remark@+1 {{non-neg}}
    %c0_i64 = arith.constant 0 : i64
    // expected-remark@+2 {{unsigned : [1, 1] signed : [1, 1]}}
    // expected-remark@+1 {{non-neg}}
    %c1_i32 = arith.constant 1 : i32
    // expected-remark@+2 {{unsigned : [100, 100] signed : [100, 100]}}
    // expected-remark@+1 {{non-neg}}
    %c100_i32 = arith.constant 100 : i32
    %0 = tt.addptr %arg0, %c1_i32 : !tt.ptr<i64>, i32
    %1 = scf.for %arg1 = %c1_i32 to %c100_i32 step %c1_i32 iter_args(%arg2 = %0) -> (!tt.ptr<i64>)  : i32 {
      tt.store %arg2, %c0_i64 : !tt.ptr<i64>
      %2 = tt.addptr %arg2, %c1_i32 : !tt.ptr<i64>, i32
      scf.yield %2 : !tt.ptr<i64>
    }
    tt.return
  }
}

// -----

// CHECK-LABEL:   tt.func @scalar_if
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @scalar_if(%arg0: !tt.ptr<f32>, %arg1: tensor<1024xf32>, %arg2: i1) -> f32 {
    // expected-remark@+2 {{unsigned : [1, 1] signed : [1, 1]}}
    // expected-remark@+1 {{non-neg}}
    %c1_i32 = arith.constant 1 : i32
    // expected-remark@+2 {{unsigned : [100, 100] signed : [100, 100]}}
    // expected-remark@+1 {{non-neg}}
    %c100_i32 = arith.constant 100 : i32
    %0 = tt.addptr %arg0, %c1_i32 : !tt.ptr<f32>, i32
    %1 = scf.if %arg2 -> (!tt.ptr<f32>) {
      %3 = tt.addptr %0, %c1_i32 : !tt.ptr<f32>, i32
      scf.yield %3 : !tt.ptr<f32>
    } else {
      %3 = tt.addptr %0, %c100_i32 : !tt.ptr<f32>, i32
      scf.yield %3 : !tt.ptr<f32>
    }
    %2 = tt.load %1 : !tt.ptr<f32>
    tt.return %2 : f32
  }
}

// -----

// CHECK-LABEL:   tt.func @scalar_cond_branch
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @scalar_cond_branch(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i1) -> f32 {
    cf.cond_br %arg2, ^bb1(%arg0 : !tt.ptr<f32>), ^bb2(%arg1 : !tt.ptr<f32>)
  ^bb1(%0: !tt.ptr<f32>):  // pred: ^bb0
    %1 = tt.load %0 : !tt.ptr<f32>
    tt.return %1 : f32
  ^bb2(%2: !tt.ptr<f32>):  // pred: ^bb0
    %3 = tt.load %2 : !tt.ptr<f32>
    tt.return %3 : f32
  }
}

// -----

// CHECK-LABEL:   tt.func @flipFlopForOpSimple
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @flipFlopForOpSimple(%arg0: !tt.ptr<f32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    // expected-remark@+2 {{unsigned : [1024, 1024] signed : [1024, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %c1024_i32 = arith.constant 1024 : i32
    // expected-remark@+2 {{unsigned : [0, 0] signed : [0, 0]}}
    // expected-remark@+1 {{non-neg}}
    %c0 = arith.constant 0 : index
    // expected-remark@+2 {{unsigned : [128, 128] signed : [128, 128]}}
    // expected-remark@+1 {{non-neg}}
    %c128 = arith.constant 128 : index
    // expected-remark@+2 {{unsigned : [1, 1] signed : [1, 1]}}
    // expected-remark@+1 {{non-neg}}
    %c1 = arith.constant 1 : index
    // expected-remark@+2 {{unsigned : [0, 65535] signed : [0, 65535]}}
    // expected-remark@+1 {{non-neg}}
    %0 = tt.get_program_id x : i32
    // expected-remark@+2 {{unsigned : [0, 67107840] signed : [0, 67107840]}}
    // expected-remark@+1 {{non-neg}}
    %1 = arith.muli %0, %c1024_i32 : i32
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %4 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
    %5 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %6 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
    // expected-remark@+2 {{result 1: non-neg}}
    // expected-remark@+1 {{result 3: non-neg}}
    %7:5 = scf.for %arg2 = %c0 to %c128 step %c1 iter_args(%arg3 = %5, %arg4 = %6, %arg5 = %3, %arg6 = %4, %arg7 = %arg1) -> (!tt.ptr<f32>, tensor<1024xi64>, !tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>) {
      %14 = tt.addptr %arg5, %1 : !tt.ptr<f32>, i32
      // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
      // expected-remark@+1 {{non-neg}}
      %15 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
      // expected-remark@+2 {{unsigned : [0, 132096] signed : [0, 132096]}}
      // expected-remark@+1 {{non-neg}}
      %16 = arith.addi %15, %arg6 : tensor<1024xi64>
      %17 = tt.splat %14 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
      %18 = tt.addptr %17, %16 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
      %19 = tt.load %18 : tensor<1024x!tt.ptr<f32>>
      %20 = arith.addf %19, %arg7 : tensor<1024xf32>
      scf.yield %14, %16, %arg3, %arg4, %20 : !tt.ptr<f32>, tensor<1024xi64>, !tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>
    }
    %8 = tt.addptr %7#0, %1 : !tt.ptr<f32>, i32
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %9 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
    // expected-remark@+2 {{unsigned : [0, 132096] signed : [0, 132096]}}
    // expected-remark@+1 {{non-neg}}
    %10 = arith.addi %9, %7#1 : tensor<1024xi64>
    %11 = tt.splat %8 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %12 = tt.addptr %11, %10 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
    %13 = tt.load %12 : tensor<1024x!tt.ptr<f32>>
    tt.return %13 : tensor<1024xf32>
  }
}

// -----

// CHECK-LABEL:   tt.func @flipFlopForOpComplex
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @flipFlopForOpComplex(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: tensor<1024xf32>) -> (tensor<1024xf32>, tensor<1024xf32>) {
    // expected-remark@+2 {{unsigned : [1024, 1024] signed : [1024, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %c1024_i32 = arith.constant 1024 : i32
    // expected-remark@+2 {{unsigned : [0, 0] signed : [0, 0]}}
    // expected-remark@+1 {{non-neg}}
    %c0 = arith.constant 0 : index
    // expected-remark@+2 {{unsigned : [128, 128] signed : [128, 128]}}
    // expected-remark@+1 {{non-neg}}
    %c128 = arith.constant 128 : index
    // expected-remark@+2 {{unsigned : [1, 1] signed : [1, 1]}}
    // expected-remark@+1 {{non-neg}}
    %c1 = arith.constant 1 : index
    // expected-remark@+2 {{unsigned : [0, 65535] signed : [0, 65535]}}
    // expected-remark@+1 {{non-neg}}
    %0 = tt.get_program_id x : i32
    // expected-remark@+2 {{unsigned : [0, 67107840] signed : [0, 67107840]}}
    // expected-remark@+1 {{non-neg}}
    %1 = arith.muli %0, %c1024_i32 : i32
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %4 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
    %5 = tt.addptr %arg1, %1 : !tt.ptr<f32>, i32
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %6 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
    // expected-remark@+2 {{result 1: non-neg}}
    // expected-remark@+1 {{result 4: non-neg}}
    %7:6 = scf.for %arg3 = %c0 to %c128 step %c1 iter_args(%arg4 = %3, %arg5 = %4, %arg6 = %arg2, %arg7 = %5, %arg8 = %6, %arg9 = %arg2) -> (!tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>, !tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>) {
      %20 = tt.addptr %arg4, %1 : !tt.ptr<f32>, i32
      // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
      // expected-remark@+1 {{non-neg}}
      %21 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
      // expected-remark@+2 {{unsigned : [0, 132096] signed : [0, 132096]}}
      // expected-remark@+1 {{non-neg}}
      %22 = arith.addi %21, %arg5 : tensor<1024xi64>
      %23 = tt.splat %20 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
      %24 = tt.addptr %23, %22 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
      %25 = tt.load %24 : tensor<1024x!tt.ptr<f32>>
      %26 = arith.addf %25, %arg6 : tensor<1024xf32>
      %27 = tt.addptr %arg7, %1 : !tt.ptr<f32>, i32
      // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
      // expected-remark@+1 {{non-neg}}
      %28 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
      // expected-remark@+2 {{unsigned : [0, 132096] signed : [0, 132096]}}
      // expected-remark@+1 {{non-neg}}
      %29 = arith.addi %28, %arg8 : tensor<1024xi64>
      %30 = tt.splat %27 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
      %31 = tt.addptr %30, %29 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
      %32 = tt.load %31 : tensor<1024x!tt.ptr<f32>>
      %33 = arith.addf %32, %arg9 : tensor<1024xf32>
      scf.yield %27, %29, %33, %20, %22, %26 : !tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>, !tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>
    }
    %8 = tt.addptr %7#0, %1 : !tt.ptr<f32>, i32
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %9 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
    // expected-remark@+2 {{unsigned : [0, 132096] signed : [0, 132096]}}
    // expected-remark@+1 {{non-neg}}
    %10 = arith.addi %9, %7#1 : tensor<1024xi64>
    %11 = tt.splat %8 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %12 = tt.addptr %11, %10 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
    %13 = tt.load %12 : tensor<1024x!tt.ptr<f32>>
    %14 = tt.addptr %7#3, %1 : !tt.ptr<f32>, i32
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %15 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
    // expected-remark@+2 {{unsigned : [0, 132096] signed : [0, 132096]}}
    // expected-remark@+1 {{non-neg}}
    %16 = arith.addi %15, %7#4 : tensor<1024xi64>
    %17 = tt.splat %14 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %18 = tt.addptr %17, %16 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
    %19 = tt.load %18 : tensor<1024x!tt.ptr<f32>>
    tt.return %13, %19 : tensor<1024xf32>, tensor<1024xf32>
  }
}

// -----

// CHECK-LABEL:   tt.func @forOpDynamicKBound
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @forOpDynamicKBound(%arg0: !tt.ptr<f32>, %arg1: tensor<1024xf32>, %K: index) -> tensor<1024xf32> {
    // expected-remark@+2 {{unsigned : [1024, 1024] signed : [1024, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %c1024_i32 = arith.constant 1024 : i32
    // expected-remark@+2 {{unsigned : [0, 0] signed : [0, 0]}}
    // expected-remark@+1 {{non-neg}}
    %c0 = arith.constant 0 : index
    // expected-remark@+2 {{unsigned : [128, 128] signed : [128, 128]}}
    // expected-remark@+1 {{non-neg}}
    %c128 = arith.constant 128 : index
    // expected-remark@+2 {{unsigned : [1, 1] signed : [1, 1]}}
    // expected-remark@+1 {{non-neg}}
    %c1 = arith.constant 1 : index
    // expected-remark@+2 {{unsigned : [0, 65535] signed : [0, 65535]}}
    // expected-remark@+1 {{non-neg}}
    %0 = tt.get_program_id x : i32
    // expected-remark@+2 {{unsigned : [0, 67107840] signed : [0, 67107840]}}
    // expected-remark@+1 {{non-neg}}
    %1 = arith.muli %0, %c1024_i32 : i32
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %4 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
    %5:3 = scf.for %arg2 = %c0 to %c128 step %K iter_args(%arg3 = %3, %arg4 = %4, %arg5 = %arg1) -> (!tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>) {
      %12 = tt.addptr %arg3, %1 : !tt.ptr<f32>, i32
      // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
      // expected-remark@+1 {{non-neg}}
      %13 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
      // expected-remark@+1 {{unsigned : [0, 18446744073709551615] signed : [-9223372036854775808, 9223372036854775807]}}
      %14 = arith.addi %13, %arg4 : tensor<1024xi64>
      %15 = tt.splat %12 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
      %16 = tt.addptr %15, %14 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
      %17 = tt.load %16 : tensor<1024x!tt.ptr<f32>>
      %18 = arith.addf %17, %arg5 : tensor<1024xf32>
      scf.yield %12, %14, %18 : !tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>
    }
    %6 = tt.addptr %5#0, %1 : !tt.ptr<f32>, i32
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %7 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
    // expected-remark@+1 {{unsigned : [0, 18446744073709551615] signed : [-9223372036854775808, 9223372036854775807]}}
    %8 = arith.addi %7, %5#1 : tensor<1024xi64>
    %9 = tt.splat %6 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %10 = tt.addptr %9, %8 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
    %11 = tt.load %10 : tensor<1024x!tt.ptr<f32>>
    tt.return %11 : tensor<1024xf32>
  }
}

// -----

// CHECK-LABEL:   tt.func @DynamicKBound
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @DynamicKBound(%K: i32) {
    // expected-remark@+2 {{unsigned : [1024, 1024] signed : [1024, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %c1024_i32 = arith.constant 1024 : i32
    // expected-remark@+2 {{unsigned : [128, 128] signed : [128, 128]}}
    // expected-remark@+1 {{non-neg}}
    %c128 = arith.constant 128 : i32
    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %cmp = arith.cmpi sle, %K, %c128 : i32
    llvm.intr.assume %cmp : i1
    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %condtest = arith.cmpi sle, %K, %c1024_i32 : i32
    tt.return
  }
}

// -----

// CHECK-LABEL:   tt.func @unsupportedAssumption
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @unsupportedAssumption(%K: i32) {
    // expected-remark@+2 {{unsigned : [1024, 1024] signed : [1024, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %c1024_i32 = arith.constant 1024 : i32
    // expected-remark@+2 {{unsigned : [128, 128] signed : [128, 128]}}
    // expected-remark@+1 {{non-neg}}
    %c128 = arith.constant 128 : i32
    // expected-remark@+2 {{unsigned : [0, 1] signed : [-1, 0]}}
    // expected-remark@+1 {{unsigned arithmetic not currently supported}}
    %cmp = arith.cmpi ule, %K, %c128 : i32
    llvm.intr.assume %cmp : i1
    // expected-remark@+1 {{unsigned : [0, 1] signed : [-1, 0]}}
    %condtest = arith.cmpi sle, %K, %c1024_i32 : i32
    tt.return
  }
}

// -----

// CHECK-LABEL:   tt.func @moreDynamicKBound
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @moreDynamicKBound(
        %Keqlhs: i32,
        %Ksgelhs: i32,
        %Ksgtlhs: i32,
        %Kslelhs: i32,
        %Ksltlhs: i32,
        %Keqrhs: i32,
        %Ksgerhs: i32,
        %Ksgtrhs: i32,
        %Kslerhs: i32,
        %Ksltrhs: i32
    ) {
    // expected-remark@+2 {{unsigned : [0, 0] signed : [0, 0]}}
    // expected-remark@+1 {{non-neg}}
    %c0 = arith.constant 0 : i32
    // expected-remark@+2 {{unsigned : [16, 16] signed : [16, 16]}}
    // expected-remark@+1 {{non-neg}}
    %c16 = arith.constant 16 : i32
    // expected-remark@+2 {{unsigned : [32, 32] signed : [32, 32]}}
    // expected-remark@+1 {{non-neg}}
    %c32 = arith.constant 32 : i32
    // expected-remark@+2 {{unsigned : [64, 64] signed : [64, 64]}}
    // expected-remark@+1 {{non-neg}}
    %c64 = arith.constant 64 : i32
    // expected-remark@+2 {{unsigned : [128, 128] signed : [128, 128]}}
    // expected-remark@+1 {{non-neg}}
    %c128 = arith.constant 128 : i32
    // expected-remark@+2 {{unsigned : [256, 256] signed : [256, 256]}}
    // expected-remark@+1 {{non-neg}}
    %c256 = arith.constant 256 : i32
    // expected-remark@+2 {{unsigned : [1024, 1024] signed : [1024, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %c1024_i32 = arith.constant 1024 : i32

    //// eq comparison

    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %assumeeqlhs = arith.cmpi eq, %Keqlhs, %c128 : i32
    llvm.intr.assume %assumeeqlhs : i1
    // expected-remark@+2 {{unsigned : [128, 128] signed : [128, 128]}}
    // expected-remark@+1 {{non-neg}}
    %testeqlhs1 = arith.addi %Keqlhs, %c0 : i32
    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %testeqlhs2 = arith.cmpi ne, %Keqlhs, %c256 : i32

    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %assumeeqrhs = arith.cmpi eq, %c64, %Keqrhs : i32
    llvm.intr.assume %assumeeqrhs : i1
    // expected-remark@+2 {{unsigned : [64, 64] signed : [64, 64]}}
    // expected-remark@+1 {{non-neg}}
    %testeqrhs1 = arith.addi %Keqrhs, %c0 : i32
    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %testeqrhs2 = arith.cmpi ne, %Keqrhs, %c256 : i32

    //// sge comparison

    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %assumesgelhs = arith.cmpi sge, %Ksgelhs, %c128 : i32
    llvm.intr.assume %assumesgelhs : i1
    // expected-remark@+2 {{unsigned : [128, 2147483647] signed : [128, 2147483647]}}
    // expected-remark@+1 {{non-neg}}
    %testsgelhs1 = arith.addi %Ksgelhs, %c0 : i32
    // expected-remark@+1 {{unsigned : [0, 1] signed : [-1, 0]}}
    %testsgelhs2 = arith.cmpi sge, %Ksgelhs, %c1024_i32 : i32

    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %assumesgerhs = arith.cmpi sge, %c128, %Ksgerhs  : i32
    llvm.intr.assume %assumesgerhs : i1
    // expected-remark@+1 {{unsigned : [0, 4294967295] signed : [-2147483648, 128]}}
    %testsgerhs1 = arith.addi %Ksgerhs, %c0 : i32
    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %testsgerhs2 = arith.cmpi sge, %c1024_i32, %Ksgerhs : i32

    //// sgt comparison

    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %assumesgtlhs = arith.cmpi sgt, %Ksgtlhs, %c128 : i32
    llvm.intr.assume %assumesgtlhs : i1
    // expected-remark@+2 {{unsigned : [129, 2147483647] signed : [129, 2147483647]}}
    // expected-remark@+1 {{non-neg}}
    %testsgtlhs1 = arith.addi %Ksgtlhs, %c0 : i32
    // expected-remark@+1 {{unsigned : [0, 1] signed : [-1, 0]}}
    %testsgtlhs2 = arith.cmpi sgt, %Ksgtlhs, %c1024_i32 : i32

    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %assumesgtrhs = arith.cmpi sgt, %c128, %Ksgtrhs  : i32
    llvm.intr.assume %assumesgtrhs : i1
    // expected-remark@+1 {{unsigned : [0, 4294967295] signed : [-2147483648, 127]}}
    %testsgtrhs1 = arith.addi %Ksgtrhs, %c0 : i32
    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %testsgtrhs2 = arith.cmpi sgt, %c1024_i32, %Ksgtrhs : i32

    //// sle comparison

    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %assumeslelhs = arith.cmpi sle, %Kslelhs, %c128 : i32
    llvm.intr.assume %assumeslelhs : i1
    // expected-remark@+1 {{unsigned : [0, 4294967295] signed : [-2147483648, 128]}}
    %testslelhs1 = arith.addi %Kslelhs, %c0 : i32
    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %testslelhs2 = arith.cmpi sle, %Kslelhs, %c1024_i32 : i32

    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %assumeslerhs = arith.cmpi sle, %c128, %Kslerhs  : i32
    llvm.intr.assume %assumeslerhs : i1
    // expected-remark@+2 {{unsigned : [128, 2147483647] signed : [128, 2147483647]}}
    // expected-remark@+1 {{non-neg}}
    %testslerhs1 = arith.addi %Kslerhs, %c0 : i32
    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %testslerhs2 = arith.cmpi sle, %c64, %Kslerhs : i32

    //// slt comparison

    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %assumesltlhs = arith.cmpi slt, %Ksltlhs, %c128 : i32
    llvm.intr.assume %assumesltlhs : i1
    // expected-remark@+1 {{unsigned : [0, 4294967295] signed : [-2147483648, 127]}}
    %testsltlhs1 = arith.addi %Ksltlhs, %c0 : i32
    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %testsltlhs2 = arith.cmpi slt, %Ksltlhs, %c1024_i32 : i32

    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %assumesltrhs = arith.cmpi slt, %c128, %Ksltrhs  : i32
    llvm.intr.assume %assumesltrhs : i1
    // expected-remark@+2 {{unsigned : [129, 2147483647] signed : [129, 2147483647]}}
    // expected-remark@+1 {{non-neg}}
    %testsltrhs1 = arith.addi %Ksltrhs, %c0 : i32
    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %testsltrhs2 = arith.cmpi slt, %c64, %Ksltrhs : i32

    tt.return
  }
}

// -----


// CHECK-LABEL: join_cat_transitive_nonneg
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @join_cat_transitive_nonneg(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>) {
    // expected-remark@+2 {{unsigned : [0, 8] signed : [0, 8]}}
    // expected-remark@+1 {{non-neg}}
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    // expected-remark@+2 {{unsigned : [2, 10] signed : [2, 10]}}
    // expected-remark@+1 {{non-neg}}
    %1 = tt.make_range {end = 10 : i32, start = 2 : i32} : tensor<8xi32>
    // expected-remark@+2 {{unsigned : [0, 10] signed : [0, 10]}}
    // expected-remark@+1 {{non-neg}}
    %2 = tt.join %0, %1 : tensor<8xi32> -> tensor<8x2xi32>
    // expected-remark@+2 {{unsigned : [0, 4] signed : [0, 4]}}
    // expected-remark@+1 {{non-neg}}
    %3 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    // expected-remark@+2 {{unsigned : [4, 8] signed : [4, 8]}}
    // expected-remark@+1 {{non-neg}}
    %4 = tt.make_range {end = 8 : i32, start = 4 : i32} : tensor<4xi32>
    // expected-remark@+2 {{unsigned : [0, 8] signed : [0, 8]}}
    // expected-remark@+1 {{non-neg}}
    %5 = tt.join %3, %4 : tensor<4xi32> -> tensor<4x2xi32>
    // expected-remark@+2 {{unsigned : [0, 8] signed : [0, 8]}}
    // expected-remark@+1 {{non-neg}}
    %6 = tt.cat %5, %5 : tensor<4x2xi32> -> tensor<8x2xi32>
    // expected-remark@+2 {{unsigned : [0, 18] signed : [0, 18]}}
    // expected-remark@+1 {{non-neg}}
    %7 = arith.addi %2, %6 : tensor<8x2xi32>
    // expected-remark@+2 {{unsigned : [0, 0] signed : [0, 0]}}
    // expected-remark@+1 {{non-neg}}
    %zeros = arith.constant dense<0> : tensor<8x1xi32>
    // expected-remark@+2 {{unsigned : [1, 1] signed : [1, 1]}}
    // expected-remark@+1 {{non-neg}}
    %ones = arith.constant dense<1> : tensor<8x1xi32>
    // expected-remark@+2 {{unsigned : [0, 18] signed : [0, 18]}}
    // expected-remark@+1 {{non-neg}}
    %8 = tt.gather %7[%zeros] {axis = 1 : i32} : (tensor<8x2xi32>, tensor<8x1xi32>) -> tensor<8x1xi32>
    // expected-remark@+2 {{unsigned : [0, 18] signed : [0, 18]}}
    // expected-remark@+1 {{non-neg}}
    %9 = tt.gather %7[%ones] {axis = 1 : i32} : (tensor<8x2xi32>, tensor<8x1xi32>) -> tensor<8x1xi32>
    // expected-remark@+2 {{unsigned : [0, 36] signed : [0, 36]}}
    // expected-remark@+1 {{non-neg}}
    %10 = arith.addi %8, %9 : tensor<8x1xi32>
    // expected-remark@+2 {{unsigned : [0, 36] signed : [0, 36]}}
    // expected-remark@+1 {{non-neg}}
    %11 = tt.reshape %10 allow_reorder : tensor<8x1xi32> -> tensor<8xi32>
    tt.return
  }
}

// -----

// CHECK-LABEL: histo_nonneg
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @histo_nonneg(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>, %arg2 : tensor<256xi32>) {
    // expected-remark@+2 {{unsigned : [0, 4294967295] signed : [0, -1]}}
    // expected-remark@+1 {{non-neg}}
    %0 = tt.histogram %arg2 : tensor<256xi32> -> tensor<8xi32>
    // expected-remark@+2 {{unsigned : [0, 8] signed : [0, 8]}}
    // expected-remark@+1 {{non-neg}}
    %1 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    tt.return
  }
}

// -----

// CHECK-LABEL: get_num_prog_nonneg
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @get_num_prog_nonneg(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>, %arg2 : i32) {
    // expected-remark@+2 {{unsigned : [0, 65536] signed : [0, 65536]}}
    // expected-remark@+1 {{non-neg}}
    %0 = tt.get_num_programs x : i32
    // expected-remark@+2 {{unsigned : [0, 65536] signed : [0, 65536]}}
    // expected-remark@+1 {{non-neg}}
    %1 = tt.get_num_programs y : i32
    // expected-remark@+2 {{unsigned : [0, 65536] signed : [0, 65536]}}
    // expected-remark@+1 {{non-neg}}
    %2 = tt.get_num_programs z : i32
    // expected-remark@+2 {{unsigned : [0, 65536] signed : [0, 65536]}}
    // expected-remark@+1 {{non-neg}}
    %3 = arith.minsi %0, %1 : i32
    // expected-remark@+2 {{unsigned : [0, 65536] signed : [0, 65536]}}
    // expected-remark@+1 {{non-neg}}
    %4 = arith.minsi %2, %3 : i32
    // expected-remark@+2 {{unsigned : [0, 2147483647] signed : [0, 2147483647]}}
    // expected-remark@+1 {{non-neg}}
    %5 = arith.maxsi %arg2, %4 : i32
    // expected-remark@+2 {{[0, 2147483647] signed : [0, 2147483647]}}
    // expected-remark@+1 {{non-neg}}
    %6 = tt.splat %5 : i32 -> tensor<8xi32>
    // expected-remark@+2 {{unsigned : [0, 8] signed : [0, 8]}}
    // expected-remark@+1 {{non-neg}}
    %7 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    // expected-remark@+1 {{unsigned : [0, 2147483655] signed : [-2147483648, 2147483647]}}
    %8 = arith.addi %6, %7 : tensor<8xi32>
    tt.return
  }
}

// -----

// CHECK-LABEL: unary_triton_ops_transitive_nonneg
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @unary_triton_ops_transitive_nonneg(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>) {
    // expected-remark@+2 {{unsigned : [5, 5] signed : [5, 5]}}
    // expected-remark@+1 {{non-neg}}
    %c10_i32 = arith.constant 5 : i32
    // expected-remark@+2 {{unsigned : [0, 16] signed : [0, 16]}}
    // expected-remark@+1 {{non-neg}}
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    // expected-remark@+2 {{unsigned : [0, 16] signed : [0, 16]}}
    // expected-remark@+1 {{non-neg}}
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    // expected-remark@+2 {{unsigned : [0, 16] signed : [0, 16]}}
    // expected-remark@+1 {{non-neg}}
    %2 = tt.reshape %1 allow_reorder : tensor<1x16xi32> -> tensor<8x2xi32>
    // expected-remark@+2 {{unsigned : [0, 16] signed : [0, 16]}}
    // expected-remark@+1 {{non-neg}}
    %3 = tt.reshape %1 allow_reorder : tensor<1x16xi32> -> tensor<2x8xi32>
    // expected-remark@+2 {{unsigned : [0, 16] signed : [0, 16]}}
    // expected-remark@+1 {{non-neg}}
    %4 = tt.trans %3 {order = array<i32: 1, 0>} : tensor<2x8xi32> -> tensor<8x2xi32>
    // expected-remark@+2 {{unsigned : [0, 16] signed : [0, 16]}}
    // expected-remark@+1 {{non-neg}}
    %5 = ttg.convert_layout %4 : tensor<8x2xi32> -> tensor<8x2xi32>
    // expected-remark@+2 {{unsigned : [0, 32] signed : [0, 32]}}
    // expected-remark@+1 {{non-neg}}
    %6 = arith.addi %5, %2 : tensor<8x2xi32>
    // expected-remark@+2 {{unsigned : [2, 10] signed : [2, 10]}}
    // expected-remark@+1 {{non-neg}}
    %7 = tt.make_range {end = 10 : i32, start = 2 : i32} : tensor<8xi32>
    // expected-remark@+2 {{unsigned : [2, 10] signed : [2, 10]}}
    // expected-remark@+1 {{non-neg}}
    %8 = ttg.convert_layout %7 : tensor<8xi32> -> tensor<8xi32>
    // expected-remark@+2 {{unsigned : [2, 10] signed : [2, 10]}}
    // expected-remark@+1 {{non-neg}}
    %9 = tt.expand_dims %8 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    // expected-remark@+2 {{unsigned : [2, 10] signed : [2, 10]}}
    // expected-remark@+1 {{non-neg}}
    %10 = tt.broadcast %9 : tensor<1x8xi32> -> tensor<2x8xi32>
    // expected-remark@+2 {{unsigned : [2, 10] signed : [2, 10]}}
    // expected-remark@+1 {{non-neg}}
    %11 = tt.reshape %10 allow_reorder : tensor<2x8xi32> -> tensor<8x2xi32>
    // expected-remark@+2 {{unsigned : [5, 5] signed : [5, 5]}}
    // expected-remark@+1 {{non-neg}}
    %12 = tt.splat %c10_i32 : i32 -> tensor<8x2xi32>
    // expected-remark@+2 {{unsigned : [7, 15] signed : [7, 15]}}
    // expected-remark@+1 {{non-neg}}
    %13 = arith.addi %11, %12 : tensor<8x2xi32>
    // expected-remark@+2 {{unsigned : [0, 15] signed : [0, 15]}}
    // expected-remark@+1 {{non-neg}}
    %14 = arith.minsi %13, %5 : tensor<8x2xi32>
    // expected-remark@+4 {{result 0: unsigned : [2, 10] signed : [2, 10]}}
    // expected-remark@+3 {{result 1: unsigned : [2, 10] signed : [2, 10]}}
    // expected-remark@+2 {{result 0: non-neg}}
    // expected-remark@+1 {{result 1: non-neg}}
    %15, %16 = tt.split %11: tensor<8x2xi32> -> tensor<8xi32>
    %17 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<8x!tt.ptr<bf16>>
    %18 = tt.addptr %17, %15 : tensor<8x!tt.ptr<bf16>>, tensor<8xi32>
    %19 = tt.load %18 : tensor<8x!tt.ptr<bf16>>
    %20 = tt.addptr %17, %16 : tensor<8x!tt.ptr<bf16>>, tensor<8xi32>
    %21 = tt.load %20 : tensor<8x!tt.ptr<bf16>>
    %22 = arith.addf %19, %21 : tensor<8xbf16>
    %23 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<8x!tt.ptr<bf16>>
    %24 = tt.addptr %23, %7 : tensor<8x!tt.ptr<bf16>>, tensor<8xi32>
    tt.store %24, %22 : tensor<8x!tt.ptr<bf16>>
    tt.return
  }
}
