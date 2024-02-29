// RUN: mlir_fusions_opt %s -split-input-file -xla-gpu-lower-tensors | FileCheck %s

module {
  func.func private @add(%arg0: f32, %arg1: f32) -> f32 {
    %sum = arith.addf %arg0, %arg1 : f32
    func.return %sum : f32
  }

  func.func private @tensorarg(%arg0: tensor<43xf32> {xla.invariant, xla.slice_index = 0}, %arg1: index) -> f32 {
    %v1 = arith.constant 2.0 : f32
    %v2 = tensor.extract %arg0[%arg1] : tensor<43xf32>
    %sum = func.call @add(%v1, %v2) : (f32, f32) -> f32
    func.return %sum : f32
  }

  func.func @tensorcall(%arg0: tensor<43xf32> {xla.slice_index = 0}, %arg1: index) -> f32 {
    %call = func.call @tensorarg(%arg0, %arg1) : (tensor<43xf32>, index) -> f32
    func.return %call : f32
  }

  func.func @stores(%arg0: tensor<17xf32> {xla.slice_index = 0}, %arg1: tensor<43xf32> {xla.slice_index = 1}) -> tensor<43xf32> {
    %c17 = arith.constant 17 : index
    %c23 = arith.constant 23 : index
    %cst = arith.constant 3.0 : f32
    %out = tensor.insert %cst into %arg1[%c17] : tensor<43xf32>
    %out2 = tensor.insert %cst into %out[%c23] : tensor<43xf32>
    func.return %out2 : tensor<43xf32>
  }
}

// CHECK:        func.func private @add(%{{.*}}: f32, %{{.*}}: f32) -> f32 {
// CHECK-NEXT:     arith.addf
// CHECK-NEXT:     return

// CHECK:        func.func private @tensorarg(%[[ARG0:.*]]: !llvm.ptr
// CHECK-SAME:        {xla.invariant, xla.slice_index = 0 : i64}, %[[ARG1:.*]]: index) -> f32 {
// CHECK-DAG:       %[[C2:.*]] = arith.constant 2.000000e+00
// CHECK-DAG:       %[[IDX:.*]] = arith.index_castui %[[ARG1]] : index to i32
// CHECK-DAG:       %[[PTR:.*]] = llvm.getelementptr inbounds %[[ARG0]][%[[IDX]]]
// CHECK-DAG:       %[[V2:.*]] = llvm.load %[[PTR]] invariant
// CHECK:           %[[RET:.*]] = call @add(%[[C2]], %[[V2]])
// CHECK:           return %[[RET]]

// CHECK:        func.func @tensorcall(%[[ARG0:.*]]: !llvm.ptr
// CHECK-SAME:        {xla.slice_index = 0 : i64}, %[[ARG1:.*]]: index)
// CHECK:           %[[RET:.*]] = call @tensorarg(%[[ARG0]], %[[ARG1]])
// CHECK:           return %[[RET]]

// CHECK:        func.func @stores(
// CHECK-SAME:        %[[ARG0:.*]]: !llvm.ptr {xla.slice_index = 0 : i64},
// CHECK-SAME:        %[[ARG1:.*]]: !llvm.ptr {xla.slice_index = 1 : i64})
// CHECK-NEXT:      %[[CST:.*]] = arith.constant 3.000000e+00 : f32
// CHECK-NEXT:      %[[PTR1:.*]] = llvm.getelementptr inbounds %[[ARG1]][17]
// CHECK-NEXT:      llvm.store %[[CST]], %[[PTR1]]
// CHECK-NEXT:      %[[PTR2:.*]] = llvm.getelementptr inbounds %[[ARG1]][23]
// CHECK-NEXT:      llvm.store %[[CST]], %[[PTR2]]
// CHECK-NEXT:      return

// -----

module {
  func.func @layout(
      %arg0: tensor<2x3xf32, dense<[0, 1]> : tensor<2xi64>>,
      %arg1: index, %arg2: index) -> f32 {
    %v = tensor.extract %arg0[%arg1, %arg2]
        : tensor<2x3xf32, dense<[0, 1]> : tensor<2xi64>>
    func.return %v : f32
  }
}

// CHECK:      #[[MAP:.*]] = affine_map<(d0, d1) -> (d0 + d1 * 2)>
// CHECK:      @layout(%[[ARG0:.*]]: !llvm.ptr,
// CHECK-SAME:     %[[X:.*]]: index, %[[Y:.*]]: index
// CHECK:        %[[IDX:.*]] = affine.apply #[[MAP]](%[[X]], %[[Y]])
// CHECK:        %[[IDX_CAST:.*]] = arith.index_castui %[[IDX]] : index to i32
// CHECK:        %[[PTR:.*]] = llvm.getelementptr inbounds %[[ARG0]][%[[IDX_CAST]]]
// CHECK:        llvm.load %[[PTR]]

// -----

module {
  func.func @store_control_flow(
     %arg0: tensor<2xf32>,
     %arg1: index
  ) -> tensor<2xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cst = arith.constant 0.0 : f32
    %cst2 = arith.constant 1.0 : f32

    %for = scf.for %i = %c0 to %c2 step %c1 iter_args(%arg2 = %arg0) -> tensor<2xf32> {
      %new_out = tensor.insert %cst into %arg2[%i] : tensor<2xf32>
      scf.yield %new_out : tensor<2xf32>
    }

    %inbounds = arith.cmpi sle, %arg1, %c1 : index
    %result = scf.if %inbounds -> tensor<2xf32> {
      %if = tensor.insert %cst2 into %for[%arg1] : tensor<2xf32>
      scf.yield %if : tensor<2xf32>
    } else {
      scf.yield %for : tensor<2xf32>
    }
    func.return %result : tensor<2xf32>
  }
}

// CHECK:     @store_control_flow(%[[ARG0:.*]]: !llvm.ptr, %[[X:.*]]: index) {
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK:       scf.for %[[I:.*]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK:         %[[CAST:.*]] = arith.index_castui %[[I]] : index to i32
// CHECK:         %[[PTR:.*]] = llvm.getelementptr inbounds %[[ARG0]][%[[CAST]]]
// CHECK:         llvm.store {{.*}}, %[[PTR]]
// CHECK:       %[[INBOUNDS:.*]] = arith.cmpi
// CHECK:       scf.if %[[INBOUNDS]] {
// CHECK:         llvm.store
// CHECK-NEXT:  }
// CHECK-NEXT:  return

// -----

module {
  func.func @large_tensor(
      %arg0: tensor<1024x1024x1024x6xf32>,
      %arg1: index) -> f32 {
    %v = tensor.extract %arg0[%arg1, %arg1, %arg1, %arg1] : tensor<1024x1024x1024x6xf32>
    func.return %v : f32
  }
}

// CHECK: @large_tensor
// CHECK: arith.index_castui {{.*}} : index to i64

// -----

module {
  func.func @complex_tensor_insert(
      %arg0: tensor<10xcomplex<f32>>) -> tensor<10xcomplex<f32>> {
    %c1 = arith.constant 1 : index
    %real = arith.constant 3.0 : f32
    %imag = arith.constant 2.0 : f32
    %complex = complex.create %real, %imag : complex<f32>
    %out = tensor.insert %complex into %arg0[%c1] : tensor<10xcomplex<f32>>
    func.return %out : tensor<10xcomplex<f32>>
  }
}

// CHECK: @complex_tensor_insert(%[[ARG0:.*]]: !llvm.ptr
// CHECK: %[[C:.*]] = complex.create
// CHECK: %[[GEP:.*]] = llvm.getelementptr inbounds %[[ARG0]][1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(f32, f32)>
// CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[C]] : complex<f32> to !llvm.struct<(f32, f32)>
// CHECK: llvm.store %[[CAST]], %[[GEP]] : !llvm.struct<(f32, f32)>, !llvm.ptr

// -----

module {
  func.func @complex_tensor_extract(
      %arg0: tensor<10xcomplex<f32>>) -> complex<f32> {
    %c1 = arith.constant 1 : index
    %v2 = tensor.extract %arg0[%c1] : tensor<10xcomplex<f32>>
    func.return %v2 : complex<f32>
  }
}

// CHECK: @complex_tensor_extract(%[[ARG0:.*]]: !llvm.ptr
// CHECK: %[[GEP:.*]] = llvm.getelementptr inbounds %[[ARG0]][1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(f32, f32)>
// CHECK: %[[LOAD:.*]] = llvm.load %[[GEP]] : !llvm.ptr -> !llvm.struct<(f32, f32)>
// CHECK: builtin.unrealized_conversion_cast %[[LOAD]] : !llvm.struct<(f32, f32)> to complex<f32>

// -----

module {
  // This example is a bit silly, in real life there wouldn't be a loop (the
  // loop body would be executed by different threads). We're just doing it this
  // way so control flow with shared memory is tested as well.
  func.func @transpose_shared(%in: tensor<32x32xf32>,
                              %out: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index

    %shared = xla_gpu.allocate_shared : tensor<32x32xf32>
    %loaded_tile = scf.for %i = %c0 to %c32 step %c1
        iter_args(%tile = %shared) -> tensor<32x32xf32> {
      %inner_loaded_tile = scf.for %j = %c0 to %c32 step %c1
          iter_args(%inner_tile = %tile) -> tensor<32x32xf32> {
        %v = tensor.extract %in[%i, %j] : tensor<32x32xf32>
        %inserted = tensor.insert %v into %inner_tile[%i, %j]
            : tensor<32x32xf32>
        scf.yield %inserted : tensor<32x32xf32>
      }
      scf.yield %inner_loaded_tile : tensor<32x32xf32>
    }

    %synced = xla_gpu.sync_threads %shared : tensor<32x32xf32>
    %written_tile = scf.for %i = %c0 to %c32 step %c1
        iter_args(%written = %out) -> tensor<32x32xf32> {
      %inner_written_tile = scf.for %j = %c0 to %c32 step %c1
          iter_args(%inner_written = %written) -> tensor<32x32xf32> {
        %v = tensor.extract %shared[%j, %i] : tensor<32x32xf32>
        %inserted = tensor.insert %v into %inner_written[%i, %j]
            : tensor<32x32xf32>
        scf.yield %inserted : tensor<32x32xf32>
      }
      scf.yield %inner_written_tile : tensor<32x32xf32>
    }

    return %written_tile : tensor<32x32xf32>
  }
}

// CHECK:      llvm.mlir.global private @[[SHARED:shared_.*]]()
// CHECK-SAME:     {addr_space = 3 : i32} : !llvm.array<1024 x f32>
// CHECK:      @transpose_shared
// CHECK:        %[[ADDR:.*]] = llvm.mlir.addressof @[[SHARED]] : !llvm.ptr<3>
// CHECK:        %[[CAST:.*]] = llvm.addrspacecast %[[ADDR]]
// CHECK-SAME:       : !llvm.ptr<3> to !llvm.ptr
// CHECK:        scf.for
// CHECK:          scf.for
// CHECK:            %[[ELEM_ADDR:.*]] = llvm.getelementptr inbounds %[[CAST]]
// CHECK:            llvm.store {{.*}} %[[ELEM_ADDR]]
// CHECK:        gpu.barrier
// CHECK:        scf.for
// CHECK:          scf.for
// CHECK:            %[[ELEM_ADDR:.*]] = llvm.getelementptr inbounds %[[CAST]]
// CHECK:            llvm.load %[[ELEM_ADDR]]

