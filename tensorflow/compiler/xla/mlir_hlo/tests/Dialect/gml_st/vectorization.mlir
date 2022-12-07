// RUN: mlir-hlo-opt %s --vectorize-gml-st-loops --split-input-file |\
// RUN: FileCheck %s

// RUN: mlir-hlo-opt %s --split-input-file \
// RUN:     --vectorize-gml-st-loops="vectorize-gml-st-ops=true" \
// RUN: | FileCheck %s --check-prefix=VECTORIZE-LOOP-NO-LABEL

#map0 = affine_map<(d0) -> (d0)>
func.func @tiled_add(%A: tensor<8xf32>, %B: tensor<8xf32>,
                  %C: tensor<8xf32>) -> tensor<8xf32> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  %sum = gml_st.loop (%i) = (%c0) to (%c8) step (%c2)
       ins (%A_ = %A: tensor<8xf32>, %B_ = %B: tensor<8xf32>)
       outs (%C_ = %C: tensor<8xf32>) {
    %A_sub = tensor.extract_slice %A_[%i] [2] [1]
      : tensor<8xf32> to tensor<2xf32>
    %B_sub = tensor.extract_slice %B_[%i] [2] [1]
      : tensor<8xf32> to tensor<2xf32>
    %C_sub = tensor.extract_slice %C_[%i] [2] [1]
      : tensor<8xf32> to tensor<2xf32>
    %sum_sub = linalg.map
      ins(%A_sub, %B_sub : tensor<2xf32>, tensor<2xf32>)
      outs(%C_sub : tensor<2xf32>)
      (%a: f32, %b: f32) {
        %0 = arith.addf %a, %b : f32
        linalg.yield %0 : f32
      }
    %update = tensor.insert_slice %sum_sub into %C_[%i] [2] [1]
      : tensor<2xf32> into tensor<8xf32>
    gml_st.yield %update : tensor<8xf32>
  }
  func.return %sum : tensor<8xf32>
}
// CHECK-LABEL: func @tiled_add

// CHECK-DAG:  %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index

// CHECK: gml_st.loop (%[[IV:arg[0-9]]]) =
// CHECK-SAME: ins (%[[A:arg[0-9]]] = %{{arg[0-9]}}: tensor<8xf32>,
// CHECK-SAME:      %[[B:arg[0-9]]] = %{{arg[0-9]}}: tensor<8xf32>
// CHECK-SAME: outs (%[[C:arg[0-9]]] = %{{arg[0-9]}}: tensor<8xf32>)

// CHECK-NEXT: %[[LHS:.*]] = vector.transfer_read %[[A]][%[[IV]]], %[[CST]]
// CHECK-SAME:   {in_bounds = [true]} : tensor<8xf32>, vector<2xf32>
// CHECK-NEXT: %[[RHS:.*]] = vector.transfer_read %[[B]][%[[IV]]], %[[CST]]
// CHECK-SAME:   {in_bounds = [true]} : tensor<8xf32>, vector<2xf32>

// CHECK-NEXT: %[[SUM:.*]] = arith.addf %[[LHS]], %[[RHS]] : vector<2xf32>

// CHECK-NEXT: %{{.*}} = vector.transfer_write %[[SUM]], %[[C]][%[[IV]]]
// CHECK-SAME:   {in_bounds = [true]} : vector<2xf32>, tensor<8xf32>

// -----

func.func @tiled_reduction_2d(%in: tensor<80x60xf32>) -> tensor<80xf32> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c60 = arith.constant 60 : index
  %c80 = arith.constant 80 : index
  %cst = arith.constant 0.000000e+00 : f32

  %init = tensor.empty() : tensor<80xf32>
  %out = linalg.fill ins(%cst : f32) outs(%init : tensor<80xf32>) -> tensor<80xf32>

  %sum = gml_st.loop (%i, %j) = (%c0, %c0) to (%c80, %c60) step (%c4, %c4)
          ins (%in_ = %in: tensor<80x60xf32>, %cst_ = %cst: f32)
          outs (%out_ = %out: tensor<80xf32>)
          iterators[#gml_st.iterator_type<parallel>,
                    #gml_st.iterator_type<reduction>] {
    %in_sub = tensor.extract_slice %in_[%i, %j] [4, 4] [1, 1]
        : tensor<80x60xf32> to tensor<4x4xf32>
    %out_sub = tensor.extract_slice %out_[%i] [4] [1]
        : tensor<80xf32> to tensor<4xf32>
    %local_fill = linalg.fill ins(%cst_ : f32) outs(%out_sub : tensor<4xf32>) -> tensor<4xf32>
    %reduced_tile = linalg.reduce
        ins(%in_sub : tensor<4x4xf32>)
        outs(%local_fill : tensor<4xf32>)
        dimensions = [0]
      (%a: f32, %b: f32) {
        %0 = arith.addf %a, %b : f32
        linalg.yield %0 : f32
      }
    %expand = tensor.expand_shape %reduced_tile [[0, 1]]
      : tensor<4xf32> into tensor<1x4xf32>
    %acc = linalg.reduce
          ins(%expand : tensor<1x4xf32>)
          outs(%out_sub : tensor<4xf32>)
          dimensions = [0]
     (%arg4: f32, %arg5: f32) {
      %9 = arith.addf %arg4, %arg5 : f32
      linalg.yield %9 : f32
    }

    %update = tensor.insert_slice %acc into %out_[%i] [4] [1]
        : tensor<4xf32> into tensor<80xf32>
    gml_st.yield %update : tensor<80xf32>
  }
  func.return %sum : tensor<80xf32>
}

// CHECK-LABEL: func @tiled_reduction_2d

// CHECK: gml_st.loop
// CHECK-SAME: ins (%{{arg[0-9]}} = %{{arg[0-9]}}: tensor<80x60xf32>,
// CHECK-SAME:      %[[CST:arg[0-9]]] = %{{.*}}: f32

// CHECK: %[[BCAST:.*]] = vector.broadcast %[[CST]] : f32 to vector<4xf32>
// CHECK-NOT: vector.transfer_write %[[BCAST]]
// CHECK: vector.multi_reduction <add>, %{{.*}}, %[[BCAST]] [0] : vector<4x4xf32> to vector<4xf32>

// -----

func.func @reduction_1d(%arg0: tensor<16xf32>) -> tensor<f32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %0 = tensor.empty() : tensor<f32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<f32>) -> tensor<f32>
  %2 = tensor.empty() : tensor<8xf32>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<8xf32>) -> tensor<8xf32>
  %4 = gml_st.loop (%arg1) = (%c0) to (%c16) step (%c8)
      ins (%arg2 = %arg0: tensor<16xf32>)
      outs (%arg3 = %3: tensor<8xf32>)
      iterators[#gml_st.iterator_type<reduction>] {
    %6 = tensor.extract_slice %arg2[%arg1] [8] [1]
      : tensor<16xf32> to tensor<8xf32>
    %7 = tensor.expand_shape %6 [[0, 1]]
      : tensor<8xf32> into tensor<1x8xf32>
    %8 = linalg.reduce
          ins(%7 : tensor<1x8xf32>)
          outs(%arg3 : tensor<8xf32>)
          dimensions = [0]
     (%arg4: f32, %arg5: f32) {
      %9 = arith.addf %arg4, %arg5 : f32
      linalg.yield %9 : f32
    }
    gml_st.yield %8 : tensor<8xf32>
  }
  %5 = linalg.reduce
    ins(%4 : tensor<8xf32>)
    outs(%1 : tensor<f32>)
    dimensions = [0]
    (%arg1: f32, %arg2: f32) {
    %6 = arith.addf %arg1, %arg2 : f32
    linalg.yield %6 : f32
  }
  func.return %5 : tensor<f32>
}
// CHECK-LABEL: func @reduction_1d

// CHECK: gml_st.loop
// CHECK-SAME: ins (%[[IN:arg[0-9]]] = %{{arg[0-9]}}: tensor<16xf32>)

// CHECK: %[[VECTOR:.*]] = vector.transfer_read %[[IN]]
// CHECK: vector.shape_cast %[[VECTOR]] : vector<8xf32> to vector<1x8xf32>
// CHECK-NOT: tensor.expand_shape
// CHECK: vector.multi_reduction

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
func.func @test_transfer_read_of_one_dim_expand_shape(
    %in: tensor<10xf32>) -> tensor<5xf32> {
  %c0 = arith.constant 0 : index
  %min_float = arith.constant dense<-3.402820e+38> : vector<5xf32>
  %zero_float = arith.constant 0.000000e+00 : f32
  %0 = tensor.expand_shape %in [[0, 1]] : tensor<10xf32> into tensor<2x5xf32>
  %1 = tensor.empty() : tensor<5xf32>
  %2 = vector.transfer_read %0[%c0, %c0], %zero_float
    {in_bounds = [true, true], permutation_map = #map0}
    : tensor<2x5xf32>, vector<2x5xf32>
  %3 = vector.multi_reduction <maxf>, %2, %min_float [0]
    : vector<2x5xf32> to vector<5xf32>
  %4 = vector.transfer_write %3, %1[%c0] {in_bounds = [true]}
    : vector<5xf32>, tensor<5xf32>
  func.return %4 : tensor<5xf32>
}
// CHECK-LABEL: func @test_transfer_read_of_one_dim_expand_shape(
// CHECK-SAME: %[[IN:.*]]: tensor<10xf32>
// CHECK-DAG: %[[MIN_FLOAT:.*]] = arith.constant dense<-3.402820e+38> : vector<5xf32>
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[ZERO_FLOAT:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[INIT_TENSOR:.*]] = tensor.empty() : tensor<5xf32>
// CHECK: %[[TRANSFER_READ:.*]] = vector.transfer_read %[[IN]][%[[C0]]], %[[ZERO_FLOAT]] {in_bounds = [true]} : tensor<10xf32>, vector<10xf32>
// CHECK: %[[SHAPE_CAST:.*]] = vector.shape_cast %[[TRANSFER_READ]] : vector<10xf32> to vector<2x5xf32>
// CHECK: %[[MULTI_REDUCTION:.*]] = vector.multi_reduction <maxf>, %[[SHAPE_CAST]], %[[MIN_FLOAT]] [0] : vector<2x5xf32> to vector<5xf32>
// CHECK: %[[TRANSFER_WRITE:.*]] = vector.transfer_write %[[MULTI_REDUCTION]], %[[INIT_TENSOR]][%[[C0]]] {in_bounds = [true]} : vector<5xf32>, tensor<5xf32>
// CHECK: return %[[TRANSFER_WRITE]] : tensor<5xf32>

// -----

#map0 = affine_map<(d0, d1) -> (d0, 0)>
func.func @test_transfer_read_of_one_dim_expand_shape_different_shape(
    %in: tensor<1xf32>) -> tensor<18xf32> {
  %c0 = arith.constant 0 : index
  %min_float = arith.constant dense<-3.402820e+38> : vector<18xf32>
  %zero_float = arith.constant 0.000000e+00 : f32
  %0 = tensor.expand_shape %in [[0, 1]] : tensor<1xf32> into tensor<1x1xf32>
  %1 = tensor.empty() : tensor<18xf32>
  %2 = vector.transfer_read %0[%c0, %c0], %zero_float
    {in_bounds = [true, true], permutation_map = #map0}
    : tensor<1x1xf32>, vector<1x18xf32>
  %3 = vector.multi_reduction <maxf>, %2, %min_float [0]
    : vector<1x18xf32> to vector<18xf32>
  %4 = vector.transfer_write %3, %1[%c0] {in_bounds = [true]}
    : vector<18xf32>, tensor<18xf32>
  func.return %4 : tensor<18xf32>
}
// CHECK-LABEL: func @test_transfer_read_of_one_dim_expand_shape_different_shape
// CHECK: %{{.*}} = tensor.expand_shape

// -----

func.func @do_not_vectorize_large_untiled_fill() -> tensor<2x1000xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<2x1000xf32>
  %out = linalg.fill ins(%cst : f32) outs(%init : tensor<2x1000xf32>) -> tensor<2x1000xf32>
  func.return %out : tensor<2x1000xf32>
}
// CHECK-LABEL: func @do_not_vectorize_large_untiled_fill
// CHECK: linalg.fill

// -----

func.func @vectorize_small_untiled_fill() -> tensor<128xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<128xf32>
  %out = linalg.fill ins(%cst : f32) outs(%init : tensor<128xf32>) -> tensor<128xf32>
  func.return %out : tensor<128xf32>
}
// CHECK-LABEL: func @vectorize_small_untiled_fill
// CHECK: vector.transfer_write

// -----

func.func @do_not_vectorize_materialize_outside_loop() -> tensor<8x1xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c0 = arith.constant 0 : index
  %0 = tensor.empty() : tensor<10x1xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<10x1xf32>) -> tensor<10x1xf32>
  %6 = gml_st.tile [0, 0] [8, 1] [1, 1] : !gml_st.tile<8x1>
  %3 = gml_st.materialize %1[%6] : tensor<10x1xf32>[!gml_st.tile<8x1>] to tensor<8x1xf32>
  %4 = gml_st.loop (%arg2, %arg3) = (%c0, %c0) to (%c8, %c1) step (%c1, %c8) ins (%arg4 = %cst: f32) outs (%arg5 = %3: tensor<8x1xf32>) {
    %10 = affine.min affine_map<(d0) -> (-d0 + 1, 8)>(%arg3)
    %extracted_slice = tensor.extract_slice %arg5[%arg2, %arg3] [1, %10] [1, 1] : tensor<8x1xf32> to tensor<1x?xf32>
    %11 = linalg.fill ins(%arg4 : f32) outs(%extracted_slice : tensor<1x?xf32>) -> tensor<1x?xf32>
    %inserted_slice_1 = tensor.insert_slice %11 into %arg5[%arg2, %arg3] [1, %10] [1, 1] : tensor<1x?xf32> into tensor<8x1xf32>
    gml_st.yield %inserted_slice_1 : tensor<8x1xf32>
  }
  return %4 : tensor<8x1xf32>
}
// VECTORIZE-LOOP-NO-LABEL-LABEL: func @do_not_vectorize_materialize_outside_loop
// VECTORIZE-LOOP-NO-LABEL:         %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<10x1xf32>
// VECTORIZE-LOOP-NO-LABEL:         %[[INIT:.*]] = tensor.empty() : tensor<10x1xf32>
// VECTORIZE-LOOP-NO-LABEL:         %[[WRITE:.*]] = vector.transfer_write %[[CST]], %[[INIT]]{{.*}} tensor<10x1xf32>
// VECTORIZE-LOOP-NO-LABEL:         %[[TILE:.*]] = gml_st.tile [0, 0] [8, 1] [1, 1]
// VECTORIZE-LOOP-NO-LABEL:         gml_st.materialize %[[WRITE]][%[[TILE]]] : {{.*}} to tensor<8x1xf32>
