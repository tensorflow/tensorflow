// RUN: mlir-hlo-opt %s --split-input-file --allow-unregistered-dialect | \
// RUN: mlir-hlo-opt --verify-diagnostics --split-input-file \
// RUN:     --allow-unregistered-dialect | \
// RUN: FileCheck %s

// CHECK-LABEL: @types
func.func @types() {
  // CHECK: %{{.*}} = gml_st.tile [0, 0] [42, 16] [1, 1] : !gml_st.tile<42x16>
  %3 = gml_st.tile [0, 0] [42, 16] [1, 1] : !gml_st.tile<42x16>
  func.return
}

// -----

// CHECK-LABEL: @dynamic_types
// CHECK-SAME: (%[[SIZE:.*]]: index)
func.func @dynamic_types(%size : index) {
  // CHECK: %{{.*}} = gml_st.tile [0, 0] [42, 8] [1, 1] : !gml_st.tile<42x8>
  %3 = gml_st.tile [0, 0] [42, 8] [1, 1] : !gml_st.tile<42x8>
  func.return
}

// -----

// CHECK-LABEL: @materialize_vector
// CHECK-SAME: %[[VECTOR:.*]]: vector<64x32xf32>
func.func @materialize_vector(%vector: vector<64x32xf32>) {
  // CHECK: %{{.*}} = gml_st.materialize %[[VECTOR]]
  // CHECK-SAME: : vector<64x32xf32>
  %0 = gml_st.materialize %vector[0, 0][42, 16][1, 1]
    : vector<64x32xf32> to vector<42x16xf32>
  func.return
}

// -----

// CHECK-LABEL: @materialize_0d_vector
// CHECK-SAME: %[[VECTOR:.*]]: vector<f32>
func.func @materialize_0d_vector(%vector: vector<f32>) {
  // CHECK: %{{.*}} = gml_st.materialize %[[VECTOR]]
  // CHECK-SAME: : vector<f32> to vector<f32>
  %0 = gml_st.materialize %vector[][][]
    : vector<f32> to vector<f32>
  func.return
}

// -----

// CHECK-LABEL: @distribute_vector
// CHECK-SAME: %[[VECTOR:.*]]: vector<42x16xf32>,
// CHECK-SAME: %[[TILE:.*]]: !gml_st.tile<42x16>
func.func @distribute_vector(%vector: vector<42x16xf32>,
                              %tile: !gml_st.tile<42x16>) {
  // CHECK: %{{.*}} = gml_st.distribute %[[VECTOR]] into[%[[TILE]]]
  // CHECK-SAME: : vector<42x16xf32> into vector<64x32xf32>[!gml_st.tile<42x16>]
  %0 = gml_st.distribute %vector into[%tile]
    : vector<42x16xf32> into vector<64x32xf32>[!gml_st.tile<42x16>]
  func.return
}

// -----

// CHECK-LABEL: @distribute_0d_vector
// CHECK-SAME: %[[VECTOR:.*]]: vector<f32>,
// CHECK-SAME: %[[TILE:.*]]: !gml_st.tile<>
func.func @distribute_0d_vector(%vector: vector<f32>, %tile: !gml_st.tile<>) {
  // CHECK: %{{.*}} = gml_st.distribute %[[VECTOR]] into[%[[TILE]]]
  // CHECK-SAME: : vector<f32> into vector<f32>[!gml_st.tile<>]
  %0 = gml_st.distribute %vector into[%tile]
    : vector<f32> into vector<f32>[!gml_st.tile<>]
  func.return
}

// -----

#id_1d = affine_map<(d0) -> (d0)>

func.func @parallel_loop(%lhs: tensor<8xf32>, %rhs: tensor<8xf32>,
                         %output: tensor<8xf32>) -> tensor<8xf32> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index

  %sum = gml_st.parallel (%i) = (%c0) to (%c8) step (%c4)
      outs(%out_ = %output : tensor<8xf32>) {
    %lhs_sub = tensor.extract_slice %lhs[%i] [4] [1]
      : tensor<8xf32> to tensor<4xf32>
    %rhs_sub = tensor.extract_slice %rhs[%i] [4] [1]
      : tensor<8xf32> to tensor<4xf32>
    %out_sub = tensor.extract_slice %output[%i] [4] [1]
      : tensor<8xf32> to tensor<4xf32>

    %result_sub = linalg.generic {
        indexing_maps = [#id_1d, #id_1d, #id_1d],
        iterator_types = ["parallel"]}
        ins(%lhs_sub, %rhs_sub : tensor<4xf32>, tensor<4xf32>)
        outs(%out_sub : tensor<4xf32>) {
      ^bb(%l: f32, %r: f32, %o: f32) :
        %s = arith.addf %l, %r : f32
        linalg.yield %s : f32
    } -> tensor<4xf32>

    %tile = gml_st.tile [%i] [4] [1] : !gml_st.tile<4>
    gml_st.set_yield %result_sub into %out_[%tile]
      : tensor<4xf32> into tensor<8xf32>[!gml_st.tile<4>]
  } : tensor<8xf32>
  func.return %sum : tensor<8xf32>
}
// CHECK-LABEL: func @parallel_loop

// -----

func.func @loop_on_points(%output: tensor<8xf32>) -> tensor<8xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c0_f32 = arith.constant 0.0 : f32

  %sum = gml_st.parallel (%i) = (%c0) to (%c8) step (%c1)
      outs(%out_ = %output : tensor<8xf32>) {
    %tile = gml_st.tile [%i] [1] [1] : !gml_st.tile<1>
    gml_st.set_yield %c0_f32 into %out_[%tile]
      : f32 into tensor<8xf32>[!gml_st.tile<1>]
  } : tensor<8xf32>
  func.return %sum : tensor<8xf32>
}
// CHECK-LABEL: func @loop_on_points

// -----
func.func @parallel_with_distribution(%output: tensor<8xf32>) -> tensor<8xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c0_f32 = arith.constant 0.0 : f32

  %sum = gml_st.parallel (%i) = (%c0) to (%c8) step (%c1)
      outs(%out_ = %output : tensor<8xf32>) distribution ("x") {
    %tile = gml_st.tile [%i] [1] [1] : !gml_st.tile<1>
    gml_st.set_yield %c0_f32 into %out_[%tile]
      : f32 into tensor<8xf32>[!gml_st.tile<1>]
  } : tensor<8xf32>
  func.return %sum : tensor<8xf32>
}
// CHECK-LABEL: func @parallel_with_distribution
// CHECK: gml_st.parallel {{.*}} distribution ("x")

// -----

func.func @loop_on_vector(%output: vector<8xf32>, %fill: vector<2xf32>)
    -> vector<8xf32> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index

  %sum = gml_st.parallel (%i) = (%c0) to (%c8) step (%c2) 
      outs(%out_ = %output : vector<8xf32>)  {
    %tile = gml_st.tile [%i] [2] [1] : !gml_st.tile<2>
    gml_st.set_yield %fill into %out_[%tile]
      : vector<2xf32> into vector<8xf32>[!gml_st.tile<2>]
  } : vector<8xf32>
  func.return %sum : vector<8xf32>
}
// CHECK-LABEL: func @loop_on_vector

// -----

#id_1d = affine_map<(d0) -> (d0)>

func.func @for_loop(%lhs: tensor<8xf32>, %rhs: tensor<8xf32>,
                    %output: tensor<8xf32>, %output2: tensor<8xf32>)
		    -> (tensor<8xf32>, tensor<8xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index

  %sum, %sum2 = gml_st.for (%i) = (%c0) to (%c8) step (%c4)
      outs(%out_ = %output : tensor<8xf32>, %out2_ = %output2 : tensor<8xf32>) {
    %lhs_sub = tensor.extract_slice %lhs [%i] [4] [1]
      : tensor<8xf32> to tensor<4xf32>
    %rhs_sub = tensor.extract_slice %rhs [%i] [4] [1]
      : tensor<8xf32> to tensor<4xf32>
    %out_sub = tensor.extract_slice %out_ [%i] [4] [1]
      : tensor<8xf32> to tensor<4xf32>
    %out2_sub = tensor.extract_slice %out_ [%i] [4] [1]
      : tensor<8xf32> to tensor<4xf32>

    %result_sub = linalg.generic {
        indexing_maps = [#id_1d, #id_1d, #id_1d],
        iterator_types = ["parallel"]}
        ins(%lhs_sub, %rhs_sub : tensor<4xf32>, tensor<4xf32>)
        outs(%out_sub : tensor<4xf32>) {
      ^bb(%l: f32, %r: f32, %o: f32) :
        %s = arith.addf %l, %r : f32
        linalg.yield %s : f32
    } -> tensor<4xf32>

    %tile = gml_st.tile [%i] [4] [1] : !gml_st.tile<4>
    gml_st.set_yield %result_sub into %out_[%tile]
      : tensor<4xf32> into tensor<8xf32>[!gml_st.tile<4>],
      %result_sub into %out2_[%tile]
      : tensor<4xf32> into tensor<8xf32>[!gml_st.tile<4>]
  } : tensor<8xf32>, tensor<8xf32>
  func.return %sum, %sum2 : tensor<8xf32>, tensor<8xf32>
}
// CHECK-LABEL: func @for_loop

// -----

#id_1d = affine_map<(d0) -> (d0)>

func.func @trivial_acc_region(%lhs: tensor<8xf32>,
    %rhs: tensor<8xf32>, %output: tensor<8xf32>) -> tensor<8xf32> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index

  %sum = gml_st.parallel (%i) = (%c0) to (%c8) step (%c4)
      outs(%out_ = %output : tensor<8xf32>)  {
    %lhs_sub = tensor.extract_slice %lhs [%i] [4] [1]
      : tensor<8xf32> to tensor<4xf32>
    %rhs_sub = tensor.extract_slice %rhs[%i] [4] [1]
      : tensor<8xf32> to tensor<4xf32>
    %out_sub = tensor.extract_slice %output[%i] [4] [1]
      : tensor<8xf32> to tensor<4xf32>

    %result_sub = linalg.generic {
        indexing_maps = [#id_1d, #id_1d, #id_1d],
        iterator_types = ["parallel"]}
        ins(%lhs_sub, %rhs_sub : tensor<4xf32>, tensor<4xf32>)
        outs(%out_sub : tensor<4xf32>) {
      ^bb(%l: f32, %r: f32, %o: f32) :
        %s = arith.addf %l, %r : f32
        linalg.yield %s : f32
    } -> tensor<4xf32>

    %tile = gml_st.tile [%i] [4] [1] : !gml_st.tile<4>
    gml_st.set_yield %result_sub into %out_[%tile]
      acc (%new, %old: tensor<4xf32>) {
        gml_st.yield %new : tensor<4xf32>
      } : tensor<4xf32> into tensor<8xf32>[!gml_st.tile<4>]
  } : tensor<8xf32>
  func.return %sum : tensor<8xf32>
}

// CHECK-LABEL: func @trivial_acc_region
// CHECK:       gml_st.set_yield %{{.*}} into %{{.*}}[%{{.*}}]
// CHECK-SAME:    acc (%[[NEW:.*]], %{{.*}}: tensor<4xf32>) {
// CHECK:           gml_st.yield %[[NEW]] : tensor<4xf32>
// CHECK:       } : tensor<4xf32> into tensor<8xf32>[!gml_st.tile<4>]

// -----

#id_1d = affine_map<(d0) -> (d0)>

func.func @two_acc_region(%lhs: tensor<8xf32>, %rhs: tensor<8xf32>,
    %output: tensor<8xf32>, %output_2: tensor<8xf32>) -> tensor<8xf32> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index

  %result:2 = gml_st.parallel (%i) = (%c0) to (%c8) step (%c4)
      outs(%out_ = %output : tensor<8xf32>, %out2_ = %output_2 : tensor<8xf32>) {
    %lhs_sub = tensor.extract_slice %lhs [%i] [4] [1]
      : tensor<8xf32> to tensor<4xf32>
    %rhs_sub = tensor.extract_slice %rhs [%i] [4] [1]
      : tensor<8xf32> to tensor<4xf32>
    %out_sub = tensor.extract_slice %output [%i] [4] [1]
      : tensor<8xf32> to tensor<4xf32>
    %out_2_sub = tensor.extract_slice %output_2 [%i] [4] [1]
      : tensor<8xf32> to tensor<4xf32>

    %result_sub = linalg.generic {
        indexing_maps = [#id_1d, #id_1d, #id_1d],
        iterator_types = ["parallel"]}
        ins(%lhs_sub, %rhs_sub : tensor<4xf32>, tensor<4xf32>)
        outs(%out_sub : tensor<4xf32>) {
      ^bb(%l: f32, %r: f32, %o: f32) :
        %s = arith.addf %l, %r : f32
        linalg.yield %s : f32
    } -> tensor<4xf32>

    %tile = gml_st.tile [%i] [4] [1] : !gml_st.tile<4>
    gml_st.set_yield
      %result_sub into %out_[%tile] acc (%new, %old: tensor<4xf32>) {
        gml_st.yield %new : tensor<4xf32>
      } : tensor<4xf32> into tensor<8xf32>[!gml_st.tile<4>],
      %result_sub into %out2_[%tile] acc (%new, %old: tensor<4xf32>) {
        %sum = linalg.generic {
           indexing_maps = [#id_1d, #id_1d],
           iterator_types = ["parallel"]}
           ins(%new: tensor<4xf32>)
          outs(%old : tensor<4xf32>) {
        ^bb(%n: f32, %o: f32) :
          %s = arith.addf %n, %o : f32
          linalg.yield %s : f32
        } -> tensor<4xf32>
        gml_st.yield %sum : tensor<4xf32>
      } : tensor<4xf32> into tensor<8xf32>[!gml_st.tile<4>]

  } : tensor<8xf32>, tensor<8xf32>
  func.return %result#0 : tensor<8xf32>
}

// CHECK-LABEL: func @two_acc_region
// CHECK:       gml_st.set_yield %[[RES:.*]] into %{{.*}}[%{{.*}}]
// CHECK-SAME:    acc (%[[NEW:.*]], %{{.*}}: tensor<4xf32>) {
// CHECK-NEXT:           gml_st.yield %[[NEW]] : tensor<4xf32>

// CHECK:  %[[RES]] into %{{.*}}[%{{.*}}]
// CHECK-SAME:    acc (%[[NEW:.*]], %{{.*}}: tensor<4xf32>) {
// CHECK-NEXT:           linalg.generic


// -----

#id_1d = affine_map<(d0) -> (d0)>

func.func @accumulator_region(%lhs: tensor<8xf32>,
    %rhs: tensor<8xf32>, %output: tensor<8xf32>) -> tensor<8xf32> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index

  %sum = gml_st.parallel (%i) = (%c0) to (%c8) step (%c4)
      outs(%out_ = %output : tensor<8xf32>) {
    %lhs_sub = tensor.extract_slice %lhs [%i] [4] [1]
      : tensor<8xf32> to tensor<4xf32>
    %rhs_sub = tensor.extract_slice %rhs [%i] [4] [1]
      : tensor<8xf32> to tensor<4xf32>
    %out_sub = tensor.extract_slice %output [%i] [4] [1]
      : tensor<8xf32> to tensor<4xf32>

    %result_sub = linalg.generic {
        indexing_maps = [#id_1d, #id_1d, #id_1d],
        iterator_types = ["parallel"]}
        ins(%lhs_sub, %rhs_sub : tensor<4xf32>, tensor<4xf32>)
        outs(%out_sub : tensor<4xf32>) {
      ^bb(%l: f32, %r: f32, %o: f32) :
        %s = arith.addf %l, %r : f32
        linalg.yield %s : f32
    } -> tensor<4xf32>

    %tile = gml_st.tile [%i] [4] [1] : !gml_st.tile<4>
    gml_st.set_yield %result_sub into %out_[%tile]
      acc (%new, %old: tensor<4xf32>) {
        %sum = linalg.generic {
           indexing_maps = [#id_1d, #id_1d],
           iterator_types = ["parallel"]}
           ins(%new: tensor<4xf32>)
          outs(%old : tensor<4xf32>) {
        ^bb(%n: f32, %o: f32) :
          %s = arith.addf %n, %o : f32
          linalg.yield %s : f32
        } -> tensor<4xf32>
        gml_st.yield %sum : tensor<4xf32>
      } : tensor<4xf32> into tensor<8xf32>[!gml_st.tile<4>]
  } : tensor<8xf32>
  func.return %sum : tensor<8xf32>
}
// CHECK-LABEL: func @accumulator_region
// CHECK:       gml_st.set_yield %{{.*}} into %{{.*}}[%{{.*}}]
// CHECK-SAME:    acc (%{{.*}}, %{{.*}}: tensor<4xf32>) {
// CHECK:           %[[SUM:.*]] = linalg.generic
// CHECK:           gml_st.yield %[[SUM]] : tensor<4xf32>
// CHECK:       } : tensor<4xf32> into tensor<8xf32>[!gml_st.tile<4>]

// -----

#map_0d = affine_map<(d0) -> ()>
#id_1d = affine_map<(d0) -> (d0)>

func.func @reduce_tiles(%arg: tensor<8xf32>,
                        %output: tensor<f32>) -> tensor<f32> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c0_f32 = arith.constant 0.0 : f32

  %sum = gml_st.parallel (%i) = (%c0) to (%c8) step (%c4)
      outs(%out_ = %output : tensor<f32>) {
    %arg_sub = tensor.extract_slice %arg [%i] [4] [1]
      : tensor<8xf32> to tensor<4xf32>

    %local_init = tensor.empty() : tensor<f32>
    %local_fill = linalg.fill
      ins(%c0_f32: f32)
      outs(%local_init: tensor<f32>) -> tensor<f32>

    %result_sub = linalg.generic {
        indexing_maps = [#id_1d, #map_0d],
        iterator_types = ["reduction"]}
        ins(%arg_sub: tensor<4xf32>)
        outs(%local_fill : tensor<f32>) {
      ^bb(%a: f32, %o: f32) :
        %s = arith.addf %a, %o : f32
        linalg.yield %s : f32
    } -> tensor<f32>

    %init_tile = gml_st.tile [] [] [] : !gml_st.tile<>
    gml_st.set_yield %result_sub into %out_[%init_tile]
        acc (%in, %out: tensor<f32>) {
      %in_pt = tensor.extract %in[] : tensor<f32>
      %out_pt = tensor.extract %out[] : tensor<f32>
      %sum_pt = arith.addf %in_pt, %out_pt : f32
      %sum = tensor.from_elements %sum_pt : tensor<f32>
      gml_st.yield %sum : tensor<f32>
    } : tensor<f32> into tensor<f32>[!gml_st.tile<>]
  } : tensor<f32>
  func.return %sum : tensor<f32>
}
// CHECK-LABEL: func @reduce_tiles
// CHECK:       gml_st.set_yield
// CHECK-SAME:    acc (%{{.*}}, %{{.*}}: tensor<f32>)
// CHECK:       } : tensor<f32> into tensor<f32>[!gml_st.tile<>]

// -----

#id_1d = affine_map<(d0) -> (d0)>
#id_2d = affine_map<(d0, d1) -> (d0, d1)>
#map_1d = affine_map<(d0, d1) -> (d1)>

func.func @column_reduction(%arg: tensor<128x16xf32>,
                            %out: tensor<16xf32>) -> tensor<16xf32> {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c128 = arith.constant 128 : index
  %cst = arith.constant 0.000000e+00 : f32

  %sum = gml_st.parallel (%i, %j) = (%c0, %c0) to (%c128, %c16) step (%c8, %c8)
      outs(%out_ = %out : tensor<16xf32>) {
    %arg_sub = tensor.extract_slice %arg[%i, %j] [8, 8] [1, 1]
      : tensor<128x16xf32> to tensor<8x8xf32>

    %init = tensor.empty() : tensor<8xf32>
    %fill = linalg.fill ins(%cst : f32)
                        outs(%init : tensor<8xf32>) -> tensor<8xf32>

    %result_sub = linalg.generic {
        indexing_maps = [#id_2d, #map_1d],
        iterator_types = ["reduction", "parallel"]}
        ins(%arg_sub: tensor<8x8xf32>)
        outs(%fill : tensor<8xf32>) {
      ^bb(%a: f32, %o: f32) :
        %s = arith.addf %a, %o : f32
        linalg.yield %s : f32
    } -> tensor<8xf32>

    %out_tile = gml_st.tile [%j] [8] [1] : !gml_st.tile<8>
    gml_st.set_yield %result_sub into %out_[%out_tile]
        acc (%new, %old: tensor<8xf32>) {
      %acc = linalg.generic {
          indexing_maps = [#id_1d, #id_1d],
          iterator_types = ["parallel"]}
          ins(%new: tensor<8xf32>)
          outs(%old : tensor<8xf32>) {
        ^bb(%n: f32, %o: f32) :
          %s = arith.addf %n, %o : f32
          linalg.yield %s : f32
      } -> tensor<8xf32>
      gml_st.yield %acc : tensor<8xf32>
    } : tensor<8xf32> into tensor<16xf32>[!gml_st.tile<8>]
  } : tensor<16xf32>
  func.return %sum : tensor<16xf32>
}
// CHECK-LABEL: func @column_reduction
// CHECK:       gml_st.set_yield
// CHECK-SAME:    acc (%{{.*}}, %{{.*}}: tensor<8xf32>)
// CHECK:       } : tensor<8xf32> into tensor<16xf32>[!gml_st.tile<8>]

// -----

#id_1d = affine_map<(d0) -> (d0)>
#id_2d = affine_map<(d0, d1) -> (d0, d1)>
#map_1d = affine_map<(d0, d1) -> (d1)>

func.func @sequential_column_reduction(%arg: tensor<128x16xf32>,
                                       %out: tensor<16xf32>) -> tensor<16xf32> {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c128 = arith.constant 128 : index

  %sum = gml_st.for (%i, %j) = (%c0, %c0) to (%c128, %c16) step (%c8, %c8)
      outs(%out_ = %out : tensor<16xf32>) {
    %arg_sub = tensor.extract_slice %arg[%i, %j] [8, 8] [1, 1]
      : tensor<128x16xf32> to tensor<8x8xf32>

    %out_sub = tensor.extract_slice %out_[%j] [8] [1]
      : tensor<16xf32> to tensor<8xf32>

    %result_sub = linalg.generic {
        indexing_maps = [#id_2d, #map_1d],
        iterator_types = ["reduction", "parallel"]}
        ins(%arg_sub: tensor<8x8xf32>)
        outs(%out_sub : tensor<8xf32>) {
      ^bb(%a: f32, %o: f32) :
        %s = arith.addf %a, %o : f32
        linalg.yield %s : f32
    } -> tensor<8xf32>

    %out_tile = gml_st.tile [%j] [8] [1] : !gml_st.tile<8>
    gml_st.set_yield %result_sub into %out_[%out_tile]
      : tensor<8xf32> into tensor<16xf32>[!gml_st.tile<8>]
  } : tensor<16xf32>
  func.return %sum : tensor<16xf32>
}
// CHECK-LABEL: func @sequential_column_reduction
// CHECK:       gml_st.set_yield

// -----

func.func @fusion_cluster(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>,
    %init: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = gml_st.fusion (%a0 = %arg0 : tensor<?x?xf32>,
                      %a1 = %arg1 : tensor<?x?xf32>,
                      %in = %init : tensor<?x?xf32>) {
    %map0 = linalg.map { math.exp }
      ins(%a0 : tensor<?x?xf32>)
      outs(%in : tensor<?x?xf32>)
    %map1 = linalg.map { arith.mulf }
      ins(%map0, %a1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%in : tensor<?x?xf32>)
    gml_st.yield %map1 : tensor<?x?xf32>
  } { "some_attr" = 1 } : tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: func @fusion_cluster
// CHECK:       gml_st.fusion
// CHECK:         linalg.map
// CHECK:         linalg.map
// CHECK:         gml_st.yield
