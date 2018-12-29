// RUN: mlir-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK: #map0 = (d0) -> (d0 + 1)

// CHECK: #map1 = (d0, d1) -> (d0 + 1, d1 + 2)
#map5 = (d0, d1) -> (d0 + 1, d1 + 2)

// CHECK: #map2 = (d0, d1)[s0, s1] -> (d0 + s1, d1 + s0)
// CHECK: #map3 = ()[s0] -> (s0 + 1)
// CHECK-DAG: #[[map_proj_d0d1_d0:map[0-9]+]] = (d0, d1) -> (d0)
// CHECK-DAG: #[[map_proj_d0d1_d1:map[0-9]+]] = (d0, d1) -> (d1)
// CHECK-DAG: #[[map_proj_d0d1_d1d0:map[0-9]+]] = (d0, d1) -> (d1, d0)

// CHECK-LABEL: cfgfunc @cfgfunc_with_ops(f32) {
cfgfunc @cfgfunc_with_ops(f32) {
^bb0(%a : f32):
  // CHECK: %0 = "getTensor"() : () -> tensor<4x4x?xf32>
  %t = "getTensor"() : () -> tensor<4x4x?xf32>

  // CHECK: %1 = dim %0, 2 : tensor<4x4x?xf32>
  %t2 = "dim"(%t){index: 2} : (tensor<4x4x?xf32>) -> index

  // CHECK: %2 = addf %arg0, %arg0 : f32
  %x = "addf"(%a, %a) : (f32,f32) -> (f32)

  // CHECK:   return
  return
}

// CHECK-LABEL: cfgfunc @standard_instrs(tensor<4x4x?xf32>, f32, i32, index) {
cfgfunc @standard_instrs(tensor<4x4x?xf32>, f32, i32, index) {
// CHECK: ^bb0(%arg0: tensor<4x4x?xf32>, %arg1: f32, %arg2: i32, %arg3: index):
^bb42(%t: tensor<4x4x?xf32>, %f: f32, %i: i32, %idx : index):
  // CHECK: %0 = dim %arg0, 2 : tensor<4x4x?xf32>
  %a = "dim"(%t){index: 2} : (tensor<4x4x?xf32>) -> index

  // CHECK: %1 = dim %arg0, 2 : tensor<4x4x?xf32>
  %a2 = dim %t, 2 : tensor<4x4x?xf32>

  // CHECK: %2 = addf %arg1, %arg1 : f32
  %f2 = "addf"(%f, %f) : (f32,f32) -> f32

  // CHECK: %3 = addf %2, %2 : f32
  %f3 = addf %f2, %f2 : f32

  // CHECK: %4 = addi %arg2, %arg2 : i32
  %i2 = "addi"(%i, %i) : (i32,i32) -> i32

  // CHECK: %5 = addi %4, %4 : i32
  %i3 = addi %i2, %i2 : i32

  // CHECK: %{{[0-9]+}} = addi %arg3, %arg3 : index
  %idx1 = addi %idx, %idx : index

  // CHECK: %{{[0-9]+}} = addi %arg3, %{{[0-9]+}} : index
  %idx2 = "addi"(%idx, %idx1) : (index, index) -> index

  // CHECK: %8 = subf %arg1, %arg1 : f32
  %f4 = "subf"(%f, %f) : (f32,f32) -> f32

  // CHECK: %9 = subf %8, %8 : f32
  %f5 = subf %f4, %f4 : f32

  // CHECK: %10 = subi %arg2, %arg2 : i32
  %i4 = "subi"(%i, %i) : (i32,i32) -> i32

  // CHECK: %11 = subi %10, %10 : i32
  %i5 = subi %i4, %i4 : i32

  // CHECK: %12 = mulf %2, %2 : f32
  %f6 = mulf %f2, %f2 : f32

  // CHECK: %13 = muli %4, %4 : i32
  %i6 = muli %i2, %i2 : i32

  // CHECK: %c42_i32 = constant 42 : i32
  %x = "constant"(){value: 42 : i32} : () -> i32

  // CHECK: %c42_i32_0 = constant 42 : i32
  %7 = constant 42 : i32

  // CHECK: %c43 = constant {crazy: "foo"} 43 : index
  %8 = constant {crazy: "foo"} 43: index

  // CHECK: %cst = constant 4.300000e+01 : bf16
  %9 = constant 43.0 : bf16

  // CHECK: %f = constant @cfgfunc_with_ops : (f32) -> ()
  %10 = constant @cfgfunc_with_ops : (f32) -> ()

  // CHECK: %f_1 = constant @affine_apply : () -> ()
  %11 = constant @affine_apply : () -> ()

  // CHECK: %f_2 = constant @affine_apply : () -> ()
  %12 = constant @affine_apply : () -> ()

  // CHECK: %cst_3 = constant splat<vector<4xi32>, 0> : vector<4xi32>
  %13 = constant splat<vector<4 x i32>, 0> : vector<4 x i32>

  // CHECK: %cst_4 = constant splat<tensor<42xi32>, 0> : tensor<42xi32>
  %tci32 = constant splat<tensor<42 x i32>, 0> : tensor<42 x i32>

  // CHECK: %cst_5 = constant splat<vector<42xi32>, 0> : vector<42xi32>
  %vci32 = constant splat<vector<42 x i32>, 0> : vector<42 x i32>

  // CHECK: %{{[0-9]+}} = cmpi "eq", %{{[0-9]+}}, %{{[0-9]+}} : i32
  %14 = cmpi "eq", %i3, %i4 : i32

  // Predicate 1 means inequality comparison.
  // CHECK: %{{[0-9]+}} = cmpi "ne", %{{[0-9]+}}, %{{[0-9]+}} : i32
  %15 = "cmpi"(%i3, %i4) {predicate: 1} : (i32, i32) -> i1

  // CHECK: %{{[0-9]+}} = cmpi "slt", %cst_3, %cst_3 : vector<4xi32>
  %16 = cmpi "slt", %13, %13 : vector<4 x i32>

  // CHECK: %{{[0-9]+}} = cmpi "ne", %cst_3, %cst_3 : vector<4xi32>
  %17 = "cmpi"(%13, %13) {predicate: 1} : (vector<4 x i32>, vector<4 x i32>) -> vector<4 x i1>

  // CHECK: %{{[0-9]+}} = cmpi "slt", %arg3, %arg3 : index
  %18 = cmpi "slt", %idx, %idx : index

  // CHECK: %{{[0-9]+}} = cmpi "eq", %cst_4, %cst_4 : tensor<42xi32>
  %19 = cmpi "eq", %tci32, %tci32 : tensor<42 x i32>

  // CHECK: %{{[0-9]+}} = cmpi "eq", %cst_5, %cst_5 : vector<42xi32>
  %20 = cmpi "eq", %vci32, %vci32 : vector<42 x i32>

  // CHECK: %{{[0-9]+}} = select %{{[0-9]+}}, %arg3, %arg3 : index
  %21 = select %18, %idx, %idx : index

  // CHECK: %{{[0-9]+}} = select %{{[0-9]+}}, %cst_4, %cst_4 : tensor<42xi32>
  %22 = select %19, %tci32, %tci32 : tensor<42 x i32>

  // CHECK: %{{[0-9]+}} = select %{{[0-9]+}}, %cst_5, %cst_5 : vector<42xi32>
  %23 = select %20, %vci32, %vci32 : vector<42 x i32>

  // CHECK: %{{[0-9]+}} = select %{{[0-9]+}}, %arg3, %arg3 : index
  %24 = "select"(%18, %idx, %idx) : (i1, index, index) -> index

  // CHECK: %{{[0-9]+}} = select %{{[0-9]+}}, %cst_4, %cst_4 : tensor<42xi32>
  %25 = "select"(%19, %tci32, %tci32) : (tensor<42 x i1>, tensor<42 x i32>, tensor<42 x i32>) -> tensor<42 x i32>

  return
}

// CHECK-LABEL: cfgfunc @affine_apply() {
cfgfunc @affine_apply() {
^bb0:
  %i = "constant"() {value: 0: index} : () -> index
  %j = "constant"() {value: 1: index} : () -> index

  // CHECK: affine_apply #map0(%c0)
  %a = "affine_apply" (%i) { map: (d0) -> (d0 + 1) } :
    (index) -> (index)

  // CHECK: affine_apply #map1(%c0, %c1)
  %b = "affine_apply" (%i, %j) { map: #map5 } :
    (index, index) -> (index, index)

  // CHECK: affine_apply #map2(%c0, %c1)[%c1, %c0]
  %c = affine_apply (i,j)[m,n] -> (i+n, j+m)(%i, %j)[%j, %i]

  // CHECK: affine_apply #map3()[%c0]
  %d = affine_apply ()[x] -> (x+1)()[%i]

  return
}

// CHECK-LABEL: cfgfunc @load_store
cfgfunc @load_store(memref<4x4xi32>, index) {
^bb0(%0: memref<4x4xi32>, %1: index):
  // CHECK: %0 = load %arg0[%arg1, %arg1] : memref<4x4xi32>
  %2 = "load"(%0, %1, %1) : (memref<4x4xi32>, index, index)->i32

  // CHECK: %1 = load %arg0[%arg1, %arg1] : memref<4x4xi32>
  %3 = load %0[%1, %1] : memref<4x4xi32>

  return
}

// CHECK-LABEL: mlfunc @return_op(%arg0 : i32) -> i32 {
mlfunc @return_op(%a : i32) -> i32 {
  // CHECK: return %arg0 : i32
  "return" (%a) : (i32)->()
}

// CHECK-LABEL: mlfunc @calls(%arg0 : i32) {
mlfunc @calls(%arg0 : i32) {
  // CHECK: %0 = call @return_op(%arg0) : (i32) -> i32
  %x = call @return_op(%arg0) : (i32) -> i32
  // CHECK: %1 = call @return_op(%0) : (i32) -> i32
  %y = call @return_op(%x) : (i32) -> i32
  // CHECK: %2 = call @return_op(%0) : (i32) -> i32
  %z = "call"(%x) {callee: @return_op : (i32) -> i32} : (i32) -> i32

  // CHECK: %f = constant @affine_apply : () -> ()
  %f = constant @affine_apply : () -> ()

  // CHECK: call_indirect %f() : () -> ()
  call_indirect %f() : () -> ()

  // CHECK: %f_0 = constant @return_op : (i32) -> i32
  %f_0 = constant @return_op : (i32) -> i32

  // CHECK: %3 = call_indirect %f_0(%arg0) : (i32) -> i32
  %2 = call_indirect %f_0(%arg0) : (i32) -> i32

  // CHECK: %4 = call_indirect %f_0(%arg0) : (i32) -> i32
  %3 = "call_indirect"(%f_0, %arg0) : ((i32) -> i32, i32) -> i32

  return
}

// CHECK-LABEL: mlfunc @extract_element(%arg0 : tensor<*xi32>, %arg1 : tensor<4x4xf32>) -> i32 {
mlfunc @extract_element(%arg0 : tensor<*xi32>, %arg1 : tensor<4x4xf32>) -> i32 {
  %c0 = "constant"() {value: 0: index} : () -> index

  // CHECK: %0 = extract_element %arg0[%c0, %c0, %c0, %c0] : tensor<*xi32>
  %0 = extract_element %arg0[%c0, %c0, %c0, %c0] : tensor<*xi32>

  // CHECK: %1 = extract_element %arg1[%c0, %c0] : tensor<4x4xf32>
  %1 = extract_element %arg1[%c0, %c0] : tensor<4x4xf32>

  return %0 : i32
}

// CHECK-LABEL: mlfunc @tensor_cast(%arg0
mlfunc @tensor_cast(%arg0 : tensor<*xf32>, %arg1 : tensor<4x4xf32>, %arg2 : tensor<?x?xf32>) {
  // CHECK: %0 = tensor_cast %arg0 : tensor<*xf32> to tensor<?x?xf32>
  %0 = tensor_cast %arg0 : tensor<*xf32> to tensor<?x?xf32>

  // CHECK: %1 = tensor_cast %arg1 : tensor<4x4xf32> to tensor<*xf32>
  %1 = tensor_cast %arg1 : tensor<4x4xf32> to tensor<*xf32>

  // CHECK: %2 = tensor_cast %arg2 : tensor<?x?xf32> to tensor<4x?xf32>
  %2 = tensor_cast %arg2 : tensor<?x?xf32> to tensor<4x?xf32>

  // CHECK: %3 = tensor_cast %2 : tensor<4x?xf32> to tensor<?x?xf32>
  %3 = tensor_cast %2 : tensor<4x?xf32> to tensor<?x?xf32>

  return
}

// CHECK-LABEL: mlfunc @memref_cast(%arg0
mlfunc @memref_cast(%arg0 : memref<4xf32>, %arg1 : memref<?xf32>) {
  // CHECK: %0 = memref_cast %arg0 : memref<4xf32> to memref<?xf32>
  %0 = memref_cast %arg0 : memref<4xf32> to memref<?xf32>

  // CHECK: %1 = memref_cast %arg1 : memref<?xf32> to memref<4xf32>
  %1 = memref_cast %arg1 : memref<?xf32> to memref<4xf32>
  return
}

// CHECK-LABEL: mlfunc @test_dimop(%arg0
mlfunc @test_dimop(%arg0 : tensor<4x4x?xf32>) {
  // CHECK: %0 = dim %arg0, 2 : tensor<4x4x?xf32>
  %0 = dim %arg0, 2 : tensor<4x4x?xf32>
  // use dim as an affine_int to ensure type correctness
  %1 = affine_apply (d0) -> (d0)(%0)
  return
}


// CHECK-LABEL: mlfunc @test_vector_transfer_ops(%arg0
mlfunc @test_vector_transfer_ops(%arg0 : memref<?x?xf32>) {
  %c3 = constant 3 : index
  %cst = constant 3.0 : f32
  // CHECK: %0 = vector_transfer_read %arg0, %c3, %c3 {permutation_map: #[[map_proj_d0d1_d0]]} : (memref<?x?xf32>, index, index) -> vector<128xf32>
  %0 = vector_transfer_read %arg0, %c3, %c3 {permutation_map: (d0, d1)->(d0)} : (memref<?x?xf32>, index, index) -> vector<128xf32>
  // CHECK: %1 = vector_transfer_read %arg0, %c3, %c3 {permutation_map: #[[map_proj_d0d1_d1d0]]} : (memref<?x?xf32>, index, index) -> vector<3x7xf32>
  %1 = vector_transfer_read %arg0, %c3, %c3 {permutation_map: (d0, d1)->(d1, d0)} : (memref<?x?xf32>, index, index) -> vector<3x7xf32>
  // CHECK: %2 = vector_transfer_read %arg0, %c3, %c3, %cst {permutation_map: #[[map_proj_d0d1_d0]]} : (memref<?x?xf32>, index, index, f32) -> vector<128xf32>
  %2 = vector_transfer_read %arg0, %c3, %c3, %cst {permutation_map: (d0, d1)->(d0)} : (memref<?x?xf32>, index, index, f32) -> vector<128xf32>
  // CHECK: %3 = vector_transfer_read %arg0, %c3, %c3, %cst {permutation_map: #[[map_proj_d0d1_d1]]} : (memref<?x?xf32>, index, index, f32) -> vector<128xf32>
  %3 = vector_transfer_read %arg0, %c3, %c3, %cst {permutation_map: (d0, d1)->(d1)} : (memref<?x?xf32>, index, index, f32) -> vector<128xf32>
  //
  // CHECK: vector_transfer_write %0, %arg0, %c3, %c3 {permutation_map: #[[map_proj_d0d1_d0]]} : vector<128xf32>, memref<?x?xf32>, index, index
  vector_transfer_write %0, %arg0, %c3, %c3 {permutation_map: (d0, d1)->(d0)} : vector<128xf32>, memref<?x?xf32>, index, index
  // CHECK: vector_transfer_write %1, %arg0, %c3, %c3 {permutation_map: #[[map_proj_d0d1_d1d0]]} : vector<3x7xf32>, memref<?x?xf32>, index, index
  vector_transfer_write %1, %arg0, %c3, %c3 {permutation_map: (d0, d1)->(d1, d0)} : vector<3x7xf32>, memref<?x?xf32>, index, index
  return
}
