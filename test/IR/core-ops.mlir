// RUN: mlir-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: mlir-opt -mlir-print-op-generic %s | mlir-opt | FileCheck %s

// CHECK: #map0 = (d0) -> (d0 + 1)

// CHECK: #map1 = ()[s0] -> (s0 + 1)

// CHECK-DAG: #[[VIEW_MAP1:map[0-9]+]] = (d0, d1) -> (d0 * 4 + d1)
// CHECK-DAG: #[[VIEW_MAP2:map[0-9]+]] = (d0, d1)[s0, s1] -> (d0 * s1 + d1 + s0)
// CHECK-DAG: #[[VIEW_MAP3:map[0-9]+]] = (d0, d1)[s0] -> (d0 * s0 + d1)

// CHECK-DAG: #[[BASE_MAP0:map[0-9]+]] = (d0, d1, d2) -> (d0 * 64 + d1 * 4 + d2)
// CHECK-DAG: #[[SUBVIEW_MAP0:map[0-9]+]] = (d0, d1, d2)[s0, s1, s2, s3] -> (d0 * s1 + d1 * s2 + d2 * s3 + s0)

// CHECK-DAG: #[[BASE_MAP1:map[0-9]+]] = (d0)[s0] -> (d0 + s0)
// CHECK-DAG: #[[SUBVIEW_MAP1:map[0-9]+]] = (d0)[s0, s1] -> (d0 * s1 + s0)

// CHECK-DAG: #[[BASE_MAP2:map[0-9]+]] = (d0, d1) -> (d0 * 22 + d1)
// CHECK-DAG: #[[SUBVIEW_MAP2:map[0-9]+]] = (d0, d1)[s0, s1, s2] -> (d0 * s1 + d1 * s2 + s0)

// CHECK-LABEL: func @func_with_ops(%arg0: f32) {
func @func_with_ops(f32) {
^bb0(%a : f32):
  // CHECK: %0 = "getTensor"() : () -> tensor<4x4x?xf32>
  %t = "getTensor"() : () -> tensor<4x4x?xf32>

  // CHECK: %1 = dim %0, 2 : tensor<4x4x?xf32>
  %t2 = "std.dim"(%t){index = 2} : (tensor<4x4x?xf32>) -> index

  // CHECK: %2 = addf %arg0, %arg0 : f32
  %x = "std.addf"(%a, %a) : (f32,f32) -> (f32)

  // CHECK:   return
  return
}

// CHECK-LABEL: func @standard_instrs(%arg0: tensor<4x4x?xf32>, %arg1: f32, %arg2: i32, %arg3: index, %arg4: i64, %arg5: f16) {
func @standard_instrs(tensor<4x4x?xf32>, f32, i32, index, i64, f16) {
^bb42(%t: tensor<4x4x?xf32>, %f: f32, %i: i32, %idx : index, %j: i64, %half: f16):
  // CHECK: %0 = dim %arg0, 2 : tensor<4x4x?xf32>
  %a = "std.dim"(%t){index = 2} : (tensor<4x4x?xf32>) -> index

  // CHECK: %1 = dim %arg0, 2 : tensor<4x4x?xf32>
  %a2 = dim %t, 2 : tensor<4x4x?xf32>

  // CHECK: %2 = addf %arg1, %arg1 : f32
  %f2 = "std.addf"(%f, %f) : (f32,f32) -> f32

  // CHECK: %3 = addf %2, %2 : f32
  %f3 = addf %f2, %f2 : f32

  // CHECK: %4 = addi %arg2, %arg2 : i32
  %i2 = "std.addi"(%i, %i) : (i32,i32) -> i32

  // CHECK: %5 = addi %4, %4 : i32
  %i3 = addi %i2, %i2 : i32

  // CHECK: %{{[0-9]+}} = addi %arg3, %arg3 : index
  %idx1 = addi %idx, %idx : index

  // CHECK: %{{[0-9]+}} = addi %arg3, %{{[0-9]+}} : index
  %idx2 = "std.addi"(%idx, %idx1) : (index, index) -> index

  // CHECK: %8 = subf %arg1, %arg1 : f32
  %f4 = "std.subf"(%f, %f) : (f32,f32) -> f32

  // CHECK: %9 = subf %8, %8 : f32
  %f5 = subf %f4, %f4 : f32

  // CHECK: %10 = subi %arg2, %arg2 : i32
  %i4 = "std.subi"(%i, %i) : (i32,i32) -> i32

  // CHECK: %11 = subi %10, %10 : i32
  %i5 = subi %i4, %i4 : i32

  // CHECK: %12 = mulf %2, %2 : f32
  %f6 = mulf %f2, %f2 : f32

  // CHECK: %13 = muli %4, %4 : i32
  %i6 = muli %i2, %i2 : i32

  // CHECK: %c42_i32 = constant 42 : i32
  %x = "std.constant"(){value = 42 : i32} : () -> i32

  // CHECK: %c42_i32_0 = constant 42 : i32
  %7 = constant 42 : i32

  // CHECK: %c43 = constant {crazy = "std.foo"} 43 : index
  %8 = constant {crazy = "std.foo"} 43: index

  // CHECK: %cst = constant 4.300000e+01 : bf16
  %9 = constant 43.0 : bf16

  // CHECK: %f = constant @func_with_ops : (f32) -> ()
  %10 = constant @func_with_ops : (f32) -> ()

  // CHECK: %f_1 = constant @affine_apply : () -> ()
  %11 = constant @affine_apply : () -> ()

  // CHECK: %f_2 = constant @affine_apply : () -> ()
  %12 = constant @affine_apply : () -> ()

  // CHECK: %cst_3 = constant dense<0> : vector<4xi32>
  %13 = constant dense<0> : vector<4 x i32>

  // CHECK: %cst_4 = constant dense<0> : tensor<42xi32>
  %tci32 = constant dense<0> : tensor<42 x i32>

  // CHECK: %cst_5 = constant dense<0> : vector<42xi32>
  %vci32 = constant dense<0> : vector<42 x i32>

  // CHECK: %{{[0-9]+}} = cmpi "eq", %{{[0-9]+}}, %{{[0-9]+}} : i32
  %14 = cmpi "eq", %i3, %i4 : i32

  // Predicate 1 means inequality comparison.
  // CHECK: %{{[0-9]+}} = cmpi "ne", %{{[0-9]+}}, %{{[0-9]+}} : i32
  %15 = "std.cmpi"(%i3, %i4) {predicate = 1} : (i32, i32) -> i1

  // CHECK: %{{[0-9]+}} = cmpi "slt", %cst_3, %cst_3 : vector<4xi32>
  %16 = cmpi "slt", %13, %13 : vector<4 x i32>

  // CHECK: %{{[0-9]+}} = cmpi "ne", %cst_3, %cst_3 : vector<4xi32>
  %17 = "std.cmpi"(%13, %13) {predicate = 1} : (vector<4 x i32>, vector<4 x i32>) -> vector<4 x i1>

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
  %24 = "std.select"(%18, %idx, %idx) : (i1, index, index) -> index

  // CHECK: %{{[0-9]+}} = select %{{[0-9]+}}, %cst_4, %cst_4 : tensor<42xi32>
  %25 = "std.select"(%19, %tci32, %tci32) : (tensor<42 x i1>, tensor<42 x i32>, tensor<42 x i32>) -> tensor<42 x i32>

  // CHECK: %{{[0-9]+}} = divis %arg2, %arg2 : i32
  %26 = divis %i, %i : i32

  // CHECK: %{{[0-9]+}} = divis %arg3, %arg3 : index
  %27 = divis %idx, %idx : index

  // CHECK: %{{[0-9]+}} = divis %cst_5, %cst_5 : vector<42xi32>
  %28 = divis %vci32, %vci32 : vector<42 x i32>

  // CHECK: %{{[0-9]+}} = divis %cst_4, %cst_4 : tensor<42xi32>
  %29 = divis %tci32, %tci32 : tensor<42 x i32>

  // CHECK: %{{[0-9]+}} = divis %arg2, %arg2 : i32
  %30 = "std.divis"(%i, %i) : (i32, i32) -> i32

  // CHECK: %{{[0-9]+}} = diviu %arg2, %arg2 : i32
  %31 = diviu %i, %i : i32

  // CHECK: %{{[0-9]+}} = diviu %arg3, %arg3 : index
  %32 = diviu %idx, %idx : index

  // CHECK: %{{[0-9]+}} = diviu %cst_5, %cst_5 : vector<42xi32>
  %33 = diviu %vci32, %vci32 : vector<42 x i32>

  // CHECK: %{{[0-9]+}} = diviu %cst_4, %cst_4 : tensor<42xi32>
  %34 = diviu %tci32, %tci32 : tensor<42 x i32>

  // CHECK: %{{[0-9]+}} = diviu %arg2, %arg2 : i32
  %35 = "std.diviu"(%i, %i) : (i32, i32) -> i32

  // CHECK: %{{[0-9]+}} = remis %arg2, %arg2 : i32
  %36 = remis %i, %i : i32

  // CHECK: %{{[0-9]+}} = remis %arg3, %arg3 : index
  %37 = remis %idx, %idx : index

  // CHECK: %{{[0-9]+}} = remis %cst_5, %cst_5 : vector<42xi32>
  %38 = remis %vci32, %vci32 : vector<42 x i32>

  // CHECK: %{{[0-9]+}} = remis %cst_4, %cst_4 : tensor<42xi32>
  %39 = remis %tci32, %tci32 : tensor<42 x i32>

  // CHECK: %{{[0-9]+}} = remis %arg2, %arg2 : i32
  %40 = "std.remis"(%i, %i) : (i32, i32) -> i32

  // CHECK: %{{[0-9]+}} = remiu %arg2, %arg2 : i32
  %41 = remiu %i, %i : i32

  // CHECK: %{{[0-9]+}} = remiu %arg3, %arg3 : index
  %42 = remiu %idx, %idx : index

  // CHECK: %{{[0-9]+}} = remiu %cst_5, %cst_5 : vector<42xi32>
  %43 = remiu %vci32, %vci32 : vector<42 x i32>

  // CHECK: %{{[0-9]+}} = remiu %cst_4, %cst_4 : tensor<42xi32>
  %44 = remiu %tci32, %tci32 : tensor<42 x i32>

  // CHECK: %{{[0-9]+}} = remiu %arg2, %arg2 : i32
  %45 = "std.remiu"(%i, %i) : (i32, i32) -> i32

  // CHECK: %{{[0-9]+}} = divf %arg1, %arg1 : f32
  %46 = "std.divf"(%f, %f) : (f32,f32) -> f32

  // CHECK: %{{[0-9]+}} = divf %arg1, %arg1 : f32
  %47 = divf %f, %f : f32

  // CHECK: %{{[0-9]+}} = divf %arg0, %arg0 : tensor<4x4x?xf32>
  %48 = divf %t, %t : tensor<4x4x?xf32>

  // CHECK: %{{[0-9]+}} = remf %arg1, %arg1 : f32
  %49 = "std.remf"(%f, %f) : (f32,f32) -> f32

  // CHECK: %{{[0-9]+}} = remf %arg1, %arg1 : f32
  %50 = remf %f, %f : f32

  // CHECK: %{{[0-9]+}} = remf %arg0, %arg0 : tensor<4x4x?xf32>
  %51 = remf %t, %t : tensor<4x4x?xf32>

  // CHECK: %{{[0-9]+}} = and %arg2, %arg2 : i32
  %52 = "std.and"(%i, %i) : (i32,i32) -> i32

  // CHECK: %{{[0-9]+}} = and %arg2, %arg2 : i32
  %53 = and %i, %i : i32

  // CHECK: %{{[0-9]+}} = and %cst_5, %cst_5 : vector<42xi32>
  %54 = std.and %vci32, %vci32 : vector<42 x i32>

  // CHECK: %{{[0-9]+}} = and %cst_4, %cst_4 : tensor<42xi32>
  %55 = and %tci32, %tci32 : tensor<42 x i32>

  // CHECK: %{{[0-9]+}} = or %arg2, %arg2 : i32
  %56 = "std.or"(%i, %i) : (i32,i32) -> i32

  // CHECK: %{{[0-9]+}} = or %arg2, %arg2 : i32
  %57 = or %i, %i : i32

  // CHECK: %{{[0-9]+}} = or %cst_5, %cst_5 : vector<42xi32>
  %58 = std.or %vci32, %vci32 : vector<42 x i32>

  // CHECK: %{{[0-9]+}} = or %cst_4, %cst_4 : tensor<42xi32>
  %59 = or %tci32, %tci32 : tensor<42 x i32>

  // CHECK: %{{[0-9]+}} = xor %arg2, %arg2 : i32
  %60 = "std.xor"(%i, %i) : (i32,i32) -> i32

  // CHECK: %{{[0-9]+}} = xor %arg2, %arg2 : i32
  %61 = xor %i, %i : i32

  // CHECK: %{{[0-9]+}} = xor %cst_5, %cst_5 : vector<42xi32>
  %62 = std.xor %vci32, %vci32 : vector<42 x i32>

  // CHECK: %{{[0-9]+}} = xor %cst_4, %cst_4 : tensor<42xi32>
  %63 = xor %tci32, %tci32 : tensor<42 x i32>

  %64 = constant dense<0.> : vector<4 x f32>
  %tcf32 = constant dense<0.> : tensor<42 x f32>
  %vcf32 = constant dense<0.> : vector<4 x f32>

  // CHECK: %{{[0-9]+}} = cmpf "ogt", %{{[0-9]+}}, %{{[0-9]+}} : f32
  %65 = cmpf "ogt", %f3, %f4 : f32

  // Predicate 0 means ordered equality comparison.
  // CHECK: %{{[0-9]+}} = cmpf "oeq", %{{[0-9]+}}, %{{[0-9]+}} : f32
  %66 = "std.cmpf"(%f3, %f4) {predicate = 1} : (f32, f32) -> i1

  // CHECK: %{{[0-9]+}} = cmpf "olt", %cst_8, %cst_8 : vector<4xf32>
  %67 = cmpf "olt", %vcf32, %vcf32 : vector<4 x f32>

  // CHECK: %{{[0-9]+}} = cmpf "oeq", %cst_8, %cst_8 : vector<4xf32>
  %68 = "std.cmpf"(%vcf32, %vcf32) {predicate = 1} : (vector<4 x f32>, vector<4 x f32>) -> vector<4 x i1>

  // CHECK: %{{[0-9]+}} = cmpf "oeq", %cst_7, %cst_7 : tensor<42xf32>
  %69 = cmpf "oeq", %tcf32, %tcf32 : tensor<42 x f32>

  // CHECK: %{{[0-9]+}} = cmpf "oeq", %cst_8, %cst_8 : vector<4xf32>
  %70 = cmpf "oeq", %vcf32, %vcf32 : vector<4 x f32>

  // CHECK: %{{[0-9]+}} = rank %arg0 : tensor<4x4x?xf32>
  %71 = "std.rank"(%t) : (tensor<4x4x?xf32>) -> index

  // CHECK: %{{[0-9]+}} = rank %arg0 : tensor<4x4x?xf32>
  %72 = rank %t : tensor<4x4x?xf32>

  // CHECK: = constant unit
  %73 = constant unit

  // CHECK: constant true
  %74 = constant true

  // CHECK: constant false
  %75 = constant false

  // CHECK: = index_cast {{.*}} : index to i64
  %76 = index_cast %idx : index to i64

  // CHECK: = index_cast {{.*}} : i32 to index
  %77 = index_cast %i : i32 to index

  // CHECK: = sitofp {{.*}} : i32 to f32
  %78 = sitofp %i : i32 to f32

  // CHECK: = sitofp {{.*}} : i32 to f64
  %79 = sitofp %i : i32 to f64

  // CHECK: = sitofp {{.*}} : i64 to f32
  %80 = sitofp %j : i64 to f32

  // CHECK: = sitofp {{.*}} : i64 to f64
  %81 = sitofp %j : i64 to f64

  // CHECK: = sexti %arg2 : i32 to i64
  %82 = "std.sexti"(%i) : (i32) -> i64

  // CHECK: = sexti %arg2 : i32 to i64
  %83 = sexti %i : i32 to i64

  // CHECK: %{{[0-9]+}} = sexti %cst_5 : vector<42xi32>
  %84 = sexti %vci32 : vector<42 x i32> to vector<42 x i64>

  // CHECK: %{{[0-9]+}} = sexti %cst_4 : tensor<42xi32>
  %85 = sexti %tci32 : tensor<42 x i32> to tensor<42 x i64>

  // CHECK: = zexti %arg2 : i32 to i64
  %86 = "std.zexti"(%i) : (i32) -> i64

  // CHECK: = zexti %arg2 : i32 to i64
  %87 = zexti %i : i32 to i64

  // CHECK: %{{[0-9]+}} = zexti %cst_5 : vector<42xi32>
  %88 = zexti %vci32 : vector<42 x i32> to vector<42 x i64>

  // CHECK: %{{[0-9]+}} = zexti %cst_4 : tensor<42xi32>
  %89 = zexti %tci32 : tensor<42 x i32> to tensor<42 x i64>

  // CHECK: = trunci %arg2 : i32 to i16
  %90 = "std.trunci"(%i) : (i32) -> i16

  // CHECK: = trunci %arg2 : i32 to i16
  %91 = trunci %i : i32 to i16

  // CHECK: %{{[0-9]+}} = trunci %cst_5 : vector<42xi32>
  %92 = trunci %vci32 : vector<42 x i32> to vector<42 x i16>

  // CHECK: %{{[0-9]+}} = trunci %cst_4 : tensor<42xi32>
  %93 = trunci %tci32 : tensor<42 x i32> to tensor<42 x i16>

  // CHECK: = fpext {{.*}} : f16 to f32
  %94 = fpext %half : f16 to f32

  // CHECK: = fptrunc {{.*}} : f32 to f16
  %95 = fptrunc %f : f32 to f16

  // CHECK: %{{[0-9]+}} = exp %arg1 : f32
  %96 = "std.exp"(%f) : (f32) -> f32

  // CHECK: %{{[0-9]+}} = exp %arg1 : f32
  %97 = exp %f : f32

  // CHECK: %{{[0-9]+}} = exp %arg0 : tensor<4x4x?xf32>
  %98 = exp %t : tensor<4x4x?xf32>
  return
}

// CHECK-LABEL: func @affine_apply() {
func @affine_apply() {
  %i = "std.constant"() {value = 0: index} : () -> index
  %j = "std.constant"() {value = 1: index} : () -> index

  // CHECK: affine.apply #map0(%c0)
  %a = "affine.apply" (%i) { map = (d0) -> (d0 + 1) } :
    (index) -> (index)

  // CHECK: affine.apply #map1()[%c0]
  %b = affine.apply ()[x] -> (x+1)()[%i]

  return
}

// CHECK-LABEL: func @load_store
func @load_store(memref<4x4xi32>, index) {
^bb0(%0: memref<4x4xi32>, %1: index):
  // CHECK: %0 = load %arg0[%arg1, %arg1] : memref<4x4xi32>
  %2 = "std.load"(%0, %1, %1) : (memref<4x4xi32>, index, index)->i32

  // CHECK: %1 = load %arg0[%arg1, %arg1] : memref<4x4xi32>
  %3 = load %0[%1, %1] : memref<4x4xi32>

  return
}

// Test with zero-dimensional operands using no index in load/store.
// CHECK-LABEL: func @zero_dim_no_idx
func @zero_dim_no_idx(%arg0 : memref<i32>, %arg1 : memref<i32>, %arg2 : memref<i32>) {
  %0 = std.load %arg0[] : memref<i32>
  std.store %0, %arg1[] : memref<i32>
  return
  // CHECK: %0 = load %{{.*}}[] : memref<i32>
  // CHECK: store %{{.*}}, %{{.*}}[] : memref<i32>
}

// CHECK-LABEL: func @return_op(%arg0: i32) -> i32 {
func @return_op(%a : i32) -> i32 {
  // CHECK: return %arg0 : i32
  "std.return" (%a) : (i32)->()
}

// CHECK-LABEL: func @calls(%arg0: i32) {
func @calls(%arg0: i32) {
  // CHECK: %0 = call @return_op(%arg0) : (i32) -> i32
  %x = call @return_op(%arg0) : (i32) -> i32
  // CHECK: %1 = call @return_op(%0) : (i32) -> i32
  %y = call @return_op(%x) : (i32) -> i32
  // CHECK: %2 = call @return_op(%0) : (i32) -> i32
  %z = "std.call"(%x) {callee = @return_op} : (i32) -> i32

  // CHECK: %f = constant @affine_apply : () -> ()
  %f = constant @affine_apply : () -> ()

  // CHECK: call_indirect %f() : () -> ()
  call_indirect %f() : () -> ()

  // CHECK: %f_0 = constant @return_op : (i32) -> i32
  %f_0 = constant @return_op : (i32) -> i32

  // CHECK: %3 = call_indirect %f_0(%arg0) : (i32) -> i32
  %2 = call_indirect %f_0(%arg0) : (i32) -> i32

  // CHECK: %4 = call_indirect %f_0(%arg0) : (i32) -> i32
  %3 = "std.call_indirect"(%f_0, %arg0) : ((i32) -> i32, i32) -> i32

  return
}

// CHECK-LABEL: func @extract_element(%arg0: tensor<*xi32>, %arg1: tensor<4x4xf32>) -> i32 {
func @extract_element(%arg0: tensor<*xi32>, %arg1 : tensor<4x4xf32>) -> i32 {
  %c0 = "std.constant"() {value = 0: index} : () -> index

  // CHECK: %0 = extract_element %arg0[%c0, %c0, %c0, %c0] : tensor<*xi32>
  %0 = extract_element %arg0[%c0, %c0, %c0, %c0] : tensor<*xi32>

  // CHECK: %1 = extract_element %arg1[%c0, %c0] : tensor<4x4xf32>
  %1 = extract_element %arg1[%c0, %c0] : tensor<4x4xf32>

  return %0 : i32
}

// CHECK-LABEL: func @tensor_cast(%arg0
func @tensor_cast(%arg0: tensor<*xf32>, %arg1 : tensor<4x4xf32>, %arg2: tensor<?x?xf32>) {
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

// CHECK-LABEL: func @memref_cast(%arg0
func @memref_cast(%arg0: memref<4xf32>, %arg1 : memref<?xf32>) {
  // CHECK: %0 = memref_cast %arg0 : memref<4xf32> to memref<?xf32>
  %0 = memref_cast %arg0 : memref<4xf32> to memref<?xf32>

  // CHECK: %1 = memref_cast %arg1 : memref<?xf32> to memref<4xf32>
  %1 = memref_cast %arg1 : memref<?xf32> to memref<4xf32>
  return
}

// CHECK-LABEL: func @memref_view(%arg0
func @memref_view(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = alloc() : memref<2048xi8>
  // Test two dynamic sizes and dynamic offset.
  // CHECK: %{{.*}} = std.view %0[%arg2][%arg0, %arg1] : memref<2048xi8> to memref<?x?xf32, #[[VIEW_MAP2]]>
  %1 = view %0[%arg2][%arg0, %arg1]
    : memref<2048xi8> to memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + d1 + s0)>

  // Test two dynamic sizes and static offset.
  // CHECK: %{{.*}} = std.view %0[][%arg0, %arg1] : memref<2048xi8> to memref<?x?xf32, #[[VIEW_MAP3]]>
  %2 = view %0[][%arg0, %arg1]
    : memref<2048xi8> to memref<?x?xf32, (d0, d1)[s0] -> (d0 * s0 + d1)>

  // Test one dynamic size and dynamic offset.
  // CHECK: %{{.*}} = std.view %0[%arg2][%arg1] : memref<2048xi8> to memref<4x?xf32, #[[VIEW_MAP2]]>
  %3 = view %0[%arg2][%arg1]
    : memref<2048xi8> to memref<4x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + d1 + s0)>

  // Test one dynamic size and static offset.
  // CHECK: %{{.*}} = std.view %0[][%arg0] : memref<2048xi8> to memref<?x4xf32, #[[VIEW_MAP1]]>
  %4 = view %0[][%arg0]
    : memref<2048xi8> to memref<?x4xf32, (d0, d1) -> (d0 * 4 + d1)>

  // Test static sizes and static offset.
  // CHECK: %{{.*}} = std.view %0[][] : memref<2048xi8> to memref<64x4xf32, #[[VIEW_MAP1]]>
  %5 = view %0[][]
    : memref<2048xi8> to memref<64x4xf32, (d0, d1) -> (d0 * 4 + d1)>
  return
}

// CHECK-LABEL: func @memref_subview(%arg0
func @memref_subview(%arg0 : index, %arg1 : index, %arg2 : index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index

  //%2 = alloc() : memref<64xf32, (d0) -> (d0)>

  %0 = alloc() : memref<8x16x4xf32, (d0, d1, d2) -> (d0 * 64 + d1 * 4 + d2)>
  // CHECK: std.subview %0[%c0, %c0, %c0][%arg0, %arg1, %arg2][%c1, %c1, %c1] : memref<8x16x4xf32, #[[BASE_MAP0]]> to memref<?x?x?xf32, #[[SUBVIEW_MAP0]]>
  %1 = subview %0[%c0, %c0, %c0][%arg0, %arg1, %arg2][%c1, %c1, %c1]
    : memref<8x16x4xf32, (d0, d1, d2) -> (d0 * 64 + d1 * 4 + d2)> to
      memref<?x?x?xf32,
       (d0, d1, d2)[s0, s1, s2, s3] -> (d0 * s1 + d1 * s2 + d2 * s3 + s0)>

  %2 = alloc()[%arg2] : memref<64xf32, (d0)[s0] -> (d0 + s0)>
 // CHECK: std.subview %2[%c1][%arg0][%c1] : memref<64xf32, #[[BASE_MAP1]]> to memref<?xf32, #[[SUBVIEW_MAP1]]>
  %3 = subview %2[%c1][%arg0][%c1]
    : memref<64xf32, (d0)[s0] -> (d0 + s0)> to
      memref<?xf32, (d0)[s0, s1] -> (d0 * s1 + s0)>

  %4 = alloc() : memref<64x22xf32, (d0, d1) -> (d0 * 22 + d1)>
  // CHECK: std.subview %4[%c0, %c1][%arg0, %arg1][%c1, %c0] : memref<64x22xf32, #[[BASE_MAP2]]> to memref<?x?xf32, #[[SUBVIEW_MAP2]]>
  %5 = subview %4[%c0, %c1][%arg0, %arg1][%c1, %c0]
    : memref<64x22xf32, (d0, d1) -> (d0 * 22 + d1)> to
      memref<?x?xf32, (d0, d1)[s0, s1, s2] -> (d0 * s1 + d1 * s2 + s0)> 
  return
}

// CHECK-LABEL: func @test_dimop(%arg0
func @test_dimop(%arg0: tensor<4x4x?xf32>) {
  // CHECK: %0 = dim %arg0, 2 : tensor<4x4x?xf32>
  %0 = dim %arg0, 2 : tensor<4x4x?xf32>
  // use dim as an index to ensure type correctness
  %1 = affine.apply (d0) -> (d0)(%0)
  return
}

// CHECK-LABEL: func @test_splat_op
// CHECK-SAME: [[S:%arg[0-9]+]]: f32
func @test_splat_op(%s : f32) {
  %v = splat %s : vector<8xf32>
  // CHECK: splat [[S]] : vector<8xf32>
  %t = splat %s : tensor<8xf32>
  // CHECK: splat [[S]] : tensor<8xf32>
  %u = "std.splat"(%s) : (f32) -> vector<4xf32>
  // CHECK: splat [[S]] : vector<4xf32>
  return
}

// CHECK-LABEL: func @tensor_load_store
func @tensor_load_store(%0 : memref<4x4xi32>) {
  // CHECK: %[[TENSOR:.*]] = tensor_load %[[MEMREF:.*]] : memref<4x4xi32>
  %1 = tensor_load %0 : memref<4x4xi32>
  // CHECK: tensor_store %[[TENSOR]], %[[MEMREF]] : memref<4x4xi32>
  tensor_store %1, %0 : memref<4x4xi32>
  return
}
