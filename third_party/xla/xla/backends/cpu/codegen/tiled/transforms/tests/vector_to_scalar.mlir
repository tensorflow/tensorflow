// RUN: fusion_compiler_opt %s -xtile-cpu-vector-to-scalar -split-input-file | FileCheck %s

func.func @vector_to_scalar_0d(%arg0 : vector<f32>, %arg1 : vector<f32>) -> vector<f32> {
  // CHECK-DAG: %[[SCALAR0:.*]] = vector.extract %arg0[]
  // CHECK-DAG: %[[SCALAR1:.*]] = vector.extract %arg1[]
  // CHECK: %[[SCALAR_ADD:.*]] = arith.addf %[[SCALAR0]], %[[SCALAR1]] : f32
  // CHECK: %[[VECTOR_ADD:.*]] = vector.from_elements %[[SCALAR_ADD]] : vector<f32>
  %add = arith.addf %arg0, %arg1 : vector<f32>
  // CHECK: return %[[VECTOR_ADD]] : vector<f32>
  return %add : vector<f32>
}

//-----

func.func @vector_to_scalar_1d(%arg0 : vector<1xf32>, %arg1 : vector<1xf32>) -> vector<1xf32> {
  // CHECK-DAG: %[[SCALAR0:.*]] = vector.extract %arg0[0]
  // CHECK-DAG: %[[SCALAR1:.*]] = vector.extract %arg1[0]
  // CHECK: %[[SCALAR_MUL:.*]] = arith.mulf %[[SCALAR0]], %[[SCALAR1]] : f32
  // CHECK: %[[VECTOR_MUL:.*]] = vector.from_elements %[[SCALAR_MUL]] : vector<1xf32>
  %mul = arith.mulf %arg0, %arg1 : vector<1xf32>
  // CHECK: return %[[VECTOR_MUL]] : vector<1xf32>
  return %mul : vector<1xf32>
}

//-----

func.func @vector_to_scalar_2d(%arg0 : vector<1x1xf32>) -> vector<1x1xf32> {
  // CHECK: %[[SCALAR0:.*]] = vector.extract %arg0[0, 0]
  // CHECK: %[[SCALAR_COS:.*]] = math.cos %[[SCALAR0]] : f32
  // CHECK: %[[VECTOR_COS:.*]] = vector.from_elements %[[SCALAR_COS]] : vector<1x1xf32>
  %cos = math.cos %arg0 : vector<1x1xf32>
  // CHECK: return %[[VECTOR_COS]] : vector<1x1xf32>
  return %cos : vector<1x1xf32>
}

//-----

func.func @vector_to_scalar_constant() -> vector<1x1xf32> {
  // CHECK: %[[SCALAR:.*]] = arith.constant 1.000000e+00 : f32
  // CHECK: %[[VECTOR:.*]] = vector.from_elements %[[SCALAR]] : vector<1x1xf32>
  %cos = arith.constant dense<1.0> : vector<1x1xf32>
  // CHECK: return %[[VECTOR]] : vector<1x1xf32>
  return %cos : vector<1x1xf32>
}

//-----

func.func @skips_multi_element(%arg0 : vector<2xf32>) -> vector<2xf32> {
  // CHECK: %[[RES:.*]] = math.sin %arg0 : vector<2xf32>
  %sin = math.sin %arg0 : vector<2xf32>
  // CHECK: return %[[RES]] : vector<2xf32>
  return %sin : vector<2xf32>
}
