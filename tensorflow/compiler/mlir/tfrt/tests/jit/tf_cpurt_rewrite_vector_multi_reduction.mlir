// Copyright 2022 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: tf-tfrt-opt %s --tf-cpurt-rewrite-vector-multi-reduction | FileCheck %s

func @vector_multi_reduction(%arg0: vector<2x4xf32>) -> vector<2xf32> {
    %0 = vector.multi_reduction <mul>, %arg0 [1] : vector<2x4xf32> to vector<2xf32>
    return %0 : vector<2xf32>
}
// CHECK-LABEL: func @vector_multi_reduction
//  CHECK-SAME:   %[[INPUT:.+]]: vector<2x4xf32>
//       CHECK:       %[[RESULT_VEC_0:.+]] = arith.constant dense<{{.*}}> : vector<2xf32>
//       CHECK:       %[[C0:.+]] = arith.constant 0 : index
//       CHECK:       %[[C1:.+]] = arith.constant 1 : index
//       CHECK:       %[[V0:.+]] = vector.extract %[[INPUT]][0]
//       CHECK:       %[[RV0:.+]] = vector.reduction "mul", %[[V0]] : vector<4xf32> into f32
//       CHECK:       %[[RESULT_VEC_1:.+]] = vector.insertelement %[[RV0:.+]], %[[RESULT_VEC_0]][%[[C0]] : index] : vector<2xf32>
//       CHECK:       %[[V1:.+]] = vector.extract %[[INPUT]][1]
//       CHECK:       %[[RV1:.+]] = vector.reduction "mul", %[[V1]] : vector<4xf32> into f32
//       CHECK:       %[[RESULT_VEC:.+]] = vector.insertelement %[[RV1:.+]], %[[RESULT_VEC_1]][%[[C1]] : index] : vector<2xf32>
//       CHECK:       return %[[RESULT_VEC]]

func @vector_multi_reduction_to_scalar(%arg0: vector<2x4xf32>) -> f32 {
    %0 = vector.multi_reduction <mul>, %arg0 [0, 1] : vector<2x4xf32> to f32
    return %0 : f32
}
// CHECK-LABEL: func @vector_multi_reduction_to_scalar
//  CHECK-SAME:   %[[INPUT:.+]]: vector<2x4xf32>
//       CHECK:   %[[CASTED:.*]] = vector.shape_cast %[[INPUT]] : vector<2x4xf32> to vector<8xf32>
//       CHECK:   %[[REDUCED:.*]] = vector.reduction "mul", %[[CASTED]] : vector<8xf32> into f32
//       CHECK:   %[[INSERTED:.*]] = vector.insertelement %[[REDUCED]], {{.*}} : vector<1xf32>
//       CHECK:   %[[RES:.*]] = vector.extract %[[INSERTED]][0] : vector<1xf32>
//       CHECK:   return %[[RES]]

func @vector_reduction_inner(%arg0: vector<2x3x4x5xi32>) -> vector<2x3xi32> {
    %0 = vector.multi_reduction <add>, %arg0 [2, 3] : vector<2x3x4x5xi32> to vector<2x3xi32>
    return %0 : vector<2x3xi32>
}
// CHECK-LABEL: func @vector_reduction_inner
//  CHECK-SAME:   %[[INPUT:.+]]: vector<2x3x4x5xi32>
//       CHECK:       %[[FLAT_RESULT_VEC_0:.+]] = arith.constant dense<0> : vector<6xi32>
//   CHECK-DAG:       %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:       %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:       %[[C2:.+]] = arith.constant 2 : index
//   CHECK-DAG:       %[[C3:.+]] = arith.constant 3 : index
//   CHECK-DAG:       %[[C4:.+]] = arith.constant 4 : index
//   CHECK-DAG:       %[[C5:.+]] = arith.constant 5 : index
//       CHECK:       %[[RESHAPED_INPUT:.+]] = vector.shape_cast %[[INPUT]] : vector<2x3x4x5xi32> to vector<6x20xi32>
//       CHECK:       %[[V0:.+]] = vector.extract %[[RESHAPED_INPUT]][0] : vector<6x20xi32>
//       CHECK:       %[[V0R:.+]] = vector.reduction "add", %[[V0]] : vector<20xi32> into i32
//       CHECK:       %[[FLAT_RESULT_VEC_1:.+]] = vector.insertelement %[[V0R]], %[[FLAT_RESULT_VEC_0]][%[[C0]] : index] : vector<6xi32>
//       CHECK:       %[[V1:.+]] = vector.extract %[[RESHAPED_INPUT]][1] : vector<6x20xi32>
//       CHECK:       %[[V1R:.+]] = vector.reduction "add", %[[V1]] : vector<20xi32> into i32
//       CHECK:       %[[FLAT_RESULT_VEC_2:.+]] = vector.insertelement %[[V1R]], %[[FLAT_RESULT_VEC_1]][%[[C1]] : index] : vector<6xi32>
//       CHECK:       %[[V2:.+]] = vector.extract %[[RESHAPED_INPUT]][2] : vector<6x20xi32>
//       CHECK:       %[[V2R:.+]] = vector.reduction "add", %[[V2]] : vector<20xi32> into i32
//       CHECK:       %[[FLAT_RESULT_VEC_3:.+]] = vector.insertelement %[[V2R]], %[[FLAT_RESULT_VEC_2]][%[[C2]] : index] : vector<6xi32>
//       CHECK:       %[[V3:.+]] = vector.extract %[[RESHAPED_INPUT]][3] : vector<6x20xi32>
//       CHECK:       %[[V3R:.+]] = vector.reduction "add", %[[V3]] : vector<20xi32> into i32
//       CHECK:       %[[FLAT_RESULT_VEC_4:.+]] = vector.insertelement %[[V3R]], %[[FLAT_RESULT_VEC_3]][%[[C3]] : index] : vector<6xi32>
//       CHECK:       %[[V4:.+]] = vector.extract %[[RESHAPED_INPUT]][4] : vector<6x20xi32>
//       CHECK:       %[[V4R:.+]] = vector.reduction "add", %[[V4]] : vector<20xi32> into i32
//       CHECK:       %[[FLAT_RESULT_VEC_5:.+]] = vector.insertelement %[[V4R]], %[[FLAT_RESULT_VEC_4]][%[[C4]] : index] : vector<6xi32>
///       CHECK:      %[[V5:.+]] = vector.extract %[[RESHAPED_INPUT]][5] : vector<6x20xi32>
//       CHECK:       %[[V5R:.+]] = vector.reduction "add", %[[V5]] : vector<20xi32> into i32
//       CHECK:       %[[FLAT_RESULT_VEC:.+]] = vector.insertelement %[[V5R]], %[[FLAT_RESULT_VEC_5]][%[[C5]] : index] : vector<6xi32>
//       CHECK:       %[[RESULT:.+]] = vector.shape_cast %[[FLAT_RESULT_VEC]] : vector<6xi32> to vector<2x3xi32>
//       CHECK:       return %[[RESULT]]


func @vector_multi_reduction_transposed(%arg0: vector<2x3x4x5xf32>) -> vector<2x5xf32> {
    %0 = vector.multi_reduction <add>, %arg0 [1, 2] : vector<2x3x4x5xf32> to vector<2x5xf32>
    return %0 : vector<2x5xf32>
}

// CHECK-LABEL: func @vector_multi_reduction_transposed
//  CHECK-SAME:    %[[INPUT:.+]]: vector<2x3x4x5xf32>
//       CHECK:     %[[TRANSPOSED_INPUT:.+]] = vector.transpose %[[INPUT]], [0, 3, 1, 2] : vector<2x3x4x5xf32> to vector<2x5x3x4xf32>
//       CHECK:     vector.shape_cast %[[TRANSPOSED_INPUT]] : vector<2x5x3x4xf32> to vector<10x12xf32>
//       CHECK:     %[[RESULT:.+]] = vector.shape_cast %{{.*}} : vector<10xf32> to vector<2x5xf32>
//       CHECK:       return %[[RESULT]]

func @vector_multi_reduction_ordering(%arg0: vector<3x2x4xf32>) -> vector<2x4xf32> {
    %0 = vector.multi_reduction <mul>, %arg0 [0] : vector<3x2x4xf32> to vector<2x4xf32>
    return %0 : vector<2x4xf32>
}
// CHECK-LABEL: func @vector_multi_reduction_ordering
//  CHECK-SAME:   %[[INPUT:.+]]: vector<3x2x4xf32>
//       CHECK:       %[[RESULT_VEC_0:.+]] = arith.constant dense<{{.*}}> : vector<8xf32>
//       CHECK:       %[[C0:.+]] = arith.constant 0 : index
//       CHECK:       %[[C1:.+]] = arith.constant 1 : index
//       CHECK:       %[[C2:.+]] = arith.constant 2 : index
//       CHECK:       %[[C3:.+]] = arith.constant 3 : index
//       CHECK:       %[[C4:.+]] = arith.constant 4 : index
//       CHECK:       %[[C5:.+]] = arith.constant 5 : index
//       CHECK:       %[[C6:.+]] = arith.constant 6 : index
//       CHECK:       %[[C7:.+]] = arith.constant 7 : index
//       CHECK:       %[[TRANSPOSED_INPUT:.+]] = vector.transpose %[[INPUT]], [1, 2, 0] : vector<3x2x4xf32> to vector<2x4x3xf32>
//       CHECK:       %[[V0:.+]] = vector.extract %[[TRANSPOSED_INPUT]][0, 0]
//       CHECK:       %[[RV0:.+]] = vector.reduction "mul", %[[V0]] : vector<3xf32> into f32
//       CHECK:       %[[RESULT_VEC_1:.+]] = vector.insertelement %[[RV0:.+]], %[[RESULT_VEC_0]][%[[C0]] : index] : vector<8xf32>
//       CHECK:       %[[V1:.+]] = vector.extract %[[TRANSPOSED_INPUT]][0, 1]
//       CHECK:       %[[RV1:.+]] = vector.reduction "mul", %[[V1]] : vector<3xf32> into f32
//       CHECK:       %[[RESULT_VEC_2:.+]] = vector.insertelement %[[RV1:.+]], %[[RESULT_VEC_1]][%[[C1]] : index] : vector<8xf32>
//       CHECK:       %[[V2:.+]] = vector.extract %[[TRANSPOSED_INPUT]][0, 2]
//       CHECK:       %[[RV2:.+]] = vector.reduction "mul", %[[V2]] : vector<3xf32> into f32
//       CHECK:       %[[RESULT_VEC_3:.+]] = vector.insertelement %[[RV2:.+]], %[[RESULT_VEC_2]][%[[C2]] : index] : vector<8xf32>
//       CHECK:       %[[V3:.+]] = vector.extract %[[TRANSPOSED_INPUT]][0, 3]
//       CHECK:       %[[RV3:.+]] = vector.reduction "mul", %[[V3]] : vector<3xf32> into f32
//       CHECK:       %[[RESULT_VEC_4:.+]] = vector.insertelement %[[RV3:.+]], %[[RESULT_VEC_3]][%[[C3]] : index] : vector<8xf32>
//       CHECK:       %[[V4:.+]] = vector.extract %[[TRANSPOSED_INPUT]][1, 0]
//       CHECK:       %[[RV4:.+]] = vector.reduction "mul", %[[V4]] : vector<3xf32> into f32
//       CHECK:       %[[RESULT_VEC_5:.+]] = vector.insertelement %[[RV4:.+]], %[[RESULT_VEC_4]][%[[C4]] : index] : vector<8xf32>
//       CHECK:       %[[V5:.+]] = vector.extract %[[TRANSPOSED_INPUT]][1, 1]
//       CHECK:       %[[RV5:.+]] = vector.reduction "mul", %[[V5]] : vector<3xf32> into f32
//       CHECK:       %[[RESULT_VEC_6:.+]] = vector.insertelement %[[RV5:.+]], %[[RESULT_VEC_5]][%[[C5]] : index] : vector<8xf32>
//       CHECK:       %[[V6:.+]] = vector.extract %[[TRANSPOSED_INPUT]][1, 2]
//       CHECK:       %[[RV6:.+]] = vector.reduction "mul", %[[V6]] : vector<3xf32> into f32
//       CHECK:       %[[RESULT_VEC_7:.+]] = vector.insertelement %[[RV6:.+]], %[[RESULT_VEC_6]][%[[C6]] : index] : vector<8xf32>
//       CHECK:       %[[V7:.+]] = vector.extract %[[TRANSPOSED_INPUT]][1, 3]
//       CHECK:       %[[RV7:.+]] = vector.reduction "mul", %[[V7]] : vector<3xf32> into f32
//       CHECK:       %[[RESULT_VEC:.+]] = vector.insertelement %[[RV7:.+]], %[[RESULT_VEC_7]][%[[C7]] : index] : vector<8xf32>
//       CHECK:       %[[RESHAPED_VEC:.+]] = vector.shape_cast %[[RESULT_VEC]] : vector<8xf32> to vector<2x4xf32>
//       CHECK:       return %[[RESHAPED_VEC]]
