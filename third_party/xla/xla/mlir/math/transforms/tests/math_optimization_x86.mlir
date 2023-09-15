// REQUIRES: x86-registered-target

// RUN: xla-runtime-opt %s --xla-math-optimization \
// RUN:   | FileCheck %s

// RUN: xla-runtime-opt %s --xla-math-optimization=enable-avx2 \
// RUN:   | FileCheck --check-prefix=AVX2 %s

// We only approximate rsqrt for vectors and when the AVX2 option is enabled.
// CHECK-LABEL:   func @rsqrt_scalar
// AVX2-LABEL:    func @rsqrt_scalar
// CHECK:           math.rsqrt
// AVX2:            math.rsqrt
func.func @rsqrt_scalar(%arg0: f32) -> f32 {
  %0 = math.rsqrt %arg0 : f32
  func.return %0 : f32
}
// CHECK-LABEL:   func @rsqrt_vector_8xf32
// CHECK:           math.rsqrt
// AVX2-LABEL:    func @rsqrt_vector_8xf32(
// AVX2-SAME:       %[[VAL_0:.*]]: vector<8xf32>) -> vector<8xf32> {
// AVX2:   %[[VAL_1:.*]] = arith.constant dense<0x7F800000> : vector<8xf32>
// AVX2:   %[[VAL_2:.*]] = arith.constant dense<1.500000e+00> : vector<8xf32>
// AVX2:   %[[VAL_3:.*]] = arith.constant dense<-5.000000e-01> : vector<8xf32>
// AVX2:   %[[VAL_4:.*]] = arith.constant dense<1.17549435E-38> : vector<8xf32>
// AVX2:   %[[VAL_5:.*]] = arith.mulf %[[VAL_0]], %[[VAL_3]] : vector<8xf32>
// AVX2:   %[[VAL_6:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_4]] : vector<8xf32>
// AVX2:   %[[VAL_7:.*]] = arith.cmpf oeq, %[[VAL_0]], %[[VAL_1]] : vector<8xf32>
// AVX2:   %[[VAL_8:.*]] = arith.ori %[[VAL_6]], %[[VAL_7]] : vector<8xi1>
// AVX2:   %[[VAL_9:.*]] = x86vector.avx.rsqrt %[[VAL_0]] : vector<8xf32>
// AVX2:   %[[VAL_10:.*]] = arith.mulf %[[VAL_5]], %[[VAL_9]] : vector<8xf32>
// AVX2:   %[[VAL_11:.*]] = math.fma %[[VAL_9]], %[[VAL_10]], %[[VAL_2]] : vector<8xf32>
// AVX2:   %[[VAL_12:.*]] = arith.mulf %[[VAL_9]], %[[VAL_11]] : vector<8xf32>
// AVX2:   %[[VAL_13:.*]] = arith.select %[[VAL_8]], %[[VAL_9]], %[[VAL_12]] : vector<8xi1>, vector<8xf32>
// AVX2:   return %[[VAL_13]] : vector<8xf32>
// AVX2: }
func.func @rsqrt_vector_8xf32(%arg0: vector<8xf32>) -> vector<8xf32> {
  %0 = math.rsqrt %arg0 : vector<8xf32>
  func.return %0 : vector<8xf32>
}
// Virtual vector width is not a multiple of an AVX2 vector width.
//
// CHECK-LABEL:  func @rsqrt_vector_5xf32
// CHECK:          math.rsqrt
// AVX2-LABEL:   func @rsqrt_vector_5xf32
// AVX2:           math.rsqrt
func.func @rsqrt_vector_5xf32(%arg0: vector<5xf32>) -> vector<5xf32> {
  %0 = math.rsqrt %arg0 : vector<5xf32>
  func.return %0 : vector<5xf32>
}
// One dimensional virtual vector expanded and unrolled into multiple AVX2-sized
// vectors.
//
// CHECK-LABEL: func @rsqrt_vector_16xf32
// CHECK:         math.rsqrt
// AVX2-LABEL:  func @rsqrt_vector_16xf32(
// AVX2-SAME:     %[[ARG:.*]]: vector<16xf32>
// AVX2-SAME:   ) -> vector<16xf32>
// AVX2:          %[[INIT:.*]] = arith.constant dense<0.000000e+00> : vector<2x8xf32>
// AVX2:          %[[EXPAND:.*]] = vector.shape_cast %[[ARG]] : vector<16xf32> to vector<2x8xf32>
// AVX2:          %[[VEC0:.*]] = vector.extract %[[EXPAND]][0]
// AVX2:          %[[RSQRT0:.*]] = x86vector.avx.rsqrt %[[VEC0]]
// AVX2:          %[[VEC1:.*]] = vector.extract %[[EXPAND]][1]
// AVX2:          %[[RSQRT1:.*]] = x86vector.avx.rsqrt %[[VEC1]]
// AVX2:          %[[RESULT0:.*]] = vector.insert %[[RSQRT0]], %[[INIT]] [0]
// AVX2:          %[[RESULT1:.*]] = vector.insert %[[RSQRT1]], %[[RESULT0]] [1]
// AVX2:          %[[RSQRT:.*]] = vector.shape_cast %[[RESULT1]] : vector<2x8xf32> to vector<16xf32>
func.func @rsqrt_vector_16xf32(%arg0: vector<16xf32>) -> vector<16xf32> {
  %0 = math.rsqrt %arg0 : vector<16xf32>
  func.return %0 : vector<16xf32>
}
// Two dimensional virtual vector unrolled into multiple AVX2-sized vectors.
//
// CHECK-LABEL: func @rsqrt_vector_2x8xf32
// CHECK:         math.rsqrt
// AVX2-LABEL:  func @rsqrt_vector_2x8xf32(
// AVX2-SAME:     %[[ARG:.*]]: vector<2x8xf32>
// AVX2-SAME:   ) -> vector<2x8xf32>
// AVX2:          %[[INIT:.*]] = arith.constant dense<0.000000e+00> : vector<2x8xf32>
// AVX2-NOT:      vector.shape_cast
// AVX2:          %[[VEC0:.*]] = vector.extract %[[ARG]][0]
// AVX2:          %[[RSQRT0:.*]] = x86vector.avx.rsqrt %[[VEC0]]
// AVX2:          %[[VEC1:.*]] = vector.extract %[[ARG]][1]
// AVX2:          %[[RSQRT1:.*]] = x86vector.avx.rsqrt %[[VEC1]]
// AVX2:          %[[RESULT0:.*]] = vector.insert %[[RSQRT0]], %[[INIT]] [0]
// AVX2:          %[[RESULT1:.*]] = vector.insert %[[RSQRT1]], %[[RESULT0]] [1]
// AVX2-NOT:      vector.shape_cast
func.func @rsqrt_vector_2x8xf32(%arg0: vector<2x8xf32>) -> vector<2x8xf32> {
  %0 = math.rsqrt %arg0 : vector<2x8xf32>
  func.return %0 : vector<2x8xf32>
}
// Two dimensional virtual vector expanded and unrolled into multiple AVX2-sized
// vectors.
//
// CHECK-LABEL: func @rsqrt_vector_2x16xf32
// CHECK:         math.rsqrt
// AVX2-LABEL:  func @rsqrt_vector_2x16xf32(
// AVX2-SAME:     %[[ARG:.*]]: vector<2x16xf32>
// AVX2-SAME:   ) -> vector<2x16xf32>
// AVX2:          %[[INIT:.*]] = arith.constant dense<0.000000e+00> : vector<2x2x8xf32>
// AVX2:          %[[EXPAND:.*]] = vector.shape_cast %[[ARG]] : vector<2x16xf32> to vector<2x2x8xf32>
// AVX2:          %[[VEC00:.*]] = vector.extract %[[EXPAND]][0, 0]
// AVX2:          %[[RSQRT00:.*]] = x86vector.avx.rsqrt %[[VEC00]]
// AVX2:          %[[VEC01:.*]] = vector.extract %[[EXPAND]][0, 1]
// AVX2:          %[[RSQRT01:.*]] = x86vector.avx.rsqrt %[[VEC01]]
// AVX2:          %[[VEC10:.*]] = vector.extract %[[EXPAND]][1, 0]
// AVX2:          %[[RSQRT10:.*]] = x86vector.avx.rsqrt %[[VEC10]]
// AVX2:          %[[VEC11:.*]] = vector.extract %[[EXPAND]][1, 1]
// AVX2:          %[[RSQRT11:.*]] = x86vector.avx.rsqrt %[[VEC11]]
// AVX2:          %[[RESULT0:.*]] = vector.insert %[[RSQRT00]], %[[INIT]] [0, 0]
// AVX2:          %[[RESULT1:.*]] = vector.insert %[[RSQRT01]], %[[RESULT0]] [0, 1]
// AVX2:          %[[RESULT2:.*]] = vector.insert %[[RSQRT10]], %[[RESULT1]] [1, 0]
// AVX2:          %[[RESULT3:.*]] = vector.insert %[[RSQRT11]], %[[RESULT2]] [1, 1]
// AVX2:          %[[RSQRT:.*]] = vector.shape_cast %[[RESULT3]] : vector<2x2x8xf32> to vector<2x16xf32>
func.func @rsqrt_vector_2x16xf32(%arg0: vector<2x16xf32>) -> vector<2x16xf32> {
  %0 = math.rsqrt %arg0 : vector<2x16xf32>
  func.return %0 : vector<2x16xf32>
}
