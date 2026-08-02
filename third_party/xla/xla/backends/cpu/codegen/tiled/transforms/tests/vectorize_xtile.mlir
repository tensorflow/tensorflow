// RUN: fusion_compiler_opt --xtile-cpu-vectorize-xtile %s | FileCheck %s

// CHECK: #indexing_map = #xla.indexing_map<"(d0) -> (-d0 + 128 - 8){{.*}}">

func.func @test_addf(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<8xf32> {
  %0 = arith.addf %arg0, %arg1 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_addf
// CHECK: arith.addf %{{.*}}, %{{.*}} : vector<8xf32>

func.func @test_subf(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<8xf32> {
  %0 = arith.subf %arg0, %arg1 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_subf
// CHECK: arith.subf %{{.*}}, %{{.*}} : vector<8xf32>

func.func @test_mulf(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<8xf32> {
  %0 = arith.mulf %arg0, %arg1 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_mulf
// CHECK: arith.mulf %{{.*}}, %{{.*}} : vector<8xf32>

func.func @test_divf(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<8xf32> {
  %0 = arith.divf %arg0, %arg1 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_divf
// CHECK: arith.divf %{{.*}}, %{{.*}} : vector<8xf32>

func.func @test_remf(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<8xf32> {
  %0 = arith.remf %arg0, %arg1 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_remf
// CHECK: arith.remf %{{.*}}, %{{.*}} : vector<8xf32>

func.func @test_maximumf(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<8xf32> {
  %0 = arith.maximumf %arg0, %arg1 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_maximumf
// CHECK: arith.maximumf %{{.*}}, %{{.*}} : vector<8xf32>

func.func @test_minimumf(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<8xf32> {
  %0 = arith.minimumf %arg0, %arg1 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_minimumf
// CHECK: arith.minimumf %{{.*}}, %{{.*}} : vector<8xf32>

func.func @test_negf(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = arith.negf %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_negf
// CHECK: arith.negf %{{.*}} : vector<8xf32>

func.func @test_addi(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
  %0 = arith.addi %arg0, %arg1 : tensor<8xi32>
  return %0 : tensor<8xi32>
}
// CHECK-LABEL: @test_addi
// CHECK: arith.addi %{{.*}}, %{{.*}} : vector<8xi32>

func.func @test_subi(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
  %0 = arith.subi %arg0, %arg1 : tensor<8xi32>
  return %0 : tensor<8xi32>
}
// CHECK-LABEL: @test_subi
// CHECK: arith.subi %{{.*}}, %{{.*}} : vector<8xi32>

func.func @test_muli(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
  %0 = arith.muli %arg0, %arg1 : tensor<8xi32>
  return %0 : tensor<8xi32>
}
// CHECK-LABEL: @test_muli
// CHECK: arith.muli %{{.*}}, %{{.*}} : vector<8xi32>

func.func @test_divsi(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
  %0 = arith.divsi %arg0, %arg1 : tensor<8xi32>
  return %0 : tensor<8xi32>
}
// CHECK-LABEL: @test_divsi
// CHECK: arith.divsi %{{.*}}, %{{.*}} : vector<8xi32>

func.func @test_divui(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
  %0 = arith.divui %arg0, %arg1 : tensor<8xi32>
  return %0 : tensor<8xi32>
}
// CHECK-LABEL: @test_divui
// CHECK: arith.divui %{{.*}}, %{{.*}} : vector<8xi32>

func.func @test_remsi(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
  %0 = arith.remsi %arg0, %arg1 : tensor<8xi32>
  return %0 : tensor<8xi32>
}
// CHECK-LABEL: @test_remsi
// CHECK: arith.remsi %{{.*}}, %{{.*}} : vector<8xi32>

func.func @test_remui(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
  %0 = arith.remui %arg0, %arg1 : tensor<8xi32>
  return %0 : tensor<8xi32>
}
// CHECK-LABEL: @test_remui
// CHECK: arith.remui %{{.*}}, %{{.*}} : vector<8xi32>

func.func @test_maxsi(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
  %0 = arith.maxsi %arg0, %arg1 : tensor<8xi32>
  return %0 : tensor<8xi32>
}
// CHECK-LABEL: @test_maxsi
// CHECK: arith.maxsi %{{.*}}, %{{.*}} : vector<8xi32>

func.func @test_maxui(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
  %0 = arith.maxui %arg0, %arg1 : tensor<8xi32>
  return %0 : tensor<8xi32>
}
// CHECK-LABEL: @test_maxui
// CHECK: arith.maxui %{{.*}}, %{{.*}} : vector<8xi32>

func.func @test_minsi(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
  %0 = arith.minsi %arg0, %arg1 : tensor<8xi32>
  return %0 : tensor<8xi32>
}
// CHECK-LABEL: @test_minsi
// CHECK: arith.minsi %{{.*}}, %{{.*}} : vector<8xi32>

func.func @test_minui(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
  %0 = arith.minui %arg0, %arg1 : tensor<8xi32>
  return %0 : tensor<8xi32>
}
// CHECK-LABEL: @test_minui
// CHECK: arith.minui %{{.*}}, %{{.*}} : vector<8xi32>

func.func @test_andi(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
  %0 = arith.andi %arg0, %arg1 : tensor<8xi32>
  return %0 : tensor<8xi32>
}
// CHECK-LABEL: @test_andi
// CHECK: arith.andi %{{.*}}, %{{.*}} : vector<8xi32>

func.func @test_ori(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
  %0 = arith.ori %arg0, %arg1 : tensor<8xi32>
  return %0 : tensor<8xi32>
}
// CHECK-LABEL: @test_ori
// CHECK: arith.ori %{{.*}}, %{{.*}} : vector<8xi32>

func.func @test_xori(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
  %0 = arith.xori %arg0, %arg1 : tensor<8xi32>
  return %0 : tensor<8xi32>
}
// CHECK-LABEL: @test_xori
// CHECK: arith.xori %{{.*}}, %{{.*}} : vector<8xi32>

func.func @test_select(%arg0: tensor<8xi1>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>) -> tensor<8xf32> {
  %0 = arith.select %arg0, %arg1, %arg2 : tensor<8xi1>, tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_select
// CHECK: arith.select %{{.*}}, %{{.*}}, %{{.*}} : vector<8xi1>, vector<8xf32>

func.func @test_cmpf(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<8xi1> {
  %0 = arith.cmpf oeq, %arg0, %arg1 : tensor<8xf32>
  return %0 : tensor<8xi1>
}
// CHECK-LABEL: @test_cmpf
// CHECK: arith.cmpf {{.*}}, %{{.*}}, %{{.*}} : vector<8xf32>

func.func @test_cmpi(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi1> {
  %0 = arith.cmpi eq, %arg0, %arg1 : tensor<8xi32>
  return %0 : tensor<8xi1>
}
// CHECK-LABEL: @test_cmpi
// CHECK: arith.cmpi {{.*}}, %{{.*}}, %{{.*}} : vector<8xi32>

func.func @test_extf(%arg0: tensor<8xf16>) -> tensor<8xf32> {
  %0 = arith.extf %arg0 : tensor<8xf16> to tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_extf
// CHECK: arith.extf %{{.*}} : vector<8xf16> to vector<8xf32>

func.func @test_truncf(%arg0: tensor<8xf32>) -> tensor<8xf16> {
  %0 = arith.truncf %arg0 : tensor<8xf32> to tensor<8xf16>
  return %0 : tensor<8xf16>
}
// CHECK-LABEL: @test_truncf
// CHECK: arith.truncf %{{.*}} : vector<8xf32> to vector<8xf16>

func.func @test_extsi(%arg0: tensor<8xi16>) -> tensor<8xi32> {
  %0 = arith.extsi %arg0 : tensor<8xi16> to tensor<8xi32>
  return %0 : tensor<8xi32>
}
// CHECK-LABEL: @test_extsi
// CHECK: arith.extsi %{{.*}} : vector<8xi16> to vector<8xi32>

func.func @test_extui(%arg0: tensor<8xi16>) -> tensor<8xi32> {
  %0 = arith.extui %arg0 : tensor<8xi16> to tensor<8xi32>
  return %0 : tensor<8xi32>
}
// CHECK-LABEL: @test_extui
// CHECK: arith.extui %{{.*}} : vector<8xi16> to vector<8xi32>

func.func @test_fptosi(%arg0: tensor<8xf32>) -> tensor<8xi32> {
  %0 = arith.fptosi %arg0 : tensor<8xf32> to tensor<8xi32>
  return %0 : tensor<8xi32>
}
// CHECK-LABEL: @test_fptosi
// CHECK: arith.fptosi %{{.*}} : vector<8xf32> to vector<8xi32>

func.func @test_fptoui(%arg0: tensor<8xf32>) -> tensor<8xi32> {
  %0 = arith.fptoui %arg0 : tensor<8xf32> to tensor<8xi32>
  return %0 : tensor<8xi32>
}
// CHECK-LABEL: @test_fptoui
// CHECK: arith.fptoui %{{.*}} : vector<8xf32> to vector<8xi32>

func.func @test_sitofp(%arg0: tensor<8xi32>) -> tensor<8xf32> {
  %0 = arith.sitofp %arg0 : tensor<8xi32> to tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_sitofp
// CHECK: arith.sitofp %{{.*}} : vector<8xi32> to vector<8xf32>

func.func @test_uitofp(%arg0: tensor<8xi32>) -> tensor<8xf32> {
  %0 = arith.uitofp %arg0 : tensor<8xi32> to tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_uitofp
// CHECK: arith.uitofp %{{.*}} : vector<8xi32> to vector<8xf32>

func.func @test_trunci(%arg0: tensor<8xi32>) -> tensor<8xi16> {
  %0 = arith.trunci %arg0 : tensor<8xi32> to tensor<8xi16>
  return %0 : tensor<8xi16>
}
// CHECK-LABEL: @test_trunci
// CHECK: arith.trunci %{{.*}} : vector<8xi32> to vector<8xi16>

func.func @test_index_cast(%arg0: tensor<8xindex>) -> tensor<8xi32> {
  %0 = arith.index_cast %arg0 : tensor<8xindex> to tensor<8xi32>
  return %0 : tensor<8xi32>
}
// CHECK-LABEL: @test_index_cast
// CHECK: arith.index_cast %{{.*}} : vector<8xindex> to vector<8xi32>

func.func @test_absf(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = math.absf %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_absf
// CHECK: math.absf %{{.*}} : vector<8xf32>

func.func @test_ceil(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = math.ceil %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_ceil
// CHECK: math.ceil %{{.*}} : vector<8xf32>

func.func @test_floor(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = math.floor %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_floor
// CHECK: math.floor %{{.*}} : vector<8xf32>

func.func @test_roundeven(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = math.roundeven %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_roundeven
// CHECK: math.roundeven %{{.*}} : vector<8xf32>

func.func @test_acos(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = math.acos %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_acos
// CHECK: math.acos %{{.*}} : vector<8xf32>

func.func @test_acosh(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = math.acosh %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_acosh
// CHECK: math.acosh %{{.*}} : vector<8xf32>

func.func @test_asin(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = math.asin %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_asin
// CHECK: math.asin %{{.*}} : vector<8xf32>

func.func @test_asinh(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = math.asinh %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_asinh
// CHECK: math.asinh %{{.*}} : vector<8xf32>

func.func @test_atanh(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = math.atanh %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_atanh
// CHECK: math.atanh %{{.*}} : vector<8xf32>

func.func @test_cos(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = math.cos %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_cos
// CHECK: math.cos %{{.*}} : vector<8xf32>

func.func @test_cosh(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = math.cosh %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_cosh
// CHECK: math.cosh %{{.*}} : vector<8xf32>

func.func @test_exp(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = math.exp %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_exp
// CHECK: math.exp %{{.*}} : vector<8xf32>

func.func @test_erf(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = math.erf %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_erf
// CHECK: math.erf %{{.*}} : vector<8xf32>

func.func @test_expm1(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = math.expm1 %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_expm1
// CHECK: math.expm1 %{{.*}} : vector<8xf32>

func.func @test_log(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = math.log %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_log
// CHECK: math.log %{{.*}} : vector<8xf32>

func.func @test_log1p(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = math.log1p %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_log1p
// CHECK: math.log1p %{{.*}} : vector<8xf32>

func.func @test_rsqrt(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = math.rsqrt %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_rsqrt
// CHECK: math.rsqrt %{{.*}} : vector<8xf32>

func.func @test_sin(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = math.sin %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_sin
// CHECK: math.sin %{{.*}} : vector<8xf32>

func.func @test_sinh(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = math.sinh %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_sinh
// CHECK: math.sinh %{{.*}} : vector<8xf32>

func.func @test_sqrt(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = math.sqrt %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_sqrt
// CHECK: math.sqrt %{{.*}} : vector<8xf32>

func.func @test_tan(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = math.tan %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_tan
// CHECK: math.tan %{{.*}} : vector<8xf32>

func.func @test_tanh(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = math.tanh %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_tanh
// CHECK: math.tanh %{{.*}} : vector<8xf32>

func.func @test_cbrt(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = math.cbrt %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_cbrt
// CHECK: math.cbrt %{{.*}} : vector<8xf32>

func.func @test_absi(%arg0: tensor<8xi32>) -> tensor<8xi32> {
  %0 = math.absi %arg0 : tensor<8xi32>
  return %0 : tensor<8xi32>
}
// CHECK-LABEL: @test_absi
// CHECK: math.absi %{{.*}} : vector<8xi32>

func.func @test_atan2(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<8xf32> {
  %0 = math.atan2 %arg0, %arg1 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_atan2
// CHECK: math.atan2 %{{.*}}, %{{.*}} : vector<8xf32>

func.func @test_powf(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<8xf32> {
  %0 = math.powf %arg0, %arg1 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_powf
// CHECK: math.powf %{{.*}}, %{{.*}} : vector<8xf32>

func.func @test_ipowi(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
  %0 = math.ipowi %arg0, %arg1 : tensor<8xi32>
  return %0 : tensor<8xi32>
}
// CHECK-LABEL: @test_ipowi
// CHECK: math.ipowi %{{.*}}, %{{.*}} : vector<8xi32>

func.func @test_isfinite(%arg0: tensor<8xf32>) -> tensor<8xi1> {
  %0 = math.isfinite %arg0 : tensor<8xf32>
  return %0 : tensor<8xi1>
}
// CHECK-LABEL: @test_isfinite
// CHECK: math.isfinite %{{.*}} : vector<8xf32>

func.func @test_extract_aligned(%arg0: memref<128xf32>, %arg1: index) -> tensor<8xf32> {
  %c0 = arith.constant 0 : index
  %0 = xtile.extract %arg0[%c0] [8] [1] : memref<128xf32> -> tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_extract_aligned
// CHECK: %[[INDEXING:[^:]+]] = xla.apply_indexing #indexing_map(%{{.*}})
// CHECK: %[[COND:.*]] = arith.cmpi sge, %[[INDEXING]], %{{.*}} : index
// CHECK: %[[PAD:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[RES:.*]] = scf.if %[[COND]] -> (vector<8xf32>) {
// CHECK:   %[[READ:.*]] = vector.transfer_read %{{.*}}[%{{.*}}], %[[PAD]] {in_bounds = [true]} : memref<128xf32>, vector<8xf32>
// CHECK:   scf.yield %[[READ]] : vector<8xf32>
// CHECK: } else {
// CHECK:   %[[MASK_DIM:.*]] = arith.maxsi
// CHECK:   %[[MASK:.*]] = vector.create_mask %[[MASK_DIM]] : vector<8xi1>
// CHECK:   %[[MASKED_READ:.*]] = vector.transfer_read %{{.*}}[%{{.*}}], %[[PAD]], %[[MASK]] : memref<128xf32>, vector<8xf32>
// CHECK:   scf.yield %[[MASKED_READ]] : vector<8xf32>
// CHECK: }

func.func @test_extract_unaligned(%arg0: memref<128xf32>, %arg1: index) -> tensor<8xf32> {
  %0 = xtile.extract %arg0[%arg1] [8] [1] : memref<128xf32> -> tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_extract_unaligned
// CHECK: %[[INDEXING:[^:]+]] = xla.apply_indexing #indexing_map(%{{.*}})
// CHECK: %[[COND:.*]] = arith.cmpi sge, %[[INDEXING]], %{{.*}} : index
// CHECK: %[[PAD:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[RES:.*]] = scf.if %[[COND]] -> (vector<8xf32>) {
// CHECK:   %[[READ:.*]] = vector.transfer_read %{{.*}}[%{{.*}}], %[[PAD]] {in_bounds = [true]} : memref<128xf32>, vector<8xf32>
// CHECK:   scf.yield %[[READ]] : vector<8xf32>
// CHECK: } else {
// CHECK:   %[[MASK_DIM:.*]] = arith.maxsi
// CHECK:   %[[MASK:.*]] = vector.create_mask %[[MASK_DIM]] : vector<8xi1>
// CHECK:   %[[MASKED_READ:.*]] = vector.transfer_read %{{.*}}[%{{.*}}], %[[PAD]], %[[MASK]] : memref<128xf32>, vector<8xf32>
// CHECK:   scf.yield %[[MASKED_READ]] : vector<8xf32>
// CHECK: }

func.func @test_insert_aligned(%arg0: tensor<8xf32>, %arg1: memref<128xf32>) {
  %c0 = arith.constant 0 : index
  xtile.insert %arg0 into %arg1[%c0] [8] [1] : tensor<8xf32> -> memref<128xf32>
  return
}
// CHECK-LABEL: @test_insert_aligned
// CHECK: %[[INDEXING:[^:]+]] = xla.apply_indexing #indexing_map(%{{.*}})
// CHECK: %[[COND:.*]] = arith.cmpi sge, %[[INDEXING]], %{{.*}} : index
// CHECK: scf.if %[[COND]] {
// CHECK:   vector.transfer_write %{{.*}}, %{{.*}}[%{{.*}}] {in_bounds = [true]} : vector<8xf32>, memref<128xf32>
// CHECK: } else {
// CHECK:   %[[MASK_DIM:.*]] = arith.maxsi
// CHECK:   %[[MASK:.*]] = vector.create_mask %[[MASK_DIM]] : vector<8xi1>
// CHECK:   vector.transfer_write %{{.*}}, %{{.*}}[%{{.*}}], %[[MASK]] : vector<8xf32>, memref<128xf32>
// CHECK: }

func.func @test_insert_unaligned(%arg0: tensor<8xf32>, %arg1: memref<128xf32>, %arg2: index) {
  xtile.insert %arg0 into %arg1[%arg2] [8] [1] : tensor<8xf32> -> memref<128xf32>
  return
}
// CHECK-LABEL: @test_insert_unaligned
// CHECK: %[[INDEXING:[^:]+]] = xla.apply_indexing #indexing_map(%{{.*}})
// CHECK: %[[COND:.*]] = arith.cmpi sge, %[[INDEXING]], %{{.*}} : index
// CHECK: scf.if %[[COND]] {
// CHECK:   vector.transfer_write %{{.*}}, %{{.*}}[%{{.*}}] {in_bounds = [true]} : vector<8xf32>, memref<128xf32>
// CHECK: } else {
// CHECK:   %[[MASK_DIM:.*]] = arith.maxsi
// CHECK:   %[[MASK:.*]] = vector.create_mask %[[MASK_DIM]] : vector<8xi1>
// CHECK:   vector.transfer_write %{{.*}}, %{{.*}}[%{{.*}}], %[[MASK]] : vector<8xf32>, memref<128xf32>
// CHECK: }

