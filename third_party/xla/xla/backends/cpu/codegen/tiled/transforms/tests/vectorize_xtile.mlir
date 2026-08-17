// RUN: fusion_compiler_opt %s --xtile-cpu-vectorize-xtile --split-input-file | FileCheck %s

func.func @test_addf(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<8xf32> {
  %0 = arith.addf %arg0, %arg1 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_addf
// CHECK: arith.addf %{{.*}}, %{{.*}} : vector<8xf32>

// -----

func.func @test_subf(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<8xf32> {
  %0 = arith.subf %arg0, %arg1 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_subf
// CHECK: arith.subf %{{.*}}, %{{.*}} : vector<8xf32>

// -----

func.func @test_mulf(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<8xf32> {
  %0 = arith.mulf %arg0, %arg1 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_mulf
// CHECK: arith.mulf %{{.*}}, %{{.*}} : vector<8xf32>

// -----

func.func @test_divf(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<8xf32> {
  %0 = arith.divf %arg0, %arg1 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_divf
// CHECK: arith.divf %{{.*}}, %{{.*}} : vector<8xf32>

// -----

func.func @test_remf(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<8xf32> {
  %0 = arith.remf %arg0, %arg1 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_remf
// CHECK: arith.remf %{{.*}}, %{{.*}} : vector<8xf32>

// -----

func.func @test_maximumf(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<8xf32> {
  %0 = arith.maximumf %arg0, %arg1 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_maximumf
// CHECK: arith.maximumf %{{.*}}, %{{.*}} : vector<8xf32>

// -----

func.func @test_minimumf(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<8xf32> {
  %0 = arith.minimumf %arg0, %arg1 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_minimumf
// CHECK: arith.minimumf %{{.*}}, %{{.*}} : vector<8xf32>

// -----

func.func @test_negf(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = arith.negf %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_negf
// CHECK: arith.negf %{{.*}} : vector<8xf32>

// -----

func.func @test_addi(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
  %0 = arith.addi %arg0, %arg1 : tensor<8xi32>
  return %0 : tensor<8xi32>
}
// CHECK-LABEL: @test_addi
// CHECK: arith.addi %{{.*}}, %{{.*}} : vector<8xi32>

// -----

func.func @test_subi(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
  %0 = arith.subi %arg0, %arg1 : tensor<8xi32>
  return %0 : tensor<8xi32>
}
// CHECK-LABEL: @test_subi
// CHECK: arith.subi %{{.*}}, %{{.*}} : vector<8xi32>

// -----

func.func @test_muli(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
  %0 = arith.muli %arg0, %arg1 : tensor<8xi32>
  return %0 : tensor<8xi32>
}
// CHECK-LABEL: @test_muli
// CHECK: arith.muli %{{.*}}, %{{.*}} : vector<8xi32>

// -----

func.func @test_divsi(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
  %0 = arith.divsi %arg0, %arg1 : tensor<8xi32>
  return %0 : tensor<8xi32>
}
// CHECK-LABEL: @test_divsi
// CHECK: arith.divsi %{{.*}}, %{{.*}} : vector<8xi32>

// -----

func.func @test_divui(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
  %0 = arith.divui %arg0, %arg1 : tensor<8xi32>
  return %0 : tensor<8xi32>
}
// CHECK-LABEL: @test_divui
// CHECK: arith.divui %{{.*}}, %{{.*}} : vector<8xi32>

// -----

func.func @test_remsi(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
  %0 = arith.remsi %arg0, %arg1 : tensor<8xi32>
  return %0 : tensor<8xi32>
}
// CHECK-LABEL: @test_remsi
// CHECK: arith.remsi %{{.*}}, %{{.*}} : vector<8xi32>

// -----

func.func @test_remui(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
  %0 = arith.remui %arg0, %arg1 : tensor<8xi32>
  return %0 : tensor<8xi32>
}
// CHECK-LABEL: @test_remui
// CHECK: arith.remui %{{.*}}, %{{.*}} : vector<8xi32>

// -----

func.func @test_maxsi(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
  %0 = arith.maxsi %arg0, %arg1 : tensor<8xi32>
  return %0 : tensor<8xi32>
}
// CHECK-LABEL: @test_maxsi
// CHECK: arith.maxsi %{{.*}}, %{{.*}} : vector<8xi32>

// -----

func.func @test_maxui(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
  %0 = arith.maxui %arg0, %arg1 : tensor<8xi32>
  return %0 : tensor<8xi32>
}
// CHECK-LABEL: @test_maxui
// CHECK: arith.maxui %{{.*}}, %{{.*}} : vector<8xi32>

// -----

func.func @test_minsi(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
  %0 = arith.minsi %arg0, %arg1 : tensor<8xi32>
  return %0 : tensor<8xi32>
}
// CHECK-LABEL: @test_minsi
// CHECK: arith.minsi %{{.*}}, %{{.*}} : vector<8xi32>

// -----

func.func @test_minui(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
  %0 = arith.minui %arg0, %arg1 : tensor<8xi32>
  return %0 : tensor<8xi32>
}
// CHECK-LABEL: @test_minui
// CHECK: arith.minui %{{.*}}, %{{.*}} : vector<8xi32>

// -----

func.func @test_andi(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
  %0 = arith.andi %arg0, %arg1 : tensor<8xi32>
  return %0 : tensor<8xi32>
}
// CHECK-LABEL: @test_andi
// CHECK: arith.andi %{{.*}}, %{{.*}} : vector<8xi32>

// -----

func.func @test_ori(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
  %0 = arith.ori %arg0, %arg1 : tensor<8xi32>
  return %0 : tensor<8xi32>
}
// CHECK-LABEL: @test_ori
// CHECK: arith.ori %{{.*}}, %{{.*}} : vector<8xi32>

// -----

func.func @test_xori(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
  %0 = arith.xori %arg0, %arg1 : tensor<8xi32>
  return %0 : tensor<8xi32>
}
// CHECK-LABEL: @test_xori
// CHECK: arith.xori %{{.*}}, %{{.*}} : vector<8xi32>

// -----

func.func @test_shli(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
  %0 = arith.shli %arg0, %arg1 : tensor<8xi32>
  return %0 : tensor<8xi32>
}
// CHECK-LABEL: @test_shli
// CHECK: arith.shli %{{.*}}, %{{.*}} : vector<8xi32>

// -----

func.func @test_shrsi(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
  %0 = arith.shrsi %arg0, %arg1 : tensor<8xi32>
  return %0 : tensor<8xi32>
}
// CHECK-LABEL: @test_shrsi
// CHECK: arith.shrsi %{{.*}}, %{{.*}} : vector<8xi32>

// -----

func.func @test_shrui(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
  %0 = arith.shrui %arg0, %arg1 : tensor<8xi32>
  return %0 : tensor<8xi32>
}
// CHECK-LABEL: @test_shrui
// CHECK: arith.shrui %{{.*}}, %{{.*}} : vector<8xi32>

// -----

func.func @test_select(%arg0: tensor<8xi1>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>) -> tensor<8xf32> {
  %0 = arith.select %arg0, %arg1, %arg2 : tensor<8xi1>, tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_select
// CHECK: arith.select %{{.*}}, %{{.*}}, %{{.*}} : vector<8xi1>, vector<8xf32>

// -----

func.func @test_cmpf(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<8xi1> {
  %0 = arith.cmpf oeq, %arg0, %arg1 : tensor<8xf32>
  return %0 : tensor<8xi1>
}
// CHECK-LABEL: @test_cmpf
// CHECK: arith.cmpf {{.*}}, %{{.*}}, %{{.*}} : vector<8xf32>

// -----

func.func @test_cmpi(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi1> {
  %0 = arith.cmpi eq, %arg0, %arg1 : tensor<8xi32>
  return %0 : tensor<8xi1>
}
// CHECK-LABEL: @test_cmpi
// CHECK: arith.cmpi {{.*}}, %{{.*}}, %{{.*}} : vector<8xi32>

// -----

func.func @test_extf(%arg0: tensor<8xf16>) -> tensor<8xf32> {
  %0 = arith.extf %arg0 : tensor<8xf16> to tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_extf
// CHECK: arith.extf %{{.*}} : vector<8xf16> to vector<8xf32>

// -----

func.func @test_truncf(%arg0: tensor<8xf32>) -> tensor<8xf16> {
  %0 = arith.truncf %arg0 : tensor<8xf32> to tensor<8xf16>
  return %0 : tensor<8xf16>
}
// CHECK-LABEL: @test_truncf
// CHECK: arith.truncf %{{.*}} : vector<8xf32> to vector<8xf16>

// -----

func.func @test_extsi(%arg0: tensor<8xi16>) -> tensor<8xi32> {
  %0 = arith.extsi %arg0 : tensor<8xi16> to tensor<8xi32>
  return %0 : tensor<8xi32>
}
// CHECK-LABEL: @test_extsi
// CHECK: arith.extsi %{{.*}} : vector<8xi16> to vector<8xi32>

// -----

func.func @test_extui(%arg0: tensor<8xi16>) -> tensor<8xi32> {
  %0 = arith.extui %arg0 : tensor<8xi16> to tensor<8xi32>
  return %0 : tensor<8xi32>
}
// CHECK-LABEL: @test_extui
// CHECK: arith.extui %{{.*}} : vector<8xi16> to vector<8xi32>

// -----

func.func @test_fptosi(%arg0: tensor<8xf32>) -> tensor<8xi32> {
  %0 = arith.fptosi %arg0 : tensor<8xf32> to tensor<8xi32>
  return %0 : tensor<8xi32>
}
// CHECK-LABEL: @test_fptosi
// CHECK: arith.fptosi %{{.*}} : vector<8xf32> to vector<8xi32>

// -----

func.func @test_fptoui(%arg0: tensor<8xf32>) -> tensor<8xi32> {
  %0 = arith.fptoui %arg0 : tensor<8xf32> to tensor<8xi32>
  return %0 : tensor<8xi32>
}
// CHECK-LABEL: @test_fptoui
// CHECK: arith.fptoui %{{.*}} : vector<8xf32> to vector<8xi32>

// -----

func.func @test_sitofp(%arg0: tensor<8xi32>) -> tensor<8xf32> {
  %0 = arith.sitofp %arg0 : tensor<8xi32> to tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_sitofp
// CHECK: arith.sitofp %{{.*}} : vector<8xi32> to vector<8xf32>

// -----

func.func @test_uitofp(%arg0: tensor<8xi32>) -> tensor<8xf32> {
  %0 = arith.uitofp %arg0 : tensor<8xi32> to tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_uitofp
// CHECK: arith.uitofp %{{.*}} : vector<8xi32> to vector<8xf32>

// -----

func.func @test_trunci(%arg0: tensor<8xi32>) -> tensor<8xi16> {
  %0 = arith.trunci %arg0 : tensor<8xi32> to tensor<8xi16>
  return %0 : tensor<8xi16>
}
// CHECK-LABEL: @test_trunci
// CHECK: arith.trunci %{{.*}} : vector<8xi32> to vector<8xi16>

// -----

func.func @test_index_cast(%arg0: tensor<8xindex>) -> tensor<8xi32> {
  %0 = arith.index_cast %arg0 : tensor<8xindex> to tensor<8xi32>
  return %0 : tensor<8xi32>
}
// CHECK-LABEL: @test_index_cast
// CHECK: arith.index_cast %{{.*}} : vector<8xindex> to vector<8xi32>

// -----

func.func @test_index_castui(%arg0: tensor<8xindex>) -> tensor<8xi32> {
  %0 = arith.index_castui %arg0 : tensor<8xindex> to tensor<8xi32>
  return %0 : tensor<8xi32>
}
// CHECK-LABEL: @test_index_castui
// CHECK: arith.index_castui %{{.*}} : vector<8xindex> to vector<8xi32>

// -----

func.func @test_arith_bitcast(%arg0: tensor<8xf32>) -> tensor<8xi32> {
  %0 = arith.bitcast %arg0 : tensor<8xf32> to tensor<8xi32>
  return %0 : tensor<8xi32>
}
// CHECK-LABEL: @test_arith_bitcast
// CHECK: arith.bitcast %{{.*}} : vector<8xf32> to vector<8xi32>

// -----

func.func @test_tensor_bitcast(%arg0: tensor<8xf32>) -> tensor<8xi32> {
  %0 = tensor.bitcast %arg0 : tensor<8xf32> to tensor<8xi32>
  return %0 : tensor<8xi32>
}
// CHECK-LABEL: @test_tensor_bitcast
// CHECK: arith.bitcast %{{.*}} : vector<8xf32> to vector<8xi32>

// -----

func.func @test_absf(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = math.absf %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_absf
// CHECK: math.absf %{{.*}} : vector<8xf32>

// -----

func.func @test_ceil(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = math.ceil %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_ceil
// CHECK: math.ceil %{{.*}} : vector<8xf32>

// -----

func.func @test_floor(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = math.floor %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_floor
// CHECK: math.floor %{{.*}} : vector<8xf32>

// -----

func.func @test_roundeven(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = math.roundeven %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_roundeven
// CHECK: math.roundeven %{{.*}} : vector<8xf32>

// -----

func.func @test_acos(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = math.acos %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_acos
// CHECK: math.acos %{{.*}} : vector<8xf32>

// -----

func.func @test_acosh(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = math.acosh %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_acosh
// CHECK: math.acosh %{{.*}} : vector<8xf32>

// -----

func.func @test_asin(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = math.asin %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_asin
// CHECK: math.asin %{{.*}} : vector<8xf32>

// -----

func.func @test_asinh(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = math.asinh %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_asinh
// CHECK: math.asinh %{{.*}} : vector<8xf32>

// -----

func.func @test_atanh(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = math.atanh %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_atanh
// CHECK: math.atanh %{{.*}} : vector<8xf32>

// -----

func.func @test_cos(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = math.cos %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_cos
// CHECK: math.cos %{{.*}} : vector<8xf32>

// -----

func.func @test_cosh(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = math.cosh %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_cosh
// CHECK: math.cosh %{{.*}} : vector<8xf32>

// -----

func.func @test_exp(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = math.exp %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_exp
// CHECK: math.exp %{{.*}} : vector<8xf32>

// -----

func.func @test_erf(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = math.erf %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_erf
// CHECK: math.erf %{{.*}} : vector<8xf32>

// -----

func.func @test_expm1(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = math.expm1 %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_expm1
// CHECK: math.expm1 %{{.*}} : vector<8xf32>

// -----

func.func @test_log(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = math.log %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_log
// CHECK: math.log %{{.*}} : vector<8xf32>

// -----

func.func @test_log1p(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = math.log1p %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_log1p
// CHECK: math.log1p %{{.*}} : vector<8xf32>

// -----

func.func @test_rsqrt(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = math.rsqrt %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_rsqrt
// CHECK: math.rsqrt %{{.*}} : vector<8xf32>

// -----

func.func @test_sin(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = math.sin %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_sin
// CHECK: math.sin %{{.*}} : vector<8xf32>

// -----

func.func @test_sinh(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = math.sinh %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_sinh
// CHECK: math.sinh %{{.*}} : vector<8xf32>

// -----

func.func @test_sqrt(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = math.sqrt %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_sqrt
// CHECK: math.sqrt %{{.*}} : vector<8xf32>

// -----

func.func @test_tan(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = math.tan %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_tan
// CHECK: math.tan %{{.*}} : vector<8xf32>

// -----

func.func @test_tanh(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = math.tanh %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_tanh
// CHECK: math.tanh %{{.*}} : vector<8xf32>

// -----

func.func @test_cbrt(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = math.cbrt %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_cbrt
// CHECK: math.cbrt %{{.*}} : vector<8xf32>

// -----

func.func @test_absi(%arg0: tensor<8xi32>) -> tensor<8xi32> {
  %0 = math.absi %arg0 : tensor<8xi32>
  return %0 : tensor<8xi32>
}
// CHECK-LABEL: @test_absi
// CHECK: math.absi %{{.*}} : vector<8xi32>

// -----

func.func @test_atan2(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<8xf32> {
  %0 = math.atan2 %arg0, %arg1 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_atan2
// CHECK: math.atan2 %{{.*}}, %{{.*}} : vector<8xf32>

// -----

func.func @test_powf(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<8xf32> {
  %0 = math.powf %arg0, %arg1 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_powf
// CHECK: math.powf %{{.*}}, %{{.*}} : vector<8xf32>

// -----

func.func @test_ipowi(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
  %0 = math.ipowi %arg0, %arg1 : tensor<8xi32>
  return %0 : tensor<8xi32>
}
// CHECK-LABEL: @test_ipowi
// CHECK: math.ipowi %{{.*}}, %{{.*}} : vector<8xi32>

// -----

func.func @test_isfinite(%arg0: tensor<8xf32>) -> tensor<8xi1> {
  %0 = math.isfinite %arg0 : tensor<8xf32>
  return %0 : tensor<8xi1>
}
// CHECK-LABEL: @test_isfinite
// CHECK: math.isfinite %{{.*}} : vector<8xf32>

// -----

func.func @test_extract_aligned(%arg0: memref<128xf32>, %arg1: index) -> tensor<8xf32> {
  %c0 = arith.constant 0 : index
  %0 = xtile.extract %arg0[%c0] [8] [1] : memref<128xf32> -> tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK: #indexing_map = #xla.indexing_map<"(d0) -> (-d0 + 128 - 8){{.*}}">
// CHECK-LABEL: @test_extract_aligned
// CHECK-DAG: %[[PAD:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG: %[[INDEXING:[^:]+]] = xla.apply_indexing #indexing_map(%{{.*}})
// CHECK: %[[COND:.*]] = arith.cmpi sge, %[[INDEXING]], %{{.*}} : index
// CHECK: %[[RES:.*]] = scf.if %[[COND]] -> (vector<8xf32>) {
// CHECK:   %[[READ:.*]] = vector.transfer_read %{{.*}}[%{{.*}}], %[[PAD]] {in_bounds = [true]} : memref<128xf32>, vector<8xf32>
// CHECK:   scf.yield %[[READ]] : vector<8xf32>
// CHECK: } else {
// CHECK:   %[[MASK_DIM:.*]] = arith.maxsi
// CHECK:   %[[MASK:.*]] = vector.create_mask %[[MASK_DIM]] : vector<8xi1>
// CHECK:   %[[MASKED_READ:.*]] = vector.transfer_read %{{.*}}[%{{.*}}], %[[PAD]], %[[MASK]] : memref<128xf32>, vector<8xf32>
// CHECK:   scf.yield %[[MASKED_READ]] : vector<8xf32>
// CHECK: }

// -----

func.func @test_extract_unaligned(%arg0: memref<128xf32>, %arg1: index) -> tensor<8xf32> {
  %0 = xtile.extract %arg0[%arg1] [8] [1] : memref<128xf32> -> tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK: #indexing_map = #xla.indexing_map<"(d0) -> (-d0 + 128 - 8){{.*}}">
// CHECK-LABEL: @test_extract_unaligned
// CHECK-DAG: %[[PAD:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG: %[[INDEXING:[^:]+]] = xla.apply_indexing #indexing_map(%{{.*}})
// CHECK: %[[COND:.*]] = arith.cmpi sge, %[[INDEXING]], %{{.*}} : index
// CHECK: %[[RES:.*]] = scf.if %[[COND]] -> (vector<8xf32>) {
// CHECK:   %[[READ:.*]] = vector.transfer_read %{{.*}}[%{{.*}}], %[[PAD]] {in_bounds = [true]} : memref<128xf32>, vector<8xf32>
// CHECK:   scf.yield %[[READ]] : vector<8xf32>
// CHECK: } else {
// CHECK:   %[[MASK_DIM:.*]] = arith.maxsi
// CHECK:   %[[MASK:.*]] = vector.create_mask %[[MASK_DIM]] : vector<8xi1>
// CHECK:   %[[MASKED_READ:.*]] = vector.transfer_read %{{.*}}[%{{.*}}], %[[PAD]], %[[MASK]] : memref<128xf32>, vector<8xf32>
// CHECK:   scf.yield %[[MASKED_READ]] : vector<8xf32>
// CHECK: }

// -----

func.func @test_insert_aligned(%arg0: tensor<8xf32>, %arg1: memref<128xf32>) {
  %c0 = arith.constant 0 : index
  xtile.insert %arg0 into %arg1[%c0] [8] [1] : tensor<8xf32> -> memref<128xf32>
  return
}
// CHECK: #indexing_map = #xla.indexing_map<"(d0) -> (-d0 + 128 - 8){{.*}}">
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

// -----

func.func @test_insert_unaligned(%arg0: tensor<8xf32>, %arg1: memref<128xf32>, %arg2: index) {
  xtile.insert %arg0 into %arg1[%arg2] [8] [1] : tensor<8xf32> -> memref<128xf32>
  return
}
// CHECK: #indexing_map = #xla.indexing_map<"(d0) -> (-d0 + 128 - 8){{.*}}">
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

// -----

func.func @test_extract_strided(%arg0: memref<12x1xf64>, %arg1: index) -> tensor<2x1xf64> {
  %c0 = arith.constant 0 : index
  %0 = xtile.extract %arg0[%arg1, %c0] [2, 1] [3, 1] : memref<12x1xf64> -> tensor<2x1xf64>
  return %0 : tensor<2x1xf64>
}
// CHECK-LABEL: @test_extract_strided
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<2x1xf64>
// CHECK: %[[LOAD0:.*]] = memref.load %{{.*}}[%{{.*}}, %[[C0]]] : memref<12x1xf64>
// CHECK: %[[INS0:.*]] = vector.insert %[[LOAD0]], %[[CST]] [0, 0] : f64 into vector<2x1xf64>
// CHECK: %[[C3:.*]] = arith.constant 3 : index
// CHECK: %[[OFF1:.*]] = arith.addi %{{.*}}, %[[C3]] : index
// CHECK: %[[LOAD1:.*]] = memref.load %{{.*}}[%[[OFF1]], %[[C0]]] : memref<12x1xf64>
// CHECK: %[[INS1:.*]] = vector.insert %[[LOAD1]], %[[INS0]] [1, 0] : f64 into vector<2x1xf64>
// CHECK: %[[RET:.*]] = builtin.unrealized_conversion_cast %[[INS1]] : vector<2x1xf64> to tensor<2x1xf64>
// CHECK: return %[[RET]]

// -----

func.func @test_insert_strided(%arg0: tensor<2x1xf64>, %arg1: memref<12x1xf64>, %arg2: index) {
  %c0 = arith.constant 0 : index
  xtile.insert %arg0 into %arg1[%arg2, %c0] [2, 1] [3, 1] : tensor<2x1xf64> -> memref<12x1xf64>
  return
}
// CHECK-LABEL: @test_insert_strided
// CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %{{.*}} : tensor<2x1xf64> to vector<2x1xf64>
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[EXT0:.*]] = vector.extract %[[CAST]][0, 0] : f64 from vector<2x1xf64>
// CHECK: memref.store %[[EXT0]], %{{.*}}[%{{.*}}, %[[C0]]] : memref<12x1xf64>
// CHECK: %[[C3:.*]] = arith.constant 3 : index
// CHECK: %[[OFF1:.*]] = arith.addi %{{.*}}, %[[C3]] : index
// CHECK: %[[EXT1:.*]] = vector.extract %[[CAST]][1, 0] : f64 from vector<2x1xf64>
// CHECK: memref.store %[[EXT1]], %{{.*}}[%[[OFF1]], %[[C0]]] : memref<12x1xf64>
// CHECK: return

// -----

func.func @test_extract_multi_strided(%arg0: memref<8x8xf32>, %arg1: index, %arg2: index) -> tensor<2x2xf32> {
  %0 = xtile.extract %arg0[%arg1, %arg2] [2, 2] [2, 3] : memref<8x8xf32> -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}
// CHECK-LABEL: @test_extract_multi_strided
// CHECK: %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<2x2xf32>
// CHECK: %[[LOAD0:.*]] = memref.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<8x8xf32>
// CHECK: %[[INS0:.*]] = vector.insert %[[LOAD0]], %[[CST]] [0, 0] : f32 into vector<2x2xf32>
// CHECK: %[[OFF_D1_1:.*]] = arith.addi %{{.*}}, %{{.*}} : index
// CHECK: %[[LOAD1:.*]] = memref.load %{{.*}}[%{{.*}}, %[[OFF_D1_1]]] : memref<8x8xf32>
// CHECK: %[[INS1:.*]] = vector.insert %[[LOAD1]], %[[INS0]] [0, 1] : f32 into vector<2x2xf32>
// CHECK: %[[OFF_D0_1:.*]] = arith.addi %{{.*}}, %{{.*}} : index
// CHECK: %[[LOAD2:.*]] = memref.load %{{.*}}[%[[OFF_D0_1]], %{{.*}}] : memref<8x8xf32>
// CHECK: %[[INS2:.*]] = vector.insert %[[LOAD2]], %[[INS1]] [1, 0] : f32 into vector<2x2xf32>
// CHECK: %[[OFF_D0_2:.*]] = arith.addi %{{.*}}, %{{.*}} : index
// CHECK: %[[OFF_D1_2:.*]] = arith.addi %{{.*}}, %{{.*}} : index
// CHECK: %[[LOAD3:.*]] = memref.load %{{.*}}[%[[OFF_D0_2]], %[[OFF_D1_2]]] : memref<8x8xf32>
// CHECK: %[[INS3:.*]] = vector.insert %[[LOAD3]], %[[INS2]] [1, 1] : f32 into vector<2x2xf32>
// CHECK: %[[RET:.*]] = builtin.unrealized_conversion_cast %[[INS3]] : vector<2x2xf32> to tensor<2x2xf32>
// CHECK: return %[[RET]]

// -----

func.func @test_insert_multi_strided(%arg0: tensor<2x2xf32>, %arg1: memref<8x8xf32>, %arg2: index, %arg3: index) {
  xtile.insert %arg0 into %arg1[%arg2, %arg3] [2, 2] [2, 3] : tensor<2x2xf32> -> memref<8x8xf32>
  return
}
// CHECK-LABEL: @test_insert_multi_strided
// CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %{{.*}} : tensor<2x2xf32> to vector<2x2xf32>
// CHECK: %[[EXT0:.*]] = vector.extract %[[CAST]][0, 0] : f32 from vector<2x2xf32>
// CHECK: memref.store %[[EXT0]], %{{.*}}[%{{.*}}, %{{.*}}] : memref<8x8xf32>
// CHECK: %[[OFF_D1_1:.*]] = arith.addi %{{.*}}, %{{.*}} : index
// CHECK: %[[EXT1:.*]] = vector.extract %[[CAST]][0, 1] : f32 from vector<2x2xf32>
// CHECK: memref.store %[[EXT1]], %{{.*}}[%{{.*}}, %[[OFF_D1_1]]] : memref<8x8xf32>
// CHECK: %[[OFF_D0_1:.*]] = arith.addi %{{.*}}, %{{.*}} : index
// CHECK: %[[EXT2:.*]] = vector.extract %[[CAST]][1, 0] : f32 from vector<2x2xf32>
// CHECK: memref.store %[[EXT2]], %{{.*}}[%[[OFF_D0_1]], %{{.*}}] : memref<8x8xf32>
// CHECK: %[[OFF_D0_2:.*]] = arith.addi %{{.*}}, %{{.*}} : index
// CHECK: %[[OFF_D1_2:.*]] = arith.addi %{{.*}}, %{{.*}} : index
// CHECK: %[[EXT3:.*]] = vector.extract %[[CAST]][1, 1] : f32 from vector<2x2xf32>
// CHECK: memref.store %[[EXT3]], %{{.*}}[%[[OFF_D0_2]], %[[OFF_D1_2]]] : memref<8x8xf32>
// CHECK: return

// -----

func.func @test_broadcast(%arg0: tensor<16xf32>) -> tensor<8x16xf32> {
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}
// CHECK-LABEL: @test_broadcast
// CHECK: %[[SCAST:.*]] = vector.shape_cast %{{.*}}: vector<16xf32> to vector<1x16xf32>
// CHECK: %[[BCAST:.*]] = vector.broadcast %[[SCAST]] : vector<1x16xf32> to vector<8x16xf32>

// -----

func.func @test_transpose(%arg0: tensor<16x16x8xf32>) -> tensor<8x16x16xf32> {
  %0 = stablehlo.transpose %arg0, dims = [2, 0, 1] : (tensor<16x16x8xf32>) -> tensor<8x16x16xf32>
  return %0 : tensor<8x16x16xf32>
}
// CHECK-LABEL: @test_transpose
// CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %{{.*}} : tensor<16x16x8xf32> to vector<16x16x8xf32>
// CHECK: %[[TRANS:.*]] = vector.transpose %[[CAST]], [2, 0, 1] : vector<16x16x8xf32> to vector<8x16x16xf32>
// CHECK: %[[RET:.*]] = builtin.unrealized_conversion_cast %[[TRANS]] : vector<8x16x16xf32> to tensor<8x16x16xf32>
// CHECK: return %[[RET]]

// -----

func.func @test_transpose_2d(%arg0: tensor<16x32xf32>) -> tensor<32x16xf32> {
  %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<16x32xf32>) -> tensor<32x16xf32>
  return %0 : tensor<32x16xf32>
}
// CHECK-LABEL: @test_transpose_2d
// CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %{{.*}} : tensor<16x32xf32> to vector<16x32xf32>
// CHECK: %[[TRANS:.*]] = vector.transpose %[[CAST]], [1, 0] : vector<16x32xf32> to vector<32x16xf32>
// CHECK: %[[RET:.*]] = builtin.unrealized_conversion_cast %[[TRANS]] : vector<32x16xf32> to tensor<32x16xf32>
// CHECK: return %[[RET]]

// -----

func.func @test_constant() -> tensor<8x8xf32> {
  %0 = arith.constant dense<1.000000e+00> : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}
// CHECK-LABEL: @test_constant
// CHECK: %[[CST:.*]] = arith.constant dense<1.000000e+00> : vector<8x8xf32>
// CHECK: %[[RET:.*]] = builtin.unrealized_conversion_cast %[[CST]] : vector<8x8xf32> to tensor<8x8xf32>
// CHECK: return %[[RET]]

// -----

func.func @test_dot_general(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>, %arg2: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  %1 = arith.addf %0, %arg2 : tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}
// CHECK-LABEL: @test_dot_general
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x8xf32>, %[[ARG1:.*]]: tensor<8x8xf32>, %[[ARG2:.*]]: tensor<8x8xf32>)
// CHECK-DAG: %[[LHS:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : tensor<8x8xf32> to vector<8x8xf32>
// CHECK-DAG: %[[RHS:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : tensor<8x8xf32> to vector<8x8xf32>
// CHECK-DAG: %[[ACC:.*]] = builtin.unrealized_conversion_cast %[[ARG2]] : tensor<8x8xf32> to vector<8x8xf32>
// CHECK: %[[RES:.*]] = vector.contract {indexing_maps = [{{.*}}], iterator_types = ["reduction", "parallel", "parallel"], kind = #vector.kind<add>} %[[LHS]], %[[RHS]], %[[ACC]] : vector<8x8xf32>, vector<8x8xf32> into vector<8x8xf32>
// CHECK: %[[RET:.*]] = builtin.unrealized_conversion_cast %[[RES]] : vector<8x8xf32> to tensor<8x8xf32>
// CHECK: return %[[RET]]

// -----

func.func @test_dot_general_batch(%arg0: tensor<2x8x8xf32>, %arg1: tensor<2x8x8xf32>, %arg2: tensor<2x8x8xf32>) -> tensor<2x8x8xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x8x8xf32>, tensor<2x8x8xf32>) -> tensor<2x8x8xf32>
  %1 = arith.addf %0, %arg2 : tensor<2x8x8xf32>
  return %1 : tensor<2x8x8xf32>
}
// CHECK-LABEL: @test_dot_general_batch
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2x8x8xf32>, %[[ARG1:.*]]: tensor<2x8x8xf32>, %[[ARG2:.*]]: tensor<2x8x8xf32>)
// CHECK-DAG: %[[LHS:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : tensor<2x8x8xf32> to vector<2x8x8xf32>
// CHECK-DAG: %[[RHS:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : tensor<2x8x8xf32> to vector<2x8x8xf32>
// CHECK-DAG: %[[ACC:.*]] = builtin.unrealized_conversion_cast %[[ARG2]] : tensor<2x8x8xf32> to vector<2x8x8xf32>
// CHECK: %[[RES:.*]] = vector.contract {indexing_maps = [{{.*}}], iterator_types = ["parallel", "reduction", "parallel", "parallel"], kind = #vector.kind<add>} %[[LHS]], %[[RHS]], %[[ACC]] : vector<2x8x8xf32>, vector<2x8x8xf32> into vector<2x8x8xf32>
// CHECK: %[[RET:.*]] = builtin.unrealized_conversion_cast %[[RES]] : vector<2x8x8xf32> to tensor<2x8x8xf32>
// CHECK: return %[[RET]]

// -----

func.func @test_iota() -> tensor<8xi32> {
  %0 = stablehlo.iota dim = 0 : tensor<8xi32>
  return %0 : tensor<8xi32>
}
// CHECK-LABEL: @test_iota
// CHECK: %[[CST:.*]] = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : vector<8xi32>
// CHECK: %[[RET:.*]] = builtin.unrealized_conversion_cast %[[CST]] : vector<8xi32> to tensor<8xi32>
// CHECK: return %[[RET]]

// -----

func.func @test_iota_f32() -> tensor<8xf32> {
  %0 = stablehlo.iota dim = 0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_iota_f32
// CHECK: %[[CST:.*]] = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : vector<8xi32>
// CHECK: %[[CONV:.*]] = arith.sitofp %[[CST]] : vector<8xi32> to vector<8xf32>
// CHECK: %[[RET:.*]] = builtin.unrealized_conversion_cast %[[CONV]] : vector<8xf32> to tensor<8xf32>
// CHECK: return %[[RET]]

// -----

func.func @test_iota_2d() -> tensor<4x8xi32> {
  %0 = stablehlo.iota dim = 1 : tensor<4x8xi32>
  return %0 : tensor<4x8xi32>
}
// CHECK-LABEL: @test_iota_2d
// CHECK: %[[CST:.*]] = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : vector<8xi32>
// CHECK: %[[CAST:.*]] = vector.shape_cast %[[CST]] : vector<8xi32> to vector<1x8xi32>
// CHECK: %[[BCAST:.*]] = vector.broadcast %[[CAST]] : vector<1x8xi32> to vector<4x8xi32>
// CHECK: %[[RET:.*]] = builtin.unrealized_conversion_cast %[[BCAST]] : vector<4x8xi32> to tensor<4x8xi32>
// CHECK: return %[[RET]]

// -----

func.func @test_reduce(%arg0: tensor<8x32xf32>, %arg1: tensor<f32>) -> tensor<8xf32> {
  %0 = stablehlo.reduce(%arg0 init: %arg1) across dimensions = [1] : (tensor<8x32xf32>, tensor<f32>) -> tensor<8xf32>
   reducer(%arg2: tensor<f32>, %arg3: tensor<f32>)  {
    %1 = arith.addf %arg2, %arg3 : tensor<f32>
    stablehlo.return %1 : tensor<f32>
  }
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @test_reduce
// CHECK: %[[INIT_CAST:.*]] = builtin.unrealized_conversion_cast %{{.*}} : tensor<f32> to vector<f32>
// CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %{{.*}} : tensor<8x32xf32> to vector<8x32xf32>
// CHECK: %[[INIT:.*]] = vector.extract %[[INIT_CAST]][] : f32 from vector<f32>
// CHECK: %[[ACC:.*]] = vector.broadcast %[[INIT]] : f32 to vector<8xf32>
// CHECK: %[[REDUCE:.*]] = vector.multi_reduction <add>, %[[CAST]], %[[ACC]] [1] : vector<8x32xf32> to vector<8xf32>
// CHECK: %[[RET:.*]] = builtin.unrealized_conversion_cast %[[REDUCE]] : vector<8xf32> to tensor<8xf32>
// CHECK: return %[[RET]]

// -----

func.func @test_reduce_1d_0d(%arg0: tensor<8xf32>, %arg1: tensor<f32>) -> tensor<f32> {
  %0 = stablehlo.reduce(%arg0 init: %arg1) across dimensions = [0] : (tensor<8xf32>, tensor<f32>) -> tensor<f32>
   reducer(%arg2: tensor<f32>, %arg3: tensor<f32>)  {
    %1 = arith.addf %arg2, %arg3 : tensor<f32>
    stablehlo.return %1 : tensor<f32>
  }
  return %0 : tensor<f32>
}
// CHECK-LABEL: @test_reduce_1d_0d
// CHECK: %[[INIT_CAST:.*]] = builtin.unrealized_conversion_cast %{{.*}} : tensor<f32> to vector<f32>
// CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %{{.*}} : tensor<8xf32> to vector<8xf32>
// CHECK: %[[INIT:.*]] = vector.extract %[[INIT_CAST]][] : f32 from vector<f32>
// CHECK: %[[REDUCE:.*]] = vector.multi_reduction <add>, %[[CAST]], %[[INIT]] [0] : vector<8xf32> to f32
// CHECK: %[[BCAST:.*]] = vector.broadcast %[[REDUCE]] : f32 to vector<f32>
// CHECK: %[[RET:.*]] = builtin.unrealized_conversion_cast %[[BCAST]] : vector<f32> to tensor<f32>
// CHECK: return %[[RET]]

// -----

func.func @test_reshape(%arg0: tensor<16x32xf32>) -> tensor<512xf32> {
  %0 = stablehlo.reshape %arg0 : (tensor<16x32xf32>) -> tensor<512xf32>
  return %0 : tensor<512xf32>
}
// CHECK-LABEL: @test_reshape
// CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %{{.*}} : tensor<16x32xf32> to vector<16x32xf32>
// CHECK: %[[RESHAPE:.*]] = vector.shape_cast %[[CAST]] : vector<16x32xf32> to vector<512xf32>
// CHECK: %[[RET:.*]] = builtin.unrealized_conversion_cast %[[RESHAPE]] : vector<512xf32> to tensor<512xf32>
// CHECK: return %[[RET]]

// -----

func.func @test_mask(%arg0: tensor<4x8xf32>, %arg1: f32) -> tensor<4x8xf32> {
  %0 = xtile.mask %arg0 bounds [3, 6], %arg1 : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}
// CHECK-LABEL: @test_mask
// CHECK: %[[SRC:.*]] = builtin.unrealized_conversion_cast %{{.*}} : tensor<4x8xf32> to vector<4x8xf32>
// CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG: %[[C6:.*]] = arith.constant 6 : index
// CHECK: %[[MASK:.*]] = vector.create_mask %[[C3]], %[[C6]] : vector<4x8xi1>
// CHECK: %[[BCAST:.*]] = vector.broadcast %{{.*}} : f32 to vector<4x8xf32>
// CHECK: %[[SELECT:.*]] = arith.select %[[MASK]], %[[SRC]], %[[BCAST]] : vector<4x8xi1>, vector<4x8xf32>
// CHECK: %[[RET:.*]] = builtin.unrealized_conversion_cast %[[SELECT]] : vector<4x8xf32> to tensor<4x8xf32>
// CHECK: return %[[RET]]

// -----

func.func @test_slice(%arg0: tensor<8x16xf32>) -> tensor<4x8xf32> {
  %0 = "stablehlo.slice"(%arg0) {limit_indices = array<i64: 6, 12>, start_indices = array<i64: 2, 4>, strides = array<i64: 1, 1>} : (tensor<8x16xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}
// CHECK-LABEL: @test_slice
// CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %{{.*}} : tensor<8x16xf32> to vector<8x16xf32>
// CHECK: %[[SLICE:.*]] = vector.extract_strided_slice %[[CAST]] {offsets = [2, 4], sizes = [4, 8], strides = [1, 1]} : vector<8x16xf32> to vector<4x8xf32>
// CHECK: %[[RET:.*]] = builtin.unrealized_conversion_cast %[[SLICE]] : vector<4x8xf32> to tensor<4x8xf32>
// CHECK: return %[[RET]]

// -----

func.func @test_slice_strided(%arg0: tensor<8x16xf32>) -> tensor<2x4xf32> {
  %0 = "stablehlo.slice"(%arg0) {limit_indices = array<i64: 6, 12>, start_indices = array<i64: 2, 4>, strides = array<i64: 2, 2>} : (tensor<8x16xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}
// CHECK-LABEL: @test_slice_strided
// CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %{{.*}} : tensor<8x16xf32> to vector<8x16xf32>
// CHECK: %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<2x4xf32>
// CHECK: %[[EXT0:.*]] = vector.extract %[[CAST]][2, 4] : f32 from vector<8x16xf32>
// CHECK: %[[INS0:.*]] = vector.insert %[[EXT0]], %[[CST]] [0, 0] : f32 into vector<2x4xf32>
// CHECK: %[[EXT1:.*]] = vector.extract %[[CAST]][2, 6] : f32 from vector<8x16xf32>
// CHECK: %[[INS1:.*]] = vector.insert %[[EXT1]], %[[INS0]] [0, 1] : f32 into vector<2x4xf32>
// CHECK: %[[RET:.*]] = builtin.unrealized_conversion_cast %{{.*}} : vector<2x4xf32> to tensor<2x4xf32>
// CHECK: return %[[RET]]

// -----

func.func @test_concatenate(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> tensor<4x16xf32> {
  %0 = stablehlo.concatenate %arg0, %arg1, dim = 1 : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}
// CHECK-LABEL: @test_concatenate
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x8xf32>, %[[ARG1:.*]]: tensor<4x8xf32>)
// CHECK-DAG: %[[LHS:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : tensor<4x8xf32> to vector<4x8xf32>
// CHECK-DAG: %[[RHS:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : tensor<4x8xf32> to vector<4x8xf32>
// CHECK: %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<4x16xf32>
// CHECK: %[[INSERT0:.*]] = vector.insert_strided_slice %[[LHS]], %[[CST]] {offsets = [0, 0], strides = [1, 1]} : vector<4x8xf32> into vector<4x16xf32>
// CHECK: %[[INSERT1:.*]] = vector.insert_strided_slice %[[RHS]], %[[INSERT0]] {offsets = [0, 8], strides = [1, 1]} : vector<4x8xf32> into vector<4x16xf32>
// CHECK: %[[RET:.*]] = builtin.unrealized_conversion_cast %[[INSERT1]] : vector<4x16xf32> to tensor<4x16xf32>
// CHECK: return %[[RET]]

