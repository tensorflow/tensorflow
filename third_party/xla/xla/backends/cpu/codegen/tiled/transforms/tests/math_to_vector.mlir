// RUN: fusion_compiler_opt %s --xtile-cpu-elemental-tensor-to-vector -split-input-file | FileCheck %s

func.func @absf(%input : tensor<1024xf32>) -> tensor<1024xf32> {
  // CHECK: math.absf %{{.*}} : vector<1024xf32>
  %abs = math.absf %input : tensor<1024xf32>
  return %abs : tensor<1024xf32>
}

// -----

func.func @absi(%input : tensor<1024xi32>) -> tensor<1024xi32> {
  // CHECK: math.absi %{{.*}} : vector<1024xi32>
  %abs = math.absi %input : tensor<1024xi32>
  return %abs : tensor<1024xi32>
}

// -----

func.func @acos(%input : tensor<1024xf32>) -> tensor<1024xf32> {
  // CHECK: math.acos %{{.*}} : vector<1024xf32>
  %res = math.acos %input : tensor<1024xf32>
  return %res : tensor<1024xf32>
}

// -----

func.func @acosh(%input : tensor<1024xf32>) -> tensor<1024xf32> {
  // CHECK: math.acosh %{{.*}} : vector<1024xf32>
  %res = math.acosh %input : tensor<1024xf32>
  return %res : tensor<1024xf32>
}

// -----

func.func @asin(%input : tensor<1024xf32>) -> tensor<1024xf32> {
  // CHECK: math.asin %{{.*}} : vector<1024xf32>
  %res = math.asin %input : tensor<1024xf32>
  return %res : tensor<1024xf32>
}

// -----

func.func @asinh(%input : tensor<1024xf32>) -> tensor<1024xf32> {
  // CHECK: math.asinh %{{.*}} : vector<1024xf32>
  %res = math.asinh %input : tensor<1024xf32>
  return %res : tensor<1024xf32>
}

// -----

func.func @atan2(%input1 : tensor<1024xf32>, %input2 : tensor<1024xf32>) -> tensor<1024xf32> {
  // CHECK: math.atan2 %{{.*}}, %{{.*}} : vector<1024xf32>
  %res = math.atan2 %input1, %input2 : tensor<1024xf32>
  return %res : tensor<1024xf32>
}

// -----

func.func @atan(%input : tensor<1024xf32>) -> tensor<1024xf32> {
  // CHECK: math.atan %{{.*}} : vector<1024xf32>
  %res = math.atan %input : tensor<1024xf32>
  return %res : tensor<1024xf32>
}

// -----

func.func @atanh(%input : tensor<1024xf32>) -> tensor<1024xf32> {
  // CHECK: math.atanh %{{.*}} : vector<1024xf32>
  %res = math.atanh %input : tensor<1024xf32>
  return %res : tensor<1024xf32>
}

// -----

func.func @cbrt(%input : tensor<1024xf32>) -> tensor<1024xf32> {
  // CHECK: math.cbrt %{{.*}} : vector<1024xf32>
  %res = math.cbrt %input : tensor<1024xf32>
  return %res : tensor<1024xf32>
}

// -----

func.func @ceil(%input : tensor<1024xf32>) -> tensor<1024xf32> {
  // CHECK: math.ceil %{{.*}} : vector<1024xf32>
  %res = math.ceil %input : tensor<1024xf32>
  return %res : tensor<1024xf32>
}

// -----

func.func @clampf(%input : tensor<1024xf32>, %low : tensor<1024xf32>, %high : tensor<1024xf32>) -> tensor<1024xf32> {
  // CHECK: math.clampf %{{.*}} to [%{{.*}}, %{{.*}}] : vector<1024xf32>
  %res = math.clampf %input to [%low, %high] : tensor<1024xf32>
  return %res : tensor<1024xf32>
}

// -----

func.func @copysign(%mag : tensor<1024xf32>, %sign : tensor<1024xf32>) -> tensor<1024xf32> {
  // CHECK: math.copysign %{{.*}}, %{{.*}} : vector<1024xf32>
  %res = math.copysign %mag, %sign : tensor<1024xf32>
  return %res : tensor<1024xf32>
}

// -----

func.func @cos(%input : tensor<1024xf32>) -> tensor<1024xf32> {
  // CHECK: math.cos %{{.*}} : vector<1024xf32>
  %res = math.cos %input : tensor<1024xf32>
  return %res : tensor<1024xf32>
}

// -----

func.func @cosh(%input : tensor<1024xf32>) -> tensor<1024xf32> {
  // CHECK: math.cosh %{{.*}} : vector<1024xf32>
  %res = math.cosh %input : tensor<1024xf32>
  return %res : tensor<1024xf32>
}

// -----

func.func @ctlz(%input : tensor<1024xi32>) -> tensor<1024xi32> {
  // CHECK: math.ctlz %{{.*}} : vector<1024xi32>
  %res = math.ctlz %input : tensor<1024xi32>
  return %res : tensor<1024xi32>
}

// -----

func.func @cttz(%input : tensor<1024xi32>) -> tensor<1024xi32> {
  // CHECK: math.cttz %{{.*}} : vector<1024xi32>
  %res = math.cttz %input : tensor<1024xi32>
  return %res : tensor<1024xi32>
}

// -----

func.func @ctpop(%input : tensor<1024xi32>) -> tensor<1024xi32> {
  // CHECK: math.ctpop %{{.*}} : vector<1024xi32>
  %res = math.ctpop %input : tensor<1024xi32>
  return %res : tensor<1024xi32>
}

// -----

func.func @erf(%input : tensor<1024xf32>) -> tensor<1024xf32> {
  // CHECK: math.erf %{{.*}} : vector<1024xf32>
  %res = math.erf %input : tensor<1024xf32>
  return %res : tensor<1024xf32>
}

// -----

func.func @erfc(%input : tensor<1024xf32>) -> tensor<1024xf32> {
  // CHECK: math.erfc %{{.*}} : vector<1024xf32>
  %res = math.erfc %input : tensor<1024xf32>
  return %res : tensor<1024xf32>
}

// -----

func.func @exp2(%input : tensor<1024xf32>) -> tensor<1024xf32> {
  // CHECK: math.exp2 %{{.*}} : vector<1024xf32>
  %res = math.exp2 %input : tensor<1024xf32>
  return %res : tensor<1024xf32>
}

// -----

func.func @expm1(%input : tensor<1024xf32>) -> tensor<1024xf32> {
  // CHECK: math.expm1 %{{.*}} : vector<1024xf32>
  %res = math.expm1 %input : tensor<1024xf32>
  return %res : tensor<1024xf32>
}

// -----

func.func @exp(%input : tensor<1024xf32>) -> tensor<1024xf32> {
  // CHECK: math.exp %{{.*}} : vector<1024xf32>
  %res = math.exp %input : tensor<1024xf32>
  return %res : tensor<1024xf32>
}

// -----

func.func @fpowi(%base : tensor<1024xf32>, %exp : tensor<1024xi32>) -> tensor<1024xf32> {
  // CHECK: math.fpowi %{{.*}}, %{{.*}} : vector<1024xf32>
  %res = math.fpowi %base, %exp : tensor<1024xf32>, tensor<1024xi32>
  return %res : tensor<1024xf32>
}

// -----

func.func @floor(%input : tensor<1024xf32>) -> tensor<1024xf32> {
  // CHECK: math.floor %{{.*}} : vector<1024xf32>
  %res = math.floor %input : tensor<1024xf32>
  return %res : tensor<1024xf32>
}

// -----

func.func @fma(%a : tensor<1024xf32>, %b : tensor<1024xf32>, %c : tensor<1024xf32>) -> tensor<1024xf32> {
  // CHECK: math.fma %{{.*}}, %{{.*}}, %{{.*}} : vector<1024xf32>
  %res = math.fma %a, %b, %c : tensor<1024xf32>
  return %res : tensor<1024xf32>
}

// -----

func.func @ipowi(%base : tensor<1024xi32>, %exp : tensor<1024xi32>) -> tensor<1024xi32> {
  // CHECK: math.ipowi %{{.*}}, %{{.*}} : vector<1024xi32>
  %res = math.ipowi %base, %exp : tensor<1024xi32>
  return %res : tensor<1024xi32>
}

// -----

func.func @isfinite(%input : tensor<1024xf32>) -> tensor<1024xi1> {
  // CHECK: math.isfinite %{{.*}} : vector<1024xf32>
  %res = math.isfinite %input : tensor<1024xf32>
  return %res : tensor<1024xi1>
}

// -----

func.func @isinf(%input : tensor<1024xf32>) -> tensor<1024xi1> {
  // CHECK: math.isinf %{{.*}} : vector<1024xf32>
  %res = math.isinf %input : tensor<1024xf32>
  return %res : tensor<1024xi1>
}

// -----

func.func @isnan(%input : tensor<1024xf32>) -> tensor<1024xi1> {
  // CHECK: math.isnan %{{.*}} : vector<1024xf32>
  %res = math.isnan %input : tensor<1024xf32>
  return %res : tensor<1024xi1>
}

// -----

func.func @isnormal(%input : tensor<1024xf32>) -> tensor<1024xi1> {
  // CHECK: math.isnormal %{{.*}} : vector<1024xf32>
  %res = math.isnormal %input : tensor<1024xf32>
  return %res : tensor<1024xi1>
}

// -----

func.func @log10(%input : tensor<1024xf32>) -> tensor<1024xf32> {
  // CHECK: math.log10 %{{.*}} : vector<1024xf32>
  %res = math.log10 %input : tensor<1024xf32>
  return %res : tensor<1024xf32>
}

// -----

func.func @log1p(%input : tensor<1024xf32>) -> tensor<1024xf32> {
  // CHECK: math.log1p %{{.*}} : vector<1024xf32>
  %res = math.log1p %input : tensor<1024xf32>
  return %res : tensor<1024xf32>
}

// -----

func.func @log2(%input : tensor<1024xf32>) -> tensor<1024xf32> {
  // CHECK: math.log2 %{{.*}} : vector<1024xf32>
  %res = math.log2 %input : tensor<1024xf32>
  return %res : tensor<1024xf32>
}

// -----

func.func @log(%input : tensor<1024xf32>) -> tensor<1024xf32> {
  // CHECK: math.log %{{.*}} : vector<1024xf32>
  %res = math.log %input : tensor<1024xf32>
  return %res : tensor<1024xf32>
}

// -----

func.func @powf(%base : tensor<1024xf32>, %exp : tensor<1024xf32>) -> tensor<1024xf32> {
  // CHECK: math.powf %{{.*}}, %{{.*}} : vector<1024xf32>
  %res = math.powf %base, %exp : tensor<1024xf32>
  return %res : tensor<1024xf32>
}

// -----

func.func @roundeven(%input : tensor<1024xf32>) -> tensor<1024xf32> {
  // CHECK: math.roundeven %{{.*}} : vector<1024xf32>
  %res = math.roundeven %input : tensor<1024xf32>
  return %res : tensor<1024xf32>
}

// -----

func.func @round(%input : tensor<1024xf32>) -> tensor<1024xf32> {
  // CHECK: math.round %{{.*}} : vector<1024xf32>
  %res = math.round %input : tensor<1024xf32>
  return %res : tensor<1024xf32>
}

// -----

func.func @rsqrt(%input : tensor<1024xf32>) -> tensor<1024xf32> {
  // CHECK: math.rsqrt %{{.*}} : vector<1024xf32>
  %res = math.rsqrt %input : tensor<1024xf32>
  return %res : tensor<1024xf32>
}

// -----

func.func @sin(%input : tensor<1024xf32>) -> tensor<1024xf32> {
  // CHECK: math.sin %{{.*}} : vector<1024xf32>
  %res = math.sin %input : tensor<1024xf32>
  return %res : tensor<1024xf32>
}

// -----

func.func @sincos(%input : tensor<1024xf32>) -> (tensor<1024xf32>, tensor<1024xf32>) {
  // CHECK: math.sincos %{{.*}} : vector<1024xf32>
  %sin, %cos = math.sincos %input : tensor<1024xf32>
  return %sin, %cos : tensor<1024xf32>, tensor<1024xf32>
}

// -----

func.func @sinh(%input : tensor<1024xf32>) -> tensor<1024xf32> {
  // CHECK: math.sinh %{{.*}} : vector<1024xf32>
  %res = math.sinh %input : tensor<1024xf32>
  return %res : tensor<1024xf32>
}

// -----

func.func @sqrt(%input : tensor<1024xf32>) -> tensor<1024xf32> {
  // CHECK: math.sqrt %{{.*}} : vector<1024xf32>
  %res = math.sqrt %input : tensor<1024xf32>
  return %res : tensor<1024xf32>
}

// -----

func.func @tan(%input : tensor<1024xf32>) -> tensor<1024xf32> {
  // CHECK: math.tan %{{.*}} : vector<1024xf32>
  %res = math.tan %input : tensor<1024xf32>
  return %res : tensor<1024xf32>
}

// -----

func.func @tanh(%input : tensor<1024xf32>) -> tensor<1024xf32> {
  // CHECK: math.tanh %{{.*}} : vector<1024xf32>
  %res = math.tanh %input : tensor<1024xf32>
  return %res : tensor<1024xf32>
}

// -----

func.func @trunc(%input : tensor<1024xf32>) -> tensor<1024xf32> {
  // CHECK: math.trunc %{{.*}} : vector<1024xf32>
  %res = math.trunc %input : tensor<1024xf32>
  return %res : tensor<1024xf32>
}