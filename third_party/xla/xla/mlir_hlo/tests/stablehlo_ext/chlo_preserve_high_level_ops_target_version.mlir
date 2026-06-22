// RUN: mlir-hlo-opt %s --stablehlo-ext-chlo-preserve-high-level-ops=target=1.13.0 --split-input-file | FileCheck %s --check-prefix=V13
// RUN: mlir-hlo-opt %s --stablehlo-ext-chlo-preserve-high-level-ops=target=1.14.0 --split-input-file | FileCheck %s --check-prefix=V14
// RUN: mlir-hlo-opt %s --stablehlo-ext-chlo-preserve-high-level-ops --split-input-file | FileCheck %s --check-prefix=V14

// chlo.scan is preserved as a region-bearing stablehlo.composite, which
// serializes as VHLO composite_v2 (StableHLO >= 1.14.0). For a target that
// predates 1.14.0 that composite cannot be downgraded, so the pass must leave
// chlo.scan in place (to be decomposed downstream). For >= 1.14.0, and when no
// target is given, it is still preserved as a composite.

// V13-LABEL: func @scan_target_gated
// V14-LABEL: func @scan_target_gated
func.func @scan_target_gated(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // V13-NOT: stablehlo.composite
  // V13: chlo.scan
  // V14: stablehlo.composite "chlo.scan"
  %0 = chlo.scan(%arg0) inits() dimension=0 {
  ^bb0(%arg1: tensor<2xf32>):
    stablehlo.return %arg1 : tensor<2xf32>
  } : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// -----

// A non-region CHLO op (chlo.erf maps to composite_v1) is representable before
// 1.14.0, so it is preserved as a composite regardless of target version.

// V13-LABEL: func @erf_preserved_any_target
// V14-LABEL: func @erf_preserved_any_target
func.func @erf_preserved_any_target(%arg0: tensor<3x20x20xbf16>) -> tensor<?x20x20xbf16> {
  // V13: stablehlo.composite "chlo.erf"
  // V14: stablehlo.composite "chlo.erf"
  %0 = chlo.erf %arg0 : tensor<3x20x20xbf16> -> tensor<?x20x20xbf16>
  return %0 : tensor<?x20x20xbf16>
}
