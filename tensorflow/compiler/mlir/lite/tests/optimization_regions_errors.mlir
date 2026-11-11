// RUN: litert-opt %s -tfl-mark-optimization-regions -split-input-file -verify-diagnostics

// Only `optimize = "none"` is understood. Anything else in
// `composite_attributes` is rejected when the regions are stamped, before
// anything else looks at them: the rest of the vocabulary (`disable`,
// `enable`, other `optimize` values) is reserved for per-entry selection, and
// a misspelt key must not silently mean "default".

func.func @main(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // expected-error @+1 {{odml.optimization_region: only optimize = "none" is supported; per-entry selection is a separate change (got {optimize = "some"})}}
  %0 = stablehlo.composite "odml.optimization_region" %arg0 {composite_attributes = {optimize = "some"}, decomposition = @body} : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}
func.func private @body(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  func.return %arg0 : tensor<2x2xf32>
}

// -----

func.func @main(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // expected-error @+1 {{odml.optimization_region: only optimize = "none" is supported; per-entry selection is a separate change (got {disable = ["bmm_to_fc"]})}}
  %0 = stablehlo.composite "odml.optimization_region" %arg0 {composite_attributes = {disable = ["bmm_to_fc"]}, decomposition = @body} : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}
func.func private @body(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  func.return %arg0 : tensor<2x2xf32>
}

// -----

func.func @main(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // expected-error @+1 {{odml.optimization_region: only optimize = "none" is supported; per-entry selection is a separate change (got {name = "attn", optimize = "none"})}}
  %0 = stablehlo.composite "odml.optimization_region" %arg0 {composite_attributes = {name = "attn", optimize = "none"}, decomposition = @body} : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}
func.func private @body(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  func.return %arg0 : tensor<2x2xf32>
}

// -----

// A decomposition shared by regions that disagree has no single answer.

func.func @main(%arg0: tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>) {
  %0 = stablehlo.composite "odml.optimization_region" %arg0 {composite_attributes = {optimize = "none"}, decomposition = @body} : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // expected-error @+1 {{odml.optimization_region decomposition @body is used with conflicting odml.optimization_region attributes}}
  %1 = stablehlo.composite "odml.optimization_region" %arg0 {decomposition = @body} : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0, %1 : tensor<2x2xf32>, tensor<2x2xf32>
}
func.func private @body(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  func.return %arg0 : tensor<2x2xf32>
}
