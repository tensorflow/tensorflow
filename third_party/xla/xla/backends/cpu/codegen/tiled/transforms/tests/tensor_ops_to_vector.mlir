// RUN: fusion_compiler_opt %s -xtile-cpu-tensor-ops-to-vector -split-input-file | FileCheck %s

func.func @from_elements(%input : f32) -> tensor<f32> {
  // CHECK: vector.from_elements %{{.*}} : vector<f32>
  %result = tensor.from_elements %input : tensor<f32>
  return %result : tensor<f32>
}

// -----

func.func @extract(%input : tensor<f32>) -> f32 {
  // CHECK: vector.extract %{{.*}}[] : f32 from vector<f32>
  %result = tensor.extract %input[] : tensor<f32>
  return %result : f32
}
