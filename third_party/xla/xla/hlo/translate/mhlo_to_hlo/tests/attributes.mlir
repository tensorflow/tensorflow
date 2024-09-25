// RUN: xla-translate -split-input-file -mlir-hlo-to-hlo-text %s | FileCheck %s

// CHECK-LABEL: HloModule dot_algorithm_f8_f8_f32
module @dot_algorithm_f8_f8_f32 {
  func.func @main(%arg0: tensor<2x2x2xf32>, %arg1: tensor<2x2x2xf32>) -> tensor<2x2x2xf32> {
    // CHECK: f32[2,2,2] dot(f32[2,2,2] {{.*}}, f32[2,2,2] {{.*}}), {{.*}}, algorithm=dot_any_f8_any_f8_f32
    %0 = "mhlo.dot_general"(%arg0, %arg1) <{
      dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>,
      precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>],
      algorithm = #mhlo.dot_algorithm<
        lhs_precision_type = f8E4M3FNUZ,
        rhs_precision_type = f8E4M3FNUZ,
        accumulation_type = f32,
        lhs_component_count = 1,
        rhs_component_count = 1,
        num_primitive_operations = 1,
        allow_imprecise_accumulation = false
      >
    }> : (tensor<2x2x2xf32>, tensor<2x2x2xf32>) -> tensor<2x2x2xf32>  return %0 : tensor<2x2x2xf32>
  }
}

// -----

// CHECK-LABEL: HloModule dot_algorithm_f8_f8_f32_fast_accum
module @dot_algorithm_f8_f8_f32_fast_accum {
  func.func @main(%arg0: tensor<2x2x2xf32>, %arg1: tensor<2x2x2xf32>) -> tensor<2x2x2xf32> {
    // CHECK: f32[2,2,2] dot(f32[2,2,2] {{.*}}, f32[2,2,2] {{.*}}), {{.*}}, algorithm=dot_any_f8_any_f8_f32_fast_accum
    %0 = "mhlo.dot_general"(%arg0, %arg1) <{
      dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>,
      precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>],
      algorithm = #mhlo.dot_algorithm<
        lhs_precision_type = f8E4M3FNUZ,
        rhs_precision_type = f8E4M3FNUZ,
        accumulation_type = f32,
        lhs_component_count = 1,
        rhs_component_count = 1,
        num_primitive_operations = 1,
        allow_imprecise_accumulation = true
      >
    }> : (tensor<2x2x2xf32>, tensor<2x2x2xf32>) -> tensor<2x2x2xf32>  return %0 : tensor<2x2x2xf32>
  }
}

// -----

// CHECK-LABEL: HloModule dot_algorithm_f16_f16_f16
module @dot_algorithm_f16_f16_f16 {
  func.func @main(%arg0: tensor<2x2x2xf32>, %arg1: tensor<2x2x2xf32>) -> tensor<2x2x2xf32> {
    // CHECK: f32[2,2,2] dot(f32[2,2,2] {{.*}}, f32[2,2,2] {{.*}}), {{.*}}, algorithm=dot_f16_f16_f16
    %0 = "mhlo.dot_general"(%arg0, %arg1) <{
      dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>,
      precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>],
      algorithm = #mhlo.dot_algorithm<
        lhs_precision_type = f16,
        rhs_precision_type = f16,
        accumulation_type = f16,
        lhs_component_count = 1,
        rhs_component_count = 1,
        num_primitive_operations = 1,
        allow_imprecise_accumulation = false
      >
    }> : (tensor<2x2x2xf32>, tensor<2x2x2xf32>) -> tensor<2x2x2xf32>  return %0 : tensor<2x2x2xf32>
  }
}

// -----

// CHECK-LABEL: HloModule dot_algorithm_f16_f16_f32
module @dot_algorithm_f16_f16_f32 {
  func.func @main(%arg0: tensor<2x2x2xf32>, %arg1: tensor<2x2x2xf32>) -> tensor<2x2x2xf32> {
    // CHECK: f32[2,2,2] dot(f32[2,2,2] {{.*}}, f32[2,2,2] {{.*}}), {{.*}}, algorithm=dot_f16_f16_f32
    %0 = "mhlo.dot_general"(%arg0, %arg1) <{
      dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>,
      precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>],
      algorithm = #mhlo.dot_algorithm<
        lhs_precision_type = f16,
        rhs_precision_type = f16,
        accumulation_type = f32,
        lhs_component_count = 1,
        rhs_component_count = 1,
        num_primitive_operations = 1,
        allow_imprecise_accumulation = false
      >
    }> : (tensor<2x2x2xf32>, tensor<2x2x2xf32>) -> tensor<2x2x2xf32>  return %0 : tensor<2x2x2xf32>
  }
}

// -----

// CHECK-LABEL: HloModule dot_algorithm_bf16_bf16_bf16
module @dot_algorithm_bf16_bf16_bf16 {
  func.func @main(%arg0: tensor<2x2x2xbf16>, %arg1: tensor<2x2x2xbf16>) -> tensor<2x2x2xbf16> {
    // CHECK: bf16[2,2,2] dot(bf16[2,2,2] {{.*}}, bf16[2,2,2] {{.*}}), {{.*}}, algorithm=dot_bf16_bf16_bf16
    %0 = "mhlo.dot_general"(%arg0, %arg1) <{
      dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>,
      precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>],
      algorithm = #mhlo.dot_algorithm<
        lhs_precision_type = bf16,
        rhs_precision_type = bf16,
        accumulation_type = bf16,
        lhs_component_count = 1,
        rhs_component_count = 1,
        num_primitive_operations = 1,
        allow_imprecise_accumulation = false
      >
    }> : (tensor<2x2x2xbf16>, tensor<2x2x2xbf16>) -> tensor<2x2x2xbf16>  return %0 : tensor<2x2x2xbf16>
  }
}

// -----

// CHECK-LABEL: HloModule dot_algorithm_bf16_bf16_f32
module @dot_algorithm_bf16_bf16_f32 {
  func.func @main(%arg0: tensor<2x2x2xbf16>, %arg1: tensor<2x2x2xbf16>) -> tensor<2x2x2xbf16> {
    // CHECK: bf16[2,2,2] dot(bf16[2,2,2] {{.*}}, bf16[2,2,2] {{.*}}), {{.*}}, algorithm=dot_bf16_bf16_f32
    %0 = "mhlo.dot_general"(%arg0, %arg1) <{
      dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>,
      precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>],
      algorithm = #mhlo.dot_algorithm<
        lhs_precision_type = bf16,
        rhs_precision_type = bf16,
        accumulation_type = f32,
        lhs_component_count = 1,
        rhs_component_count = 1,
        num_primitive_operations = 1,
        allow_imprecise_accumulation = false
      >
    }> : (tensor<2x2x2xbf16>, tensor<2x2x2xbf16>) -> tensor<2x2x2xbf16>  return %0 : tensor<2x2x2xbf16>
  }
}

// -----

// CHECK-LABEL: HloModule dot_algorithm_bf16_bf16_f32_x3
module @dot_algorithm_bf16_bf16_f32_x3 {
  func.func @main(%arg0: tensor<2x2x2xbf16>, %arg1: tensor<2x2x2xbf16>) -> tensor<2x2x2xbf16> {
    // CHECK: bf16[2,2,2] dot(bf16[2,2,2] {{.*}}, bf16[2,2,2] {{.*}}), {{.*}}, algorithm=dot_bf16_bf16_f32_x3
    %0 = "mhlo.dot_general"(%arg0, %arg1) <{
      dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>,
      precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>],
      algorithm = #mhlo.dot_algorithm<
        lhs_precision_type = bf16,
        rhs_precision_type = bf16,
        accumulation_type = f32,
        lhs_component_count = 1,
        rhs_component_count = 1,
        num_primitive_operations = 3,
        allow_imprecise_accumulation = false
      >
    }> : (tensor<2x2x2xbf16>, tensor<2x2x2xbf16>) -> tensor<2x2x2xbf16>  return %0 : tensor<2x2x2xbf16>
  }
}

// -----

// CHECK-LABEL: HloModule dot_algorithm_bf16_bf16_f32_x6
module @dot_algorithm_bf16_bf16_f32_x6 {
  func.func @main(%arg0: tensor<2x2x2xbf16>, %arg1: tensor<2x2x2xbf16>) -> tensor<2x2x2xbf16> {
    // CHECK: bf16[2,2,2] dot(bf16[2,2,2] {{.*}}, bf16[2,2,2] {{.*}}), {{.*}}, algorithm=dot_bf16_bf16_f32_x6
    %0 = "mhlo.dot_general"(%arg0, %arg1) <{
      dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>,
      precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>],
      algorithm = #mhlo.dot_algorithm<
        lhs_precision_type = bf16,
        rhs_precision_type = bf16,
        accumulation_type = f32,
        lhs_component_count = 1,
        rhs_component_count = 1,
        num_primitive_operations = 6,
        allow_imprecise_accumulation = false
      >
    }> : (tensor<2x2x2xbf16>, tensor<2x2x2xbf16>) -> tensor<2x2x2xbf16>  return %0 : tensor<2x2x2xbf16>
  }
}

// -----

// CHECK-LABEL: HloModule dot_algorithm_tf32_tf32_f32
module @dot_algorithm_tf32_tf32_f32 {
  func.func @main(%arg0: tensor<2x2x2xf32>, %arg1: tensor<2x2x2xf32>) -> tensor<2x2x2xf32> {
    // CHECK: f32[2,2,2] dot(f32[2,2,2] {{.*}}, f32[2,2,2] {{.*}}), {{.*}}, algorithm=dot_tf32_tf32_f32
    %0 = "mhlo.dot_general"(%arg0, %arg1) <{
      dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>,
      precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>],
      algorithm = #mhlo.dot_algorithm<
        lhs_precision_type = tf32,
        rhs_precision_type = tf32,
        accumulation_type = f32,
        lhs_component_count = 1,
        rhs_component_count = 1,
        num_primitive_operations = 1,
        allow_imprecise_accumulation = false
      >
    }> : (tensor<2x2x2xf32>, tensor<2x2x2xf32>) -> tensor<2x2x2xf32>  return %0 : tensor<2x2x2xf32>
  }
}

// -----

// CHECK-LABEL: HloModule dot_algorithm_tf32_tf32_f32_x3
module @dot_algorithm_tf32_tf32_f32_x3 {
  func.func @main(%arg0: tensor<2x2x2xf32>, %arg1: tensor<2x2x2xf32>) -> tensor<2x2x2xf32> {
    // CHECK: f32[2,2,2] dot(f32[2,2,2] {{.*}}, f32[2,2,2] {{.*}}), {{.*}}, algorithm=dot_tf32_tf32_f32_x3
    %0 = "mhlo.dot_general"(%arg0, %arg1) <{
      dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>,
      precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>],
      algorithm = #mhlo.dot_algorithm<
        lhs_precision_type = tf32,
        rhs_precision_type = tf32,
        accumulation_type = f32,
        lhs_component_count = 1,
        rhs_component_count = 1,
        num_primitive_operations = 3,
        allow_imprecise_accumulation = false
      >
    }> : (tensor<2x2x2xf32>, tensor<2x2x2xf32>) -> tensor<2x2x2xf32>  return %0 : tensor<2x2x2xf32>
  }
}

// -----

// CHECK-LABEL: HloModule dot_algorithm_f32_f32_f32
module @dot_algorithm_f32_f32_f32 {
  func.func @main(%arg0: tensor<2x2x2xf32>, %arg1: tensor<2x2x2xf32>) -> tensor<2x2x2xf32> {
    // CHECK: f32[2,2,2] dot(f32[2,2,2] {{.*}}, f32[2,2,2] {{.*}}), {{.*}}, algorithm=dot_f32_f32_f32
    %0 = "mhlo.dot_general"(%arg0, %arg1) <{
      dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>,
      precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>],
      algorithm = #mhlo.dot_algorithm<
        lhs_precision_type = f32,
        rhs_precision_type = f32,
        accumulation_type = f32,
        lhs_component_count = 1,
        rhs_component_count = 1,
        num_primitive_operations = 1,
        allow_imprecise_accumulation = false
      >
    }> : (tensor<2x2x2xf32>, tensor<2x2x2xf32>) -> tensor<2x2x2xf32>  return %0 : tensor<2x2x2xf32>
  }
}

// -----

// CHECK-LABEL: HloModule dot_algorithm_f64_f64_f64
module @dot_algorithm_f64_f64_f64 {
  func.func @main(%arg0: tensor<2x2x2xf64>, %arg1: tensor<2x2x2xf64>) -> tensor<2x2x2xf64> {
    // CHECK: f64[2,2,2] dot(f64[2,2,2] {{.*}}, f64[2,2,2] {{.*}}), {{.*}}, algorithm=dot_f64_f64_f64
    %0 = "mhlo.dot_general"(%arg0, %arg1) <{
      dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>,
      precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>],
      algorithm = #mhlo.dot_algorithm<
        lhs_precision_type = f64,
        rhs_precision_type = f64,
        accumulation_type = f64,
        lhs_component_count = 1,
        rhs_component_count = 1,
        num_primitive_operations = 1,
        allow_imprecise_accumulation = false
      >
    }> : (tensor<2x2x2xf64>, tensor<2x2x2xf64>) -> tensor<2x2x2xf64>  return %0 : tensor<2x2x2xf64>
  }
}
