// RUN: mlir-hlo-opt %s -verify-diagnostics -split-input-file -allow-unregistered-dialect | FileCheck %s

// CHECK: func @select_and_scatter
func.func @select_and_scatter(
    %arg0: tensor<10x24x24x64xf32>,
    %arg1: tensor<10x12x12x64xf32>) -> tensor<10x24x24x64xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>

    %1 = "mhlo.select_and_scatter"(%arg0, %arg1, %0) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = "mhlo.compare"(%arg3, %arg4) {
        compare_type = #mhlo<comparison_type TOTALORDER>,
        comparison_direction = #mhlo<comparison_direction GE>
        } : (tensor<f32>, tensor<f32>) -> tensor<i1>
      "mhlo.return"(%2) : (tensor<i1>) -> ()
    },  {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%2) : (tensor<f32>) -> ()
    }) {
      window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>,
      window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>
    } : (tensor<10x24x24x64xf32>, tensor<10x12x12x64xf32>, tensor<f32>) ->
          tensor<10x24x24x64xf32>

    func.return %1 : tensor<10x24x24x64xf32>
}


// CHECK: func @select_and_scatter_with_promotable_types
func.func @select_and_scatter_with_promotable_types(
    %arg0: tensor<10x24x24x64xf32>,
    %arg1: tensor<10x12x12x64xf32>) -> () {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = "mhlo.select_and_scatter"(%arg0, %arg1, %0) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = "mhlo.compare"(%arg3, %arg4) {
        comparison_direction = #mhlo<comparison_direction GE>
        } : (tensor<f32>, tensor<f32>) -> tensor<i1>
      "mhlo.return"(%2) : (tensor<i1>) -> ()
    },  {
    ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
      %2 = mhlo.add %arg3, %arg4 : tensor<f64>
      "mhlo.return"(%2) : (tensor<f64>) -> ()
    }) {
      window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>,
      window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>,
      padding = dense<0> : tensor<4x2xi64>
    } : (tensor<10x24x24x64xf32>, tensor<10x12x12x64xf32>, tensor<f32>) ->
          tensor<10x24x24x64xf64>
    func.return
}


// CHECK: func @select_and_scatter_with_promotable_quantized_types
func.func @select_and_scatter_with_promotable_quantized_types(
  %arg0: tensor<10x24x24x64x!quant.uniform<i8:f32, 2.000000e+00:15>>,
  %arg1: tensor<10x12x12x64x!quant.uniform<i8:f32, 2.000000e+00:15>>,
  %arg2 : tensor<!quant.uniform<i8:f32, 2.000000e+00:15>>) ->
  tensor<10x24x24x64x!quant.uniform<i32:f32, 2.000000e+00:15>> {

  %1 = "mhlo.select_and_scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%arg3: tensor<!quant.uniform<i8:f32, 2.000000e+00:15>>, %arg4: tensor<!quant.uniform<i8:f32, 2.000000e+00:15>>):
    %2 = "mhlo.compare"(%arg3, %arg4) {
      compare_type = #mhlo<comparison_type TOTALORDER>,
      comparison_direction = #mhlo<comparison_direction GE>
      } : (tensor<!quant.uniform<i8:f32, 2.000000e+00:15>>, tensor<!quant.uniform<i8:f32, 2.000000e+00:15>>) -> tensor<i1>
    "mhlo.return"(%2) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg3: tensor<!quant.uniform<i32:f32, 2.000000e+00:15>>, %arg4: tensor<!quant.uniform<i32:f32, 2.000000e+00:15>>):
    %2 = mhlo.add %arg3, %arg4 : tensor<!quant.uniform<i32:f32, 2.000000e+00:15>>
    "mhlo.return"(%2) : (tensor<!quant.uniform<i32:f32, 2.000000e+00:15>>) -> ()
  }) {
    window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>,
    window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>
  } : (tensor<10x24x24x64x!quant.uniform<i8:f32, 2.000000e+00:15>>,
      tensor<10x12x12x64x!quant.uniform<i8:f32, 2.000000e+00:15>>,
      tensor<!quant.uniform<i8:f32, 2.000000e+00:15>>) ->
      tensor<10x24x24x64x!quant.uniform<i32:f32, 2.000000e+00:15>>
  func.return %1 : tensor<10x24x24x64x!quant.uniform<i32:f32, 2.000000e+00:15>>
}

// -----

func.func @select_and_scatter_invalid_select_computation(
    %arg0: tensor<10x24x24x64xf32>,
    %arg1: tensor<10x12x12x64xf32>) -> tensor<10x24x24x64xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>

    // expected-error @+1 {{expects the select-region to take 2 parameters, but takes 1}}
    %1 = "mhlo.select_and_scatter"(%arg0, %arg1, %0) ({
    ^bb0(%arg3: tensor<f32>):
      %2 = "mhlo.compare"(%arg3, %arg3) {
        compare_type = #mhlo<comparison_type TOTALORDER>,
        comparison_direction = #mhlo<comparison_direction GE>
        } : (tensor<f32>, tensor<f32>) -> tensor<i1>
      "mhlo.return"(%2) : (tensor<i1>) -> ()
    },  {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%2) : (tensor<f32>) -> ()
    }) {
      window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>,
      window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>
    } : (tensor<10x24x24x64xf32>, tensor<10x12x12x64xf32>, tensor<f32>) ->
          tensor<10x24x24x64xf32>

    func.return %1 : tensor<10x24x24x64xf32>
}

// -----

func.func @select_and_scatter_invalid_select_computation(
    %arg0: tensor<10x24x24x64xf32>,
    %arg1: tensor<10x12x12x64xf32>) -> tensor<10x24x24x64xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>

    // expected-error @+1 {{expects the type of select-region's parameter at index 0 to be 'tensor<f32>', but got 'tensor<i32>'}}
    %1 = "mhlo.select_and_scatter"(%arg0, %arg1, %0) ({
    ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
      %2 = "mhlo.compare"(%arg3, %arg4) {
        comparison_direction = #mhlo<comparison_direction GE>
        } : (tensor<i32>, tensor<i32>) -> tensor<i1>
      "mhlo.return"(%2) : (tensor<i1>) -> ()
    },  {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%2) : (tensor<f32>) -> ()
    }) {
      window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>,
      window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>
    } : (tensor<10x24x24x64xf32>, tensor<10x12x12x64xf32>, tensor<f32>) ->
          tensor<10x24x24x64xf32>

    func.return %1 : tensor<10x24x24x64xf32>
}

// -----

func.func @select_and_scatter_invalid_select_computation(
    %arg0: tensor<10x24x24x64xf32>,
    %arg1: tensor<10x12x12x64xf32>) -> tensor<10x24x24x64xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>

    // expected-error @+1 {{expects the type of select-region's parameter at index 0 to be 'tensor<f32>', but got 'tensor<1xf32>'}}
    %1 = "mhlo.select_and_scatter"(%arg0, %arg1, %0) ({
    ^bb0(%arg3: tensor<1xf32>, %arg4: tensor<1xf32>):
      %2 = "mhlo.compare"(%arg3, %arg4) {
        comparison_direction = #mhlo<comparison_direction GE>
        } : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
      %3 = "mhlo.reshape"(%2) : (tensor<1xi1>) -> tensor<i1>
      "mhlo.return"(%3) : (tensor<i1>) -> ()
    },  {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%2) : (tensor<f32>) -> ()
    }) {
      window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>,
      window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>
    } : (tensor<10x24x24x64xf32>, tensor<10x12x12x64xf32>, tensor<f32>) ->
          tensor<10x24x24x64xf32>

    func.return %1 : tensor<10x24x24x64xf32>
}

// -----

func.func @select_and_scatter_invalid_select_computation(
    %arg0: tensor<10x24x24x64xf32>,
    %arg1: tensor<10x12x12x64xf32>) -> tensor<10x24x24x64xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>

    // expected-error @+1 {{expects select-region to return single value, but got: 2}}
    %1 = "mhlo.select_and_scatter"(%arg0, %arg1, %0) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = "mhlo.compare"(%arg3, %arg4) {
        comparison_direction = #mhlo<comparison_direction GE>
        } : (tensor<f32>, tensor<f32>) -> tensor<i1>
      "mhlo.return"(%2, %2) : (tensor<i1>, tensor<i1>) -> ()
    },  {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%2) : (tensor<f32>) -> ()
    }) {
      window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>,
      window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>
    } : (tensor<10x24x24x64xf32>, tensor<10x12x12x64xf32>, tensor<f32>) ->
          tensor<10x24x24x64xf32>

    func.return %1 : tensor<10x24x24x64xf32>
}

// -----

func.func @select_and_scatter_invalid_select_computation(
    %arg0: tensor<10x24x24x64xf32>,
    %arg1: tensor<10x12x12x64xf32>) -> tensor<10x24x24x64xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>

    // expected-error @+1 {{expects the return-type of select-region to be tensor<i1>, but got: 'tensor<1xi1>'}}
    %1 = "mhlo.select_and_scatter"(%arg0, %arg1, %0) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = "mhlo.compare"(%arg3, %arg4) {
        comparison_direction = #mhlo<comparison_direction GE>
        } : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %3 = "mhlo.reshape"(%2) : (tensor<i1>) -> tensor<1xi1>
      "mhlo.return"(%3) : (tensor<1xi1>) -> ()
    },  {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%2) : (tensor<f32>) -> ()
    }) {
      window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>,
      window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>
    } : (tensor<10x24x24x64xf32>, tensor<10x12x12x64xf32>, tensor<f32>) ->
          tensor<10x24x24x64xf32>

    func.return %1 : tensor<10x24x24x64xf32>
}

// -----

func.func @select_and_scatter_invalid_select_computation(
    %arg0: tensor<10x24x24x64xf32>,
    %arg1: tensor<10x12x12x64xf32>) -> tensor<10x24x24x64xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>

    // expected-error @+1 {{expects the return-type of select-region to be tensor<i1>, but got: 'tuple<tensor<i1>>'}}
    %1 = "mhlo.select_and_scatter"(%arg0, %arg1, %0) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = "mhlo.compare"(%arg3, %arg4) {
        comparison_direction = #mhlo<comparison_direction GE>
        } : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %3 = "mhlo.tuple"(%2) : (tensor<i1>) -> tuple<tensor<i1>>
      "mhlo.return"(%3) : (tuple<tensor<i1>>) -> ()
    },  {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%2) : (tensor<f32>) -> ()
    }) {
      window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>,
      window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>
    } : (tensor<10x24x24x64xf32>, tensor<10x12x12x64xf32>, tensor<f32>) ->
          tensor<10x24x24x64xf32>

    func.return %1 : tensor<10x24x24x64xf32>
}

// -----

func.func @select_and_scatter_invalid_attributes(
    %arg0: tensor<10x24x24x64xf32>,
    %arg1: tensor<10x12x12x64xf32>) -> tensor<10x24x24x64xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>

    // expected-error @+1 {{expects window-dimensions size == operand rank, but got window-dimensions size: 0 and operand-type: 'tensor<10x24x24x64xf32>' with rank = 4.}}
    %1 = "mhlo.select_and_scatter"(%arg0, %arg1, %0) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = "mhlo.compare"(%arg3, %arg4) {
        comparison_direction = #mhlo<comparison_direction GE>
        } : (tensor<f32>, tensor<f32>) -> tensor<i1>
      "mhlo.return"(%2) : (tensor<i1>) -> ()
    },  {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%2) : (tensor<f32>) -> ()
    }) {
      window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>
    } : (tensor<10x24x24x64xf32>, tensor<10x12x12x64xf32>, tensor<f32>) ->
          tensor<10x24x24x64xf32>

    func.return %1 : tensor<10x24x24x64xf32>
}

// -----

func.func @select_and_scatter_invalid_attributes(
    %arg0: tensor<10x24x24x64xf32>,
    %arg1: tensor<10x12x12x64xf32>) -> tensor<10x24x24x64xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>

    // expected-error @+1 {{expects window-strides to have same dimension-size as size of window dimensions (4), but got: 3}}
    %1 = "mhlo.select_and_scatter"(%arg0, %arg1, %0) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = "mhlo.compare"(%arg3, %arg4) {
        comparison_direction = #mhlo<comparison_direction GE>
        } : (tensor<f32>, tensor<f32>) -> tensor<i1>
      "mhlo.return"(%2) : (tensor<i1>) -> ()
    },  {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%2) : (tensor<f32>) -> ()
    }) {
      window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>,
      window_strides = dense<[1, 2, 2]> : tensor<3xi64>
    } : (tensor<10x24x24x64xf32>, tensor<10x12x12x64xf32>, tensor<f32>) ->
          tensor<10x24x24x64xf32>

    func.return %1 : tensor<10x24x24x64xf32>
}

// -----

func.func @select_and_scatter_invalid_attributes(
    %arg0: tensor<10x24x24x64xf32>,
    %arg1: tensor<10x12x12x64xf32>) -> tensor<10x24x24x64xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>

    // expected-error @+1 {{expects padding-entries to have same dimension-size as size of window dimensions (4), but got: 3}}
    %1 = "mhlo.select_and_scatter"(%arg0, %arg1, %0) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = "mhlo.compare"(%arg3, %arg4) {
        comparison_direction = #mhlo<comparison_direction GE>
        } : (tensor<f32>, tensor<f32>) -> tensor<i1>
      "mhlo.return"(%2) : (tensor<i1>) -> ()
    },  {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%2) : (tensor<f32>) -> ()
    }) {
      window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>,
      window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>,
      padding = dense<0> : tensor<3x2xi64>
    } : (tensor<10x24x24x64xf32>, tensor<10x12x12x64xf32>, tensor<f32>) ->
          tensor<10x24x24x64xf32>

    func.return %1 : tensor<10x24x24x64xf32>
}

// -----

func.func @select_and_scatter_invalid_attributes(
    %arg0: tensor<10x24x24x64xf32>,
    %arg1: tensor<10x12x12x64xf32>) -> tensor<10x24x24x64xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>

    // expected-error @+1 {{expects the shape of padding-attribute to be {N, 2}, but got {4, 1}.}}
    %1 = "mhlo.select_and_scatter"(%arg0, %arg1, %0) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = "mhlo.compare"(%arg3, %arg4) {
        comparison_direction = #mhlo<comparison_direction GE>
        } : (tensor<f32>, tensor<f32>) -> tensor<i1>
      "mhlo.return"(%2) : (tensor<i1>) -> ()
    },  {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%2) : (tensor<f32>) -> ()
    }) {
      window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>,
      window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>,
      padding = dense<[[2], [2], [0], [0]]> : tensor<4x1xi64>
    } : (tensor<10x24x24x64xf32>, tensor<10x12x12x64xf32>, tensor<f32>) ->
          tensor<10x24x24x64xf32>

    func.return %1 : tensor<10x24x24x64xf32>
}

// -----

func.func @select_and_scatter_invalid_attributes(
    %arg0: tensor<10x24x24x64xf32>,
    %arg1: tensor<10x12x12x64xf32>) -> tensor<10x24x24x64xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>

    // expected-error @+1 {{expects window to have positive value for 3-th window dimension, but got 0.}}
    %1 = "mhlo.select_and_scatter"(%arg0, %arg1, %0) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = "mhlo.compare"(%arg3, %arg4) {
        comparison_direction = #mhlo<comparison_direction GE>
        } : (tensor<f32>, tensor<f32>) -> tensor<i1>
      "mhlo.return"(%2) : (tensor<i1>) -> ()
    },  {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%2) : (tensor<f32>) -> ()
    }) {
      window_dimensions = dense<[1, 2, 2, 0]> : tensor<4xi64>,
      window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>,
      padding = dense<0> : tensor<4x2xi64>
    } : (tensor<10x24x24x64xf32>, tensor<10x12x12x64xf32>, tensor<f32>) ->
          tensor<10x24x24x64xf32>

    func.return %1 : tensor<10x24x24x64xf32>
}

// -----

func.func @select_and_scatter_invalid_attributes(
    %arg0: tensor<10x24x24x64xf32>,
    %arg1: tensor<10x12x12x64xf32>) -> tensor<10x24x24x64xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>

    // expected-error @+1 {{expects window to have positive stride for 0-th window dimension, but got 0.}}
    %1 = "mhlo.select_and_scatter"(%arg0, %arg1, %0) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = "mhlo.compare"(%arg3, %arg4) {
        comparison_direction = #mhlo<comparison_direction GE>
        } : (tensor<f32>, tensor<f32>) -> tensor<i1>
      "mhlo.return"(%2) : (tensor<i1>) -> ()
    },  {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%2) : (tensor<f32>) -> ()
    }) {
      window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>,
      window_strides = dense<[0, 2, 2, 1]> : tensor<4xi64>,
      padding = dense<0> : tensor<4x2xi64>
    } : (tensor<10x24x24x64xf32>, tensor<10x12x12x64xf32>, tensor<f32>) ->
          tensor<10x24x24x64xf32>

    func.return %1 : tensor<10x24x24x64xf32>
}

// -----

func.func @select_and_scatter_invalid_ret_type(
    %arg0: tensor<10x24x24x64xf32>,
    %arg1: tensor<10x12x12x64xf32>) -> () {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>

    // expected-error @+2 {{'mhlo.select_and_scatter' op failed to infer returned types}}
    // expected-error @+1 {{inferred type(s) 'tensor<10x24x24x64xf32>' are incompatible with return type(s) of operation 'tensor<10x24x24x32xf32>'}}
    %1 = "mhlo.select_and_scatter"(%arg0, %arg1, %0) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = "mhlo.compare"(%arg3, %arg4) {
        comparison_direction = #mhlo<comparison_direction GE>
        } : (tensor<f32>, tensor<f32>) -> tensor<i1>
      "mhlo.return"(%2) : (tensor<i1>) -> ()
    },  {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%2) : (tensor<f32>) -> ()
    }) {
      window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>,
      window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>,
      padding = dense<0> : tensor<4x2xi64>
    } : (tensor<10x24x24x64xf32>, tensor<10x12x12x64xf32>, tensor<f32>) ->
          tensor<10x24x24x32xf32>

    func.return
}

// -----

func.func @select_and_scatter_invalid_ret_type(
    %arg0: tensor<10x24x24x64xf32>,
    %arg1: tensor<10x12x12x64xf32>) -> () {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>

    // expected-error @+2 {{'mhlo.select_and_scatter' op failed to infer returned types}}
    // expected-error @+1 {{inferred type(s) 'tensor<10x24x24x64xf32>' are incompatible with return type(s) of operation 'tensor<10x24x24x64xi32>'}}
    %1 = "mhlo.select_and_scatter"(%arg0, %arg1, %0) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = "mhlo.compare"(%arg3, %arg4) {
        comparison_direction = #mhlo<comparison_direction GE>
        } : (tensor<f32>, tensor<f32>) -> tensor<i1>
      "mhlo.return"(%2) : (tensor<i1>) -> ()
    },  {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%2) : (tensor<f32>) -> ()
    }) {
      window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>,
      window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>,
      padding = dense<0> : tensor<4x2xi64>
    } : (tensor<10x24x24x64xf32>, tensor<10x12x12x64xf32>, tensor<f32>) ->
          tensor<10x24x24x64xi32>

    func.return
}

// -----

func.func @select_and_scatter_invalid_scatter_reducer(
    %arg0: tensor<10x24x24x64xf32>,
    %arg1: tensor<10x12x12x64xf32>) -> () {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>

    // expected-error @+1 {{Reduction-region must take 2 parameters, but takes 1 parameter(s)}}
    %1 = "mhlo.select_and_scatter"(%arg0, %arg1, %0) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = "mhlo.compare"(%arg3, %arg4) {
        comparison_direction = #mhlo<comparison_direction GE>
        } : (tensor<f32>, tensor<f32>) -> tensor<i1>
      "mhlo.return"(%2) : (tensor<i1>) -> ()
    },  {
    ^bb0(%arg3: tensor<f32>):
      %2 = mhlo.add %arg3, %arg3 : tensor<f32>
      "mhlo.return"(%2) : (tensor<f32>) -> ()
    }) {
      window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>,
      window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>,
      padding = dense<0> : tensor<4x2xi64>
    } : (tensor<10x24x24x64xf32>, tensor<10x12x12x64xf32>, tensor<f32>) ->
          tensor<10x24x24x64xf32>

    func.return
}

// -----

func.func @select_and_scatter_invalid_scatter_reducer(
    %arg0: tensor<10x24x24x64xf32>,
    %arg1: tensor<10x12x12x64xf32>) -> () {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>

    // expected-error @+1 {{The reduction-region expected to return some value(s)}}
    %1 = "mhlo.select_and_scatter"(%arg0, %arg1, %0) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = "mhlo.compare"(%arg3, %arg4) {
        comparison_direction = #mhlo<comparison_direction GE>
        } : (tensor<f32>, tensor<f32>) -> tensor<i1>
      "mhlo.return"(%2) : (tensor<i1>) -> ()
    },  {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"() : () -> ()
    }) {
      window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>,
      window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>,
      padding = dense<0> : tensor<4x2xi64>
    } : (tensor<10x24x24x64xf32>, tensor<10x12x12x64xf32>, tensor<f32>) ->
          tensor<10x24x24x64xf32>

    func.return
}

// -----

func.func @select_and_scatter_invalid_scatter_reducer(
    %arg0: tensor<10x24x24x64xf32>,
    %arg1: tensor<10x12x12x64xf32>) -> () {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>

    // expected-error @+1 {{Reduction-region here must produce 1 tensors, but produces 2 instead}}
    %1 = "mhlo.select_and_scatter"(%arg0, %arg1, %0) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = "mhlo.compare"(%arg3, %arg4) {
        comparison_direction = #mhlo<comparison_direction GE>
        } : (tensor<f32>, tensor<f32>) -> tensor<i1>
      "mhlo.return"(%2) : (tensor<i1>) -> ()
    },  {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%2, %2) : (tensor<f32>, tensor<f32>) -> ()
    }) {
      window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>,
      window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>,
      padding = dense<0> : tensor<4x2xi64>
    } : (tensor<10x24x24x64xf32>, tensor<10x12x12x64xf32>, tensor<f32>) ->
          tensor<10x24x24x64xf32>

    func.return
}

// -----

func.func @select_and_scatter_invalid_scatter_reducer(
    %arg0: tensor<10x24x24x64xf32>,
    %arg1: tensor<10x12x12x64xf32>) -> () {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>

    // expected-error @+1 {{Reduction-region here must produce tensor-typed result(s), but produces 'tuple<tensor<f32>>' instead}}
    %1 = "mhlo.select_and_scatter"(%arg0, %arg1, %0) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = "mhlo.compare"(%arg3, %arg4) {
        comparison_direction = #mhlo<comparison_direction GE>
        } : (tensor<f32>, tensor<f32>) -> tensor<i1>
      "mhlo.return"(%2) : (tensor<i1>) -> ()
    },  {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = mhlo.add %arg3, %arg4 : tensor<f32>
      %3 = "mhlo.tuple"(%2) : (tensor<f32>) -> tuple<tensor<f32>>
      "mhlo.return"(%3) : (tuple<tensor<f32>>) -> ()
    }) {
      window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>,
      window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>,
      padding = dense<0> : tensor<4x2xi64>
    } : (tensor<10x24x24x64xf32>, tensor<10x12x12x64xf32>, tensor<f32>) ->
          tensor<10x24x24x64xf32>

    func.return
}

// -----

func.func @select_and_scatter_invalid_scatter_reducer(
    %arg0: tensor<10x24x24x64xf32>,
    %arg1: tensor<10x12x12x64xf32>) -> () {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>

    // expected-error @+1 {{Reduction-region here must produce tensor-typed result(s), but produces 'f32' instead}}
    %1 = "mhlo.select_and_scatter"(%arg0, %arg1, %0) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = "mhlo.compare"(%arg3, %arg4) {
        comparison_direction = #mhlo<comparison_direction GE>
        } : (tensor<f32>, tensor<f32>) -> tensor<i1>
      "mhlo.return"(%2) : (tensor<i1>) -> ()
    },  {
    ^bb0(%arg3: f32, %arg4: f32):
      %2 = "llvm.add" (%arg3, %arg4) : (f32, f32) -> f32
      "mhlo.return"(%2) : (f32) -> ()
    }) {
      window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>,
      window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>,
      padding = dense<0> : tensor<4x2xi64>
    } : (tensor<10x24x24x64xf32>, tensor<10x12x12x64xf32>, tensor<f32>) ->
          tensor<10x24x24x64xf32>

    func.return
}

// -----

func.func @select_and_scatter_invalid_scatter_reducer(
    %arg0: tensor<10x24x24x64xf32>,
    %arg1: tensor<10x12x12x64xf32>) -> () {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>

    // expected-error @+1 {{The type of reduction-region's parameter at index 0 is different than the corresponding result type: 'tensor<f32>' vs 'tensor<i32>'}}
    %1 = "mhlo.select_and_scatter"(%arg0, %arg1, %0) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = "mhlo.compare"(%arg3, %arg4) {
        comparison_direction = #mhlo<comparison_direction GE>
        } : (tensor<f32>, tensor<f32>) -> tensor<i1>
      "mhlo.return"(%2) : (tensor<i1>) -> ()
    },  {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = mhlo.constant dense<0> : tensor<i32>
      "mhlo.return"(%2) : (tensor<i32>) -> ()
    }) {
      window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>,
      window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>,
      padding = dense<0> : tensor<4x2xi64>
    } : (tensor<10x24x24x64xf32>, tensor<10x12x12x64xf32>, tensor<f32>) ->
          tensor<10x24x24x64xf32>

    func.return
}

// -----

func.func @select_and_scatter_invalid_scatter_reducer(
    %arg0: tensor<10x24x24x64xf32>,
    %arg1: tensor<10x12x12x64xf32>) -> () {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>

    // expected-error @+1 {{The type of reduction-region's parameter at index 1 is different than the corresponding result type: 'tensor<i32>' vs 'tensor<f32>'}}
    %1 = "mhlo.select_and_scatter"(%arg0, %arg1, %0) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = "mhlo.compare"(%arg3, %arg4) {
        comparison_direction = #mhlo<comparison_direction GE>
        } : (tensor<f32>, tensor<f32>) -> tensor<i1>
      "mhlo.return"(%2) : (tensor<i1>) -> ()
    },  {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<i32>):
      %2 = mhlo.add %arg3, %arg3 : tensor<f32>
      "mhlo.return"(%2) : (tensor<f32>) -> ()
    }) {
      window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>,
      window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>,
      padding = dense<0> : tensor<4x2xi64>
    } : (tensor<10x24x24x64xf32>, tensor<10x12x12x64xf32>, tensor<f32>) ->
          tensor<10x24x24x64xf32>

    func.return
}

// -----

func.func @select_and_scatter_invalid_scatter_reducer(
    %arg0: tensor<10x24x24x64xf32>,
    %arg1: tensor<10x12x12x64xf32>) -> () {
    %0 = mhlo.constant dense<0> : tensor<i32>

    // expected-error @+1 {{The element-type of reduction-region's argument at index 1 is expected to be promotable from 'f32', but got 'i32'}}
    %1 = "mhlo.select_and_scatter"(%arg0, %arg1, %0) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = "mhlo.compare"(%arg3, %arg4) {
        comparison_direction = #mhlo<comparison_direction GE>
        } : (tensor<f32>, tensor<f32>) -> tensor<i1>
      "mhlo.return"(%2) : (tensor<i1>) -> ()
    },  {
    ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
      %2 = mhlo.add %arg3, %arg4 : tensor<i32>
      "mhlo.return"(%2) : (tensor<i32>) -> ()
    }) {
      window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>,
      window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>,
      padding = dense<0> : tensor<4x2xi64>
    } : (tensor<10x24x24x64xf32>, tensor<10x12x12x64xf32>, tensor<i32>) ->
          tensor<10x24x24x64xf32>

    func.return
}

// -----

func.func @select_and_scatter_invalid_scatter_reducer(
    %arg0: tensor<10x24x24x64xf32>,
    %arg1: tensor<10x12x12x64xf32>) -> () {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>

    // expected-error @+1 {{The element-type of reduction-region's result type at index 0 is expected to be promotable from the op's corresponding init-value element-type: 'tensor<i32>' vs 'tensor<f32>'}}
    %1 = "mhlo.select_and_scatter"(%arg0, %arg1, %0) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = "mhlo.compare"(%arg3, %arg4) {
        comparison_direction = #mhlo<comparison_direction GE>
        } : (tensor<f32>, tensor<f32>) -> tensor<i1>
      "mhlo.return"(%2) : (tensor<i1>) -> ()
    },  {
    ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
      %2 = mhlo.add %arg3, %arg4 : tensor<i32>
      "mhlo.return"(%2) : (tensor<i32>) -> ()
    }) {
      window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>,
      window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>,
      padding = dense<0> : tensor<4x2xi64>
    } : (tensor<10x24x24x64xf32>, tensor<10x12x12x64xf32>, tensor<f32>) ->
          tensor<10x24x24x64xf32>

    func.return
}

// -----

func.func @select_and_scatter_invalid_scatter_reducer(
    %arg0: tensor<10x24x24x64xf32>,
    %arg1: tensor<10x12x12x64xf32>) -> () {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<1xf32>

    // expected-error @+1 {{The rank of reduction-region's argument at index 1 is expected to be <= 0, got 1}}
    %1 = "mhlo.select_and_scatter"(%arg0, %arg1, %0) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = "mhlo.compare"(%arg3, %arg4) {
        comparison_direction = #mhlo<comparison_direction GE>
        } : (tensor<f32>, tensor<f32>) -> tensor<i1>
      "mhlo.return"(%2) : (tensor<i1>) -> ()
    },  {
    ^bb0(%arg3: tensor<1xf32>, %arg4: tensor<1xf32>):
      %2 = mhlo.add %arg3, %arg4 : tensor<1xf32>
      "mhlo.return"(%2) : (tensor<1xf32>) -> ()
    }) {
      window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>,
      window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>,
      padding = dense<0> : tensor<4x2xi64>
    } : (tensor<10x24x24x64xf32>, tensor<10x12x12x64xf32>, tensor<1xf32>) ->
          tensor<10x24x24x64xf32>

    func.return
}

// -----

func.func @select_and_scatter_invalid_source_operand(
    %arg0: tensor<10x24x24x64xf32>,
    %arg1: tensor<10x12x12x32xf32>) -> () {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>

    // expected-error @+1 {{expects source-type to be 'tensor<10x12x12x64xf32>', but got'tensor<10x12x12x32xf32>'}}
    %1 = "mhlo.select_and_scatter"(%arg0, %arg1, %0) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = "mhlo.compare"(%arg3, %arg4) {
        comparison_direction = #mhlo<comparison_direction GE>
        } : (tensor<f32>, tensor<f32>) -> tensor<i1>
      "mhlo.return"(%2) : (tensor<i1>) -> ()
    },  {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%2) : (tensor<f32>) -> ()
    }) {
      window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>,
      window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>,
      padding = dense<0> : tensor<4x2xi64>
    } : (tensor<10x24x24x64xf32>, tensor<10x12x12x32xf32>, tensor<f32>) ->
          tensor<10x24x24x64xf32>

    func.return
}

// -----

func.func @select_and_scatter_invalid_source_operand(
    %arg0: tensor<10x24x24x64xf32>,
    %arg1: tensor<10x12x12x64xi32>) -> () {
    %0 = mhlo.constant dense<0> : tensor<i32>

    // expected-error @+1 {{expects source-type to be 'tensor<10x12x12x64xf32>', but got'tensor<10x12x12x64xi32>'}}
    %1 = "mhlo.select_and_scatter"(%arg0, %arg1, %0) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = "mhlo.compare"(%arg3, %arg4) {
        comparison_direction = #mhlo<comparison_direction GE>
        } : (tensor<f32>, tensor<f32>) -> tensor<i1>
      "mhlo.return"(%2) : (tensor<i1>) -> ()
    },  {
    ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
      %2 = mhlo.add %arg3, %arg4 : tensor<i32>
      "mhlo.return"(%2) : (tensor<i32>) -> ()
    }) {
      window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>,
      window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>,
      padding = dense<0> : tensor<4x2xi64>
    } : (tensor<10x24x24x64xf32>, tensor<10x12x12x64xi32>, tensor<i32>) ->
          tensor<10x24x24x64xf32>

    func.return
}
