// RUN: mlir-hlo-opt %s -verify-diagnostics -split-input-file -allow-unregistered-dialect | FileCheck %s

// CHECK: func @reduce_window
func.func @reduce_window(%arg0: tensor<4x2xf32>, %arg1: tensor<4x2xi32>,
                    %init0: tensor<f32>, %init1: tensor<i32>) ->
                      (tensor<2x2xf32>, tensor<2x2xi32>) {
  %0:2 = "mhlo.reduce_window"(%arg0, %arg1, %init0, %init1) ({
         ^bb0(%a0: tensor<f32>, %a1: tensor<i32>,
                %b0: tensor<f32>, %b1: tensor<i32>):
              %2 = mhlo.add %a0, %b0 : tensor<f32>
              %3 = mhlo.add %a1, %b1 : tensor<i32>
              "mhlo.return"(%2, %3) : (tensor<f32>, tensor<i32>) -> ()
            })
         { padding = dense<[[2, 2], [0, 0]]> : tensor<2x2xi64>,
           window_dimensions = dense<[5, 1]> : tensor<2xi64>,
           window_strides = dense<[3, 1]> : tensor<2xi64> }
         : (tensor<4x2xf32>, tensor<4x2xi32>, tensor<f32>, tensor<i32>) ->
              (tensor<2x2xf32>, tensor<2x2xi32>)
  func.return %0#0, %0#1 : tensor<2x2xf32>, tensor<2x2xi32>
}

// -----

func.func @reduce_window_with_unranked_dynamic_dims(%arg0: tensor<*xf32>,
    %arg1: tensor<4x?xi32>, %init0: tensor<f32>, %init1: tensor<i32>) ->
        (tensor<?x?xf32>, tensor<*xi32>) {
  %0:2 = "mhlo.reduce_window"(%arg0, %arg1, %init0, %init1) ({
         ^bb0(%a0: tensor<f32>, %a1: tensor<i32>,
                %b0: tensor<f32>, %b1: tensor<i32>):
              %2 = mhlo.add %a0, %b0 : tensor<f32>
              %3 = mhlo.add %a1, %b1 : tensor<i32>
              "mhlo.return"(%2, %3) : (tensor<f32>, tensor<i32>) -> ()
            })
         { padding = dense<[[2, 2], [0, 0]]> : tensor<2x2xi64>,
           window_dimensions = dense<[5, 1]> : tensor<2xi64>,
           window_strides = dense<[3, 1]> : tensor<2xi64>,
           base_dilations = dense<[1,1]> : tensor<2xi64>,
           window_dilations = dense<[1,1]> : tensor<2xi64> }
         : (tensor<*xf32>, tensor<4x?xi32>, tensor<f32>, tensor<i32>) ->
              (tensor<?x?xf32>, tensor<*xi32>)
  func.return %0#0, %0#1 : tensor<?x?xf32>, tensor<*xi32>
}

// -----

func.func @reduce_window_with_non_scalar_block_arg1(%arg0: tensor<4x2xf32>,
    %init0: tensor<4xf32>) -> tensor<2x1xf32> {
  %0 = "mhlo.reduce_window"(%arg0, %init0) ({
         ^bb0(%a0: tensor<4xf32>, %b0: tensor<4xf32>):
              %2 = mhlo.add %a0, %b0 : tensor<4xf32>
              "mhlo.return"(%2) : (tensor<4xf32>) -> ()
            })
         {
           padding = dense<[[2, 2], [0, 0]]> : tensor<2x2xi64>,
           window_dimensions = dense<[4, 2]> : tensor<2xi64>,
           window_strides = dense<[3, 1]> : tensor<2xi64>
         }
         : (tensor<4x2xf32>, tensor<4xf32>) -> (tensor<2x1xf32>)
  func.return %0 : tensor<2x1xf32>
}

// -----

func.func @reduce_window_with_non_scalar_block_arg2(%arg0: tensor<4x2xf32>,
    %init0: tensor<2xf32>) -> tensor<2x1xf32> {
  %0 = "mhlo.reduce_window"(%arg0, %init0) ({
         ^bb0(%a0: tensor<2xf32>, %b0: tensor<2xf32>):
              %2 = mhlo.add %a0, %b0 : tensor<2xf32>
              "mhlo.return"(%2) : (tensor<2xf32>) -> ()
            })
         {
           padding = dense<[[2, 2], [0, 0]]> : tensor<2x2xi64>,
           window_dimensions = dense<[4, 2]> : tensor<2xi64>,
           window_strides = dense<[3, 1]> : tensor<2xi64>
         }
         : (tensor<4x2xf32>, tensor<2xf32>) -> (tensor<2x1xf32>)
  func.return %0 : tensor<2x1xf32>
}

// -----

func.func @reduce_window_invalid_inputs(%arg0: tensor<4x2xf32>,
    %arg1: tensor<4x3xi32>, %init0: tensor<f32>, %init1: tensor<i32>) ->
    (tensor<2x2xf32>, tensor<2x2xi32>) {
  // expected-error @+1 {{expects all inputs to have compatible shapes}}
  %0:2 = "mhlo.reduce_window"(%arg0, %arg1, %init0, %init1) ({
         ^bb0(%a0: tensor<f32>, %a1: tensor<i32>,
                %b0: tensor<f32>, %b1: tensor<i32>):
              %2 = mhlo.add %a0, %b0 : tensor<f32>
              %3 = mhlo.add %a1, %b1 : tensor<i32>
              "mhlo.return"(%2, %3) : (tensor<f32>, tensor<i32>) -> ()
            })
         { padding = dense<[[2, 2], [0, 0]]> : tensor<2x2xi64>,
           window_dimensions = dense<[5, 1]> : tensor<2xi64>,
           window_strides = dense<[3, 1]> : tensor<2xi64> }
         : (tensor<4x2xf32>, tensor<4x3xi32>, tensor<f32>, tensor<i32>) ->
              (tensor<2x2xf32>, tensor<2x2xi32>)
  func.return %0#0, %0#1 : tensor<2x2xf32>, tensor<2x2xi32>
}

// -----

func.func @reduce_window_invalid_inputs(%arg0: tensor<4x2xf32>,
    %arg1: tensor<4x2xi32>, %init0: tensor<f32>, %init1: tensor<i32>) ->
    (tensor<2x2xf32>, tensor<2x2xi32>) {
  // expected-error @+1 {{expects window-dimensions size == input rank, but got window-dimensions size: 1 and input: 'tensor<4x2xf32>' with rank = 2.}}
  %0:2 = "mhlo.reduce_window"(%arg0, %arg1, %init0, %init1) ({
         ^bb0(%a0: tensor<f32>, %a1: tensor<i32>,
                %b0: tensor<f32>, %b1: tensor<i32>):
              %2 = mhlo.add %a0, %b0 : tensor<f32>
              %3 = mhlo.add %a1, %b1 : tensor<i32>
              "mhlo.return"(%2, %3) : (tensor<f32>, tensor<i32>) -> ()
            })
         { padding = dense<[[2, 2], [0, 0]]> : tensor<2x2xi64>,
           window_dimensions = dense<[5]> : tensor<1xi64>,
           window_strides = dense<[3, 1]> : tensor<2xi64> }
         : (tensor<4x2xf32>, tensor<4x2xi32>, tensor<f32>, tensor<i32>) ->
              (tensor<2x2xf32>, tensor<2x2xi32>)
  func.return %0#0, %0#1 : tensor<2x2xf32>, tensor<2x2xi32>
}

// -----

func.func @reduce_window_invalid_attributes(%arg0: tensor<4x2xf32>,
    %arg1: tensor<4x2xi32>, %init0: tensor<f32>, %init1: tensor<i32>) ->
    (tensor<2x2xf32>, tensor<2x2xi32>) {
  // expected-error @+1 {{requires attribute 'window_dimensions'}}
  %0:2 = "mhlo.reduce_window"(%arg0, %arg1, %init0, %init1) ({
         ^bb0(%a0: tensor<f32>, %a1: tensor<i32>,
                %b0: tensor<f32>, %b1: tensor<i32>):
              %2 = mhlo.add %a0, %b0 : tensor<f32>
              %3 = mhlo.add %a1, %b1 : tensor<i32>
              "mhlo.return"(%2, %3) : (tensor<f32>, tensor<i32>) -> ()
            })
         { padding = dense<[[2, 2], [0, 0]]> : tensor<2x2xi64>,
           window_strides = dense<[3, 1]> : tensor<2xi64> }
         : (tensor<4x2xf32>, tensor<4x2xi32>, tensor<f32>, tensor<i32>) ->
              (tensor<2x2xf32>, tensor<2x2xi32>)
  func.return %0#0, %0#1 : tensor<2x2xf32>, tensor<2x2xi32>
}

// -----

func.func @reduce_window_invalid_attributes(%arg0: tensor<4x2xf32>,
    %arg1: tensor<4x2xi32>, %init0: tensor<f32>, %init1: tensor<i32>) ->
    (tensor<2x2xf32>, tensor<2x2xi32>) {
  // expected-error @+1 {{expects window-strides to have same dimension-size as size of window dimensions (2), but got: 1.}}
  %0:2 = "mhlo.reduce_window"(%arg0, %arg1, %init0, %init1) ({
         ^bb0(%a0: tensor<f32>, %a1: tensor<i32>,
                %b0: tensor<f32>, %b1: tensor<i32>):
              %2 = mhlo.add %a0, %b0 : tensor<f32>
              %3 = mhlo.add %a1, %b1 : tensor<i32>
              "mhlo.return"(%2, %3) : (tensor<f32>, tensor<i32>) -> ()
            })
         { padding = dense<[[2, 2], [0, 0]]> : tensor<2x2xi64>,
           window_dimensions = dense<[5, 1]> : tensor<2xi64>,
           window_strides = dense<[1]> : tensor<1xi64> }
         : (tensor<4x2xf32>, tensor<4x2xi32>, tensor<f32>, tensor<i32>) ->
              (tensor<2x2xf32>, tensor<2x2xi32>)
  func.return %0#0, %0#1 : tensor<2x2xf32>, tensor<2x2xi32>
}

// -----

func.func @reduce_window_invalid_attributes(%arg0: tensor<4x2xf32>,
    %arg1: tensor<4x2xi32>, %init0: tensor<f32>, %init1: tensor<i32>) ->
    (tensor<2x2xf32>, tensor<2x2xi32>) {
  // expected-error @+1 {{base-dilation factors to have same dimension-size as size of window dimensions (2), but got: 1.}}
  %0:2 = "mhlo.reduce_window"(%arg0, %arg1, %init0, %init1) ({
         ^bb0(%a0: tensor<f32>, %a1: tensor<i32>,
                %b0: tensor<f32>, %b1: tensor<i32>):
              %2 = mhlo.add %a0, %b0 : tensor<f32>
              %3 = mhlo.add %a1, %b1 : tensor<i32>
              "mhlo.return"(%2, %3) : (tensor<f32>, tensor<i32>) -> ()
            })
         { padding = dense<[[2, 2], [0, 0]]> : tensor<2x2xi64>,
           window_dimensions = dense<[5, 1]> : tensor<2xi64>,
           window_strides = dense<[3, 1]> : tensor<2xi64>,
           base_dilations = dense<[1]> : tensor<1xi64>,
           window_dilations = dense<[1, 1]> : tensor<2xi64> }
         : (tensor<4x2xf32>, tensor<4x2xi32>, tensor<f32>, tensor<i32>) ->
              (tensor<2x2xf32>, tensor<2x2xi32>)
  func.return %0#0, %0#1 : tensor<2x2xf32>, tensor<2x2xi32>
}

// -----

func.func @reduce_window_invalid_attributes(%arg0: tensor<4x2xf32>,
    %arg1: tensor<4x2xi32>, %init0: tensor<f32>, %init1: tensor<i32>) ->
    (tensor<2x2xf32>, tensor<2x2xi32>) {
  // expected-error @+1 {{window-dilation factors to have same dimension-size as size of window dimensions (2), but got: 1.}}
  %0:2 = "mhlo.reduce_window"(%arg0, %arg1, %init0, %init1) ({
         ^bb0(%a0: tensor<f32>, %a1: tensor<i32>,
                %b0: tensor<f32>, %b1: tensor<i32>):
              %2 = mhlo.add %a0, %b0 : tensor<f32>
              %3 = mhlo.add %a1, %b1 : tensor<i32>
              "mhlo.return"(%2, %3) : (tensor<f32>, tensor<i32>) -> ()
            })
         { padding = dense<[[2, 2], [0, 0]]> : tensor<2x2xi64>,
           window_dimensions = dense<[5, 1]> : tensor<2xi64>,
           window_strides = dense<[3, 1]> : tensor<2xi64>,
           base_dilations = dense<[1,1]> : tensor<2xi64>,
           window_dilations = dense<[1]> : tensor<1xi64> }
         : (tensor<4x2xf32>, tensor<4x2xi32>, tensor<f32>, tensor<i32>) ->
              (tensor<2x2xf32>, tensor<2x2xi32>)
  func.return %0#0, %0#1 : tensor<2x2xf32>, tensor<2x2xi32>
}

// -----

func.func @reduce_window_invalid_attributes(%arg0: tensor<4x2xf32>,
    %arg1: tensor<4x2xi32>, %init0: tensor<f32>, %init1: tensor<i32>) ->
    (tensor<2x2xf32>, tensor<2x2xi32>) {
  // expected-error @+1 {{expects padding-entries to have same dimension-size as size of window dimensions (2), but got: 1.}}
  %0:2 = "mhlo.reduce_window"(%arg0, %arg1, %init0, %init1) ({
         ^bb0(%a0: tensor<f32>, %a1: tensor<i32>,
                %b0: tensor<f32>, %b1: tensor<i32>):
              %2 = mhlo.add %a0, %b0 : tensor<f32>
              %3 = mhlo.add %a1, %b1 : tensor<i32>
              "mhlo.return"(%2, %3) : (tensor<f32>, tensor<i32>) -> ()
            })
         { padding = dense<[[2, 2]]> : tensor<1x2xi64>,
           window_dimensions = dense<[5, 1]> : tensor<2xi64>,
           window_strides = dense<[3, 1]> : tensor<2xi64> }
         : (tensor<4x2xf32>, tensor<4x2xi32>, tensor<f32>, tensor<i32>) ->
              (tensor<2x2xf32>, tensor<2x2xi32>)
  func.return %0#0, %0#1 : tensor<2x2xf32>, tensor<2x2xi32>
}

// -----

func.func @reduce_window_invalid_padding_attributes(%arg0: tensor<4x2xf32>,
    %arg1: tensor<4x2xi32>, %init0: tensor<f32>, %init1: tensor<i32>) ->
    (tensor<2x2xf32>, tensor<2x2xi32>) {
  // expected-error @+1 {{expects the shape of padding-attribute to be {N, 2}, but got {4, 1}.}}
  %0:2 = "mhlo.reduce_window"(%arg0, %arg1, %init0, %init1) ({
         ^bb0(%a0: tensor<f32>, %a1: tensor<i32>,
                %b0: tensor<f32>, %b1: tensor<i32>):
              %2 = mhlo.add %a0, %b0 : tensor<f32>
              %3 = mhlo.add %a1, %b1 : tensor<i32>
              "mhlo.return"(%2, %3) : (tensor<f32>, tensor<i32>) -> ()
            })
         { padding = dense<[[2], [2], [0], [0]]> : tensor<4x1xi64>,
           window_dimensions = dense<[5, 1]> : tensor<2xi64>,
           window_strides = dense<[3, 1]> : tensor<2xi64> }
         : (tensor<4x2xf32>, tensor<4x2xi32>, tensor<f32>, tensor<i32>) ->
              (tensor<2x2xf32>, tensor<2x2xi32>)
  func.return %0#0, %0#1 : tensor<2x2xf32>, tensor<2x2xi32>
}

// -----

func.func @reduce_window(%arg0: tensor<4x2xf32>, %arg1: tensor<4x2xi32>,
                    %init0: tensor<f32>, %init1: tensor<i32>) ->
                      (tensor<2x2xf32>, tensor<2x2xi32>) {
  // expected-error @+1 {{expects the shape of window_dimensions attribute to be 1-D, but got {1, 2}}}
  %0:2 = "mhlo.reduce_window"(%arg0, %arg1, %init0, %init1) ({
         ^bb0(%a0: tensor<f32>, %a1: tensor<i32>,
                %b0: tensor<f32>, %b1: tensor<i32>):
              %2 = mhlo.add %a0, %b0 : tensor<f32>
              %3 = mhlo.add %a1, %b1 : tensor<i32>
              "mhlo.return"(%2, %3) : (tensor<f32>, tensor<i32>) -> ()
            })
         { padding = dense<[[2, 2], [0, 0]]> : tensor<2x2xi64>,
           window_dimensions = dense<[[5, 1]]> : tensor<1x2xi64>,
           window_strides = dense<[3, 1]> : tensor<2xi64> }
         : (tensor<4x2xf32>, tensor<4x2xi32>, tensor<f32>, tensor<i32>) ->
              (tensor<2x2xf32>, tensor<2x2xi32>)
  func.return %0#0, %0#1 : tensor<2x2xf32>, tensor<2x2xi32>
}

// -----

func.func @reduce_window(%arg0: tensor<4x2xf32>, %arg1: tensor<4x2xi32>,
                    %init0: tensor<f32>, %init1: tensor<i32>) ->
                      (tensor<2x2xf32>, tensor<2x2xi32>) {
  // expected-error @+1 {{expects the shape of window_strides attribute to be 1-D, but got {1, 2}}}
  %0:2 = "mhlo.reduce_window"(%arg0, %arg1, %init0, %init1) ({
         ^bb0(%a0: tensor<f32>, %a1: tensor<i32>,
                %b0: tensor<f32>, %b1: tensor<i32>):
              %2 = mhlo.add %a0, %b0 : tensor<f32>
              %3 = mhlo.add %a1, %b1 : tensor<i32>
              "mhlo.return"(%2, %3) : (tensor<f32>, tensor<i32>) -> ()
            })
         { padding = dense<[[2, 2], [0, 0]]> : tensor<2x2xi64>,
           window_dimensions = dense<[5, 1]> : tensor<2xi64>,
           window_strides = dense<[[3, 1]]> : tensor<1x2xi64> }
         : (tensor<4x2xf32>, tensor<4x2xi32>, tensor<f32>, tensor<i32>) ->
              (tensor<2x2xf32>, tensor<2x2xi32>)
  func.return %0#0, %0#1 : tensor<2x2xf32>, tensor<2x2xi32>
}

// -----

func.func @reduce_window_with_unranked_dynamic_dims(%arg0: tensor<*xf32>,
    %arg1: tensor<4x?xi32>, %init0: tensor<f32>, %init1: tensor<i32>) ->
        (tensor<?x?xf32>, tensor<*xi32>) {
  // expected-error @+1 {{expects the shape of base_dilations attribute to be 1-D, but got {1, 2}}}
  %0:2 = "mhlo.reduce_window"(%arg0, %arg1, %init0, %init1) ({
         ^bb0(%a0: tensor<f32>, %a1: tensor<i32>,
                %b0: tensor<f32>, %b1: tensor<i32>):
              %2 = mhlo.add %a0, %b0 : tensor<f32>
              %3 = mhlo.add %a1, %b1 : tensor<i32>
              "mhlo.return"(%2, %3) : (tensor<f32>, tensor<i32>) -> ()
            })
         { padding = dense<[[2, 2], [0, 0]]> : tensor<2x2xi64>,
           window_dimensions = dense<[5, 1]> : tensor<2xi64>,
           window_strides = dense<[3, 1]> : tensor<2xi64>,
           base_dilations = dense<[[1, 1]]> : tensor<1x2xi64>,
           window_dilations = dense<[1, 1]> : tensor<2xi64> }
         : (tensor<*xf32>, tensor<4x?xi32>, tensor<f32>, tensor<i32>) ->
              (tensor<?x?xf32>, tensor<*xi32>)
  func.return %0#0, %0#1 : tensor<?x?xf32>, tensor<*xi32>
}

// -----

func.func @reduce_window_with_unranked_dynamic_dims(%arg0: tensor<*xf32>,
    %arg1: tensor<4x?xi32>, %init0: tensor<f32>, %init1: tensor<i32>) ->
        (tensor<?x?xf32>, tensor<*xi32>) {
  // expected-error @+1 {{expects the shape of window_dilations attribute to be 1-D, but got {1, 2}}}
  %0:2 = "mhlo.reduce_window"(%arg0, %arg1, %init0, %init1) ({
         ^bb0(%a0: tensor<f32>, %a1: tensor<i32>,
                %b0: tensor<f32>, %b1: tensor<i32>):
              %2 = mhlo.add %a0, %b0 : tensor<f32>
              %3 = mhlo.add %a1, %b1 : tensor<i32>
              "mhlo.return"(%2, %3) : (tensor<f32>, tensor<i32>) -> ()
            })
         { padding = dense<[[2, 2], [0, 0]]> : tensor<2x2xi64>,
           window_dimensions = dense<[5, 1]> : tensor<2xi64>,
           window_strides = dense<[3, 1]> : tensor<2xi64>,
           base_dilations = dense<[1, 1]> : tensor<2xi64>,
           window_dilations = dense<[[1, 1]]> : tensor<1x2xi64> }
         : (tensor<*xf32>, tensor<4x?xi32>, tensor<f32>, tensor<i32>) ->
              (tensor<?x?xf32>, tensor<*xi32>)
  func.return %0#0, %0#1 : tensor<?x?xf32>, tensor<*xi32>
}

// -----

func.func @reduce_window_invalid_attributes(%arg0: tensor<4x2xf32>, %arg1: tensor<4x2xi32>,
                    %init0: tensor<f32>, %init1: tensor<i32>) ->
                      (tensor<2x2xf32>, tensor<2x2xi32>) {
  // expected-error @+1 {{expects the shape of window_dimensions attribute to be 1-D}}
  %0:2 = "mhlo.reduce_window"(%arg0, %arg1, %init0, %init1) ({
         ^bb0(%a0: tensor<f32>, %a1: tensor<i32>,
                %b0: tensor<f32>, %b1: tensor<i32>):
              %2 = mhlo.add %a0, %b0 : tensor<f32>
              %3 = mhlo.add %a1, %b1 : tensor<i32>
              "mhlo.return"(%2, %3) : (tensor<f32>, tensor<i32>) -> ()
            })
         { padding = dense<[[2, 2], [0, 0]]> : tensor<2x2xi64>,
           window_dimensions = dense<[[5, 1]]> : tensor<1x2xi64>,
           window_strides = dense<[3, 1]> : tensor<2xi64> }
         : (tensor<4x2xf32>, tensor<4x2xi32>, tensor<f32>, tensor<i32>) ->
              (tensor<2x2xf32>, tensor<2x2xi32>)
  func.return %0#0, %0#1 : tensor<2x2xf32>, tensor<2x2xi32>
}

// -----

func.func @reduce_window_invalid_attributes(%arg0: tensor<4x2xf32>,
    %arg1: tensor<4x2xi32>, %init0: tensor<f32>, %init1: tensor<i32>) ->
    (tensor<2x2xf32>, tensor<2x2xi32>) {
  // expected-error @+1 {{expects window to have positive value for 1-th window dimension, but got 0.}}
  %0:2 = "mhlo.reduce_window"(%arg0, %arg1, %init0, %init1) ({
         ^bb0(%a0: tensor<f32>, %a1: tensor<i32>,
                %b0: tensor<f32>, %b1: tensor<i32>):
              %2 = mhlo.add %a0, %b0 : tensor<f32>
              %3 = mhlo.add %a1, %b1 : tensor<i32>
              "mhlo.return"(%2, %3) : (tensor<f32>, tensor<i32>) -> ()
            })
         { padding = dense<[[2, 2], [0, 0]]> : tensor<2x2xi64>,
           window_dimensions = dense<[5, 0]> : tensor<2xi64>,
           window_strides = dense<[3, 1]> : tensor<2xi64> }
         : (tensor<4x2xf32>, tensor<4x2xi32>, tensor<f32>, tensor<i32>) ->
              (tensor<2x2xf32>, tensor<2x2xi32>)
  func.return %0#0, %0#1 : tensor<2x2xf32>, tensor<2x2xi32>
}

// -----

func.func @reduce_window_invalid_attributes(%arg0: tensor<4x2xf32>,
    %arg1: tensor<4x2xi32>, %init0: tensor<f32>, %init1: tensor<i32>) ->
    (tensor<2x2xf32>, tensor<2x2xi32>) {
  // expected-error @+1 {{expects window to have positive stride for 0-th window dimension, but got 0.}}
  %0:2 = "mhlo.reduce_window"(%arg0, %arg1, %init0, %init1) ({
         ^bb0(%a0: tensor<f32>, %a1: tensor<i32>,
                %b0: tensor<f32>, %b1: tensor<i32>):
              %2 = mhlo.add %a0, %b0 : tensor<f32>
              %3 = mhlo.add %a1, %b1 : tensor<i32>
              "mhlo.return"(%2, %3) : (tensor<f32>, tensor<i32>) -> ()
            })
         { padding = dense<[[2, 2], [0, 0]]> : tensor<2x2xi64>,
           window_dimensions = dense<[5, 1]> : tensor<2xi64>,
           window_strides = dense<[0, 1]> : tensor<2xi64> }
         : (tensor<4x2xf32>, tensor<4x2xi32>, tensor<f32>, tensor<i32>) ->
              (tensor<2x2xf32>, tensor<2x2xi32>)
  func.return %0#0, %0#1 : tensor<2x2xf32>, tensor<2x2xi32>
}

// -----

func.func @reduce_window_invalid_attributes(%arg0: tensor<4x2xf32>,
    %arg1: tensor<4x2xi32>, %init0: tensor<f32>, %init1: tensor<i32>) ->
    (tensor<2x2xf32>, tensor<2x2xi32>) {
  // expected-error @+1 {{expects window to have positive base dilation factor for 1-th window dimension, but got 0.}}
  %0:2 = "mhlo.reduce_window"(%arg0, %arg1, %init0, %init1) ({
         ^bb0(%a0: tensor<f32>, %a1: tensor<i32>,
                %b0: tensor<f32>, %b1: tensor<i32>):
              %2 = mhlo.add %a0, %b0 : tensor<f32>
              %3 = mhlo.add %a1, %b1 : tensor<i32>
              "mhlo.return"(%2, %3) : (tensor<f32>, tensor<i32>) -> ()
            })
         { padding = dense<[[2, 2], [0, 0]]> : tensor<2x2xi64>,
           window_dimensions = dense<[5, 1]> : tensor<2xi64>,
           window_strides = dense<[3, 1]> : tensor<2xi64>,
           base_dilations = dense<[1,0]> : tensor<2xi64>,
           window_dilations = dense<[1,1]> : tensor<2xi64> }
         : (tensor<4x2xf32>, tensor<4x2xi32>, tensor<f32>, tensor<i32>) ->
              (tensor<2x2xf32>, tensor<2x2xi32>)
  func.return %0#0, %0#1 : tensor<2x2xf32>, tensor<2x2xi32>
}

// -----

func.func @reduce_window_invalid_attributes(%arg0: tensor<4x2xf32>,
    %arg1: tensor<4x2xi32>, %init0: tensor<f32>, %init1: tensor<i32>) ->
    (tensor<2x2xf32>, tensor<2x2xi32>) {
  // expected-error @+1 {{expects window to have positive window dilation factor for 0-th window dimension, but got 0.}}
  %0:2 = "mhlo.reduce_window"(%arg0, %arg1, %init0, %init1) ({
         ^bb0(%a0: tensor<f32>, %a1: tensor<i32>,
                %b0: tensor<f32>, %b1: tensor<i32>):
              %2 = mhlo.add %a0, %b0 : tensor<f32>
              %3 = mhlo.add %a1, %b1 : tensor<i32>
              "mhlo.return"(%2, %3) : (tensor<f32>, tensor<i32>) -> ()
            })
         { padding = dense<[[2, 2], [0, 0]]> : tensor<2x2xi64>,
           window_dimensions = dense<[5, 1]> : tensor<2xi64>,
           window_strides = dense<[3, 1]> : tensor<2xi64>,
           base_dilations = dense<[1,1]> : tensor<2xi64>,
           window_dilations = dense<[0,1]> : tensor<2xi64> }
         : (tensor<4x2xf32>, tensor<4x2xi32>, tensor<f32>, tensor<i32>) ->
              (tensor<2x2xf32>, tensor<2x2xi32>)
  func.return %0#0, %0#1 : tensor<2x2xf32>, tensor<2x2xi32>
}

// -----

func.func @reduce_window_invalid_ret_type(%arg0: tensor<4x2xf32>,
    %arg1: tensor<4x2xi32>, %init0: tensor<f32>, %init1: tensor<i32>) ->
        tensor<2x2xf32> {
  // expected-error @+1 {{inferred type(s) 'tensor<2x2xf32>', 'tensor<2x2xi32>' are incompatible with return type(s) of operation 'tensor<2x2xf32>'}}
  %0 = "mhlo.reduce_window"(%arg0, %arg1, %init0, %init1) ({
         ^bb0(%a0: tensor<f32>, %a1: tensor<i32>,
                %b0: tensor<f32>, %b1: tensor<i32>):
              %2 = mhlo.add %a0, %b0 : tensor<f32>
              %3 = mhlo.add %a1, %b1 : tensor<i32>
              "mhlo.return"(%2, %3) : (tensor<f32>, tensor<i32>) -> ()
            })
         { padding = dense<[[2, 2], [0, 0]]> : tensor<2x2xi64>,
           window_dimensions = dense<[5, 1]> : tensor<2xi64>,
           window_strides = dense<[3, 1]> : tensor<2xi64>
         }
         : (tensor<4x2xf32>, tensor<4x2xi32>, tensor<f32>, tensor<i32>) ->
              tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

func.func @reduce_window_invalid_ret_type(%arg0: tensor<4x2xf32>,
    %arg1: tensor<4x2xi32>, %init0: tensor<f32>, %init1: tensor<i32>) ->
        (tensor<2x2xf32>, tensor<2x3xi32>) {
  // expected-error @+1 {{inferred type(s) 'tensor<2x2xf32>', 'tensor<2x2xi32>' are incompatible with return type(s) of operation 'tensor<2x2xf32>', 'tensor<2x3xi32>'}}
  %0:2 = "mhlo.reduce_window"(%arg0, %arg1, %init0, %init1) ({
         ^bb0(%a0: tensor<f32>, %a1: tensor<i32>,
                %b0: tensor<f32>, %b1: tensor<i32>):
              %2 = mhlo.add %a0, %b0 : tensor<f32>
              %3 = mhlo.add %a1, %b1 : tensor<i32>
              "mhlo.return"(%2, %3) : (tensor<f32>, tensor<i32>) -> ()
            })
         { padding = dense<[[2, 2], [0, 0]]> : tensor<2x2xi64>,
           window_dimensions = dense<[5, 1]> : tensor<2xi64>,
           window_strides = dense<[3, 1]> : tensor<2xi64>
         }
         : (tensor<4x2xf32>, tensor<4x2xi32>, tensor<f32>, tensor<i32>) ->
              (tensor<2x2xf32>, tensor<2x3xi32>)
  func.return %0#0, %0#1 : tensor<2x2xf32>, tensor<2x3xi32>
}

// -----

func.func @reduce_window_invalid_ret_type(%arg0: tensor<4x2xf32>,
    %arg1: tensor<4x2xi32>, %init0: tensor<f32>, %init1: tensor<i32>) ->
        (tensor<2x2xi32>, tensor<2x2xi32>) {
  // expected-error @+1 {{inferred type(s) 'tensor<2x2xf32>', 'tensor<2x2xi32>' are incompatible with return type(s) of operation 'tensor<2x2xi32>', 'tensor<2x2xi32>'}}
  %0:2 = "mhlo.reduce_window"(%arg0, %arg1, %init0, %init1) ({
         ^bb0(%a0: tensor<f32>, %a1: tensor<i32>,
                %b0: tensor<f32>, %b1: tensor<i32>):
              %2 = mhlo.add %a0, %b0 : tensor<f32>
              %3 = mhlo.add %a1, %b1 : tensor<i32>
              "mhlo.return"(%2, %3) : (tensor<f32>, tensor<i32>) -> ()
            })
         { padding = dense<[[2, 2], [0, 0]]> : tensor<2x2xi64>,
           window_dimensions = dense<[5, 1]> : tensor<2xi64>,
           window_strides = dense<[3, 1]> : tensor<2xi64>
         }
         : (tensor<4x2xf32>, tensor<4x2xi32>, tensor<f32>, tensor<i32>) ->
              (tensor<2x2xi32>, tensor<2x2xi32>)
  func.return %0#0, %0#1 : tensor<2x2xi32>, tensor<2x2xi32>
}

// -----

func.func @reduce_window_invalid_reducer(%arg0: tensor<4x2xf32>,
    %arg1: tensor<4x2xi32>, %init0: tensor<f32>, %init1: tensor<i32>) ->
    (tensor<2x2xf32>, tensor<2x2xi32>) {
  // expected-error@+1 {{Reduction-region must take 4 parameters, but takes 2 parameter(s)}}
  %0:2 = "mhlo.reduce_window"(%arg0, %arg1, %init0, %init1) ({
         ^bb0(%a0: tensor<f32>, %b0: tensor<f32>):
              %2 = mhlo.add %a0, %b0 : tensor<f32>
              "mhlo.return"(%2) : (tensor<f32>) -> ()
            })
         { padding = dense<[[2, 2], [0, 0]]> : tensor<2x2xi64>,
           window_dimensions = dense<[5, 1]> : tensor<2xi64>,
           window_strides = dense<[3, 1]> : tensor<2xi64> }
         : (tensor<4x2xf32>, tensor<4x2xi32>, tensor<f32>, tensor<i32>) ->
              (tensor<2x2xf32>, tensor<2x2xi32>)
  func.return %0#0, %0#1 : tensor<2x2xf32>, tensor<2x2xi32>
}

// -----

func.func @reduce_window_invalid_reducer(%arg0: tensor<4x2xf32>,
    %arg1: tensor<4x2xi32>, %init0: tensor<f32>, %init1: tensor<i32>) ->
    (tensor<2x2xf32>, tensor<2x2xi32>) {
  // expected-error@+1 {{The reduction-region expected to return some value(s)}}
  %0:2 = "mhlo.reduce_window"(%arg0, %arg1, %init0, %init1) ({
         ^bb0(%a0: tensor<f32>, %a1: tensor<i32>,
                %b0: tensor<f32>, %b1: tensor<i32>):
              %2 = mhlo.add %a0, %b0 : tensor<f32>
              %3 = mhlo.add %a1, %b1 : tensor<i32>
              "mhlo.return"() : () -> ()
            })
         { padding = dense<[[2, 2], [0, 0]]> : tensor<2x2xi64>,
           window_dimensions = dense<[5, 1]> : tensor<2xi64>,
           window_strides = dense<[3, 1]> : tensor<2xi64> }
         : (tensor<4x2xf32>, tensor<4x2xi32>, tensor<f32>, tensor<i32>) ->
              (tensor<2x2xf32>, tensor<2x2xi32>)
  func.return %0#0, %0#1 : tensor<2x2xf32>, tensor<2x2xi32>
}

// -----

func.func @reduce_window_invalid_reducer(%arg0: tensor<4x2xf32>,
    %arg1: tensor<4x2xi32>, %init0: tensor<f32>, %init1: tensor<i32>) ->
    (tensor<2x2xf32>, tensor<2x2xi32>) {
  // expected-error@+1 {{Reduction-region here must produce 2 tensors, but produces 3 instead}}
  %0:2 = "mhlo.reduce_window"(%arg0, %arg1, %init0, %init1) ({
         ^bb0(%a0: tensor<f32>, %a1: tensor<i32>,
                %b0: tensor<f32>, %b1: tensor<i32>):
              %2 = mhlo.add %a0, %b0 : tensor<f32>
              %3 = mhlo.add %a1, %b1 : tensor<i32>
              "mhlo.return"(%2,%3,%2) : (tensor<f32>, tensor<i32>, tensor<f32>)
                  -> ()
            })
         { padding = dense<[[2, 2], [0, 0]]> : tensor<2x2xi64>,
           window_dimensions = dense<[5, 1]> : tensor<2xi64>,
           window_strides = dense<[3, 1]> : tensor<2xi64> }
         : (tensor<4x2xf32>, tensor<4x2xi32>, tensor<f32>, tensor<i32>) ->
              (tensor<2x2xf32>, tensor<2x2xi32>)
  func.return %0#0, %0#1 : tensor<2x2xf32>, tensor<2x2xi32>
}

// -----

func.func @reduce_window_invalid_reducer(%arg0: tensor<4x2xf32>,
    %arg1: tensor<4x2xi32>, %init0: tensor<f32>, %init1: tensor<i32>) ->
    (tensor<2x2xf32>, tensor<2x2xi32>) {
  // expected-error@+1 {{Reduction-region here must produce tensor-typed result(s), but produces 'tuple<tensor<f32>, tensor<i32>>' instead}}
  %0:2 = "mhlo.reduce_window"(%arg0, %arg1, %init0, %init1) ({
         ^bb0(%a0: tensor<f32>, %a1: tensor<i32>, %b0: tensor<f32>,
                %b1: tensor<i32>):
              %2 = mhlo.add %a0, %b0 : tensor<f32>
              %3 = mhlo.add %a1, %b1 : tensor<i32>
              %4 = "mhlo.tuple"(%2, %3) : (tensor<f32>, tensor<i32>) ->
                  tuple<tensor<f32>, tensor<i32>>
              "mhlo.return"(%4,%2) :
                  (tuple<tensor<f32>, tensor<i32>>, tensor<f32>) -> ()
            })
         { padding = dense<[[2, 2], [0, 0]]> : tensor<2x2xi64>,
           window_dimensions = dense<[5, 1]> : tensor<2xi64>,
           window_strides = dense<[3, 1]> : tensor<2xi64> }
         : (tensor<4x2xf32>, tensor<4x2xi32>, tensor<f32>, tensor<i32>) ->
              (tensor<2x2xf32>, tensor<2x2xi32>)
  func.return %0#0, %0#1 : tensor<2x2xf32>, tensor<2x2xi32>
}

// -----

func.func @reduce_window_invalid_reducer(%arg0: tensor<4x2xf32>,
    %arg1: tensor<4x2xi32>, %init0: tensor<f32>, %init1: tensor<i32>) ->
    (tensor<2x2xf32>, tensor<2x2xi32>) {
  // expected-error@+1 {{The type of reduction-region's parameter at index 0 is different than the corresponding result type: 'tensor<f32>' vs 'tensor<i32>'}}
  %0:2 = "mhlo.reduce_window"(%arg0, %arg1, %init0, %init1) ({
         ^bb0(%a0: tensor<f32>, %a1: tensor<i32>, %b0: tensor<f32>,
                %b1: tensor<i32>):
              %2 = mhlo.add %a0, %b0 : tensor<f32>
              %3 = mhlo.add %a1, %b1 : tensor<i32>
              "mhlo.return"(%3,%2) : (tensor<i32>, tensor<f32>) -> ()
            })
         { padding = dense<[[2, 2], [0, 0]]> : tensor<2x2xi64>,
           window_dimensions = dense<[5, 1]> : tensor<2xi64>,
           window_strides = dense<[3, 1]> : tensor<2xi64> }
         : (tensor<4x2xf32>, tensor<4x2xi32>, tensor<f32>, tensor<i32>) ->
              (tensor<2x2xf32>, tensor<2x2xi32>)
  func.return %0#0, %0#1 : tensor<2x2xf32>, tensor<2x2xi32>
}

// -----

func.func @reduce_window_invalid_reducer(%arg0: tensor<4x2xf32>,
    %arg1: tensor<4x2xi32>, %init0: tensor<f32>, %init1: tensor<i32>) ->
    (tensor<2x2xf32>, tensor<2x2xi32>) {
  // expected-error@+1 {{The type of reduction-region's parameter at index 1 is different than the corresponding result type: 'tensor<i32>' vs 'tensor<f32>'}}
  %0:2 = "mhlo.reduce_window"(%arg0, %arg1, %init0, %init1) ({
         ^bb0(%a0: tensor<f32>, %a1: tensor<i32>, %b0: tensor<f32>,
                %b1: tensor<i32>):
              %2 = mhlo.add %a0, %b0 : tensor<f32>
              %3 = mhlo.add %a1, %b1 : tensor<i32>
              "mhlo.return"(%2,%2) : (tensor<f32>, tensor<f32>) -> ()
            })
         { padding = dense<[[2, 2], [0, 0]]> : tensor<2x2xi64>,
           window_dimensions = dense<[5, 1]> : tensor<2xi64>,
           window_strides = dense<[3, 1]> : tensor<2xi64> }
         : (tensor<4x2xf32>, tensor<4x2xi32>, tensor<f32>, tensor<i32>) ->
              (tensor<2x2xf32>, tensor<2x2xi32>)
  func.return %0#0, %0#1 : tensor<2x2xf32>, tensor<2x2xi32>
}

// -----

func.func @reduce_window_invalid_reducer(%arg0: tensor<4x2xf32>,
    %arg1: tensor<4x2xi32>, %init0: tensor<f32>, %init1: tensor<i32>) ->
    (tensor<2x2xf32>, tensor<2x2xi32>) {
  // expected-error@+1 {{The type of reduction-region's parameter at index 2 is different than the corresponding result type: 'tensor<i32>' vs 'tensor<f32>'}}
  %0:2 = "mhlo.reduce_window"(%arg0, %arg1, %init0, %init1) ({
         ^bb0(%a0: tensor<f32>, %a1: tensor<i32>, %b1: tensor<i32>,
                %b0: tensor<f32>):
              %2 = mhlo.add %a0, %b0 : tensor<f32>
              %3 = mhlo.add %a1, %b1 : tensor<i32>
              "mhlo.return"(%2,%3) : (tensor<f32>, tensor<i32>) -> ()
            })
         { padding = dense<[[2, 2], [0, 0]]> : tensor<2x2xi64>,
           window_dimensions = dense<[5, 1]> : tensor<2xi64>,
           window_strides = dense<[3, 1]> : tensor<2xi64> }
         : (tensor<4x2xf32>, tensor<4x2xi32>, tensor<f32>, tensor<i32>) ->
              (tensor<2x2xf32>, tensor<2x2xi32>)
  func.return %0#0, %0#1 : tensor<2x2xf32>, tensor<2x2xi32>
}

// -----

func.func @reduce_window_invalid_reducer(%arg0: tensor<4x2xf32>,
    %arg1: tensor<4x2xi32>, %init0: tensor<f32>, %init1: tensor<i32>) ->
    (tensor<2x2xf32>, tensor<2x2xi32>) {
  // expected-error@+1 {{The type of reduction-region's result type at index 0 differs from the op's corresponding init-value type: 'tensor<f32>' vs 'tensor<i32>'}}
  %0:2 = "mhlo.reduce_window"(%arg0, %arg1, %init1, %init0) ({
         ^bb0(%a0: tensor<f32>, %a1: tensor<i32>, %b0: tensor<f32>,
                %b1: tensor<i32>):
              %2 = mhlo.add %a0, %b0 : tensor<f32>
              %3 = mhlo.add %a1, %b1 : tensor<i32>
              "mhlo.return"(%2,%3) : (tensor<f32>, tensor<i32>) -> ()
            })
         { padding = dense<[[2, 2], [0, 0]]> : tensor<2x2xi64>,
           window_dimensions = dense<[5, 1]> : tensor<2xi64>,
           window_strides = dense<[3, 1]> : tensor<2xi64> }
         : (tensor<4x2xf32>, tensor<4x2xi32>, tensor<i32>, tensor<f32>) ->
              (tensor<2x2xf32>, tensor<2x2xi32>)
  func.return %0#0, %0#1 : tensor<2x2xf32>, tensor<2x2xi32>
}

// -----

func.func @reduce_window_invalid_reducer(%arg0: tensor<4x2xf32>,
    %arg1: tensor<4x2xi32>, %init0: tensor<f32>, %init1: tensor<i32>) ->
    (tensor<2x2xf32>, tensor<2x2xi32>) {
  // expected-error@+1 {{The element-type of reduction-region's argument at index 2 is expected to be 'i32', but got 'tensor<f32>' as its type.}}
  %0:2 = "mhlo.reduce_window"(%arg1, %arg0, %init0, %init1) ({
         ^bb0(%a0: tensor<f32>, %a1: tensor<i32>, %b0: tensor<f32>,
                %b1: tensor<i32>):
              %2 = mhlo.add %a0, %b0 : tensor<f32>
              %3 = mhlo.add %a1, %b1 : tensor<i32>
              "mhlo.return"(%2,%3) : (tensor<f32>, tensor<i32>) -> ()
            })
         {
           padding = dense<[[2, 2], [0, 0]]> : tensor<2x2xi64>,
           window_dimensions = dense<[5, 1]> : tensor<2xi64>,
           window_strides = dense<[3, 1]> : tensor<2xi64>
         }
         : (tensor<4x2xi32>, tensor<4x2xf32>, tensor<f32>, tensor<i32>) ->
              (tensor<2x2xf32>, tensor<2x2xi32>)
  func.return %0#0, %0#1 : tensor<2x2xf32>, tensor<2x2xi32>
}

// -----

func.func @reduce_window_invalid_reducer(%arg0: tensor<f32>, %init0: tensor<1xf32>)
  -> (tensor<f32>) {
  // expected-error@+1 {{The rank of reduction-region's argument at index 1 is expected to be <= 0, got 1}}
  %0 = "mhlo.reduce_window"(%arg0, %init0) ({
         ^bb0(%a0: tensor<1xf32>, %b0: tensor<1xf32>):
              %2 = mhlo.add %a0, %b0 : tensor<1xf32>
              "mhlo.return"(%2) : (tensor<1xf32>) -> ()
            })
         {
           window_dimensions = dense<> : tensor<0xi64>
         }
         : (tensor<f32>, tensor<1xf32>) -> (tensor<f32>)
  func.return %0 : tensor<f32>
}

// -----

func.func @reduce_window_invalid_reducer(%arg0: tensor<4x2xf32>, %init0: tensor<4x2xf32>)
  -> tensor<2x2xf32> {
  // expected-error@+1 {{The shape of reduction-region's argument at index 1 is not compatible with that of reduce-op's input-parameter at index 0}}
  %0 = "mhlo.reduce_window"(%arg0, %init0) ({
         ^bb0(%a0: tensor<4x2xf32>, %b0: tensor<4x2xf32>):
              %2 = mhlo.add %a0, %b0 : tensor<4x2xf32>
              "mhlo.return"(%2) : (tensor<4x2xf32>) -> ()
            })
         {
           padding = dense<[[2, 2], [0, 0]]> : tensor<2x2xi64>,
           window_dimensions = dense<[5, 1]> : tensor<2xi64>,
           window_strides = dense<[3, 1]> : tensor<2xi64>
         }
         : (tensor<4x2xf32>, tensor<4x2xf32>) -> (tensor<2x2xf32>)
  func.return %0 : tensor<2x2xf32>
}

// -----

func.func @reduce_window_invalid_reducer(%arg0: tensor<4x2xf32>, %init0: tensor<4x2xf32>)
  -> tensor<2x1xf32> {
  %0 = "mhlo.reduce_window"(%arg0, %init0) ({
         ^bb0(%a0: tensor<4x2xf32>, %b0: tensor<4x2xf32>):
              %2 = mhlo.add %a0, %b0 : tensor<4x2xf32>
              "mhlo.return"(%2) : (tensor<4x2xf32>) -> ()
            })
         {
           padding = dense<[[2, 2], [0, 0]]> : tensor<2x2xi64>,
           window_dimensions = dense<[4, 2]> : tensor<2xi64>,
           window_strides = dense<[3, 1]> : tensor<2xi64>
         }
         : (tensor<4x2xf32>, tensor<4x2xf32>) -> (tensor<2x1xf32>)
  func.return %0 : tensor<2x1xf32>
}

// -----

func.func @reduce_window_invalid_reducer(%arg0: tensor<4x2xf32>,
    %arg1: tensor<4x2xi32>, %init0: tensor<f32>, %init1: tensor<i32>) ->
    (tensor<2x2xf32>, tensor<2x2xi32>) {
  // expected-error@+1 {{Reduction-region here must produce tensor-typed result(s), but produces 'tuple<tensor<f32>, tensor<i32>>' instead}}
  %0:2 = "mhlo.reduce_window"(%arg0, %arg1, %init0, %init1) ({
         ^bb0(%a0: tensor<f32>, %a1: tensor<i32>, %b0: tensor<f32>,
                %b1: tensor<i32>):
              %2 = mhlo.add %a0, %b0 : tensor<f32>
              %3 = mhlo.add %a1, %b1 : tensor<i32>
              %4 = "mhlo.tuple"(%2, %3) : (tensor<f32>, tensor<i32>) ->
                  tuple<tensor<f32>, tensor<i32>>
              "mhlo.return"(%4,%2) :
                  (tuple<tensor<f32>, tensor<i32>>, tensor<f32>) -> ()
            })
         { padding = dense<[[2, 2], [0, 0]]> : tensor<2x2xi64>,
           window_dimensions = dense<[5, 1]> : tensor<2xi64>,
           window_strides = dense<[3, 1]> : tensor<2xi64> }
         : (tensor<4x2xf32>, tensor<4x2xi32>, tensor<f32>, tensor<i32>) ->
              (tensor<2x2xf32>, tensor<2x2xi32>)
  func.return %0#0, %0#1 : tensor<2x2xf32>, tensor<2x2xi32>
}
