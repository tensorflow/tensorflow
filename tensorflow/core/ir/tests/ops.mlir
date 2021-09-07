// RUN: tfg-opt-no-passes %s | tfg-opt-no-passes | FileCheck %s

// Exercise some basic custom syntax

// CHECK-LABEL: tfg.graph
// CHECK-SAME:  #tf_type.version<producer = 42, min_consumer = 33>
tfg.graph #tf_type.version<producer = 42, min_consumer = 33> {
  // CHECK: %placeholder, %ctl = placeholder : () -> (tensor<*xi32>)
  // CHECK: %AddV2, %ctl_0 = AddV2(%placeholder, %placeholder_1) device("GPU") assigned_device("TPU") {some_attribute = "some attr!"} : (tensor<*xi32>, tensor<*xi32>) -> (tensor<*xi32>)
  // CHECK: %placeholder_1, %ctl_2 = placeholder device("CPU") name("foobar") : () -> (tensor<*xi32>)
  %arg0, %ctl = "tfg.placeholder"() : () -> (tensor<*xi32>, !tf_type.control)
  %add, %ctl3 = "tfg.AddV2"(%arg0, %arg1) {"_mlir_device" = "GPU", _mlir_assigned_device = "TPU", some_attribute = "some attr!"} : (tensor<*xi32>, tensor<*xi32>) -> (tensor<*xi32>, !tf_type.control)
  %arg1, %ctl2 = "tfg.placeholder"() {"_mlir_device" = "CPU", _mlir_name = "foobar"} : () -> (tensor<*xi32>, !tf_type.control)
  // Unused argument
  %ctl4 = tfg._Arg
}

// CHECK-LABEL: tfg.graph
// CHECK-SAME:  #tf_type.version<producer = 1, min_consumer = 1, bad_consumers = [1, 2, 5, 12]>
tfg.graph #tf_type.version<producer = 1, min_consumer = 1, bad_consumers = [1, 2, 5, 12]> {
}

// Verify:
// - the naming of the result using OpAsmDialectInterface.
// - the implicit `.ctl` entry block arguments.
// - the specific format for the FuncOp.
// As such this test does not use any regex intentionally.

// CHECK:  tfg.func @foo(%input: tensor<10xf32> {tfg.name = "input"},
// CHECK-NEXT:           %another_input: tensor<10xf32> {tfg.name = "another_input"})
// CHECK-NEXT:      -> (tensor<10xf32> {tfg.name = "result1"},
// CHECK-NEXT:          tensor<10xf32> {tfg.name = "result2"})
// CHECK-NEXT:    attributes {description = "function foo"} {
tfg.func @foo(%arg0 : tensor<10xf32> {tfg.name = "input"},
              %arg1 : tensor<10xf32> {tfg.name = "another_input"})
    -> (tensor<10xf32> {tfg.name = "result1"}, tensor<10xf32> {tfg.name = "result2"})
  attributes {description = "function foo"} {
// CHECK: %placeholder:3, %ctl = placeholder : () -> (tensor<*xi32>, tensor<*xi32>, tensor<*xi32>)
  %placeholder:3, %ctl0 = placeholder : () -> (tensor<*xi32>, tensor<*xi32>, tensor<*xi32>)
// CHECK: %AddV2, %ctl_0 = AddV2(%input, %another_input) [%another_input.ctl] device("GPU") : (tensor<10xf32>, tensor<10xf32>) -> (tensor<10xf32>)
  %add, %ctl = AddV2(%arg0, %arg1) [%arg1.ctl] device("GPU") : (tensor<10xf32>, tensor<10xf32>) -> (tensor<10xf32>)
// CHECK: %AddV2_1, %ctl_2 = AddV2(%another_input, %input) [%ctl_0] device("GPU") {some_attribute = "some attr!"} : (tensor<10xf32>, tensor<10xf32>) -> (tensor<10xf32>)
  %add_1, %ctl2 = AddV2(%arg1, %arg0) [%ctl] device("GPU") {some_attribute = "some attr!"} : (tensor<10xf32>, tensor<10xf32>) -> (tensor<10xf32>)
// CHECK: return(%AddV2, %AddV2_1) [%ctl] : tensor<10xf32>, tensor<10xf32>
  return(%add, %add_1) [%ctl0] : tensor<10xf32>, tensor<10xf32>
}
// CHECK-LABEL: tfg.func @bar()
// CHECK-NEXT: gradient = @foo
tfg.func @bar() attributes {gradient = @foo} {
  return
}

// CHECK:  tfg.func generic @gfoo(%input: !tf_type.tensor {tfg.name = "input"},
// CHECK-NEXT:           %another_input: !tf_type.tensor {tfg.name = "another_input"})
// CHECK-NEXT:      -> (!tf_type.tensor {tfg.name = "result1"},
// CHECK-NEXT:          !tf_type.tensor {tfg.name = "result2"})
// CHECK-NEXT:    attributes {description = "function foo"} {
tfg.func generic @gfoo(%arg0 : !tf_type.tensor {tfg.name = "input"},
              %arg1 : !tf_type.tensor {tfg.name = "another_input"})
    -> (!tf_type.tensor {tfg.name = "result1"}, !tf_type.tensor {tfg.name = "result2"})
  attributes {description = "function foo"} {
// CHECK: %placeholder, %ctl = placeholder : () -> (!tf_type.tensor)
  %placeholder, %ctl0 = placeholder : () -> (!tf_type.tensor)
// CHECK: %AddV2, %ctl_0 = AddV2(%input, %another_input) [%another_input.ctl] device("GPU") : (!tf_type.tensor, !tf_type.tensor) -> (!tf_type.tensor)
  %add, %ctl = AddV2(%arg0, %arg1) [%arg1.ctl] device("GPU") : (!tf_type.tensor, !tf_type.tensor) -> (!tf_type.tensor)
// CHECK: get_result(%AddV2) "output" : 0
  %add_output = get_result(%add) "output" : 0
// CHECK: %AddV2_1, %ctl_2 = AddV2(%another_input, %input) [%ctl_0] device("GPU") {some_attribute = "some attr!"} : (!tf_type.tensor, !tf_type.tensor) -> (!tf_type.tensor)
  %add_1, %ctl2 = AddV2(%arg1, %arg0) [%ctl] device("GPU") {some_attribute = "some attr!"} : (!tf_type.tensor, !tf_type.tensor) -> (!tf_type.tensor)
// CHECK: return(%AddV2, %AddV2_1) [%ctl] : !tf_type.tensor, !tf_type.tensor
  return(%add, %add_1) [%ctl0] : !tf_type.tensor, !tf_type.tensor
}
