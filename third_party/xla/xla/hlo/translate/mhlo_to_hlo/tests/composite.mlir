// RUN: xla-translate -split-input-file -mlir-hlo-to-hlo-text %s | FileCheck %s

module @composite {
  // CHECK: HloModule composite, entry_computation_layout={()->f32[]}
  // CHECK: %add.2 (Arg_0.3: f32[]) -> f32[] {
  // CHECK:   %Arg_0.3 = f32[] parameter(0)
  // CHECK:   %constant.4 = f32[] constant(2)
  // CHECK:   ROOT %add.5 = f32[] add(f32[] %Arg_0.3, f32[] %constant.4)
  // CHECK: }
  // CHECK: ENTRY %main.7 () -> f32[] {
  // CHECK:   %constant.1 = f32[] constant(42)
  // CHECK:   ROOT %call.6 = f32[] call(f32[] %constant.1), to_apply=%add.2, is_composite=true, frontend_attributes={composite.attributes={n = 1 : i32, tensor = dense<1> : tensor<i32>},composite.name="foo.bar",composite.version="1"}
  // CHECK: }
  func.func @main() -> tensor<f32> {
    %0 = mhlo.constant dense<4.200000e+01> : tensor<f32>
    %1 = mhlo.composite "foo.bar" %0 {
      composite_attributes = {
        n = 1 : i32,
        tensor = dense<1> : tensor<i32>
      },
      decomposition = @add,
      version = 1 : i32
    } : (tensor<f32>) -> tensor<f32>
    return %1 : tensor<f32>
  }
  func.func @add(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = mhlo.constant dense<2.000000e+00> : tensor<f32>
    %1 = mhlo.add %arg0, %0 : tensor<f32>
    return %1 : tensor<f32>
  }
}

// -----

// zero-output composite
module @composite {
  //CHECK: HloModule composite, entry_computation_layout={()->()}
  //CHECK: %return.2 (Arg_0.3: f32[]) -> () {
  //CHECK:   %Arg_0.3 = f32[] parameter(0)
  //CHECK:   ROOT %tuple.4 = () tuple()
  //CHECK: }
  //CHECK: ENTRY %main.7 () -> () {
  //CHECK:   %constant.1 = f32[] constant(42)
  //CHECK:   %call.5 = () call(f32[] %constant.1), to_apply=%return.2, is_composite=true, frontend_attributes={composite.attributes={n = 1 : i32, tensor = dense<1> : tensor<i32>},composite.name="foo.bar",composite.version="1"}
  //CHECK:   ROOT %tuple.6 = () tuple()
  //CHECK: }
  func.func @main() -> () {
    %0 = mhlo.constant dense<4.200000e+01> : tensor<f32>
    "mhlo.composite"(%0) {
      name = "foo.bar",
      composite_attributes = {
        n = 1 : i32,
        tensor = dense<1> : tensor<i32>
      },
      decomposition = @return,
      version = 1 : i32
    } : (tensor<f32>) -> ()
    return
  }
  func.func @return(%arg0: tensor<f32>) -> () {
    return
  }
}

// -----

// multi-output composite
module @composite {
  //CHECK: HloModule composite, entry_computation_layout={()->(f32[], f32[])}
  //CHECK: %add.2 (Arg_0.3: f32[]) -> (f32[], f32[]) {
  //CHECK:   %Arg_0.3 = f32[] parameter(0)
  //CHECK:   %constant.4 = f32[] constant(2)
  //CHECK:   %add.5 = f32[] add(f32[] %Arg_0.3, f32[] %constant.4)
  //CHECK:   ROOT %tuple.6 = (f32[], f32[]) tuple(f32[] %add.5, f32[] %add.5)
  //CHECK: }
  //CHECK: ENTRY %main.11 () -> (f32[], f32[]) {
  //CHECK:   %constant.1 = f32[] constant(42)
  //CHECK:   %call.7 = (f32[], f32[]) call(f32[] %constant.1), to_apply=%add.2, is_composite=true, frontend_attributes={composite.attributes={n = 1 : i32, tensor = dense<1> : tensor<i32>},composite.name="foo.bar",composite.version="1"}
  //CHECK:   %get-tuple-element.8 = f32[] get-tuple-element((f32[], f32[]) %call.7), index=0
  //CHECK:   %get-tuple-element.9 = f32[] get-tuple-element((f32[], f32[]) %call.7), index=1
  //CHECK:   ROOT %tuple.10 = (f32[], f32[]) tuple(f32[] %get-tuple-element.8, f32[] %get-tuple-element.9)
  //CHECK: }
  func.func @main() -> (tensor<f32>, tensor<f32>) {
    %0 = mhlo.constant dense<4.200000e+01> : tensor<f32>
    %result:2 = "mhlo.composite"(%0) {
      name = "foo.bar",
      composite_attributes = {
        n = 1 : i32,
        tensor = dense<1> : tensor<i32>
      },
      decomposition = @add,
      version = 1 : i32
    } : (tensor<f32>) -> (tensor<f32>, tensor<f32>)
    return %result#0, %result#1 : tensor<f32>, tensor<f32>
  }
  func.func @add(%arg0: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
    %0 = mhlo.constant dense<2.000000e+00> : tensor<f32>
    %1 = mhlo.add %arg0, %0 : tensor<f32>
    return %1, %1 : tensor<f32>, tensor<f32>
  }
}

// -----

// optional composite attributes
module @composite {
  // CHECK: HloModule composite, entry_computation_layout={()->f32[]}
  // CHECK: %add.2 (Arg_0.3: f32[]) -> f32[] {
  // CHECK:   %Arg_0.3 = f32[] parameter(0)
  // CHECK:   %constant.4 = f32[] constant(2)
  // CHECK:   ROOT %add.5 = f32[] add(f32[] %Arg_0.3, f32[] %constant.4)
  // CHECK: }
  // CHECK: ENTRY %main.7 () -> f32[] {
  // CHECK:   %constant.1 = f32[] constant(42)
  // CHECK:   ROOT %call.6 = f32[] call(f32[] %constant.1), to_apply=%add.2, is_composite=true, frontend_attributes={composite.attributes={},composite.name="foo.bar",composite.version="1"}
  // CHECK: }
  func.func @main() -> tensor<f32> {
    %0 = mhlo.constant dense<4.200000e+01> : tensor<f32>
    %1 = mhlo.composite "foo.bar" %0 {
      decomposition = @add,
      version = 1 : i32
    } : (tensor<f32>) -> tensor<f32>
    return %1 : tensor<f32>
  }
  func.func @add(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = mhlo.constant dense<2.000000e+00> : tensor<f32>
    %1 = mhlo.add %arg0, %0 : tensor<f32>
    return %1 : tensor<f32>
  }
}

// -----

// optional composite version
module @composite {
  // CHECK: HloModule composite, entry_computation_layout={()->f32[]}
  // CHECK: %add.2 (Arg_0.3: f32[]) -> f32[] {
  // CHECK:   %Arg_0.3 = f32[] parameter(0)
  // CHECK:   %constant.4 = f32[] constant(2)
  // CHECK:   ROOT %add.5 = f32[] add(f32[] %Arg_0.3, f32[] %constant.4)
  // CHECK: }
  // CHECK: ENTRY %main.7 () -> f32[] {
  // CHECK:   %constant.1 = f32[] constant(42)
  // CHECK:   ROOT %call.6 = f32[] call(f32[] %constant.1), to_apply=%add.2, is_composite=true, frontend_attributes={composite.attributes={n = 1 : i32, tensor = dense<1> : tensor<i32>},composite.name="foo.bar",composite.version="0"}
  // CHECK: }
  func.func @main() -> tensor<f32> {
    %0 = mhlo.constant dense<4.200000e+01> : tensor<f32>
    %1 = mhlo.composite "foo.bar" %0 {
      composite_attributes = {
        n = 1 : i32,
        tensor = dense<1> : tensor<i32>
      },
      decomposition = @add
    } : (tensor<f32>) -> tensor<f32>
    return %1 : tensor<f32>
  }
  func.func @add(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = mhlo.constant dense<2.000000e+00> : tensor<f32>
    %1 = mhlo.add %arg0, %0 : tensor<f32>
    return %1 : tensor<f32>
  }
}

// -----

// optional composite attributes and version
module @composite {
  // CHECK: HloModule composite, entry_computation_layout={()->f32[]}
  // CHECK: %add.2 (Arg_0.3: f32[]) -> f32[] {
  // CHECK:   %Arg_0.3 = f32[] parameter(0)
  // CHECK:   %constant.4 = f32[] constant(2)
  // CHECK:   ROOT %add.5 = f32[] add(f32[] %Arg_0.3, f32[] %constant.4)
  // CHECK: }
  // CHECK: ENTRY %main.7 () -> f32[] {
  // CHECK:   %constant.1 = f32[] constant(42)
  // CHECK:   ROOT %call.6 = f32[] call(f32[] %constant.1), to_apply=%add.2, is_composite=true, frontend_attributes={composite.attributes={},composite.name="foo.bar",composite.version="0"}
  // CHECK: }
  func.func @main() -> tensor<f32> {
    %0 = mhlo.constant dense<4.200000e+01> : tensor<f32>
    %1 = mhlo.composite "foo.bar" %0 {
      decomposition = @add
    } : (tensor<f32>) -> tensor<f32>
    return %1 : tensor<f32>
  }
  func.func @add(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = mhlo.constant dense<2.000000e+00> : tensor<f32>
    %1 = mhlo.add %arg0, %0 : tensor<f32>
    return %1 : tensor<f32>
  }
}
