// RUN: mlir-translate -test-spirv-roundtrip %s | FileCheck %s

spv.module "Logical" "GLSL450" {
  spv.globalVariable @var1 : !spv.ptr<!spv.array<4xf32>, Input>
  func @fmain() -> i32 {
    %0 = spv.constant 16 : i32
    %1 = spv._address_of @var1 : !spv.ptr<!spv.array<4xf32>, Input>
    // CHECK: {{%.*}} = spv.FunctionCall @f_0({{%.*}}) : (i32) -> i32
    %3 = spv.FunctionCall @f_0(%0) : (i32) -> i32
    // CHECK: spv.FunctionCall @f_1({{%.*}}, {{%.*}}) : (i32, !spv.ptr<!spv.array<4 x f32>, Input>) -> ()
    spv.FunctionCall @f_1(%3, %1) : (i32, !spv.ptr<!spv.array<4xf32>, Input>) ->  ()
    // CHECK: {{%.*}} =  spv.FunctionCall @f_2({{%.*}}) : (!spv.ptr<!spv.array<4 x f32>, Input>) -> !spv.ptr<!spv.array<4 x f32>, Input>
    %4 = spv.FunctionCall @f_2(%1) : (!spv.ptr<!spv.array<4xf32>, Input>) -> !spv.ptr<!spv.array<4xf32>, Input>
    spv.ReturnValue %3 : i32
  }
  func @f_0(%arg0 : i32) -> i32 {
    spv.ReturnValue %arg0 : i32
  }
  func @f_1(%arg0 : i32, %arg1 : !spv.ptr<!spv.array<4xf32>, Input>) -> () {
    spv.Return
  }
  func @f_2(%arg0 : !spv.ptr<!spv.array<4xf32>, Input>) -> !spv.ptr<!spv.array<4xf32>, Input> {
    spv.ReturnValue %arg0 : !spv.ptr<!spv.array<4xf32>, Input>
  }

  func @f_loop_with_function_call(%count : i32) -> () {
    %zero = spv.constant 0: i32
    %var = spv.Variable init(%zero) : !spv.ptr<i32, Function>
    spv.loop {
      spv.Branch ^header
    ^header:
      %val0 = spv.Load "Function" %var : i32
      %cmp = spv.SLessThan %val0, %count : i32
      spv.BranchConditional %cmp, ^body, ^merge
    ^body:
      spv.Branch ^continue
    ^continue:
      // CHECK: spv.FunctionCall @f_inc({{%.*}}) : (!spv.ptr<i32, Function>) -> ()
      spv.FunctionCall @f_inc(%var) : (!spv.ptr<i32, Function>) -> ()
      spv.Branch ^header
    ^merge:
      spv._merge
    }
    spv.Return
  }
  func @f_inc(%arg0 : !spv.ptr<i32, Function>) -> () {
      %one = spv.constant 1 : i32
      %0 = spv.Load "Function" %arg0 : i32
      %1 = spv.IAdd %0, %one : i32
      spv.Store "Function" %arg0, %1 : i32
      spv.Return
  }
}
