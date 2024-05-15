// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @absi() -> i32 {
  %a = arith.constant -1 : i32
  %ret = math.absi %a : i32
  return %ret : i32
}

// CHECK-LABEL: @absi
// CHECK-NEXT: Results
// CHECK-NEXT: 1

func.func @absf() -> (f32, f32) {
  %a = arith.constant 1.0 : f32
  %b = arith.constant -1.0 : f32
  %c = math.absf %a : f32
  %d = math.absf %b : f32
  return %c, %d : f32, f32
}

// CHECK-LABEL: @absf
// CHECK-NEXT: Results
// CHECK-NEXT: 1.000000e+00
// CHECK-NEXT: 1.000000e+00

func.func @atan() -> f32 {
  %c1 = arith.constant 1.0 : f32
  %ret = math.atan %c1 : f32
  return %ret : f32
}

// CHECK-LABEL: @atan
// CHECK-NEXT: Results
// CHECK-NEXT: 7.853982e-01

func.func @atan2() -> f32 {
  %c10 = arith.constant 10.0 : f32
  %c1 = arith.constant 1.0 : f32
  %ret = math.atan2 %c10, %c1 : f32
  return %ret : f32
}

// CHECK-LABEL: @atan2
// CHECK-NEXT: Results
// CHECK-NEXT: 1.471128e+00

func.func @cbrt() -> f32 {
  %c-27 = arith.constant -27.0 : f32
  %ret = math.cbrt %c-27 : f32
  return %ret : f32
}

// CHECK-LABEL: @cbrt
// CHECK-NEXT: Results
// CHECK-NEXT: -3.000000e+00

func.func @ceil() -> f32 {
  %a = arith.constant 1.234 : f32
  %ret = math.ceil %a : f32
  return %ret : f32
}

// CHECK-LABEL: @ceil
// CHECK-NEXT: Results
// CHECK-NEXT: 2.000000e+00

func.func @ctlz() -> i8 {
  %c1 = arith.constant 1 : i8
  %ret = math.ctlz %c1 : i8
  return %ret : i8
}

// CHECK-LABEL: @ctlz
// CHECK-NEXT: Results
// CHECK-NEXT: 7

func.func @ctpop() -> i8 {
  %c1 = arith.constant 17 : i8
  %ret = math.ctpop %c1 : i8
  return %ret : i8
}

// CHECK-LABEL: @ctpop
// CHECK-NEXT: Results
// CHECK-NEXT: i8: 2

func.func @ctz() -> i8 {
  %c1 = arith.constant 8 : i8
  %ret = math.cttz %c1 : i8
  return %ret : i8
}

// CHECK-LABEL: @ctz
// CHECK-NEXT: Results
// CHECK-NEXT: 3

func.func @copysign() -> (f32, f32) {
  %a = arith.constant 1.5 : f32
  %b = arith.constant -10.0 : f32
  %c = math.copysign %a, %b : f32
  %d = math.copysign %a, %a : f32
  return %c, %d : f32, f32
}

// CHECK-LABEL: @copysign
// CHECK-NEXT: Results
// CHECK-NEXT: -1.500000e+00
// CHECK-NEXT: 1.500000e+00

func.func @cos() -> f32 {
  %a = arith.constant 0.0 : f32
  %b = math.cos %a : f32
  return %b : f32
}

// CHECK-LABEL: @cos
// CHECK-NEXT: Results
// CHECK-NEXT: 1.000000e+00

func.func @erf() -> f32 {
  %a = arith.constant 1.0 : f32
  %b = math.erf %a : f32
  return %b : f32
}

// CHECK-LABEL: @erf
// CHECK-NEXT: Results
// CHECK-NEXT: 8.427008e-01

func.func @exp() -> f32 {
  %a = arith.constant 1.0 : f32
  %b = math.exp %a : f32
  return %b : f32
}

// CHECK-LABEL: @exp
// CHECK-NEXT: Results
// CHECK-NEXT: 2.718282e+00

func.func @exp2() -> f32 {
  %a = arith.constant 3.0 : f32
  %b = math.exp2 %a : f32
  return %b : f32
}

// CHECK-LABEL: @exp2
// CHECK-NEXT: Results
// CHECK-NEXT: 8.000000e+00

func.func @expm1() -> f32 {
  %a = arith.constant 1.0e-06 : f32
  %b = math.expm1 %a : f32
  return %b : f32
}

// CHECK-LABEL: @expm1
// CHECK-NEXT: Results
// CHECK-NEXT: 1.000000e-06

func.func @floor() -> (f32, f32) {
  %a = arith.constant 1.5 : f32
  %b = arith.constant -10.5 : f32
  %c = math.floor %a : f32
  %d = math.floor %b : f32
  return %c, %d : f32, f32
}

// CHECK-LABEL: @floor
// CHECK-NEXT: Results
// CHECK-NEXT: 1.000000e+00
// CHECK-NEXT: -1.100000e+01

func.func @ipowi() -> i32 {
  %a = arith.constant 2 : i32
  %b = arith.constant 3 : i32
  %c = math.ipowi %a, %b : i32
  return %c : i32
}

// CHECK-LABEL: @ipowi
// CHECK-NEXT: Results
// CHECK-NEXT: i32: 8

func.func @log() -> f32 {
  %a = arith.constant 1.0 : f32
  %ret = math.log %a : f32
  return %ret : f32
}

// CHECK-LABEL: @log
// CHECK-NEXT: Results
// CHECK-NEXT: 0.000000e+00

func.func @log10() -> f32 {
  %a = arith.constant 1.0e12 : f32
  %ret = math.log10 %a : f32
  return %ret : f32
}

// CHECK-LABEL: @log
// CHECK-NEXT: Results
// CHECK-NEXT: 1.200000e+01

func.func @log1p() -> f32 {
  %a = arith.constant 1.0e-10 : f32
  %ret = math.log1p %a : f32
  return %ret : f32
}

// CHECK-LABEL: @log1p
// CHECK-NEXT: Results
// CHECK-NEXT: 1.000000e-10

func.func @log2() -> f32 {
  %a = arith.constant 65536.0 : f32
  %ret = math.log2 %a : f32
  return %ret : f32
}

// CHECK-LABEL: @log2
// CHECK-NEXT: Results
// CHECK-NEXT: 1.600000e+01

func.func @powf() -> f32 {
  %a = arith.constant 2.0 : f32
  %b = arith.constant 3.0 : f32
  %c = math.powf %a, %b : f32
  return %c : f32
}

// CHECK-LABEL: @powf
// CHECK-NEXT: Results
// CHECK-NEXT: 8.000000e+00

//REGISTER_MLIR_INTERPRETER_OP("math.round", applyCwiseMap<Round>);
//REGISTER_MLIR_INTERPRETER_OP("math.roundeven",  applyCwiseMap<NearbyInt>);
//REGISTER_MLIR_INTERPRETER_OP("math.rsqrt", applyCwiseMap<RSqrt>);

func.func @sin() -> f32 {
  %a = arith.constant 0.0 : f32
  %b = math.sin %a : f32
  return %b : f32
}

// CHECK-LABEL: @sin
// CHECK-NEXT: Results
// CHECK-NEXT: 0.000000e+00

//REGISTER_MLIR_INTERPRETER_OP("math.sqrt", applyCwiseMap<Sqrt>);
//REGISTER_MLIR_INTERPRETER_OP("math.tan", applyCwiseMap<Tan>);
//REGISTER_MLIR_INTERPRETER_OP("math.tanh", applyCwiseMap<TanH>);
//REGISTER_MLIR_INTERPRETER_OP("math.trunc", applyCwiseMap<Trunc>);
