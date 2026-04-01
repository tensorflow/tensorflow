// RUN: xla-opt %s -split-input-file -triton-xla-math-to-libdevice=' \
// RUN: libdevice_path=/path/to/libdevice triple=amdgcn-amd-amdhsa' \
// RUN: | FileCheck %s

// -----
// f32: no casting, uses __ocml_*_f32 symbol.

func.func @cos(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
  %result = math.cos %arg0 : tensor<1024xf32>
  return %result : tensor<1024xf32>
}

// CHECK:       tt.extern_elementwise %arg0
// CHECK-SAME:    {libname = "libdevice", libpath = "/path/to/libdevice",
// CHECK-SAME:    pure = true, symbol = "__ocml_cos_f32"}

// -----
// f16 with a native f16 implementation: no upcast/downcast, uses __ocml_*_f16 symbol.

func.func @cos_f16(%arg0: tensor<1024xf16>) -> tensor<1024xf16> {
  %result = math.cos %arg0 : tensor<1024xf16>
  return %result : tensor<1024xf16>
}

// CHECK-NOT:   arith.extf
// CHECK:       tt.extern_elementwise %arg0
// CHECK-SAME:    {libname = "libdevice", libpath = "/path/to/libdevice",
// CHECK-SAME:    pure = true, symbol = "__ocml_cos_f16"}
// CHECK-NOT:   arith.truncf

// -----
// f16 without a native f16 implementation: upcast to f32, downcast back.

func.func @roundeven_f16(%arg0: tensor<1024xf16>) -> tensor<1024xf16> {
  %result = math.roundeven %arg0 : tensor<1024xf16>
  return %result : tensor<1024xf16>
}

// CHECK:       %[[CAST:.*]] = arith.extf %arg0 : tensor<1024xf16> to tensor<1024xf32>
// CHECK:       tt.extern_elementwise %[[CAST]]
// CHECK-SAME:    {libname = "libdevice", libpath = "/path/to/libdevice",
// CHECK-SAME:    pure = true, symbol = "__ocml_rint_f32"}
// CHECK:       arith.truncf {{.*}} : tensor<1024xf32> to tensor<1024xf16>

// -----
// bf16: always upcast to f32, downcast back (no native bf16 implementation).

func.func @exp_bf16(%arg0: tensor<1024xbf16>) -> tensor<1024xbf16> {
  %result = math.exp %arg0 : tensor<1024xbf16>
  return %result : tensor<1024xbf16>
}

// CHECK:       %[[CAST:.*]] = arith.extf %arg0 : tensor<1024xbf16> to tensor<1024xf32>
// CHECK:       tt.extern_elementwise %[[CAST]]
// CHECK-SAME:    {libname = "libdevice", libpath = "/path/to/libdevice",
// CHECK-SAME:    pure = true, symbol = "__ocml_exp_f32"}
// CHECK:       arith.truncf {{.*}} : tensor<1024xf32> to tensor<1024xbf16>

// -----
// f16 with a native f16 implementation: no upcast/downcast.

func.func @exp_f16(%arg0: tensor<1024xf16>) -> tensor<1024xf16> {
  %result = math.exp %arg0 : tensor<1024xf16>
  return %result : tensor<1024xf16>
}

// CHECK-NOT:   arith.extf
// CHECK:       tt.extern_elementwise %arg0
// CHECK-SAME:    {libname = "libdevice", libpath = "/path/to/libdevice",
// CHECK-SAME:    pure = true, symbol = "__ocml_exp_f16"}
// CHECK-NOT:   arith.truncf
