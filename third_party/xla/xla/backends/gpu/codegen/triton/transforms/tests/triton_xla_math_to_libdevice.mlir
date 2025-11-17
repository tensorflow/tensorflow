// RUN: xla-opt %s -split-input-file -triton-xla-math-to-libdevice=' \
// RUN: libdevice_path=/path/to/libdevice triple=nvptx64-unknown-unknown' \
// RUN: | FileCheck %s

func.func @main(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
  %result = math.acos %arg0 : tensor<1024xf32>
  return %result : tensor<1024xf32>
}

// CHECK:       tt.extern_elementwise %arg0
// CHECK-SAME:    {libname = "libdevice", libpath = "/path/to/libdevice",
// CHECK-SAME:    pure = true, symbol = "__nv_acosf"}

// -----

func.func @acosh(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
  %result = math.acosh %arg0 : tensor<1024xf32>
  return %result : tensor<1024xf32>
}

// CHECK:       tt.extern_elementwise %arg0
// CHECK-SAME:    {libname = "libdevice", libpath = "/path/to/libdevice",
// CHECK-SAME:    pure = true, symbol = "__nv_acoshf"}

// -----

func.func @asin(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
  %result = math.asin %arg0 : tensor<1024xf32>
  return %result : tensor<1024xf32>
}

// CHECK:       tt.extern_elementwise %arg0
// CHECK-SAME:    {libname = "libdevice", libpath = "/path/to/libdevice",
// CHECK-SAME:    pure = true, symbol = "__nv_asinf"}

// -----

func.func @asinh(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
  %result = math.asinh %arg0 : tensor<1024xf32>
  return %result : tensor<1024xf32>
}

// CHECK:       tt.extern_elementwise %arg0
// CHECK-SAME:    {libname = "libdevice", libpath = "/path/to/libdevice",
// CHECK-SAME:    pure = true, symbol = "__nv_asinhf"}

// -----

func.func @atan2(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
  %result = math.atan2 %arg0, %arg1 : tensor<1024xf32>
  return %result : tensor<1024xf32>
}

// CHECK:       tt.extern_elementwise %arg0, %arg1
// CHECK-SAME:    {libname = "libdevice", libpath = "/path/to/libdevice",
// CHECK-SAME:    pure = true, symbol = "__nv_atan2f"}

// -----

func.func @atanh(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
  %result = math.atanh %arg0 : tensor<1024xf32>
  return %result : tensor<1024xf32>
}

// CHECK:       tt.extern_elementwise %arg0
// CHECK-SAME:    {libname = "libdevice", libpath = "/path/to/libdevice",
// CHECK-SAME:    pure = true, symbol = "__nv_atanhf"}

// -----

func.func @cos(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
  %result = math.cos %arg0 : tensor<1024xf32>
  return %result : tensor<1024xf32>
}

// CHECK:       tt.extern_elementwise %arg0
// CHECK-SAME:    {libname = "libdevice", libpath = "/path/to/libdevice",
// CHECK-SAME:    pure = true, symbol = "__nv_cosf"}

// -----

func.func @cosh(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
  %result = math.cosh %arg0 : tensor<1024xf32>
  return %result : tensor<1024xf32>
}

// CHECK:       tt.extern_elementwise %arg0
// CHECK-SAME:    {libname = "libdevice", libpath = "/path/to/libdevice",
// CHECK-SAME:    pure = true, symbol = "__nv_coshf"}

// -----

func.func @exp(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
  %result = math.exp %arg0 : tensor<1024xf32>
  return %result : tensor<1024xf32>
}

// CHECK:       tt.extern_elementwise %arg0
// CHECK-SAME:    {libname = "libdevice", libpath = "/path/to/libdevice",
// CHECK-SAME:    pure = true, symbol = "__nv_expf"}

// -----

func.func @exp_bf16(%arg0: tensor<1024xbf16>) -> tensor<1024xbf16> {
  %result = math.exp %arg0 : tensor<1024xbf16>
  return %result : tensor<1024xbf16>
}

// CHECK:       %[[CAST:.*]] = arith.extf %arg0 : tensor<1024xbf16> to tensor<1024xf32>
// CHECK:       tt.extern_elementwise %[[CAST]]
// CHECK-SAME:    {libname = "libdevice", libpath = "/path/to/libdevice",
// CHECK-SAME:    pure = true, symbol = "__nv_fast_expf"}
// CHECK:       arith.truncf {{.*}} : tensor<1024xf32> to tensor<1024xbf16>

// -----

func.func @exp_f16(%arg0: tensor<1024xf16>) -> tensor<1024xf16> {
  %result = math.exp %arg0 : tensor<1024xf16>
  return %result : tensor<1024xf16>
}

// CHECK:       %[[CAST:.*]] = arith.extf %arg0 : tensor<1024xf16> to tensor<1024xf32>
// CHECK:       tt.extern_elementwise %[[CAST]]
// CHECK-SAME:    {libname = "libdevice", libpath = "/path/to/libdevice",
// CHECK-SAME:    pure = true, symbol = "__nv_fast_expf"}
// CHECK:       arith.truncf {{.*}} : tensor<1024xf32> to tensor<1024xf16>

// -----

func.func @erf(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
  %result = math.erf %arg0 : tensor<1024xf32>
  return %result : tensor<1024xf32>
}

// CHECK:       tt.extern_elementwise %arg0
// CHECK-SAME:    {libname = "libdevice", libpath = "/path/to/libdevice",
// CHECK-SAME:    pure = true, symbol = "__nv_erff"}

// -----

func.func @expm1(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
  %result = math.expm1 %arg0 : tensor<1024xf32>
  return %result : tensor<1024xf32>
}

// CHECK:       tt.extern_elementwise %arg0
// CHECK-SAME:    {libname = "libdevice", libpath = "/path/to/libdevice",
// CHECK-SAME:    pure = true, symbol = "__nv_expm1f"}

// -----

func.func @log(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
  %result = math.log %arg0 : tensor<1024xf32>
  return %result : tensor<1024xf32>
}

// CHECK:       tt.extern_elementwise %arg0
// CHECK-SAME:    {libname = "libdevice", libpath = "/path/to/libdevice",
// CHECK-SAME:    pure = true, symbol = "__nv_logf"}

// -----


func.func @log_bf16(%arg0: tensor<1024xbf16>) -> tensor<1024xbf16> {
  %result = math.log %arg0 : tensor<1024xbf16>
  return %result : tensor<1024xbf16>
}

// CHECK:       %[[CAST:.*]] = arith.extf %arg0 : tensor<1024xbf16> to tensor<1024xf32>
// CHECK:       tt.extern_elementwise %[[CAST]]
// CHECK-SAME:    {libname = "libdevice", libpath = "/path/to/libdevice",
// CHECK-SAME:    pure = true, symbol = "__nv_fast_logf"}
// CHECK:       arith.truncf {{.*}} : tensor<1024xf32> to tensor<1024xbf16>

// -----

func.func @log_f16(%arg0: tensor<1024xf16>) -> tensor<1024xf16> {
  %result = math.log %arg0 : tensor<1024xf16>
  return %result : tensor<1024xf16>
}

// CHECK:       %[[CAST:.*]] = arith.extf %arg0 : tensor<1024xf16> to tensor<1024xf32>
// CHECK:       tt.extern_elementwise %[[CAST]]
// CHECK-SAME:    {libname = "libdevice", libpath = "/path/to/libdevice",
// CHECK-SAME:    pure = true, symbol = "__nv_fast_logf"}
// CHECK:       arith.truncf {{.*}} : tensor<1024xf32> to tensor<1024xf16>

// -----

func.func @log1p(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
  %result = math.log1p %arg0 : tensor<1024xf32>
  return %result : tensor<1024xf32>
}

// CHECK:       tt.extern_elementwise %arg0
// CHECK-SAME:    {libname = "libdevice", libpath = "/path/to/libdevice",
// CHECK-SAME:    pure = true, symbol = "__nv_log1pf"}

// -----

func.func @powf(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
  %result = math.powf %arg0, %arg1 : tensor<1024xf32>
  return %result : tensor<1024xf32>
}

// CHECK:       tt.extern_elementwise %arg0, %arg1
// CHECK-SAME:    {libname = "libdevice", libpath = "/path/to/libdevice",
// CHECK-SAME:    pure = true, symbol = "__nv_powf"}

// -----

func.func @remf(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
  %result = arith.remf %arg0, %arg1 : tensor<1024xf32>
  return %result : tensor<1024xf32>
}

// CHECK:       tt.extern_elementwise %arg0, %arg1
// CHECK-SAME:    {libname = "libdevice", libpath = "/path/to/libdevice",
// CHECK-SAME:    pure = true, symbol = "__nv_fmodf"}

// -----

func.func @rsqrt(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
  %result = math.rsqrt %arg0 : tensor<1024xf32>
  return %result : tensor<1024xf32>
}

// CHECK:       tt.extern_elementwise %arg0
// CHECK-SAME:    {libname = "libdevice", libpath = "/path/to/libdevice",
// CHECK-SAME:    pure = true, symbol = "__nv_rsqrtf"}

// -----

func.func @sin(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
  %result = math.sin %arg0 : tensor<1024xf32>
  return %result : tensor<1024xf32>
}

// CHECK:       tt.extern_elementwise %arg0
// CHECK-SAME:    {libname = "libdevice", libpath = "/path/to/libdevice",
// CHECK-SAME:    pure = true, symbol = "__nv_sinf"}

// -----

func.func @sinh(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
  %result = math.sinh %arg0 : tensor<1024xf32>
  return %result : tensor<1024xf32>
}

// CHECK:       tt.extern_elementwise %arg0
// CHECK-SAME:    {libname = "libdevice", libpath = "/path/to/libdevice",
// CHECK-SAME:    pure = true, symbol = "__nv_sinhf"}

// -----

func.func @sqrt(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
  %result = math.sqrt %arg0 : tensor<1024xf32>
  return %result : tensor<1024xf32>
}

// CHECK:       tt.extern_elementwise %arg0
// CHECK-SAME:    {libname = "libdevice", libpath = "/path/to/libdevice",
// CHECK-SAME:    pure = true, symbol = "__nv_sqrtf"}

// -----

func.func @tan(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
  %result = math.tan %arg0 : tensor<1024xf32>
  return %result : tensor<1024xf32>
}

// CHECK:       tt.extern_elementwise %arg0
// CHECK-SAME:    {libname = "libdevice", libpath = "/path/to/libdevice",
// CHECK-SAME:    pure = true, symbol = "__nv_tanf"}

// -----

func.func @tanh(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
  %result = math.tanh %arg0 : tensor<1024xf32>
  return %result : tensor<1024xf32>
}

// CHECK:       tt.extern_elementwise %arg0
// CHECK-SAME:    {libname = "libdevice", libpath = "/path/to/libdevice",
// CHECK-SAME:    pure = true, symbol = "__nv_tanhf"}

// -----

func.func @cbrt(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
  %result = math.cbrt %arg0 : tensor<1024xf32>
  return %result : tensor<1024xf32>
}

// CHECK:       tt.extern_elementwise %arg0
// CHECK-SAME:    {libname = "libdevice", libpath = "/path/to/libdevice",
// CHECK-SAME:    pure = true, symbol = "__nv_cbrtf"}
