// RUN: triton-opt --split-input-file %s --verify-diagnostics

#blocked = #ttg.blocked<{
    sizePerThread=[1, 1],
    threadsPerWarp=[16, 1],
    warpsPerCTA=[4, 1],
    order=[0, 1],
    CTAsPerCGA=[2, 1],
    CTASplitNum=[1, 1],
    CTAOrder=[0, 1]
}>
module attributes {
    "ttg.num-warps" = 4 : i32,
    "ttg.num-ctas" = 2 : i32,
    "ttg.threads-per-warp" = 32 : i32
} {
    tt.func public @fn(%arg0: !tt.ptr<i32>) {
        // expected-error @+1 {{threads per warp}}
        %t = tt.splat %arg0 : !tt.ptr<i32,1> -> tensor<8x1x!tt.ptr<i32,1>, #blocked>
        tt.return
    }
}

// -----

#blocked = #ttg.blocked<{
    sizePerThread=[1, 1],
    threadsPerWarp=[32, 1],
    warpsPerCTA=[4, 2],
    order=[0, 1],
    CTAsPerCGA=[2, 1],
    CTASplitNum=[1, 1],
    CTAOrder=[0, 1]
}>
module attributes {
    "ttg.num-warps" = 4 : i32,
    "ttg.num-ctas" = 2 : i32,
    "ttg.threads-per-warp" = 32 : i32
} {
    tt.func public @fn(%arg0: !tt.ptr<i32>) {
        // expected-error @+1 {{warps per CTA}}
        %t = tt.splat %arg0 : !tt.ptr<i32,1> -> tensor<8x1x!tt.ptr<i32,1>, #blocked>
        tt.return
    }
}

// -----

#blocked = #ttg.blocked<{
    sizePerThread=[1, 1],
    threadsPerWarp=[32, 1],
    warpsPerCTA=[4, 1],
    order=[0, 1],
    CTAsPerCGA=[1, 1],
    CTASplitNum=[1, 1],
    CTAOrder=[0, 1]
}>
module attributes {
    "ttg.num-warps" = 4 : i32,
    "ttg.num-ctas" = 2 : i32,
    "ttg.threads-per-warp" = 32 : i32
} {
    tt.func public @fn(%arg0: !tt.ptr<i32>) {
        // expected-error @+1 {{CTAs per CGA}}
        %t = tt.splat %arg0 : !tt.ptr<i32,1> -> tensor<8x1x!tt.ptr<i32,1>, #blocked>
        tt.return
    }
}

// -----

#blocked = #ttg.blocked<{
    sizePerThread=[1, 1],
    threadsPerWarp=[32, 1],
    warpsPerCTA=[4, 1],
    order=[0, 1],
    CTAsPerCGA=[1, 2],
    CTASplitNum=[1, 1],
    CTAOrder=[0, 1]
}>
module attributes {
    "ttg.num-warps" = 4 : i32,
    "ttg.num-ctas" = 2 : i32,
    "ttg.threads-per-warp" = 32 : i32
} {
    tt.func public @fn(%arg0: !tt.ptr<i32>) {
        // Note it's a 3d tensor here, but #blocked is 2D.
        // expected-error @+1 {{rank}}
        %t = tt.splat %arg0 : !tt.ptr<i32,1> -> tensor<8x1x1x!tt.ptr<i32,1>, #blocked>
        tt.return
    }
}

// -----

#blocked = #ttg.blocked<{
    sizePerThread=[1, 1],
    threadsPerWarp=[32, 1],
    warpsPerCTA=[4, 1],
    order=[0, 1],
    CTAsPerCGA=[1, 2],
    CTASplitNum=[1, 1],
    CTAOrder=[0, 1]
}>
module attributes {
    "ttg.num-warps" = 4 : i32,
    "ttg.num-ctas" = 2 : i32,
    "ttg.threads-per-warp" = 32 : i32
} {
    tt.func public @fn(%arg0: tensor<8xf32, #blocked>) {
        // expected-error @+1 {{rank}}
        %t = tt.expand_dims %arg0 {axis = 0 : i32} : tensor<8xf32, #blocked> -> tensor<8x1xf32, #blocked>
        tt.return
    }
}
