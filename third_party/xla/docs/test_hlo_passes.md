# Writing unit tests for HLO passes

There are different ways to write unit test for HLO passes. This page describes
the preferred method to ensure consistency and readability.

## `FileCheck` with `CHECK` lines interleaved

Most HLO passes can be tested using
[`FileCheck`](https://llvm.org/docs/CommandGuide/FileCheck.html) tests.
Interleave `CHECK` lines in input HLO module texts, and make sure to use `//
CHECK` instead of `; CHECK` uniformly as the `FileCheck` delimiter.

For example, you can re-write the
[`fusion cc_test` for a `priotity_fusion` pass](https://github.com/openxla/xla/blob/fe30942a406659bff75399a2a10585bbd1287e07/xla/service/gpu/transforms/priority_fusion_test.cc#L133-L149)
as follows:

```
TEST_F(PriorityFusionTest, FuseBroadcastIntoBitcastConsumers) {
  absl::string_view kHlo = R"(
    HloModule test_module

    // CHECK: ENTRY main
    ENTRY main {
      // CHECK-NEXT: %[[PARAM:.*]] = f32[96]{0} parameter(0)
      param_0 = f32[96]{0} parameter(0)
      broadcast = f32[8,96,128,7]{3,2,1,0} broadcast(param_0), dimensions={1}
      bitcast.6079.2 = f32[8,24,4,128,7]{4,3,2,1,0} bitcast(broadcast)
      // CHECK-NEXT: ROOT %{{.*}} fusion(%[[PARAM]]) {{.*}}
      ROOT transpose.1990.2 = f32[8,24,128,7,4]{4,3,2,1,0} transpose(bitcast.6079.2), dimensions={0,1,3,4,2}
    }
  )";
  RunAndFilecheckHloRewrite(kHlo, std::move(priority_fusion_));
}
```

Note: Currently, the codebase has some tests where input HLO module and expected
module are written separately. Inlining the `CHECK` lines is the preferred
method for future tests. It enables better readability and a similar signature
as MLIR based tests
[like in `stablehlo_aggressive_folder.mlir`](https://github.com/openxla/stablehlo/blob/main/stablehlo/tests/transforms/stablehlo_aggressive_folder.mlir#L31-L39).

## `LIT` runner and `hlo-opt`

Where feasible, use [`LIT`](https://llvm.org/docs/CommandGuide/lit.html) runner
and `hlo-opt`, and place `CHECK` lines locally next to the input IR they
correspond to. Again, make sure to use `// CHECK` instead of `; CHECK` as the
delimiter.

For example, some
[GPU tests](https://github.com/openxla/xla/tree/main/xla/service/gpu/tests) can
be written as follows:

```
// RUN: hlo-opt %s --platform=gpu --stage=llvm-before-optimizations --xla_gpu_target_config_filename=%S/../../../tools/hlo_opt/gpu_specs/%{GPU}.txtpb | FileCheck --check-prefixes=CHECK-%{PTX} %s

HloModule Test, is_scheduled=true
fused_computation {
  param_0 = f32[100,200]{1,0} parameter(0)
  ROOT b.1 = f32[200,100]{1,0} transpose(f32[100,200]{1,0} param_0), dimensions={1,0}
}
ENTRY main {
  a = f32[100, 200]{1,0} parameter(0)
  // CHECK-PTX:         call void @llvm.nvvm.barrier0
  // CHECK-GCN:         call void @llvm.amdgcn.s.barrier
  ROOT wrapped_b = f32[200,100]{1,0} fusion(f32[100,200]{1,0} a), kind=kInput, calls=fused_computation
}
```

## Automated `CHECK`-generation script

Writing test checks manually can be a lot of work, so it's often more practical
to run an optimizer, read over the results to make sure they match expectations,
and then convert the optimized HLO into `CHECK` directives. To simplify this
process, you can use
[`generate_hlo_test_checks.py`](https://github.com/openxla/xla/tree/main/xla/hlo/tools/generate_hlo_test_checks.py)
to automatically insert generated `CHECK` directives above each test case in an
HLO file.

> IMPORTANT: This tool inherently assumes that the pass's current behavior is
> correct, so make sure to look over the generated `CHECK` lines yourself and
> confirm that they match the output you expect.

## (Don't) Graph traversal

Refrain from writing tests that travel leaf nodes of the result graph and match
with expected op. These tests are tedious to write, difficult to quickly read,
and more difficult to debug and fix. Use one of the above options instead.
