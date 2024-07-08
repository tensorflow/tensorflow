// RUN: xla_sdy_opt %s -sdy-import-pipeline 2>&1 | FileCheck %s

// Make sure this temp attr doesn't exist anymore.
// CHECK-NOT: xla.sdy.sharding

// CHECK: sdy.mesh @mesh = <"a"=2, "b"=2, "c"=2>
sdy.mesh @mesh = <"a"=2, "b"=2, "c"=2>

func.func @main(
  // CHECK: %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {"b"}p4]>})
  %arg0: tensor<8x16xf32>           {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {"b"}p4]>}
  ) -> (tensor<8x16xf32>) {
  %0 = mhlo.add %arg0, %arg0 : tensor<8x16xf32>
  %1 = mhlo.add %0, %0 : tensor<8x16xf32>
  return %1 : tensor<8x16xf32>
}

