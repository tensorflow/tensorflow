// RUN: ifrt-opt %s -split-input-file -verify-diagnostics

func.func @devices_should_be_distinct() {
  // expected-error@+2 {{`devices` has duplicated id 0}}
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<4x4xi32>, "shard0", [0,0]>
  return
}
