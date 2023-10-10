// RUN: tfg-opt-no-passes --split-input-file --verify-diagnostics %s

tfg.func @test(%arg0: tensor<i32>) -> (tensor<i32>) {
  // expected-error@+1 {{expected 1 region attribute(s) but got 0}}
  %Case, %ctl = CaseRegion %arg0 {
    yield(%arg0) : tensor<i32>
  } {region_attrs = []}
  : (tensor<i32>) -> (tensor<i32>)
  return(%Case) : tensor<i32>
}

// -----

tfg.func @test(%arg0: tensor<i32>) -> (tensor<i32>) {
  // expected-error@+1 {{has 1 result(s) but preserved attributes has 2}}
  %Case, %ctl = CaseRegion %arg0 {
    yield(%arg0) : tensor<i32>
  } {region_attrs = [#tfg.region_attrs<{} [] [{}, {}]>]}
  : (tensor<i32>) -> (tensor<i32>)
  return(%Case) : tensor<i32>
}

// -----

tfg.func @test(%arg: tensor<i32>) -> (tensor<i32>) {
  // expected-error@+1 {{has 2 argument(s) but preserved attributes has 3}}
  %For, %ctl_1 = ForRegion(%arg) from %arg to %arg by %arg {
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: !tf_type.control, %arg3: !tf_type.control):
    yield(%arg1) : tensor<i32>
  } {region_attrs = #tfg.region_attrs<{} [{}, {}, {}] []>}
  : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> (tensor<i32>)
  return(%For) : tensor<i32>
}

// -----

tfg.graph #tf_type.version<producer = 42, min_consumer = 33> {
  %Op:2, %ctl = Op : () -> (tensor<*xi32>, tensor<*xi32>)
  // expected-error@+1 {{has 1 result(s) but preserved attributes has 2}}
  %WhileRegion:2, %ctl_0 = WhileRegion(%Op#0, %Op#1) [%ctl] {
  ^bb0(%arg0: tensor<*xi32>, %arg1: tensor<*xi32>, %arg2: !tf_type.control, %arg3: !tf_type.control):
    %cond, %ctl_1 = Op : () -> (tensor<*xi1>)
    condition %cond : tensor<*xi1> (%arg0, %arg1) : tensor<*xi32>, tensor<*xi32>
  } do {
  ^bb0(%arg0: tensor<*xi32>, %arg1: tensor<*xi32>, %arg2: !tf_type.control, %arg3: !tf_type.control):
    yield(%arg0, %arg1) : tensor<*xi32>, tensor<*xi32>
  } {parallel_iterations = 10 : i64,
     body_region_attrs = #tfg.region_attrs<{} [{}, {}] [{}, {}]>,
     cond_region_attrs = #tfg.region_attrs<{} [{}, {}] [{}, {}]>}
  : (tensor<*xi32>, tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi32>)
}
