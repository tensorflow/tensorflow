// RUN: mlir-opt %s -pipeline-data-transfer | FileCheck %s

// CHECK-LABEL: mlfunc @loop_nest_simple() {
// CHECK:       %c8 = constant 8 : affineint
// CHECK-NEXT:  %c0 = constant 0 : affineint
// CHECK-NEXT:  %0 = "foo"(%c0) : (affineint) -> affineint
// CHECK-NEXT:  for %i0 = 1 to 7 {
// CHECK-NEXT:    %1 = "foo"(%i0) : (affineint) -> affineint
// CHECK-NEXT:    %2 = affine_apply #map0(%i0)
// CHECK-NEXT:    %3 = "bar"(%2) : (affineint) -> affineint
// CHECK-NEXT:  }
// CHECK-NEXT:  %4 = affine_apply #map0(%c8)
// CHECK-NEXT:  %5 = "bar"(%4) : (affineint) -> affineint
// CHECK-NEXT:  return
mlfunc @loop_nest_simple() {
  for %i = 0 to 7 {
    %y = "foo"(%i) : (affineint) -> affineint
    %x = "bar"(%i) : (affineint) -> affineint
  }
  return
}

// CHECK-LABEL: mlfunc @loop_nest_dma() {
// CHECK:       %c8 = constant 8 : affineint
// CHECK-NEXT:  %c0 = constant 0 : affineint
// CHECK-NEXT:  %0 = affine_apply #map1(%c0)
// CHECK-NEXT:  %1 = "dma.enqueue"(%0) : (affineint) -> affineint
// CHECK-NEXT:  %2 = "dma.enqueue"(%0) : (affineint) -> affineint
// CHECK-NEXT:  for %i0 = 1 to 7 {
// CHECK-NEXT:    %3 = affine_apply #map1(%i0)
// CHECK-NEXT:    %4 = "dma.enqueue"(%3) : (affineint) -> affineint
// CHECK-NEXT:    %5 = "dma.enqueue"(%3) : (affineint) -> affineint
// CHECK-NEXT:    %6 = affine_apply #map0(%i0)
// CHECK-NEXT:    %7 = affine_apply #map1(%6)
// CHECK-NEXT:    %8 = "dma.wait"(%7) : (affineint) -> affineint
// CHECK-NEXT:    %9 = "compute1"(%7) : (affineint) -> affineint
// CHECK-NEXT:  }
// CHECK-NEXT:  %10 = affine_apply #map0(%c8)
// CHECK-NEXT:  %11 = affine_apply #map1(%10)
// CHECK-NEXT:  %12 = "dma.wait"(%11) : (affineint) -> affineint
// CHECK-NEXT:  %13 = "compute1"(%11) : (affineint) -> affineint
// CHECK-NEXT:  return
mlfunc @loop_nest_dma() {
  for %i = 0 to 7 {
    %pingpong = affine_apply (d0) -> (d0 mod 2) (%i)
    "dma.enqueue"(%pingpong) : (affineint) -> affineint
    "dma.enqueue"(%pingpong) : (affineint) -> affineint
    %pongping = affine_apply (d0) -> (d0 mod 2) (%i)
    "dma.wait"(%pongping) : (affineint) -> affineint
    "compute1"(%pongping) : (affineint) -> affineint
  }
  return
}

// CHECK-LABEL: mlfunc @loop_nest_bound_map(%arg0 : affineint) {
// CHECK:       %0 = affine_apply #map2()[%arg0]
// CHECK-NEXT:  %1 = "foo"(%0) : (affineint) -> affineint
// CHECK-NEXT:  %2 = "bar"(%0) : (affineint) -> affineint
// CHECK-NEXT:  for %i0 = #map3()[%arg0] to #map4()[%arg0] {
// CHECK-NEXT:    %3 = "foo"(%i0) : (affineint) -> affineint
// CHECK-NEXT:    %4 = "bar"(%i0) : (affineint) -> affineint
// CHECK-NEXT:    %5 = affine_apply #map0(%i0)
// CHECK-NEXT:    %6 = "foo_bar"(%5) : (affineint) -> affineint
// CHECK-NEXT:    %7 = "bar_foo"(%5) : (affineint) -> affineint
// CHECK-NEXT:  }
// CHECK-NEXT:  %8 = affine_apply #map5()[%arg0]
// CHECK-NEXT:  %9 = affine_apply #map0(%8)
// CHECK-NEXT:  %10 = "foo_bar"(%9) : (affineint) -> affineint
// CHECK-NEXT:  %11 = "bar_foo"(%9) : (affineint) -> affineint
// CHECK-NEXT:  return
mlfunc @loop_nest_bound_map(%N : affineint) {
  for %i = %N to ()[s0] -> (s0 + 7)()[%N] {
    "foo"(%i) : (affineint) -> affineint
    "bar"(%i) : (affineint) -> affineint
    "foo_bar"(%i) : (affineint) -> (affineint)
    "bar_foo"(%i) : (affineint) -> (affineint)
  }
  return
}
