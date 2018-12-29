// RUN: mlir-opt -lower-affine-apply %s | FileCheck %s

#map0 = () -> (0)
#map1 = ()[s0] -> (s0)
#map2 = (d0) -> (d0)
#map3 = (d0)[s0] -> (d0 + s0 + 1)
#map4 = (d0,d1,d2,d3)[s0,s1,s2] -> (d0 + 2*d1 + 3*d2 + 4*d3 + 5*s0 + 6*s1 + 7*s2)
#map5 = (d0,d1,d2) -> (d0,d1,d2)
#map6 = (d0,d1,d2) -> (d0 + d1 + d2)

// CHECK-LABEL: cfgfunc @affine_applies()
cfgfunc @affine_applies() {
^bb0:
// CHECK: %c0 = constant 0 : index
  %zero = affine_apply #map0()

// Identity maps are just discarded.
// CHECK-NEXT: %c101 = constant 101 : index
  %101 = constant 101 : index
  %symbZero = affine_apply #map1()[%zero]
// CHECK-NEXT: %c102 = constant 102 : index
  %102 = constant 102 : index
  %copy = affine_apply #map2(%zero)

// CHECK-NEXT: %0 = addi %c0, %c0 : index
// CHECK-NEXT: %c1 = constant 1 : index
// CHECK-NEXT: %1 = addi %0, %c1 : index
  %one = affine_apply #map3(%symbZero)[%zero]

// CHECK-NEXT: %c103 = constant 103 : index
// CHECK-NEXT: %c104 = constant 104 : index
// CHECK-NEXT: %c105 = constant 105 : index
// CHECK-NEXT: %c106 = constant 106 : index
// CHECK-NEXT: %c107 = constant 107 : index
// CHECK-NEXT: %c108 = constant 108 : index
// CHECK-NEXT: %c109 = constant 109 : index
  %103 = constant 103 : index
  %104 = constant 104 : index
  %105 = constant 105 : index
  %106 = constant 106 : index
  %107 = constant 107 : index
  %108 = constant 108 : index
  %109 = constant 109 : index
// CHECK-NEXT: %c2 = constant 2 : index
// CHECK-NEXT: %2 = muli %c104, %c2 : index
// CHECK-NEXT: %3 = addi %c103, %2 : index
// CHECK-NEXT: %c3 = constant 3 : index
// CHECK-NEXT: %4 = muli %c105, %c3 : index
// CHECK-NEXT: %5 = addi %3, %4 : index
// CHECK-NEXT: %c4 = constant 4 : index
// CHECK-NEXT: %6 = muli %c106, %c4 : index
// CHECK-NEXT: %7 = addi %5, %6 : index
// CHECK-NEXT: %c5 = constant 5 : index
// CHECK-NEXT: %8 = muli %c107, %c5 : index
// CHECK-NEXT: %9 = addi %7, %8 : index
// CHECK-NEXT: %c6 = constant 6 : index
// CHECK-NEXT: %10 = muli %c108, %c6 : index
// CHECK-NEXT: %11 = addi %9, %10 : index
// CHECK-NEXT: %c7 = constant 7 : index
// CHECK-NEXT: %12 = muli %c109, %c7 : index
// CHECK-NEXT: %13 = addi %11, %12 : index
  %four = affine_apply #map4(%103,%104,%105,%106)[%107,%108,%109]
  return
}

// CHECK-LABEL: cfgfunc @multiresult_affine_apply()
cfgfunc @multiresult_affine_apply() {
// CHECK: ^bb0
^bb0:
// CHECK-NEXT: %c1 = constant 1 : index
// CHECK-NEXT: %0 = addi %c1, %c1 : index
// CHECK-NEXT: %1 = addi %0, %c1 : index
  %one = constant 1 : index
  %tuple = affine_apply #map5 (%one, %one, %one)
  %three = affine_apply #map6 (%tuple#0, %tuple#1, %tuple#2)
  return
}

// CHECK-LABEL: cfgfunc @args_ret_affine_apply
cfgfunc @args_ret_affine_apply(index, index) -> (index, index) {
// CHECK: ^bb0(%arg0: index, %arg1: index):
^bb0(%0 : index, %1 : index):
// CHECK-NEXT: return %arg0, %arg1 : index, index
  %00 = affine_apply #map2 (%0)
  %11 = affine_apply #map1 ()[%1]
  return %00, %11 : index, index
}
