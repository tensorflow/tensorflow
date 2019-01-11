// RUN: mlir-opt %s -vectorizer-test -normalize-maps |  FileCheck %s

// CHECK-DAG: #[[ZERO:[a-zA-Z0-9]+]] = () -> (0)
// CHECK-DAG: #[[ID1:[a-zA-Z0-9]+]] = (d0) -> (d0)
// CHECK-DAG: #[[D0TIMES2:[a-zA-Z0-9]+]] = (d0) -> (d0 * 2)
// CHECK-DAG: #[[D0PLUSD1:[a-zA-Z0-9]+]] = (d0, d1) -> (d0 + d1)
// CHECK-DAG: #[[MINSD0PLUSD1:[a-zA-Z0-9]+]] = (d0, d1) -> (d0 * -1 + d1)
// CHECK-DAG: #[[D0MINUSD1:[a-zA-Z0-9]+]] = (d0, d1) -> (d0 - d1)

// CHECK-LABEL: func @simple()
func @simple() {
  for %i0 = 0 to 7 {
    %0 = affine_apply (d0) -> (d0) (%i0)
    %1 = affine_apply (d0) -> (d0) (%0)
    %2 = affine_apply (d0, d1) -> (d0 + d1) (%0, %0)
    %3 = affine_apply (d0, d1) -> (d0 - d1) (%0, %0)
  }
  // CHECK-NEXT: for %i0 = 0 to 7
  // CHECK-NEXT:   {{.*}} affine_apply #[[ID1]](%i0)
  // CHECK-NEXT:   {{.*}} affine_apply #[[D0TIMES2]](%i0)
  // CHECK-NEXT:   {{.*}} affine_apply #[[ZERO]]()

  for %i1 = 0 to 7 {
    for %i2 = 0 to 42 {
      %20 = affine_apply (d0, d1) -> (d1) (%i1, %i2)
      %21 = affine_apply (d0, d1) -> (d0) (%i1, %i2)
      %22 = affine_apply (d0, d1) -> (d0 + d1) (%20, %21)
      %23 = affine_apply (d0, d1) -> (d0 - d1) (%20, %21)
      %24 = affine_apply (d0, d1) -> (-d0 + d1) (%20, %21)
    }
  }
  // CHECK:      for %i1 = 0 to 7
  // CHECK-NEXT:   for %i2 = 0 to 42
  // CHECK-NEXT:     {{.*}} affine_apply #[[D0PLUSD1]](%i1, %i2)
  // CHECK-NEXT:     {{.*}} affine_apply #[[MINSD0PLUSD1]](%i1, %i2)
  // CHECK-NEXT:     {{.*}} affine_apply #[[D0MINUSD1]](%i1, %i2)

  for %i3 = 0 to 16 {
    for %i4 = 0 to 47 step 2 {
      for %i5 = 0 to 78 step 16 {
        %50 = affine_apply (d0) -> (d0) (%i3)
        %51 = affine_apply (d0) -> (d0) (%i4)
        %52 = affine_apply (d0) -> (d0) (%i5)
        %53 = affine_apply (d0, d1, d2) -> (d0) (%50, %51, %52)
        %54 = affine_apply (d0, d1, d2) -> (d1) (%50, %51, %52)
        %55 = affine_apply (d0, d1, d2) -> (d2) (%50, %51, %52)
      }
    }
  }
  // CHECK:      for %i3 = 0 to 16
  // CHECK-NEXT:   for %i4 = 0 to 47 step 2
  // CHECK-NEXT:     for %i5 = 0 to 78 step 16
  // CHECK-NEXT:       {{.*}} affine_apply #[[ID1]](%i3)
  // CHECK-NEXT:       {{.*}} affine_apply #[[ID1]](%i4)
  // CHECK-NEXT:       {{.*}} affine_apply #[[ID1]](%i5)

  return
}
