// RUN: mlir-opt %s -affine-vectorizer-test -normalize-maps |  FileCheck %s

// CHECK-DAG: #[[ZERO:[a-zA-Z0-9]+]] = () -> (0)
// CHECK-DAG: #[[ID1:[a-zA-Z0-9]+]] = (d0) -> (d0)
// CHECK-DAG: #[[D0TIMES2:[a-zA-Z0-9]+]] = (d0) -> (d0 * 2)
// CHECK-DAG: #[[D0PLUSD1:[a-zA-Z0-9]+]] = (d0, d1) -> (d0 + d1)
// CHECK-DAG: #[[MINSD0PLUSD1:[a-zA-Z0-9]+]] = (d0, d1) -> (-d0 + d1)
// CHECK-DAG: #[[D0MINUSD1:[a-zA-Z0-9]+]] = (d0, d1) -> (d0 - d1)

// CHECK-LABEL: func @simple()
func @simple() {
  affine.for %i0 = 0 to 7 {
    %0 = affine.apply (d0) -> (d0) (%i0)
    %1 = affine.apply (d0) -> (d0) (%0)
    %2 = affine.apply (d0, d1) -> (d0 + d1) (%0, %0)
    %3 = affine.apply (d0, d1) -> (d0 - d1) (%0, %0)
  }
  // CHECK-NEXT: affine.for %{{.*}} = 0 to 7
  // CHECK-NEXT:   {{.*}} affine.apply #[[ID1]](%{{.*}})
  // CHECK-NEXT:   {{.*}} affine.apply #[[D0TIMES2]](%{{.*}})
  // CHECK-NEXT:   {{.*}} affine.apply #[[ZERO]]()

  affine.for %i1 = 0 to 7 {
    affine.for %i2 = 0 to 42 {
      %20 = affine.apply (d0, d1) -> (d1) (%i1, %i2)
      %21 = affine.apply (d0, d1) -> (d0) (%i1, %i2)
      %22 = affine.apply (d0, d1) -> (d0 + d1) (%20, %21)
      %23 = affine.apply (d0, d1) -> (d0 - d1) (%20, %21)
      %24 = affine.apply (d0, d1) -> (-d0 + d1) (%20, %21)
    }
  }
  //      CHECK: affine.for %{{.*}} = 0 to 7
  // CHECK-NEXT:   affine.for %{{.*}} = 0 to 42
  // CHECK-NEXT:     {{.*}} affine.apply #[[D0PLUSD1]](%{{.*}}, %{{.*}})
  // CHECK-NEXT:     {{.*}} affine.apply #[[MINSD0PLUSD1]](%{{.*}}, %{{.*}})
  // CHECK-NEXT:     {{.*}} affine.apply #[[D0MINUSD1]](%{{.*}}, %{{.*}})

  affine.for %i3 = 0 to 16 {
    affine.for %i4 = 0 to 47 step 2 {
      affine.for %i5 = 0 to 78 step 16 {
        %50 = affine.apply (d0) -> (d0) (%i3)
        %51 = affine.apply (d0) -> (d0) (%i4)
        %52 = affine.apply (d0) -> (d0) (%i5)
        %53 = affine.apply (d0, d1, d2) -> (d0) (%50, %51, %52)
        %54 = affine.apply (d0, d1, d2) -> (d1) (%50, %51, %52)
        %55 = affine.apply (d0, d1, d2) -> (d2) (%50, %51, %52)
      }
    }
  }
  // CHECK:      affine.for %{{.*}} = 0 to 16
  // CHECK-NEXT:   affine.for %{{.*}} = 0 to 47 step 2
  // CHECK-NEXT:     affine.for %{{.*}} = 0 to 78 step 16
  // CHECK-NEXT:       {{.*}} affine.apply #[[ID1]](%{{.*}})
  // CHECK-NEXT:       {{.*}} affine.apply #[[ID1]](%{{.*}})
  // CHECK-NEXT:       {{.*}} affine.apply #[[ID1]](%{{.*}})

  return
}
