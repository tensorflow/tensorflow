
```mlir {.mlir}
func @matmul(%A: memref<?x?xf32, offset: ?, strides: [?, 1]>,
             %B: memref<?x?xf32, offset: ?, strides: [?, 1]>,
             %C: memref<?x?xf32, offset: ?, strides: [?, 1]>) {
  "s.matmul"(%A, %B, %C) : memref<?x?xf32, offset: ?, strides: [?, 1]>,
                              memref<?x?xf32, offset: ?, strides: [?, 1]>,
                              memref<?x?xf32, offset: ?, strides: [?, 1]>
  return
}
```

```mlir {.mlir}
func @matmul(%A: memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>, %B: memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>, %C: memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>) {
  ...
  loop.for %i = %c0 to %6 step %c2000 {
    loop.for %k = %c0_0 to %7 step %c3000 {
      loop.for %k = %c0_1 to %8 step %c4000 {
        %9 = affine.apply (d0) -> (d0 + 2000)(%i)
        ...
        %16 = "s.subview" %A[%12, %13, %c1, %14, %15, %c1] : memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
        ...
        %21 = "s.subview" %B[%17, %18, %c1, %19, %20, %c1] : memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
        ...
        %26 = "s.subview" %C[%22, %23, %c1, %24, %25, %c1] : memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
        "s.matmul"(%16, %21, %26) {__internal_linalg_transform__ = "L3"} : memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>, memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>, memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
      }
    }
  }
}
```

```mlir {.mlir}
func @matmul(%A: memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>, %B: memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>, %C: memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>) {
  ...
  loop.for %i = %c0 to %0 step %c2000 {
    loop.for %j = %c0 to %2 step %c3000 {
      loop.for %k = %c0 to %1 step %c4000 {
        %3 = affine.apply (d0) -> (d0)(%i)
        ...
        %7 = "s.subview" %A[%3, %4, %c1, %5, %6, %c1] : memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
        ...
        %12 = "s.subview" %B[%8, %9, %c1, %10, %11, %c1] : memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
        ...
        %17 = "s.subview" %C[%13, %14, %c1, %15, %16, %c1] : memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
        ...
        loop.for %ii = %c0_0 to %24 step %c200 {
          loop.for %jj = %c0_1 to %25 step %c300 {
            loop.for %kk = %c0_2 to %26 step %c400 {
              %27 = affine.apply (d0) -> (d0 + 200)(%ii)
              ...
              %34 = "s.subview" %7[%30, %31, %c1, %32, %33, %c1] : memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
              ...
              %39 = "s.subview" %12[%35, %36, %c1, %37, %38, %c1] : memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
              ...
              %44 = "s.subview" %17[%40, %41, %c1, %42, %43, %c1] : memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
              "s.matmul"(%34, %39, %44) {__internal_linalg_transform__ = "L2"} : memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>, memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>, memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
            }
          }
        }
      }
    }
  }
  return
}
```

```mlir {.mlir}
func @matmul(%A: memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>, %B: memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>, %C: memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>) {
  ...
  loop.for %i = %c0 to %0 step %c2000 {
    loop.for %j = %c0 to %2 step %c3000 {
      loop.for %k = %c0 to %1 step %c4000 {
        %3 = affine.apply (d0) -> (d0 + 2000)(%i)
        ...
        %5 = "s.subview" %A[%D, %3, %c1, %arg5, %4, %c1] : memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
        ...
        %8 = "s.subview" %B[%arg5, %6, %c1, %E, %7, %c1] : memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
        ...
        %11 = "s.subview" %C[%D, %9, %c1, %E, %10, %c1] : memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
        ...
        loop.for %arg6 = %c0 to %12 step %c200 {
          loop.for %arg7 = %c0 to %14 step %c300 {
            loop.for %arg8 = %c0 to %13 step %c400 {
              %15 = affine.apply (d0) -> (d0)(%arg6)
              ...
              %19 = "s.subview" %5[%15, %16, %c1, %17, %18, %c1] : memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
              ...
              %24 = "s.subview" %8[%20, %21, %c1, %22, %23, %c1] : memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
              ...
              %29 = "s.subview" %11[%25, %26, %c1, %27, %28, %c1] : memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
              ...
              loop.for %arg9 = %c0_0 to %36 step %c20 {
                loop.for %B0 = %c0_1 to %37 step %c30 {
                  loop.for %B1 = %c0_2 to %38 step %c40 {
                    %39 = affine.apply (d0) -> (d0 + 20)(%arg9)
                    ...
                    %46 = "s.subview" %19[%42, %43, %c1, %44, %45, %c1] : memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
                    ...
                    %51 = "s.subview" %24[%47, %48, %c1, %49, %50, %c1] : memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
                    ...
                    %56 = "s.subview" %29[%52, %53, %c1, %54, %55, %c1] : memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
                    "s.matmul"(%46, %51, %56) {__internal_linalg_transform__ = "L1"} : memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>, memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>, memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return
}
```

```mlir {.mlir}
func @fusion_test(%A: memref<?x?xf32, offset: ?, strides: [?, 1]>,
                  %B: memref<?x?xf32, offset: ?, strides: [?, 1]>,
                  %C: memref<?x?xf32, offset: ?, strides: [?, 1]>,
                  %D: memref<?x?xf32, offset: ?, strides: [?, 1]>,
                  %E: memref<?x?xf32, offset: ?, strides: [?, 1]>) {
  // This will not be fused as it would violate dependencies. It will get
  // tiled for all levels of the memory hierarchy.
  "s.matmul"(%A, %A, %C) : memref<?x?xf32, offset: ?, strides: [?, 1]>,
                              memref<?x?xf32, offset: ?, strides: [?, 1]>,
                              memref<?x?xf32, offset: ?, strides: [?, 1]>

  // This will be fused.
  "s.matmul"(%A, %B, %C) : memref<?x?xf32, offset: ?, strides: [?, 1]>,
                              memref<?x?xf32, offset: ?, strides: [?, 1]>,
                              memref<?x?xf32, offset: ?, strides: [?, 1]>

  // This will  not be fused or transformed at all since there are no patterns
  // on it. However it will be reordered because there are no dependencies.
  "s.generic" #some_generic_trait %A, %D {
    ^bb(%a: f32, %b: f32) :
      "s.yield" %a : f32
  } : memref<?x?xf32, offset: ?, strides: [?, 1]>,
      memref<?x?xf32, offset: ?, strides: [?, 1]>

  "s.matmul"(%C, %D, %E) : memref<?x?xf32, offset: ?, strides: [?, 1]>,
                              memref<?x?xf32, offset: ?, strides: [?, 1]>,
                              memref<?x?xf32, offset: ?, strides: [?, 1]>

  return
}
```

```mlir {.mlir}
func @fusion_test(%A: memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>, %B: memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>, %C: memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>, %D: memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>, %E: memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>) {
  ...
  "s.matmul"(%A, %A, %C) : memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>, memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>, memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
  "s.generic" #some_generic_trait %A, %D {
  ^bb0(%arg5: f32, %arg6: f32): // no predecessors
    "s.yield" %arg5 : f32
  }: memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>, memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
  ...
  loop.for %arg5 = %c0_0 to %6 step %c100 {
    loop.for %arg6 = %c0_1 to %7 step %c150 {
      %9 = affine.apply (d0) -> (d0 + 100)(%arg5)
      ...
      %14 = "s.subview" %C[%11, %12, %c1, %c0_3, %13, %c1] : memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
      ...
      %18 = "s.subview" %D[%c0_5, %15, %c1, %16, %17, %c1] : memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
      ...
      %23 = "s.subview" %E[%19, %20, %c1, %21, %22, %c1] : memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
      ...
      %25 = "s.subview" %A[%11, %12, %c1, %c0_10, %24, %c11] : memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
      %26 = "s.subview" %B[%c0_10, %24, %c11, %c0_3, %13, %c1] : memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
      %27 = "s.subview" %C[%11, %12, %c1, %c0_3, %13, %c1] : memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
      "s.matmul"(%25, %26, %27) {__internal_linalg_transform__ = "L1"} : memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>, memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>, memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
      "s.matmul"(%14, %18, %23) {__internal_linalg_transform__ = "L1"} : memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>, memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>, memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
    }
  }
  return
}
```


```mlir {.mlir}
func @fusion_test(%A: memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>, %B: memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>, %C: memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>, %D: memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>, %E: memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>) {
  ...
  loop.for %arg5 = %c0_0 to %6 step %c2000 {
    loop.for %arg6 = %c0_1 to %7 step %c3000 {
      loop.for %arg7 = %c0_2 to %8 step %c4000 {
        %11 = affine.apply (d0) -> (d0 + 2000)(%arg5)
        ...
        %18 = "s.subview" %A[%14, %15, %c1, %16, %17, %c1] : memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
        ...
        %23 = "s.subview" %A[%19, %20, %c1, %21, %22, %c1] : memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
        ...
        %28 = "s.subview" %C[%24, %25, %c1, %26, %27, %c1] : memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
        "s.matmul"(%18, %23, %28) {__internal_linalg_transform__ = "L3"} : memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>, memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>, memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
      }
    }
  }
  ...
  "s.generic" #some_generic_trait %A, %D {
  ^bb0(%arg5: f32, %arg6: f32): // no predecessors
    "s.yield" %arg5 : f32
  }: memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>, memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
  ...
  loop.for %arg5 = %c0 to %9 step %c100 {
    loop.for %arg6 = %c0 to %10 step %c150 {
      %11 = affine.apply (d0) -> (d0)(%arg5)
      ...
      %14 = "s.subview" %C[%11, %12, %c1, %c0, %13, %c1] : memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
      ...
      %18 = "s.subview" %D[%c0, %15, %c1, %16, %17, %c1] : memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
      ...
      %23 = "s.subview" %E[%19, %20, %c1, %21, %22, %c1] : memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
      ...
      %25 = "s.subview" %A[%11, %12, %c1, %c0, %24, %c1] : memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
      %26 = "s.subview" %B[%c0, %24, %c1, %c0, %13, %c1] : memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
      %27 = "s.subview" %C[%11, %12, %c1, %c0, %13, %c1] : memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
      "s.matmul"(%25, %26, %27) {__internal_linalg_transform__ = "L1"} : memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>, memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>, memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
      "s.matmul"(%14, %18, %23) {__internal_linalg_transform__ = "L1"} : memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>, memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>, memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
    }
  }
  return
}
```


```mlir {.mlir}
func @fusion_test(%A: memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>, %B: memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>, %C: memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>, %D: memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>, %E: memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>) {
  ...
  loop.for %arg5 = %c0 to %0 step %c2000 {
    loop.for %arg6 = %c0 to %2 step %c3000 {
      loop.for %arg7 = %c0 to %1 step %c4000 {
        %5 = affine.apply (d0) -> (d0)(%arg5)
        ...
        %9 = "s.subview" %A[%5, %6, %c1, %7, %8, %c1] : memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
        ...
        %14 = "s.subview" %A[%10, %11, %c1, %12, %13, %c1] : memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
        ...
        %19 = "s.subview" %C[%15, %16, %c1, %17, %18, %c1] : memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
        "s.matmul"(%9, %14, %19) {__internal_linalg_transform__ = "L3"} : memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>, memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>, memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
      }
    }
  }
  ...
  "s.generic" #some_generic_trait %A, %D {
  ^bb0(%arg5: f32, %arg6: f32): // no predecessors
    "s.yield" %arg5 : f32
  }: memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>, memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
  ...
  loop.for %arg5 = %c0 to %3 step %c100 {
    loop.for %arg6 = %c0 to %4 step %c150 {
      %5 = affine.apply (d0) -> (d0)(%arg5)
      ...
      %8 = "s.subview" %C[%5, %6, %c1, %c0, %7, %c1] : memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
      ...
      %12 = "s.subview" %D[%c0, %9, %c1, %10, %11, %c1] : memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
      ...
      %17 = "s.subview" %E[%13, %14, %c1, %15, %16, %c1] : memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
      %19 = "s.subview" %A[%5, %6, %c1, %c0, %18, %c1] : memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
      %20 = "s.subview" %B[%c0, %18, %c1, %c0, %7, %c1] : memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
      %21 = "s.subview" %C[%5, %6, %c1, %c0, %7, %c1] : memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
      "s.matmul"(%19, %20, %21) {__internal_linalg_transform__ = "L1"} : memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>, memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>, memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
      ...
      loop.for %arg7 = %c0_0 to %28 step %c2 {
        loop.for %arg8 = %c0_1 to %29 step %c3 {
          loop.for %arg9 = %c0_2 to %30 step %c4 {
            %31 = affine.apply (d0) -> (d0 + 2)(%arg7)
            ...
            %38 = "s.subview" %8[%34, %35, %c1, %36, %37, %c1] : memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
            ...
            %43 = "s.subview" %12[%39, %40, %c1, %41, %42, %c1] : memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
            ...
            %48 = "s.subview" %17[%44, %45, %c1, %46, %47, %c1] : memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
            "s.matmul"(%38, %43, %48) {__internal_linalg_transform__ = "REG"} : memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>, memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>, memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
          }
        }
      }
    }
  }
  return
}
```
