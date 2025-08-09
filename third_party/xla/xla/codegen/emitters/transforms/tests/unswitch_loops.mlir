// RUN: emitters_opt %s -split-input-file -xla-unswitch-loops | FileCheck %s

module {
  func.func @unswitchable(
     %arg0: tensor<2xf32>,
     %arg1: index
  ) -> tensor<2xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cst3 = arith.constant 3.0 : f32
    %cst4 = arith.constant 4.0 : f32
    %cond = arith.cmpi sle, %arg1, %c1 : index

    %for = scf.for %i = %c0 to %c2 step %c1 iter_args(%arg2 = %arg0) -> tensor<2xf32> {
      %result = scf.if %cond -> tensor<2xf32> {
        %set_3 = tensor.insert %cst3 into %arg2[%i] : tensor<2xf32>
        scf.yield %set_3 : tensor<2xf32>
      } else {
        %set_4 = tensor.insert %cst4 into %arg2[%i] : tensor<2xf32>
        scf.yield %set_4 : tensor<2xf32>
      }
      scf.yield %result : tensor<2xf32>
    }

    func.return %for : tensor<2xf32>
  }
}

// CHECK:      @unswitchable(%[[ARG0:.*]]: tensor<2xf32>, %[[ARG1:.*]]: index)
// CHECK:        %[[CST3:.*]] = arith.constant 3.0
// CHECK:        %[[CST4:.*]] = arith.constant 4.0
// CHECK:        %[[COND:.*]] = arith.cmpi sle, %[[ARG1]]
// CHECK-NEXT:   scf.if %[[COND]]
// CHECK-NEXT:     scf.for
// CHECK-NEXT:       tensor.insert %[[CST3]]
// CHECK-NEXT:       scf.yield
// CHECK-NEXT:     }
// CHECK:        } else {
// CHECK-NEXT:     scf.for
// CHECK-NEXT:       tensor.insert %[[CST4]]
// CHECK-NEXT:       scf.yield
// CHECK-NEXT:   }


// -----

// COM: Check that a ddeply nested loop still works.

#indexing_map = #xla.indexing_map<"(d0, d1, d2, d3, d4, d5, d6, bl_x, d8) -> (((bl_x * 3 + d8) mod 4) * 4320 + (((bl_x * 3 + d8) floordiv 4) mod 2) * 2160 + ((bl_x * 3 + d8) floordiv 8) * 720 + d1 * 2 + d2 * 6 + d3 * 17280 + d4 * 18 + d5 * 72 + d6 * 34560 + d0), domain: d0 in [0, 1], d1 in [0, 2], d2 in [0, 2], d3 in [0, 1], d4 in [0, 3], d5 in [0, 9], d6 in [0, 127], bl_x in [0, 7], d8 in [0, 2]">
#indexing_map1 = #xla.indexing_map<"(d0, d1, d2, d3, d4, d5, d6, bl_x, d8) -> (d0 * 92160 + d1 * 30720 + d2 * 10240 + d3 * 5120 + d4 * 1280 + d5 * 128 + bl_x * 552960 + d8 * 184320 + d6), domain: d0 in [0, 1], d1 in [0, 2], d2 in [0, 2], d3 in [0, 1], d4 in [0, 3], d5 in [0, 9], d6 in [0, 127], bl_x in [0, 7], d8 in [0, 2]">
module {
  func.func @deeply_nested_loop(%arg0: tensor<4423680xbf16>, %arg1: tensor<4423680xf32>) -> tensor<4423680xf32> {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c10 = arith.constant 10 : index
    %c128 = arith.constant 128 : index
    %c7 = arith.constant 7 : index
    %0 = xla.workgroup_id  x {xla.range = [0 : index, 7 : index]}
    %1 = arith.cmpi sge, %0, %c0 : index
    %2 = arith.cmpi sle, %0, %c7 : index
    %3 = arith.andi %1, %2 : i1
    %5 = scf.for %arg4 = %c0 to %c3 step %c1 iter_args(%arg5 = %arg1) -> (tensor<4423680xf32>) {
      %6 = scf.for %arg6 = %c0 to %c2 step %c1 iter_args(%arg7 = %arg5) -> (tensor<4423680xf32>) {
        %7 = scf.for %arg8 = %c0 to %c3 step %c1 iter_args(%arg9 = %arg7) -> (tensor<4423680xf32>) {
          %8 = scf.for %arg10 = %c0 to %c3 step %c1 iter_args(%arg11 = %arg9) -> (tensor<4423680xf32>) {
            %9 = scf.for %arg12 = %c0 to %c2 step %c1 iter_args(%arg13 = %arg11) -> (tensor<4423680xf32>) {
              %10 = scf.for %arg14 = %c0 to %c4 step %c1 iter_args(%arg15 = %arg13) -> (tensor<4423680xf32>) {
                %11 = scf.for %arg16 = %c0 to %c10 step %c1 iter_args(%arg17 = %arg15) -> (tensor<4423680xf32>) {
                  %12 = scf.for %arg18 = %c0 to %c128 step %c1 iter_args(%arg19 = %arg17) -> (tensor<4423680xf32>) {
                    %13 = scf.if %3 -> (tensor<4423680xf32>) {
                      %14 = xla.apply_indexing #indexing_map(%arg6, %arg8, %arg10, %arg12, %arg14, %arg16, %arg18, %0, %arg4)
                      %extracted = tensor.extract %arg0[%14] : tensor<4423680xbf16>
                      %15 = arith.extf %extracted : bf16 to f32
                      %16 = xla.apply_indexing #indexing_map1(%arg6, %arg8, %arg10, %arg12, %arg14, %arg16, %arg18, %0, %arg4)
                      %inserted = tensor.insert %15 into %arg19[%16] : tensor<4423680xf32>
                      scf.yield %inserted : tensor<4423680xf32>
                    } else {
                      scf.yield %arg19 : tensor<4423680xf32>
                    }
                    scf.yield %13 : tensor<4423680xf32>
                  }
                  scf.yield %12 : tensor<4423680xf32>
                }
                scf.yield %11 : tensor<4423680xf32>
              }
              scf.yield %10 : tensor<4423680xf32>
            }
            scf.yield %9 : tensor<4423680xf32>
          }
          scf.yield %8 : tensor<4423680xf32>
        }
        scf.yield %7 : tensor<4423680xf32>
      }
      scf.yield %6 : tensor<4423680xf32>
    }
    return %5 : tensor<4423680xf32>
  }
}

// CHECK: scf.if
// CHECK-NEXT: scf.for
// CHECK-NEXT: scf.for
// CHECK-NEXT: scf.for
// CHECK-NEXT: scf.for
// CHECK-NEXT: scf.for
// CHECK-NEXT: scf.for
// CHECK-NEXT: scf.for
// CHECK-NEXT: scf.for
