// RUN: lhlo-tfrt-opt %s \
// RUN:   -lmhlo-to-tfrt-while \
// RUN: | FileCheck %s

// Performs while(cond) { dst = src; cond = val; }
func @while(
  %dst : memref<8xf32>,
  %src : memref<8xf32>,
  %cond : memref<i1>,
  %val : memref<i1>
) {
  "lmhlo.while"(%cond) ({
    gpu.memcpy %cond, %val : memref<i1>, memref<i1>
    "lmhlo.terminator"() : () -> ()
  }, {
    gpu.memcpy %dst, %src : memref<8xf32>, memref<8xf32>
    "lmhlo.terminator"() : () -> ()
  }) : (memref<i1>) -> ()
  return
}

// CHECK-LABEL: func @while_cond(
// CHECK-SAME:    %arg0: memref<i1>,
// CHECK-SAME:    %arg1: memref<i1>,
// CHECK-SAME:    %arg2: memref<8xf32>,
// CHECK-SAME:    %arg3: memref<8xf32>
// CHECK-SAME:  ) -> i1 {
// CHECK-NEXT:    gpu.memcpy  %arg0, %arg1 : memref<i1>, memref<i1>
// CHECK-NEXT:    %[[value:.*]] = memref.load %arg0[] : memref<i1>
// CHECK-NEXT:    return %[[value]] : i1
// CHECK-NEXT:  }

// CHECK-LABEL: func @while_body(
// CHECK-SAME:    %arg0: memref<i1>,
// CHECK-SAME:    %arg1: memref<i1>,
// CHECK-SAME:    %arg2: memref<8xf32>,
// CHECK-SAME:    %arg3: memref<8xf32>
// CHECK-SAME:  ) -> (memref<i1>, memref<i1>, memref<8xf32>, memref<8xf32>, i1)
// CHECK-NEXT:    gpu.memcpy  %arg2, %arg3 : memref<8xf32>, memref<8xf32>
// CHECK-NEXT:    %[[value:.*]] = tfrt.call
// CHECK-SAME:        @while_cond(%arg0, %arg1, %arg2, %arg3)
// CHECK-NEXT:    return %arg0, %arg1, %arg2, %arg3, %[[value]]
// CHECK-NEXT:  }

// CHECK-LABEL: func @while(
// CHECK-SAME:    %arg0: memref<8xf32>,
// CHECK-SAME:    %arg1: memref<8xf32>,
// CHECK-SAME:    %arg2: memref<i1>,
// CHECK-SAME:    %arg3: memref<i1>
// CHECK-SAME:  ) {
// CHECK-NEXT:    %[[value:.*]] = tfrt.call
// CHECK-SAME:        @while_cond(%arg2, %arg3, %arg0, %arg1)
// CHECK-NEXT:    %[[result:.*]]:4 = tfrt.while %[[value]]
// CHECK-SAME:        @while_body(%arg2, %arg3, %arg0, %arg1)
// CHECK-NEXT:    return
// CHECK-NEXT:  }
