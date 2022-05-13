// Performs while(cond) { dst = src; cond = val; }
func.func @while(
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
  func.return
}

// RUN: lhlo-tfrt-opt %s \
// RUN:   -lmhlo-to-tfrt-branch \
// RUN: | FileCheck %s --check-prefix=WHILE

// WHILE-LABEL: func @while_cond(
// WHILE-SAME:    %arg0: memref<i1>,
// WHILE-SAME:    %arg1: memref<i1>,
// WHILE-SAME:    %arg2: memref<8xf32>,
// WHILE-SAME:    %arg3: memref<8xf32>
// WHILE-SAME:  ) -> i1 {
// WHILE-NEXT:    gpu.memcpy  %arg0, %arg1 : memref<i1>, memref<i1>
// WHILE-NEXT:    %[[value:.*]] = memref.load %arg0[] : memref<i1>
// WHILE-NEXT:    return %[[value]] : i1
// WHILE-NEXT:  }

// WHILE-LABEL: func @while_body(
// WHILE-SAME:    %arg0: memref<i1>,
// WHILE-SAME:    %arg1: memref<i1>,
// WHILE-SAME:    %arg2: memref<8xf32>,
// WHILE-SAME:    %arg3: memref<8xf32>
// WHILE-SAME:  ) -> (memref<i1>, memref<i1>, memref<8xf32>, memref<8xf32>, i1)
// WHILE-NEXT:    gpu.memcpy  %arg2, %arg3 : memref<8xf32>, memref<8xf32>
// WHILE-NEXT:    %[[value:.*]] = tfrt.call
// WHILE-SAME:        @while_cond(%arg0, %arg1, %arg2, %arg3)
// WHILE-NEXT:    return %arg0, %arg1, %arg2, %arg3, %[[value]]
// WHILE-NEXT:  }

// WHILE-LABEL: func @while(
// WHILE-SAME:    %arg0: memref<8xf32>,
// WHILE-SAME:    %arg1: memref<8xf32>,
// WHILE-SAME:    %arg2: memref<i1>,
// WHILE-SAME:    %arg3: memref<i1>
// WHILE-SAME:  ) {
// WHILE-NEXT:    %[[value:.*]] = tfrt.call
// WHILE-SAME:        @while_cond(%arg2, %arg3, %arg0, %arg1)
// WHILE-NEXT:    %[[result:.*]]:4 = tfrt.while %[[value]]
// WHILE-SAME:        @while_body(%arg2, %arg3, %arg0, %arg1)
// WHILE-NEXT:    return
// WHILE-NEXT:  }

// RUN: lhlo-tfrt-opt %s \
// RUN:   -lmhlo-to-tfrt-gpu \
// RUN: | FileCheck %s --check-prefix=GPU

// GPU-LABEL: func @while_cond(
// GPU-SAME:    %arg0: !tfrt.chain,
// GPU-SAME:    %arg1: !tfrt_gpu.stream,
// GPU-SAME:    %arg2: !tfrt_gpu.buffer,
// GPU-SAME:    %arg3: !tfrt_gpu.buffer,
// GPU-SAME:    %arg4: !tfrt_gpu.buffer,
// GPU-SAME:    %arg5: !tfrt_gpu.buffer
// GPU-SAME:  ) -> (!tfrt.chain, i1) {
// GPU-NEXT:   %[[ch0:.*]] = tfrt_gpu.mem.copy %arg2, %arg3, %arg1, %arg0
// GPU-NEXT:   %[[ctx:.*]] = tfrt_gpu.stream.get_context %arg1
// GPU-NEXT:   %[[size:.*]] = tfrt.constant.i64 1
// GPU-NEXT:   %[[host:.*]] = tfrt_gpu.mem.allocate_host %[[ctx]], %[[size]], %[[ch0]]
// GPU-NEXT:   %[[ch1:.*]] = tfrt_gpu.mem.copy %[[host]], %arg2, %arg1, %[[ch0]]
// GPU-NEXT:   %[[ch2:.*]] = tfrt_gpu.stream.synchronize %arg1, %[[ch1]]
// GPU-NEXT:   %[[result:.*]] = tfrt_gpu.mem.load %[[host]], %[[ch2]] : i1
// GPU-NEXT:   tfrt.return %[[ch2]], %[[result]] : !tfrt.chain, i1
// GPU-NEXT: }

// GPU-LABEL: func @while_body(
// GPU-SAME:    %arg0: !tfrt.chain,
// GPU-SAME:    %arg1: !tfrt_gpu.stream,
// GPU-SAME:    %arg2: !tfrt_gpu.buffer,
// GPU-SAME:    %arg3: !tfrt_gpu.buffer,
// GPU-SAME:    %arg4: !tfrt_gpu.buffer,
// GPU-SAME:    %arg5: !tfrt_gpu.buffer
// GPU-SAME:  ) -> (
// GPU-SAME:    !tfrt.chain,
// GPU-SAME:    !tfrt_gpu.buffer,
// GPU-SAME:    !tfrt_gpu.buffer,
// GPU-SAME:    !tfrt_gpu.buffer,
// GPU-SAME:    !tfrt_gpu.buffer,
// GPU-SAME:    i1
// GPU-SAME:  ) {
// GPU-NEXT:   %[[ch0:.*]] = tfrt_gpu.mem.copy %arg4, %arg5, %arg1, %arg0
// GPU-NEXT:   %[[results:.*]]:2 = tfrt.call
// GPU-SAME:       @while_cond(%[[ch0]], %arg1, %arg2, %arg3, %arg4, %arg5)
// GPU-NEXT:   tfrt.return
// GPU-SAME:       %[[results]]#0, %arg2, %arg3, %arg4, %arg5, %[[results]]#1
// GPU-NEXT: }

// GPU-LABEL: func @while(
// GPU-SAME:    %arg0: !tfrt.chain,
// GPU-SAME:    %arg1: !tfrt_gpu.stream,
// GPU-SAME:    %arg2: !tfrt_gpu.buffer,
// GPU-SAME:    %arg3: !tfrt_gpu.buffer,
// GPU-SAME:    %arg4: !tfrt_gpu.buffer,
// GPU-SAME:    %arg5: !tfrt_gpu.buffer
// GPU-SAME:  ) -> !tfrt.chain {
// GPU-NEXT:   %[[cond:.*]]:2 = tfrt.call
// GPU-SAME:       @while_cond(%arg0, %arg1, %arg4, %arg5, %arg2, %arg3)
// GPU-NEXT:   %[[while:.*]]:6 = tfrt.while %[[cond]]#1
// GPU-SAME:       @while_body(%[[cond]]#0, %arg1, %arg4, %arg5, %arg2, %arg3)
// GPU-SAME:       parallel_iterations(1)
// GPU-NEXT:   tfrt.return %[[while]]#0
// GPU-NEXT: }

