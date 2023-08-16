// RUN: export MSAN_OPTIONS=intercept_strpbrk=0
// RUN: xla-gpu2-opt %s --xla-gpu2-finalize-graph-dispatches                   \
// RUN:                 --split-input-file                                     \
// RUN:   | FileCheck %s

func.func @graph_dispatch(%ctx: !xla_gpu.execution_context) {
  xla_gpu.graph.dispatch graph(%g: !xla_gpu.graph) {
    func.call @sink(%ctx, %g)
      : (!xla_gpu.execution_context, !xla_gpu.graph) -> ()
  }
  return
}

func.func private @sink(%ctx: !xla_gpu.execution_context, %g: !xla_gpu.graph)

// CHECK-LABEL: func @graph_dispatch(
// CHECK:            %[[CTX:.*]]: !xla_gpu.execution_context
// CHECK: ) {
// CHECK:   %[[G:.*]] = call @__xla_gpu.graph.create(%[[CTX]])
// CHECK:   call @xla_gpu.graph.execute(%[[CTX]], %[[G]])
// CHECK: }

// CHECK: func private @__xla_gpu.graph.create(
// CHECK:   %[[CTX_ARG:.*]]: !xla_gpu.execution_context
// CHECK: ) -> !xla_gpu.graph {
// CHECK:   %[[GG:.*]] = call @xla_gpu.graph.create(%[[CTX_ARG]])
// CHECK:   call @sink(%[[CTX_ARG]], %[[GG]])
// CHECK:   return %[[GG]] : !xla_gpu.graph
// CHECK: }