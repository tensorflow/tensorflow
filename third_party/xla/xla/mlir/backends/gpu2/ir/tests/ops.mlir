// RUN: xla-gpu2-opt %s -split-input-file | FileCheck %s

func.func @graph_region() {
  xla_gpu.graph.region {
    %0 = arith.constant 0 : index
  }
  return
}

// CHECK-LABEL: func @graph_region()
// CHECK: xla_gpu.graph.region {
// CHECK:   arith.constant 0 : index
// CHECK: }

// -----

func.func private @sink(%arg0: !xla_gpu.graph)

func.func @graph_dispatch() {
  xla_gpu.graph.dispatch graph(%g: !xla_gpu.graph) {
    func.call @sink(%g) : (!xla_gpu.graph) -> ()
  }
  return
}

// CHECK-LABEL: func @graph_dispatch()
// CHECK: xla_gpu.graph.dispatch graph(%[[G:.*]]: !xla_gpu.graph) {
// CHECK:   func.call @sink(%[[G]]) : (!xla_gpu.graph) -> ()
// CHECK: }
