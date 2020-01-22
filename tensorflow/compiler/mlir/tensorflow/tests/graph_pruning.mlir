// RUN: tf-opt %s -tf-executor-graph-pruning | FileCheck %s --dump-input=fail

// Two islands chained by data-flow contributing to the graph return are
// preserved.
// CHECK-LABEL: func @chained_islands(
func @chained_islands(%arg0 : i32) -> i32 {
// CHECK: island
// CHECK: island
  %0 = tf_executor.graph {
    %1:2 = tf_executor.island {
      tf_executor.yield %arg0 : i32
    }
    %2:2 = tf_executor.island {
      tf_executor.yield %1#0 : i32
    }
    tf_executor.fetch %2#0 : i32
  }
  return %0 : i32
}

// Check that empty islands that don't contribute to the fetch are removed.
// CHECK-LABEL: func @empty_islands(
func @empty_islands() {
// CHECK-NOT: tf_executor.island
  tf_executor.graph {
    %0 = tf_executor.island {
      tf_executor.yield
    }
    %1 = tf_executor.island {
      tf_executor.yield
    }
    tf_executor.fetch
  }
  return
}

// Check that an unused island that doesn't contribute to the fetch is removed.
// CHECK-LABEL: func @dead_island(
func @dead_island(%arg0 : i32) -> i32 {
// CHECK: tf_executor.island
// CHECK-NOT: tf_executor.island
  %0 = tf_executor.graph {
    %1:2 = tf_executor.island {
      %a = "op.A"(%arg0) : (i32) -> i32
      %b = "op.B"(%a) : (i32) -> i32
      tf_executor.yield %b : i32
    }
    %2:2 = tf_executor.island {
      %a = "op.A"(%1#0) : (i32) -> i32
      tf_executor.yield %a : i32
    }
    tf_executor.fetch %1#0 : i32
  }
  return %0 : i32
}


// Check that NextIteration.sink node isn't deleted when the source is still
// used, even though it does not have any result.
// CHECK-LABEL: func @nextiteration_sink_preserved(
func @nextiteration_sink_preserved(%arg0 : i32) -> i32 {
// CHECK: tf_executor.NextIteration.Source
// CHECK: tf_executor.NextIteration.Sink
  %0 = tf_executor.graph {
    %1:3 = tf_executor.NextIteration.Source : i32
    tf_executor.NextIteration.Sink[%1#1] %1#0 : i32
    tf_executor.fetch %1#0 : i32
  }
  return %0 : i32
}

// Check that NextIteration.sink node is deleted when the source does not have
// any user other than the sink.
// CHECK-LABEL: func @nextiteration_deleted(
func @nextiteration_deleted(%arg0 : i32) -> i32 {
// CHECK-NOT: tf_executor.NextIteration.Source
// CHECK-NOT: tf_executor.NextIteration.Sink
  %0 = tf_executor.graph {
    %1:3 = tf_executor.NextIteration.Source : i32
    // intentionally take an output dependency on the source here.
    tf_executor.NextIteration.Sink[%1#1] %1#0 : i32
    tf_executor.fetch %arg0 : i32
  }
  return %0 : i32
}

