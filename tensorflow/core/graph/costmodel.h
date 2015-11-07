#ifndef TENSORFLOW_GRAPH_COSTMODEL_H_
#define TENSORFLOW_GRAPH_COSTMODEL_H_

#include <unordered_map>
#include <vector>

#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {
typedef std::unordered_map<string, int32> NodeNameToCostIdMap;

class StepStats;

// CostModel keeps track of the following runtime statistics for nodes
// of a single Graph:
//    * The total number of times a node has executed.
//    * The accumulated execution time (in microseconds) of a node.
//    * The accumulated size (in bytes) of each node's output.
//
// This class is NOT thread-safe.
class CostModel {
 public:
  // If "global" is true, maintains costs based on Node::cost_id, otherwise
  // maintains costs based on Node::id.
  explicit CostModel(bool is_global) : is_global_(is_global) {}

  // Assigns min_count_ as a function of the median count for a Node.
  // This value is then used for suppressing the time/size costs of
  // infrequent operations.
  // NOTE(tucker): Maybe this should move to a subclass of CostModel.
  void SuppressInfrequent();

  bool is_global() const { return is_global_; }

  // Initializes cost model for 'g'.
  void InitFromGraph(const Graph& g);

  // Merges costs from cm.
  // REQUIRES: is_global_ is true for this and for "cm"
  void MergeFromGlobal(const CostModel& cm);

  // Merges costs from "cm", which has been computed relative to "g".
  // REQUIRES: is_global_ is true for this, and false for "cm".
  void MergeFromLocal(const Graph& g, const CostModel& cm);

  void MergeFromStats(const NodeNameToCostIdMap& map, const StepStats& ss);

  // Sets the number of outputs of "node".
  void SetNumOutputs(const Node* node, int num_outputs);

  // Records that "node" has executed "num_count" more times.
  void RecordCount(const Node* node, int num_count);

  // Returns how many times "node" has been executed.
  int32 TotalCount(const Node* node) const;

  // Records that "output_slot" of "node" has produced tensors of
  // aggregated "bytes".
  void RecordSize(const Node* node, int output_slot, Bytes bytes);

  // Returns total bytes of tensors produced by "node"s output slot.
  Bytes TotalBytes(const Node* node, int output_slot) const;

  // Returns a prediction for the size of the tensor at the
  // output_slot produced by one execution of "node".
  Bytes SizeEstimate(const Node* node, int output_slot) const;

  // Records that Executions of "node" have taken "time" microseconds.
  void RecordTime(const Node* node, Microseconds time);

  // Returns the total execution time for "node".
  Microseconds TotalTime(const Node* node) const;

  // Returns a prediction for one execution of "node".
  Microseconds TimeEstimate(const Node* node) const;

  // Check that an estimate is available for every OP node in graph.
  void CheckInitialized(const Graph& graph) const;

  // Helper routines to encapsulate static estimatation heuristics

  // Compute an estimate of the time to copy "b" bytes over the network,
  // given a fixed cost of "network_latency_millis" milliseconds and
  // an estimated bandwidth of "estimated_gbps" gigabits per second (note that
  // this value is in gigabits, not gigabytes).
  static Microseconds CopyTimeEstimate(Bytes b, double network_latency_millis,
                                       double estimated_gbps);
  static Microseconds ComputationTimeEstimate(int64 mathops);

  // Write the contents of the CostModel to the INFO log.
  void WriteToLog();

 private:
  const bool is_global_;
  inline int Id(const Node* n) const {
    if (is_global_) {
      return n->cost_id();
    } else {
      return n->id();
    }
  }
  // Resizes vectors so that they are large enough for "id".
  void Ensure(int id);

  // Nodes and Edges whose count is < this value
  // get type/byte estimates of 0.
  int32 min_count_ = 0;

  // Number of times each Node has been executed.
  std::vector<int32> count_;
  // Cumulative execution time.
  std::vector<Microseconds> time_;
  // Cumulative Bytes output on each channel.
  std::vector<gtl::InlinedVector<Bytes, 2> > slot_bytes_;

  TF_DISALLOW_COPY_AND_ASSIGN(CostModel);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_GRAPH_COSTMODEL_H_
