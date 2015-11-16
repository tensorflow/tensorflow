#ifndef TENSORFLOW_COMMON_RUNTIME_SIMPLE_PLACER_H_
#define TENSORFLOW_COMMON_RUNTIME_SIMPLE_PLACER_H_

#include <string>
#include <unordered_map>

#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/status.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

// A placement algorithm that assigns the nodes of the given Graph to
// devices the given DeviceSet, respecting the following constraints:
//
// 1. Existing device assignments remain unchanged.
// 2. Requested (partial or complete) device specifications in the
//    are granted.
// 3. Nodes connected by edges of a reference type are colocated on
//    the same device.
// 4. Given nodes "A" and "B", if node "B" has the device specification
//    "@A", nodes "A" and "B" will be colocated on the same device.
//
// The implementation builds a constraint graph with the same set of
// nodes, and edges that represent colocation constraints between
// nodes.  Each connected component in the resulting constraint graph
// is then assigned to a single device.
//
// TODO(mrry): "Soft" constraints, such as "place node 'x' as close as
// possible to node 'y' while respecting the other constraints"?
// TODO(mrry): Create a common interface for this and the other
// placement algorithms so that they may be injected into the graph
// builder.
class SimplePlacer {
 public:
  // A map from graph node names to numerical IDs (in a Graph object).
  typedef std::unordered_map<string, int> NodeNameToIdMap;

  // Creates an instance of the SimplePlacer algorithm for the given
  // Graph "graph" (nodes in which may or may not be assigned) on the
  // given DeviceSet "devices". The "name_to_id_map" maps the names of
  // nodes in "g" to their numerical ID.
  //
  // REQUIRES: for all mappings (k, v) in "name_to_id_map",
  // graph.FindNodeId(v)->name() == k.
  //
  // The "graph", "devices", and "name_to_id_map" pointer arguments
  // are borrowed by this SimplePlacer, and must outlive it.
  SimplePlacer(Graph* graph, const DeviceSet* devices,
               const NodeNameToIdMap* name_to_id_map,
               const SessionOptions* options);

  SimplePlacer(Graph* graph, const DeviceSet* devices,
               const NodeNameToIdMap* name_to_id_map);

  ~SimplePlacer();

  // Assigns each node in this SimplePlacer's graph to a device in its
  // set of devices.
  //
  // This method is not thread-safe.
  // Run() may be invoked at most once.
  Status Run();

 private:
  Status GetNodeByName(const string& name, Node** out_node) const;

  Graph* const graph_;                           // Not owned.
  const DeviceSet* const devices_;               // Not owned.
  const NodeNameToIdMap* const name_to_id_map_;  // Not owned.
  const SessionOptions* options_;                // Not owned.

  TF_DISALLOW_COPY_AND_ASSIGN(SimplePlacer);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMMON_RUNTIME_SIMPLE_PLACER_H_
