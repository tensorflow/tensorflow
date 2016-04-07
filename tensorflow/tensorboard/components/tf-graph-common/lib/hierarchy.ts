/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
/**
 * Package for the Graph Hierarchy for TensorFlow graph.
 */
module tf.graph.hierarchy {

/**
 * Class used as output for getPredecessors and getSuccessors methods
 */
export interface Edges {
  control: string[];
  regular: string[];
}

export interface Hierarchy {
  root: Metanode;
  templates: {[templateId: string]: string[]};
  /** List of all device names */
  devices: string[];
  /** True if at least one tensor in the graph has shape information */
  hasShapeInfo: boolean;
  /** The maximum size across all meta edges. Used for scaling thickness. */
  maxMetaEdgeSize: number;
  getNodeMap(): {[nodeName: string]: GroupNode|OpNode};
  node(name: string): GroupNode|OpNode;
  setNode(name: string, node: GroupNode|OpNode): void;
  getBridgegraph(nodeName: string): graphlib.Graph<GroupNode|OpNode, Metaedge>;
  getPredecessors(nodeName: string): Edges;
  getSuccessors(nodeName: string): Edges;
  getTopologicalOrdering(nodeName: string): { [childName: string]: number };
  getTemplateIndex(): (string) => number;
}

/**
 * Class for the Graph Hierarchy for TensorFlow graph.
 */
class HierarchyImpl implements Hierarchy {
  root: Metanode;
  templates: {[templateId: string]: string[]};
  private index: {[nodeName: string]: GroupNode|OpNode};
  devices: string[];
  hasShapeInfo = false;
  maxMetaEdgeSize = 1;
  orderings: { [nodeName: string]: { [childName: string]: number } };

  constructor() {
    this.root = createMetanode(ROOT_NAME, {compound: true});
    this.templates = null;
    this.devices = null;
    /**
     * @type {Object} Dictionary object that maps node name to the node
     * (could be op-node, metanode, or series-node)
     */
    this.index = {};
    this.index[ROOT_NAME] = this.root;
    this.orderings = {};
  }

  getNodeMap(): {[nodeName: string]: GroupNode|OpNode} {
    return this.index;
  }

  node(name: string): GroupNode|OpNode {
    return this.index[name];
  }

  setNode(name: string, node: GroupNode|OpNode): void {
    this.index[name] = node;
  }

  /**
   * Given the name of a node in this hierarchy, get its bridgegraph, creating
   * it on the fly if necessary. If the node is not a GroupNode, then this
   * method returns null. If the provided name does not map to a node in the
   * hierarchy, an error will be thrown.
   */
  getBridgegraph(nodeName: string): graphlib.Graph<GroupNode|OpNode, Metaedge> {
    let node = this.index[nodeName];
    if (!node) {
      throw Error("Could not find node in hierarchy: " + nodeName);
    }
    if (!("metagraph" in node)) {
      return null;
    }
    let groupNode = <GroupNode> node;
    if (groupNode.bridgegraph) {
      return groupNode.bridgegraph;
    }
    let bridgegraph = groupNode.bridgegraph =
        createGraph<GroupNode|OpNode, Metaedge>(
            "BRIDGEGRAPH", GraphType.BRIDGE);
    if (!node.parentNode || !("metagraph" in node.parentNode)) {
      return bridgegraph;
    }

    let parentNode = <GroupNode>node.parentNode;
    let parentMetagraph = parentNode.metagraph;
    let parentBridgegraph = this.getBridgegraph(parentNode.name);

    // For each of the parent node's two Metaedge containing graphs, process
    // each Metaedge involving this node.
    _.each([parentMetagraph, parentBridgegraph], parentGraph => {
      _(parentGraph.edges())
        .filter(e => e.v === nodeName || e.w === nodeName)
        .each(parentEdgeObj => {

          let inbound = parentEdgeObj.w === nodeName;
          let parentMetaedge = parentGraph.edge(parentEdgeObj);

          // The parent's Metaedge represents some number of underlying
          // BaseEdges from the original full graph. For each of those, we need
          // to determine which immediate child is involved and make sure
          // there's a Metaedge in the bridgegraph that covers it.
          _.each(parentMetaedge.baseEdgeList, baseEdge => {

            // Based on the direction, figure out which is the descendant node
            // and which is the "other" node (sibling of parent or ancestor).
            let [descendantName, otherName] =
              inbound ?
                [baseEdge.w, parentEdgeObj.v] :
                [baseEdge.v, parentEdgeObj.w];

            // Determine the immediate child containing this descendant node.
            let childName = this.getChildName(nodeName, descendantName);

            // Look for an existing Metaedge in the bridgegraph (or create a
            // new one) that covers the relationship between child and other.
            let bridgeEdgeObj = <graphlib.EdgeObject> {
              v: inbound ? otherName : childName,
              w: inbound ? childName : otherName,
            };
            let bridgeMetaedge = bridgegraph.edge(bridgeEdgeObj);
            if (!bridgeMetaedge) {
              bridgeMetaedge = createMetaedge(bridgeEdgeObj.v, bridgeEdgeObj.w);
              bridgeMetaedge.inbound = inbound;
              bridgegraph.setEdge(bridgeEdgeObj.v, bridgeEdgeObj.w,
                  bridgeMetaedge);
            }

            // Copy the BaseEdge from the parent's Metaedge into this
            // bridgegraph Metaedge.
            bridgeMetaedge.addBaseEdge(baseEdge, this);
          });
        })
        .value(); // force lodash chain execution.
    });

    return bridgegraph;
  }

  /**
   * Utility function for determining the name of the immediate child under a
   * node for a given descendant path. If the descendant corresponds to no
   * immediate child, an error is thrown.
   */
  getChildName(nodeName: string, descendantName: string): string {
    // Walk up the hierarchy from the descendant to find the child.
    let currentNode: Node = this.index[descendantName];
    while (currentNode) {
      if (currentNode.parentNode && currentNode.parentNode.name === nodeName) {
        return currentNode.name;
      }
      currentNode = currentNode.parentNode;
    }
    throw Error("Could not find immediate child for descendant: " +
        descendantName);
  };

  /**
   * Given the name of a node, return the names of its predecessors.
   * For an OpNode, this will contain the targets from the underlying BaseEdges.
   * For a GroupNode, this will contain the targets truncated to siblings of
   * the shared ancestor.
   *
   * For example, consider an original non-control BaseEdge A/B/C->Z/Y/X. Their
   * shared ancestor is the ROOT node. A and Z are the highest siblings. Here
   * are the results of calling getPredecessors():
   *
   *  - getPredecessors("Z/Y/X") === {regular: ["A/B/C"], control: []};
   *  - getPredecessors("Z/Y") === {regular: ["A"], control: []};
   *  - getPredecessors("Z") === {regular: ["A"], control: []};
   *
   * The reason getPredecessors("Z/Y") returns ["A"] (and not ["A/B"] as you
   * might intuitively expect) is because it's not clear how far down the
   * other end of the hierarchy to traverse in the general case.
   *
   * Continuing this example, say there was another BaseEdge A/K->Z/Y/W. When
   * we look at Z/Y's predecessors, the best we can say is ["A"] without getting
   * into the details of which of of Z/Y's descendant nodes have predecessors to
   * which of A's descendants.
   *
   * On the other hand, for an OpNode it's clear what the final predecessors
   * ought to be. There is no ambiguity.
   */
  getPredecessors(nodeName: string): Edges {
    let node = this.index[nodeName];
    if (!node) {
      throw Error("Could not find node with name: " + nodeName);
    }

    let predecessors = this.getOneWayEdges(node, true);

    // Add embedded predecessors, such as constants.
    if (!node.isGroupNode) {
      _.each((<OpNode>node).inEmbeddings, embeddedNode => {
        predecessors.regular.push(embeddedNode.name);
      });
    }
    return predecessors;
  }

  /**
   * Given the name of a node, return an array of the names of its successors.
   * For an OpNode, this will contain the targets from the underlying BaseEdges.
   * For a GroupNode, this will contain the targets truncated to sibling of
   * the shared ancestor.
   *
   * This is the inverse of getPredecessors(). See that method's documentation
   * for an in-depth example.
   */
  getSuccessors(nodeName: string): Edges {
    let node = this.index[nodeName];
    if (!node) {
      throw Error("Could not find node with name: " + nodeName);
    }

    let successors = this.getOneWayEdges(node, false);

    // Add embedded successors, such as summaries.
    if (!node.isGroupNode) {
      _.each((<OpNode>node).outEmbeddings, embeddedNode => {
        successors.regular.push(embeddedNode.name);
      });
    }
    return successors;
  }

  /** Helper method for getPredecessors and getSuccessors */
  getOneWayEdges(node: GroupNode|OpNode, inEdges: boolean) {
    let edges = { control: [], regular: [] };
    // A node with no parent cannot have any edges.
    if (!node.parentNode || !node.parentNode.isGroupNode) {
      return edges;
    }
    let parentNode = <GroupNode> node.parentNode;
    let metagraph = parentNode.metagraph;
    let bridgegraph = this.getBridgegraph(parentNode.name);
    findEdgeTargetsInGraph(metagraph, node, inEdges, edges);
    findEdgeTargetsInGraph(bridgegraph, node, inEdges, edges);
    return edges;
  }

  /**
   * For a given GroupNode, get or calculate an object which describes a
   * topological ordering of child nodes within that GroupNode's metagraph.
   *
   * This ordering is used when rendering bridge control edges which are
   * sometimes backwards relative to the dataflow.
   *
   * For example, say we have a graph with two edges A->B and A->C, and we're
   * interested in the ordering under ROOT. In this case, any of the following
   * would be legitimate return values:
   *
   *  - { "A": 0, "B": 1, "C": 2 } -- most likely
   *  - { "A": 0, "B": 2, "C": 1 } -- less likely
   *  - { "A": 12, "B": 100, "C": 99 } -- unlikely, but still OK
   *
   * The algorithm does not guarantee that all numbers from 0-N (where N is
   * the number of nodes) appear exactly once. Rather it guarantees that if
   * there is a path between two nodes, the earlier one will have a lower
   * number in the ordering hash.
   *
   * When generating the ordering, we ignore control Metaedges (those which
   * represent only BaseEdges that have isControlDependency set to true).
   *
   * If there is no node with the specified name, an error is thrown. If the
   * node with the specified name is not a group node, null is returned.
   */
  getTopologicalOrdering(nodeName: string): { [childName: string]: number } {
    let node = this.index[nodeName];
    if (!node) {
      throw Error("Could not find node with name: " + nodeName);
    }
    if (!node.isGroupNode) {
      return null;
    }
    if (nodeName in this.orderings) {
      return this.orderings[nodeName];
    }

    // Mapping of a child node names to lists of their successors.
    let successors: { [childName: string]: string[] } = {};

    // Set of node names which have appeared as a destination.
    let destinations: { [childName: string]: boolean } = {};

    let metagraph = (<GroupNode> node).metagraph;
    _.each(metagraph.edges(), (e: graphlib.EdgeObject) => {
      if (!metagraph.edge(e).numRegularEdges) {
        return; // Skip control edges.
      }

      // Keep track of successors and destinations.
      if (!(e.v in successors)) {
        successors[e.v] = [];
      }
      successors[e.v].push(e.w);
      destinations[e.w] = true;
    });

    // Seed the queue with true sources (those that are not destinations).
    let queue: string[] =
      _.difference(_.keys(successors), _.keys(destinations));

    // Produce an ordering by traversing the graph breadth first.
    let ordering = this.orderings[nodeName] = {};
    let index = 0;
    while (queue.length) {
      let childName = queue.shift();
      ordering[childName] = index++;
      _.each(successors[childName], succName => queue.push(succName));
      delete successors[childName]; // Prevent cycles from infinite looping.
    }
    return ordering;
  }

  /**
   * Returns a d3 Ordinal function that can be used to look up the index of
   * a node based on its template id.
   */
  getTemplateIndex(): (string) => number {
    let templateNames = d3.keys(this.templates);
    let templateIndex = d3.scale.ordinal()
        .domain(templateNames)
        .range(d3.range(0, templateNames.length));
    return (templateId: string) => <number>templateIndex(templateId);
  }
}

/**
 * Internal utility function - given a graph (should be either a metagraph or a
 * bridgegraph) and a node which is known to be in that graph, determine
 * the other ends of edges that involve that node in the direction specified
 * by whether it's inbound.
 *
 * For example if you wanted to find the predecessors of a node, you'd call
 * this method for the parent's metagraph and bridgegraph, specifying inbound
 * as true (look at the source of inbound edges to the specified node).
 *
 * Discovered target names are appended to the targets array.
 */
function findEdgeTargetsInGraph(
    graph: graphlib.Graph<GroupNode|OpNode, Metaedge>,
    node: Node, inbound: boolean, targets: Edges): void {
  let edges = inbound ? graph.inEdges(node.name) : graph.outEdges(node.name);
  _.each(edges, e => {
    let otherName = inbound ? e.v : e.w;
    let metaedge = graph.edge(e);

    if (node.isGroupNode && metaedge.baseEdgeList.length > 1) {
      let targetList = metaedge.numRegularEdges
        ? targets.regular : targets.control;
      targetList.push(otherName);
    } else {
      // Enumerate all the base edges if the node is an OpNode, or the
      // metaedge has only 1 edge in it.
      _.each(metaedge.baseEdgeList, (baseEdge: BaseEdge) => {
        let targetList = baseEdge.isControlDependency
          ? targets.control : targets.regular;
        targetList.push(inbound ? baseEdge.v : baseEdge.w);
      });
    }
  });
}

export interface HierarchyParams {
  verifyTemplate: boolean;
  seriesNodeMinSize: number;
  seriesMap: { [name: string]: tf.graph.SeriesGroupingType };
}

/**
 * @param graph The raw graph.
 * @param params Parameters used when building a hierarchy.
 */
export function build(graph: tf.graph.SlimGraph, params: HierarchyParams,
    tracker: ProgressTracker): Promise<Hierarchy|void> {
  let h = new HierarchyImpl();
  let seriesNames: { [name: string]: string } = {};
  return runAsyncTask("Adding nodes", 20, () => {
    // Get all the possible device names.
    let deviceNames = {};
    _.each(graph.nodes, (node, nodeName) => {
      if (node.device != null) {
        deviceNames[node.device] = true;
      }
    });
    h.devices = _.keys(deviceNames);
    addNodes(h, graph);
  }, tracker)
  .then(() => {
    return runAsyncTask("Detect series", 20, () => {
      if (params.seriesNodeMinSize > 0) {
        groupSeries(h.root, h, seriesNames, params.seriesNodeMinSize,
          params.seriesMap);
      }
    }, tracker);
  })
  .then(() => {
    return runAsyncTask("Adding edges", 30, () => {
      addEdges(h, graph, seriesNames);
    }, tracker);
  })
  .then(() => {
    return runAsyncTask("Finding similar subgraphs", 30, () => {
      h.templates = template.detect(h, params.verifyTemplate);
    }, tracker);
  })
  .then(() => {
    return h;
  });
};

export function joinAndAggregateStats(h: Hierarchy, stats: StepStats) {
  // Get all the possible device names.
  let deviceNames = {};
  _.each(h.root.leaves(), nodeName => {
    let leaf = <OpNode> h.node(nodeName);
    if (leaf.device != null) {
      deviceNames[leaf.device] = true;
    }
  });
  h.devices = _.keys(deviceNames);

  // Reset stats for each group node.
  _.each(h.getNodeMap(), (node, nodeName) => {
    if (node.isGroupNode) {
      node.stats = new NodeStats(0, 0, null);
      (<GroupNode>node).deviceHistogram = {};
    }
  });

  // Bubble-up the stats and device distribution from leaves to parents.
  _.each(h.root.leaves(), nodeName => {
    let leaf = <OpNode> h.node(nodeName);
    let node = <GroupNode|OpNode> leaf;
    while (node.parentNode != null) {
      if (leaf.device != null) {
        let deviceHistogram = (<GroupNode>node.parentNode).deviceHistogram;
        deviceHistogram[leaf.device] = (deviceHistogram[leaf.device] || 0) + 1;
      }
      if (leaf.stats != null) {
        node.parentNode.stats.combine(leaf.stats);
      }
      node = <GroupNode> node.parentNode;
    }
  });
}

/**
 * Creates the metanodes in the hierarchical graph and assigns parent-child
 * relationship between them.
 */
function addNodes(h: Hierarchy, graph: SlimGraph) {
  _.each(graph.nodes, (node, nodeName) => {
    let path = getHierarchicalPath(node.name);
    let parent: Metanode = h.root;

    parent.depth = Math.max(path.length, parent.depth);

    // Create parent metanodes for each depth. For example if the node name
    // is 'a/b/c', then create metanodes 'a' and 'a/b', where 'a/b' is a child
    // of a.
    for (let i = 0; i < path.length; i++) {
      parent.depth = Math.max(parent.depth, path.length - i);
      parent.cardinality += node.cardinality;
      parent.opHistogram[node.op] = (parent.opHistogram[node.op] || 0) + 1;
      if (node.device != null) {
        parent.deviceHistogram[node.device] =
            (parent.deviceHistogram[node.device] || 0) + 1;
      }
      if (i === path.length - 1) { break; }
      let name = path[i];
      let child = <Metanode>h.node(name);
      if (!child) {
        child = createMetanode(name);
        child.parentNode = parent;
        h.setNode(name, child);
        parent.metagraph.setNode(name, child);
      }
      parent = child;
    }
    // Assuming node name is 'a/b/c', assign the OpNode as a child of the
    // metanode 'a/b'.
    h.setNode(node.name, node);
    node.parentNode = parent;
    parent.metagraph.setNode(node.name, node);

    // Add each of the in-embeddings and out-embeddings in the hierarchy.
    _.each(node.inEmbeddings, function(embedding) {
      h.setNode(embedding.name, embedding);
      embedding.parentNode = node;
    });
    _.each(node.outEmbeddings, function(embedding) {
      h.setNode(embedding.name, embedding);
      embedding.parentNode = node;
    });
  });
};

/**
 * For each metanode in the hierarchical graph, this method adds:
 * the edges in the metagraph. These are edges between nodes
 * that share the same parent.
 */
function addEdges(h: Hierarchy, graph: SlimGraph,
    seriesNames: { [name: string]: string }) {

  let nodeIndex = h.getNodeMap();

  // Ancestor paths for the source and destination nodes of an edge. These are
  // reused for each edge rather than allocating new ones. It's about 10% faster
  // than allocating new ones on each pass through the loop.
  let sourcePath: string[] = [];
  let destPath: string[] = [];

  // Insert the ancestor path for a node into the provided array, including the
  // node itself. Return the index of the last node inserted (always ROOT).
  let getPath = (node: Node, path: string[]): number => {
    let i = 0;
    while (node) {
      path[i++] = node.name;
      node = node.parentNode;
    }
    return i - 1;
  };

  _.each(graph.edges, baseEdge => {

    // Get the hierarchical paths for the source and destination of the edge.
    let sourceAncestorIndex = getPath(graph.nodes[baseEdge.v], sourcePath);
    let destAncestorIndex = getPath(graph.nodes[baseEdge.w], destPath);

    // If the hierarchical path cannot be found for either endpoint, then we
    // cannot create the edge. This happens for example when a node has a
    // control dependency on a summary node, which are embedded.
    if (sourceAncestorIndex === -1 || destAncestorIndex === -1) {
      return;
    }

    // Find the lowest shared ancestor between source and dest by looking for
    // the highest nodes that differ between their ancestor paths.
    while (sourcePath[sourceAncestorIndex] === destPath[destAncestorIndex]) {
      sourceAncestorIndex--;
      destAncestorIndex--;
      if (sourceAncestorIndex < 0 || destAncestorIndex < 0) {
        // This would only occur if the two nodes were the same (a cycle in the
        // graph), or if one endpoint was a strict ancestor of the other. The
        // latter shouldn't happen because we rename nodes which are both
        // metanodes and op nodes. E.g. "A/B" becomes "A/B/(B)".
        throw Error("No difference found between ancestor paths.");
      }
    }

    let sharedAncestorNode =
      <GroupNode>nodeIndex[sourcePath[sourceAncestorIndex + 1]];
    let sourceAncestorName = sourcePath[sourceAncestorIndex];
    let destAncestorName = destPath[destAncestorIndex];

    // Find or create the Metaedge which should contain this BaseEdge inside
    // the shared ancestor.
    let metaedge =
      sharedAncestorNode.metagraph.edge(sourceAncestorName, destAncestorName);
    if (!metaedge) {
      metaedge = createMetaedge(sourceAncestorName, destAncestorName);
      sharedAncestorNode.metagraph
        .setEdge(sourceAncestorName, destAncestorName, metaedge);
    }
    if (!sharedAncestorNode.hasNonControlEdges &&
        !baseEdge.isControlDependency) {
      sharedAncestorNode.hasNonControlEdges = true;
    }
    metaedge.addBaseEdge(baseEdge, h);
  });
};

/**
 * Using the hierarchy template information, detect series in the provided
 * metanode.  For each detected series, create a new SeriesNode
 * and remove series members from the metanode's metagraph and move them to
 * the new series node's metagraph.
 *
 * @param metanode
 * @param hierarchy
 * @param seriesNames Map of node names to their series they are contained in.
 *     This should be provided empty and is populated by this method.
 * @param threshold If the series has this many nodes or more, then group them
 *     into a series.
 * @param map Map of series names to their series grouping type, if one has
 *     been set.
 * @return A dictionary from node name to series node name that contains the
 *     node.
 */
function groupSeries(metanode: Metanode, hierarchy: Hierarchy,
    seriesNames: { [name: string]: string }, threshold: number,
    map: { [name: string]: tf.graph.SeriesGroupingType }) {
  let metagraph = metanode.metagraph;
  _.each(metagraph.nodes(), n => {
    let child = metagraph.node(n);
    if (child.type === tf.graph.NodeType.META) {
      groupSeries(<Metanode>child, hierarchy, seriesNames, threshold, map);
    }
  });

  let clusters = clusterNodes(metagraph);
  let seriesDict = detectSeries(clusters, metagraph);

  // Add each series node to the graph and add its grouped children to its own
  // metagraph.
  _.each(seriesDict, function(seriesNode: SeriesNode, seriesName: string) {
    let nodeMemberNames = seriesNode.metagraph.nodes();
    _.each(nodeMemberNames, n => {
      let child = <OpNode>metagraph.node(n);
      if (!child.owningSeries) {
        child.owningSeries = seriesName;
      }
    });
    // If the series contains less than the threshold number of nodes and
    // this series has not been adding to the series map, then set this
    // series to be shown ungrouped in the map.
    if (nodeMemberNames.length < threshold && !(seriesNode.name in map)) {
      map[seriesNode.name] = tf.graph.SeriesGroupingType.UNGROUP;
    }
    // If the series is in the map as ungrouped then do not group the series.
    if (seriesNode.name in map
      && map[seriesNode.name] === tf.graph.SeriesGroupingType.UNGROUP) {
      return;
    }
    hierarchy.setNode(seriesName, seriesNode); // add to the index
    metagraph.setNode(seriesName, seriesNode);
    _.each(nodeMemberNames, n => {
      let child = <OpNode> metagraph.node(n);
      seriesNode.metagraph.setNode(n, child);
      seriesNode.parentNode = child.parentNode;
      seriesNode.cardinality++;
      if (child.device != null) {
        seriesNode.deviceHistogram[child.device] =
            (seriesNode.deviceHistogram[child.device] || 0) + 1;
      }
      child.parentNode = seriesNode;
      seriesNames[n] = seriesName;
      // Remove now-grouped node from its original parent's metagraph.
      metagraph.removeNode(n);
    });
  });
};

/** cluster op-nodes with similar op */
function clusterNodes(metagraph: graphlib.Graph<GroupNode|OpNode, Metaedge>):
    {[clusterId: string]: string[]} {
  let result: {[clusterId: string]: string[]} = {};
  return  _.reduce(metagraph.nodes(),
      (clusters: {[clusterId: string]: string[]}, n: string) => {
    let child = metagraph.node(n);
    if (child.type === NodeType.META) {
      // skip metanodes
      return clusters;
    }
    let template = (<OpNode>child).op;
    if (template) {
      clusters[template] = clusters[template] || [];
      clusters[template].push(child.name);
    }
    return clusters;
  }, result);
}

/**
 * For each cluster of op-nodes based op type, try to detect groupings.
 * Infer series name using by trying to find pattern "<number>" in the node
 * name.
 *
 * @param clusters Dictionary output from clusterNodes().
 * @param metagraph
 * @return A dictionary from series name => seriesNode
 */
function detectSeries(clusters: {[clusterId: string]: string[]},
     metagraph: graphlib.Graph<GroupNode|OpNode, Metaedge>):
     {[seriesName: string]: SeriesNode} {
  let seriesDict: {[seriesName: string]: SeriesNode} = {};
  _.each(clusters, function(members, clusterId: string) {
    if (members.length <= 1) { return; } // isolated clusters can't make series

    /** @type {Object}  A dictionary mapping seriesName to seriesInfoArray,
     * which is an array that contains objects with name, id, prefix, suffix,
     * and parent properties.
     */
    let candidatesDict: {[seriesName: string]: SeriesNode[]} = {};

    // Group all nodes that have the same name, with the exception of a
    // number at the end of the name after an underscore, which is allowed to
    // vary.
    _.each(members, function(name: string) {
      let isGroup = name.charAt(name.length - 1) === "*";
      let namepath = name.split("/");
      let leaf = namepath[namepath.length - 1];
      let parent = namepath.slice(0, namepath.length - 1).join("/");
      let matches = leaf.match(/^(\D*)_(\d+)$/);

      let prefix;
      let id;
      let suffix = "";
      if (matches) { // if found "<number>" in the name, assign id.
        prefix = matches[1]; // the front non-numeric characters
        id = matches[2]; // the digits
      } else { // for node without "_<number>", make them zero-th items.
        prefix = isGroup ? leaf.substr(0, leaf.length - 1) : leaf;
        id = 0;
        suffix = isGroup ? "*" : "";
      }
      let seriesName = getSeriesNodeName(prefix, suffix, parent);
      candidatesDict[seriesName] = candidatesDict[seriesName] || [];
      let seriesNode = createSeriesNode(prefix, suffix, parent, +id, name);
      candidatesDict[seriesName].push(seriesNode);
    });

    // In each group of nodes, group nodes in bunches that have monotonically
    // increasing numbers in their names.  Each of these bunches is a series.
    _.each(candidatesDict, function(seriesInfoArray: SeriesNode[], seriesName) {
      if (seriesInfoArray.length < 2) {
        return;
      }
      seriesInfoArray.sort(function(a, b) {
        return (+a.clusterId) - (+b.clusterId);
      });

      // Loop through the nodes sorted by its detected series number, grouping
      // all nodes with monotonically-increasing series numbers.
      let seriesNodes = [seriesInfoArray[0]];
      for (let index = 1; index < seriesInfoArray.length; index++) {
        let nextNode = seriesInfoArray[index];
        if (nextNode.clusterId === seriesNodes[seriesNodes.length - 1].clusterId
            + 1) {
          seriesNodes.push(nextNode);
          continue;
        }
        addSeriesToDict(seriesNodes, seriesDict, +clusterId, metagraph);
        seriesNodes = [nextNode];
      }
      addSeriesToDict(seriesNodes, seriesDict, +clusterId, metagraph);
    });
  });
  return seriesDict;
}

/**
 * Add a series to the provided dictionary mapping series names to series.
 *
 * @param seriesNodes the nodes in the series. Contains
 *     name, id, prefix, suffix and parent properties of the node.
 * @param seriesDict the dictionary of series
 * @param clusterId ID of the template of the nodes of the series
 * @param metagraph
 */
function addSeriesToDict(seriesNodes: SeriesNode[],
    seriesDict: {[seriesName: string]: SeriesNode},
    clusterId: number,
    metagraph: graphlib.Graph<GroupNode|OpNode, Metaedge>) {
  if (seriesNodes.length > 1) {
    let curSeriesName = getSeriesNodeName(
      seriesNodes[0].prefix, seriesNodes[0].suffix,
      seriesNodes[0].parent, seriesNodes[0].clusterId,
      seriesNodes[seriesNodes.length - 1].clusterId);
    let curSeriesNode = createSeriesNode(seriesNodes[0].prefix,
      seriesNodes[0].suffix, seriesNodes[0].parent, clusterId,
      curSeriesName);
    _.each(seriesNodes, function(node) {
      curSeriesNode.ids.push(node.clusterId);
      curSeriesNode.metagraph.setNode(node.name, metagraph.node(node.name));
    });
    seriesDict[curSeriesName] = curSeriesNode;
  }
}

} // close module tf.graph.hierarchy
