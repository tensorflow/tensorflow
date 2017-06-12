/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the 'License');
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an 'AS IS' BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
/**
 * Package for the Render Hierarchy for TensorFlow graph.
 */
module tf.graph.render {

export type Point = {x: number, y: number};

/**
 * Color parameters for op nodes.
 */
export let OpNodeColors = {DEFAULT_FILL: 'white', DEFAULT_STROKE: '#b2b2b2'};

/**
 * Color parameters for node encoding.
 * @type {Object}
 */
export let MetanodeColors = {
  /**
   * Default fill and stroke to use when no other information is available.
   */
  DEFAULT_FILL: '#d9d9d9',
  DEFAULT_STROKE: '#a6a6a6',
  SATURATION: 0.6,
  LIGHTNESS: 0.85,
  /**
   * Neutral color to use when the node is expanded (used when coloring by
   * compute time, memory and device).
   */
  EXPANDED_COLOR: '#f0f0f0',
  /**
   * Standard hue values for node color palette.
   */
  HUES: [220, 100, 180, 40, 20, 340, 260, 300, 140, 60],
  STRUCTURE_PALETTE(id: number, lightened?: boolean) {
    // The code below is a flexible way to computationally create a set
    // of colors that go well together.
    let hues = MetanodeColors.HUES;
    let n = hues.length;
    let hue = hues[id % n];
    let m = Math.sin(hue * Math.PI / 360);
    let sat = lightened ? 30 : 90 - 60 * m;
    let light = lightened ? 95 : 80;
    return d3.hsl(hue, .01 * sat, .01 * light).toString();
  },
  DEVICE_PALETTE(index: number): string {
    return MetanodeColors.STRUCTURE_PALETTE(index);
  },
  XLA_CLUSTER_PALETTE(index: number): string {
    return MetanodeColors.STRUCTURE_PALETTE(index);
  },
  UNKNOWN: '#eee',
  GRADIENT_OUTLINE: '#888'
};

/**
 * Color parameters for op nodes.
 */
export let SeriesNodeColors = {
  DEFAULT_FILL: 'white',
  DEFAULT_STROKE: '#b2b2b2'
};

/**
 * Parameters that affect how the graph is rendered on the screen.
 */
const PARAMS = {
  /**
   * Whether to extract high degree nodes from the core part of the graph.
   */
  enableExtraction: true,
  /**
   * The minimum number of nodes for a graph to have in order for high in and
   * out degree nodes to be extracted in auxiliary. The aim here is to prevent
   * nodes from being extracted from small graphs.
   */
  minNodeCountForExtraction: 15,
  /**
   * The minimum in or out degree a node must have in order to be possibly
   * extracted.
   */
  minDegreeForExtraction: 5,
  /**
   * Maximum number of control edges a node can have before they aren't
   * displayed.
   */
  maxControlDegree: 4,
  /**
   * Maximum in (for outbound bridge paths) or out (for inbound bridge paths)
   * degree of a node allowed for a bridge path to be rendered to it from a
   * subhierarchy of nodes. Having a max prevents having too many nodes emanate
   * from a subhierarchy and crowding up.
   */
  maxBridgePathDegree: 4,
  /**
   * Types patterns for predefined out-extract nodes, which are
   * sink-like nodes that will be extracted from the main graph.
   */
  outExtractTypes: [
    'NoOp'  // NoOps are sink-like used for managing control dependencies.
  ],

  /**
   * Types patterns for predefined in-extract nodes, which are
   * source-like nodes that will be extracted from the main graph.
   */
  inExtractTypes: [],

  /**
   * When removing edges from a high degree node, remove all of its edges if
   * detachAllEdgesForHighDegree is true.  Otherwise remove all in-edges if
   * the node has high in-degree, or all out-edges if the node has high
   * out-degree.
   */
  detachAllEdgesForHighDegree: true,

  /**
   * After extracting high in/out degree nodes and predefined
   * source-like/sink-like, extract isolated nodes to the side
   * if this extractIsolatedNodesWithAnnotationsOnOneSide is true.
   */
  extractIsolatedNodesWithAnnotationsOnOneSide: true,

  /**
   * Whether to add bridge nodes and edges to the core when building the
   * subhierarchy of an expanded metanode. See buildSubhierarchy().
   */
  enableBridgegraph: true,

  /**
   * 2 colors, for the minimum and maximum value respectively, whenever we
   * have a gradient scale.
   */
  minMaxColors: ['#fff5f0', '#fb6a4a'],

  /**
   * Maximum number of annotations to be displayed on a node before an
   * ellipsis is used.
   */
  maxAnnotations: 5
};

/**
 * Stores the rendering information, such as x and y coordinates,
 * for each node in the graph.
 */
export class RenderGraphInfo {
  hierarchy: hierarchy.Hierarchy;
  private displayingStats: boolean;
  private index: {[nodeName: string]: RenderNodeInfo};
  private renderedOpNames: string[];
  private deviceColorMap: d3.ScaleOrdinal<string, string>;
  private xlaClusterColorMap: d3.ScaleOrdinal<string, string>;
  private memoryUsageScale: d3.ScaleLinear<string, string>;
  private computeTimeScale: d3.ScaleLinear<string, string>;
  /** Scale for the thickness of edges when there is no shape information. */
  edgeWidthScale:
      d3.ScaleLinear<number, number> | d3.ScalePower<number, number>;
  // Since the rendering information for each node is constructed lazily,
  // upon node's expansion by the user, we keep a map between the node's name
  // and whether the rendering information was already constructed for that
  // node.
  private hasSubhierarchy: {[nodeName: string]: boolean};
  root: RenderGroupNodeInfo;
  traceInputs: Boolean;

  constructor(hierarchy: hierarchy.Hierarchy, displayingStats: boolean) {
    this.hierarchy = hierarchy;
    this.displayingStats = displayingStats;
    this.index = {};
    this.renderedOpNames = [];

    this.computeScales();
    // Maps node name to whether the rendering hierarchy was already
    // constructed.
    this.hasSubhierarchy = {};
    this.root = new RenderGroupNodeInfo(hierarchy.root);
    this.index[hierarchy.root.name] = this.root;
    this.renderedOpNames.push(hierarchy.root.name);
    this.buildSubhierarchy(hierarchy.root.name);
    this.root.expanded = true;
    this.traceInputs = false;
  }

  computeScales() {
    this.deviceColorMap = d3.scaleOrdinal<string>()
        .domain(this.hierarchy.devices)
        .range(_.map(d3.range(this.hierarchy.devices.length),
                     MetanodeColors.DEVICE_PALETTE));

    this.xlaClusterColorMap =
        d3.scaleOrdinal<string>()
            .domain(this.hierarchy.xlaClusters)
            .range(_.map(
                d3.range(this.hierarchy.xlaClusters.length),
                MetanodeColors.XLA_CLUSTER_PALETTE));

    let topLevelGraph = this.hierarchy.root.metagraph;
    // Find the maximum and minimum memory usage.
    let memoryExtent = d3.extent(topLevelGraph.nodes(),
        (nodeName, index) => {
      let node = topLevelGraph.node(nodeName);
      // Some ops don't have stats at all.
      if (node.stats != null) {
        return node.stats.totalBytes;
      }
    });
    this.memoryUsageScale = d3.scaleLinear<string, string>()
        .domain(memoryExtent)
        .range(PARAMS.minMaxColors);

    // Find also the minimum and maximum compute time.
    let computeTimeExtent = d3.extent(topLevelGraph.nodes(),
        (nodeName, index) => {
      let node = topLevelGraph.node(nodeName);
      // Some ops don't have stats at all.
      if (node.stats != null) {
        return node.stats.getTotalMicros();
      }
    });
    this.computeTimeScale = d3.scaleLinear<string, string>()
        .domain(computeTimeExtent)
        .range(PARAMS.minMaxColors);

    this.edgeWidthScale = this.hierarchy.hasShapeInfo ?
      scene.edge.EDGE_WIDTH_SCALE :
      d3.scaleLinear()
        .domain([1, this.hierarchy.maxMetaEdgeSize])
        .range([scene.edge.MIN_EDGE_WIDTH, scene.edge.MAX_EDGE_WIDTH]);
  }

  /**
   * Get a previously created RenderNodeInfo by its node name.
   */
  getRenderNodeByName(nodeName: string): RenderNodeInfo {
    return this.index[nodeName];
  }

  /**
   * Get the underlying node in the hierarchical graph by its name.
   */
  getNodeByName(nodeName: string): Node {
    return this.hierarchy.node(nodeName);
  }

  /**
   * Get a previously created RenderNodeInfo for the specified node name,
   * or create one if it hasn't been created yet.
   */
  getOrCreateRenderNodeByName(nodeName: string): RenderNodeInfo {
    // Polymer may invoke this with null.
    if (!nodeName) {
      return null;
    }

    if (nodeName in this.index) {
      return this.index[nodeName];
    }

    let node = this.hierarchy.node(nodeName);
    // Exit early if the node does not exist in the hierarchy. This can happen
    // when a graph is reloaded while the infocard points to a node not visible
    // at the top-level.
    if (!node) {
      return null;
    }
    let renderInfo = node.isGroupNode ?
        new RenderGroupNodeInfo(<GroupNode>node) :
        new RenderNodeInfo(node);
    this.index[nodeName] = renderInfo;
    this.renderedOpNames.push(nodeName);

    if (node.stats) {
      renderInfo.memoryColor = this.memoryUsageScale(node.stats.totalBytes);
      renderInfo.computeTimeColor =
          this.computeTimeScale(node.stats.getTotalMicros());
    }

    if (!node.isGroupNode) {
      let clusterName = (node as OpNode).xlaCluster;
      if (clusterName) {
        renderInfo.xlaClusterColor = this.xlaClusterColorMap(clusterName);
      }
    }

    // We only fade nodes when we're displaying stats.
    renderInfo.isFadedOut = this.displayingStats &&
        !tf.graph.util.hasDisplayableNodeStats(node.stats);

    if (node.isGroupNode) {
      // Make a list of tuples (device, proportion), where proportion
      // is the fraction of op nodes that have that device.
      let pairs = _.pairs((<GroupNode>node).deviceHistogram);
      if (pairs.length > 0) {
        // Compute the total # of devices.
        let numDevices = _.sum(pairs, _.last);
        renderInfo.deviceColors = _.map(pairs, pair => ({
              color: this.deviceColorMap(pair[0]),
              // Normalize to a proportion of total # of devices.
              proportion: pair[1] / numDevices
            }));
      }
    } else {
      let device = (<OpNode>renderInfo.node).device;
      if (device) {
        renderInfo.deviceColors = [{
          color: this.deviceColorMap(device),
          proportion: 1.0
        }];
      }
    }

    return this.index[nodeName];
  }

  /**
   * Return the nearest ancestor node, including itself, that is visible
   * in the visualization. This method is used so that we can select
   * (highlight) a node that isn't drawn yet, by selecting (highlighting)
   * its nearest ancestor that has been drawn.
   */
  getNearestVisibleAncestor(name: string): string {
    let path = getHierarchicalPath(name);
    for (let i = 0; i < path.length; i++) {
      let nodeName = path[i];
      // Op nodes have expanded set to false by default.
      if (!this.getRenderNodeByName(nodeName).expanded) {
        return nodeName;
      }
    }
    // Fallthrough. If everything was expanded return the node.
    return name;
  }

  // TODO(jimbo): Delete this an any code it touches (all deprecated).
  setDepth(depth: number): void {
    setGroupNodeDepth(this.root, +depth);
  }

  /**
   * Returns true if the renderNode is an isolated node within its parent node.
   */
  isNodeAuxiliary(renderNode: RenderNodeInfo): boolean {
    let parentNode = <RenderGroupNodeInfo>this.getRenderNodeByName(
      renderNode.node.parentNode.name);
    let found = _.find(parentNode.isolatedInExtract, node => {
      return node.node.name === renderNode.node.name;
    });
    if (found) {
      return true;
    }
    found = _.find(parentNode.isolatedOutExtract, node => {
      return node.node.name === renderNode.node.name;
    });
    return !!found;
  }

  /**
   * Returns a list of ops that have been rendered so far for this graph. More
   * ops may later be rendered if the user expands nodes for instance. The list
   * returned here can only stay the same size or grow on successive calls.
   */
  getNamesOfRenderedOps(): string[] {
    return this.renderedOpNames;
  }

  buildSubhierarchy(nodeName: string): void {
    // Terminate if the rendering hierarchy was already constructed
    // for this node.
    if (nodeName in this.hasSubhierarchy) {
      return;
    }

    let renderNodeInfo = this.index[nodeName];

    // If it is not a meta node or a series node, don't do anything.
    if (renderNodeInfo.node.type !== NodeType.META &&
        renderNodeInfo.node.type !== NodeType.SERIES) {
      return;
    }

    // At this point we know the rendering information is about a group node.
    let renderGroupNodeInfo = <RenderGroupNodeInfo> renderNodeInfo;
    let metagraph = renderGroupNodeInfo.node.metagraph;
    let coreGraph = renderGroupNodeInfo.coreGraph;

    // Create render nodes to represent each child from the metagraph. Although
    // these will initially be added to the coreGraph, they may later be
    // extracted. Also, due to extraction, the coreGraph may contain disjoint
    // groups between which there is no visible path (other than annotations).
    _.each(metagraph.nodes(), childName => {

      let childRenderInfo = this.getOrCreateRenderNodeByName(childName);
      let childNode = childRenderInfo.node;

      coreGraph.setNode(childName, childRenderInfo);

      if (!childNode.isGroupNode) {
        _.each((<OpNode>childNode).inEmbeddings, embedding => {
          let renderMetaedgeInfo = new RenderMetaedgeInfo(null);
          addInAnnotation(childRenderInfo, embedding, null, renderMetaedgeInfo,
              AnnotationType.CONSTANT);
          this.index[embedding.name] = new RenderNodeInfo(embedding);
        });
        _.each((<OpNode>childNode).outEmbeddings, embedding => {
          let renderMetaedgeInfo = new RenderMetaedgeInfo(null);
          addOutAnnotation(childRenderInfo, embedding, null, renderMetaedgeInfo,
              AnnotationType.SUMMARY);
          this.index[embedding.name] = new RenderNodeInfo(embedding);
        });
      }

    });

    // Add render metaedge info for edges in the metagraph.
    _.each(metagraph.edges(), edgeObj => {
      let metaedge = metagraph.edge(edgeObj);
      let renderMetaedgeInfo = new RenderMetaedgeInfo(metaedge);
      renderMetaedgeInfo.isFadedOut =
          this.index[edgeObj.v].isFadedOut || this.index[edgeObj.w].isFadedOut;
      coreGraph.setEdge(edgeObj.v, edgeObj.w, renderMetaedgeInfo);
    });

    if (PARAMS.enableExtraction &&
        renderGroupNodeInfo.node.type === NodeType.META) {
      extractHighDegrees(renderGroupNodeInfo);
    }

    // Record that we constructed the rendering hierarchy for this node, so we
    // don't construct it another time.
    this.hasSubhierarchy[nodeName] = true;

    // Look up the parent node's render information and short circuit if none.
    let parentNode = renderGroupNodeInfo.node.parentNode;
    if (!parentNode) {
      return;
    }
    let parentNodeInfo =
      <RenderGroupNodeInfo> this.index[parentNode.name];

    // Utility function for computing the name of a bridge node.
    let getBridgeNodeName = (inbound, ...rest) =>
        rest.concat([inbound ? 'IN' : 'OUT']).join('~~');

    // Build out the bridgegraph.
    let bridgegraph = this.hierarchy.getBridgegraph(nodeName);

    // Look for popular nodes so we can make annotations instead of paths.
    let otherCounts = {
      // Counts of edges coming INTO other nodes by name (outgoing from self).
      in: <{[nodeName: string]: number}> {},
      // Counts of edges going OUT from other nodes by name (coming into self).
      out: <{[nodeName: string]: number}> {},
      // Counts of all control edges involving other nodes by name.
      control: <{[nodeName: string]: number}> {},
    };
    _.each(bridgegraph.edges(), e => {
      // An edge is inbound if its destination node is in the metagraph.
      let inbound = !!metagraph.node(e.w);
      let otherName = inbound ? e.v : e.w;
      let metaedge = bridgegraph.edge(e);
      if (!metaedge.numRegularEdges) {
        otherCounts.control[otherName] =
          (otherCounts.control[otherName] || 0) + 1;
      } else if (inbound) {
        otherCounts.out[otherName] = (otherCounts.out[otherName] || 0) + 1;
      } else {
        otherCounts.in[otherName] = (otherCounts.in[otherName] || 0) + 1;
      }
    });

    // Add annotations and edges for bridgegraph relationships.
    let hierarchyNodeMap = this.hierarchy.getNodeMap();
    _.each(bridgegraph.edges(), bridgeEdgeObj => {
      let bridgeMetaedge = bridgegraph.edge(bridgeEdgeObj);

      // Determine whether this bridge edge is incoming by checking the
      // metagraph for a node that matches the destination end.
      let inbound = !!metagraph.node(bridgeEdgeObj.w);

      // Based on the direction of the edge, one endpoint will be an immediate
      // child of this renderNodeInfo, and the other endpoint will be a sibling
      // of the parent (or an ancestor further up).
      let [childName, otherName] =
        inbound ?
          [bridgeEdgeObj.w, bridgeEdgeObj.v] :
          [bridgeEdgeObj.v, bridgeEdgeObj.w];

      let childRenderInfo = this.index[childName];
      let otherRenderInfo = this.index[otherName];
      let otherNode =
        otherRenderInfo ?
          otherRenderInfo.node :
          hierarchyNodeMap[otherName];

      // Determine whether this edge is a control edge between nodes where
      // either node is high-degree with respect to control edges. This will
      // be a signal to show it as an annotation instead of a bridge edge.
      let isHighDegreeControlEdge = !bridgeMetaedge.numRegularEdges &&
        otherCounts.control[otherName] > PARAMS.maxControlDegree;

      let [, childAnnotations] =
        inbound ?
          [renderNodeInfo.inAnnotations, childRenderInfo.inAnnotations] :
          [renderNodeInfo.outAnnotations, childRenderInfo.outAnnotations];

      // Don't render a bridge path if the other node has in or out degree above
      // a threshold, lest bridge paths emanating out of a metagraph crowd up,
      // as was the case for the Fatcat LSTM lstm_1 > lstm_1 metagraph.
      let otherDegreeCount =
          (inbound ? otherCounts.out : otherCounts.in)[otherName];
      let isOtherHighDegree = otherDegreeCount > PARAMS.maxBridgePathDegree;

      // The adjoining render metaedge info from the parent's coreGraph, if any.
      // It will either be a Metaedge involving this node directly, if it
      // previously came from a metagraph, or it'll be a Metaedge involving
      // a previously created bridge node standing in for the other node.
      let adjoiningMetaedge = null;

      // We can only hope to render a bridge path if:
      //  - bridgegraph paths are enabled,
      //  - the other node is not too high-degree,
      //  - the child is in the core (not extracted for being high-degree), and
      //  - there's a path (in the traversal sense) between child and other.
      let canDrawBridgePath = false;
      if (PARAMS.enableBridgegraph &&
          !isOtherHighDegree &&
          !isHighDegreeControlEdge &&
          childRenderInfo.isInCore()) {

        // Utility function for finding an adjoining metaedge.
        let findAdjoiningMetaedge = targetName => {
          let adjoiningEdgeObj: graphlib.EdgeObject =
            inbound ?
              { v: targetName, w: nodeName } :
              { v: nodeName, w: targetName };
          return <RenderMetaedgeInfo>
            parentNodeInfo.coreGraph.edge(adjoiningEdgeObj);
        };

        adjoiningMetaedge = findAdjoiningMetaedge(otherName);
        if (!adjoiningMetaedge) {
          adjoiningMetaedge = findAdjoiningMetaedge(
              getBridgeNodeName(inbound, otherName, parentNode.name));
        }

        canDrawBridgePath = !!adjoiningMetaedge;
      }

      // Although dataflow edges are acyclic, control dependency edges may
      // actually point 'backwards' in the graph. If this bridgeMetaedge is
      // a control dependency, we need to determine whether it's backwards
      // pointing so that we render it appropriately.
      //
      // For instance, say we're rendering a graph with nodes named A/B and Z/Y,
      // and we're currently rendering the bridgegraph for A. Further, let's say
      // that there was an original BaseEdge from A/B->Z/Y and a CONTROL EDGE
      // from Z/Y=>A/B.
      //
      //     +----------------+
      //     | A              |
      //     |  +-----+       |         +------+
      //     |  | B   |>----->|>------->| Z    |
      //     |  |     |       |         |      |
      //     |  |     |   *   |         |      |
      //     |  |     |<=====<|<=======<|      |
      //     |  +-----+       |         +------+
      //     +----------------+
      //
      // When we render the subhierarchy for Metanode A, we'll come across a
      // control-only Metaedge in the bridgegraph from Z=>A/B (*). The question
      // is whether this edge is backwards.
      //
      // To answer that question, we follow the chain of adjoining metaedges
      // until we reach the topmost one. In this case, that's the control-only
      // Metaedge Z=>A in the ROOT's metagraph. We determine that this edge
      // is backwards by looking at the topological ordering of ROOT's metagraph
      // (which ignores control edges) and seeing that Z comes AFTER A.
      //
      // The property of being backwards is independent of whether the edge
      // is inbound or outbound. In the preceding example, if we were building
      // the subhierarchy for Z, we'd find bridge edge Z/Y=>A, walk to its
      // topmost adjoining metaedge Z=>A and discover that it's backwards.
      let backwards = false;
      if (adjoiningMetaedge && !bridgeMetaedge.numRegularEdges) {
        // Find the top-most adjoining render metaedge information, and the
        // GroupNode whose metagraph must contain the associated metaedge.
        let topAdjoiningMetaedge = adjoiningMetaedge;
        let topGroupNode = parentNodeInfo.node;
        while (topAdjoiningMetaedge.adjoiningMetaedge) {
          topAdjoiningMetaedge = topAdjoiningMetaedge.adjoiningMetaedge;
          topGroupNode = <GroupNode>topGroupNode.parentNode;
        }

        // Check against the topological ordering for the top node. The current
        // bridge metaedge we're evaluating is backwards if its source comes
        // after its destination.
        let ordering = this.hierarchy.getTopologicalOrdering(topGroupNode.name);
        let e = topAdjoiningMetaedge.metaedge;
        backwards = ordering[e.v] > ordering[e.w];
      }

      // Render backwards control edges as annotations.
      canDrawBridgePath = canDrawBridgePath && !backwards;

      // If we can't make a bridge path for any reason, then we add an
      // annotation instead.
      if (!canDrawBridgePath) {
        childAnnotations.push(new Annotation(
            otherNode,
            otherRenderInfo,
            new RenderMetaedgeInfo(bridgeMetaedge),
            AnnotationType.SHORTCUT,
            inbound));
        return;
      }

      // At this point, all conditions have been met for drawing a bridge path.

      // Find or create the IN/OUT node representing otherNode.
      let bridgeContainerName = getBridgeNodeName(inbound, nodeName);
      let bridgeNodeName = getBridgeNodeName(inbound, otherName, nodeName);
      let bridgeNodeRenderInfo = coreGraph.node(bridgeNodeName);
      if (!bridgeNodeRenderInfo) {

        // Find or create the directional container for the bridge node.
        let bridgeContainerInfo = coreGraph.node(bridgeContainerName);
        if (!bridgeContainerInfo) {
          let bridgeContainerNode: BridgeNode = {
            // Important node properties.
            name: bridgeContainerName,
            type: NodeType.BRIDGE,
            // Unused node properties.
            isGroupNode: false,
            cardinality: 0,
            parentNode: null,
            stats: null,
            include: InclusionType.UNSPECIFIED,
            // BridgeNode properties.
            inbound: inbound,
            nodeAttributes: {},
          };
          bridgeContainerInfo =
            new RenderNodeInfo(bridgeContainerNode);
          this.index[bridgeContainerName] = bridgeContainerInfo;
          coreGraph.setNode(bridgeContainerName, bridgeContainerInfo);
        }

        let bridgeNode: BridgeNode = {
          // Important node properties.
          name: bridgeNodeName,
          type: NodeType.BRIDGE,
          // Unimportant node properties.
          isGroupNode: false,
          cardinality: 1,
          parentNode: null,
          stats: null,
          include: InclusionType.UNSPECIFIED,
          // BridgeNode properties.
          inbound: inbound,
          nodeAttributes: {},
        };
        bridgeNodeRenderInfo = new RenderNodeInfo(bridgeNode);
        this.index[bridgeNodeName] = bridgeNodeRenderInfo;
        coreGraph.setNode(bridgeNodeName, bridgeNodeRenderInfo);

        // Set bridgeNode to be a graphlib child of the container node.
        coreGraph.setParent(bridgeNodeName, bridgeContainerName);
        bridgeContainerInfo.node.cardinality++;
      }

      // Create and add a bridge render metaedge.
      let bridgeRenderMetaedge =
        new RenderMetaedgeInfo(bridgeMetaedge);
      bridgeRenderMetaedge.adjoiningMetaedge = adjoiningMetaedge;
      inbound ?
        coreGraph.setEdge(bridgeNodeName, childName, bridgeRenderMetaedge) :
        coreGraph.setEdge(childName, bridgeNodeName, bridgeRenderMetaedge);

    }); // End _.each(bridgegraph.edges).

    // For each bridge container (IN and/or OUT), add structural edges between
    // terminal nodes and that container. A terminal node is one which has no
    // non-bridge edges in the direction of the container.
    //
    // For example, consider a Metanode A which contains two child nodes A/B
    // and A/C. Let's say it has one edge in the metagraph from A/B->A/C, and
    // one edge in the bridgegraph from Z->A/C.
    //
    // At this point, we've added a container bridge node IN to house all
    // incoming bridge nodes. We've also added a bridge node Z' (with parent IN)
    // to A, and a bridge edge from Z'->C.
    //
    //     +----------------------+
    //     | A          +---+     |
    //     |    +------>| C |     |
    //     |    |       +---+     |
    //     |    |         ^       |
    //     |    |         |       |
    //     |    |    +----|----+  |
    //     |    |    | IN |    |  |
    //     |  +---+  |  +---+  |  |
    //     |  | B |  |  | Z'|  |  |
    //     |  +---+  |  +---+  |  |
    //     |         +---------+  |
    //     +----------------------+
    //
    // With no other help, dagre would lay out B and Z' on the same level,
    // because both of them have no incoming edges. In other words, B is a
    // terminal node in the INCOMING direction.
    //
    // But we want to force dagre to lay out Z' (and everything in IN) lower
    // than all non-bridge nodes, so that there's enough room for the bridge
    // edges after they've been adjusted to meet up with paths coming in from
    // outside.
    //
    // To force Z' (and all other bridge nodes) to be lowest in the graph, we
    // identify terminal nodes like B and give them structural edges to
    // a new structural bridge node S which we add to IN.
    //
    //     +----------------------+
    //     | A          +---+     |
    //     |       +--->| C |     |
    //     |       |    +---+     |
    //     |     +---+    ^       |
    //     |     | B |    |       |
    //     |     +---+    |       |
    //     |       ^      |       |
    //     |       |      |       |
    //     |  +----|------|----+  |
    //     |  |IN  |      |    |  |
    //     |  |  +---+  +---+  |  |
    //     |  |  | S |  | Z'|  |  |
    //     |  |  +---+  +---+  |  |
    //     |  +----------------+  |
    //     +----------------------+
    //
    // This ensures that dagre will lay out the bridge containers strictly at
    // the ends of the graph. The structural edges will never be seen in the
    // visualization except as a debugging aid.
    _.each([true, false], inbound => {
      let bridgeContainerName = getBridgeNodeName(inbound, nodeName);
      let bridgeContainerInfo = coreGraph.node(bridgeContainerName);
      if (!bridgeContainerInfo) {
        return;
      }
      _.each(coreGraph.nodes(), childName => {
        // Short-circuit if this child is a bridge node or it's not a terminal
        // node in the direction we're interested in.
        let childNodeInfo = coreGraph.node(childName);
        if (childNodeInfo.node.type === NodeType.BRIDGE) {
          return;
        }
        let isTerminal = inbound ?
          !coreGraph.predecessors(childName).length :
          !coreGraph.successors(childName).length;
        if (!isTerminal) {
          return;
        }

        // Find or create a bridge node in the container for all structural
        // metaedges. It would have been nice to skip this step and simply
        // set a metaedge between the terminal node and the container node, but
        // in that case, something about the graph upsets dagre.layout()'s
        // longestPath algorithm (was getting errors due to an undefined).
        let structuralNodeName =
            getBridgeNodeName(inbound, nodeName, 'STRUCTURAL_TARGET');
        let structuralRenderInfo = coreGraph.node(structuralNodeName);
        if (!structuralRenderInfo) {
          let bridgeNode: BridgeNode = {
            // Important Node properties.
            name: structuralNodeName,
            type: NodeType.BRIDGE,
            // Unimportant Node properties.
            isGroupNode: false,
            cardinality: 1,
            parentNode: null,
            stats: null,
            include: InclusionType.UNSPECIFIED,
            // BridgeNode properties.
            inbound: inbound,
            nodeAttributes: {},
          };
          structuralRenderInfo = new RenderNodeInfo(bridgeNode);
          structuralRenderInfo.structural = true;
          this.index[structuralNodeName] = structuralRenderInfo;
          coreGraph.setNode(structuralNodeName, structuralRenderInfo);
          bridgeContainerInfo.node.cardinality++;
          coreGraph.setParent(structuralNodeName, bridgeContainerName);
        }

        // Create the structural Metaedge and insert it.
        let structuralMetaedgeInfo = new RenderMetaedgeInfo(null);
        structuralMetaedgeInfo.structural = true;
        structuralMetaedgeInfo.weight--; // Reduce weight for dagre layout.
        inbound ?
          coreGraph.setEdge(
              structuralNodeName, childName, structuralMetaedgeInfo) :
          coreGraph.setEdge(
              childName, structuralNodeName, structuralMetaedgeInfo);
      });
    });
  }
}

/**
 * A class for rendering annotation object which contains label
 * about the node embedded as annotation, type of annotation and the location
 * of both the annotation's node and edge.
 *
 * Annotation objects include embedded constants, embedded summary, and
 * edge shortcuts.
 */
export class Annotation {
  node: Node;
  renderNodeInfo: RenderNodeInfo;
  renderMetaedgeInfo: RenderMetaedgeInfo;
  annotationType: AnnotationType;
  /**
   * Center position of annotation relative to the host
   * node's center x.
   */
  dx: number;
  /**
   * Center position of annotation relative to the host
   * node's center y.
   */
  dy: number;
  width: number;
  height: number;
  /**
   * The names of nodes on either side of this edge.
   */
  v: string;
  w: string;
  /**
   * A flag whether it is an in-annotation (if true) or
   * out-annotation  (if false).
   */
  isIn: boolean;
  /** Label horizontal offset from the end of the node shape */
  labelOffset: number;
  /**
   * Array of points for edges from the annotation to its host
   * node. Each point contains the point location, relative to
   * the host node's center.
   */
  points: {dx: number, dy: number}[];

  /**
   * Creates a new Annotation.
   *
   * @param node The underlying node this annotation points to.
   * @param renderNodeInfo The render information for the underlying node
   *     this annotation points to. This can be null if the annotation
   *     denotes an embedding (constant, summary), in which case we
   *     use the node property.
   * @param renderMetaedgeInfo The render information for the edge associated
   *     with the annotation.
   * @param type The type of the annotation.
   * @param isIn True if it is an in-annotation. False if it is an
   *     out-annotation.
   */
  constructor(node: Node, renderNodeInfo: RenderNodeInfo,
      renderMetaedgeInfo: RenderMetaedgeInfo, type: AnnotationType,
      isIn: boolean) {
    this.node = node;
    this.renderNodeInfo = renderNodeInfo;
    this.renderMetaedgeInfo = renderMetaedgeInfo;
    this.annotationType = type;
    // Properties specified by layout
    this.dx = 0;
    this.dy = 0;
    this.width = 0;
    this.height = 0;
    // Properties needed for generating an ID for the edge's path element if
    // this annotation is associated with a metaedge.
    if (renderMetaedgeInfo && renderMetaedgeInfo.metaedge) {
      this.v = renderMetaedgeInfo.metaedge.v;
      this.w = renderMetaedgeInfo.metaedge.w;
    }

    this.isIn = isIn;
    this.points = [];
  }
};

export enum AnnotationType {SHORTCUT, CONSTANT, SUMMARY, ELLIPSIS};

/**
 * Manages a list of annotations. Two will be used for each
 * RenderNodeInfo, one for in annotations and one for out annotations.
 */
export class AnnotationList {
  /**
   * List of visually drawable annotations, may include an ellipses annotation
   * if the number added exceeds the number specified by maxAnnotations.
   */
  list: Annotation[];

  /**
   * Set of nodes which have been added as annotations to this list, so we can
   * prevent duplicates.
   */
  nodeNames: { [nodeName: string]: boolean };

  constructor() {
    this.list = [];
    this.nodeNames = {};
  }

  /**
   * Append an annotation to the list, or a stand-in ellipsis annotation instead
   * if this would make it too many.
   */
  push(annotation: Annotation): void {
    if (annotation.node.name in this.nodeNames) {
      return; // Skip duplicate annotation.
    }
    this.nodeNames[annotation.node.name] = true;

    if (this.list.length < PARAMS.maxAnnotations) {
      this.list.push(annotation);
      return;
    }

    let lastAnnotation = this.list[this.list.length - 1];
    if (lastAnnotation.annotationType === AnnotationType.ELLIPSIS) {
      let ellipsisNode = <EllipsisNode>lastAnnotation.node;
      ellipsisNode.setNumMoreNodes(++ellipsisNode.numMoreNodes);
      return;
    }

    let ellipsisNode = new tf.graph.EllipsisNodeImpl(1);
    this.list.push(new Annotation(ellipsisNode,
        new RenderNodeInfo(ellipsisNode), null,
        AnnotationType.ELLIPSIS, annotation.isIn));
  }
}

/**
 * Contains rendering information about a node in the hierarchical graph.
 */
export class RenderNodeInfo {
  /** Reference to the original underlying Node from the hierarchical graph. */
  node: Node;
  /** Whether the node is expanded or not. */
  expanded: boolean;
  /**
   * List of rendering information about in-annotations like constants and
   * shortcuts to high-degree nodes.
   */
  inAnnotations: AnnotationList;
  /**
   * List of rendering information about out-annotations (e.g. summary nodes)
   */
  outAnnotations: AnnotationList;

  // --- Params specified by layout --- //

  /** Center x position */
  x: number;
  /** Center y position */
  y: number;
  /**
   * Total width of the node's shape, including in- and out-annotations. This
   * property is used by dagre to layout the graph.
   */
  width: number;
  /**
   * Total height of the node's shape, including in- and out-annotations. This
   * property is used by dagre to layout the graph.
   */
  height: number;
  /**
   * Size of the main box of the node, excluding in- and out-annotations. This
   * property is used to draw the rectangle/ellipse shape denoting the node.
   */
  coreBox: {
    width: number,
    height: number,
  };

  /** Width of the bounding box for all in-annotations. */
  inboxWidth: number;
  /** Width of the bounding box for all out-annotations. */
  outboxWidth: number;
  /**
   * Whether the node should be excluded from the scene.
   * This is only used when there are too many items in a series so we only
   * want to include top N ones.
   */
  // TODO(jimbo): Now that series rendering is non-recursive, remove this and
  // all its uses from the code base.
  excluded: boolean;

  // --- Params used in drawing the bridge paths --- //

  /**
   * All bridge nodes are meant to be invisible, but whereas most represent a
   * relationship from the underlying graph hierarchy, some exist solely for
   * layout reasons. Specifically, those bridge nodes which have only structural
   * rendering metaedges.
   */
  structural: boolean;

  // --- Params for the size of the node box --- //

  /** Label vertical offset from the center of node shape */
  labelOffset: number;
  /** Rectangle radius (for making rounded rectangle) */
  radius: number;

  // --- Params for expanded node --- //

  /** Label height for expanded node. */
  labelHeight: number;
  // Paddings between inner subscene and the border of the expanded node.
  paddingTop: number;
  paddingLeft: number;
  paddingRight: number;
  paddingBottom: number;

  /**
   * Whether a node is extracted as source-like (having high out-degree or
   * matching predefined in-extract pattern.)
   */
  isInExtract: boolean;
  /**
   * Whether a node is extracted as sink-like (having high in-degree or matching
   * predefined out-extract pattern.)
   */
  isOutExtract: boolean;

  /**
   * List of (color, proportion) tuples based on the proportion of devices of
   * its children. If this node is an op node, this list will have only one
   * color with proportion 1.0.
   */
  deviceColors: Array<{color: string, proportion: number}>;

  /**
   * Color according to the XLA cluster of this node.
   */
  xlaClusterColor: string;

  /**
   * Color according to the memory usage of this node.
   */
  memoryColor: string;

  /**
   * Color according to the compute time of this node.
   */
  computeTimeColor: string;

  /**
   * Whether this node is faded out. Used when displaying stats.
   */
  isFadedOut: boolean;

  constructor(node: Node) {
    this.node = node;
    this.expanded = false;
    this.inAnnotations = new AnnotationList();
    this.outAnnotations = new AnnotationList();
    // Params specified by layout
    this.x = 0;
    this.y = 0;
    this.width = 0;
    this.height = 0;
    this.inboxWidth = 0;
    this.outboxWidth = 0;

    this.excluded = false;

    // Params for bridge paths.
    this.structural = false;

    // Params for node box.
    this.labelOffset = 0;
    this.radius = 0;

    // Params for expanded node
    this.labelHeight = 0;
    this.paddingTop = 0;
    this.paddingLeft = 0;
    this.paddingRight = 0;
    this.paddingBottom = 0;
    this.isInExtract = false;
    this.isOutExtract = false;
    this.coreBox = {width: 0, height: 0};

    // By default, we don't fade nodes out. Default to false for safety.
    this.isFadedOut = false;
  }

  isInCore(): boolean {
    return !this.isInExtract && !this.isOutExtract;
  }
}

/**
 * Contains rendering information about a Metaedge from the underlying
 * hierarchical graph. It may be from either a metagraph or a bridgegraph.
 */
export class RenderMetaedgeInfo {
  /**
   * Reference to the original underlying Metaedge from the hierarchical graph,
   * if any. This will be null for the edges which connect OpNodes to their
   * embeddings, for example.
   */
  metaedge: Metaedge;

  /**
   * Reference to the adjoining RenderMetaedgeInfo from the parent's
   * coreGraph. This is used during layout to determine the point at which this
   * edge should touch the node's bounding box. This property will be null for
   * edges which terminate at a node on both ends (all non-bridge edges).
   */
  adjoiningMetaedge: RenderMetaedgeInfo;

  /**
   * Most of the time, a RenderMetaedgeInfo object represents a real
   * edge between nodes in the underlying graph structure. But sometimes, an
   * edge only exists for layout purposes. These structural edges are added
   * during buildSubhierarchy() to force dagre.layout() to put bridge nodes
   * at the ends of the flow.
   * @see buildSubhierarchy()
   */
  structural: boolean;

  /**
   * Weight of the edge, used by dagre when deciding how important an edge is.
   * Edges with higher weight are made shorter and straighter. The default
   * dagre uses is 1.
   */
  weight: number;

  /**
   * X and Y coordinate pairs of the points in the path of the edge.
   * @see tf.graph.node.subsceneAdjustPaths
   */
  points: Point[];

  /**
   * D3 selection of the group containing the path that displays this edge.
   */
  edgeGroup: d3.Selection<RenderMetaedgeInfo & any, any, any, any>;

  /** Id of the <marker> used as a start-marker for the edge path. */
  startMarkerId: string;

  /** Id of the <marker> used as an end-marker for the edge path. */
  endMarkerId: string;

  /**
   * Whether this edge is faded out. Used for fading out unused edges when
   * displaying run statistics.
   */
  isFadedOut: boolean;

  constructor(metaedge: Metaedge) {
    this.metaedge = metaedge;
    this.adjoiningMetaedge = null;
    this.structural = false;
    this.weight = 1;
    this.isFadedOut = false;
  }
}

function addInAnnotation(node: RenderNodeInfo, predecessor: Node,
    predecessorRenderInfo: RenderNodeInfo,
    edge: RenderMetaedgeInfo, type: AnnotationType): void {
  let annotation = new Annotation(predecessor, predecessorRenderInfo, edge,
      type, true);
  node.inAnnotations.push(annotation);
}

function addOutAnnotation(node: RenderNodeInfo, successor: Node,
    successorRenderInfo: RenderNodeInfo, edge: RenderMetaedgeInfo,
    type: AnnotationType): void {
  let annotation = new Annotation(successor, successorRenderInfo, edge,
      type, false);
  node.outAnnotations.push(annotation);
}

function setGraphDepth(graph: graphlib.Graph<RenderNodeInfo, any>,
    depth: number) {
  _.each(graph.nodes(), nodeName => {
    let child = graph.node(nodeName);
    child.expanded = depth > 1; // set all child of depth 1 to collapsed
    if (depth > 0) {
      switch (child.node.type) {
        case NodeType.META:
        case NodeType.SERIES:
          setGroupNodeDepth(<RenderGroupNodeInfo>child, depth - 1);
          break;
        // Do nothing for leaf
      }
    }
  });
};

export class RenderGroupNodeInfo extends RenderNodeInfo {
  node: GroupNode;
  /**
   * The core graph is derived from the underlying node's metagraph, minus
   * the extracted source-like and sink-like nodes.
   */
  coreGraph: graphlib.Graph<RenderNodeInfo, RenderMetaedgeInfo>;
  /** Size of the bounding box for a metanode's isolated in-extract children. */
  inExtractBox: {width: number, height: number};
  /**
   * Size of the bounding box for a metanode's isolated out-extract children.
   */
  outExtractBox: {width: number, height: number};
  /** Array of isolated in-extract nodes. */
  isolatedInExtract: RenderNodeInfo[];
  /** Array of isolated out-extract nodes. */
  isolatedOutExtract: RenderNodeInfo[];

  constructor(groupNode: GroupNode) {
    super(groupNode);
    let metagraph = groupNode.metagraph;
    let gl = metagraph.graph();
    this.coreGraph =
        createGraph<RenderNodeInfo, RenderMetaedgeInfo>(
            gl.name, GraphType.CORE, { compound: true });
    this.inExtractBox = {width: 0, height: 0};
    this.outExtractBox = {width: 0, height: 0};
    this.isolatedInExtract = [];
    this.isolatedOutExtract = [];
  }
}

function setGroupNodeDepth(renderInfo: RenderGroupNodeInfo,
    depth: number): void {
  if (renderInfo.coreGraph) {
    setGraphDepth(renderInfo.coreGraph, depth);
  }
}

/**
 * Remove an edge from the graph and add annotations to both ends of the edge.
 *
 * @param The core graph.
 * @param v Source name.
 * @param w Sink name.
 */
function createShortcut(
    graph: graphlib.Graph<RenderNodeInfo, RenderMetaedgeInfo>,
    v: string, w: string) {
  let src = graph.node(v);
  let sink = graph.node(w);
  let edge = graph.edge(v, w);

  // If either of the nodes is explicitly included in the main graph and
  // both nodes are in the main graph then do not create the shortcut
  // and instead keep the real edge.
  if ((src.node.include === InclusionType.INCLUDE ||
       sink.node.include === InclusionType.INCLUDE) &&
      src.node.include !== InclusionType.EXCLUDE &&
      sink.node.include !== InclusionType.EXCLUDE) {
    return;
  }

  // Add each annotation.
  addOutAnnotation(src, sink.node, sink, edge, AnnotationType.SHORTCUT);
  addInAnnotation(sink, src.node, src, edge, AnnotationType.SHORTCUT);

  // Remove the edge from the core graph.
  graph.removeEdge(v, w);
}

/**
 * Remove edges from a node, and set its isOutExtract property to true,
 * and remove the node and move it to isolatedOutExtract.
 *
 * If detachAllEdgesForHighDegree or forceDetach is true, extract all of its
 * edges. Otherwise, only extract all in-edges.
 */
function makeOutExtract(renderNode: RenderGroupNodeInfo, n: string,
    forceDetach?: boolean) {
  let graph = renderNode.coreGraph;
  let child = graph.node(n);
  child.isOutExtract = true;

  _.each(graph.predecessors(n), (p, index) => {
    createShortcut(graph, p, n);
  });

  if (PARAMS.detachAllEdgesForHighDegree || forceDetach) {
    _.each(graph.successors(n), (s, index) => {
      createShortcut(graph, n, s);
    });
  }

  // Remove the node from the core graph if it no longer has neighbors.
  if (graph.neighbors(n).length === 0) {
    child.node.include = InclusionType.EXCLUDE;
    renderNode.isolatedOutExtract.push(child);
    graph.removeNode(n);
  }
}

/**
 * Remove edges from a node, set its isInExtract property to true,
 * and remove the node and move it to isolatedInExtract.
 *
 * If detachAllEdgesForHighDegree or forceDetach is true, extract all of its
 * edges. Otherwise, only remove all out-edges.
 */
export function makeInExtract(renderNode: RenderGroupNodeInfo, n: string,
    forceDetach?: boolean) {
  let graph = renderNode.coreGraph;
  let child = graph.node(n);
  child.isInExtract = true;

  _.each(graph.successors(n), (s, index) => {
    createShortcut(graph, n, s);
  });

  if (PARAMS.detachAllEdgesForHighDegree || forceDetach) {
    _.each(graph.predecessors(n), (p, index) => {
      createShortcut(graph, p, n);
    });
  }

  // Remove the node from the core graph if it no longer has neighbors.
  if (graph.neighbors(n).length === 0) {
    child.node.include = InclusionType.EXCLUDE;
    renderNode.isolatedInExtract.push(child);
    graph.removeNode(n);
  }
}

/**
 * Check whether the node's type is a member of the given list of types.
 *
 * @param node Node.
 * @param types List of type to match.
 */
function hasTypeIn(node: Node, types: string[]): boolean {
  if (node.type === NodeType.OP) {
    for (let i = 0; i < types.length; i++) {
      if ((<OpNode>node).op === types[i]) { return true; }
    }
  } else if (node.type === NodeType.META) {
    let rootOpNode = (<Metanode>node).getRootOp();
    if (rootOpNode) {
      for (let i = 0; i < types.length; i++) {
        if (rootOpNode.op === types[i]) { return true; }
      }
    }
  }
  return false;
}

/** Move nodes that are specified to be excluded out of the core graph. */
function extractSpecifiedNodes(renderNode: RenderGroupNodeInfo) {
  let graph = renderNode.coreGraph;
  _.each(graph.nodes(), n => {
    let renderInfo = graph.node(n);
    if (renderInfo.node.include === InclusionType.EXCLUDE) {
      if (renderNode.coreGraph.outEdges(n).length >
          renderNode.coreGraph.inEdges(n).length) {
        makeOutExtract(renderNode, n, true);
      } else {
        makeInExtract(renderNode, n, true);
      }
    }
  });
}

/** Remove edges from pre-defined out-extract patterns */
function extractPredefinedSink(renderNode: RenderGroupNodeInfo) {
  let graph = renderNode.coreGraph;
  _.each(graph.nodes(), n => {
    let renderInfo = graph.node(n);
    if (renderInfo.node.include !== InclusionType.UNSPECIFIED) {
      return;
    }
    if (hasTypeIn(renderInfo.node, PARAMS.outExtractTypes)) {
      makeOutExtract(renderNode, n);
    }
  });
}

/** Remove edges from pre-defined in-extract patterns */
function extractPredefinedSource(renderNode) {
  let graph = renderNode.coreGraph;
  _.each(graph.nodes(), n => {
    let renderInfo = graph.node(n);
    if (renderInfo.node.include !== InclusionType.UNSPECIFIED) {
      return;
    }
    if (hasTypeIn(renderInfo.node, PARAMS.inExtractTypes)) {
      makeInExtract(renderNode, n);
    }
  });
}

/** Extract nodes deemed to have either high in-degree or high out-degree. */
function extractHighInOrOutDegree(renderNode: RenderGroupNodeInfo) {
  let graph = renderNode.coreGraph;

  // Create mappings from node to in and out degrees. Count the number of valid
  // nodes along the way.
  let nodeToInDegree = {};
  let nodeToOutDegree = {};
  let validNodeCount = 0;
  _.each(graph.nodes(), currentNode => {
    if (graph.node(currentNode).node.include !== InclusionType.UNSPECIFIED) {
      // This node is not included in the first place.
      return;
    }

    // Count the in and out degrees based on only regular edges, unless there
    // are no regular edges, in which case use the number of control edges.
    // This is done so that control edges don't affect if nodes are extracted
    // from the core graph, unless the node is only used for control.
    let inDegree =
        _.reduce(graph.predecessors(currentNode), (inDegree, pred) => {
          let metaedge = graph.edge(pred, currentNode).metaedge;
          return inDegree + (metaedge.numRegularEdges ? 1 : 0);
        }, 0);
    if (inDegree === 0 && graph.predecessors(currentNode).length > 0) {
      inDegree = graph.predecessors(currentNode).length;
    }

    let outDegree =
        _.reduce(graph.successors(currentNode), (outDegree, succ) => {
          let metaedge = graph.edge(currentNode, succ).metaedge;
          return outDegree + (metaedge.numRegularEdges ? 1 : 0);
        }, 0);
    if (outDegree === 0 && graph.successors(currentNode).length > 0) {
      outDegree = graph.successors(currentNode).length;
    }

    // Store the in and out degrees of this node to avoid recomputing.
    nodeToInDegree[currentNode] = inDegree;
    nodeToOutDegree[currentNode] = outDegree;
    validNodeCount++;
  });

  if (validNodeCount < PARAMS.minNodeCountForExtraction) {
    // This graph has few nodes. Do not extract any nodes.
    return;
  }

  // We only extract if the node has a min in or out degree greater than this.
  let minUpperBound = PARAMS.minDegreeForExtraction - 1;

  // Mark for extraction nodes with in-degree > Q3 + (Q3 - Q1).
  let q3Index = Math.round(validNodeCount * 0.75);
  let q1Index = Math.round(validNodeCount * 0.25);
  let sortedByInDegree = Object.keys(nodeToInDegree).sort((node0, node1) => {
    return nodeToInDegree[node0] - nodeToInDegree[node1];
  });
  let inDegreeQ3 = nodeToInDegree[sortedByInDegree[q3Index]];
  let inDegreeQ1 = nodeToInDegree[sortedByInDegree[q1Index]];
  let inDegreeUpperBound = inDegreeQ3 + inDegreeQ3 - inDegreeQ1;
  // Only extract if the upper bound is high enough.
  inDegreeUpperBound = Math.max(inDegreeUpperBound, minUpperBound);
  for (let i = validNodeCount - 1;
       nodeToInDegree[sortedByInDegree[i]] > inDegreeUpperBound; i--) {
    // Extract a high in-degree node.
    makeInExtract(renderNode, sortedByInDegree[i]);
  }

  // Mark for extraction nodes with out-degree > Q3 + (Q3 - Q1) * 4.
  let sortedByOutDegree = Object.keys(nodeToOutDegree).sort((node0, node1) => {
    return nodeToOutDegree[node0] - nodeToOutDegree[node1];
  });
  let outDegreeQ3 = nodeToOutDegree[sortedByOutDegree[q3Index]];
  let outDegreeQ1 = nodeToOutDegree[sortedByOutDegree[q1Index]];
  // The upper bound for extracting out-degree nodes is higher than that for
  // extracting in-degree ones (Note the "* 4") because, in practice, some
  // graphs look worse with a smaller out-degree bound. For instance, a smaller
  // out-degree bound removes the convolution nodes from cifar 10 train's graph.
  let outDegreeUpperBound = outDegreeQ3 + (outDegreeQ3 - outDegreeQ1) * 4;
  // Only extract if the upper bound is high enough.
  outDegreeUpperBound = Math.max(outDegreeUpperBound, minUpperBound);
  for (let i = validNodeCount - 1;
       nodeToOutDegree[sortedByOutDegree[i]] > outDegreeUpperBound; i--) {
    let node = graph.node(sortedByOutDegree[i]);
    if (!node || node.isInExtract) {
      // This node has already been extracted due to high in-degree. It might
      // have been removed from the graph in general (during in-degree
      // extraction) due to a lack of neighbors. Do not extract this node twice.
      continue;
    }

    // Extract a high out-degree node that has not already been extracted.
    makeOutExtract(renderNode, sortedByOutDegree[i]);
  }
}

/** Remove control edges from nodes that have too many control edges */
function removeControlEdges(renderNode: RenderGroupNodeInfo) {
  let graph = renderNode.coreGraph;

  // Collect control edges into a map by node name.
  let map = <{[nodeName: string]: graphlib.EdgeObject[]}>{};
  _.each(graph.edges(), e => {
    if (!graph.edge(e).metaedge.numRegularEdges) {
      (map[e.v] = map[e.v] || []).push(e);
      (map[e.w] = map[e.w] || []).push(e);
    }
  });

  // For each node with too many control edges, turn them into annotations.
  _.each(map, (edges, nodeName) => {
    if (edges.length > PARAMS.maxControlDegree) {
      _.each(edges, e => createShortcut(graph, e.v, e.w));
    }
  });
}

/**
 * Given an integer, picks a hue that is far apart from other colors.
 * The formula for picking color that avoid collision is:
 *     hue = (color range * golden ratio * index) % color range
 */
export function mapIndexToHue(id: number): number {
  let GOLDEN_RATIO = 1.61803398875;
  // Hue of 0 is reserved for the gray nodes.
  let MIN_HUE = 1;
  let MAX_HUE = 359;
  let COLOR_RANGE = MAX_HUE - MIN_HUE;
  return MIN_HUE + ((COLOR_RANGE * GOLDEN_RATIO * id) % COLOR_RANGE);
};

/**
 * Remove edges and add to annotation instead.
 *
 * For root node, consider predefined types for source and sink.
 * We do not extract predefined type from non-root so that Variables and the
 * sgd node (op type = 'NoOp') do not get extract from inside own group.
 *
 * The order of extraction is important here as swapping the order can totally
 * screw up the graph layout.
 *
 * @param {Render.Node} renderNode Node to manipulate.
 */
function extractHighDegrees(renderNode: RenderGroupNodeInfo) {

  extractSpecifiedNodes(renderNode);

  if (PARAMS.outExtractTypes) {
    extractPredefinedSink(renderNode);
  }

  // This has to come before extract high in-degree to protect the core part
  // that takes many variables.
  if (PARAMS.inExtractTypes) {
    extractPredefinedSource(renderNode);
  }

  extractHighInOrOutDegree(renderNode);

  if (PARAMS.maxControlDegree) {
    removeControlEdges(renderNode);
  }

  // Extract isolated nodes, which can be
  // (1) source-like and sink-like nodes that are not originally isolated but
  //     become isolated after further removal.
  // (2) isolated nodes with annotations on one-side.  These might be either
  //     - nodes that originally have high out-degree but because we remove
  //       high in-degree nodes first, they no longer have high in-degree when
  //       we check.  (Detecting all high-degree before removing also leads to
  //       another problem.)
  //     - nodes that do not have high degree, but their neighbors are all
  //       extracted, so it might make sense to extract them too.

  let graph = renderNode.coreGraph;
  _.each(graph.nodes(), n => {
    let child = graph.node(n);
    let degree = graph.neighbors(n).length;
    if (child.node.include !== InclusionType.UNSPECIFIED) {
      return;
    }
    if (degree === 0) {
      let hasOutAnnotations = child.outAnnotations.list.length > 0;
      let hasInAnnotations = child.inAnnotations.list.length > 0;

      if (child.isInExtract) { // Is source-like.
        // This case only happens if detachAllEdgesForHighDegree is false.
        // (Otherwise all source-like nodes are all isolated already.)
        renderNode.isolatedInExtract.push(child);
        child.node.include = InclusionType.EXCLUDE;
        graph.removeNode(n);
      } else if (child.isOutExtract) { // Is sink-like.
        // This case only happens if detachAllEdgesForHighDegree is false.
        // // (Otherwise all sink-like nodes are all isolated already.)
        renderNode.isolatedOutExtract.push(child);
        child.node.include = InclusionType.EXCLUDE;
        graph.removeNode(n);
      } else if (PARAMS.extractIsolatedNodesWithAnnotationsOnOneSide) {
        if (hasOutAnnotations && !hasInAnnotations) {
          child.isInExtract = true; // for ones with high out-annotations
          renderNode.isolatedInExtract.push(child);
          child.node.include = InclusionType.EXCLUDE;
          graph.removeNode(n);
        } else if (hasInAnnotations && !hasOutAnnotations) {
          child.isOutExtract = true; // for ones with high in-annotations
          renderNode.isolatedOutExtract.push(child);
          child.node.include = InclusionType.EXCLUDE;
          graph.removeNode(n);
        } else {
          // if a low degree node has both in- & out- annotations, do nothing
          // because it is unclear which side it should go to.
        }
      }
    }
  });
}

/**
 * Expands nodes in the graph until the desired node is visible.
 *
 * @param scene The scene polymer component.
 * @param renderHierarchy The render hierarchy.
 * @param tensorName The name of a tensor.
 * @return A string that is the name of the node representing the given tensor.
 *     Note that the original tensor name might differ from this returned node
 *     name. Specifically, for instance, the tensor name usually ends with an
 *     output slot index (such as :0), while the node name lacks that suffix.
 */
export function expandUntilNodeIsShown(
    scene, renderHierarchy, tensorName: string) {
  const splitTensorName = tensorName.split('/');

  // Graph names do not take into account the output slot. Strip it.
  const lastNodeNameMatch =
      splitTensorName[splitTensorName.length - 1].match(/(.*):\d+/);
  if (lastNodeNameMatch.length === 2) {
    splitTensorName[splitTensorName.length - 1] = lastNodeNameMatch[1];
  }

  let nodeName = splitTensorName[0];
  let renderNode = renderHierarchy.getRenderNodeByName(nodeName);
  for (let i = 1; i < splitTensorName.length; i++) {
    // Op nodes are not expandable.
    if (renderNode.node.type === tf.graph.NodeType.OP) {
      break;
    }
    renderHierarchy.buildSubhierarchy(nodeName);
    renderNode.expanded = true;
    scene.setNodeExpanded(renderNode);
    nodeName += '/' + splitTensorName[i];
    renderNode = renderHierarchy.getRenderNodeByName(nodeName);
  }

  return renderNode.node.name;
}

} // close module tf.graph.render
