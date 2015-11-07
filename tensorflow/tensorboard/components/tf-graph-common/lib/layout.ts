/// <reference path="graph.ts" />
/// <reference path="render.ts" />

module tf.graph.layout {

/** Set of parameters that define the look and feel of the graph. */
export const PARAMS = {
  animation: {
    /** Default duration for graph animations in ms. */
    duration: 250
  },
  graph: {
    /** Graph parameter for metanode. */
    meta: {
      /**
       * Dagre's nodesep param - number of pixels that
       * separate nodes horizontally in the layout.
       *
       * See https://github.com/cpettitt/dagre/wiki#configuring-the-layout
       */
      nodeSep: 110,
      /**
       * Dagre's ranksep param - number of pixels
       * between each rank in the layout.
       *
       * See https://github.com/cpettitt/dagre/wiki#configuring-the-layout
       */
      rankSep: 25
    },
    /** Graph parameter for metanode. */
    series: {
      /**
       * Dagre's nodesep param - number of pixels that
       * separate nodes horizontally in the layout.
       *
       * See https://github.com/cpettitt/dagre/wiki#configuring-the-layout
       */
      nodeSep: 90,
      /**
       * Dagre's ranksep param - number of pixels
       * between each rank in the layout.
       *
       * See https://github.com/cpettitt/dagre/wiki#configuring-the-layout
       */
      rankSep: 25,
    },
    /**
     * Padding is used to correctly position the graph SVG inside of its parent
     * element. The padding amounts are applied using an SVG transform of X and
     * Y coordinates.
     */
    padding: {
      paddingTop: 40,
      paddingLeft: 20
    }
  },
  subscene: {
    meta: {
      paddingTop: 10,
      paddingBottom: 10,
      paddingLeft: 10,
      paddingRight: 10,
      /**
       * Used to leave room for the label on top of the highest node in
       * the core graph.
       */
      labelHeight: 20,
      /** X-space between each extracted node and the core graph. */
      extractXOffset: 50,
      /** Y-space between each extracted node. */
      extractYOffset: 20
    },
    series: {
      paddingTop: 10,
      paddingBottom: 10,
      paddingLeft: 10,
      paddingRight: 10,
      labelHeight: 10
    }
  },
  nodeSize: {
    /** Size of meta nodes. */
    meta: {
      radius: 5,
      width: 60,
      /** A scale for the node's height based on number of nodes inside */
      height: d3.scale.linear().domain([1, 200]).range([15, 60]).clamp(true),
      /** The radius of the circle denoting the expand button. */
      expandButtonRadius: 3
    },
    /** Size of op nodes. */
    op: {
      width: 15,
      height: 6,
      radius: 3, // for making annotation touching ellipse
      labelOffset: -8
    },
    /** Size of series nodes. */
    series: {
      expanded: {
        // For expanded series nodes, width and height will be
        // computed to account for the subscene.
        radius: 10,
        labelOffset: 0,
      },
      vertical: {
        // When unexpanded, series whose underlying metagraphs contain
        // one or more non-control edges will show as a vertical stack
        // of ellipses.
        width: 16,
        height: 13,
        labelOffset: -13,
      },
      horizontal: {
        // When unexpanded, series whose underlying metagraphs contain
        // no non-control edges will show as a horizontal stack of
        // ellipses.
        width: 24,
        height: 8,
        radius: 10, // Forces annotations to center line.
        labelOffset: -10,
      },
    },
    /** Size of bridge nodes. */
    bridge: {
      // NOTE: bridge nodes will normally be invisible, but they must
      // take up some space so that the layout step leaves room for
      // their edges.
      width: 20,
      height: 20,
      radius: 2,
      labelOffset: 0
    }
  },
  shortcutSize: {
    /** Size of shortcuts for op nodes */
    op: {
      width: 10,
      height: 4
    },
    /** Size of shortcuts for meta nodes */
    meta: {
      width: 12,
      height: 4,
      radius: 1
    },
    /** Size of shortcuts for series nodes */
    series: {
      width: 14,
      height: 4,
    }
  },
  annotations: {
    /** X-space between the shape and each annotation-node. */
    xOffset: 10,
    /** Y-space between each annotation-node. */
    yOffset: 3,
    /** X-space between each annotation-node and its label. */
    labelOffset: 2,
    /** Estimate max width for annotation label */
    labelWidth: 35
  },
  constant: {
    size: {
      width: 4,
      height: 4
    }
  },
  series: {
    /** Maximum number of repeated item for unexpanded series node. */
    maxStackCount: 3,
    /**
     * Positioning offset ratio for collapsed stack
     * of parallel series (series without edges between its members).
     */
    parallelStackOffsetRatio: 0.2,
    /**
     * Positioning offset ratio for collapsed stack
     * of tower series (series with edges between its members).
     */
    towerStackOffsetRatio: 0.5
  },
  minimap : {
    /** The maximum width/height the minimap can have. */
    size: 150
  }
};

/** Calculate layout for a scene of a group node. */
export function scene(renderNodeInfo: render.RenderGroupNodeInformation)
    : void {
  // Update layout, size, and annotations of its children nodes and edges.
  if (renderNodeInfo.node.isGroupNode) {
    layoutChildren(renderNodeInfo);
  }

  // Update position of its children nodes and edges
  if (renderNodeInfo.node.type === NodeType.META) {
    layoutMetanode(renderNodeInfo);
  } else if (renderNodeInfo.node.type === NodeType.SERIES) {
    layoutSeriesNode(renderNodeInfo);
  }
};

/**
 * Update layout, size, and annotations of its children nodes and edges.
 */
function layoutChildren(renderNodeInfo: render.RenderGroupNodeInformation)
    : void {
  let children = renderNodeInfo.coreGraph.nodes().map(n => {
    return renderNodeInfo.coreGraph.node(n);
  }).concat(renderNodeInfo.isolatedInExtract,
      renderNodeInfo.isolatedOutExtract);

  _.each(children, childNodeInfo => {
    // Set size of each child
    switch (childNodeInfo.node.type) {
      case NodeType.OP:
        _.extend(childNodeInfo, PARAMS.nodeSize.op);
        break;
      case NodeType.BRIDGE:
        _.extend(childNodeInfo, PARAMS.nodeSize.bridge);
        break;
      case NodeType.META:
        if (!childNodeInfo.expanded) {
          // set fixed width and scalable height based on cardinality
          _.extend(childNodeInfo, PARAMS.nodeSize.meta);
          childNodeInfo.height =
              PARAMS.nodeSize.meta.height(childNodeInfo.node.cardinality);
        } else {
          let childGroupNodeInfo =
            <render.RenderGroupNodeInformation>childNodeInfo;
          scene(childGroupNodeInfo); // Recursively layout its subscene.
        }
        break;
      case NodeType.SERIES:
        if (childNodeInfo.expanded) {
          _.extend(childNodeInfo, PARAMS.nodeSize.series.expanded);
          let childGroupNodeInfo =
            <render.RenderGroupNodeInformation>childNodeInfo;
          scene(childGroupNodeInfo); // Recursively layout its subscene.
        } else {
          let childGroupNodeInfo =
            <render.RenderGroupNodeInformation>childNodeInfo;
          let seriesParams =
            childGroupNodeInfo.node.hasNonControlEdges ?
              PARAMS.nodeSize.series.vertical :
              PARAMS.nodeSize.series.horizontal;
          _.extend(childNodeInfo, seriesParams);
        }
        break;
      default:
        throw Error("Unrecognized node type: " + childNodeInfo.node.type);
    }

    // Layout each child's annotations
    layoutAnnotation(childNodeInfo);
  });
}

/**
 * Calculate layout for a graph using dagre
 * @param graph the graph to be laid out
 * @param params layout parameters
 * @return width and height of the core graph
 */
function dagreLayout(graph: graphlib.Graph<any, any>, params)
    : {height: number, width: number} {
  _.extend(graph.graph(), {
      nodeSep: params.nodeSep,
      rankSep: params.rankSep
    });

  let bridgeNodeNames = [];
  let nonBridgeNodeNames = [];

  // Split out nodes into bridge and non-bridge nodes, and calculate the total
  // width we should use for bridge nodes.
  _.each(graph.nodes(), nodeName => {
    let nodeInfo = graph.node(nodeName);
    if (nodeInfo.node.type === NodeType.BRIDGE) {
      bridgeNodeNames.push(nodeName);
    } else {
      nonBridgeNodeNames.push(nodeName);
    }
  });

  // If there are no non-bridge nodes, then the graph has zero size.
  if (!nonBridgeNodeNames.length) {
    return {
      width: 0,
      height: 0,
    };
  }

  dagre.layout(graph);

  let graphLabel = graph.graph();

  // Calculate the true bounding box of the graph by iterating over nodes and
  // edges rather than accepting dagre's word for it. In particular, we should
  // ignore the extra-wide bridge nodes and bridge edges, and allow for
  // annotation boxes and labels.
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  _.each(nonBridgeNodeNames, nodeName => {
    let nodeInfo = graph.node(nodeName);
    let w = 0.5 * nodeInfo.width;
    let x1 = nodeInfo.x - w - nodeInfo.inboxWidth;
    let x2 = nodeInfo.x + w + nodeInfo.outboxWidth;
    minX = x1 < minX ? x1 : minX;
    maxX = x2 > maxX ? x2 : maxX;
    let labelLength =
        nodeName.length - nodeName.lastIndexOf(NAMESPACE_DELIM);
    // TODO(jimbo): Account for font width rather than using a magic number.
    let charWidth = 3; // 3 pixels per character.
    let lw = 0.5 * labelLength * charWidth;
    let lx1 = nodeInfo.x - lw;
    let lx2 = nodeInfo.x + lw;
    minX = lx1 < minX ? lx1 : minX;
    maxX = lx2 > maxX ? lx2 : maxX;
    // TODO(jimbo): Account for the height of labels above op nodes here.
    let h = 0.5 * nodeInfo.outerHeight;
    let y1 = nodeInfo.y - h;
    let y2 = nodeInfo.y + h;
    minY = y1 < minY ? y1 : minY;
    maxY = y2 > maxY ? y2 : maxY;
  });
  _.each(graph.edges(), edgeObj => {
    let renderMetaedgeInfo = graph.edge(edgeObj);
    if (renderMetaedgeInfo.structural) {
      return; // Skip structural edges from min/max calculations.
    }
    _.each(renderMetaedgeInfo.points,
      (point: { x: number, y: number }) => {
        minX = point.x < minX ? point.x : minX;
        maxX = point.x > maxX ? point.x : maxX;
        minY = point.y < minY ? point.y : minY;
        maxY = point.y > maxY ? point.y : maxY;
      });
  });

  // Shift all nodes and edge points to account for the left-padding amount,
  // and the invisble bridge nodes.
  _.each(graph.nodes(), nodeName => {
    let nodeInfo = graph.node(nodeName);
    nodeInfo.x -= minX;
    nodeInfo.y -= minY;
  });
  _.each(graph.edges(), edgeObj => {
    _.each(graph.edge(edgeObj).points,
      (point: { x: number, y: number }) => {
        point.x -= minX;
        point.y -= minY;
      });
  });

  return {
    width: maxX - minX,
    height: maxY - minY,
  };
}

/** Layout a metanode. */
function layoutMetanode(renderNodeInfo): void {
  // First, copy params specific to meta nodes onto this render info object.
  let params = PARAMS.subscene.meta;
  renderNodeInfo = _.extend(renderNodeInfo, params);

  // Invoke dagre.layout() on the core graph and record the bounding box
  // dimensions.
  _.extend(renderNodeInfo.coreBox,
      dagreLayout(renderNodeInfo.coreGraph, PARAMS.graph.meta));

  // Calculate the position of nodes in isolatedInExtract relative to the
  // top-left corner of inExtractBox (the bounding box for all inExtract nodes)
  // and calculate the size of the inExtractBox.
  let hasInExtract = renderNodeInfo.isolatedInExtract.length > 0;

  renderNodeInfo.inExtractBox.width = hasInExtract ?
    _(renderNodeInfo.isolatedInExtract).pluck("outerWidth").max() : 0;

  renderNodeInfo.inExtractBox.height =
    _.reduce(renderNodeInfo.isolatedInExtract, (height, child: any, i) => {
      let yOffset = i > 0 ? params.extractYOffset : 0;
      // use outerWidth/Height here to avoid overlaps between extracts
      child.x = renderNodeInfo.inExtractBox.width / 2;
      child.y = height + yOffset + child.outerHeight / 2;
      return height + yOffset + child.outerHeight;
    }, 0);

  // Calculate the position of nodes in isolatedOutExtract relative to the
  // top-left corner of outExtractBox (the bounding box for all outExtract
  // nodes) and calculate the size of the outExtractBox.
  let hasOutExtract = renderNodeInfo.isolatedOutExtract.length > 0;
  renderNodeInfo.outExtractBox.width = hasOutExtract ?
    _(renderNodeInfo.isolatedOutExtract).pluck("outerWidth").max() : 0;

  renderNodeInfo.outExtractBox.height =
    _.reduce(renderNodeInfo.isolatedOutExtract, (height, child: any, i) => {
      let yOffset = i > 0 ? params.extractYOffset : 0;
      // use outerWidth/Height here to avoid overlaps between extracts
      child.x = renderNodeInfo.outExtractBox.width / 2;
      child.y = height + yOffset + child.outerHeight / 2;
      return height + yOffset + child.outerHeight;
    }, 0);

  // Determine the whole metanode's width (from left to right).
  renderNodeInfo.width =
    params.paddingLeft + renderNodeInfo.coreBox.width + params.paddingRight +
    (hasInExtract ?
      renderNodeInfo.inExtractBox.width + params.extractXOffset : 0) +
    (hasOutExtract ?
      params.extractXOffset + renderNodeInfo.outExtractBox.width : 0);

  // TODO(jimbo): Remove labelHeight and instead incorporate into box sizes.
  // Determine the whole metanode's height (from top to bottom).
  renderNodeInfo.height =
    renderNodeInfo.labelHeight +
    params.paddingTop +
    Math.max(
        renderNodeInfo.inExtractBox.height,
        renderNodeInfo.coreBox.height,
        renderNodeInfo.outExtractBox.height
    ) +
    params.paddingBottom;
}

/**
 * Calculate layout for series node's core graph. Only called for an expanded
 * series.
 */
function layoutSeriesNode(node: render.RenderGroupNodeInformation): void {
  let graph = node.coreGraph;

  let params = PARAMS.subscene.series;
  _.extend(node, params);

  // Layout the core.
  _.extend(node.coreBox,
      dagreLayout(node.coreGraph, PARAMS.graph.series));

  _.each(graph.nodes(), nodeName => {
    graph.node(nodeName).excluded = false;
  });

  // Series do not have in/outExtractBox so no need to include them here.
  node.width = node.coreBox.width + params.paddingLeft + params.paddingRight;
  node.height = node.coreBox.height + params.paddingTop + params.paddingBottom;
}

/**
 * Calculate layout for annotations of a given node.
 * This will modify positions of the the given node and its annotations.
 *
 * @see tf.graph.render.Node and tf.graph.render.Annotation
 * for description of each property of each render node.
 *
 */
 function layoutAnnotation(renderNodeInfo: render.RenderNodeInformation): void {
  // If the render node is an expanded metanode, then its annotations will not
  // be visible and we should skip the annotation calculations.
  if (renderNodeInfo.expanded) {
    _.extend(renderNodeInfo, {
      inboxWidth: 0,
      inboxHeight: 0,
      outboxWidth: 0,
      outboxHeight: 0,
      outerWidth: renderNodeInfo.width,
      outerHeight: renderNodeInfo.height
    });
    return;
  }

  let inAnnotations = renderNodeInfo.inAnnotations.list;
  let outAnnotations = renderNodeInfo.outAnnotations.list;

  // Calculate size for in-annotations
  _.each(inAnnotations, a => sizeAnnotation(a));

  // Calculate size for out-annotations
  _.each(outAnnotations, a => sizeAnnotation(a));

  let params = PARAMS.annotations;
  renderNodeInfo.inboxWidth =
    inAnnotations.length > 0 ?
      (<any>_(inAnnotations).pluck("width").max()) +
          params.xOffset + params.labelWidth + params.labelOffset :
      0;

  renderNodeInfo.outboxWidth =
    outAnnotations.length > 0 ?
      (<any>_(outAnnotations).pluck("width").max()) +
          params.xOffset + params.labelWidth + params.labelOffset :
      0;

  // Calculate annotation node position (a.dx, a.dy)
  // and total height for in-annotations
  // After this chunk of code:
  // inboxHeight = sum of annotation heights+ (annotation.length - 1 * yOffset)
  let inboxHeight = _.reduce(inAnnotations,
      (height, a: any, i) => {
        let yOffset = i > 0 ? params.yOffset : 0;
        a.dx = -(renderNodeInfo.width + a.width) / 2 - params.xOffset;
        a.dy = height + yOffset + a.height / 2;
        return height + yOffset + a.height;
      }, 0);

  _.each(inAnnotations, (a: any) => {
    a.dy -= inboxHeight / 2;

    a.labelOffset = params.labelOffset;
  });

  // Calculate annotation node position position (a.dx, a.dy)
  // and total height for out-annotations
  // After this chunk of code:
  // outboxHeight = sum of annotation heights +
  //                (annotation.length - 1 * yOffset)
  let outboxHeight = _.reduce(outAnnotations,
      (height, a: any, i) => {
        let yOffset = i > 0 ? params.yOffset : 0;
        a.dx = (renderNodeInfo.width + a.width) / 2 + params.xOffset;
        a.dy = height + yOffset + a.height / 2;
        return height + yOffset + a.height;
      }, 0);

  _.each(outAnnotations, (a: any) => {
    // adjust by (half of ) the total height
    // so dy is relative to the host node's center.
    a.dy -= outboxHeight / 2;

    a.labelOffset = params.labelOffset;
  });

  // Creating scales for touch point between the in-annotation edges
  // and their hosts.

  let inTouchHeight =
      Math.min(renderNodeInfo.height / 2 - renderNodeInfo.radius,
          inboxHeight / 2);
  inTouchHeight = inTouchHeight < 0 ? 0 : inTouchHeight;

  let inY = d3.scale.linear()
    .domain([0, inAnnotations.length - 1])
    .range([-inTouchHeight, inTouchHeight]);

  // Calculate annotation edge position
  _.each(inAnnotations, (a: any, i) => {
    a.points = [
      // The annotation node end
      {
        dx: a.dx + a.width / 2,
        dy: a.dy
      },

      // The host node end
      {
        dx: - renderNodeInfo.width / 2,
        // only use scale if there are more than one,
        // otherwise center it vertically
        dy: inAnnotations.length > 1 ? inY(i) : 0
      }
    ];
  });

  // Creating scales for touch point between the out-annotation edges
  // and their hosts.
  let outTouchHeight =
      Math.min(renderNodeInfo.height / 2 - renderNodeInfo.radius,
          outboxHeight / 2);
  outTouchHeight = outTouchHeight < 0 ? 0 : outTouchHeight;
  let outY = d3.scale.linear()
    .domain([0, outAnnotations.length - 1])
    .range([-outTouchHeight, outTouchHeight]);

  _.each(outAnnotations, (a: any, i) => {
    // Add point from the border of the annotation node
    a.points = [
      // The host node end
      {
        dx: renderNodeInfo.width / 2,
        // only use scale if there are more than one,
        // otherwise center it vertically
        dy: outAnnotations.length > 1 ? outY(i) : 0
      },
      // The annotation node end
      {
        dx: a.dx - a.width / 2,
        dy: a.dy
      }
    ];
  });

  renderNodeInfo.outerWidth = renderNodeInfo.width + renderNodeInfo.inboxWidth +
      renderNodeInfo.outboxWidth;
  renderNodeInfo.outerHeight =
      Math.max(renderNodeInfo.height, inboxHeight, outboxHeight);
}

/**
 * Set size of an annotation node.
 */
function sizeAnnotation(a: render.Annotation): void {
  switch (a.annotationType) {
    case render.AnnotationType.CONSTANT:
      _.extend(a, PARAMS.constant.size);
      break;
    case render.AnnotationType.SHORTCUT:
      if (a.node.type === NodeType.OP) {
        _.extend(a, PARAMS.shortcutSize.op);
      } else if (a.node.type === NodeType.META) {
        _.extend(a, PARAMS.shortcutSize.meta);
      } else if (a.node.type === NodeType.SERIES) {
        _.extend(a, PARAMS.shortcutSize.series);
      } else {
        throw Error("Invalid node type: " + a.node.type);
      }
      break;
    case render.AnnotationType.SUMMARY:
      _.extend(a, PARAMS.constant.size);
      break;
  }
}

} // close module
