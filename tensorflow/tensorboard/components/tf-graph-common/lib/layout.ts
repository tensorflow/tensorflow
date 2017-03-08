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
      nodeSep: 5,
      /**
       * Dagre's ranksep param - number of pixels
       * between each rank in the layout.
       *
       * See https://github.com/cpettitt/dagre/wiki#configuring-the-layout
       */
      rankSep: 25,
      /**
       * Dagre's edgesep param - number of pixels that separate
       * edges horizontally in the layout.
       */
      edgeSep: 5,
    },
    /** Graph parameter for metanode. */
    series: {
      /**
       * Dagre's nodesep param - number of pixels that
       * separate nodes horizontally in the layout.
       *
       * See https://github.com/cpettitt/dagre/wiki#configuring-the-layout
       */
      nodeSep: 5,
      /**
       * Dagre's ranksep param - number of pixels
       * between each rank in the layout.
       *
       * See https://github.com/cpettitt/dagre/wiki#configuring-the-layout
       */
      rankSep: 25,
      /**
       * Dagre's edgesep param - number of pixels that separate
       * edges horizontally in the layout.
       */
      edgeSep: 5
    },
    /**
     * Padding is used to correctly position the graph SVG inside of its parent
     * element. The padding amounts are applied using an SVG transform of X and
     * Y coordinates.
     */
    padding: {paddingTop: 40, paddingLeft: 20}
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
      extractXOffset: 15,
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
      maxLabelWidth: 52,
      /** A scale for the node's height based on number of nodes inside */
      height: d3.scale.linear().domain([1, 200]).range([15, 60]).clamp(true),
      /** The radius of the circle denoting the expand button. */
      expandButtonRadius: 3
    },
    /** Size of op nodes. */
    op: {
      width: 15,
      height: 6,
      radius: 3,  // for making annotation touching ellipse
      labelOffset: -8,
      maxLabelWidth: 30
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
        radius: 10,  // Forces annotations to center line.
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
    op: {width: 10, height: 4},
    /** Size of shortcuts for meta nodes */
    meta: {width: 12, height: 4, radius: 1},
    /** Size of shortcuts for series nodes */
    series: {
      width: 14,
      height: 4,
    }
  },
  annotations: {
    /** Maximum possible width of the bounding box for in annotations */
    inboxWidth: 50,
    /** Maximum possible width of the bounding box for out annotations */
    outboxWidth: 50,
    /** X-space between the shape and each annotation-node. */
    xOffset: 10,
    /** Y-space between each annotation-node. */
    yOffset: 3,
    /** X-space between each annotation-node and its label. */
    labelOffset: 2,
    /** Defines the max width for annotation label */
    maxLabelWidth: 120
  },
  constant: {size: {width: 4, height: 4}},
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
  minimap: {
    /** The maximum width/height the minimap can have. */
    size: 150
  }
};

/** Calculate layout for a scene of a group node. */
export function layoutScene(renderNodeInfo: render.RenderGroupNodeInfo): void {
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
 * Updates the total width of an unexpanded node which includes the size of its
 * in and out annotations.
 */
function updateTotalWidthOfNode(renderInfo: render.RenderNodeInfo): void {
  renderInfo.inboxWidth = renderInfo.inAnnotations.list.length > 0 ?
      PARAMS.annotations.inboxWidth : 0;
  renderInfo.outboxWidth = renderInfo.outAnnotations.list.length > 0 ?
      PARAMS.annotations.outboxWidth : 0;
  // Assign the width of the core box (the main shape of the node).
  renderInfo.coreBox.width = renderInfo.width;
  renderInfo.coreBox.height = renderInfo.height;
  // TODO(jimbo): Account for font width rather than using a magic number.
  let labelLength = renderInfo.node.name.length -
      renderInfo.node.name.lastIndexOf(NAMESPACE_DELIM) - 1;
  let charWidth = 3; // 3 pixels per character.
  // Compute the total width of the node.
  renderInfo.width = Math.max(renderInfo.coreBox.width +
      renderInfo.inboxWidth + renderInfo.outboxWidth,
      labelLength * charWidth);

}

/**
 * Update layout, size, and annotations of its children nodes and edges.
 */
function layoutChildren(renderNodeInfo: render.RenderGroupNodeInfo): void {
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
          // Set fixed width and scalable height based on cardinality
          _.extend(childNodeInfo, PARAMS.nodeSize.meta);
          childNodeInfo.height =
              PARAMS.nodeSize.meta.height(childNodeInfo.node.cardinality);
        } else {
          let childGroupNodeInfo =
            <render.RenderGroupNodeInfo>childNodeInfo;
          layoutScene(childGroupNodeInfo); // Recursively layout its subscene.
        }
        break;
      case NodeType.SERIES:
        if (childNodeInfo.expanded) {
          _.extend(childNodeInfo, PARAMS.nodeSize.series.expanded);
          let childGroupNodeInfo =
            <render.RenderGroupNodeInfo>childNodeInfo;
          layoutScene(childGroupNodeInfo); // Recursively layout its subscene.
        } else {
          let childGroupNodeInfo =
            <render.RenderGroupNodeInfo>childNodeInfo;
          let seriesParams =
            childGroupNodeInfo.node.hasNonControlEdges ?
              PARAMS.nodeSize.series.vertical :
              PARAMS.nodeSize.series.horizontal;
          _.extend(childNodeInfo, seriesParams);
        }
        break;
      default:
        throw Error('Unrecognized node type: ' + childNodeInfo.node.type);
    }
    // Compute total width of un-expanded nodes. Width of expanded nodes
    // has already been computed.
    if (!childNodeInfo.expanded) {
      updateTotalWidthOfNode(childNodeInfo);
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
function dagreLayout(
    graph: graphlib.Graph<render.RenderNodeInfo, render.RenderMetaedgeInfo>,
    params): {height: number, width: number} {
  _.extend(graph.graph(), {
    nodesep: params.nodeSep,
    ranksep: params.rankSep,
    edgesep: params.edgeSep
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
    let x1 = nodeInfo.x - w;
    let x2 = nodeInfo.x + w;
    minX = x1 < minX ? x1 : minX;
    maxX = x2 > maxX ? x2 : maxX;
    // TODO(jimbo): Account for the height of labels above op nodes here.
    let h = 0.5 * nodeInfo.height;
    let y1 = nodeInfo.y - h;
    let y2 = nodeInfo.y + h;
    minY = y1 < minY ? y1 : minY;
    maxY = y2 > maxY ? y2 : maxY;
  });
  _.each(graph.edges(), edgeObj => {
    let edgeInfo = graph.edge(edgeObj);
    if (edgeInfo.structural) {
      return; // Skip structural edges from min/max calculations.
    }

    // Since the node size passed to dagre includes the in and out
    // annotations, the endpoints of the edge produced by dagre may not
    // point to the actual node shape (rectangle, ellipse). We correct the
    // end-points by finding the intersection of a line between the
    // next-to-last (next-to-first) point and the destination (source)
    // rectangle.
    let sourceNode = graph.node(edgeInfo.metaedge.v);
    let destNode = graph.node(edgeInfo.metaedge.w);

    // Straight 3-points edges are special case, since they are curved after
    // our default correction. To keep them straight, we remove the mid point
    // and correct the first and the last point to be the center of the
    // source and destination node respectively.
    if (edgeInfo.points.length === 3 && isStraightLine(edgeInfo.points)) {
      if (sourceNode != null) {
        let cxSource = sourceNode.expanded ?
            sourceNode.x : computeCXPositionOfNodeShape(sourceNode);
        edgeInfo.points[0].x = cxSource;
      }
      if (destNode != null) {
        let cxDest = destNode.expanded ?
            destNode.x : computeCXPositionOfNodeShape(destNode);
        edgeInfo.points[2].x = cxDest;
      }
      // Remove the middle point so the edge doesn't curve.
      edgeInfo.points = [edgeInfo.points[0], edgeInfo.points[1]];
    }
    // Correct the destination endpoint of the edge.
    let nextToLastPoint = edgeInfo.points[edgeInfo.points.length - 2];
    // The destination node might be null if this is a bridge edge.
    if (destNode != null) {
      edgeInfo.points[edgeInfo.points.length - 1] =
          intersectPointAndNode(nextToLastPoint, destNode);
    }
    // Correct the source endpoint of the edge.
    let secondPoint = edgeInfo.points[1];
    // The source might be null if this is a bridge edge.
    if (sourceNode != null) {
      edgeInfo.points[0] = intersectPointAndNode(secondPoint, sourceNode);
    }

    _.each(edgeInfo.points, (point: render.Point) => {
        minX = point.x < minX ? point.x : minX;
        maxX = point.x > maxX ? point.x : maxX;
        minY = point.y < minY ? point.y : minY;
        maxY = point.y > maxY ? point.y : maxY;
      });
  });

  // Shift all nodes and edge points to account for the left-padding amount,
  // and the invisible bridge nodes.
  _.each(graph.nodes(), nodeName => {
    let nodeInfo = graph.node(nodeName);
    nodeInfo.x -= minX;
    nodeInfo.y -= minY;
  });
  _.each(graph.edges(), edgeObj => {
    _.each(graph.edge(edgeObj).points, (point: render.Point) => {
        point.x -= minX;
        point.y -= minY;
      });
  });

  return {
    width: maxX - minX,
    height: maxY - minY
  };
}

/** Layout a metanode. Only called for an expanded node. */
function layoutMetanode(renderNodeInfo: render.RenderGroupNodeInfo): void {
  // First, copy params specific to meta nodes onto this render info object.
  let params = PARAMS.subscene.meta;
  _.extend(renderNodeInfo, params);
  // Invoke dagre.layout() on the core graph and record the bounding box
  // dimensions.
  _.extend(renderNodeInfo.coreBox,
      dagreLayout(renderNodeInfo.coreGraph, PARAMS.graph.meta));

  // Calculate the position of nodes in isolatedInExtract relative to the
  // top-left corner of inExtractBox (the bounding box for all inExtract nodes)
  // and calculate the size of the inExtractBox.
  let maxInExtractWidth = _.max(renderNodeInfo.isolatedInExtract,
      renderNode => renderNode.width).width;
  renderNodeInfo.inExtractBox.width = maxInExtractWidth != null ?
      maxInExtractWidth : 0;

  renderNodeInfo.inExtractBox.height =
    _.reduce(renderNodeInfo.isolatedInExtract, (height, child, i) => {
      let yOffset = i > 0 ? params.extractYOffset : 0;
      // use width/height here to avoid overlaps between extracts
      child.x = 0;
      child.y = height + yOffset + child.height / 2;
      return height + yOffset + child.height;
    }, 0);

  // Calculate the position of nodes in isolatedOutExtract relative to the
  // top-left corner of outExtractBox (the bounding box for all outExtract
  // nodes) and calculate the size of the outExtractBox.
  let maxOutExtractWidth = _.max(renderNodeInfo.isolatedOutExtract,
      renderNode => renderNode.width).width;
  renderNodeInfo.outExtractBox.width = maxOutExtractWidth != null ?
      maxOutExtractWidth : 0;

  renderNodeInfo.outExtractBox.height =
    _.reduce(renderNodeInfo.isolatedOutExtract, (height, child, i) => {
      let yOffset = i > 0 ? params.extractYOffset : 0;
      // use width/height here to avoid overlaps between extracts
      child.x = 0;
      child.y = height + yOffset + child.height / 2;
      return height + yOffset + child.height;
    }, 0);

  // Compute the total padding between the core graph, in-extract and
  // out-extract boxes.
  let numParts = 0;
  if (renderNodeInfo.isolatedInExtract.length > 0) {
    numParts++;
  }
  if (renderNodeInfo.isolatedOutExtract.length > 0) {
    numParts++;
  }
  if (renderNodeInfo.coreGraph.nodeCount() > 0) {
    numParts++;
  }
  let offset = PARAMS.subscene.meta.extractXOffset;
  let padding = numParts <= 1 ? 0 : (numParts  <= 2 ? offset : 2 * offset);

  // Add the in-extract and out-extract width to the core box width.
  renderNodeInfo.coreBox.width += renderNodeInfo.inExtractBox.width +
      renderNodeInfo.outExtractBox.width + padding;
  renderNodeInfo.coreBox.height =
    params.labelHeight +
    Math.max(
      renderNodeInfo.inExtractBox.height,
      renderNodeInfo.coreBox.height,
      renderNodeInfo.outExtractBox.height
  );
  // Determine the whole metanode's width (from left to right).
  renderNodeInfo.width = renderNodeInfo.coreBox.width +
      params.paddingLeft + params.paddingRight;

  // Determine the whole metanode's height (from top to bottom).
  renderNodeInfo.height =
      renderNodeInfo.paddingTop +
      renderNodeInfo.coreBox.height +
      renderNodeInfo.paddingBottom;
}

/**
 * Calculate layout for series node's core graph. Only called for an expanded
 * series.
 */
function layoutSeriesNode(node: render.RenderGroupNodeInfo): void {
  let graph = node.coreGraph;

  let params = PARAMS.subscene.series;
  _.extend(node, params);

  // Layout the core.
  _.extend(node.coreBox, dagreLayout(node.coreGraph, PARAMS.graph.series));

  _.each(graph.nodes(), nodeName => {
    graph.node(nodeName).excluded = false;
  });

  // Series do not have in/outExtractBox so no need to include them here.
  node.width = node.coreBox.width + params.paddingLeft + params.paddingRight;
  node.height = node.coreBox.height + params.paddingTop + params.paddingBottom;
}

/**
 * Calculate layout for annotations of a given node.
 * This will modify positions of the given node and its annotations.
 *
 * @see tf.graph.render.Node and tf.graph.render.Annotation
 * for description of each property of each render node.
 *
 */
function layoutAnnotation(renderNodeInfo: render.RenderNodeInfo): void {
  // If the render node is an expanded metanode, then its annotations will not
  // be visible and we should skip the annotation calculations.
  if (renderNodeInfo.expanded) {
    return;
  }

  let inAnnotations = renderNodeInfo.inAnnotations.list;
  let outAnnotations = renderNodeInfo.outAnnotations.list;

  // Calculate size for in-annotations
  _.each(inAnnotations, a => sizeAnnotation(a));

  // Calculate size for out-annotations
  _.each(outAnnotations, a => sizeAnnotation(a));

  let params = PARAMS.annotations;

  // Calculate annotation node position (a.dx, a.dy)
  // and total height for in-annotations
  // After this chunk of code:
  // inboxHeight = sum of annotation heights+ (annotation.length - 1 * yOffset)
  let inboxHeight = _.reduce(inAnnotations,
      (height, a, i) => {
        let yOffset = i > 0 ? params.yOffset : 0;
        a.dx = -(renderNodeInfo.coreBox.width + a.width) / 2 - params.xOffset;
        a.dy = height + yOffset + a.height / 2;
        return height + yOffset + a.height;
      }, 0);

  _.each(inAnnotations, a => {
    a.dy -= inboxHeight / 2;

    a.labelOffset = params.labelOffset;
  });

  // Calculate annotation node position (a.dx, a.dy)
  // and total height for out-annotations
  // After this chunk of code:
  // outboxHeight = sum of annotation heights +
  //                (annotation.length - 1 * yOffset)
  let outboxHeight = _.reduce(outAnnotations,
      (height, a, i) => {
        let yOffset = i > 0 ? params.yOffset : 0;
        a.dx = (renderNodeInfo.coreBox.width + a.width) / 2 + params.xOffset;
        a.dy = height + yOffset + a.height / 2;
        return height + yOffset + a.height;
      }, 0);

  _.each(outAnnotations, a => {
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
  _.each(inAnnotations, (a, i) => {
    a.points = [
      // The annotation node end
      {
        dx: a.dx + a.width / 2,
        dy: a.dy
      },

      // The host node end
      {
        dx: - renderNodeInfo.coreBox.width / 2,
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

  _.each(outAnnotations, (a, i) => {
    // Add point from the border of the annotation node
    a.points = [
      // The host node end
      {
        dx: renderNodeInfo.coreBox.width / 2,
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

  renderNodeInfo.height =
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
        throw Error('Invalid node type: ' + a.node.type);
      }
      break;
    case render.AnnotationType.SUMMARY:
      _.extend(a, PARAMS.constant.size);
      break;
  }
}

/**
 * Determines the center position of the node's shape. The position depends
 * on if the node has in and out-annotations.
 */
export function computeCXPositionOfNodeShape(renderInfo: render.RenderNodeInfo):
    number {
  if (renderInfo.expanded) {
    return renderInfo.x;
  }
  let dx = renderInfo.inAnnotations.list.length ? renderInfo.inboxWidth : 0;
  return renderInfo.x - renderInfo.width / 2 + dx +
      renderInfo.coreBox.width / 2;
}

/** Returns the angle (in degrees) between two points. */
function angleBetweenTwoPoints(a: render.Point, b: render.Point): number {
  let dx = b.x - a.x;
  let dy = b.y - a.y;
  return 180 * Math.atan(dy / dx) / Math.PI;
}

/**
 * Returns if a line going through the specified points is a straight line.
 */
function isStraightLine(points: render.Point[]) {
  let angle = angleBetweenTwoPoints(points[0], points[1]);
  for (let i = 1; i < points.length - 1; i++) {
    let newAngle = angleBetweenTwoPoints(points[i], points[i + 1]);
    // Have a tolerance of 1 degree.
    if (Math.abs(newAngle - angle) > 1) {
      return false;
    }
    angle = newAngle;
  }
  return true;
}

/**
 * Returns the intersection of a line between the provided point
 * and the provided rectangle.
 */
function intersectPointAndNode(
    point: render.Point, node: render.RenderNodeInfo): render.Point {
  // cx and cy are the center of the rectangle.
  let cx = node.expanded ?
     node.x : computeCXPositionOfNodeShape(node);
  let cy = node.y;
  // Calculate the slope
  let dx = point.x - cx;
  let dy = point.y - cy;
  let w = node.expanded ? node.width : node.coreBox.width;
  let h = node.expanded ? node.height : node.coreBox.height;
  let deltaX, deltaY;
  if (Math.abs(dy) * w / 2  > Math.abs(dx) * h / 2) {
    // The intersection is above or below the rectangle.
    if (dy < 0) {
      h = -h;
    }
    deltaX = dy === 0 ? 0 : h / 2 * dx / dy;
    deltaY = h / 2;
  } else {
    // The intersection is left or right of the rectangle.
    if (dx < 0) {
      w = -w;
    }
    deltaX = w / 2;
    deltaY = dx === 0 ? 0 : w / 2 * dy / dx;
  }
  return {x: cx + deltaX, y: cy + deltaY};
}

} // close module
