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
module tf.graph.scene.edge {

/** Delimiter between dimensions when showing sizes of tensors. */
const TENSOR_SHAPE_DELIM = '×';

/** The minimum stroke width of an edge. */
export const MIN_EDGE_WIDTH = 0.75;

/** The maximum stroke width of an edge. */
export const MAX_EDGE_WIDTH = 12;

/** The exponent used in the power scale for edge thickness. */
const EDGE_WIDTH_SCALE_EXPONENT = 0.3;

/** The domain (min and max value) for the edge width. */
const DOMAIN_EDGE_WIDTH_SCALE = [1, 5E6];

export const EDGE_WIDTH_SCALE = d3.scale.pow()
      .exponent(EDGE_WIDTH_SCALE_EXPONENT)
      .domain(DOMAIN_EDGE_WIDTH_SCALE)
      .range([MIN_EDGE_WIDTH, MAX_EDGE_WIDTH])
      .clamp(true);

let arrowheadMap =
    d3.scale.quantize().domain([MIN_EDGE_WIDTH, MAX_EDGE_WIDTH]).range([
      'small', 'medium', 'large', 'xlarge'
    ]);

/** Minimum stroke width to put edge labels in the middle of edges */
const CENTER_EDGE_LABEL_MIN_STROKE_WIDTH = 2.5;

export type EdgeData = {v: string, w: string, label: render.RenderMetaedgeInfo};

export function getEdgeKey(edgeObj: EdgeData) {
  return edgeObj.v + EDGE_KEY_DELIM + edgeObj.w;
}

/**
 * Select or Create a 'g.edges' group to a given sceneGroup
 * and builds a number of 'g.edge' groups inside the group.
 *
 * Structure Pattern:
 *
 * <g class='edges'>
 *   <g class='edge'>
 *     <path class='edgeline'/>
 *   </g>
 *   ...
 * </g>
 *
 *
 * @param sceneGroup container
 * @param graph
 * @param sceneElement <tf-graph-scene> polymer element.
 * @return selection of the created nodeGroups
 */
export function buildGroup(sceneGroup,
    graph: graphlib.Graph<render.RenderNodeInfo, render.RenderMetaedgeInfo>,
    sceneElement) {
  let edges: EdgeData[] = [];
  edges = _.reduce(graph.edges(), (edges, edgeObj) => {
    let edgeLabel = graph.edge(edgeObj);
    edges.push({
      v: edgeObj.v,
      w: edgeObj.w,
      label: edgeLabel
    });
    return edges;
  }, edges);

  let container =
      scene.selectOrCreateChild(sceneGroup, 'g', Class.Edge.CONTAINER);

  // Select all children and join with data.
  // (Note that all children of g.edges are g.edge)
  let edgeGroups = container.selectAll(function() {
    // using d3's selector function
    // See https://github.com/mbostock/d3/releases/tag/v2.0.0
    // (It's not listed in the d3 wiki.)
    return this.childNodes;
  }).data(edges, getEdgeKey);

  // Make edges a group to support rendering multiple lines for metaedge
  edgeGroups.enter()
      .append('g')
      .attr('class', Class.Edge.GROUP)
      .attr('data-edge', getEdgeKey)
      .each(function(d: EdgeData) {
        let edgeGroup = d3.select(this);
        d.label.edgeGroup = edgeGroup;
        // index node group for quick highlighting
        sceneElement._edgeGroupIndex[getEdgeKey(d)] = edgeGroup;

        // Add line during enter because we're assuming that type of line
        // normally does not change.
        appendEdge(edgeGroup, d, sceneElement);
      });

  edgeGroups.each(position);
  edgeGroups.each(function(d) {
    stylize(d3.select(this), d, sceneElement);
  });

  edgeGroups.exit()
    .each(d => {
      delete sceneElement._edgeGroupIndex[getEdgeKey(d)];
    })
    .remove();
  return edgeGroups;
};

/**
 * Returns the label for the given base edge.
 * The label is the shape of the underlying tensor.
 */
export function getLabelForBaseEdge(
    baseEdge: BaseEdge, renderInfo: render.RenderGraphInfo): string {
  let node = <OpNode>renderInfo.getNodeByName(baseEdge.v);
  if (node.outputShapes == null || node.outputShapes.length === 0) {
    return null;
  }
  let shape = node.outputShapes[baseEdge.outputTensorIndex];
  if (shape == null) {
    return null;
  }
  if (shape.length === 0) {
    return 'scalar';
  }
  return shape.map(size => { return size === -1 ? '?' : size; })
      .join(TENSOR_SHAPE_DELIM);
}

/**
 * Creates the label for the given metaedge. If the metaedge consists
 * of only 1 tensor, and it's shape is known, the label will contain that
 * shape. Otherwise, the label will say the number of tensors in the metaedge.
 */
export function getLabelForEdge(metaedge: Metaedge,
    renderInfo: render.RenderGraphInfo): string {
  let isMultiEdge = metaedge.baseEdgeList.length > 1;
  return isMultiEdge ?
      metaedge.baseEdgeList.length + ' tensors' :
      getLabelForBaseEdge(metaedge.baseEdgeList[0], renderInfo);
}

/**
 * Shortens the path enought such that the tip of the start/end marker will
 * point to the start/end of the path. The marker can be of arbitrary size.
 *
 * @param points Array of path control points.
 * @param marker D3 selection of the <marker> svg element.
 * @param isStart Is the marker a `start-marker`. If false, the marker is
 *     an `end-marker`.
 * @return The new array of control points.
 */
function adjustPathPointsForMarker(points: render.Point[],
    marker: d3.Selection<any>, isStart: boolean): render.Point[] {
  let lineFunc = d3.svg.line<render.Point>()
    .x(d => d.x)
    .y(d => d.y);
  let path =
      d3.select(document.createElementNS('http://www.w3.org/2000/svg', 'path'))
          .attr('d', lineFunc(points));
  let markerWidth = +marker.attr('markerWidth');
  let viewBox = marker.attr('viewBox').split(' ').map(Number);
  let viewBoxWidth = viewBox[2] - viewBox[0];
  let refX = +marker.attr('refX');
  let pathNode = <SVGPathElement> path.node();
  if (isStart) {
    let fractionStickingOut = refX / viewBoxWidth;
    let length = markerWidth * fractionStickingOut;
    let point = pathNode.getPointAtLength(length);
    // Figure out how many segments of the path we need to remove in order
    // to shorten the path.
    let segIndex = pathNode.getPathSegAtLength(length);
    // Update the very first segment.
    points[segIndex - 1] = {x: point.x, y: point.y};
    // Ignore every point before segIndex - 1.
    return points.slice(segIndex - 1);
  } else {
    let fractionStickingOut = 1 - refX / viewBoxWidth;
    let length = pathNode.getTotalLength() - markerWidth * fractionStickingOut;
    let point = pathNode.getPointAtLength(length);
    // Figure out how many segments of the path we need to remove in order
    // to shorten the path.
    let segIndex = pathNode.getPathSegAtLength(length);
    // Update the very last segment.
    points[segIndex] = {x: point.x, y: point.y};
    // Ignore every point after segIndex.
    return points.slice(0, segIndex + 1);
  }
}

/**
 * For a given d3 selection and data object, create a path to represent the
 * edge described in d.label.
 *
 * If d.label is defined, it will be a RenderMetaedgeInfo instance. It
 * will sometimes be undefined, for example for some Annotation edges for which
 * there is no underlying Metaedge in the hierarchical graph.
 */
export function appendEdge(edgeGroup, d: EdgeData,
    sceneElement: {renderHierarchy: render.RenderGraphInfo},
    edgeClass?: string) {
  let size = 1;
  if (d.label != null && d.label.metaedge != null) {
    // There is an underlying Metaedge.
    size = d.label.metaedge.totalSize;
  }
  edgeClass = edgeClass || Class.Edge.LINE; // set default type

  if (d.label && d.label.structural) {
    edgeClass += ' ' + Class.Edge.STRUCTURAL;
  }
  // Give the path a unique id, which will be used to link
  // the textPath (edge label) to this path.
  let pathId = 'path_' + getEdgeKey(d);
  let strokeWidth = sceneElement.renderHierarchy.edgeWidthScale(size);

  let path = edgeGroup.append('path')
                 .attr({
                   'id': pathId,
                   'class': edgeClass,
                 })
                 .style({'stroke-width': strokeWidth + 'px'});

  // Check if there is a reference edge and add an arrowhead of the right size.
  if (d.label && d.label.metaedge && d.label.metaedge.numRefEdges) {
    let markerId = `ref-arrowhead-${arrowheadMap(strokeWidth)}`;
    path.style('marker-start', `url(#${markerId})`);
    d.label.startMarkerId = markerId;
  }

  if (d.label == null || d.label.metaedge == null) {
    // There is no associated metaedge, thus no text.
    // This happens for annotation edges.
    return;
  }
  let labelForEdge = getLabelForEdge(d.label.metaedge,
      sceneElement.renderHierarchy);
  if (labelForEdge == null) {
    // We have no information to show on this edge.
    return;
  }

  // Put edge label in the middle of edge only if the edge is thick enough.
  let baseline = strokeWidth > CENTER_EDGE_LABEL_MIN_STROKE_WIDTH ?
      'central' :
      'text-after-edge';

  edgeGroup.append('text')
      .append('textPath')
      .attr({
        'xlink:href': '#' + pathId,
        'startOffset': '50%',
        'text-anchor': 'middle',
        'dominant-baseline': 'central'
      })
      .text(labelForEdge);
};

export let interpolate = d3.svg.line<{x: number, y: number}>()
                             .interpolate('basis')
                             .x((d) => { return d.x;})
                             .y((d) => { return d.y;});

/**
 * Returns a tween interpolator for the endpoint of an edge path.
 */
function getEdgePathInterpolator(d: EdgeData, i: number, a: string) {
  let renderMetaedgeInfo = <render.RenderMetaedgeInfo> d.label;
  let adjoiningMetaedge = renderMetaedgeInfo.adjoiningMetaedge;
  let points = renderMetaedgeInfo.points;

  // Adjust the path so that start/end markers point to the end
  // of the path.
  if (d.label.startMarkerId) {
    points = adjustPathPointsForMarker(
        points, d3.select('#' + d.label.startMarkerId), true);
  }
  if (d.label.endMarkerId) {
    points = adjustPathPointsForMarker(
        points, d3.select('#' + d.label.endMarkerId), false);
  }

  if (!adjoiningMetaedge) {
    return d3.interpolate(a, interpolate(points));
  }

  let renderPath = this;

  // Get the adjoining path that matches the adjoining metaedge.
  let adjoiningPath =
    <SVGPathElement>((<HTMLElement>adjoiningMetaedge.edgeGroup.node())
      .firstChild);

  // Find the desired SVGPoint along the adjoining path, then convert those
  // coordinates into the space of the renderPath using its Current
  // Transformation Matrix (CTM).
  let inbound = renderMetaedgeInfo.metaedge.inbound;

  return function(t) {
    let adjoiningPoint = adjoiningPath
      .getPointAtLength(inbound ? adjoiningPath.getTotalLength() : 0)
      .matrixTransform(adjoiningPath.getCTM())
      .matrixTransform(renderPath.getCTM().inverse());

    // Update the relevant point in the renderMetaedgeInfo's points list, then
    // re-interpolate the path.
    let index = inbound ? 0 : points.length - 1;
    points[index].x = adjoiningPoint.x;
    points[index].y = adjoiningPoint.y;
    let dPath = interpolate(points);
    return dPath;
  };
}

function position(d) {
  d3.select(this)
      .select('path.' + Class.Edge.LINE)
      .transition()
      .attrTween('d', getEdgePathInterpolator);
};

/**
 * For a given d3 selection and data object, mark the edge as a control
 * dependency if it contains only control edges.
 *
 * d's label property will be a RenderMetaedgeInfo object.
 */
function stylize(edgeGroup, d: EdgeData, stylize) {
  edgeGroup.classed('faded', d.label.isFadedOut);
  let metaedge = d.label.metaedge;
  edgeGroup.select('path.' + Class.Edge.LINE)
      .classed('control-dep', metaedge && !metaedge.numRegularEdges);
};

} // close module
