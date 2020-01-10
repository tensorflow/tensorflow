/// <reference path="../graph.ts" />
/// <reference path="../render.ts" />
/// <reference path="scene.ts" />

module tf.graph.scene.edge {

let Scene = tf.graph.scene; // Aliased

export function getEdgeKey(edgeObj) {
  return edgeObj.v + tf.graph.EDGE_KEY_DELIM + edgeObj.w;
}

/**
 * Select or Create a "g.edges" group to a given sceneGroup
 * and builds a number of "g.edge" groups inside the group.
 *
 * Structure Pattern:
 *
 * <g class="edges">
 *   <g class="edge">
 *     <path class="edgeline"/>
 *   </g>
 *   ...
 * </g>
 *
 *
 * @param sceneGroup container
 * @param graph
 * @param sceneBehavior Parent scene module.
 * @return selection of the created nodeGroups
 */
export function buildGroup(sceneGroup,
  graph: graphlib.Graph<tf.graph.render.RenderNodeInformation,
    tf.graph.render.RenderMetaedgeInformation>, sceneBehavior) {
  let edgeData = _.reduce(graph.edges(), (edges, edgeObj) => {
    let edgeLabel = graph.edge(edgeObj);
    edges.push({
      v: edgeObj.v,
      w: edgeObj.w,
      label: edgeLabel
    });
    return edges;
  }, []);

  let container = scene.selectOrCreateChild(sceneGroup, "g",
     Class.Edge.CONTAINER);
  let containerNode = container.node();

  // Select all children and join with data.
  // (Note that all children of g.edges are g.edge)
  let edgeGroups = container.selectAll(function() {
    // using d3's selector function
    // See https://github.com/mbostock/d3/releases/tag/v2.0.0
    // (It's not listed in the d3 wiki.)
    return this.childNodes;
  })
    .data(edgeData, getEdgeKey);

  // Make edges a group to support rendering multiple lines for metaedge
  edgeGroups.enter()
    .append("g")
    .attr("class", Class.Edge.GROUP)
    .attr("data-edge", getEdgeKey)
    .each(function(d) {
      let edgeGroup = d3.select(this);
      d.label.edgeGroup = edgeGroup;
      // index node group for quick highlighting
      sceneBehavior._edgeGroupIndex[getEdgeKey(d)] = edgeGroup;

      // If any edges are reference edges, add the reference edge class.
      let extraEdgeClass = d.label.metaedge && d.label.metaedge.numRefEdges
        ? Class.Edge.REF_LINE + " " + Class.Edge.LINE
        : undefined;
      // Add line during enter because we're assuming that type of line
      // normally does not change.
      appendEdge(edgeGroup, d, scene, extraEdgeClass);
    });

  edgeGroups.each(position);
  edgeGroups.each(function(d) {
    stylize(d3.select(this), d, sceneBehavior);
  });

  edgeGroups.exit()
    .each(d => {
      delete sceneBehavior._edgeGroupIndex[getEdgeKey(d)];
    })
    .remove();
  return edgeGroups;
};

/**
 * For a given d3 selection and data object, create a path to represent the
 * edge described in d.label.
 *
 * If d.label is defined, it will be a RenderMetaedgeInformation instance. It
 * will sometimes be undefined, for example for some Annotation edges for which
 * there is no underlying Metaedge in the hierarchical graph.
 */
export function appendEdge(edgeGroup, d, sceneBehavior, edgeClass?) {
  edgeClass = edgeClass || Class.Edge.LINE; // set default type

  if (d.label && d.label.structural) {
    edgeClass += " " + Class.Edge.STRUCTURAL;
  }

  edgeGroup.append("path")
    .attr("class", edgeClass);
};

/**
 * Returns a tween interpolator for the endpoint of an edge path.
 */
function getEdgePathInterpolator(d, i, a) {
  let renderMetaedgeInfo = d.label;
  let adjoiningMetaedge = renderMetaedgeInfo.adjoiningMetaedge;
  if (!adjoiningMetaedge) {
    return d3.interpolate(a, interpolate(renderMetaedgeInfo.points));
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
    let points = renderMetaedgeInfo.points;
    let index = inbound ? 0 : points.length - 1;
    points[index].x = adjoiningPoint.x;
    points[index].y = adjoiningPoint.y;
    let dPath = interpolate(points);
    return dPath;
  };
}

export let interpolate = d3.svg.line()
  .interpolate("basis")
  .x((d: any) => { return d.x; })
  .y((d: any) => { return d.y; });

function position(d) {
  d3.select(this).select("path." + Class.Edge.LINE)
    .each(function(d) {
      let path = d3.select(this);
      path.transition().attrTween("d", getEdgePathInterpolator);
    });
};

/**
 * For a given d3 selection and data object, mark the edge as a control
 * dependency if it contains only control edges.
 *
 * d's label property will be a RenderMetaedgeInformation object.
 */
function stylize(edgeGroup, d, stylize) {
  let a;
  let metaedge = d.label.metaedge;
  edgeGroup
    .select("path." + Class.Edge.LINE)
    .classed("control-dep", metaedge && !metaedge.numRegularEdges);
};

} // close module
