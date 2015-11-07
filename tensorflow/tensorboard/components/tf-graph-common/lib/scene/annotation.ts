/// <reference path="../graph.ts" />
/// <reference path="../render.ts" />
/// <reference path="scene.ts" />
/// <reference path="edge.ts" />

module tf.graph.scene.annotation {

/**
 * Populate a given annotation container group
 *
 *     <g class="{in|out}-annotations"></g>
 *
 * with annotation group of the following structure:
 *
 * <g class="annotation">
 *   <g class="annotation-node">
 *   <!--
 *   Content here determined by Scene.node.buildGroup.
 *   -->
 *   </g>
 * </g>
 *
 * @param container selection of the container.
 * @param annotationData node.{in|out}Annotations
 * @param d node to build group for.
 * @param sceneBehavior polymer scene element.
 * @return selection of appended objects
 */
export function buildGroup(container, annotationData: render.AnnotationList,
  d: render.RenderNodeInformation, sceneBehavior) {
  // Select all children and join with data.
  let annotationGroups = container.selectAll(function() {
       // using d3's selector function
       // See https://github.com/mbostock/d3/releases/tag/v2.0.0
       // (It's not listed in the d3 wiki.)
         return this.childNodes;
       })
       .data(annotationData.list, d => { return d.node.name; });

  annotationGroups.enter()
    .append("g")
    .attr("data-name", a => { return a.node.name; })
    .each(function(a) {
      let aGroup = d3.select(this);

      // Add annotation to the index in the scene
      sceneBehavior.addAnnotationGroup(a, d, aGroup);
      // Append annotation edge
      let edgeType = Class.Annotation.EDGE;
      let metaedge = a.renderMetaedgeInfo && a.renderMetaedgeInfo.metaedge;
      if (metaedge && !metaedge.numRegularEdges) {
        edgeType += " " + Class.Annotation.CONTROL_EDGE;
      }
      // If any edges are reference edges, add the reference edge class.
      if (metaedge && metaedge.numRefEdges) {
        edgeType += " " + Class.Edge.REF_LINE;
      }
      edge.appendEdge(aGroup, a, sceneBehavior, edgeType);

      if (a.annotationType !== tf.graph.render.AnnotationType.ELLIPSIS) {
        addAnnotationLabelFromNode(aGroup, a);
        buildShape(aGroup, a, sceneBehavior);
      } else {
        addAnnotationLabel(aGroup, a.node.name, a, Class.Annotation.ELLIPSIS);
      }
    });

  annotationGroups
    .attr("class", a => {
      return Class.Annotation.GROUP + " " +
        annotationToClassName(a.annotationType) +
        " " + node.nodeClass(a);
    })
    .each(function(a) {
      let aGroup = d3.select(this);
      update(aGroup, d, a, sceneBehavior);
      if (a.annotationType !== tf.graph.render.AnnotationType.ELLIPSIS) {
        addInteraction(aGroup, d, sceneBehavior);
      }
    });

  annotationGroups.exit()
    .each(function(a) {
      let aGroup = d3.select(this);

      // Remove annotation from the index in the scene
      sceneBehavior.removeAnnotationGroup(a, d, aGroup);
    })
    .remove();
  return annotationGroups;
};

/**
 * Maps an annotation enum to a class name used in css rules.
 */
function annotationToClassName(annotationType: render.AnnotationType) {
  return (tf.graph.render.AnnotationType[annotationType] || "")
      .toLowerCase() || null;
}

function buildShape(aGroup, a: render.Annotation, sceneBehavior) {
  if (a.annotationType === tf.graph.render.AnnotationType.SUMMARY) {
    let image = scene.selectOrCreateChild(aGroup, "image");
    image.attr({
      "xlink:href": sceneBehavior.resolveUrl("../../lib/svg/summary-icon.svg"),
      "height": "12px",
      "width": "12px",
      "cursor": "pointer"
    });
  } else {
    let shape = node.buildShape(aGroup, a, Class.Annotation.NODE);
    // add title tag to get native tooltips
    scene.selectOrCreateChild(shape, "title").text(a.node.name);
  }
}

function addAnnotationLabelFromNode(aGroup, a: render.Annotation) {
  let namePath = a.node.name.split("/");
  let text = namePath[namePath.length - 1];
  let shortenedText = text.length > 8 ? text.substring(0, 8) + "..." : text;
  return addAnnotationLabel(aGroup, shortenedText, a, null, text);
}

function addAnnotationLabel(aGroup, label, a, additionalClassNames,
    fullLabel?) {
  let classNames = Class.Annotation.LABEL;
  if (additionalClassNames) {
    classNames += " " + additionalClassNames;
  }
  let titleText = fullLabel ? fullLabel : label;
  return aGroup.append("text")
                .attr("class", classNames)
                .attr("dy", ".35em")
                .attr("text-anchor", a.isIn ? "end" : "start")
                .text(label)
                .append("title").text(titleText);
}

function addInteraction(selection, d: render.RenderNodeInformation,
    sceneBehavior) {
  selection
    .on("mouseover", a => {
      sceneBehavior.fire("annotation-highlight", {
        name: a.node.name,
        hostName: d.node.name
      });
    })
    .on("mouseout", a => {
      sceneBehavior.fire("annotation-unhighlight", {
        name: a.node.name,
        hostName: d.node.name
      });
    })
    .on("click", a => {
      // Stop this event"s propagation so that it isn't also considered a
      // graph-select.
      (<Event>d3.event).stopPropagation();
      sceneBehavior.fire("annotation-select", {
        name: a.node.name,
        hostName: d.node.name
      });
    });
};

/**
 * Adjust annotation's position.
 *
 * @param aGroup selection of a "g.annotation" element.
 * @param d Host node data.
 * @param a annotation node data.
 * @param scene Polymer scene element.
 */
function update(aGroup, d: render.RenderNodeInformation, a: render.Annotation,
    sceneBehavior) {
  // Annotations that point to embedded nodes (constants,summary)
  // don't have a render information attached so we don't stylize these.
  // Also we don't stylize ellipsis annotations (the string "... and X more").
  if (a.renderNodeInfo &&
      a.annotationType !== tf.graph.render.AnnotationType.ELLIPSIS) {
    node.stylize(aGroup, a.renderNodeInfo, sceneBehavior,
      Class.Annotation.NODE);
  }

  if (a.annotationType === tf.graph.render.AnnotationType.SUMMARY) {
    // Update the width of the annotation to give space for the image.
    a.width += 10;
  }

  // label position
  aGroup.select("text." + Class.Annotation.LABEL).transition().attr({
    x: d.x + a.dx + (a.isIn ? -1 : 1) * (a.width / 2 + a.labelOffset),
    y: d.y + a.dy
  });

  // Some annotations (such as summary) are represented using a 12x12 image tag.
  // Purposely ommited units (e.g. pixels) since the images are vector graphics.
  // If there is an image, we adjust the location of the image to be vertically
  // centered with the node and horizontally centered between the arrow and the
  // text label.
  aGroup.select("image").transition().attr({
    x: d.x + a.dx - 3,
    y: d.y + a.dy - 6
  });

  // Node position (only one of the shape selection will be non-empty.)
  scene.positionEllipse(aGroup.select("." + Class.Annotation.NODE + " ellipse"),
                        d.x + a.dx, d.y + a.dy, a.width, a.height);
  scene.positionRect(aGroup.select("." + Class.Annotation.NODE + " rect"),
                     d.x + a.dx, d.y + a.dy, a.width, a.height);
  scene.positionRect(aGroup.select("." + Class.Annotation.NODE + " use"),
                     d.x + a.dx, d.y + a.dy, a.width, a.height);

  // Edge position
  aGroup.select("path." + Class.Annotation.EDGE).transition().attr("d", a => {
        // map relative position to absolute position
        let points = a.points.map(p => {
          return {x: p.dx + d.x, y: p.dy + d.y};
        });
        return edge.interpolate(points);
      });
};

} // close module
