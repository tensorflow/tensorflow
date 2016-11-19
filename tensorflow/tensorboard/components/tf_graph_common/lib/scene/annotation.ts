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
module tf.graph.scene.annotation {
  /**
   * Populate a given annotation container group
   *
   *     <g class='{in|out}-annotations'></g>
   *
   * with annotation group of the following structure:
   *
   * <g class='annotation'>
   *   <g class='annotation-node'>
   *   <!--
   *   Content here determined by Scene.node.buildGroup.
   *   -->
   *   </g>
   * </g>
   *
   * @param container selection of the container.
   * @param annotationData node.{in|out}Annotations
   * @param d node to build group for.
   * @param sceneElement <tf-graph-scene> polymer element.
   * @return selection of appended objects
   */
  export function buildGroup(
      container, annotationData: render.AnnotationList,
      d: render.RenderNodeInfo, sceneElement) {
    // Select all children and join with data.
    let annotationGroups =
        container
            .selectAll(function() {
              // using d3's selector function
              // See https://github.com/mbostock/d3/releases/tag/v2.0.0
              // (It's not listed in the d3 wiki.)
              return this.childNodes;
            })
            .data(annotationData.list, d => { return d.node.name; });

    annotationGroups.enter()
        .append('g')
        .attr('data-name', a => { return a.node.name; })
        .each(function(a) {
          let aGroup = d3.select(this);

          // Add annotation to the index in the scene
          sceneElement.addAnnotationGroup(a, d, aGroup);
          // Append annotation edge
          let edgeType = Class.Annotation.EDGE;
          let metaedge = a.renderMetaedgeInfo && a.renderMetaedgeInfo.metaedge;
          if (metaedge && !metaedge.numRegularEdges) {
            edgeType += ' ' + Class.Annotation.CONTROL_EDGE;
          }
          // If any edges are reference edges, add the reference edge class.
          if (metaedge && metaedge.numRefEdges) {
            edgeType += ' ' + Class.Edge.REF_LINE;
          }
          edge.appendEdge(aGroup, a, sceneElement, edgeType);

          if (a.annotationType !== render.AnnotationType.ELLIPSIS) {
            addAnnotationLabelFromNode(aGroup, a);
            buildShape(aGroup, a);
          } else {
            addAnnotationLabel(
                aGroup, a.node.name, a, Class.Annotation.ELLIPSIS);
          }
        });

    annotationGroups
        .attr(
            'class',
            a => {
              return Class.Annotation.GROUP + ' ' +
                  annotationToClassName(a.annotationType) + ' ' +
                  node.nodeClass(a);
            })
        .each(function(a) {
          let aGroup = d3.select(this);
          update(aGroup, d, a, sceneElement);
          if (a.annotationType !== render.AnnotationType.ELLIPSIS) {
            addInteraction(aGroup, d, a, sceneElement);
          }
        });

    annotationGroups.exit()
        .each(function(a) {
          let aGroup = d3.select(this);

          // Remove annotation from the index in the scene
          sceneElement.removeAnnotationGroup(a, d, aGroup);
        })
        .remove();
    return annotationGroups;
};

/**
 * Maps an annotation enum to a class name used in css rules.
 */
function annotationToClassName(annotationType: render.AnnotationType) {
  return (render.AnnotationType[annotationType] || '').toLowerCase() || null;
}

function buildShape(aGroup, a: render.Annotation) {
  if (a.annotationType === render.AnnotationType.SUMMARY) {
    let summary = selectOrCreateChild(aGroup, 'use');
    summary.attr({
      'class': 'summary',
      'xlink:href': '#summary-icon',
      'cursor': 'pointer'
    });
  } else {
    let shape = node.buildShape(aGroup, a, Class.Annotation.NODE);
    // add title tag to get native tooltips
    selectOrCreateChild(shape, 'title').text(a.node.name);
  }
}

function addAnnotationLabelFromNode(aGroup, a: render.Annotation) {
  let namePath = a.node.name.split('/');
  let text = namePath[namePath.length - 1];
  return addAnnotationLabel(aGroup, text, a, null);
}

function addAnnotationLabel(
    aGroup, label: string, a: render.Annotation, additionalClassNames) {
  let classNames = Class.Annotation.LABEL;
  if (additionalClassNames) {
    classNames += ' ' + additionalClassNames;
  }
  let txtElement = aGroup.append('text')
                       .attr('class', classNames)
                       .attr('dy', '.35em')
                       .attr('text-anchor', a.isIn ? 'end' : 'start')
                       .text(label);

  return tf.graph.scene.node.enforceLabelWidth(txtElement, -1);
}

function addInteraction(selection, d: render.RenderNodeInfo,
    annotation: render.Annotation, sceneElement) {
  selection
      .on('mouseover',
          a => {
            sceneElement.fire(
                'annotation-highlight',
                {name: a.node.name, hostName: d.node.name});
          })
      .on('mouseout',
          a => {
            sceneElement.fire(
                'annotation-unhighlight',
                {name: a.node.name, hostName: d.node.name});
          })
      .on('click', a => {
        // Stop this event's propagation so that it isn't also considered a
        // graph-select.
        (<Event>d3.event).stopPropagation();
        sceneElement.fire(
            'annotation-select', {name: a.node.name, hostName: d.node.name});
      });
  if (annotation.annotationType !== render.AnnotationType.SUMMARY &&
      annotation.annotationType !== render.AnnotationType.CONSTANT) {
    selection.on(
        'contextmenu', contextmenu.getMenu(
                           node.getContextMenu(annotation.node, sceneElement)));
  }
};

/**
 * Adjust annotation's position.
 *
 * @param aGroup selection of a 'g.annotation' element.
 * @param d Host node data.
 * @param a annotation node data.
 * @param sceneElement <tf-graph-scene> polymer element.
 */
function update(aGroup, d: render.RenderNodeInfo, a: render.Annotation,
    sceneElement) {
  let cx = layout.computeCXPositionOfNodeShape(d);
  // Annotations that point to embedded nodes (constants,summary)
  // don't have a render information attached so we don't stylize these.
  // Also we don't stylize ellipsis annotations (the string '... and X more').
  if (a.renderNodeInfo &&
      a.annotationType !== render.AnnotationType.ELLIPSIS) {
    node.stylize(aGroup, a.renderNodeInfo, sceneElement,
      Class.Annotation.NODE);
  }

  if (a.annotationType === render.AnnotationType.SUMMARY) {
    // Update the width of the annotation to give space for the image.
    a.width += 10;
  }

  // label position
  aGroup.select('text.' + Class.Annotation.LABEL).transition().attr({
    x: cx + a.dx + (a.isIn ? -1 : 1) * (a.width / 2 + a.labelOffset),
    y: d.y + a.dy
  });

  // Some annotations (such as summary) are represented using a 12x12 image tag.
  // Purposely omitted units (e.g. pixels) since the images are vector graphics.
  // If there is an image, we adjust the location of the image to be vertically
  // centered with the node and horizontally centered between the arrow and the
  // text label.
  aGroup.select('use.summary').transition().attr({
    x: cx + a.dx - 3,
    y: d.y + a.dy - 6
  });

  // Node position (only one of the shape selection will be non-empty.)
  positionEllipse(
      aGroup.select('.' + Class.Annotation.NODE + ' ellipse'), cx + a.dx,
      d.y + a.dy, a.width, a.height);
  positionRect(
      aGroup.select('.' + Class.Annotation.NODE + ' rect'), cx + a.dx,
      d.y + a.dy, a.width, a.height);
  positionRect(
      aGroup.select('.' + Class.Annotation.NODE + ' use'), cx + a.dx,
      d.y + a.dy, a.width, a.height);

  // Edge position
  aGroup.select('path.' + Class.Annotation.EDGE).transition().attr('d', a => {
    // map relative position to absolute position
    let points = a.points.map(p => { return {x: p.dx + cx, y: p.dy + d.y}; });
    return edge.interpolate(points);
  });
};

} // close module
