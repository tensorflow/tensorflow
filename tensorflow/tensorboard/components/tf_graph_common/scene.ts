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
module tf.graph.scene {
  const svgNamespace = 'http://www.w3.org/2000/svg';

  /** Enums element class of objects in the scene */
  export let Class = {
    Node: {
      // <g> element that contains nodes.
      CONTAINER: 'nodes',
      // <g> element that contains detail about a node.
      GROUP: 'node',
      // <g> element that contains visual elements (like rect, ellipse).
      SHAPE: 'nodeshape',
      // <*> element(s) under SHAPE that should receive color updates.
      COLOR_TARGET: 'nodecolortarget',
      // <text> element showing the node's label.
      LABEL: 'nodelabel',
      // <g> element that contains all visuals for the expand/collapse
      // button for expandable group nodes.
      BUTTON_CONTAINER: 'buttoncontainer',
      // <circle> element that surrounds expand/collapse buttons.
      BUTTON_CIRCLE: 'buttoncircle',
      // <path> element of the expand button.
      EXPAND_BUTTON: 'expandbutton',
      // <path> element of the collapse button.
      COLLAPSE_BUTTON: 'collapsebutton'
    },
    Edge: {
      CONTAINER: 'edges',
      GROUP: 'edge',
      LINE: 'edgeline',
      REFERENCE_EDGE: 'referenceedge',
      REF_LINE: 'refline',
      STRUCTURAL: 'structural'
    },
    Annotation: {
      OUTBOX: 'out-annotations',
      INBOX: 'in-annotations',
      GROUP: 'annotation',
      NODE: 'annotation-node',
      EDGE: 'annotation-edge',
      CONTROL_EDGE: 'annotation-control-edge',
      LABEL: 'annotation-label',
      ELLIPSIS: 'annotation-ellipsis'
    },
    Scene: {
      GROUP: 'scene',
      CORE: 'core',
      INEXTRACT: 'in-extract',
      OUTEXTRACT: 'out-extract'
    },
    Subscene: {GROUP: 'subscene'},
    OPNODE: 'op',
    METANODE: 'meta',
    SERIESNODE: 'series',
    BRIDGENODE: 'bridge',
    ELLIPSISNODE: 'ellipsis'
  };

  /**
   * A health pill encapsulates an overview of tensor element values. The value
   * field is a list of 12 numbers that shed light on the status of the tensor.
   * Visualized in health pills are the 3rd through 8th (inclusive) numbers of
   * health pill values. Those 6 numbers are counts of tensor elements that fall
   * under -Inf, negative, 0, positive, +Inf, NaN (in that order).
   *
   * Please keep this interface consistent with HealthPillDatum within
   * backend.ts.
   */
  export interface HealthPill {
    device_name: string;
    node_name: string;
    output_slot: number;
    dtype: string;
    shape: number[];
    value: number[];
    wall_time: number;
    step: number;
  }

  interface HealthPillNumericStats {
    min: number;
    max: number;
    mean: number;
    stddev: number;
  }

  /**
   * Encapsulates how to render a single entry in a health pill. Each entry
   * corresponds to a category of tensor element values.
   */
  export interface HealthPillEntry {
    background_color: string;
    label: string;
  }
  ;
  export let healthPillEntries: HealthPillEntry[] = [
    {
      background_color: '#CC2F2C',
      label: 'NaN',
    },
    {
      background_color: '#FF8D00',
      label: '-∞',
    },
    {
      background_color: '#EAEAEA',
      label: '-',
    },
    {
      background_color: '#A5A5A5',
      label: '0',
    },
    {
      background_color: '#262626',
      label: '+',
    },
    {
      background_color: '#003ED4',
      label: '+∞',
    },
  ];

  /**
   * Helper method for fitting the graph in the svg view.
   *
   * @param svg The main svg.
   * @param zoomG The svg group used for panning and zooming.
   * @param d3zoom The zoom behavior.
   * @param callback Called when the fitting is done.
   */
  export function fit(svg, zoomG, d3zoom, callback) {
    let svgRect = svg.getBoundingClientRect();
    let sceneSize = null;
    try {
      sceneSize = zoomG.getBBox();
      if (sceneSize.width === 0) {
        // There is no scene anymore. We have been detached from the dom.
        return;
      }
    } catch (e) {
      // Firefox produced NS_ERROR_FAILURE if we have been
      // detached from the dom.
      return;
    }
    let scale = 0.9 *
        Math.min(
            svgRect.width / sceneSize.width, svgRect.height / sceneSize.height,
            2);
    let params = layout.PARAMS.graph;
    const transform = d3.zoomIdentity
        .scale(scale)
        .translate(params.padding.paddingLeft, params.padding.paddingTop);

    d3.select(svg)
        .transition()
        .duration(500)
        .call(d3zoom.transform, transform)
        .on('end.fitted', () => {
          // Remove the listener for the zoomend event,
          // so we don't get called at the end of regular zoom events,
          // just those that fit the graph to screen.
          d3zoom.on('end.fitted', null);
          callback();
        });
};

/**
 * Helper method for panning the graph to center on the provided node,
 * if the node is currently off-screen.
 *
 * @param nodeName The node to center the graph on
 * @param svg The root SVG element for the graph
 * @param zoomG The svg group used for panning and zooming.
 * @param d3zoom The zoom behavior.
 * @return True if the graph had to be panned to display the
 *            provided node.
 */
export function panToNode(nodeName: String, svg, zoomG, d3zoom): boolean {
  let node = <SVGAElement>d3
                 .select('[data-name="' + nodeName + '"].' + Class.Node.GROUP)
                 .node();
  if (!node) {
    return false;
  }

  // Check if the selected node is off-screen in either
  // X or Y dimension in either direction.
  let nodeBox = node.getBBox();
  let nodeCtm = node.getScreenCTM();
  let pointTL = svg.createSVGPoint();
  let pointBR = svg.createSVGPoint();
  pointTL.x = nodeBox.x;
  pointTL.y = nodeBox.y;
  pointBR.x = nodeBox.x + nodeBox.width;
  pointBR.y = nodeBox.y + nodeBox.height;
  pointTL = pointTL.matrixTransform(nodeCtm);
  pointBR = pointBR.matrixTransform(nodeCtm);
  let isOutsideOfBounds = (start, end, bound) => {
    return end < 0 || start > bound;
  };
  let svgRect = svg.getBoundingClientRect();
  if (isOutsideOfBounds(pointTL.x, pointBR.x, svgRect.width) ||
      isOutsideOfBounds(pointTL.y, pointBR.y, svgRect.height)) {
    // Determine the amount to translate the graph in both X and Y dimensions in
    // order to center the selected node. This takes into account the position
    // of the node, the size of the svg scene, the amount the scene has been
    // scaled by through zooming, and any previous transforms already performed
    // by this logic.
    let centerX = (pointTL.x + pointBR.x) / 2;
    let centerY = (pointTL.y + pointBR.y) / 2;
    let dx = ((svgRect.width / 2) - centerX);
    let dy = ((svgRect.height / 2) - centerY);

    // We translate by this amount. We divide the X and Y translations by the
    // scale to undo how translateBy scales the translations (in d3 v4).
    const svgTransform = d3.zoomTransform(svg);
    d3.select(svg).transition().duration(500).call(
        d3zoom.translateBy, dx / svgTransform.k, dy / svgTransform.k);

    return true;
  }
  return false;
};

/**
 * Given a container d3 selection, select a child svg element of a given tag
 * and class if exists or append / insert one otherwise.  If multiple children
 * matches the tag and class name, returns only the first one.
 *
 * @param container
 * @param tagName tag name.
 * @param className (optional) Class name or a list of class names.
 * @param before (optional) reference DOM node for insertion.
 * @return selection of the element
 */
export function selectOrCreateChild(
    container, tagName: string, className?: string | string[], before?): d3.Selection<any, any, any, any> {
  let child = selectChild(container, tagName, className);
  if (!child.empty()) {
    return child;
  }
  let newElement =
      document.createElementNS('http://www.w3.org/2000/svg', tagName);

  if (className instanceof Array) {
    for (let i = 0; i < className.length; i++) {
      newElement.classList.add(className[i]);
    }
  } else {
    newElement.classList.add(className);
  }

  if (before) { // if before exists, insert
    container.node().insertBefore(newElement, before);
  } else { // otherwise, append
    container.node().appendChild(newElement);
  }
  return d3.select(newElement)
           // need to bind data to emulate d3_selection.append
           .datum(container.datum());
};

/**
 * Given a container d3 selection, select a child element of a given tag and
 * class. If multiple children matches the tag and class name, returns only
 * the first one.
 *
 * @param container
 * @param tagName tag name.
 * @param className (optional) Class name or list of class names.
 * @return selection of the element, or an empty selection
 */
export function selectChild(
    container, tagName: string, className?: string | string[]): d3.Selection<any, any, any, any> {
  let children = container.node().childNodes;
  for (let i = 0; i < children.length; i++) {
    let child = children[i];
    if (child.tagName === tagName) {
      if (className instanceof Array) {
        let hasAllClasses = true;
        for (let j = 0; j < className.length; j++) {
          hasAllClasses =
              hasAllClasses && child.classList.contains(className[j]);
        }
        if (hasAllClasses) {
          return d3.select(child);
        }
      } else if ((!className || child.classList.contains(className))) {
        return d3.select(child);
      }
    }
  }
  return d3.select(null);
};

/**
 * Select or create a sceneGroup and build/update its nodes and edges.
 *
 * Structure Pattern:
 *
 * <g class='scene'>
 *   <g class='core'>
 *     <g class='edges'>
 *       ... stuff from tf.graph.scene.edges.build ...
 *     </g>
 *     <g class='nodes'>
 *       ... stuff from tf.graph.scene.nodes.build ...
 *     </g>
 *   </g>
 *   <g class='in-extract'>
 *     <g class='nodes'>
 *       ... stuff from tf.graph.scene.nodes.build ...
 *     </g>
 *   </g>
 *   <g class='out-extract'>
 *     <g class='nodes'>
 *       ... stuff from tf.graph.scene.nodes.build ...
 *     </g>
 *   </g>
 * </g>
 *
 * @param container D3 selection of the parent.
 * @param renderNode render node of a metanode or series node.
 * @param sceneElement <tf-graph-scene> polymer element.
 * @param sceneClass class attribute of the scene (default='scene').
 */
export function buildGroup(container,
    renderNode: render.RenderGroupNodeInfo,
    sceneElement,
    sceneClass: string): d3.Selection<any, any, any, any> {
  sceneClass = sceneClass || Class.Scene.GROUP;
  let isNewSceneGroup = selectChild(container, 'g', sceneClass).empty();
  let sceneGroup = selectOrCreateChild(container, 'g', sceneClass);

  // core
  let coreGroup = selectOrCreateChild(sceneGroup, 'g', Class.Scene.CORE);
  let coreNodes = _.reduce(renderNode.coreGraph.nodes(), (nodes, name) => {
                    let node = renderNode.coreGraph.node(name);
                    if (!node.excluded) {
                      nodes.push(node);
                    }
                    return nodes;
                  }, []);

  if (renderNode.node.type === NodeType.SERIES) {
    // For series, we want the first item on top, so reverse the array so
    // the first item in the series becomes last item in the top, and thus
    // is rendered on the top.
    coreNodes.reverse();
  }

  // Create the layer of edges for this scene (paths).
  edge.buildGroup(coreGroup, renderNode.coreGraph, sceneElement);

  // Create the layer of nodes for this scene (ellipses, rects etc).
  node.buildGroup(coreGroup, coreNodes, sceneElement);

  // In-extract
  if (renderNode.isolatedInExtract.length > 0) {
    let inExtractGroup =
        selectOrCreateChild(sceneGroup, 'g', Class.Scene.INEXTRACT);
    node.buildGroup(inExtractGroup, renderNode.isolatedInExtract,
        sceneElement);
  } else {
    selectChild(sceneGroup, 'g', Class.Scene.INEXTRACT).remove();
  }

  // Out-extract
  if (renderNode.isolatedOutExtract.length > 0) {
    let outExtractGroup =
        selectOrCreateChild(sceneGroup, 'g', Class.Scene.OUTEXTRACT);
    node.buildGroup(outExtractGroup, renderNode.isolatedOutExtract,
        sceneElement);
  } else {
    selectChild(sceneGroup, 'g', Class.Scene.OUTEXTRACT).remove();
  }

  position(sceneGroup, renderNode);

  // Fade in the scene group if it didn't already exist.
  if (isNewSceneGroup) {
    sceneGroup.attr('opacity', 0).transition().attr('opacity', 1);
  }

  return sceneGroup;
};

/**
 * Given a scene's svg group, set  g.in-extract, g.coreGraph, g.out-extract svg
 * groups' position relative to the scene.
 *
 * @param sceneGroup
 * @param renderNode render node of a metanode or series node.
 */
function position(sceneGroup, renderNode: render.RenderGroupNodeInfo) {
  // Translate scenes down by the label height so that when showing graphs in
  // expanded metanodes, the graphs are below the labels.  Do not shift them
  // down for series nodes as series nodes don't have labels inside of their
  // bounding boxes.
  let yTranslate = renderNode.node.type === NodeType.SERIES ?
    0 : layout.PARAMS.subscene.meta.labelHeight;

  // core
  translate(selectChild(sceneGroup, 'g', Class.Scene.CORE), 0, yTranslate);

  // in-extract
  let hasInExtract = renderNode.isolatedInExtract.length > 0;
  let hasOutExtract = renderNode.isolatedOutExtract.length > 0;

  if (hasInExtract) {
    let offset = layout.PARAMS.subscene.meta.extractXOffset;
    let inExtractX = renderNode.coreBox.width -
      renderNode.inExtractBox.width / 2 - renderNode.outExtractBox.width -
          (hasOutExtract ? offset : 0);
    translate(
        selectChild(sceneGroup, 'g', Class.Scene.INEXTRACT), inExtractX,
        yTranslate);
  }

  // out-extract
  if (hasOutExtract) {
    let outExtractX = renderNode.coreBox.width -
      renderNode.outExtractBox.width / 2;
    translate(
        selectChild(sceneGroup, 'g', Class.Scene.OUTEXTRACT), outExtractX,
        yTranslate);
  }
};

/** Adds a click listener to a group that fires a graph-select event */
export function addGraphClickListener(graphGroup, sceneElement) {
  d3.select(graphGroup).on('click', () => {
    sceneElement.fire('graph-select');
  });
};

/** Helper for adding transform: translate(x0, y0) */
export function translate(selection, x0: number, y0: number) {
  // If it is already placed on the screen, make it a transition.
  if (selection.attr('transform') != null) {
    selection = selection.transition('position');
  }
  selection.attr('transform', 'translate(' + x0 + ',' + y0 + ')');
};

/**
 * Helper for setting position of a svg rect
 * @param rect rect to set position of.
 * @param cx Center x.
 * @param cy Center x.
 * @param width Width to set.
 * @param height Height to set.
 */
export function positionRect(rect, cx: number, cy: number, width: number,
    height: number) {
  rect.transition()
    .attr('x', cx - width / 2)
    .attr('y', cy - height / 2)
    .attr('width', width)
    .attr('height', height);
};

/**
 * Helper for setting position of a svg expand/collapse button
 * @param button container group
 * @param renderNode the render node of the group node to position
 *        the button on.
 */
export function positionButton(button, renderNode: render.RenderNodeInfo) {
  let cx = layout.computeCXPositionOfNodeShape(renderNode);
  // Position the button in the top-right corner of the group node,
  // with space given the draw the button inside of the corner.
  let width = renderNode.expanded ?
      renderNode.width : renderNode.coreBox.width;
  let height = renderNode.expanded ?
      renderNode.height : renderNode.coreBox.height;
  let x = cx + width / 2 - 6;
  let y = renderNode.y - height / 2 + 6;
  // For unexpanded series nodes, the button has special placement due
  // to the unique visuals of this group node.
  if (renderNode.node.type === NodeType.SERIES && !renderNode.expanded) {
    x += 10;
    y -= 2;
  }
  let translateStr = 'translate(' + x + ',' + y + ')';
  button.selectAll('path').transition().attr('transform', translateStr);
  button.select('circle').transition().attr(
      {cx: x, cy: y, r: layout.PARAMS.nodeSize.meta.expandButtonRadius});
};

/**
 * Helper for setting position of a svg ellipse
 * @param ellipse ellipse to set position of.
 * @param cx Center x.
 * @param cy Center x.
 * @param width Width to set.
 * @param height Height to set.
 */
export function positionEllipse(ellipse, cx: number, cy: number,
    width: number, height: number) {
  ellipse.transition()
    .attr('cx', cx)
    .attr('cy', cy)
    .attr('rx', width / 2)
    .attr('ry', height / 2);
};

/**
 * @param {number} stat A stat for a health pill (such as mean or variance).
 * @param {boolean} shouldRoundOnesDigit Whether to round this number to the
 *     ones digit. Useful for say int, uint, and bool output types.
 * @return {string} A human-friendly string representation of that stat.
 */
export function humanizeHealthPillStat(stat, shouldRoundOnesDigit) {
  if (shouldRoundOnesDigit) {
    return stat.toFixed(0);
  }

  if (Math.abs(stat) >= 1) {
    return stat.toFixed(1);
  }
  return stat.toExponential(1);
}

/**
 * Get text content describing a health pill.
 */
function _getHealthPillTextContent(healthPill: HealthPill,
                                   totalCount: number,
                                   elementsBreakdown: number[],
                                   numericStats: HealthPillNumericStats) {
  let text = 'Device: ' + healthPill.device_name + '\n';
  text += 'dtype: ' + healthPill.dtype + '\n';

  let shapeStr = '(scalar)';
  if (healthPill.shape.length > 0) {
    shapeStr = '(' + healthPill.shape.join(',') + ')';
  }
  text += '\nshape: ' + shapeStr + '\n\n';

  text += '#(elements): ' + totalCount + '\n';
  const breakdownItems = [];
  for (let i = 0; i < elementsBreakdown.length; i++) {
    if (elementsBreakdown[i] > 0) {
      breakdownItems.push(
          '#(' + healthPillEntries[i].label + '): ' + elementsBreakdown[i]);
    }
  }
  text += breakdownItems.join(', ') + '\n\n';

  // In some cases (e.g., size-0 tensors; all elements are nan or inf) the
  // min/max and mean/stddev stats are meaningless.
  if (numericStats.max >= numericStats.min) {
    text += 'min: ' + numericStats.min + ', max: ' + numericStats.max + '\n';
    text += 'mean: ' + numericStats.mean + ', stddev: ' + numericStats.stddev;
  }

  return text;
}

/**
 * Renders a health pill for an op atop a node.
 */
function _addHealthPill(
    nodeGroupElement: SVGElement, healthPill: HealthPill,
    nodeInfo: render.RenderNodeInfo) {
  // Check if text already exists at location.
  d3.select(nodeGroupElement.parentNode as any).selectAll('.health-pill').remove();

  if (!nodeInfo || !healthPill) {
    return;
  }

  let lastHealthPillData = healthPill.value;

  // For now, we only visualize the 6 values that summarize counts of tensor
  // elements of various categories: -Inf, negative, 0, positive, Inf, and NaN.
  const lastHealthPillElementsBreakdown = lastHealthPillData.slice(2, 8);
  let totalCount = lastHealthPillData[1];
  const numericStats: HealthPillNumericStats = {
      min: lastHealthPillData[8],
      max: lastHealthPillData[9],
      mean: lastHealthPillData[10],
      stddev: Math.sqrt(lastHealthPillData[11])
  };

  let healthPillWidth = 60;
  let healthPillHeight = 10;
  if (nodeInfo.node.type === tf.graph.NodeType.OP) {
    // Use a smaller health pill for op nodes (rendered as smaller ellipses).
    healthPillWidth /= 2;
    healthPillHeight /= 2;
  }

  let healthPillGroup = document.createElementNS(svgNamespace, 'g');
  healthPillGroup.classList.add('health-pill');

  // Define the gradient for the health pill.
  let healthPillDefs = document.createElementNS(svgNamespace, 'defs');
  healthPillGroup.appendChild(healthPillDefs);
  let healthPillGradient =
      document.createElementNS(svgNamespace, 'linearGradient');
  const healthPillGradientId = 'health-pill-gradient';
  healthPillGradient.setAttribute('id', healthPillGradientId);

  let cumulativeCount = 0;
  let previousOffset = '0%';
  for (let i = 0; i < lastHealthPillElementsBreakdown.length; i++) {
    if (!lastHealthPillElementsBreakdown[i]) {
      // Exclude empty categories.
      continue;
    }
    cumulativeCount += lastHealthPillElementsBreakdown[i];

    // Create a color interval using 2 stop elements.
    let stopElement0 = document.createElementNS(svgNamespace, 'stop');
    stopElement0.setAttribute('offset', previousOffset);
    stopElement0.setAttribute(
        'stop-color', healthPillEntries[i].background_color);
    healthPillGradient.appendChild(stopElement0);

    let stopElement1 = document.createElementNS(svgNamespace, 'stop');
    let percent = (cumulativeCount * 100 / totalCount) + '%';
    stopElement1.setAttribute('offset', percent);
    stopElement1.setAttribute(
        'stop-color', healthPillEntries[i].background_color);
    healthPillGradient.appendChild(stopElement1);
    previousOffset = percent;
  }
  healthPillDefs.appendChild(healthPillGradient);

  // Create the rectangle for the health pill.
  let rect = document.createElementNS(svgNamespace, 'rect');
  rect.setAttribute('fill', 'url(#' + healthPillGradientId + ')');
  rect.setAttribute('width', String(healthPillWidth));
  rect.setAttribute('height', String(healthPillHeight));
  healthPillGroup.appendChild(rect);

  // Show a title with specific counts on hover.
  let titleSvg = document.createElementNS(svgNamespace, 'title');
  titleSvg.textContent = _getHealthPillTextContent(
      healthPill, totalCount, lastHealthPillElementsBreakdown, numericStats);
  healthPillGroup.appendChild(titleSvg);
  // TODO(cais): Make the tooltip content prettier.

  // Center this health pill just right above the node for the op.
  let healthPillX = nodeInfo.x - healthPillWidth / 2;
  let healthPillY = nodeInfo.y - healthPillHeight - nodeInfo.height / 2 - 2;
  if (nodeInfo.labelOffset < 0) {
    // The label is positioned above the node. Do not occlude the label.
    healthPillY += nodeInfo.labelOffset;
  }

  if (lastHealthPillElementsBreakdown[2] ||
      lastHealthPillElementsBreakdown[3] ||
      lastHealthPillElementsBreakdown[4]) {
    // At least 1 "non-Inf and non-NaN" value exists (a -, 0, or + value). Show
    // stats on tensor values.

    // Determine if we should display the output range as integers.
    let shouldRoundOnesDigit = false;
    let node = nodeInfo.node as OpNode;
    let attributes = node.attr;
    if (attributes && attributes.length) {
      // Find the attribute for output type if there is one.
      for (let i = 0; i < attributes.length; i++) {
        if (attributes[i].key === 'T') {
          // Note whether the output type is an integer.
          let outputType = attributes[i].value['type'];
          shouldRoundOnesDigit =
              outputType && /^DT_(BOOL|INT|UINT)/.test(outputType);
          break;
        }
      }
    }

    let statsSvg = document.createElementNS(svgNamespace, 'text');
    const minString = humanizeHealthPillStat(numericStats.min, shouldRoundOnesDigit);
    const maxString = humanizeHealthPillStat(numericStats.max, shouldRoundOnesDigit);
    if (totalCount > 1) {
      statsSvg.textContent = minString + ' ~ ' + maxString;
    } else {
      statsSvg.textContent = minString;
    }
    statsSvg.classList.add('health-pill-stats');
    statsSvg.setAttribute('x', String(healthPillWidth / 2));
    statsSvg.setAttribute('y', '-2');
    healthPillGroup.appendChild(statsSvg);
  }

  healthPillGroup.setAttribute(
      'transform', 'translate(' + healthPillX + ', ' + healthPillY + ')');

  Polymer.dom(nodeGroupElement.parentNode).appendChild(healthPillGroup);
}

/**
 * Adds health pills (which visualize tensor summaries) to a graph group.
 * @param svgRoot The root SVG element of the graph to add heath pills to.
 * @param nodeNamesToHealthPills An object mapping node name to health pill.
 * @param colors A list of colors to use.
 */
export function addHealthPills(
    svgRoot: SVGElement, nodeNamesToHealthPills: {[key: string]: HealthPill[]},
    healthPillStepIndex: number) {
  if (!nodeNamesToHealthPills) {
    // No health pill information available.
    return;
  }

  let svgRootSelection = d3.select(svgRoot);
  svgRootSelection.selectAll('g.nodeshape')
      .each(function(nodeInfo: render.RenderNodeInfo) {
        // Only show health pill data for this node if it is available.
        let healthPills = nodeNamesToHealthPills[nodeInfo.node.name];
        let healthPill = healthPills ? healthPills[healthPillStepIndex] : null;
        _addHealthPill((this as SVGElement), healthPill, nodeInfo);
      });
};

} // close module
