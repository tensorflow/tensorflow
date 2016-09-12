/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

import {BoundingBox, CollisionGrid} from './label';
import {ScatterWebGL} from './scatterWebGL';

const FONT_SIZE = 10;
const NUM_POINTS_FOG_THRESHOLD = 5000;
const MIN_POINT_SIZE = 5.0;
const IMAGE_SIZE = 30;

const LABEL_COLOR_DAY = 0x000000;
const LABEL_COLOR_NIGHT = 0xffffff;
const LABEL_STROKE_DAY = 0xffffff;
const LABEL_STROKE_NIGHT = 0x000000;
const POINT_COLOR = 0x7575D9;
const POINT_COLOR_GRAYED = 0x888888;

const TRACE_START_HUE = 60;
const TRACE_END_HUE = 360;
const TRACE_SATURATION = 1;
const TRACE_LIGHTNESS = .3;
const TRACE_DEFAULT_OPACITY = .2;
const TRACE_DEFAULT_LINEWIDTH = 2;
const TRACE_SELECTED_OPACITY = .9;
const TRACE_SELECTED_LINEWIDTH = 3;
const TRACE_DESELECTED_OPACITY = .05;

const BLENDING_DAY = THREE.MultiplyBlending;
const BLENDING_NIGHT = THREE.AdditiveBlending;

// Constants relating to the indices of buffer arrays.
/** Item size of a single point in a bufferArray representing colors */
const RGB_NUM_BYTES = 3;
/** Item size of a single point in a bufferArray representing indices */
const INDEX_NUM_BYTES = 1;
/** Item size of a single point in a bufferArray representing locations */
const XYZ_NUM_BYTES = 3;

// The maximum number of labels to draw to keep the frame rate up.
const SAMPLE_SIZE = 10000;

const VERTEX_SHADER = `
  // Index of the specific vertex (passed in as bufferAttribute), and the
  // variable that will be used to pass it to the fragment shader.
  attribute float vertexIndex;
  varying vec2 xyIndex;

  // Similar to above, but for colors.
  attribute vec3 color;
  varying vec3 vColor;

  // If the point is highlighted, this will be 1.0 (else 0.0).
  attribute float isHighlight;

  // Uniform passed in as a property from THREE.ShaderMaterial.
  uniform bool sizeAttenuation;
  uniform float pointSize;
  uniform float imageWidth;
  uniform float imageHeight;

  void main() {
    // Pass index and color values to fragment shader.
    vColor = color;
    xyIndex = vec2(mod(vertexIndex, imageWidth),
              floor(vertexIndex / imageWidth));

    // Transform current vertex by modelViewMatrix (model world position and
    // camera world position matrix).
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);

    // Project vertex in camera-space to screen coordinates using the camera's
    // projection matrix.
    gl_Position = projectionMatrix * mvPosition;

    // Create size attenuation (if we're in 3D mode) by making the size of
    // each point inversly proportional to its distance to the camera.
    float attenuatedSize = - pointSize / mvPosition.z;
    gl_PointSize = sizeAttenuation ? attenuatedSize : pointSize;

    // If the point is a highlight, make it slightly bigger than the other
    // points, and also don't let it get smaller than some threshold.
    if (isHighlight == 1.0) {
      gl_PointSize = max(gl_PointSize * 1.2, ${MIN_POINT_SIZE.toFixed(1)});
    };
  }`;

const FRAGMENT_SHADER = `
  // Values passed in from the vertex shader.
  varying vec2 xyIndex;
  varying vec3 vColor;

  // Adding in the THREEjs shader chunks for fog.
  ${THREE.ShaderChunk['common']}
  ${THREE.ShaderChunk['fog_pars_fragment']}

  // Uniforms passed in as properties from THREE.ShaderMaterial.
  uniform sampler2D texture;
  uniform float imageWidth;
  uniform float imageHeight;
  uniform bool isImage;

  void main() {
    // A mystery variable that is required to make the THREE shaderchunk for fog
    // work correctly.
    vec3 outgoingLight = vec3(0.0);

    if (isImage) {
      // Coordinates of the vertex within the entire sprite image.
      vec2 coords = (gl_PointCoord + xyIndex) / vec2(imageWidth, imageHeight);
      // Determine the color of the spritesheet at the calculate spot.
      vec4 fromTexture = texture2D(texture, coords);

      // Finally, set the fragment color.
      gl_FragColor = vec4(vColor, 1.0) * fromTexture;
    } else {
      // Discard pixels outside the radius so points are rendered as circles.
      vec2 uv = gl_PointCoord.xy - 0.5;
      if (length(uv) > 0.5) discard;

      // If the point is not an image, just color it.
      gl_FragColor = vec4(vColor, 1.0);
    }
    ${THREE.ShaderChunk['fog_fragment']}
  }`;

/**
 * ScatterWebGLPointCanvasLabels uses GL point sprites to render
 * the scatter plot dataset, and a 2D HTML canvas to render labels.
 */
export class ScatterWebGLPointsCanvasLabels extends ScatterWebGL {
  // HTML elements.
  private gc: CanvasRenderingContext2D;
  private canvas: HTMLCanvasElement;
  private labelCanvasIsCleared = true;
  private image: HTMLImageElement;

  /** The buffer attribute that holds the positions of the points. */
  private positionBuffer: THREE.BufferAttribute;

  private defaultPointColor = POINT_COLOR;
  private pointSize2D: number;
  private pointSize3D: number;
  private fog: THREE.Fog;

  private points: THREE.Points;
  private traces: THREE.Line[];
  private tracePositionBuffer: {[trace: number]: THREE.BufferAttribute} = {};

  private labelColor: number;
  private labelStroke: number;

  private blending: THREE.Blending;

  constructor(
      container: d3.Selection<any>, labelAccessor: (index: number) => string) {
    super(container, labelAccessor);

    // For now, labels are drawn on this transparent canvas with no touch events
    // rather than being rendered in webGL.
    this.canvas = container.append('canvas').node() as HTMLCanvasElement;
    this.gc = this.canvas.getContext('2d');
    d3.select(this.canvas).style({position: 'absolute', left: 0, top: 0});
    this.canvas.style.pointerEvents = 'none';
    this.onResize();
  }
  /**
   * Create points, set their locations and actually instantiate the
   * geometry.
   */
  private addSprites() {
    // Create geometry.
    this.geometry = new THREE.BufferGeometry();
    this.createBufferAttributes();
    let canvas = document.createElement('canvas');
    let image = this.image || canvas;
    // TODO(b/31390553): Pass sprite dim to the renderer.
    let spriteDim = 28.0;
    let tex = this.createTexture(image);
    let pointSize = (this.zAccessor ? this.pointSize3D : this.pointSize2D);
    if (this.image) {
      pointSize = IMAGE_SIZE;
    }
    let uniforms = {
      texture: {type: 't', value: tex},
      imageWidth: {type: 'f', value: image.width / spriteDim},
      imageHeight: {type: 'f', value: image.height / spriteDim},
      fogColor: {type: 'c', value: this.fog.color},
      fogNear: {type: 'f', value: this.fog.near},
      fogFar: {type: 'f', value: this.fog.far},
      sizeAttenuation: {type: 'bool', value: !!this.zAccessor},
      isImage: {type: 'bool', value: !!this.image},
      pointSize: {type: 'f', value: pointSize}
    };
    this.materialOptions = {
      uniforms: uniforms,
      vertexShader: VERTEX_SHADER,
      fragmentShader: FRAGMENT_SHADER,
      transparent: (this.image ? false : true),
      // When rendering points with blending, we want depthTest/Write
      // turned off.
      depthTest: (this.image ? true : false),
      depthWrite: (this.image ? true : false),
      fog: true,
      blending: (this.image ? THREE.NormalBlending : this.blending),
    };
    // Give it some material.
    let material = new THREE.ShaderMaterial(this.materialOptions);

    // And finally initialize it and add it to the scene.
    this.points = new THREE.Points(this.geometry, material);
    this.scene.add(this.points);
  }

  /**
   * Create line traces between connected points and instantiate the geometry.
   */
  private addTraces() {
    if (!this.dataSet || !this.dataSet.traces) {
      return;
    }

    this.traces = [];

    for (let i = 0; i < this.dataSet.traces.length; i++) {
      let dataTrace = this.dataSet.traces[i];

      let geometry = new THREE.BufferGeometry();
      let colors: number[] = [];

      for (let j = 0; j < dataTrace.pointIndices.length - 1; j++) {
        this.dataSet.points[dataTrace.pointIndices[j]].traceIndex = i;
        this.dataSet.points[dataTrace.pointIndices[j + 1]].traceIndex = i;

        let color1 =
            this.getPointInTraceColor(j, dataTrace.pointIndices.length);
        let color2 =
            this.getPointInTraceColor(j + 1, dataTrace.pointIndices.length);

        colors.push(
            color1.r / 255, color1.g / 255, color1.b / 255, color2.r / 255,
            color2.g / 255, color2.b / 255);
      }

      geometry.addAttribute('position', this.tracePositionBuffer[i]);
      this.tracePositionBuffer[i].needsUpdate = true;

      geometry.addAttribute(
          'color',
          new THREE.BufferAttribute(new Float32Array(colors), RGB_NUM_BYTES));

      // We use the same material for every line.
      let material = new THREE.LineBasicMaterial({
        linewidth: TRACE_DEFAULT_LINEWIDTH,
        opacity: TRACE_DEFAULT_OPACITY,
        transparent: true,
        vertexColors: THREE.VertexColors
      });

      let trace = new THREE.LineSegments(geometry, material);
      this.traces.push(trace);
      this.scene.add(trace);
    }
  }

  /** Removes all traces from the scene. */
  private removeAllTraces() {
    if (!this.traces) {
      return;
    }

    for (let i = 0; i < this.traces.length; i++) {
      this.scene.remove(this.traces[i]);
    }
    this.traces = [];
  }

  /**
   * Returns the color of a point along a trace.
   */
  private getPointInTraceColor(index: number, totalPoints: number) {
    let hue = TRACE_START_HUE +
        (TRACE_END_HUE - TRACE_START_HUE) * index / totalPoints;

    return d3.hsl(hue, TRACE_SATURATION, TRACE_LIGHTNESS).rgb();
  }

  private calibratePointSize() {
    let numPts = this.dataSet.points.length;
    let scaleConstant = 200;
    let logBase = 8;
    // Scale point size inverse-logarithmically to the number of points.
    this.pointSize3D = scaleConstant / Math.log(numPts) / Math.log(logBase);
    this.pointSize2D = this.pointSize3D / 1.5;
  }

  private setFogDistances() {
    let dists = this.getNearFarPoints();
    this.fog.near = dists.shortestDist;
    // If there are fewer points we want less fog. We do this
    // by making the "far" value (that is, the distance from the camera to the
    // far edge of the fog) proportional to the number of points.
    let multiplier = 2 -
        Math.min(this.dataSet.points.length, NUM_POINTS_FOG_THRESHOLD) /
            NUM_POINTS_FOG_THRESHOLD;
    this.fog.far = dists.furthestDist * multiplier;
  }

  private getNearFarPoints() {
    let shortestDist: number = Infinity;
    let furthestDist: number = 0;
    let camToTarget = new THREE.Vector3()
                          .copy(this.perspCamera.position)
                          .sub(this.cameraControls.target);
    for (let i = 0; i < this.dataSet.points.length; i++) {
      let point = this.getProjectedPointFromIndex(i);
      // discard points that are behind the camera
      let camToPoint =
          new THREE.Vector3().copy(this.perspCamera.position).sub(point);
      if (camToTarget.dot(camToPoint) < 0) {
        continue;
      }

      let distToCam = this.perspCamera.position.distanceToSquared(point);
      furthestDist = Math.max(furthestDist, distToCam);
      shortestDist = Math.min(shortestDist, distToCam);
    }
    furthestDist = Math.sqrt(furthestDist);
    shortestDist = Math.sqrt(shortestDist);
    return {shortestDist, furthestDist};
  }

  /** Removes all the labels. */
  private removeAllLabels() {
    // If labels are already removed, do not spend compute power to clear the
    // canvas.
    if (!this.labelCanvasIsCleared) {
      this.gc.clearRect(0, 0, this.width * this.dpr, this.height * this.dpr);
      this.labelCanvasIsCleared = true;
    }
  }

  /**
   * Reset the positions of all labels, and check for overlapps using the
   * collision grid.
   */
  private makeLabels() {
    if (this.points == null) {
      return;
    }
    // First, remove all old labels.
    this.removeAllLabels();

    if (!this.labeledPoints.length) {
      return;
    }

    this.labelCanvasIsCleared = false;

    // We never render more than ~500 labels, so when we get much past that
    // point, just break.
    let numRenderedLabels: number = 0;
    let labelHeight = parseInt(this.gc.font, 10);

    // Bounding box for collision grid.
    let boundingBox: BoundingBox = {
      loX: 0,
      hiX: this.width * this.dpr,
      loY: 0,
      hiY: this.height * this.dpr
    };

    // Make collision grid with cells proportional to window dimensions.
    let grid =
        new CollisionGrid(boundingBox, this.width / 25, this.height / 50);

    let dists = this.getNearFarPoints();
    let opacityRange = dists.furthestDist - dists.shortestDist;
    let camToTarget = new THREE.Vector3()
                          .copy(this.perspCamera.position)
                          .sub(this.cameraControls.target);

    // Setting styles for the labeled font.
    this.gc.lineWidth = 6;
    this.gc.textBaseline = 'middle';
    this.gc.font = (FONT_SIZE * this.dpr).toString() + 'px roboto';

    for (let i = 0;
         (i < this.labeledPoints.length) && !(numRenderedLabels > SAMPLE_SIZE);
         i++) {
      let index = this.labeledPoints[i];
      let point = this.getProjectedPointFromIndex(index);
      // discard points that are behind the camera
      let camToPoint =
          new THREE.Vector3().copy(this.perspCamera.position).sub(point);
      if (camToTarget.dot(camToPoint) < 0) {
        continue;
      }
      let screenCoords = this.vector3DToScreenCoords(point);
      // Have extra space between neighboring labels. Don't pack too tightly.
      let labelMargin = 2;
      // Shift the label to the right of the point circle.
      let xShift = 3;
      let textBoundingBox = {
        loX: screenCoords.x + xShift - labelMargin,
        // Computing the width of the font is expensive,
        // so we assume width of 1 at first. Then, if the label doesn't
        // conflict with other labels, we measure the actual width.
        hiX: screenCoords.x + xShift + /* labelWidth - 1 */ +1 + labelMargin,
        loY: screenCoords.y - labelHeight / 2 - labelMargin,
        hiY: screenCoords.y + labelHeight / 2 + labelMargin
      };

      if (grid.insert(textBoundingBox, true)) {
        let text = this.labelAccessor(index);
        let labelWidth = this.gc.measureText(text).width;

        // Now, check with properly computed width.
        textBoundingBox.hiX += labelWidth - 1;
        if (grid.insert(textBoundingBox) &&
            this.isLabelInBounds(labelWidth, screenCoords)) {
          let p = new THREE.Vector3(point[0], point[1], point[2]);
          let lenToCamera = this.perspCamera.position.distanceTo(p);
          // Opacity is scaled between 0.2 and 1, based on how far a label is
          // from the camera (Unless we are in 2d mode, in which case opacity is
          // just 1!)
          let opacity = this.zAccessor ?
              1.2 - (lenToCamera - dists.shortestDist) / opacityRange :
              1;
          this.formatLabel(text, screenCoords, opacity);
          numRenderedLabels++;
        }
      }
    }

    if (this.highlightedPoints.length > 0) {
      // Force-draw the first favored point with increased font size.
      let index = this.highlightedPoints[0];
      let point = this.dataSet.points[index];
      this.gc.font = (FONT_SIZE * this.dpr * 1.7).toString() + 'px roboto';
      let coords = new THREE.Vector3(
          point.projectedPoint[0], point.projectedPoint[1],
          point.projectedPoint[2]);
      let screenCoords = this.vector3DToScreenCoords(coords);
      let text = this.labelAccessor(index);
      this.formatLabel(text, screenCoords, 255);
    }
  }

  /** Checks if a given label will be within the screen's bounds. */
  private isLabelInBounds(labelWidth: number, coords: {x: number, y: number}) {
    let padding = 7;
    if ((coords.x < 0) || (coords.y < 0) ||
        (coords.x > this.width * this.dpr - labelWidth - padding) ||
        (coords.y > this.height * this.dpr)) {
      return false;
    };
    return true;
  }

  /** Add a specific label to the canvas. */
  private formatLabel(
      text: string, point: {x: number, y: number}, opacity: number) {
    let ls = new THREE.Color(this.labelStroke);
    let lc = new THREE.Color(this.labelColor);
    this.gc.strokeStyle = 'rgba(' + ls.r * 255 + ',' + ls.g * 255 + ',' +
        ls.b * 255 + ',' + opacity + ')';
    this.gc.fillStyle = 'rgba(' + lc.r * 255 + ',' + lc.g * 255 + ',' +
        lc.b * 255 + ',' + opacity + ')';
    this.gc.strokeText(text, point.x + 4, point.y);
    this.gc.fillText(text, point.x + 4, point.y);
  }

  onResize() {
    d3.select(this.canvas)
        .attr('width', this.width * this.dpr)
        .attr('height', this.height * this.dpr)
        .style({width: this.width + 'px', height: this.height + 'px'});
  }

  /**
   * Set up buffer attributes to be used for the points/images.
   */
  private createBufferAttributes() {
    // Set up buffer attribute arrays.
    let numPoints = this.dataSet.points.length;
    let colArr = new Float32Array(numPoints * RGB_NUM_BYTES);
    this.uniqueColArr = new Float32Array(numPoints * RGB_NUM_BYTES);
    let colors = new THREE.BufferAttribute(this.uniqueColArr, RGB_NUM_BYTES);
    // Assign each point a unique color in order to identify when the user
    // hovers over a point.
    for (let i = 0; i < numPoints; i++) {
      let color = new THREE.Color(i);
      colors.setXYZ(i, color.r, color.g, color.b);
    }
    colors.array = colArr;
    let hiArr = new Float32Array(numPoints);

    /** Indices cooresponding to highlighted points. */
    let highlights = new THREE.BufferAttribute(hiArr, INDEX_NUM_BYTES);

    /**
     * The actual indices of the points which we use for sizeAttenuation in
     * the shader.
     */
    let indicesShader =
        new THREE.BufferAttribute(new Float32Array(numPoints), 1);

    // Create the array of indices.
    for (let i = 0; i < numPoints; i++) {
      indicesShader.setX(i, this.dataSet.points[i].dataSourceIndex);
    }

    // Finally, add all attributes to the geometry.
    this.geometry.addAttribute('position', this.positionBuffer);
    this.positionBuffer.needsUpdate = true;
    this.geometry.addAttribute('color', colors);
    this.geometry.addAttribute('vertexIndex', indicesShader);
    this.geometry.addAttribute('isHighlight', highlights);

    // For now, nothing is highlighted.
    this.colorSprites(null);
  }

  private resetTraces() {
    if (!this.traces) {
      return;
    }
    for (let i = 0; i < this.traces.length; i++) {
      this.traces[i].material.opacity = TRACE_DEFAULT_OPACITY;
      (this.traces[i].material as THREE.LineBasicMaterial).linewidth =
          TRACE_DEFAULT_LINEWIDTH;
      this.traces[i].material.needsUpdate = true;
    }
  }

  private colorSprites(highlightStroke: ((index: number) => string)) {
    // Update attributes to change colors
    let colors = this.geometry.getAttribute('color') as THREE.BufferAttribute;
    let highlights =
        this.geometry.getAttribute('isHighlight') as THREE.BufferAttribute;
    for (let i = 0; i < this.dataSet.points.length; i++) {
      let unhighlightedColor = this.image ?
          new THREE.Color() :
          new THREE.Color(
              this.colorAccessor ? this.colorAccessor(i) :
                                   (this.defaultPointColor as any));
      colors.setXYZ(
          i, unhighlightedColor.r, unhighlightedColor.g, unhighlightedColor.b);
      highlights.setX(i, 0.0);
    }
    if (highlightStroke) {
      // Traverse in reverse so that the point we are hovering over
      // (highlightedPoints[0]) is painted last.
      for (let i = this.highlightedPoints.length - 1; i >= 0; i--) {
        let assocPoint = this.highlightedPoints[i];
        let color = new THREE.Color(highlightStroke(i));
        // Fill colors array (single array of numPoints*3 elements,
        // triples of which refer to the rgb values of a single vertex).
        colors.setXYZ(assocPoint, color.r, color.g, color.b);
        highlights.setX(assocPoint, 1.0);
      }
    }
    colors.needsUpdate = true;
    highlights.needsUpdate = true;
  }

  /* Updates the positions buffer array to reflect the actual data. */
  private updatePositionsArray() {
    // Update the points.
    for (let i = 0; i < this.dataSet.points.length; i++) {
      // Set position based on projected point.
      let pp = this.dataSet.points[i].projectedPoint;
      this.positionBuffer.setXYZ(i, pp[0], pp[1], pp[2]);
    }

    // Update the traces.
    for (let i = 0; i < this.dataSet.traces.length; i++) {
      let dataTrace = this.dataSet.traces[i];

      let vertexCount = 0;
      for (let j = 0; j < dataTrace.pointIndices.length - 1; j++) {
        let point1 = this.dataSet.points[dataTrace.pointIndices[j]];
        let point2 = this.dataSet.points[dataTrace.pointIndices[j + 1]];

        this.tracePositionBuffer[i].setXYZ(
            vertexCount, point1.projectedPoint[0], point1.projectedPoint[1],
            point1.projectedPoint[2]);
        this.tracePositionBuffer[i].setXYZ(
            vertexCount + 1, point2.projectedPoint[0], point2.projectedPoint[1],
            point2.projectedPoint[2]);
        vertexCount += 2;
      }
    }

    if (this.geometry) {
      this.positionBuffer.needsUpdate = true;

      for (let i = 0; i < this.dataSet.traces.length; i++) {
        this.tracePositionBuffer[i].needsUpdate = true;
      }
    }
  }

  /**
   * Returns an x, y, z value for each item of our data based on the accessor
   * methods.
   */
  private getPointsCoordinates() {
    // Determine max and min of each axis of our data.
    let xExtent = d3.extent(this.dataSet.points, (p, i) => this.xAccessor(i));
    let yExtent = d3.extent(this.dataSet.points, (p, i) => this.yAccessor(i));
    this.xScale.domain(xExtent).range([-1, 1]);
    this.yScale.domain(yExtent).range([-1, 1]);
    if (this.zAccessor) {
      let zExtent = d3.extent(this.dataSet.points, (p, i) => this.zAccessor(i));
      this.zScale.domain(zExtent).range([-1, 1]);
    }

    // Determine 3d coordinates of each data point.
    this.dataSet.points.forEach((d, i) => {
      d.projectedPoint[0] = this.xScale(this.xAccessor(i));
      d.projectedPoint[1] = this.yScale(this.yAccessor(i));
      d.projectedPoint[2] =
          (this.zAccessor ? this.zScale(this.zAccessor(i)) : 0);
    });
  }

  protected removeAllGeometry() {
    this.scene.remove(this.points);
    this.removeAllLabels();
    this.removeAllTraces();
  }

  /**
   * Generate a texture for the points/images and sets some initial params
   */
  protected createTexture(image: HTMLImageElement|
                          HTMLCanvasElement): THREE.Texture {
    let tex = new THREE.Texture(image);
    tex.needsUpdate = true;
    // Used if the texture isn't a power of 2.
    tex.minFilter = THREE.LinearFilter;
    tex.generateMipmaps = false;
    tex.flipY = false;
    return tex;
  }

  protected onDataSet(spriteImage: HTMLImageElement) {
    this.points = null;
    this.calibratePointSize();

    let positions =
        new Float32Array(this.dataSet.points.length * XYZ_NUM_BYTES);
    this.positionBuffer = new THREE.BufferAttribute(positions, XYZ_NUM_BYTES);

    // Set up the position buffer arrays for each trace.
    for (let i = 0; i < this.dataSet.traces.length; i++) {
      let dataTrace = this.dataSet.traces[i];
      let traces = new Float32Array(
          2 * (dataTrace.pointIndices.length - 1) * XYZ_NUM_BYTES);
      this.tracePositionBuffer[i] =
          new THREE.BufferAttribute(traces, XYZ_NUM_BYTES);
    }

    this.image = spriteImage;
  }

  protected onHighlightPoints(
      pointIndexes: number[], highlightStroke: (i: number) => string,
      favorLabels: (i: number) => boolean) {
    this.colorSprites(highlightStroke);
  }

  protected onSetColorAccessor() {
    if (this.geometry) {
      this.colorSprites(this.highlightStroke);
    }
  }

  protected onMouseClickInternal(e?: MouseEvent) {
    this.resetTraces();
    if (!this.points) {
      return false;
    }
    let selection = this.nearestPoint || null;
    this.defaultPointColor = (selection ? POINT_COLOR_GRAYED : POINT_COLOR);

    this.labeledPoints =
        this.highlightedPoints.filter((id, i) => this.favorLabels(i));

    if (selection && this.dataSet.points[selection].traceIndex) {
      for (let i = 0; i < this.traces.length; i++) {
        this.traces[i].material.opacity = TRACE_DESELECTED_OPACITY;
        this.traces[i].material.needsUpdate = true;
      }
      this.traces[this.dataSet.points[selection].traceIndex].material.opacity =
          TRACE_SELECTED_OPACITY;
      (this.traces[this.dataSet.points[selection].traceIndex].material as
           THREE.LineBasicMaterial)
          .linewidth = TRACE_SELECTED_LINEWIDTH;
      this.traces[this.dataSet.points[selection].traceIndex]
          .material.needsUpdate = true;
    }
    return true;
  }

  protected onSetDayNightMode(isNight: boolean) {
    this.labelColor = (isNight ? LABEL_COLOR_NIGHT : LABEL_COLOR_DAY);
    this.labelStroke = (isNight ? LABEL_STROKE_NIGHT : LABEL_STROKE_DAY);
    this.blending = (isNight ? BLENDING_NIGHT : BLENDING_DAY);
  }

  protected onRecreateScene() {
    this.fog = this.zAccessor ?
        new THREE.Fog(this.backgroundColor) :
        new THREE.Fog(this.backgroundColor, Infinity, Infinity);
    this.scene.fog = this.fog;
    this.addSprites();
    this.colorSprites(null);
    this.addTraces();
  }
  /**
   * Redraws the data. Should be called anytime the accessor method
   * for x and y coordinates changes, which means a new projection
   * exists and the scatter plot should repaint the points.
   */
  protected onUpdate() {
    this.getPointsCoordinates();
    this.updatePositionsArray();
    if (this.geometry) {
      this.render();
    }
  }

  protected onRender() {
    if (!this.dataSet) {
      return;
    }

    this.makeLabels();

    // We want to determine which point the user is hovering over. So, rather
    // than linearly iterating through each point to see if it is under the
    // mouse, we render another set of the points offscreen, where each point is
    // at full opacity and has its id encoded in its color. Then, we see the
    // color of the pixel under the mouse, decode the color, and get the id of
    // of the point.
    let shaderMaterial = this.points.material as THREE.ShaderMaterial;
    let colors = this.geometry.getAttribute('color') as THREE.BufferAttribute;
    // Make shallow copy of the shader options and modify the necessary values.
    let offscreenOptions =
        Object.create(this.materialOptions) as THREE.ShaderMaterialParameters;
    // Since THREE.js errors if we remove the fog, the workaround is to set the
    // near value to very far, so no points have fog.
    this.fog.near = 1000;
    this.fog.far = 10000;
    // Render offscreen as non transparent points (even when we have images).
    offscreenOptions.uniforms.isImage.value = false;
    offscreenOptions.transparent = false;
    offscreenOptions.depthTest = true;
    offscreenOptions.depthWrite = true;
    shaderMaterial.setValues(offscreenOptions);
    // Give each point a unique color.
    let origColArr = colors.array;
    colors.array = this.uniqueColArr;
    colors.needsUpdate = true;
    this.renderer.render(this.scene, this.perspCamera, this.pickingTexture);

    // Change to original color array.
    colors.array = origColArr;
    colors.needsUpdate = true;
    // Bring back the fog.
    if (this.zAccessor && this.geometry) {
      this.setFogDistances();
    }
    offscreenOptions.uniforms.isImage.value = !!this.image;
    // Bring back the standard shader material options.
    shaderMaterial.setValues(this.materialOptions);
  }
}
