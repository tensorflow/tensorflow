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
import {DataSet} from './scatter';
import {ScatterWebGL} from './scatterWebGL';
import {Point2D} from './vector';

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
  // TODO(nicholsonc): the trailing _ here is temporary.
  // ScatterWebGLPointsCanvasLabels is going to become an aggregate of
  // ScatterWebGL, at which point the naming collision between
  // ScatterWebGL.dataSet and this dataSet will go away, and we can
  // rename dataSet_ here to dataSet.
  private dataSet_: DataSet;

  private gc: CanvasRenderingContext2D;
  private canvas: HTMLCanvasElement;
  private labelCanvasIsCleared = true;
  private image: HTMLImageElement;

  private geometry: THREE.BufferGeometry;
  private materialParams: THREE.ShaderMaterialParameters;
  private positionBuffer: THREE.BufferAttribute;

  private defaultPointColor = POINT_COLOR;
  private sceneIs3D: boolean = true;
  private pointSize2D: number;
  private pointSize3D: number;
  private fog: THREE.Fog;

  private points: THREE.Points;
  private pickingColors: Float32Array;
  private renderColors: Float32Array;
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
  }

  /**
   * Create points, set their locations and actually instantiate the
   * geometry.
   */
  private addSprites(scene: THREE.Scene) {
    // Create geometry.
    this.geometry = new THREE.BufferGeometry();
    this.createBufferAttributes();
    let canvas = document.createElement('canvas');
    let image = this.image || canvas;
    // TODO(b/31390553): Pass sprite dim to the renderer.
    let spriteDim = 28.0;
    let tex = this.createTexture(image);
    let pointSize = (this.sceneIs3D ? this.pointSize3D : this.pointSize2D);
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
      sizeAttenuation: {type: 'bool', value: this.sceneIs3D},
      isImage: {type: 'bool', value: !!this.image},
      pointSize: {type: 'f', value: pointSize}
    };
    this.materialParams = {
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
    let material = new THREE.ShaderMaterial(this.materialParams);

    // And finally initialize it and add it to the scene.
    this.points = new THREE.Points(this.geometry, material);
    scene.add(this.points);
  }

  /**
   * Create line traces between connected points and instantiate the geometry.
   */
  private addTraces(scene: THREE.Scene) {
    if (!this.dataSet_ || !this.dataSet_.traces) {
      return;
    }

    this.traces = [];

    for (let i = 0; i < this.dataSet_.traces.length; i++) {
      let dataTrace = this.dataSet_.traces[i];

      let geometry = new THREE.BufferGeometry();
      let colors: number[] = [];

      for (let j = 0; j < dataTrace.pointIndices.length - 1; j++) {
        this.dataSet_.points[dataTrace.pointIndices[j]].traceIndex = i;
        this.dataSet_.points[dataTrace.pointIndices[j + 1]].traceIndex = i;

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
      scene.add(trace);
    }
  }

  /** Removes all traces from the scene. */
  private removeAllTraces(scene: THREE.Scene) {
    if (!this.traces) {
      return;
    }

    for (let i = 0; i < this.traces.length; i++) {
      scene.remove(this.traces[i]);
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
    let numPts = this.dataSet_.points.length;
    let scaleConstant = 200;
    let logBase = 8;
    // Scale point size inverse-logarithmically to the number of points.
    this.pointSize3D = scaleConstant / Math.log(numPts) / Math.log(logBase);
    this.pointSize2D = this.pointSize3D / 1.5;
  }

  private setFogDistances(nearestPointZ: number, farthestPointZ: number) {
    if (this.sceneIs3D) {
      this.fog.near = nearestPointZ;
      // If there are fewer points we want less fog. We do this
      // by making the "far" value (that is, the distance from the camera to the
      // far edge of the fog) proportional to the number of points.
      let multiplier = 2 -
          Math.min(this.dataSet_.points.length, NUM_POINTS_FOG_THRESHOLD) /
              NUM_POINTS_FOG_THRESHOLD;
      this.fog.far = farthestPointZ * multiplier;
    } else {
      this.fog.near = Infinity;
      this.fog.far = Infinity;
    }
  }

  private getNearFarPoints(
      cameraPos: THREE.Vector3, cameraTarget: THREE.Vector3) {
    let shortestDist: number = Infinity;
    let furthestDist: number = 0;
    let camToTarget = new THREE.Vector3().copy(cameraTarget).sub(cameraPos);
    for (let i = 0; i < this.dataSet_.points.length; i++) {
      let point = this.getProjectedPointFromIndex(i);
      // discard points that are behind the camera
      let camToPoint = new THREE.Vector3().copy(point).sub(cameraPos);
      if (camToTarget.dot(camToPoint) < 0) {
        continue;
      }

      let distToCam = cameraPos.distanceToSquared(point);
      furthestDist = Math.max(furthestDist, distToCam);
      shortestDist = Math.min(shortestDist, distToCam);
    }
    furthestDist = Math.sqrt(furthestDist);
    shortestDist = Math.sqrt(shortestDist);
    return [shortestDist, furthestDist];
  }

  /** Removes all the labels. */
  private removeAllLabels() {
    // If labels are already removed, do not spend compute power to clear the
    // canvas.
    let pixelWidth = this.canvas.width * window.devicePixelRatio;
    let pixelHeight = this.canvas.height * window.devicePixelRatio;
    if (!this.labelCanvasIsCleared) {
      this.gc.clearRect(0, 0, pixelWidth, pixelHeight);
      this.labelCanvasIsCleared = true;
    }
  }

  /**
   * Reset the positions of all labels, and check for overlapps using the
   * collision grid.
   */
  private makeLabels(
      cameraPos: THREE.Vector3, cameraTarget: THREE.Vector3,
      nearestPointZ: number, farthestPointZ: number) {
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
    let dpr = window.devicePixelRatio;
    let pixelWidth = this.canvas.width * dpr;
    let pixelHeight = this.canvas.height * dpr;

    // Bounding box for collision grid.
    let boundingBox:
        BoundingBox = {loX: 0, hiX: pixelWidth, loY: 0, hiY: pixelHeight};

    // Make collision grid with cells proportional to window dimensions.
    let grid =
        new CollisionGrid(boundingBox, pixelWidth / 25, pixelHeight / 50);

    let opacityRange = farthestPointZ - nearestPointZ;
    let camToTarget = new THREE.Vector3().copy(cameraPos).sub(cameraTarget);

    // Setting styles for the labeled font.
    this.gc.lineWidth = 6;
    this.gc.textBaseline = 'middle';
    this.gc.font = (FONT_SIZE * dpr).toString() + 'px roboto';

    let strokeStylePrefix: string;
    let fillStylePrefix: string;
    {
      let ls = new THREE.Color(this.labelStroke).multiplyScalar(255);
      let lc = new THREE.Color(this.labelColor).multiplyScalar(255);
      strokeStylePrefix = 'rgba(' + ls.r + ',' + ls.g + ',' + ls.b + ',';
      fillStylePrefix = 'rgba(' + lc.r + ',' + lc.g + ',' + lc.b + ',';
    }

    for (let i = 0;
         (i < this.labeledPoints.length) && !(numRenderedLabels > SAMPLE_SIZE);
         i++) {
      let index = this.labeledPoints[i];
      let point = this.getProjectedPointFromIndex(index);
      // discard points that are behind the camera
      let camToPoint = new THREE.Vector3().copy(cameraPos).sub(point);
      if (camToTarget.dot(camToPoint) < 0) {
        continue;
      }
      let screenCoords = this.vector3DToScreenCoords(point);
      // Have extra space between neighboring labels. Don't pack too tightly.
      let labelMargin = 2;
      // Shift the label to the right of the point circle.
      let xShift = 3;
      let textBoundingBox = {
        loX: screenCoords[0] + xShift - labelMargin,
        // Computing the width of the font is expensive,
        // so we assume width of 1 at first. Then, if the label doesn't
        // conflict with other labels, we measure the actual width.
        hiX: screenCoords[0] + xShift + 1 + labelMargin,
        loY: screenCoords[1] - labelHeight / 2 - labelMargin,
        hiY: screenCoords[1] + labelHeight / 2 + labelMargin
      };

      if (grid.insert(textBoundingBox, true)) {
        let text = this.labelAccessor(index);
        let labelWidth = this.gc.measureText(text).width;

        // Now, check with properly computed width.
        textBoundingBox.hiX += labelWidth - 1;
        if (grid.insert(textBoundingBox)) {
          let p = new THREE.Vector3(point[0], point[1], point[2]);
          let lenToCamera = cameraPos.distanceTo(p);
          // Opacity is scaled between 0.2 and 1, based on how far a label is
          // from the camera (Unless we are in 2d mode, in which case opacity is
          // just 1!)
          let opacity = this.sceneIs3D ?
              1.2 - (lenToCamera - nearestPointZ) / opacityRange :
              1;
          this.formatLabel(
              text, screenCoords, strokeStylePrefix, fillStylePrefix, opacity);
          numRenderedLabels++;
        }
      }
    }

    if (this.highlightedPoints.length > 0) {
      // Force-draw the first favored point with increased font size.
      let index = this.highlightedPoints[0];
      let point = this.dataSet_.points[index];
      this.gc.font = (FONT_SIZE * dpr * 1.7).toString() + 'px roboto';
      let coords = new THREE.Vector3(
          point.projectedPoint[0], point.projectedPoint[1],
          point.projectedPoint[2]);
      let screenCoords = this.vector3DToScreenCoords(coords);
      let text = this.labelAccessor(index);
      this.formatLabel(
          text, screenCoords, strokeStylePrefix, fillStylePrefix, 255);
    }
  }

  /** Add a specific label to the canvas. */
  private formatLabel(
      text: string, point: Point2D, strokeStylePrefix: string,
      fillStylePrefix: string, opacity: number) {
    this.gc.strokeStyle = strokeStylePrefix + opacity + ')';
    this.gc.fillStyle = fillStylePrefix + opacity + ')';
    this.gc.strokeText(text, point[0] + 4, point[1]);
    this.gc.fillText(text, point[0] + 4, point[1]);
  }

  onResize(newWidth: number, newHeight: number) {
    let dpr = window.devicePixelRatio;
    d3.select(this.canvas)
        .attr('width', newWidth * dpr)
        .attr('height', newHeight * dpr)
        .style({width: newWidth + 'px', height: newHeight + 'px'});
  }

  /**
   * Set up buffer attributes to be used for the points/images.
   */
  private createBufferAttributes() {
    let numPoints = this.dataSet_.points.length;
    this.pickingColors = new Float32Array(numPoints * RGB_NUM_BYTES);
    let colors = new THREE.BufferAttribute(this.pickingColors, RGB_NUM_BYTES);

    // Fill pickingColors with each point's unique id as its color.
    for (let i = 0; i < numPoints; i++) {
      let color = new THREE.Color(i);
      colors.setXYZ(i, color.r, color.g, color.b);
    }

    this.renderColors = new Float32Array(numPoints * RGB_NUM_BYTES);
    colors.array = this.renderColors;

    /** Indices cooresponding to highlighted points. */
    let hiArr = new Float32Array(numPoints);
    let highlights = new THREE.BufferAttribute(hiArr, INDEX_NUM_BYTES);

    /**
     * The actual indices of the points which we use for sizeAttenuation in
     * the shader.
     */
    let indicesShader =
        new THREE.BufferAttribute(new Float32Array(numPoints), 1);

    // Create the array of indices.
    for (let i = 0; i < numPoints; i++) {
      indicesShader.setX(i, this.dataSet_.points[i].dataSourceIndex);
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
    if (!this.geometry) {
      return;
    }
    // Update attributes to change colors
    let colors = this.geometry.getAttribute('color') as THREE.BufferAttribute;
    let highlights =
        this.geometry.getAttribute('isHighlight') as THREE.BufferAttribute;
    for (let i = 0; i < this.dataSet_.points.length; i++) {
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
    for (let i = 0; i < this.dataSet_.points.length; i++) {
      // Set position based on projected point.
      let pp = this.dataSet_.points[i].projectedPoint;
      this.positionBuffer.setXYZ(i, pp[0], pp[1], pp[2]);
    }

    // Update the traces.
    for (let i = 0; i < this.dataSet_.traces.length; i++) {
      let dataTrace = this.dataSet_.traces[i];

      let vertexCount = 0;
      for (let j = 0; j < dataTrace.pointIndices.length - 1; j++) {
        let point1 = this.dataSet_.points[dataTrace.pointIndices[j]];
        let point2 = this.dataSet_.points[dataTrace.pointIndices[j + 1]];

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

      for (let i = 0; i < this.dataSet_.traces.length; i++) {
        this.tracePositionBuffer[i].needsUpdate = true;
      }
    }
  }

  protected removeAllFromScene(scene: THREE.Scene) {
    scene.remove(this.points);
    this.removeAllLabels();
    this.removeAllTraces(scene);
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

  protected onDataSet(dataSet: DataSet, spriteImage: HTMLImageElement) {
    this.dataSet_ = dataSet;
    this.points = null;
    if (this.geometry) {
      this.geometry.dispose();
    }
    this.geometry = null;
    this.calibratePointSize();

    let positions =
        new Float32Array(this.dataSet_.points.length * XYZ_NUM_BYTES);
    this.positionBuffer = new THREE.BufferAttribute(positions, XYZ_NUM_BYTES);

    // Set up the position buffer arrays for each trace.
    for (let i = 0; i < this.dataSet_.traces.length; i++) {
      let dataTrace = this.dataSet_.traces[i];
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

    if (selection && this.dataSet_.points[selection].traceIndex) {
      for (let i = 0; i < this.traces.length; i++) {
        this.traces[i].material.opacity = TRACE_DESELECTED_OPACITY;
        this.traces[i].material.needsUpdate = true;
      }
      let traceIndex = this.dataSet_.points[selection].traceIndex;
      this.traces[traceIndex].material.opacity = TRACE_SELECTED_OPACITY;
      (this.traces[traceIndex].material as THREE.LineBasicMaterial).linewidth =
          TRACE_SELECTED_LINEWIDTH;
      this.traces[traceIndex].material.needsUpdate = true;
    }
    return true;
  }

  protected onSetDayNightMode(isNight: boolean) {
    this.labelColor = (isNight ? LABEL_COLOR_NIGHT : LABEL_COLOR_DAY);
    this.labelStroke = (isNight ? LABEL_STROKE_NIGHT : LABEL_STROKE_DAY);
    this.blending = (isNight ? BLENDING_NIGHT : BLENDING_DAY);
  }

  protected onRecreateScene(
      scene: THREE.Scene, sceneIs3D: boolean, backgroundColor: number) {
    this.sceneIs3D = sceneIs3D;
    this.fog = new THREE.Fog(backgroundColor);
    scene.fog = this.fog;
    this.addSprites(scene);
    this.colorSprites(null);
    this.addTraces(scene);
  }

  /**
   * Redraws the data. Should be called anytime the accessor method
   * for x and y coordinates changes, which means a new projection
   * exists and the scatter plot should repaint the points.
   */
  protected onUpdate() {
    this.updatePositionsArray();
    if (this.geometry) {
      this.render();
    }
  }

  protected onPickingRender(camera: THREE.Camera, cameraTarget: THREE.Vector3) {
    if (!this.geometry) {
      return;
    }
    let colors = this.geometry.getAttribute('color') as THREE.BufferAttribute;
    // Make shallow copy of the shader options and modify the necessary values.
    let pickingState =
        Object.create(this.materialParams) as THREE.ShaderMaterialParameters;
    this.fog.near =
        Infinity;  // Fog changes point colors, which alters the IDs.
    this.fog.far = Infinity;
    pickingState.transparent = false;  // Alpha blending distorts the IDs.
    pickingState.uniforms.isImage.value = false;
    pickingState.depthTest = true;
    pickingState.depthWrite = true;
    (this.points.material as THREE.ShaderMaterial).setValues(pickingState);
    colors.array = this.pickingColors;
    colors.needsUpdate = true;
  }

  protected onRender(camera: THREE.Camera, cameraTarget: THREE.Vector3) {
    if (!this.geometry) {
      return;
    }
    let nearFarPoints = this.getNearFarPoints(camera.position, cameraTarget);
    this.makeLabels(
        camera.position, cameraTarget, nearFarPoints[0], nearFarPoints[1]);
    let shaderMaterial = this.points.material as THREE.ShaderMaterial;
    let colors = this.geometry.getAttribute('color') as THREE.BufferAttribute;
    colors.array = this.renderColors;
    colors.needsUpdate = true;

    this.setFogDistances(nearFarPoints[0], nearFarPoints[1]);

    shaderMaterial.setValues(this.materialParams);
  }
}
