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

import {RenderContext} from './renderContext';
import {DataSet} from './scatterPlot';
import {ScatterPlotWebGL} from './scatterPlotWebGL';
import {ScatterPlotWebGLVisualizer} from './scatterPlotWebGLVisualizer';
import {createTexture} from './util';

const NUM_POINTS_FOG_THRESHOLD = 5000;
const MIN_POINT_SIZE = 5.0;
const IMAGE_SIZE = 30;

const POINT_COLOR = 0x7575D9;
const POINT_COLOR_GRAYED = 0x888888;

// Constants relating to the indices of buffer arrays.
/** Item size of a single point in a bufferArray representing colors */
const RGB_NUM_BYTES = 3;
/** Item size of a single point in a bufferArray representing indices */
const INDEX_NUM_BYTES = 1;
/** Item size of a single point in a bufferArray representing locations */
const XYZ_NUM_BYTES = 3;

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
 * This uses GL point sprites to render
 * the scatter plot dataset, and a 2D HTML canvas to render labels.
 */
export class ScatterPlotWebGLVisualizerSprites implements
    ScatterPlotWebGLVisualizer {
  private dataSet: DataSet;

  private image: HTMLImageElement;
  private geometry: THREE.BufferGeometry;
  private positionBuffer: THREE.BufferAttribute;
  private renderMaterial: THREE.ShaderMaterial;
  private pickingMaterial: THREE.ShaderMaterial;
  private uniforms: Object;

  private defaultPointColor = POINT_COLOR;
  private sceneIs3D: boolean = true;
  private pointSize2D: number;
  private pointSize3D: number;
  private fog: THREE.Fog;

  private points: THREE.Points;
  private pickingColors: Float32Array;
  private renderColors: Float32Array;

  constructor(scatterPlotWebGL: ScatterPlotWebGL) {
    scatterPlotWebGL.onSelection((s: number[]) => this.onSelectionChanged(s));
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
    let tex = createTexture(image);
    let pointSize = (this.sceneIs3D ? this.pointSize3D : this.pointSize2D);
    if (this.image) {
      pointSize = IMAGE_SIZE;
    }

    this.uniforms = {
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

    let haveImage = (this.image != null);

    this.renderMaterial = new THREE.ShaderMaterial({
      uniforms: this.uniforms,
      vertexShader: VERTEX_SHADER,
      fragmentShader: FRAGMENT_SHADER,
      transparent: !haveImage,
      depthTest: haveImage,
      depthWrite: haveImage,
      fog: true,
      blending: (this.image ? THREE.NormalBlending : THREE.MultiplyBlending),
    });

    this.pickingMaterial = new THREE.ShaderMaterial({
      uniforms: this.uniforms,
      vertexShader: VERTEX_SHADER,
      fragmentShader: FRAGMENT_SHADER,
      transparent: false,
      depthTest: true,
      depthWrite: true,
      fog: false,
      blending: (this.image ? THREE.NormalBlending : THREE.MultiplyBlending),
    });

    // And finally initialize it and add it to the scene.
    this.points = new THREE.Points(this.geometry, this.renderMaterial);
    scene.add(this.points);
  }

  private calibratePointSize() {
    let numPts = this.dataSet.points.length;
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
          Math.min(this.dataSet.points.length, NUM_POINTS_FOG_THRESHOLD) /
              NUM_POINTS_FOG_THRESHOLD;
      this.fog.far = farthestPointZ * multiplier;
    } else {
      this.fog.near = Infinity;
      this.fog.far = Infinity;
    }
  }

  /**
   * Set up buffer attributes to be used for the points/images.
   */
  private createBufferAttributes() {
    let numPoints = this.dataSet.points.length;
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
      indicesShader.setX(i, this.dataSet.points[i].dataSourceIndex);
    }

    // Finally, add all attributes to the geometry.
    this.geometry.addAttribute('position', this.positionBuffer);
    this.positionBuffer.needsUpdate = true;
    this.geometry.addAttribute('color', colors);
    this.geometry.addAttribute('vertexIndex', indicesShader);
    this.geometry.addAttribute('isHighlight', highlights);

    this.colorSprites(null);
    this.highlightSprites(null, null);
  }

  private colorSprites(colorAccessor: (index: number) => string) {
    if (this.geometry == null) {
      return;
    }
    let colors = this.geometry.getAttribute('color') as THREE.BufferAttribute;
    colors.array = this.renderColors;
    let getColor: (index: number) => string = (() => undefined);
    if (this.image == null) {
      getColor =
          colorAccessor ? colorAccessor : () => (this.defaultPointColor as any);
    }
    for (let i = 0; i < this.dataSet.points.length; i++) {
      let color = new THREE.Color(getColor(i));
      colors.setXYZ(i, color.r, color.g, color.b);
    }
    colors.needsUpdate = true;
  }

  private highlightSprites(
      highlightedPoints: number[], highlightStroke: (index: number) => string) {
    if (this.geometry == null) {
      return;
    }
    let highlights =
        this.geometry.getAttribute('isHighlight') as THREE.BufferAttribute;
    for (let i = 0; i < this.dataSet.points.length; i++) {
      highlights.setX(i, 0.0);
    }
    if (highlightedPoints && highlightStroke) {
      let colors = this.geometry.getAttribute('color') as THREE.BufferAttribute;
      // Traverse in reverse so that the point we are hovering over
      // (highlightedPoints[0]) is painted last.
      for (let i = highlightedPoints.length - 1; i >= 0; i--) {
        let assocPoint = highlightedPoints[i];
        let color = new THREE.Color(highlightStroke(i));
        // Fill colors array (single array of numPoints*3 elements,
        // triples of which refer to the rgb values of a single vertex).
        colors.setXYZ(assocPoint, color.r, color.g, color.b);
        highlights.setX(assocPoint, 1.0);
      }
      colors.needsUpdate = true;
    }
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

    if (this.geometry) {
      this.positionBuffer.needsUpdate = true;
    }
  }

  removeAllFromScene(scene: THREE.Scene) {
    scene.remove(this.points);
  }

  onDataSet(dataSet: DataSet, spriteImage: HTMLImageElement) {
    this.dataSet = dataSet;
    this.image = spriteImage;
    this.points = null;
    if (this.geometry) {
      this.geometry.dispose();
    }
    this.geometry = null;
    this.calibratePointSize();

    let positions =
        new Float32Array(this.dataSet.points.length * XYZ_NUM_BYTES);
    this.positionBuffer = new THREE.BufferAttribute(positions, XYZ_NUM_BYTES);
  }

  onSelectionChanged(selection: number[]) {
    this.defaultPointColor =
        (selection.length > 0) ? POINT_COLOR_GRAYED : POINT_COLOR;
  }

  onRecreateScene(
      scene: THREE.Scene, sceneIs3D: boolean, backgroundColor: number) {
    this.sceneIs3D = sceneIs3D;
    this.fog = new THREE.Fog(backgroundColor);
    scene.fog = this.fog;
    this.addSprites(scene);
    this.colorSprites(null);
    this.highlightSprites(null, null);
  }

  onUpdate() {
    this.updatePositionsArray();
  }

  onResize(newWidth: number, newHeight: number) {}
  onSetLabelAccessor(labelAccessor: (index: number) => string) {}

  onPickingRender(camera: THREE.Camera, cameraTarget: THREE.Vector3) {
    if (!this.geometry) {
      return;
    }
    // Fog changes point colors, which alters the IDs.
    this.fog.near = Infinity;
    this.fog.far = Infinity;

    this.points.material = this.pickingMaterial;
    this.pickingMaterial.uniforms.isImage.value = false;

    let colors = this.geometry.getAttribute('color') as THREE.BufferAttribute;
    colors.array = this.pickingColors;
    colors.needsUpdate = true;
  }

  onRender(rc: RenderContext) {
    if (!this.geometry) {
      return;
    }
    this.colorSprites(rc.colorAccessor);
    this.highlightSprites(rc.highlightedPoints, rc.highlightStroke);

    this.setFogDistances(
        rc.nearestCameraSpacePointZ, rc.farthestCameraSpacePointZ);

    this.points.material = this.renderMaterial;
    this.renderMaterial.uniforms.isImage.value = !!this.image;

    let colors = this.geometry.getAttribute('color') as THREE.BufferAttribute;
    colors.array = this.renderColors;
    colors.needsUpdate = true;
  }
}
