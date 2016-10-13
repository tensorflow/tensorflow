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
import {ScatterPlotVisualizer} from './scatterPlotVisualizer';
import {createTexture} from './util';

const NUM_POINTS_FOG_THRESHOLD = 5000;
const MIN_POINT_SIZE = 5.0;
const IMAGE_SIZE = 30;

// Constants relating to the indices of buffer arrays.
const RGB_NUM_ELEMENTS = 3;
const INDEX_NUM_ELEMENTS = 1;
const XYZ_NUM_ELEMENTS = 3;

const VERTEX_SHADER = `
  // Index of the specific vertex (passed in as bufferAttribute), and the
  // variable that will be used to pass it to the fragment shader.
  attribute float vertexIndex;
  attribute vec3 color;
  attribute float scaleFactor;

  varying vec2 xyIndex;
  varying vec3 vColor;

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
    vec4 cameraSpacePos = modelViewMatrix * vec4(position, 1.0);

    // Project vertex in camera-space to screen coordinates using the camera's
    // projection matrix.
    gl_Position = projectionMatrix * cameraSpacePos;

    // Create size attenuation (if we're in 3D mode) by making the size of
    // each point inversly proportional to its distance to the camera.
    float outputPointSize = pointSize;
    if (sizeAttenuation) {
      outputPointSize = -pointSize / cameraSpacePos.z;
    }

    gl_PointSize =
      max(outputPointSize * scaleFactor, ${MIN_POINT_SIZE.toFixed(1)});
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
      float uvLenSquared = dot(uv, uv);
      if (uvLenSquared > (0.5 * 0.5)) {
        discard;
      }

      // If the point is not an image, just color it.
      gl_FragColor = vec4(vColor, 1.0);
    }
    ${THREE.ShaderChunk['fog_fragment']}
  }`;

const FRAGMENT_SHADER_PICKING = `
  varying vec2 xyIndex;
  varying vec3 vColor;

  uniform bool isImage;

  void main() {
    if (isImage) {
      gl_FragColor = vec4(vColor, 1);
    } else {
      vec2 pointCenterToHere = gl_PointCoord.xy - vec2(0.5, 0.5);
      float lenSquared = dot(pointCenterToHere, pointCenterToHere);
      if (lenSquared > (0.5 * 0.5)) {
        discard;
      }
      gl_FragColor = vec4(vColor, 1);
    }
  }`;

/**
 * Uses GL point sprites to render the dataset.
 */
export class ScatterPlotVisualizerSprites implements ScatterPlotVisualizer {
  private dataSet: DataSet;

  private image: HTMLImageElement;
  private geometry: THREE.BufferGeometry;
  private renderMaterial: THREE.ShaderMaterial;
  private pickingMaterial: THREE.ShaderMaterial;
  private uniforms: Object;

  private sceneIs3D: boolean = true;
  private pointSize2D: number;
  private pointSize3D: number;
  private fog: THREE.Fog;

  private points: THREE.Points;
  private pickingColors: Float32Array;
  private renderColors: Float32Array;

  /**
   * Create points, set their locations and actually instantiate the
   * geometry.
   */
  private addSprites(scene: THREE.Scene) {
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
      isImage: {type: 'bool', value: (this.image != null)},
      pointSize: {type: 'f', value: pointSize}
    };

    const haveImage = (this.image != null);

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
      fragmentShader: FRAGMENT_SHADER_PICKING,
      transparent: false,
      depthTest: true,
      depthWrite: true,
      fog: false,
      blending: (this.image ? THREE.NormalBlending : THREE.MultiplyBlending),
    });

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

    // Fill pickingColors with each point's unique id as its color.
    this.pickingColors = new Float32Array(numPoints * RGB_NUM_ELEMENTS);
    {
      let dst = 0;
      for (let i = 0; i < numPoints; i++) {
        const c = new THREE.Color(i);
        this.pickingColors[dst++] = c.r;
        this.pickingColors[dst++] = c.g;
        this.pickingColors[dst++] = c.b;
      }
    }

    let colors =
        new THREE.BufferAttribute(this.pickingColors, RGB_NUM_ELEMENTS);
    let scaleFactors = new THREE.BufferAttribute(
        new Float32Array(numPoints), INDEX_NUM_ELEMENTS);
    let positions = new THREE.BufferAttribute(
        new Float32Array(numPoints * XYZ_NUM_ELEMENTS), XYZ_NUM_ELEMENTS);

    /**
     * The actual indices of the points which we use for sizeAttenuation in
     * the shader.
     */
    let indicesShader =
        new THREE.BufferAttribute(new Float32Array(numPoints), 1);

    // Create the array of indices.
    for (let i = 0; i < numPoints; i++) {
      indicesShader.setX(i, this.dataSet.points[i].index);
    }

    this.geometry.addAttribute('position', positions);
    this.geometry.addAttribute('color', colors);
    this.geometry.addAttribute('vertexIndex', indicesShader);
    this.geometry.addAttribute('scaleFactor', scaleFactors);
  }

  private updatePositionsArray() {
    if (this.geometry == null) {
      return;
    }
    const n = this.dataSet.points.length;
    const positions =
        this.geometry.getAttribute('position') as THREE.BufferAttribute;
    positions.array = new Float32Array(n * XYZ_NUM_ELEMENTS);
    for (let i = 0; i < n; i++) {
      let pp = this.dataSet.points[i].projectedPoint;
      positions.setXYZ(i, pp[0], pp[1], pp[2]);
    }
    positions.needsUpdate = true;
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
  }

  onRecreateScene(
      scene: THREE.Scene, sceneIs3D: boolean, backgroundColor: number) {
    this.sceneIs3D = sceneIs3D;
    this.fog = new THREE.Fog(backgroundColor);
    scene.fog = this.fog;
    if (this.dataSet) {
      this.addSprites(scene);
      this.updatePositionsArray();
    }
  }

  onUpdate() {
    this.updatePositionsArray();
  }

  onResize(newWidth: number, newHeight: number) {}
  onSetLabelAccessor(labelAccessor: (index: number) => string) {}

  onPickingRender(rc: RenderContext) {
    if (!this.geometry) {
      return;
    }

    this.points.material = this.pickingMaterial;

    let colors = this.geometry.getAttribute('color') as THREE.BufferAttribute;
    colors.array = this.pickingColors;
    colors.needsUpdate = true;

    let scaleFactors =
        this.geometry.getAttribute('scaleFactor') as THREE.BufferAttribute;
    scaleFactors.array = rc.pointScaleFactors;
    scaleFactors.needsUpdate = true;
  }

  onRender(rc: RenderContext) {
    if (!this.geometry) {
      return;
    }
    this.setFogDistances(
        rc.nearestCameraSpacePointZ, rc.farthestCameraSpacePointZ);

    this.points.material = this.renderMaterial;
    this.renderMaterial.uniforms.isImage.value = !!this.image;

    let colors = this.geometry.getAttribute('color') as THREE.BufferAttribute;
    this.renderColors = rc.pointColors;
    colors.array = this.renderColors;
    colors.needsUpdate = true;

    let scaleFactors =
        this.geometry.getAttribute('scaleFactor') as THREE.BufferAttribute;
    scaleFactors.array = rc.pointScaleFactors;
    scaleFactors.needsUpdate = true;
  }
}
