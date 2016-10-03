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
import {SelectionContext} from './selectionContext';
import {createTexture} from './util';

const FONT_SIZE = 80;
const ONE_OVER_FONT_SIZE = 1 / FONT_SIZE;
const LABEL_COLOR = 'black';
const LABEL_BACKGROUND = 'white';
const MAX_CANVAS_DIMENSION = 8192;
const NUM_GLYPHS = 256;
const RGB_ELEMENTS_PER_ENTRY = 3;
const XYZ_ELEMENTS_PER_ENTRY = 3;
const UV_ELEMENTS_PER_ENTRY = 2;
const VERTICES_PER_GLYPH = 2 * 3;  // 2 triangles, 3 verts per triangle
const POINT_COLOR = 0xFFFFFF;

/**
 * Each label is made up of triangles (two per letter.) Each vertex, then, is
 * the corner of one of these triangles (and thus the corner of a letter
 * rectangle.)
 * Each has the following attributes:
 *    posObj: The (x, y) position of the vertex within the label, where the
 *            bottom center of the word is positioned at (0, 0);
 *    position: The position of the label in worldspace.
 *    vUv: The (u, v) coordinates that index into the glyphs sheet (range 0, 1.)
 *    color: The color of the label (matches the cooresponding point's color.)
 *    wordShown: Boolean. Whether or not the label is visible.
 */

const VERTEX_SHADER = `
    attribute vec2 posObj;
    attribute vec3 color;
    varying vec2 vUv;
    varying vec3 vColor;

    float getPointScale() {
      float normalScale = 3.0;
      // Distance to the camera (world coordinates.) This is the scale factor.
      // Note that positions of verts are in world space, scaled so that the
      // lineheight is 1.
      vec4 posCamSpace = modelViewMatrix * vec4(position, 1.0);
      float distToCam = length(posCamSpace.z);
      float scale = max(min(distToCam * 10.0, normalScale), distToCam * 2.0);
      return scale * ${ONE_OVER_FONT_SIZE};
    }

    void main() {
      vUv = uv;
      vColor = color;

      // Rotate label to face camera.

      vec4 vRight = vec4(
        modelViewMatrix[0][0], modelViewMatrix[1][0], modelViewMatrix[2][0], 0);

      vec4 vUp = vec4(
        modelViewMatrix[0][1], modelViewMatrix[1][1], modelViewMatrix[2][1], 0);

      vec4 vAt = -vec4(
        modelViewMatrix[0][2], modelViewMatrix[1][2], modelViewMatrix[2][2], 0);

      mat4 pointToCamera = mat4(vRight, vUp, vAt, vec4(0, 0, 0, 1));

      vec2 posObj = posObj * getPointScale();

      vec4 posRotated = pointToCamera * vec4(posObj, 0.00001, 1.0);
      vec4 mvPosition = modelViewMatrix * (vec4(position, 0.0) + posRotated);
      gl_Position = projectionMatrix * mvPosition;
    }`;

const FRAGMENT_SHADER = `
    uniform sampler2D texture;
    uniform bool picking;
    varying vec2 vUv;
    varying vec3 vColor;

    void main() {
      if (picking) {
        gl_FragColor = vec4(vColor, 1.0);
      } else {
        vec4 fromTexture = texture2D(texture, vUv);
        gl_FragColor = vec4(vColor, 1.0) * fromTexture;
      }
    }`;

type GlyphTexture = {
  texture: THREE.Texture; lengths: Float32Array; offsets: Float32Array;
};

/**
 * Renders the text labels as 3d geometry in the world.
 */
export class ScatterPlotVisualizer3DLabels implements ScatterPlotVisualizer {
  private dataSet: DataSet;
  private scene: THREE.Scene;
  private labelAccessor: (index: number) => string;
  private geometry: THREE.BufferGeometry;
  private pickingColors: Float32Array;
  private renderColors: Float32Array;
  private material: THREE.ShaderMaterial;
  private uniforms: Object;
  private labelsMesh: THREE.Mesh;
  private positions: THREE.BufferAttribute;
  private defaultPointColor = POINT_COLOR;
  private totalVertexCount: number;
  private labelVertexMap: number[][];
  private glyphTexture: GlyphTexture;

  constructor(selectionContext: SelectionContext) {
    selectionContext.registerSelectionChangedListener(
        s => this.onSelectionChanged(s));
    this.createGlyphTexture();

    this.uniforms = {
      texture: {type: 't', value: this.glyphTexture.texture},
      picking: {type: 'bool', value: false},
    };

    this.material = new THREE.ShaderMaterial({
      uniforms: this.uniforms,
      transparent: true,
      vertexShader: VERTEX_SHADER,
      fragmentShader: FRAGMENT_SHADER,
    });
  }

  private createGlyphTexture() {
    if (this.glyphTexture) {
      this.glyphTexture.texture.dispose();
    }

    let canvas = document.createElement('canvas');
    canvas.width = MAX_CANVAS_DIMENSION;
    canvas.height = FONT_SIZE;
    let ctx = canvas.getContext('2d');
    ctx.font = 'bold ' + FONT_SIZE * 0.75 + 'px roboto';
    ctx.textBaseline = 'top';
    ctx.fillStyle = LABEL_BACKGROUND;
    ctx.rect(0, 0, canvas.width, canvas.height);
    ctx.fill();
    ctx.fillStyle = LABEL_COLOR;
    let spaceOffset = ctx.measureText(' ').width;
    // For each letter, store length, position at the encoded index.
    let glyphLengths = new Float32Array(NUM_GLYPHS);
    let glyphOffset = new Float32Array(NUM_GLYPHS);
    let leftCoord = 0;
    for (let i = 0; i < NUM_GLYPHS; i++) {
      let text = ' ' + String.fromCharCode(i);
      let textLength = ctx.measureText(text).width;
      glyphLengths[i] = textLength - spaceOffset;
      glyphOffset[i] = leftCoord;
      ctx.fillText(text, leftCoord - spaceOffset, 0);
      leftCoord += textLength;
    }
    let tex = createTexture(canvas);
    this.glyphTexture = {
      texture: tex,
      lengths: glyphLengths,
      offsets: glyphOffset
    };
  }

  private processLabelVerts() {
    let numTotalLetters = 0;
    this.labelVertexMap = [];
    for (let i = 0; i < this.dataSet.points.length; i++) {
      let label: string = this.labelAccessor(i).toString();
      let vertsArray: number[] = [];
      for (let j = 0; j < label.length; j++) {
        for (let k = 0; k < VERTICES_PER_GLYPH; k++) {
          vertsArray.push(numTotalLetters * VERTICES_PER_GLYPH + k);
        }
        numTotalLetters++;
      }
      this.labelVertexMap.push(vertsArray);
    }
    this.totalVertexCount = numTotalLetters * VERTICES_PER_GLYPH;
  }

  private createColorBuffers() {
    let numPoints = this.dataSet.points.length;
    this.pickingColors =
        new Float32Array(this.totalVertexCount * RGB_ELEMENTS_PER_ENTRY);
    this.renderColors =
        new Float32Array(this.totalVertexCount * RGB_ELEMENTS_PER_ENTRY);
    for (let i = 0; i < numPoints; i++) {
      let color = new THREE.Color(i);
      this.labelVertexMap[i].forEach((j) => {
        this.pickingColors[RGB_ELEMENTS_PER_ENTRY * j] = color.r;
        this.pickingColors[RGB_ELEMENTS_PER_ENTRY * j + 1] = color.g;
        this.pickingColors[RGB_ELEMENTS_PER_ENTRY * j + 2] = color.b;
        this.renderColors[RGB_ELEMENTS_PER_ENTRY * j] = 1.0;
        this.renderColors[RGB_ELEMENTS_PER_ENTRY * j + 1] = 1.0;
        this.renderColors[RGB_ELEMENTS_PER_ENTRY * j + 2] = 1.0;
      });
    }
  }

  private createLabelGeometry() {
    this.processLabelVerts();
    this.createColorBuffers();

    let positionArray =
        new Float32Array(this.totalVertexCount * XYZ_ELEMENTS_PER_ENTRY);
    this.positions =
        new THREE.BufferAttribute(positionArray, XYZ_ELEMENTS_PER_ENTRY);

    let posArray =
        new Float32Array(this.totalVertexCount * XYZ_ELEMENTS_PER_ENTRY);
    let uvArray =
        new Float32Array(this.totalVertexCount * UV_ELEMENTS_PER_ENTRY);
    let colorsArray =
        new Float32Array(this.totalVertexCount * RGB_ELEMENTS_PER_ENTRY);
    let positionObject = new THREE.BufferAttribute(posArray, 2);
    let uv = new THREE.BufferAttribute(uvArray, UV_ELEMENTS_PER_ENTRY);
    let colors = new THREE.BufferAttribute(colorsArray, RGB_ELEMENTS_PER_ENTRY);

    this.geometry = new THREE.BufferGeometry();
    this.geometry.addAttribute('posObj', positionObject);
    this.geometry.addAttribute('position', this.positions);
    this.geometry.addAttribute('uv', uv);
    this.geometry.addAttribute('color', colors);

    let lettersSoFar = 0;
    for (let i = 0; i < this.dataSet.points.length; i++) {
      let label: string = this.labelAccessor(i).toString();
      let leftOffset = 0;
      // Determine length of word in pixels.
      for (let j = 0; j < label.length; j++) {
        let letterCode = label.charCodeAt(j);
        leftOffset += this.glyphTexture.lengths[letterCode];
      }
      leftOffset /= -2;  // centers text horizontally around the origin
      for (let j = 0; j < label.length; j++) {
        let letterCode = label.charCodeAt(j);
        let letterWidth = this.glyphTexture.lengths[letterCode];
        let scale = FONT_SIZE;
        let right = (leftOffset + letterWidth) / scale;
        let left = (leftOffset) / scale;
        let top = FONT_SIZE / scale;

        // First triangle
        positionObject.setXY(lettersSoFar * VERTICES_PER_GLYPH + 0, left, 0);
        positionObject.setXY(lettersSoFar * VERTICES_PER_GLYPH + 1, right, 0);
        positionObject.setXY(lettersSoFar * VERTICES_PER_GLYPH + 2, left, top);

        // Second triangle
        positionObject.setXY(lettersSoFar * VERTICES_PER_GLYPH + 3, left, top);
        positionObject.setXY(lettersSoFar * VERTICES_PER_GLYPH + 4, right, 0);
        positionObject.setXY(lettersSoFar * VERTICES_PER_GLYPH + 5, right, top);

        // Set UVs based on letter.
        let uLeft = (this.glyphTexture.offsets[letterCode]);
        let uRight = (this.glyphTexture.offsets[letterCode] + letterWidth);
        // Scale so that uvs lie between 0 and 1 on the texture.
        uLeft /= MAX_CANVAS_DIMENSION;
        uRight /= MAX_CANVAS_DIMENSION;
        let vTop = 1;
        let vBottom = 0;
        uv.setXY(lettersSoFar * VERTICES_PER_GLYPH + 0, uLeft, vTop);
        uv.setXY(lettersSoFar * VERTICES_PER_GLYPH + 1, uRight, vTop);
        uv.setXY(lettersSoFar * VERTICES_PER_GLYPH + 2, uLeft, vBottom);
        uv.setXY(lettersSoFar * VERTICES_PER_GLYPH + 3, uLeft, vBottom);
        uv.setXY(lettersSoFar * VERTICES_PER_GLYPH + 4, uRight, vTop);
        uv.setXY(lettersSoFar * VERTICES_PER_GLYPH + 5, uRight, vBottom);

        lettersSoFar++;
        leftOffset += letterWidth;
      }
    }

    for (let i = 0; i < this.dataSet.points.length; i++) {
      let pp = this.dataSet.points[i].projectedPoint;
      this.labelVertexMap[i].forEach((j) => {
        this.positions.setXYZ(j, pp[0], pp[1], pp[2]);
      });
    };

    this.labelsMesh = new THREE.Mesh(this.geometry, this.material);
  }

  private destroyLabels() {
    if (this.labelsMesh) {
      if (this.scene) {
        this.scene.remove(this.labelsMesh);
      }
      this.geometry.dispose();
      this.labelsMesh = null;
    }
  }

  private createLabels() {
    this.destroyLabels();
    if (this.labelAccessor) {
      this.createLabelGeometry();
    }
  }

  private colorSprites(colorAccessor: (index: number) => string) {
    if (this.geometry == null || this.dataSet == null) {
      return;
    }
    let colors = this.geometry.getAttribute('color') as THREE.BufferAttribute;
    colors.array = this.renderColors;
    let getColor: (index: number) => string =
        colorAccessor ? colorAccessor : () => (this.defaultPointColor as any);
    for (let i = 0; i < this.dataSet.points.length; i++) {
      let color = new THREE.Color(getColor(i));
      this.labelVertexMap[i].forEach((j) => {
        colors.setXYZ(j, color.r, color.g, color.b);
      });
    }
    colors.needsUpdate = true;
  }

  private highlightSprites(
      highlightedPoints: number[], highlightStroke: (index: number) => string) {
    if (this.geometry == null || this.dataSet == null) {
      return;
    }
    if (highlightedPoints && highlightStroke) {
      let colors = this.geometry.getAttribute('color') as THREE.BufferAttribute;
      for (let i = highlightedPoints.length - 1; i >= 0; --i) {
        let assocPoint = highlightedPoints[i];
        let color = new THREE.Color(highlightStroke(i));
        this.labelVertexMap[assocPoint].forEach((j) => {
          colors.setXYZ(j, color.r, color.g, color.b);
        });
      }
      colors.needsUpdate = true;
    }
  }

  onRecreateScene(
      scene: THREE.Scene, sceneIs3D: boolean, backgroundColor: number) {
    this.scene = scene;
    if (this.labelsMesh == null) {
      this.createLabels();
    }
    if (this.labelsMesh) {
      scene.add(this.labelsMesh);
    }
  }

  removeAllFromScene(scene: THREE.Scene) {
    this.destroyLabels();
  }

  onSetLabelAccessor(labelAccessor: (index: number) => string) {
    this.labelAccessor = labelAccessor;
    this.onUpdate();
  }

  onDataSet(dataSet: DataSet, spriteImage: HTMLImageElement) {
    this.dataSet = dataSet;
  }

  onPickingRender(camera: THREE.Camera, cameraTarget: THREE.Vector3) {
    this.material.uniforms.texture.value = this.glyphTexture.texture;
    this.material.uniforms.picking.value = true;

    let colors = this.geometry.getAttribute('color') as THREE.BufferAttribute;
    colors.array = this.pickingColors;
    colors.needsUpdate = true;
  }

  onRender(rc: RenderContext) {
    this.colorSprites(rc.colorAccessor);
    this.highlightSprites(rc.highlightedPoints, rc.highlightStroke);

    this.material.uniforms.texture.value = this.glyphTexture.texture;
    this.material.uniforms.picking.value = false;

    let colors = this.geometry.getAttribute('color') as THREE.BufferAttribute;
    colors.array = this.renderColors;
    colors.needsUpdate = true;
  }

  onUpdate() {
    this.createLabels();
    if (this.labelsMesh && this.scene) {
      this.scene.add(this.labelsMesh);
    }
  }

  onSelectionChanged(selection: number[]) {}

  onResize(newWidth: number, newHeight: number) {}
}
