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

const FONT_SIZE = 80;
const LABEL_COLOR = 'black';
const LABEL_BACKGROUND = 'white';
const MAX_CANVAS_DIMENSION = 8192;
const NUM_GLYPHS = 256;
const RGB_ELEMENTS_PER_ENTRY = 3;
const XYZ_ELEMENTS_PER_ENTRY = 3;
const UV_ELEMENTS_PER_ENTRY = 2;
const VERTICES_PER_GLYPH = 2 * 3;  // 2 triangles, 3 verts per triangle

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
    varying vec2 vUv;
    attribute vec3 color;
    varying vec3 vColor;
    uniform float camPos;

    float getPointScale() {
      float normalScale =  3.0;
      // Distance to the camera (world coordinates.) This is the scale factor.
      // Note that positions of verts are in world space, scaled so that the
      // lineheight is 1.
      float distToCam = length((modelViewMatrix * vec4(position, 1.0)).z);
      float unscale = distToCam;
      float scale = max(min(unscale * 10.0, normalScale), unscale * 2.0);
      return scale * ${1 /
    FONT_SIZE};
    }

    void main() {
      vUv = uv;
      vColor = color;

      // Make label face camera.
      // 'At' and 'Up' vectors just match that of the camera.
      vec3 Vat = normalize(vec3(
        modelViewMatrix[0][2],
        modelViewMatrix[1][2],
        modelViewMatrix[2][2]));

      vec3 Vup = normalize(vec3(
        modelViewMatrix[0][1],
        modelViewMatrix[1][1],
        modelViewMatrix[2][1]));

      vec3 Vright = normalize(cross(Vup, Vat));
      Vup = cross(Vat, Vright);
      mat4 pointToCamera = mat4(Vright, 0.0, Vup, 0.0, Vat, 0.0, vec3(0), 1.0);

      vec2 posObj = posObj*getPointScale();

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
        vec4 color = vec4(vColor, 1.0);
        gl_FragColor = color + fromTexture;
      }
    }`;

type GlyphTexture = {
  texture: THREE.Texture; lengths: Float32Array; offsets: Float32Array;
};

/**
 * Renders the text labels as 3d geometry in the world.
 */
export class ScatterPlotWebGLVisualizer3DLabels implements
    ScatterPlotWebGLVisualizer {
  private dataSet: DataSet;
  private scene: THREE.Scene;
  private labelAccessor: (index: number) => string;
  private geometry: THREE.BufferGeometry;
  private material: THREE.ShaderMaterial;
  private uniforms: Object;
  private labelsMesh: THREE.Mesh;
  private positions: THREE.BufferAttribute;
  private totalVertexCount: number;
  private labelVertexMap: number[][];
  private glyphTexture: GlyphTexture;

  constructor(scatterPlotWebGL: ScatterPlotWebGL) {
    scatterPlotWebGL.onSelection((s: number[]) => this.onSelectionChanged(s));
    this.createGlyphTexture();

    this.uniforms = {
      texture: {type: 't', value: this.glyphTexture.texture},
      picking: {type: 'bool', value: false},
      camPos: {type: 'float', value: new THREE.Vector3()}
    };

    this.material = new THREE.ShaderMaterial({
      uniforms: this.uniforms,
      transparent: true,
      side: THREE.DoubleSide,
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
      let label = this.labelAccessor(i);
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

  private createLabelGeometry() {
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
      let label = this.labelAccessor(i);
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
        positionObject.setXY(lettersSoFar * VERTICES_PER_GLYPH + 1, left, top);
        positionObject.setXY(lettersSoFar * VERTICES_PER_GLYPH + 2, right, 0);

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
        uv.setXY(lettersSoFar * VERTICES_PER_GLYPH + 1, uLeft, vBottom);
        uv.setXY(lettersSoFar * VERTICES_PER_GLYPH + 2, uRight, vTop);
        uv.setXY(lettersSoFar * VERTICES_PER_GLYPH + 3, uLeft, vBottom);
        uv.setXY(lettersSoFar * VERTICES_PER_GLYPH + 4, uRight, vTop);
        uv.setXY(lettersSoFar * VERTICES_PER_GLYPH + 5, uRight, vBottom);

        lettersSoFar++;
        leftOffset += letterWidth;
      }
    }

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

  onPickingRender(camera: THREE.Camera, cameraTarget: THREE.Vector3) {}

  onRender(renderContext: RenderContext) {
    this.material.uniforms.texture.value = this.glyphTexture.texture;
    this.material.uniforms.picking.value = false;
    this.material.uniforms.camPos.value = renderContext.camera.position;
  }

  onUpdate() {
    this.processLabelVerts();
    let positionArray =
        new Float32Array(this.totalVertexCount * XYZ_ELEMENTS_PER_ENTRY);
    this.positions =
        new THREE.BufferAttribute(positionArray, XYZ_ELEMENTS_PER_ENTRY);

    this.createLabels();
    if (this.labelsMesh && this.scene) {
      this.scene.add(this.labelsMesh);
    }
    for (let i = 0; i < this.dataSet.points.length; i++) {
      let pp = this.dataSet.points[i].projectedPoint;
      this.labelVertexMap[i].forEach((j) => {
        this.positions.setXYZ(j, pp[0], pp[1], pp[2]);
      });
    };
  }

  onResize(newWidth: number, newHeight: number) {}
  onSelectionChanged(selection: number[]) {}
}
