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
import {DataSet, Mode, OnHoverListener, OnSelectionListener, Point3D, Scatter} from './scatter';
import {shuffle} from './util';

const FONT_SIZE = 10;

// Colors (in various necessary formats).
const BACKGROUND_COLOR_DAY = 0xffffff;
const BACKGROUND_COLOR_NIGHT = 0x000000;
const AXIS_COLOR = 0xb3b3b3;
const LABEL_COLOR_DAY = 0x000000;
const LABEL_COLOR_NIGHT = 0xffffff;
const LABEL_STROKE_DAY = 0xffffff;
const LABEL_STROKE_NIGHT = 0x000000;
const POINT_COLOR = 0x7575D9;
const POINT_COLOR_GRAYED = 0x888888;
const BLENDING_DAY = THREE.MultiplyBlending;
const BLENDING_NIGHT = THREE.AdditiveBlending;
const TRACE_START_HUE = 60;
const TRACE_END_HUE = 360;
const TRACE_SATURATION = 1;
const TRACE_LIGHTNESS = .3;
const TRACE_DEFAULT_OPACITY = .2;
const TRACE_DEFAULT_LINEWIDTH = 2;
const TRACE_SELECTED_OPACITY = .9;
const TRACE_SELECTED_LINEWIDTH = 3;
const TRACE_DESELECTED_OPACITY = .05;

// Various distance bounds.
const MAX_ZOOM = 10;
const MIN_ZOOM = .05;
const NUM_POINTS_FOG_THRESHOLD = 5000;
const MIN_POINT_SIZE = 5.0;
const IMAGE_SIZE = 30;

// Constants relating to the camera parameters.
/** Camera frustum vertical field of view. */
const FOV = 70;
const NEAR = 0.01;
const FAR = 100;

// Constants relating to the indices of buffer arrays.
/** Item size of a single point in a bufferArray representing colors */
const RGB_NUM_BYTES = 3;
/** Item size of a single point in a bufferArray representing indices */
const INDEX_NUM_BYTES = 1;
/** Item size of a single point in a bufferArray representing locations */
const XYZ_NUM_BYTES = 3;

// Key presses.
const SHIFT_KEY = 16;
const CTRL_KEY = 17;

// Original positions of camera and camera target, in 2d and 3d
const POS_3D = {
  x: 1.5,
  y: 1.5,
  z: 1.5
};

// Target for the camera in 3D is the center of the 1, 1, 1 square, as all our
// data is scaled to this.
const TAR_3D = {
  x: 0,
  y: 0,
  z: 0
};

const POS_2D = {
  x: 0,
  y: 0,
  z: 2
};

// In 3D, the target is the center of the xy plane.
const TAR_2D = {
  x: 0,
  y: 0,
  z: 0
};

// The maximum number of labels to draw to keep the frame rate up.
const SAMPLE_SIZE = 10000;

// Shaders for images.
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

export class ScatterWebGL implements Scatter {
  // MISC UNINITIALIZED VARIABLES.

  // Colors and options that are changed between Day and Night modes.
  private backgroundColor: number;
  private labelColor: number;
  private labelStroke: number;
  private blending: THREE.Blending;
  private isNight: boolean;

  // THREE.js necessities.
  private scene: THREE.Scene;
  private perspCamera: THREE.PerspectiveCamera;
  private renderer: THREE.WebGLRenderer;
  private cameraControls: any;
  private light: THREE.PointLight;
  private fog: THREE.Fog;

  // Data structures (and THREE.js objects) associated with points.
  private geometry: THREE.BufferGeometry;
  /** Texture for rendering offscreen in order to enable interactive hover. */
  private pickingTexture: THREE.WebGLRenderTarget;
  /** Array of unique colors for each point used in detecting hover. */
  private uniqueColArr: Float32Array;
  private materialOptions: THREE.ShaderMaterialParameters;
  private points: THREE.Points;
  private traces: THREE.Line[];
  private dataSet: DataSet;
  private shuffledData: number[];
  /** Holds the indexes of the points to be labeled. */
  private labeledPoints: number[] = [];
  private highlightedPoints: number[] = [];
  private nearestPoint: number;
  private pointSize2D: number;
  private pointSize3D: number;
  /** The buffer attribute that holds the positions of the points. */
  private positionBufferArray: THREE.BufferAttribute;

  // Accessors for rendering and labeling the points.
  private xAccessor: (index: number) => number;
  private yAccessor: (index: number) => number;
  private zAccessor: (index: number) => number;
  private labelAccessor: (index: number) => string;
  private colorAccessor: (index: number) => string;
  private highlightStroke: (i: number) => string;
  private favorLabels: (i: number) => boolean;

  // Scaling functions for each axis.
  private xScale: d3.scale.Linear<number, number>;
  private yScale: d3.scale.Linear<number, number>;
  private zScale: d3.scale.Linear<number, number>;

  // Listeners
  private onHoverListeners: OnHoverListener[] = [];
  private onSelectionListeners: OnSelectionListener[] = [];
  private lazySusanAnimation: number;

  // Other variables associated with layout and interaction.
  private height: number;
  private width: number;
  private mode: Mode;
  /** Whether the user has turned labels on or off. */
  private labelsAreOn = true;
  /** Whether the label canvas has been already cleared. */
  private labelCanvasIsCleared = true;

  private animating: boolean;
  private axis3D: THREE.AxisHelper;
  private axis2D: THREE.LineSegments;
  private dpr: number;        // The device pixelratio
  private selecting = false;  // whether or not we are selecting points.
  private mouseIsDown = false;
  // Whether the current click sequence contains a drag, so we can determine
  // whether to update the selection.
  private isDragSequence = false;
  private selectionSphere: THREE.Mesh;
  private image: HTMLImageElement;
  private animationID: number;
  /** Color of any point not selected (or NN of selected) */
  private defaultPointColor = POINT_COLOR;

  // HTML elements.
  private gc: CanvasRenderingContext2D;
  private containerNode: HTMLElement;
  private canvas: HTMLCanvasElement;

  /** Get things started up! */
  constructor(
      container: d3.Selection<any>, labelAccessor: (index: number) => string) {
    this.labelAccessor = labelAccessor;
    this.xScale = d3.scale.linear();
    this.yScale = d3.scale.linear();
    this.zScale = d3.scale.linear();

    // Set up non-THREEjs layout.
    this.containerNode = container.node() as HTMLElement;
    this.getLayoutValues();

    // For now, labels are drawn on this transparent canvas with no touch events
    // rather than being rendered in webGL.
    this.canvas = container.append('canvas').node() as HTMLCanvasElement;
    this.gc = this.canvas.getContext('2d');
    d3.select(this.canvas).style({position: 'absolute', left: 0, top: 0});
    this.canvas.style.pointerEvents = 'none';

    // Set up THREE.js.
    this.createSceneAndRenderer();
    this.setDayNightMode(false);
    this.createLight();
    this.makeCamera();
    this.resize(false);
    // Render now so no black background appears during startup.
    this.renderer.render(this.scene, this.perspCamera);
    // Add interaction listeners.
    this.addInteractionListeners();
  }

  // SET UP
  private addInteractionListeners() {
    this.containerNode.addEventListener(
        'mousemove', this.onMouseMove.bind(this));
    this.containerNode.addEventListener(
        'mousedown', this.onMouseDown.bind(this));
    this.containerNode.addEventListener('mouseup', this.onMouseUp.bind(this));
    this.containerNode.addEventListener('click', this.onClick.bind(this));
    window.addEventListener('keydown', this.onKeyDown.bind(this), false);
    window.addEventListener('keyup', this.onKeyUp.bind(this), false);
  }

  /** Updates the positions buffer array to reflect the actual data. */
  private updatePositionsArray() {
    for (let i = 0; i < this.dataSet.points.length; i++) {
      // Set position based on projected point.
      let pp = this.dataSet.points[i].projectedPoint;
      this.positionBufferArray.setXYZ(i, pp.x, pp.y, pp.z);
    }
    if (this.geometry) {
      this.positionBufferArray.needsUpdate = true;
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
      d.projectedPoint.x = this.xScale(this.xAccessor(i));
      d.projectedPoint.y = this.yScale(this.yAccessor(i));
      d.projectedPoint.z =
          (this.zAccessor ? this.zScale(this.zAccessor(i)) : 0);
    });
  }

  private createLight() {
    this.light = new THREE.PointLight(0xFFECBF, 1, 0);
    this.scene.add(this.light);
  }

  /** General setup of scene and renderer. */
  private createSceneAndRenderer() {
    this.scene = new THREE.Scene();
    this.renderer = new THREE.WebGLRenderer();
    // Accouting for retina displays.
    this.renderer.setPixelRatio(window.devicePixelRatio || 1);
    this.renderer.setSize(this.width, this.height);
    this.containerNode.appendChild(this.renderer.domElement);
    this.pickingTexture = new THREE.WebGLRenderTarget(this.width, this.height);
    this.pickingTexture.texture.minFilter = THREE.LinearFilter;
  }

  /** Set up camera and camera's controller. */
  private makeCamera() {
    this.perspCamera =
        new THREE.PerspectiveCamera(FOV, this.width / this.height, NEAR, FAR);
    this.cameraControls =
        new (THREE as any)
            .OrbitControls(this.perspCamera, this.renderer.domElement);
    this.cameraControls.mouseButtons.ORBIT = THREE.MOUSE.LEFT;
    this.cameraControls.mouseButtons.PAN = THREE.MOUSE.RIGHT;
    // Start is called when the user stars interacting with
    // orbit controls.
    this.cameraControls.addEventListener('start', () => {
      this.cameraControls.autoRotate = false;
      cancelAnimationFrame(this.lazySusanAnimation);
    });
    // Change is called everytime the user interacts with the
    // orbit controls.
    this.cameraControls.addEventListener('change', () => {
      this.removeAllLabels();
      this.render();
    });
    // End is called when the user stops interacting with the
    // orbit controls (e.g. on mouse up, after dragging).
    this.cameraControls.addEventListener('end', () => { this.makeLabels(); });
  }

  /** Sets up camera to work in 3D (called after makeCamera()). */
  private makeCamera3D() {
    // Set up the camera position at a skewed angle from the xy plane, looking
    // toward the origin
    this.cameraControls.position0.set(POS_3D.x, POS_3D.y, POS_3D.z);
    this.cameraControls.target0.set(TAR_3D.x, TAR_3D.y, TAR_3D.z);
    this.cameraControls.enableRotate = true;
    let position = new THREE.Vector3(POS_3D.x, POS_3D.y, POS_3D.z);
    let target = new THREE.Vector3(TAR_3D.x, TAR_3D.y, TAR_3D.z);
    this.animate(position, target, () => {
      // Start lazy susan after the animation is done.
      this.startLazySusanAnimation();
    });
  }

  /** Sets up camera to work in 2D (called after makeCamera()). */
  private makeCamera2D(animate?: boolean) {
    // Set the camera position in the middle of the screen, looking directly
    // toward the middle of the xy plane
    this.cameraControls.position0.set(POS_2D.x, POS_2D.y, POS_2D.z);
    this.cameraControls.target0.set(TAR_2D.x, TAR_2D.y, TAR_2D.z);
    let position = new THREE.Vector3(POS_2D.x, POS_2D.y, POS_2D.z);
    let target = new THREE.Vector3(TAR_2D.x, TAR_2D.y, TAR_2D.z);
    this.animate(position, target);
    this.cameraControls.enableRotate = false;
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

    // Note that we need two index arrays.

    /**
     * The actual indices of the points which we use for sizeAttenuation in
     * the shader.
     */
    let indicesShader =
        new THREE.BufferAttribute(new Float32Array(numPoints), 1);

    for (let i = 0; i < numPoints; i++) {
      // Create the array of indices.
      indicesShader.setX(i, this.dataSet.points[i].dataSourceIndex);
    }

    // Finally, add all attributes to the geometry.
    this.geometry.addAttribute('position', this.positionBufferArray);
    this.positionBufferArray.needsUpdate = true;
    this.geometry.addAttribute('color', colors);
    this.geometry.addAttribute('vertexIndex', indicesShader);
    this.geometry.addAttribute('isHighlight', highlights);

    // For now, nothing is highlighted.
    this.colorSprites(null);
  }

  /**
   * Generate a texture for the points/images and sets some initial params
   */
  private createTexture(image: HTMLImageElement|
                        HTMLCanvasElement): THREE.Texture {
    let tex = new THREE.Texture(image);
    tex.needsUpdate = true;
    // Used if the texture isn't a power of 2.
    tex.minFilter = THREE.LinearFilter;
    tex.generateMipmaps = false;
    tex.flipY = false;
    return tex;
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
    // TODO(smilkov): Pass sprite dim to the renderer.
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
      let vertices: number[] = [];
      let colors: number[] = [];

      for (let j = 0; j < dataTrace.pointIndices.length - 1; j++) {
        this.dataSet.points[dataTrace.pointIndices[j]].traceIndex = i;
        this.dataSet.points[dataTrace.pointIndices[j + 1]].traceIndex = i;

        let point1 = this.dataSet.points[dataTrace.pointIndices[j]];
        let point2 = this.dataSet.points[dataTrace.pointIndices[j + 1]];

        vertices.push(
            point1.projectedPoint.x, point1.projectedPoint.y,
            point1.projectedPoint.z);
        vertices.push(
            point2.projectedPoint.x, point2.projectedPoint.y,
            point2.projectedPoint.z);

        let color1 =
            this.getPointInTraceColor(j, dataTrace.pointIndices.length);
        let color2 =
            this.getPointInTraceColor(j + 1, dataTrace.pointIndices.length);

        colors.push(
            color1.r / 255, color1.g / 255, color1.b / 255, color2.r / 255,
            color2.g / 255, color2.b / 255);
      }

      geometry.addAttribute(
          'position',
          new THREE.BufferAttribute(new Float32Array(vertices), XYZ_NUM_BYTES));
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

  /**
   * Returns the color of a point along a trace.
   */
  private getPointInTraceColor(index: number, totalPoints: number) {
    let hue = TRACE_START_HUE +
        (TRACE_END_HUE - TRACE_START_HUE) * index / totalPoints;

    return d3.hsl(hue, TRACE_SATURATION, TRACE_LIGHTNESS).rgb();
  }

  /** Clean up any old axes that we may have made previously.  */
  private removeOldAxes() {
    if (this.axis3D) {
      this.scene.remove(this.axis3D);
    }
    if (this.axis2D) {
      this.scene.remove(this.axis2D);
    }
  }

  /** Add axis. */
  private addAxis3D() {
    this.axis3D = new THREE.AxisHelper();
    this.scene.add(this.axis3D);
  }

  /** Manually make axis if we're in 2d. */
  private addAxis2D() {
    let vertices = new Float32Array([
      0,
      0,
      0,
      this.xScale(1),
      0,
      0,
      0,
      0,
      0,
      0,
      this.yScale(1),
      0,
    ]);

    let axisColor = new THREE.Color(AXIS_COLOR);
    let axisColors = new Float32Array([
      axisColor.r,
      axisColor.b,
      axisColor.g,
      axisColor.r,
      axisColor.b,
      axisColor.g,
      axisColor.r,
      axisColor.b,
      axisColor.g,
      axisColor.r,
      axisColor.b,
      axisColor.g,
    ]);

    // Create line geometry based on above position and color.
    let lineGeometry = new THREE.BufferGeometry();
    lineGeometry.addAttribute(
        'position', new THREE.BufferAttribute(vertices, XYZ_NUM_BYTES));
    lineGeometry.addAttribute(
        'color', new THREE.BufferAttribute(axisColors, RGB_NUM_BYTES));

    // And use it to create the actual object and add this new axis to the
    // scene!
    let axesMaterial =
        new THREE.LineBasicMaterial({vertexColors: THREE.VertexColors});
    this.axis2D = new THREE.LineSegments(lineGeometry, axesMaterial);
    this.scene.add(this.axis2D);
  }

  // DYNAMIC (post-load) CHANGES

  /** When we stop dragging/zooming, return to normal behavior. */
  private onClick(e?: MouseEvent) {
    if (e && this.selecting || !this.points) {
      this.resetTraces();
      return;
    }
    let selection = this.nearestPoint || null;
    this.defaultPointColor = (selection ? POINT_COLOR_GRAYED : POINT_COLOR);
    // Only call event handlers if the click originated from the scatter plot.
    if (e && !this.isDragSequence) {
      this.onSelectionListeners.forEach(l => l(selection ? [selection] : []));
    }
    this.isDragSequence = false;
    this.labeledPoints =
        this.highlightedPoints.filter((id, i) => this.favorLabels(i));

    this.resetTraces();
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
    this.render();
    this.makeLabels();
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

  /** When dragging, do not redraw labels. */
  private onMouseDown(e: MouseEvent) {
    this.animating = false;
    this.isDragSequence = false;
    this.mouseIsDown = true;
    // If we are in selection mode, and we have in fact clicked a valid point,
    // create a sphere so we can select things
    if (this.selecting) {
      this.cameraControls.enabled = false;
      this.setNearestPointToMouse(e);
      if (this.nearestPoint) {
        this.createSelectionSphere();
      }
    } else if (
        !e.ctrlKey &&
        this.cameraControls.mouseButtons.ORBIT == THREE.MOUSE.RIGHT) {
      // The user happened to press the ctrl key when the tab was active,
      // unpressed the ctrl when the tab was inactive, and now he/she
      // is back to the projector tab.
      this.cameraControls.mouseButtons.ORBIT = THREE.MOUSE.LEFT;
      this.cameraControls.mouseButtons.PAN = THREE.MOUSE.RIGHT;
    } else if (
        e.ctrlKey &&
        this.cameraControls.mouseButtons.ORBIT == THREE.MOUSE.LEFT) {
      // Similarly to the situation above.
      this.cameraControls.mouseButtons.ORBIT = THREE.MOUSE.RIGHT;
      this.cameraControls.mouseButtons.PAN = THREE.MOUSE.LEFT;
    }
  }


  /** When we stop dragging/zooming, return to normal behavior. */
  private onMouseUp(e: any) {
    if (this.selecting) {
      this.cameraControls.enabled = true;
      this.scene.remove(this.selectionSphere);
      this.selectionSphere = null;
      this.render();
    }
    this.mouseIsDown = false;
  }

  /**
   * When the mouse moves, find the nearest point (if any) and send it to the
   * hoverlisteners (usually called from embedding.ts)
   */
  private onMouseMove(e: MouseEvent) {
    if (this.cameraControls.autoRotate) {
      // Cancel the lazy susan.
      this.cameraControls.autoRotate = false;
      cancelAnimationFrame(this.lazySusanAnimation);
      this.makeLabels();
    }

    // A quick check to make sure data has come in.
    if (!this.points) {
      return;
    }
    this.isDragSequence = this.mouseIsDown;
    // Depending if we're selecting or just navigating, handle accordingly.
    if (this.selecting && this.mouseIsDown) {
      if (this.selectionSphere) {
        this.adjustSelectionSphere(e);
      }
      this.render();
    } else if (!this.mouseIsDown) {
      let lastNearestPoint = this.nearestPoint;
      this.setNearestPointToMouse(e);
      if (lastNearestPoint != this.nearestPoint) {
        this.onHoverListeners.forEach(l => l(this.nearestPoint));
      }
    }
  }

  /** For using ctrl + left click as right click, and for circle select */
  private onKeyDown(e: any) {
    // If ctrl is pressed, use left click to orbit
    if (e.keyCode === CTRL_KEY) {
      this.cameraControls.mouseButtons.ORBIT = THREE.MOUSE.RIGHT;
      this.cameraControls.mouseButtons.PAN = THREE.MOUSE.LEFT;
    }

    // If shift is pressed, start selecting
    if (e.keyCode === SHIFT_KEY) {
      this.selecting = true;
      this.containerNode.style.cursor = 'crosshair';
    }
  }

  /** For using ctrl + left click as right click, and for circle select */
  private onKeyUp(e: any) {
    if (e.keyCode === CTRL_KEY) {
      this.cameraControls.mouseButtons.ORBIT = THREE.MOUSE.LEFT;
      this.cameraControls.mouseButtons.PAN = THREE.MOUSE.RIGHT;
    }

    // If shift is released, stop selecting
    if (e.keyCode === SHIFT_KEY) {
      this.selecting = (this.getMode() === Mode.SELECT);
      if (!this.selecting) {
        this.containerNode.style.cursor = 'default';
      }
      this.scene.remove(this.selectionSphere);
      this.selectionSphere = null;
      this.render();
    }
  }

  private setNearestPointToMouse(e: MouseEvent) {
    // Create buffer for reading a single pixel.
    let pixelBuffer = new Uint8Array(4);
    // No need to account for dpr (device pixel ratio) since the pickingTexture
    // has the same coordinates as the mouse (flipped on y).
    let x = e.offsetX;
    let y = e.offsetY;

    // Read the pixel under the mouse from the texture.
    this.renderer.readRenderTargetPixels(
        this.pickingTexture, x, this.pickingTexture.height - y, 1, 1,
        pixelBuffer);

    // Interpret the pixel as an ID.
    let id = (pixelBuffer[0] << 16) | (pixelBuffer[1] << 8) | pixelBuffer[2];
    this.nearestPoint =
        id != 0xffffff && id < this.dataSet.points.length ? id : null;
  }

  /** Returns the squared distance to the mouse for the i-th point. */
  private getDist2ToMouse(i: number, e: MouseEvent) {
    let point = this.getProjectedPointFromIndex(i);
    let screenCoords = this.vector3DToScreenCoords(point);
    return this.dist2D(
        [e.offsetX * this.dpr, e.offsetY * this.dpr],
        [screenCoords.x, screenCoords.y]);
  }

  private adjustSelectionSphere(e: MouseEvent) {
    let dist2 = this.getDist2ToMouse(this.nearestPoint, e) / 100;
    this.selectionSphere.scale.set(dist2, dist2, dist2);
    this.selectPoints(dist2);
  }

  private getProjectedPointFromIndex(i: number): THREE.Vector3 {
    return new THREE.Vector3(
        this.dataSet.points[i].projectedPoint.x,
        this.dataSet.points[i].projectedPoint.y,
        this.dataSet.points[i].projectedPoint.z);
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
    for (let i = 0; i < this.dataSet.points.length; i++) {
      let point = this.getProjectedPointFromIndex(i);
      if (!this.isPointWithinCameraView(point)) {
        continue;
      };
      let distToCam = this.dist3D(point, this.perspCamera.position);
      furthestDist = Math.max(furthestDist, distToCam);
      shortestDist = Math.min(shortestDist, distToCam);
    }
    return {shortestDist, furthestDist};
  }

  /**
   * Renders the scene and updates the label for the point, which is rendered
   * as a div on top of WebGL.
   */
  private render() {
    if (!this.dataSet) {
      return;
    }
    let lightPos = new THREE.Vector3().copy(this.perspCamera.position);
    lightPos.x += 1;
    lightPos.y += 1;
    this.light.position.set(lightPos.x, lightPos.y, lightPos.z);

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

    // Now render onscreen.

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
    // Render onscreen.
    this.renderer.render(this.scene, this.perspCamera);
  }

  /**
   * Make sure that the point is in view of the camera (as opposed to behind
   * this is a problem because we are projecting to the camera)
   */
  private isPointWithinCameraView(point: THREE.Vector3): boolean {
    let camToTarget = new THREE.Vector3()
                          .copy(this.perspCamera.position)
                          .sub(this.cameraControls.target);
    let camToPoint =
        new THREE.Vector3().copy(this.perspCamera.position).sub(point);
    // If the angle between the camera-target and camera-point vectors is more
    // than 90, the point is behind the camera
    if (camToPoint.angleTo(camToTarget) > Math.PI / 2) {
      return false;
    };
    return true;
  }

  private vector3DToScreenCoords(v: THREE.Vector3) {
    let vector = new THREE.Vector3().copy(v).project(this.perspCamera);
    let coords = {
      // project() returns the point in perspCamera's coordinates, with the
      // origin in the center and a positive upward y. To get it into screen
      // coordinates, normalize by adding 1 and dividing by 2.
      x: ((vector.x + 1) / 2 * this.width) * this.dpr,
      y: -((vector.y - 1) / 2 * this.height) * this.dpr
    };
    return coords;
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

  /**
   * Reset the positions of all labels, and check for overlapps using the
   * collision grid.
   */
  private makeLabels() {
    // Don't make labels if they are turned off.
    if (!this.labelsAreOn || this.points == null) {
      return;
    }
    // First, remove all old labels.
    this.removeAllLabels();

    this.labelCanvasIsCleared = false;
    // If we are passed no points to label (that is, not mousing over any
    // points) then want to label ALL the points that we can.
    if (!this.labeledPoints.length) {
      this.labeledPoints = this.shuffledData;
    }

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

    // Setting styles for the labeled font.
    this.gc.lineWidth = 6;
    this.gc.textBaseline = 'middle';
    this.gc.font = (FONT_SIZE * this.dpr).toString() + 'px roboto';

    for (let i = 0;
         (i < this.labeledPoints.length) && !(numRenderedLabels > SAMPLE_SIZE);
         i++) {
      let index = this.labeledPoints[i];
      let point = this.getProjectedPointFromIndex(index);
      if (!this.isPointWithinCameraView(point)) {
        continue;
      };
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
        let dataSet = this.dataSet;
        let text = this.labelAccessor(index);
        let labelWidth = this.gc.measureText(text).width;

        // Now, check with properly computed width.
        textBoundingBox.hiX += labelWidth - 1;
        if (grid.insert(textBoundingBox) &&
            this.isLabelInBounds(labelWidth, screenCoords)) {
          let lenToCamera = this.dist3D(point, this.perspCamera.position);
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
          point.projectedPoint.x, point.projectedPoint.y,
          point.projectedPoint.z);
      let screenCoords = this.vector3DToScreenCoords(coords);
      let text = this.labelAccessor(index);
      this.formatLabel(text, screenCoords, 255);
    }
  }

  /** Returns the distance between two points in 3d space */
  private dist3D(a: Point3D, b: Point3D): number {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    let dz = a.z - b.z;
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  }

  private dist2D(a: [number, number], b: [number, number]): number {
    let dX = a[0] - b[0];
    let dY = a[1] - b[1];
    return Math.sqrt(dX * dX + dY * dY);
  }

  /** Cancels current animation */
  private cancelAnimation() {
    if (this.animationID) {
      cancelAnimationFrame(this.animationID);
    }
  }

  private startLazySusanAnimation() {
    this.cameraControls.autoRotate = true;
    this.cameraControls.update();
    this.lazySusanAnimation =
        requestAnimationFrame(() => this.startLazySusanAnimation());
  }

  /**
   * Animates the camera between one location and another.
   * If callback is specified, it gets called when the animation is done.
   */
  private animate(
      pos: THREE.Vector3, target: THREE.Vector3, callback?: () => void) {
    this.cameraControls.autoRotate = false;
    cancelAnimationFrame(this.lazySusanAnimation);

    let currPos = this.perspCamera.position;
    let currTarget = this.cameraControls.target;
    let speed = 3;
    this.animating = true;
    let interp = (a: THREE.Vector3, b: THREE.Vector3) => {
      let x = (a.x - b.x) / speed + b.x;
      let y = (a.y - b.y) / speed + b.y;
      let z = (a.z - b.z) / speed + b.z;
      return {x: x, y: y, z: z};
    };
    // If we're still relatively far away from the target, go closer
    if (this.dist3D(currPos, pos) > .03) {
      let newTar = interp(target, currTarget);
      this.cameraControls.target.set(newTar.x, newTar.y, newTar.z);

      let newPos = interp(pos, currPos);
      this.perspCamera.position.set(newPos.x, newPos.y, newPos.z);
      this.cameraControls.update();
      this.render();
      this.animationID =
          requestAnimationFrame(() => this.animate(pos, target, callback));
    } else {
      // Once we get close enough, update flags and stop moving
      this.animating = false;
      this.cameraControls.target.set(target.x, target.y, target.z);
      this.cameraControls.update();
      this.makeLabels();
      this.render();
      if (callback) {
        callback();
      }
    }
  }

  /** Removes all points geometry from the scene. */
  private removeAll() {
    this.scene.remove(this.points);
    this.removeOldAxes();
    this.removeAllLabels();
    this.removeAllTraces();
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

  /** Removes all the labels. */
  private removeAllLabels() {
    // If labels are already removed, do not spend compute power to clear the
    // canvas.
    if (!this.labelCanvasIsCleared) {
      this.gc.clearRect(0, 0, this.width * this.dpr, this.height * this.dpr);
      this.labelCanvasIsCleared = true;
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

  /**
   * This is called when we update the data to make sure we don't have stale
   * data lying around.
   */
  private cleanVariables() {
    this.removeAll();
    if (this.geometry) {
      this.geometry.dispose();
    }
    this.geometry = null;
    this.points = null;
    this.labeledPoints = [];
    this.highlightedPoints = [];
  }

  /** Select the points inside the sphere of radius dist */
  private selectPoints(dist: number) {
    let selectedPoints: Array<number> = new Array();
    this.dataSet.points.forEach(point => {
      let pt = point.projectedPoint;
      let pointVect = new THREE.Vector3(pt.x, pt.y, pt.z);
      let distPointToSphereOrigin = new THREE.Vector3()
                                        .copy(this.selectionSphere.position)
                                        .sub(pointVect)
                                        .length();
      if (distPointToSphereOrigin < dist) {
        selectedPoints.push(this.dataSet.points.indexOf(point));
      }
    });
    this.labeledPoints = selectedPoints;
    // Whenever anything is selected, we want to set the corect point color.
    this.defaultPointColor = POINT_COLOR_GRAYED;
    this.onSelectionListeners.forEach(l => l(selectedPoints));
  }

  private createSelectionSphere() {
    let geometry = new THREE.SphereGeometry(1, 300, 100);
    let material = new THREE.MeshPhongMaterial({
      color: 0x000000,
      specular: (this.zAccessor && 0xffffff),  // In 2d, make sphere look flat.
      emissive: 0x000000,
      shininess: 10,
      shading: THREE.SmoothShading,
      opacity: 0.125,
      transparent: true,
    });
    this.selectionSphere = new THREE.Mesh(geometry, material);
    this.selectionSphere.scale.set(0, 0, 0);
    let pos = this.dataSet.points[this.nearestPoint].projectedPoint;
    this.scene.add(this.selectionSphere);
    this.selectionSphere.position.set(pos.x, pos.y, pos.z);
  }

  private getLayoutValues() {
    this.width = this.containerNode.offsetWidth;
    this.height = Math.max(1, this.containerNode.offsetHeight);
    this.dpr = window.devicePixelRatio;
  }

  // PUBLIC API

  /** Sets the data for the scatter plot. */
  setDataSet(dataSet: DataSet, spriteImage: HTMLImageElement) {
    this.dataSet = dataSet;
    this.calibratePointSize();
    let positions =
        new Float32Array(this.dataSet.points.length * XYZ_NUM_BYTES);
    this.positionBufferArray =
        new THREE.BufferAttribute(positions, XYZ_NUM_BYTES);
    this.image = spriteImage;
    this.shuffledData = new Array(this.dataSet.points.length);
    for (let i = 0; i < this.dataSet.points.length; i++) {
      this.shuffledData[i] = i;
    }
    shuffle(this.shuffledData);
    this.cleanVariables();
  }

  setColorAccessor(colorAccessor: (index: number) => string) {
    this.colorAccessor = colorAccessor;
    // Render only if there is a geometry.
    if (this.geometry) {
      this.colorSprites(this.highlightStroke);
      this.render();
    }
  }

  setXAccessor(xAccessor: (index: number) => number) {
    this.xAccessor = xAccessor;
  }

  setYAccessor(yAccessor: (index: number) => number) {
    this.yAccessor = yAccessor;
  }

  setZAccessor(zAccessor: (index: number) => number) {
    this.zAccessor = zAccessor;
  }

  setLabelAccessor(labelAccessor: (index: number) => string) {
    this.labelAccessor = labelAccessor;
    this.render();
  }

  setMode(mode: Mode) {
    this.mode = mode;
    if (mode === Mode.SELECT) {
      this.selecting = true;
      this.containerNode.style.cursor = 'crosshair';
    } else {
      this.selecting = false;
      this.containerNode.style.cursor = 'default';
    }
  }

  getMode(): Mode { return this.mode; }

  resetZoom() {
    if (this.animating) {
      return;
    }
    let resetPos = this.cameraControls.position0;
    let resetTarget = this.cameraControls.target0;
    this.removeAllLabels();
    this.animate(resetPos, resetTarget, () => {
      // Start rotating when the animation is done, if we are in 3D mode.
      if (this.zAccessor) {
        this.startLazySusanAnimation();
      }
    });
  }

  /** Zoom by moving the camera toward the target. */
  zoomStep(multiplier: number) {
    let additiveZoom = Math.log(multiplier);
    if (this.animating) {
      return;
    }

    // Zoomvect is the vector along which we want to move the camera
    // It is the (normalized) vector from the camera to its target
    let zoomVect = new THREE.Vector3()
                       .copy(this.cameraControls.target)
                       .sub(this.perspCamera.position)
                       .multiplyScalar(additiveZoom);
    let position =
        new THREE.Vector3().copy(this.perspCamera.position).add(zoomVect);

    // Make sure that we're not too far zoomed in. If not, zoom!
    if ((this.dist3D(position, this.cameraControls.target) > MIN_ZOOM) &&
        (this.dist3D(position, this.cameraControls.target) < MAX_ZOOM)) {
      this.removeAllLabels();
      this.animate(position, this.cameraControls.target);
    }
  }

  highlightPoints(
      pointIndexes: number[], highlightStroke: (i: number) => string,
      favorLabels: (i: number) => boolean): void {
    this.favorLabels = favorLabels;
    this.highlightedPoints = pointIndexes;
    this.labeledPoints = pointIndexes;
    this.highlightStroke = highlightStroke;
    this.colorSprites(highlightStroke);
    this.render();
    this.makeLabels();
  }

  getHighlightedPoints(): number[] { return this.highlightedPoints; }

  showLabels(show: boolean) {
    this.labelsAreOn = show;
    if (this.labelsAreOn) {
      this.makeLabels();
    } else {
      this.removeAllLabels();
    }
  }

  /**
   * Toggles between day and night mode (resets corresponding variables for
   * color, etc.)
   */
  setDayNightMode(isNight: boolean) {
    this.isNight = isNight;
    this.labelColor = (isNight ? LABEL_COLOR_NIGHT : LABEL_COLOR_DAY);
    this.labelStroke = (isNight ? LABEL_STROKE_NIGHT : LABEL_STROKE_DAY);
    this.backgroundColor =
        (isNight ? BACKGROUND_COLOR_NIGHT : BACKGROUND_COLOR_DAY);
    this.blending = (isNight ? BLENDING_NIGHT : BLENDING_DAY);
    this.renderer.setClearColor(this.backgroundColor);
  }

  showAxes(show: boolean) {
    // TODO(ereif): implement
  }

  setAxisLabels(xLabel: string, yLabel: string) {
    // TODO(ereif): implement
  }
  /**
   * Recreates the scene in its entirety, not only resetting the point
   * locations but also demolishing and recreating the THREEjs structures.
   */
  recreateScene() {
    this.removeAll();
    this.cancelAnimation();
    this.fog = this.zAccessor ?
        new THREE.Fog(this.backgroundColor) :
        new THREE.Fog(this.backgroundColor, Infinity, Infinity);
    this.scene.fog = this.fog;
    this.addSprites();
    this.addTraces();
    if (this.zAccessor) {
      this.addAxis3D();
      this.makeCamera3D();
    } else {
      this.addAxis2D();
      this.makeCamera2D();
    }
    this.render();
  }

  /**
   * Redraws the data. Should be called anytime the accessor method
   * for x and y coordinates changes, which means a new projection
   * exists and the scatter plot should repaint the points.
   */
  update() {
    this.cancelAnimation();
    this.getPointsCoordinates();
    this.updatePositionsArray();
    if (this.geometry) {
      this.makeLabels();
      this.render();
    }
  }

  resize(render = true) {
    this.getLayoutValues();
    this.perspCamera.aspect = this.width / this.height;
    this.perspCamera.updateProjectionMatrix();
    d3.select(this.canvas)
        .attr('width', this.width * this.dpr)
        .attr('height', this.height * this.dpr)
        .style({width: this.width + 'px', height: this.height + 'px'});
    this.renderer.setSize(this.width, this.height);
    this.pickingTexture = new THREE.WebGLRenderTarget(this.width, this.height);
    this.pickingTexture.texture.minFilter = THREE.LinearFilter;
    if (render) {
      this.render();
    };
  }

  showTickLabels(show: boolean) {
    // TODO(ereif): implement
  }

  onSelection(listener: OnSelectionListener) {
    this.onSelectionListeners.push(listener);
  }

  onHover(listener: OnHoverListener) { this.onHoverListeners.push(listener); }

  clickOnPoint(pointIndex: number) {
    this.nearestPoint = pointIndex;
    this.onClick();
  }
}
