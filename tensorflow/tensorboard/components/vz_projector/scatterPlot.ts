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

import {PointAccessor} from './data';
import {HoverContext} from './hoverContext';
import {LabelRenderParams, RenderContext} from './renderContext';
import {ScatterPlotVisualizer} from './scatterPlotVisualizer';
import {ScatterPlotVisualizerAxes} from './scatterPlotVisualizerAxes';
import {SelectionContext} from './selectionContext';
import {getNearFarPoints, getProjectedPointFromIndex, vector3DToScreenCoords} from './util';
import {dist_2D, Point2D, Point3D} from './vector';

const BACKGROUND_COLOR = 0xffffff;

/**
 * The length of the cube (diameter of the circumscribing sphere) where all the
 * points live.
 */
const CUBE_LENGTH = 2;
const MAX_ZOOM = 5 * CUBE_LENGTH;
const MIN_ZOOM = 0.025 * CUBE_LENGTH;

// Constants relating to the camera parameters.
const PERSP_CAMERA_FOV_VERTICAL = 70;
const PERSP_CAMERA_NEAR_CLIP_PLANE = 0.01;
const PERSP_CAMERA_FAR_CLIP_PLANE = 100;
const ORTHO_CAMERA_FRUSTUM_HALF_EXTENT = 1.2;

// Key presses.
const SHIFT_KEY = 16;
const CTRL_KEY = 17;

const START_CAMERA_POS_3D = new THREE.Vector3(0.6, 1.0, 1.85);
const START_CAMERA_TARGET_3D = new THREE.Vector3(0, 0, 0);
const START_CAMERA_POS_2D = new THREE.Vector3(0, 0, 1);
const START_CAMERA_TARGET_2D = new THREE.Vector3(0, 0, 0);

const ORBIT_MOUSE_ROTATION_SPEED = 1;
const ORBIT_ANIMATION_ROTATION_CYCLE_IN_SECONDS = 7;

/** The spacial data of points and lines that will be shown in the projector. */
export interface DataSet {
  points: DataPoint[];
  traces: DataTrace[];
}

/**
 * Points in 3D space that will be used in the projector. If the projector is
 * in 2D mode, the Z coordinate of the point will be 0.
 */
export interface DataPoint {
  projectedPoint: Point3D;
  /** index of the trace, used for highlighting on click */
  traceIndex?: number;
  /** index in the original data source */
  index: number;
}

/** A single collection of points which make up a trace through space. */
export interface DataTrace {
  /** Indices into the DataPoints array in the Data object. */
  pointIndices: number[];
}

export type OnCameraMoveListener =
    (cameraPosition: THREE.Vector3, cameraTarget: THREE.Vector3) => void;

/** Supported modes of interaction. */
export enum Mode {
  SELECT,
  HOVER
}

/** Defines a camera, suitable for serialization. */
export class CameraDef {
  orthographic: boolean = false;
  position: Point3D;
  target: Point3D;
  zoom: number;
}

/**
 * Maintains a three.js instantiation and context,
 * animation state, and all other logic that's
 * independent of how a 3D scatter plot is actually rendered. Also holds an
 * array of visualizers and dispatches application events to them.
 */
export class ScatterPlot {
  private dataSet: DataSet;
  private selectionContext: SelectionContext;
  private hoverContext: HoverContext;

  private spriteImage: HTMLImageElement;
  private containerNode: HTMLElement;
  private visualizers: ScatterPlotVisualizer[] = [];

  private labelAccessor: (index: number) => string;
  private onCameraMoveListeners: OnCameraMoveListener[] = [];

  // Accessors for rendering and labeling the points.
  private pointAccessors: [PointAccessor, PointAccessor, PointAccessor];

  // Scaling functions for each axis.
  private xScale: d3.scale.Linear<number, number>;
  private yScale: d3.scale.Linear<number, number>;
  private zScale: d3.scale.Linear<number, number>;

  // window layout dimensions
  private height: number;
  private width: number;

  private mode: Mode;
  private backgroundColor: number = BACKGROUND_COLOR;

  private dimensionality: number = 3;
  private renderer: THREE.WebGLRenderer;

  private scene: THREE.Scene;
  private pickingTexture: THREE.WebGLRenderTarget;
  private light: THREE.PointLight;
  private selectionSphere: THREE.Mesh;

  private cameraDef: CameraDef = null;
  private camera: THREE.Camera;
  private orbitCameraControls: any;
  private orbitAnimationId: number;

  private pointColors: Float32Array;
  private pointScaleFactors: Float32Array;
  private labels: LabelRenderParams;

  private traceColors: {[trace: number]: Float32Array};

  private selecting = false;
  private nearestPoint: number;
  private mouseIsDown = false;
  private isDragSequence = false;

  constructor(
      container: d3.Selection<any>, labelAccessor: (index: number) => string,
      selectionContext: SelectionContext, hoverContext: HoverContext) {
    this.containerNode = container.node() as HTMLElement;
    this.selectionContext = selectionContext;
    this.hoverContext = hoverContext;
    this.getLayoutValues();

    this.labelAccessor = labelAccessor;
    this.xScale = d3.scale.linear();
    this.yScale = d3.scale.linear();
    this.zScale = d3.scale.linear();

    this.scene = new THREE.Scene();
    this.renderer = new THREE.WebGLRenderer();
    this.renderer.setClearColor(BACKGROUND_COLOR, 1);
    this.containerNode.appendChild(this.renderer.domElement);
    this.light = new THREE.PointLight(0xFFECBF, 1, 0);
    this.scene.add(this.light);

    this.setDimensions(3);
    this.recreateCamera(this.makeDefaultCameraDef(this.dimensionality));
    this.renderer.render(this.scene, this.camera);

    this.addAxesToScene();
    this.addInteractionListeners();
  }

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

  private addCameraControlsEventListeners(cameraControls: any) {
    // Start is called when the user stars interacting with
    // controls.
    cameraControls.addEventListener('start', () => {
      this.stopOrbitAnimation();
      this.onCameraMoveListeners.forEach(
          l => l(this.camera.position, cameraControls.target));
    });

    // Change is called everytime the user interacts with the controls.
    cameraControls.addEventListener('change', () => {
      this.render();
    });

    // End is called when the user stops interacting with the
    // controls (e.g. on mouse up, after dragging).
    cameraControls.addEventListener('end', () => {});
  }

  private makeCamera3D(cameraDef: CameraDef, w: number, h: number) {
    let camera: THREE.PerspectiveCamera;
    {
      const aspectRatio = w / h;
      camera = new THREE.PerspectiveCamera(
          PERSP_CAMERA_FOV_VERTICAL, aspectRatio, PERSP_CAMERA_NEAR_CLIP_PLANE,
          PERSP_CAMERA_FAR_CLIP_PLANE);
      camera.position.set(
          cameraDef.position[0], cameraDef.position[1], cameraDef.position[2]);
      const at = new THREE.Vector3(
          cameraDef.target[0], cameraDef.target[1], cameraDef.target[2]);
      camera.lookAt(at);
      camera.zoom = cameraDef.zoom;
    }

    const occ =
        new (THREE as any).OrbitControls(camera, this.renderer.domElement);

    occ.enableRotate = true;
    occ.rotateSpeed = ORBIT_MOUSE_ROTATION_SPEED;
    occ.mouseButtons.ORBIT = THREE.MOUSE.LEFT;
    occ.mouseButtons.PAN = THREE.MOUSE.RIGHT;

    if (this.orbitCameraControls != null) {
      this.orbitCameraControls.dispose();
    }

    this.camera = camera;
    this.orbitCameraControls = occ;
    this.addCameraControlsEventListeners(this.orbitCameraControls);
  }

  private makeCamera2D(cameraDef: CameraDef, w: number, h: number) {
    let camera: THREE.OrthographicCamera;
    const target = new THREE.Vector3(
        cameraDef.target[0], cameraDef.target[1], cameraDef.target[2]);
    {
      const aspectRatio = w / h;
      let left = -ORTHO_CAMERA_FRUSTUM_HALF_EXTENT;
      let right = ORTHO_CAMERA_FRUSTUM_HALF_EXTENT;
      let bottom = -ORTHO_CAMERA_FRUSTUM_HALF_EXTENT;
      let top = ORTHO_CAMERA_FRUSTUM_HALF_EXTENT;
      // Scale up the larger of (w, h) to match the aspect ratio.
      if (aspectRatio > 1) {
        left *= aspectRatio;
        right *= aspectRatio;
      } else {
        top /= aspectRatio;
        bottom /= aspectRatio;
      }
      camera =
          new THREE.OrthographicCamera(left, right, top, bottom, -1000, 1000);
      camera.position.set(
          cameraDef.position[0], cameraDef.position[1], cameraDef.position[2]);
      camera.up = new THREE.Vector3(0, 1, 0);
      camera.lookAt(target);
      camera.zoom = cameraDef.zoom;
    }

    const occ =
        new (THREE as any).OrbitControls(camera, this.renderer.domElement);

    occ.target = target;
    occ.enableRotate = false;
    occ.enableDamping = false;
    occ.autoRotate = false;
    occ.mouseButtons.ORBIT = null;
    occ.mouseButtons.PAN = THREE.MOUSE.LEFT;

    if (this.orbitCameraControls != null) {
      this.orbitCameraControls.dispose();
    }

    this.camera = camera;
    this.orbitCameraControls = occ;
    this.addCameraControlsEventListeners(occ);
  }

  private makeDefaultCameraDef(dimensionality: number): CameraDef {
    const def = new CameraDef();
    def.orthographic = (dimensionality === 2);
    def.zoom = 1.0;
    if (def.orthographic) {
      def.position =
          [START_CAMERA_POS_2D.x, START_CAMERA_POS_2D.y, START_CAMERA_POS_2D.z];
      def.target = [
        START_CAMERA_TARGET_2D.x, START_CAMERA_TARGET_2D.y,
        START_CAMERA_TARGET_2D.z
      ];
    } else {
      def.position =
          [START_CAMERA_POS_3D.x, START_CAMERA_POS_3D.y, START_CAMERA_POS_3D.z];
      def.target = [
        START_CAMERA_TARGET_3D.x, START_CAMERA_TARGET_3D.y,
        START_CAMERA_TARGET_3D.z
      ];
    }
    return def;
  }

  /** Recreate the scatter plot camera from a definition structure. */
  recreateCamera(cameraDef: CameraDef) {
    if (cameraDef.orthographic) {
      this.makeCamera2D(cameraDef, this.width, this.height);
    } else {
      this.makeCamera3D(cameraDef, this.width, this.height);
    }
    this.orbitCameraControls.minDistance = MIN_ZOOM;
    this.orbitCameraControls.maxDistance = MAX_ZOOM;
    this.orbitCameraControls.update();
  }

  private onClick(e?: MouseEvent, notify = true) {
    if (e && this.selecting) {
      return;
    }
    // Only call event handlers if the click originated from the scatter plot.
    if (!this.isDragSequence && notify) {
      const selection = this.nearestPoint ? [this.nearestPoint] : [];
      this.selectionContext.notifySelectionChanged(selection);
    }
    this.isDragSequence = false;
    this.render();
  }

  private onMouseDown(e: MouseEvent) {
    this.isDragSequence = false;
    this.mouseIsDown = true;
    // If we are in selection mode, and we have in fact clicked a valid point,
    // create a sphere so we can select things
    if (this.selecting) {
      this.orbitCameraControls.enabled = false;
      this.setNearestPointToMouse(e);
      if (this.nearestPoint) {
        this.createSelectionSphere();
      }
    } else if (
        !e.ctrlKey && this.sceneIs3D() &&
        this.orbitCameraControls.mouseButtons.ORBIT === THREE.MOUSE.RIGHT) {
      // The user happened to press the ctrl key when the tab was active,
      // unpressed the ctrl when the tab was inactive, and now he/she
      // is back to the projector tab.
      this.orbitCameraControls.mouseButtons.ORBIT = THREE.MOUSE.LEFT;
      this.orbitCameraControls.mouseButtons.PAN = THREE.MOUSE.RIGHT;
    } else if (
        e.ctrlKey && this.sceneIs3D() &&
        this.orbitCameraControls.mouseButtons.ORBIT === THREE.MOUSE.LEFT) {
      // Similarly to the situation above.
      this.orbitCameraControls.mouseButtons.ORBIT = THREE.MOUSE.RIGHT;
      this.orbitCameraControls.mouseButtons.PAN = THREE.MOUSE.LEFT;
    }
  }

  /** When we stop dragging/zooming, return to normal behavior. */
  private onMouseUp(e: any) {
    if (this.selecting) {
      this.orbitCameraControls.enabled = true;
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
    this.stopOrbitAnimation();
    if (!this.dataSet) {
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
      this.setNearestPointToMouse(e);
      this.hoverContext.notifyHoverOverPoint(this.nearestPoint);
    }
  }

  /** For using ctrl + left click as right click, and for circle select */
  private onKeyDown(e: any) {
    // If ctrl is pressed, use left click to orbit
    if (e.keyCode === CTRL_KEY && this.sceneIs3D()) {
      this.orbitCameraControls.mouseButtons.ORBIT = THREE.MOUSE.RIGHT;
      this.orbitCameraControls.mouseButtons.PAN = THREE.MOUSE.LEFT;
    }

    // If shift is pressed, start selecting
    if (e.keyCode === SHIFT_KEY) {
      this.selecting = true;
      this.containerNode.style.cursor = 'crosshair';
    }
  }

  /** For using ctrl + left click as right click, and for circle select */
  private onKeyUp(e: any) {
    if (e.keyCode === CTRL_KEY && this.sceneIs3D()) {
      this.orbitCameraControls.mouseButtons.ORBIT = THREE.MOUSE.LEFT;
      this.orbitCameraControls.mouseButtons.PAN = THREE.MOUSE.RIGHT;
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
    if (this.pickingTexture == null) {
      this.nearestPoint = null;
      return;
    }

    // Create buffer for reading a single pixel.
    let pixelBuffer = new Uint8Array(4);
    const dpr = window.devicePixelRatio || 1;
    const x = e.offsetX * dpr;
    const y = e.offsetY * dpr;
    // Read the pixel under the mouse from the texture.
    this.renderer.readRenderTargetPixels(
        this.pickingTexture, x, this.pickingTexture.height - y, 1, 1,
        pixelBuffer);
    // Interpret the pixel as an ID.
    const id = (pixelBuffer[0] << 16) | (pixelBuffer[1] << 8) | pixelBuffer[2];
    this.nearestPoint =
        (id !== 0xffffff) && (id < this.dataSet.points.length) ? id : null;
  }

  /** Returns the squared distance to the mouse for the i-th point. */
  private getDist2ToMouse(i: number, e: MouseEvent) {
    let point = getProjectedPointFromIndex(this.dataSet, i);
    let screenCoords =
        vector3DToScreenCoords(this.camera, this.width, this.height, point);
    let dpr = window.devicePixelRatio || 1;
    return dist_2D(
        [e.offsetX * dpr, e.offsetY * dpr], [screenCoords[0], screenCoords[1]]);
  }

  private adjustSelectionSphere(e: MouseEvent) {
    const dist = this.getDist2ToMouse(this.nearestPoint, e) / 100;
    this.selectionSphere.scale.set(dist, dist, dist);
    const selectedPoints: number[] = [];
    this.dataSet.points.forEach(point => {
      const pt = point.projectedPoint;
      const pointVect = new THREE.Vector3(pt[0], pt[1], pt[2]);
      const distPointToSphereOrigin =
          this.selectionSphere.position.clone().sub(pointVect).length();
      if (distPointToSphereOrigin < dist) {
        selectedPoints.push(this.dataSet.points.indexOf(point));
      }
    });
    this.selectionContext.notifySelectionChanged(selectedPoints);
  }

  private removeAll() {
    this.visualizers.forEach(v => {
      v.removeAllFromScene(this.scene);
    });
  }

  private createSelectionSphere() {
    let geometry = new THREE.SphereGeometry(1, 300, 100);
    let material = new THREE.MeshPhongMaterial({
      color: 0x000000,
      specular:
          (this.sceneIs3D() && 0xffffff),  // In 2d, make sphere look flat.
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
    this.selectionSphere.position.set(pos[0], pos[1], pos[2]);
  }

  private getLayoutValues(): Point2D {
    this.width = this.containerNode.offsetWidth;
    this.height = Math.max(1, this.containerNode.offsetHeight);
    return [this.width, this.height];
  }

  /**
   * Returns an x, y, z value for each item of our data based on the accessor
   * methods.
   */
  private getPointsCoordinates() {
    const xAccessor = this.pointAccessors[0];
    const yAccessor = this.pointAccessors[1];
    const zAccessor = this.pointAccessors[2];

    // Determine max and min of each axis of our data.
    const xExtent = d3.extent(this.dataSet.points, (p, i) => xAccessor(i));
    const yExtent = d3.extent(this.dataSet.points, (p, i) => yAccessor(i));
    const range = [-CUBE_LENGTH / 2, CUBE_LENGTH / 2];

    this.xScale.domain(xExtent).range(range);
    this.yScale.domain(yExtent).range(range);

    if (zAccessor) {
      const zExtent = d3.extent(this.dataSet.points, (p, i) => zAccessor(i));
      this.zScale.domain(zExtent).range(range);
    }

    // Determine 3d coordinates of each data point.
    this.dataSet.points.forEach((d, i) => {
      d.projectedPoint[0] = this.xScale(xAccessor(i));
      d.projectedPoint[1] = this.yScale(yAccessor(i));
    });

    if (zAccessor) {
      this.dataSet.points.forEach((d, i) => {
        d.projectedPoint[2] = this.zScale(zAccessor(i));
      });
    } else {
      this.dataSet.points.forEach((d, i) => {
        d.projectedPoint[2] = 0;
      });
    }
  }

  private addAxesToScene() {
    this.addVisualizer(new ScatterPlotVisualizerAxes());
  }

  private sceneIs3D(): boolean {
    return this.dimensionality === 3;
  }

  /** Set 2d vs 3d mode. */
  setDimensions(dimensionality: number) {
    if ((dimensionality !== 2) && (dimensionality !== 3)) {
      throw new RangeError('dimensionality must be 2 or 3');
    }
    this.dimensionality = dimensionality;
    const def = this.cameraDef || this.makeDefaultCameraDef(dimensionality);
    this.recreateCamera(def);
  }

  /** Gets the current camera information, suitable for serialization. */
  getCameraDef(): CameraDef {
    const def = new CameraDef();
    const pos = this.camera.position;
    const tgt = this.orbitCameraControls.target;
    def.orthographic = !this.sceneIs3D();
    def.position = [pos.x, pos.y, pos.z];
    def.target = [tgt.x, tgt.y, tgt.z];
    def.zoom = (this.camera as any).zoom;
    return def;
  }

  /** Sets parameters for the next camera recreation. */
  setCameraDefForNextCameraCreation(def: CameraDef) {
    this.cameraDef = def;
  }

  /** Gets the current camera position. */
  getCameraPosition(): Point3D {
    const currPos = this.camera.position;
    return [currPos.x, currPos.y, currPos.z];
  }

  /** Gets the current camera target. */
  getCameraTarget(): Point3D {
    let currTarget = this.orbitCameraControls.target;
    return [currTarget.x, currTarget.y, currTarget.z];
  }

  /** Sets up the camera from given position and target coordinates. */
  setCameraPositionAndTarget(position: Point3D, target: Point3D) {
    this.stopOrbitAnimation();
    this.camera.position.set(position[0], position[1], position[2]);
    this.orbitCameraControls.target.set(target[0], target[1], target[2]);
    this.orbitCameraControls.update();
    this.render();
  }

  /** Starts orbiting the camera around its current lookat target. */
  startOrbitAnimation() {
    if (!this.sceneIs3D()) {
      return;
    }
    if (this.orbitAnimationId != null) {
      this.stopOrbitAnimation();
    }
    this.orbitCameraControls.autoRotate = true;
    this.orbitCameraControls.rotateSpeed =
        ORBIT_ANIMATION_ROTATION_CYCLE_IN_SECONDS;
    this.updateOrbitAnimation();
  }

  private updateOrbitAnimation() {
    this.orbitCameraControls.update();
    this.orbitAnimationId =
        requestAnimationFrame(() => this.updateOrbitAnimation());
  }

  /** Stops the orbiting animation on the camera. */
  stopOrbitAnimation() {
    this.orbitCameraControls.autoRotate = false;
    this.orbitCameraControls.rotateSpeed = ORBIT_MOUSE_ROTATION_SPEED;
    if (this.orbitAnimationId != null) {
      cancelAnimationFrame(this.orbitAnimationId);
      this.orbitAnimationId = null;
    }
  }

  /** Adds a visualizer to the set, will start dispatching events to it */
  addVisualizer(visualizer: ScatterPlotVisualizer) {
    this.visualizers.push(visualizer);
    if (this.dataSet) {
      visualizer.onDataSet(this.dataSet, this.spriteImage);
    }
    if (this.labelAccessor) {
      visualizer.onSetLabelAccessor(this.labelAccessor);
    }
    if (this.scene) {
      visualizer.onRecreateScene(
          this.scene, this.sceneIs3D(), this.backgroundColor);
    }
  }

  /** Removes all visualizers attached to this scatter plot. */
  removeAllVisualizers() {
    this.removeAll();
    this.visualizers = [];
    this.addAxesToScene();
  }

  recreateScene() {
    this.removeAll();
    this.visualizers.forEach(v => {
      v.onRecreateScene(this.scene, this.sceneIs3D(), this.backgroundColor);
    });
    this.resize(false);
    this.render();
  }

  /** Sets the data for the scatter plot. */
  setDataSet(dataSet: DataSet, spriteImage: HTMLImageElement) {
    this.removeAll();
    this.dataSet = dataSet;
    this.spriteImage = spriteImage;
    this.nearestPoint = null;
    this.visualizers.forEach(v => {
      v.onDataSet(dataSet, spriteImage);
    });
    this.render();
  }

  update() {
    this.getPointsCoordinates();
    this.visualizers.forEach(v => {
      v.onUpdate(this.dataSet);
    });
    this.render();
  }

  render() {
    if (!this.dataSet) {
      return;
    }

    // place the light near the camera
    {
      const lightPos = this.camera.position.clone();
      lightPos.x += 1;
      lightPos.y += 1;
      this.light.position.set(lightPos.x, lightPos.y, lightPos.z);
    }

    const cameraSpacePointExtents: [number, number] = getNearFarPoints(
        this.dataSet, this.camera.position, this.orbitCameraControls.target);

    const rc = new RenderContext(
        this.camera, this.orbitCameraControls.target, this.width, this.height,
        cameraSpacePointExtents[0], cameraSpacePointExtents[1],
        this.pointColors, this.pointScaleFactors, this.labelAccessor,
        this.labels, this.traceColors);

    // Render first pass to picking target. This render fills pickingTexture
    // with colors that are actually point ids, so that sampling the texture at
    // the mouse's current x,y coordinates will reveal the data point that the
    // mouse is over.
    this.visualizers.forEach(v => {
      v.onPickingRender(rc);
    });

    this.renderer.render(this.scene, this.camera, this.pickingTexture);

    // Render second pass to color buffer, to be displayed on the canvas.
    this.visualizers.forEach(v => {
      v.onRender(rc);
    });

    this.renderer.render(this.scene, this.camera);
  }

  setPointAccessors(pointAccessors:
                        [PointAccessor, PointAccessor, PointAccessor]) {
    this.pointAccessors = pointAccessors;
  }

  setLabelAccessor(labelAccessor: (index: number) => string) {
    this.labelAccessor = labelAccessor;
    this.visualizers.forEach(v => {
      v.onSetLabelAccessor(labelAccessor);
    });
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

  /** Set the colors for every data point. (RGB triplets) */
  setPointColors(colors: Float32Array) {
    this.pointColors = colors;
  }

  /** Set the scale factors for every data point. (scalars) */
  setPointScaleFactors(scaleFactors: Float32Array) {
    this.pointScaleFactors = scaleFactors;
  }

  /** Set the labels to rendered */
  setLabels(labels: LabelRenderParams) {
    this.labels = labels;
  }

  /** Set the colors for every data trace. (RGB triplets) */
  setTraceColors(colors: {[trace: number]: Float32Array}) {
    this.traceColors = colors;
  }

  getMode(): Mode { return this.mode; }

  resetZoom() {
    this.recreateCamera(this.makeDefaultCameraDef(this.dimensionality));
    this.render();
  }

  setDayNightMode(isNight: boolean) {
    d3.select(this.containerNode)
        .selectAll('canvas')
        .style('filter', isNight ? 'invert(100%)' : null);
  }

  showAxes(show: boolean) {}
  showTickLabels(show: boolean) {}

  resize(render = true) {
    const [oldW, oldH] = [this.width, this.height];
    const [newW, newH] = this.getLayoutValues();

    if (this.dimensionality === 3) {
      const camera = (this.camera as THREE.PerspectiveCamera);
      camera.aspect = newW / newH;
      camera.updateProjectionMatrix();
    } else {
      const camera = (this.camera as THREE.OrthographicCamera);
      // Scale the ortho frustum by however much the window changed.
      const scaleW = newW / oldW;
      const scaleH = newH / oldH;
      const newCamHalfWidth = ((camera.right - camera.left) * scaleW) / 2;
      const newCamHalfHeight = ((camera.top - camera.bottom) * scaleH) / 2;
      camera.top = newCamHalfHeight;
      camera.bottom = -newCamHalfHeight;
      camera.left = -newCamHalfWidth;
      camera.right = newCamHalfWidth;
      camera.updateProjectionMatrix();
    }

    // Accouting for retina displays.
    const dpr = window.devicePixelRatio || 1;
    this.renderer.setPixelRatio(dpr);
    this.renderer.setSize(newW, newH);

    // the picking texture needs to be exactly the same as the render texture.
    {
      const renderCanvasSize = this.renderer.getSize();
      const pixelRatio = this.renderer.getPixelRatio();
      this.pickingTexture = new THREE.WebGLRenderTarget(
          renderCanvasSize.width * pixelRatio,
          renderCanvasSize.height * pixelRatio);
      this.pickingTexture.texture.minFilter = THREE.LinearFilter;
    }

    this.visualizers.forEach(v => {
      v.onResize(newW, newH);
    });

    if (render) {
      this.render();
    };
  }

  onCameraMove(listener: OnCameraMoveListener) {
    this.onCameraMoveListeners.push(listener);
  }

  clickOnPoint(pointIndex: number) {
    this.nearestPoint = pointIndex;
    this.onClick(null, false);
  }
}
