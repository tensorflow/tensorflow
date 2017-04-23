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

import {ProjectorEventContext} from './projectorEventContext';
import {CameraType, LabelRenderParams, RenderContext} from './renderContext';
import {BoundingBox, ScatterPlotRectangleSelector} from './scatterPlotRectangleSelector';
import {ScatterPlotVisualizer} from './scatterPlotVisualizer';
import * as util from './util';
import {Point2D, Point3D} from './vector';

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

const START_CAMERA_POS_3D = new THREE.Vector3(0.45, 0.9, 1.6);
const START_CAMERA_TARGET_3D = new THREE.Vector3(0, 0, 0);
const START_CAMERA_POS_2D = new THREE.Vector3(0, 0, 4);
const START_CAMERA_TARGET_2D = new THREE.Vector3(0, 0, 0);

const ORBIT_MOUSE_ROTATION_SPEED = 1;
const ORBIT_ANIMATION_ROTATION_CYCLE_IN_SECONDS = 7;

export type OnCameraMoveListener =
    (cameraPosition: THREE.Vector3, cameraTarget: THREE.Vector3) => void;

/** Supported modes of interaction. */
export enum MouseMode {
  AREA_SELECT,
  CAMERA_AND_CLICK_SELECT
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
  private visualizers: ScatterPlotVisualizer[] = [];

  private onCameraMoveListeners: OnCameraMoveListener[] = [];

  private height: number;
  private width: number;

  private mouseMode: MouseMode;
  private backgroundColor: number = BACKGROUND_COLOR;

  private dimensionality: number = 3;
  private renderer: THREE.WebGLRenderer;

  private scene: THREE.Scene;
  private pickingTexture: THREE.WebGLRenderTarget;
  private light: THREE.PointLight;

  private cameraDef: CameraDef = null;
  private camera: THREE.Camera;
  private orbitAnimationOnNextCameraCreation: boolean = false;
  private orbitCameraControls: any;
  private orbitAnimationId: number;

  private worldSpacePointPositions: Float32Array;
  private pointColors: Float32Array;
  private pointScaleFactors: Float32Array;
  private labels: LabelRenderParams;
  private polylineColors: {[polylineIndex: number]: Float32Array};
  private polylineOpacities: Float32Array;
  private polylineWidths: Float32Array;

  private selecting = false;
  private nearestPoint: number;
  private mouseIsDown = false;
  private isDragSequence = false;
  private rectangleSelector: ScatterPlotRectangleSelector;

  constructor(
      private container: HTMLElement,
      private projectorEventContext: ProjectorEventContext) {
    this.getLayoutValues();

    this.scene = new THREE.Scene();
    this.renderer = new THREE.WebGLRenderer(
        {alpha: true, premultipliedAlpha: false, antialias: false});
    this.renderer.setClearColor(BACKGROUND_COLOR, 1);
    this.container.appendChild(this.renderer.domElement);
    this.light = new THREE.PointLight(0xFFECBF, 1, 0);
    this.scene.add(this.light);

    this.setDimensions(3);
    this.recreateCamera(this.makeDefaultCameraDef(this.dimensionality));
    this.renderer.render(this.scene, this.camera);

    this.rectangleSelector = new ScatterPlotRectangleSelector(
        this.container,
        (boundingBox: BoundingBox) => this.selectBoundingBox(boundingBox));
    this.addInteractionListeners();
  }

  private addInteractionListeners() {
    this.container.addEventListener('mousemove', this.onMouseMove.bind(this));
    this.container.addEventListener('mousedown', this.onMouseDown.bind(this));
    this.container.addEventListener('mouseup', this.onMouseUp.bind(this));
    this.container.addEventListener('click', this.onClick.bind(this));
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

  private makeOrbitControls(
      camera: THREE.Camera, cameraDef: CameraDef, cameraIs3D: boolean) {
    if (this.orbitCameraControls != null) {
      this.orbitCameraControls.dispose();
    }
    const occ =
        new (THREE as any).OrbitControls(camera, this.renderer.domElement);
    occ.target0 = new THREE.Vector3(
        cameraDef.target[0], cameraDef.target[1], cameraDef.target[2]);
    occ.position0 = new THREE.Vector3().copy(camera.position);
    occ.zoom0 = cameraDef.zoom;
    occ.enableRotate = cameraIs3D;
    occ.autoRotate = false;
    occ.rotateSpeed = ORBIT_MOUSE_ROTATION_SPEED;
    if (cameraIs3D) {
      occ.mouseButtons.ORBIT = THREE.MOUSE.LEFT;
      occ.mouseButtons.PAN = THREE.MOUSE.RIGHT;
    } else {
      occ.mouseButtons.ORBIT = null;
      occ.mouseButtons.PAN = THREE.MOUSE.LEFT;
    }
    occ.reset();

    this.camera = camera;
    this.orbitCameraControls = occ;
    this.addCameraControlsEventListeners(this.orbitCameraControls);
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
      camera.updateProjectionMatrix();
    }
    this.camera = camera;
    this.makeOrbitControls(camera, cameraDef, true);
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
      camera.updateProjectionMatrix();
    }
    this.camera = camera;
    this.makeOrbitControls(camera, cameraDef, false);
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
    if (this.orbitAnimationOnNextCameraCreation) {
      this.startOrbitAnimation();
    }
  }

  private onClick(e?: MouseEvent, notify = true) {
    if (e && this.selecting) {
      return;
    }
    // Only call event handlers if the click originated from the scatter plot.
    if (!this.isDragSequence && notify) {
      const selection = (this.nearestPoint != null) ? [this.nearestPoint] : [];
      this.projectorEventContext.notifySelectionChanged(selection);
    }
    this.isDragSequence = false;
    this.render();
  }

  private onMouseDown(e: MouseEvent) {
    this.isDragSequence = false;
    this.mouseIsDown = true;
    if (this.selecting) {
      this.orbitCameraControls.enabled = false;
      this.rectangleSelector.onMouseDown(e.offsetX, e.offsetY);
      this.setNearestPointToMouse(e);
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
      this.rectangleSelector.onMouseUp();
      this.render();
    }
    this.mouseIsDown = false;
  }

  /**
   * When the mouse moves, find the nearest point (if any) and send it to the
   * hoverlisteners (usually called from embedding.ts)
   */
  private onMouseMove(e: MouseEvent) {
    this.isDragSequence = this.mouseIsDown;
    // Depending if we're selecting or just navigating, handle accordingly.
    if (this.selecting && this.mouseIsDown) {
      this.rectangleSelector.onMouseMove(e.offsetX, e.offsetY);
      this.render();
    } else if (!this.mouseIsDown) {
      this.setNearestPointToMouse(e);
      this.projectorEventContext.notifyHoverOverPoint(this.nearestPoint);
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
      this.container.style.cursor = 'crosshair';
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
      this.selecting = (this.getMouseMode() === MouseMode.AREA_SELECT);
      if (!this.selecting) {
        this.container.style.cursor = 'default';
      }
      this.render();
    }
  }

  /**
   * Returns a list of indices of points in a bounding box from the picking
   * texture.
   * @param boundingBox The bounding box to select from.
   */
  private getPointIndicesFromPickingTexture(boundingBox: BoundingBox):
      number[] {
    if (this.worldSpacePointPositions == null) {
      return null;
    }
    const pointCount = this.worldSpacePointPositions.length / 3;
    const dpr = window.devicePixelRatio || 1;
    const x = Math.floor(boundingBox.x * dpr);
    const y = Math.floor(boundingBox.y * dpr);
    const width = Math.floor(boundingBox.width * dpr);
    const height = Math.floor(boundingBox.height * dpr);

    // Create buffer for reading all of the pixels from the texture.
    let pixelBuffer = new Uint8Array(width * height * 4);

    // Read the pixels from the bounding box.
    this.renderer.readRenderTargetPixels(
        this.pickingTexture, x, this.pickingTexture.height - y, width, height,
        pixelBuffer);

    // Keep a flat list of each point and whether they are selected or not. This
    // approach is more efficient than using an object keyed by the index.
    let pointIndicesSelection =
        new Uint8Array(this.worldSpacePointPositions.length);
    for (let i = 0; i < width * height; i++) {
      const id = (pixelBuffer[i * 4] << 16) | (pixelBuffer[i * 4 + 1] << 8) |
          pixelBuffer[i * 4 + 2];
      if (id !== 0xffffff && (id < pointCount)) {
        pointIndicesSelection[id] = 1;
      }
    }
    let pointIndices: number[] = [];
    for (let i = 0; i < pointIndicesSelection.length; i++) {
      if (pointIndicesSelection[i] === 1) {
        pointIndices.push(i);
      }
    }

    return pointIndices;
  }


  private selectBoundingBox(boundingBox: BoundingBox) {
    let pointIndices = this.getPointIndicesFromPickingTexture(boundingBox);
    this.projectorEventContext.notifySelectionChanged(pointIndices);
  }

  private setNearestPointToMouse(e: MouseEvent) {
    if (this.pickingTexture == null) {
      this.nearestPoint = null;
      return;
    }
    const boundingBox:
        BoundingBox = {x: e.offsetX, y: e.offsetY, width: 1, height: 1};
    const pointIndices = this.getPointIndicesFromPickingTexture(boundingBox);
    this.nearestPoint = (pointIndices != null) ? pointIndices[0] : null;
  }

  private getLayoutValues(): Point2D {
    this.width = this.container.offsetWidth;
    this.height = Math.max(1, this.container.offsetHeight);
    return [this.width, this.height];
  }

  private sceneIs3D(): boolean {
    return this.dimensionality === 3;
  }

  private remove3dAxisFromScene(): THREE.Object3D {
    const axes = this.scene.getObjectByName('axes');
    if (axes != null) {
      this.scene.remove(axes);
    }
    return axes;
  }

  private add3dAxis() {
    const axes = new THREE.AxisHelper();
    axes.name = 'axes';
    this.scene.add(axes);
  }

  /** Set 2d vs 3d mode. */
  setDimensions(dimensionality: number) {
    if ((dimensionality !== 2) && (dimensionality !== 3)) {
      throw new RangeError('dimensionality must be 2 or 3');
    }
    this.dimensionality = dimensionality;

    const def = this.cameraDef || this.makeDefaultCameraDef(dimensionality);
    this.recreateCamera(def);

    this.remove3dAxisFromScene();
    if (dimensionality === 3) {
      this.add3dAxis();
    }
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
  setCameraParametersForNextCameraCreation(
      def: CameraDef, orbitAnimation: boolean) {
    this.cameraDef = def;
    this.orbitAnimationOnNextCameraCreation = orbitAnimation;
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
    if (this.scene) {
      visualizer.setScene(this.scene);
    }
    visualizer.onResize(this.width, this.height);
    visualizer.onPointPositionsChanged(this.worldSpacePointPositions);
    this.visualizers.push(visualizer);
  }

  /** Removes all visualizers attached to this scatter plot. */
  removeAllVisualizers() {
    this.visualizers.forEach(v => v.dispose());
    this.visualizers = [];
  }

  /** Update scatter plot with a new array of packed xyz point positions. */
  setPointPositions(worldSpacePointPositions: Float32Array) {
    this.worldSpacePointPositions = worldSpacePointPositions;
    this.visualizers.forEach(
        v => v.onPointPositionsChanged(worldSpacePointPositions));
  }

  render() {
    {
      const lightPos = this.camera.position.clone();
      lightPos.x += 1;
      lightPos.y += 1;
      this.light.position.set(lightPos.x, lightPos.y, lightPos.z);
    }

    const cameraType = (this.camera instanceof THREE.PerspectiveCamera) ?
        CameraType.Perspective :
        CameraType.Orthographic;

    let cameraSpacePointExtents: [number, number] = [0, 0];
    if (this.worldSpacePointPositions != null) {
      cameraSpacePointExtents = util.getNearFarPoints(
          this.worldSpacePointPositions, this.camera.position,
          this.orbitCameraControls.target);
    }

    const rc = new RenderContext(
        this.camera, cameraType, this.orbitCameraControls.target, this.width,
        this.height, cameraSpacePointExtents[0], cameraSpacePointExtents[1],
        this.backgroundColor, this.pointColors, this.pointScaleFactors,
        this.labels, this.polylineColors, this.polylineOpacities,
        this.polylineWidths);

    // Render first pass to picking target. This render fills pickingTexture
    // with colors that are actually point ids, so that sampling the texture at
    // the mouse's current x,y coordinates will reveal the data point that the
    // mouse is over.
    this.visualizers.forEach(v => v.onPickingRender(rc));

    {
      const axes = this.remove3dAxisFromScene();
      this.renderer.render(this.scene, this.camera, this.pickingTexture);
      if (axes != null) {
        this.scene.add(axes);
      }
    }

    // Render second pass to color buffer, to be displayed on the canvas.
    this.visualizers.forEach(v => v.onRender(rc));

    this.renderer.render(this.scene, this.camera);
  }

  setMouseMode(mouseMode: MouseMode) {
    this.mouseMode = mouseMode;
    if (mouseMode === MouseMode.AREA_SELECT) {
      this.selecting = true;
      this.container.style.cursor = 'crosshair';
    } else {
      this.selecting = false;
      this.container.style.cursor = 'default';
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

  /** Set the colors for every data polyline. (RGB triplets) */
  setPolylineColors(colors: {[polylineIndex: number]: Float32Array}) {
    this.polylineColors = colors;
  }

  setPolylineOpacities(opacities: Float32Array) {
    this.polylineOpacities = opacities;
  }

  setPolylineWidths(widths: Float32Array) {
    this.polylineWidths = widths;
  }

  getMouseMode(): MouseMode {
    return this.mouseMode;
  }

  resetZoom() {
    this.recreateCamera(this.makeDefaultCameraDef(this.dimensionality));
    this.render();
  }

  setDayNightMode(isNight: boolean) {
    const canvases = this.container.querySelectorAll('canvas');
    const filterValue = isNight ? 'invert(100%)' : null;
    for (let i = 0; i < canvases.length; i++) {
      canvases[i].style.filter = filterValue;
    }
  }

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

    this.visualizers.forEach(v => v.onResize(newW, newH));

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
