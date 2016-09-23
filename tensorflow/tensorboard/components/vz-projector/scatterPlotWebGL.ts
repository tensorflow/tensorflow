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
import {DataSet, Mode, OnHoverListener, OnSelectionListener, ScatterPlot} from './scatterPlot';
import {ScatterPlotWebGLVisualizer} from './scatterPlotWebGLVisualizer';
import {ScatterPlotWebGLVisualizerAxes} from './scatterPlotWebGLVisualizerAxes';
import {getNearFarPoints, getProjectedPointFromIndex, vector3DToScreenCoords} from './util';
import {dist_2D} from './vector';

const BACKGROUND_COLOR = 0xffffff;

const MAX_ZOOM = 10;
const MIN_ZOOM = .05;

// Constants relating to the camera parameters.
const FOV_VERTICAL = 70;
const NEAR = 0.01;
const FAR = 100;

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

/**
 * Maintains a three.js instantiation and context,
 * animation state, and all other logic that's
 * independent of how a 3D scatter plot is actually rendered. Also holds an
 * array of visualizers and dispatches application events to them.
 */
export class ScatterPlotWebGL implements ScatterPlot {
  private dataSet: DataSet;
  private containerNode: HTMLElement;

  private visualizers: ScatterPlotWebGLVisualizer[] = [];

  private highlightedPoints: number[] = [];
  private highlightStroke: (index: number) => string;
  private labeledPoints: number[] = [];
  private favorLabels: (i: number) => boolean;
  private labelAccessor: (index: number) => string;
  private colorAccessor: (index: number) => string;
  private onHoverListeners: OnHoverListener[] = [];
  private onSelectionListeners: OnSelectionListener[] = [];
  private lazySusanAnimation: number;

  // Accessors for rendering and labeling the points.
  private xAccessor: (index: number) => number;
  private yAccessor: (index: number) => number;
  private zAccessor: (index: number) => number;

  // Scaling functions for each axis.
  private xScale: d3.scale.Linear<number, number>;
  private yScale: d3.scale.Linear<number, number>;
  private zScale: d3.scale.Linear<number, number>;

  // window layout dimensions
  private height: number;
  private width: number;

  private mode: Mode;
  private backgroundColor: number = BACKGROUND_COLOR;

  private scene: THREE.Scene;
  private renderer: THREE.WebGLRenderer;
  private perspCamera: THREE.PerspectiveCamera;
  private cameraControls: any;
  private pickingTexture: THREE.WebGLRenderTarget;
  private light: THREE.PointLight;
  private selectionSphere: THREE.Mesh;

  private animating = false;
  private selecting = false;
  private nearestPoint: number;
  private mouseIsDown = false;
  private isDragSequence = false;
  private animationID: number;

  constructor(
      container: d3.Selection<any>, labelAccessor: (index: number) => string) {
    this.containerNode = container.node() as HTMLElement;
    this.getLayoutValues();

    this.labelAccessor = labelAccessor;
    this.xScale = d3.scale.linear();
    this.yScale = d3.scale.linear();
    this.zScale = d3.scale.linear();

    // Set up THREE.js.
    this.scene = new THREE.Scene();
    this.renderer = new THREE.WebGLRenderer();
    this.renderer.setClearColor(BACKGROUND_COLOR, 1);
    this.containerNode.appendChild(this.renderer.domElement);
    this.light = new THREE.PointLight(0xFFECBF, 1, 0);
    this.scene.add(this.light);
    this.makeCamera();

    // Render now so no black background appears during startup.
    this.renderer.render(this.scene, this.perspCamera);
    this.addInteractionListeners();

    this.addVisualizer(
        new ScatterPlotWebGLVisualizerAxes(this.xScale, this.yScale));
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

  /** Set up camera and camera's controller. */
  private makeCamera() {
    this.perspCamera = new THREE.PerspectiveCamera(
        FOV_VERTICAL, this.width / this.height, NEAR, FAR);
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
      this.render();
    });
    // End is called when the user stops interacting with the
    // orbit controls (e.g. on mouse up, after dragging).
    this.cameraControls.addEventListener('end', () => {
    });
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

  private onClick(e?: MouseEvent) {
    if (e && this.selecting) {
      return;
    }
    this.labeledPoints =
        this.highlightedPoints.filter((id, i) => this.favorLabels(i));
    let selection = this.nearestPoint || null;
    // Only call event handlers if the click originated from the scatter plot.
    if (e && !this.isDragSequence) {
      this.onSelectionListeners.forEach(l => l(selection ? [selection] : []));
    }
    this.isDragSequence = false;
    this.render();
  }

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
        this.cameraControls.mouseButtons.ORBIT === THREE.MOUSE.RIGHT) {
      // The user happened to press the ctrl key when the tab was active,
      // unpressed the ctrl when the tab was inactive, and now he/she
      // is back to the projector tab.
      this.cameraControls.mouseButtons.ORBIT = THREE.MOUSE.LEFT;
      this.cameraControls.mouseButtons.PAN = THREE.MOUSE.RIGHT;
    } else if (
        e.ctrlKey &&
        this.cameraControls.mouseButtons.ORBIT === THREE.MOUSE.LEFT) {
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
      this.cameraControls.autoRotate = false;
      cancelAnimationFrame(this.lazySusanAnimation);
    }
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
      let lastNearestPoint = this.nearestPoint;
      this.setNearestPointToMouse(e);
      if (lastNearestPoint !== this.nearestPoint) {
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
    if (this.pickingTexture == null) {
      this.nearestPoint = null;
      return;
    }

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
        (id !== 0xffffff) && (id < this.dataSet.points.length) ? id : null;
  }

  /** Returns the squared distance to the mouse for the i-th point. */
  private getDist2ToMouse(i: number, e: MouseEvent) {
    let point = getProjectedPointFromIndex(this.dataSet, i);
    let screenCoords = vector3DToScreenCoords(
        this.perspCamera, this.width, this.height, point);
    let dpr = window.devicePixelRatio;
    return dist_2D(
        [e.offsetX * dpr, e.offsetY * dpr], [screenCoords[0], screenCoords[1]]);
  }

  private adjustSelectionSphere(e: MouseEvent) {
    let dist = this.getDist2ToMouse(this.nearestPoint, e) / 100;
    this.selectionSphere.scale.set(dist, dist, dist);
    let selectedPoints: number[] = [];
    this.dataSet.points.forEach(point => {
      let pt = point.projectedPoint;
      let pointVect = new THREE.Vector3(pt[0], pt[1], pt[2]);
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
    this.onSelectionListeners.forEach(l => l(selectedPoints));
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
    if (currPos.distanceTo(pos) > 0.03) {
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
      this.render();
      if (callback) {
        callback();
      }
    }
  }

  /** Removes all geometry from the scene. */
  private removeAll() {
    this.visualizers.forEach(v => {
      v.removeAllFromScene(this.scene);
    });
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
    this.selectionSphere.position.set(pos[0], pos[1], pos[2]);
  }

  private getLayoutValues() {
    this.width = this.containerNode.offsetWidth;
    this.height = Math.max(1, this.containerNode.offsetHeight);
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

  // PUBLIC API

  /** Adds a visualizer to the set, will start dispatching events to it */
  addVisualizer(visualizer: ScatterPlotWebGLVisualizer) {
    this.visualizers.push(visualizer);
  }

  recreateScene() {
    this.removeAll();
    this.cancelAnimation();
    let sceneIs3D = this.zAccessor != null;
    if (sceneIs3D) {
      this.makeCamera3D();
    } else {
      this.makeCamera2D();
    }
    this.visualizers.forEach(v => {
      v.onRecreateScene(this.scene, sceneIs3D, this.backgroundColor);
    });
    this.resize(false);
    this.render();
  }

  /** Sets the data for the scatter plot. */
  setDataSet(dataSet: DataSet, spriteImage: HTMLImageElement) {
    this.removeAll();
    this.dataSet = dataSet;
    this.labeledPoints = [];
    this.highlightedPoints = [];
    this.visualizers.forEach(v => {
      v.onDataSet(dataSet, spriteImage);
    });
  }

  update() {
    this.cancelAnimation();
    this.getPointsCoordinates();
    this.visualizers.forEach(v => {
      v.onUpdate();
    });
    this.render();
  }

  render() {
    if (!this.dataSet) {
      return;
    }

    let cameraSpacePointExtents: [number, number] = getNearFarPoints(
        this.dataSet, this.perspCamera.position, this.cameraControls.target);

    // Render first pass to picking target. This render fills pickingTexture
    // with colors that are actually point ids, so that sampling the texture at
    // the mouse's current x,y coordinates will reveal the data point that the
    // mouse is over.
    this.visualizers.forEach(v => {
      v.onPickingRender(this.perspCamera, this.cameraControls.target);
    });
    this.renderer.render(this.scene, this.perspCamera, this.pickingTexture);

    // Render second pass to color buffer, to be displayed on the canvas.
    let lightPos = new THREE.Vector3().copy(this.perspCamera.position);
    lightPos.x += 1;
    lightPos.y += 1;
    this.light.position.set(lightPos.x, lightPos.y, lightPos.z);

    let rc = new RenderContext(
        this.perspCamera, this.cameraControls.target, this.width, this.height,
        cameraSpacePointExtents[0], cameraSpacePointExtents[1],
        this.colorAccessor, this.labeledPoints, this.labelAccessor,
        this.highlightedPoints, this.highlightStroke);

    this.visualizers.forEach(v => {
      v.onRender(rc);
    });
    this.renderer.render(this.scene, this.perspCamera);
  }

  setColorAccessor(colorAccessor: (index: number) => string) {
    this.colorAccessor = colorAccessor;
    this.render();
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

  getMode(): Mode { return this.mode; }

  resetZoom() {
    if (this.animating) {
      return;
    }
    let resetPos = this.cameraControls.position0;
    let resetTarget = this.cameraControls.target0;
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
    let p = new THREE.Vector3().copy(this.perspCamera.position).add(zoomVect);
    let d = p.distanceTo(this.cameraControls.target);

    // Make sure that we're not too far zoomed in. If not, zoom!
    if ((d > MIN_ZOOM) && (d < MAX_ZOOM)) {
      this.animate(p, this.cameraControls.target);
    }
  }

  highlightPoints(
      pointIndexes: number[], highlightStroke: (i: number) => string,
      favorLabels: (i: number) => boolean) {
    this.favorLabels = favorLabels;
    this.highlightedPoints = pointIndexes;
    this.labeledPoints = pointIndexes;
    this.highlightStroke = highlightStroke;
    this.render();
  }

  getHighlightedPoints(): number[] { return this.highlightedPoints; }

  setDayNightMode(isNight: boolean) {
    d3.select(this.containerNode)
        .selectAll('canvas')
        .style('filter', isNight ? 'invert(100%)' : null);
  }

  showAxes(show: boolean) {}
  showTickLabels(show: boolean) {}
  setAxisLabels(xLabel: string, yLabel: string) {}

  resize(render = true) {
    this.getLayoutValues();
    this.perspCamera.aspect = this.width / this.height;
    this.perspCamera.updateProjectionMatrix();
    // Accouting for retina displays.
    this.renderer.setPixelRatio(window.devicePixelRatio || 1);
    this.renderer.setSize(this.width, this.height);
    this.pickingTexture = new THREE.WebGLRenderTarget(this.width, this.height);
    this.pickingTexture.texture.minFilter = THREE.LinearFilter;
    this.visualizers.forEach(v => {
      v.onResize(this.width, this.height);
    });
    if (render) {
      this.render();
    };
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
