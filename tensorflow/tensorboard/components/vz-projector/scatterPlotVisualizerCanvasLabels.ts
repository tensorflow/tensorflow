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
import {RenderContext} from './renderContext';
import {DataSet} from './scatterPlot';
import {ScatterPlotVisualizer} from './scatterPlotVisualizer';
import {getProjectedPointFromIndex, vector3DToScreenCoords} from './util';

const MAX_LABELS_ON_SCREEN = 10000;

/**
 * Creates and maintains a 2d canvas on top of the GL canvas. All labels, when
 * active, are rendered to the 2d canvas as part of the visible render pass.
 */
export class ScatterPlotVisualizerCanvasLabels implements
    ScatterPlotVisualizer {
  private dataSet: DataSet;
  private gc: CanvasRenderingContext2D;
  private canvas: HTMLCanvasElement;
  private labelsActive: boolean = true;
  private sceneIs3D: boolean = true;

  constructor(container: d3.Selection<any>) {
    this.canvas = container.append('canvas').node() as HTMLCanvasElement;
    this.gc = this.canvas.getContext('2d');
    d3.select(this.canvas).style({position: 'absolute', left: 0, top: 0});
    this.canvas.style.pointerEvents = 'none';
  }

  private removeAllLabels() {
    const pixelWidth = this.canvas.width * window.devicePixelRatio;
    const pixelHeight = this.canvas.height * window.devicePixelRatio;
    this.gc.clearRect(0, 0, pixelWidth, pixelHeight);
  }

  /** Render all of the non-overlapping visible labels to the canvas. */
  private makeLabels(rc: RenderContext) {
    if (rc.labelIndices.length === 0) {
      return;
    }

    let strokeStylePrefix: string;
    let fillStylePrefix: string;
    {
      const ls = new THREE.Color(rc.labelStrokeColor).multiplyScalar(255);
      const lc = new THREE.Color(rc.labelFillColor).multiplyScalar(255);
      strokeStylePrefix = 'rgba(' + ls.r + ',' + ls.g + ',' + ls.b + ',';
      fillStylePrefix = 'rgba(' + lc.r + ',' + lc.g + ',' + lc.b + ',';
    }

    const labelHeight = parseInt(this.gc.font, 10);
    const dpr = window.devicePixelRatio;

    let grid: CollisionGrid;
    {
      const pixw = this.canvas.width * dpr;
      const pixh = this.canvas.height * dpr;
      const bb: BoundingBox = {loX: 0, hiX: pixw, loY: 0, hiY: pixh};
      grid = new CollisionGrid(bb, pixw / 25, pixh / 50);
    }

    const opacityRange =
        rc.farthestCameraSpacePointZ - rc.nearestCameraSpacePointZ;
    const camPos = rc.camera.position;
    const camToTarget = new THREE.Vector3().copy(camPos).sub(rc.cameraTarget);

    this.gc.lineWidth = 6;
    this.gc.textBaseline = 'middle';

    // Have extra space between neighboring labels. Don't pack too tightly.
    const labelMargin = 2;
    // Shift the label to the right of the point circle.
    const xShift = 4;

    const n = Math.min(MAX_LABELS_ON_SCREEN, rc.labelIndices.length);
    for (let i = 0; i < n; ++i) {
      const index = rc.labelIndices[i];
      const point = getProjectedPointFromIndex(this.dataSet, index);

      // discard points that are behind the camera
      const camToPoint = new THREE.Vector3().copy(camPos).sub(point);
      if (camToTarget.dot(camToPoint) < 0) {
        continue;
      }

      let [x, y] = vector3DToScreenCoords(
          rc.camera, rc.screenWidth, rc.screenHeight, point);
      x += xShift;

      // Computing the width of the font is expensive,
      // so we assume width of 1 at first. Then, if the label doesn't
      // conflict with other labels, we measure the actual width.
      const textBoundingBox = {
        loX: x - labelMargin,
        hiX: x + 1 + labelMargin,
        loY: y - labelHeight / 2 - labelMargin,
        hiY: y + labelHeight / 2 + labelMargin
      };

      if (grid.insert(textBoundingBox, true)) {
        const text = rc.labelAccessor(index);
        const fontSize =
            rc.labelDefaultFontSize * rc.labelScaleFactors[i] * dpr;
        this.gc.font = fontSize + 'px roboto';

        // Now, check with properly computed width.
        textBoundingBox.hiX += this.gc.measureText(text).width - 1;
        if (grid.insert(textBoundingBox)) {
          let p = new THREE.Vector3(point[0], point[1], point[2]);
          const distFromNearestPoint =
              camPos.distanceTo(p) - rc.nearestCameraSpacePointZ;
          // Opacity is scaled between 0.2 and 1, based on how far a label is
          // from the camera (Unless we are in 2d mode, in which case opacity is
          // just 1!)
          const opacity =
              this.sceneIs3D ? 1.2 - distFromNearestPoint / opacityRange : 1;
          this.gc.strokeStyle = strokeStylePrefix + opacity + ')';
          this.gc.fillStyle = fillStylePrefix + opacity + ')';
          this.gc.strokeText(text, x, y);
          this.gc.fillText(text, x, y);
        }
      }
    }
  }

  onDataSet(dataSet: DataSet, spriteImage: HTMLImageElement) {
    this.labelsActive = (spriteImage == null);
    this.dataSet = dataSet;
  }

  onResize(newWidth: number, newHeight: number) {
    let dpr = window.devicePixelRatio;
    d3.select(this.canvas)
        .attr('width', newWidth * dpr)
        .attr('height', newHeight * dpr)
        .style({width: newWidth + 'px', height: newHeight + 'px'});
  }

  onRecreateScene(
      scene: THREE.Scene, sceneIs3D: boolean, backgroundColor: number) {
    this.sceneIs3D = sceneIs3D;
  }

  removeAllFromScene(scene: THREE.Scene) {
    this.removeAllLabels();
  }

  onUpdate() {
    this.removeAllLabels();
  }

  onRender(rc: RenderContext) {
    if (!this.labelsActive) {
      return;
    }

    this.removeAllLabels();
    this.makeLabels(rc);
  }

  onPickingRender(renderContext: RenderContext) {}
  onSetLabelAccessor(labelAccessor: (index: number) => string) {}
}
