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

const AXIS_COLOR = 0xb3b3b3;

/**
 * Maintains and renders 2d and 3d axes for the scatter plot.
 * 2d axes are scaled relative to the data set.
 */
export class ScatterPlotVisualizerAxes implements ScatterPlotVisualizer {
  private axis3D: THREE.AxisHelper;
  private axis2D: THREE.LineSegments;
  private sceneIs3D: boolean = true;
  private xScale: d3.scale.Linear<number, number>;
  private yScale: d3.scale.Linear<number, number>;

  constructor(
      xScale: d3.scale.Linear<number, number>,
      yScale: d3.scale.Linear<number, number>) {
    this.xScale = xScale;
    this.yScale = yScale;
    this.axis3D = new THREE.AxisHelper();
  }

  private createAxis2D() {
    if (this.axis2D) {
      this.axis2D.material.dispose();
      this.axis2D.geometry.dispose();
    }

    let vertices = new Float32Array([
      0,  // origin
      0,
      0,
      this.xScale(1),  // x axis
      0,
      0,
      0,  // origin
      0,
      0,
      0,  // y axis
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

    const RGB_NUM_BYTES = 3;
    const XYZ_NUM_BYTES = 3;

    let lineGeometry = new THREE.BufferGeometry();
    lineGeometry.addAttribute(
        'position', new THREE.BufferAttribute(vertices, XYZ_NUM_BYTES));
    lineGeometry.addAttribute(
        'color', new THREE.BufferAttribute(axisColors, RGB_NUM_BYTES));
    let material =
        new THREE.LineBasicMaterial({vertexColors: THREE.VertexColors});
    this.axis2D = new THREE.LineSegments(lineGeometry, material);
  }

  onDataSet(dataSet: DataSet) {}

  onRecreateScene(
      scene: THREE.Scene, sceneIs3D: boolean, backgroundColor: number) {
    this.sceneIs3D = sceneIs3D;
    if (sceneIs3D) {
      scene.add(this.axis3D);
    } else {
      this.createAxis2D();
      scene.add(this.axis2D);
    }
  }

  removeAllFromScene(scene: THREE.Scene) {
    if (this.sceneIs3D) {
      scene.remove(this.axis3D);
    } else {
      scene.remove(this.axis2D);
    }
  }

  onPickingRender(camera: THREE.Camera, cameraTarget: THREE.Vector3) {}
  onRender(renderContext: RenderContext) {}
  onUpdate() {}
  onResize(newWidth: number, newHeight: number) {}
  onSetLabelAccessor(labelAccessor: (index: number) => string) {}
}
