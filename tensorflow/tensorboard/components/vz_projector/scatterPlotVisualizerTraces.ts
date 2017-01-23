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

import {DataSet} from './data';
import {RenderContext} from './renderContext';
import {ScatterPlotVisualizer} from './scatterPlotVisualizer';
import * as util from './util';

const RGB_NUM_ELEMENTS = 3;
const XYZ_NUM_ELEMENTS = 3;

/**
 * Renders 'traces' (polylines) that connect multiple points in the dataset
 */
export class ScatterPlotVisualizerTraces implements ScatterPlotVisualizer {
  private dataSet: DataSet;
  private scene: THREE.Scene;
  private traces: THREE.Line[];
  private tracePositionBuffer: {[trace: number]: THREE.BufferAttribute} = {};
  private traceColorBuffer: {[trace: number]: THREE.BufferAttribute} = {};

  private updateTraceIndicesInDataSet(ds: DataSet) {
    for (let i = 0; i < ds.traces.length; i++) {
      const trace = ds.traces[i];
      for (let j = 0; j < trace.pointIndices.length - 1; j++) {
        ds.points[trace.pointIndices[j]].traceIndex = i;
        ds.points[trace.pointIndices[j + 1]].traceIndex = i;
      }
    }
  }

  private createTraces(scene: THREE.Scene) {
    if (!this.dataSet || !this.dataSet.traces) {
      return;
    }

    this.updateTraceIndicesInDataSet(this.dataSet);
    this.traces = [];

    for (let i = 0; i < this.dataSet.traces.length; i++) {
      const geometry = new THREE.BufferGeometry();
      geometry.addAttribute('position', this.tracePositionBuffer[i]);
      geometry.addAttribute('color', this.traceColorBuffer[i]);

      const material = new THREE.LineBasicMaterial({
        linewidth: 1,  // unused default, overwritten by width array.
        opacity: 1.0,  // unused default, overwritten by opacity array.
        transparent: true,
        vertexColors: THREE.VertexColors
      });

      const trace = new THREE.LineSegments(geometry, material);
      trace.frustumCulled = false;
      this.traces.push(trace);
      scene.add(trace);
    }
  }

  dispose() {
    if (this.traces == null) {
      return;
    }
    for (let i = 0; i < this.traces.length; i++) {
      this.scene.remove(this.traces[i]);
      this.traces[i].geometry.dispose();
    }
    this.traces = null;
    this.tracePositionBuffer = {};
    this.traceColorBuffer = {};
  }

  setScene(scene: THREE.Scene) {
    this.scene = scene;
  }

  setDataSet(dataSet: DataSet) {
    this.dataSet = dataSet;
  }

  onPointPositionsChanged(newPositions: Float32Array) {
    if ((newPositions == null) || (this.traces != null)) {
      this.dispose();
    }
    if ((newPositions == null) || (this.dataSet == null)) {
      return;
    }
    // Set up the position buffer arrays for each trace.
    for (let i = 0; i < this.dataSet.traces.length; i++) {
      let dataTrace = this.dataSet.traces[i];
      const vertexCount = 2 * (dataTrace.pointIndices.length - 1);

      let traces = new Float32Array(vertexCount * XYZ_NUM_ELEMENTS);
      this.tracePositionBuffer[i] =
          new THREE.BufferAttribute(traces, XYZ_NUM_ELEMENTS);

      let colors = new Float32Array(vertexCount * RGB_NUM_ELEMENTS);
      this.traceColorBuffer[i] =
          new THREE.BufferAttribute(colors, RGB_NUM_ELEMENTS);
    }
    for (let i = 0; i < this.dataSet.traces.length; i++) {
      const dataTrace = this.dataSet.traces[i];
      let src = 0;
      for (let j = 0; j < dataTrace.pointIndices.length - 1; j++) {
        const p1Index = dataTrace.pointIndices[j];
        const p2Index = dataTrace.pointIndices[j + 1];
        const p1 = util.vector3FromPackedArray(newPositions, p1Index);
        const p2 = util.vector3FromPackedArray(newPositions, p2Index);
        this.tracePositionBuffer[i].setXYZ(src, p1.x, p1.y, p1.z);
        this.tracePositionBuffer[i].setXYZ(src + 1, p2.x, p2.y, p2.z);
        src += 2;
      }
      this.tracePositionBuffer[i].needsUpdate = true;
    }

    if (this.traces == null) {
      this.createTraces(this.scene);
    }
  }

  onRender(renderContext: RenderContext) {
    if (this.traces == null) {
      return;
    }
    for (let i = 0; i < this.traces.length; i++) {
      this.traces[i].material.opacity = renderContext.traceOpacities[i];
      (this.traces[i].material as THREE.LineBasicMaterial).linewidth =
          renderContext.traceWidths[i];
      this.traceColorBuffer[i].array = renderContext.traceColors[i];
      this.traceColorBuffer[i].needsUpdate = true;
    }
  }

  onPickingRender(renderContext: RenderContext) {}
  onResize(newWidth: number, newHeight: number) {}
}
