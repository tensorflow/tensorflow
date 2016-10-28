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

const TRACE_DEFAULT_OPACITY = .2;
const TRACE_DEFAULT_LINEWIDTH = 2;
const TRACE_SELECTED_OPACITY = .9;
const TRACE_SELECTED_LINEWIDTH = 3;
const TRACE_DESELECTED_OPACITY = .05;

const RGB_NUM_ELEMENTS = 3;
const XYZ_NUM_ELEMENTS = 3;

/**
 * Renders 'traces' (polylines) that connect multiple points in the dataset
 */
export class ScatterPlotVisualizerTraces implements ScatterPlotVisualizer {
  private dataSet: DataSet;
  private traces: THREE.Line[];
  private tracePositionBuffer: {[trace: number]: THREE.BufferAttribute} = {};
  private traceColorBuffer: {[trace: number]: THREE.BufferAttribute} = {};

  constructor(selectionContext: SelectionContext) {
    selectionContext.registerSelectionChangedListener(
        (s: number[]) => this.onSelectionChanged(s));
  }

  /**
   * Create line traces between connected points and instantiate the geometry.
   */
  private addTraces(scene: THREE.Scene) {
    if (!this.dataSet || !this.dataSet.traces) {
      return;
    }

    this.traces = [];

    for (let i = 0; i < this.dataSet.traces.length; i++) {
      let dataTrace = this.dataSet.traces[i];

      for (let j = 0; j < dataTrace.pointIndices.length - 1; j++) {
        this.dataSet.points[dataTrace.pointIndices[j]].traceIndex = i;
        this.dataSet.points[dataTrace.pointIndices[j + 1]].traceIndex = i;
      }

      let geometry = new THREE.BufferGeometry();

      geometry.addAttribute('position', this.tracePositionBuffer[i]);
      this.tracePositionBuffer[i].needsUpdate = true;

      geometry.addAttribute('color', this.traceColorBuffer[i]);
      this.traceColorBuffer[i].needsUpdate = true;

      // We use the same material for every line.
      let material = new THREE.LineBasicMaterial({
        linewidth: TRACE_DEFAULT_LINEWIDTH,
        opacity: TRACE_DEFAULT_OPACITY,
        transparent: true,
        vertexColors: THREE.VertexColors
      });

      let trace = new THREE.LineSegments(geometry, material);
      this.traces.push(trace);
      scene.add(trace);
    }
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

  removeAllFromScene(scene: THREE.Scene) {
    if (!this.traces) {
      return;
    }
    for (let i = 0; i < this.traces.length; i++) {
      scene.remove(this.traces[i]);
    }
    this.traces = [];
  }

  onDataSet(dataSet: DataSet, spriteImage: HTMLImageElement) {
    this.dataSet = dataSet;
    if (dataSet) {
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
    }
  }

  onSelectionChanged(selection: number[]) {
    this.resetTraces();
    if (selection.length > 0) {
      let selectedIndex = selection[0];
      let traceIndex = this.dataSet.points[selectedIndex].traceIndex;
      if (traceIndex) {
        for (let i = 0; i < this.traces.length; i++) {
          this.traces[i].material.opacity = TRACE_DESELECTED_OPACITY;
          this.traces[i].material.needsUpdate = true;
        }
        this.traces[traceIndex].material.opacity = TRACE_SELECTED_OPACITY;
        (this.traces[traceIndex].material as THREE.LineBasicMaterial)
            .linewidth = TRACE_SELECTED_LINEWIDTH;
        this.traces[traceIndex].material.needsUpdate = true;
      }
    }
  }

  onRecreateScene(
      scene: THREE.Scene, sceneIs3D: boolean, backgroundColor: number) {
    this.addTraces(scene);
  }

  onUpdate() {
    if (!this.dataSet) {
      return;
    }
    for (let i = 0; i < this.dataSet.traces.length; i++) {
      let dataTrace = this.dataSet.traces[i];

      let vertexCount = 0;
      for (let j = 0; j < dataTrace.pointIndices.length - 1; j++) {
        let point1 = this.dataSet.points[dataTrace.pointIndices[j]];
        let point2 = this.dataSet.points[dataTrace.pointIndices[j + 1]];

        this.tracePositionBuffer[i].setXYZ(
            vertexCount, point1.projectedPoint[0], point1.projectedPoint[1],
            point1.projectedPoint[2]);
        this.tracePositionBuffer[i].setXYZ(
            vertexCount + 1, point2.projectedPoint[0], point2.projectedPoint[1],
            point2.projectedPoint[2]);
        vertexCount += 2;
      }
    }

    for (let i = 0; i < this.dataSet.traces.length; i++) {
      this.tracePositionBuffer[i].needsUpdate = true;
    }
  }

  onRender(renderContext: RenderContext) {
    for (let i = 0; i < this.dataSet.traces.length; i++) {
      this.traceColorBuffer[i].array = renderContext.traceColors[i];
      this.traceColorBuffer[i].needsUpdate = true;
    }
  }

  onPickingRender(renderContext: RenderContext) {}
  onResize(newWidth: number, newHeight: number) {}
  onSetLabelAccessor(labelAccessor: (index: number) => string) {}
}
