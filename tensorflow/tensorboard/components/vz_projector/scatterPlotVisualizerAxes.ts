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

/**
 * Maintains and renders 3d axes for the scatter plot.
 */
export class ScatterPlotVisualizerAxes implements ScatterPlotVisualizer {
  private axis: THREE.AxisHelper;

  constructor() {
    this.axis = new THREE.AxisHelper();
  }

  onDataSet(dataSet: DataSet) {}

  onRecreateScene(
      scene: THREE.Scene, sceneIs3D: boolean, backgroundColor: number) {
    if (sceneIs3D) {
      scene.add(this.axis);
    }
  }

  removeAllFromScene(scene: THREE.Scene) {
    scene.remove(this.axis);
  }

  onPickingRender(renderContext: RenderContext) {}
  onRender(renderContext: RenderContext) {}
  onUpdate(dataSet: DataSet) {}
  onResize(newWidth: number, newHeight: number) {}
  onSetLabelAccessor(labelAccessor: (index: number) => string) {}
}
