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


/**
 * ScatterPlotVisualizer is an interface used by ScatterPlotContainer
 * to manage and aggregate any number of concurrent visualization behaviors.
 * To add a new visualization to the 3D scatter plot, create a new class that
 * implements this interface and attach it to the ScatterPlotContainer.
 */
export interface ScatterPlotVisualizer {
  /**
   * Called when the main scatter plot builds the scene. Add all 3D geometry to
   * the scene here, set up lights, etc.
   */
  onRecreateScene(
      scene: THREE.Scene, sceneIs3D: boolean, backgroundColor: number);
  /**
   * Called when the main scatter plot tears down the scene. Remove all
   * elements from the 3D scene here.
   */
  removeAllFromScene(scene: THREE.Scene);
  /**
   * Called when the projector data set changes. Do any dataset-specific
   * initialization here.
   */
  onDataSet(dataSet: DataSet, spriteImage: HTMLImageElement);
  /**
   * Called when the label accessor (functor that maps point ids to text labels)
   * changes. The label accessor is also part of RenderContext, but visualizers
   * may need it outside of a render call, to learn when it changes.
   */
  onSetLabelAccessor(labelAccessor: (index: number) => string);
  /**
   * Called immediately before the main scatter plot performs a picking
   * (selection) render. Set up render state for any geometry to use picking IDs
   * instead of visual colors.
   */
  onPickingRender(camera: THREE.Camera, cameraTarget: THREE.Vector3);
  /**
   * Called immediately before the main scatter plot performs a color (visual)
   * render. Set up render state, lights, etc here.
   */
  onRender(renderContext: RenderContext);
  /**
   * Called when the projector updates application state (projection style,
   * etc). Generally followed by a render.
   */
  onUpdate();
  /**
   * Called when the canvas size changes.
   */
  onResize(newWidth: number, newHeight: number);
}
