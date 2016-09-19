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

import {DataSet} from './scatterPlot';

/**
 * ScatterPlotWebGLVisualizer is an interface used by ScatterPlotWebGLContainer
 * to manage and aggregate any number of concurrent visualization behaviors.
 * To add a new visualization to the 3D scatter plot, create a new class that
 * implements this interface and attach it to the ScatterPlotWebGLContainer.
 */
export interface ScatterPlotWebGLVisualizer {
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
   * Called immediately before the main scatter plot performs a picking
   * (selection) render. Set up render state for any geometry to use picking IDs
   * instead of visual colors.
   */
  onPickingRender(camera: THREE.Camera, cameraTarget: THREE.Vector3);
  /**
   * Called immediately before the main scatter plot performs a color (visual)
   * render. Set up render state, lights, etc here.
   */
  onRender(
      camera: THREE.Camera, cameraTarget: THREE.Vector3, screenWidth: number,
      screenHeight: number, colorAccessor: (index: number) => string,
      labeledPoints: number[], labelAccessor: (index: number) => string,
      highlightedPoints: number[], highlightStroke: (index: number) => string);
  /**
   * Called when the projector updates application state (day / night mode,
   * projection style, etc). Generally followed by a render.
   */
  onUpdate();
  /**
   * Called when the canvas size changes.
   */
  onResize(newWidth: number, newHeight: number);
  /**
   * Called when the application toggles between day and night mode.
   */
  onSetDayNightMode(isNight: boolean);
}
