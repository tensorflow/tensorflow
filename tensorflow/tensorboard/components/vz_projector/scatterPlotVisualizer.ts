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

/**
 * ScatterPlotVisualizer is an interface used by ScatterPlotContainer
 * to manage and aggregate any number of concurrent visualization behaviors.
 * To add a new visualization to the 3D scatter plot, create a new class that
 * implements this interface and attach it to the ScatterPlotContainer.
 */
export interface ScatterPlotVisualizer {
  /** Called to initialize the visualizer with the primary scene. */
  setScene(scene: THREE.Scene);
  /**
   * Called when the main scatter plot tears down the visualizer. Remove all
   * objects from the scene, and dispose any heavy resources.
   */
  dispose();
  /**
   * Called when the positions of the scatter plot points have changed.
   */
  onPointPositionsChanged(newWorldSpacePointPositions: Float32Array);
  /**
   * Called immediately before the main scatter plot performs a picking
   * (selection) render. Set up render state for any geometry to use picking IDs
   * instead of visual colors.
   */
  onPickingRender(renderContext: RenderContext);
  /**
   * Called immediately before the main scatter plot performs a color (visual)
   * render. Set up render state, lights, etc here.
   */
  onRender(renderContext: RenderContext);
  /**
   * Called when the canvas size changes.
   */
  onResize(newWidth: number, newHeight: number);
}
