/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http:www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

/**
 * RenderContext contains all of the state required to color and render the data
 * set. ScatterPlot passes this to every attached visualizer as part of the
 * render callback.
 */
export class RenderContext {
  camera: THREE.PerspectiveCamera;
  cameraTarget: THREE.Vector3;
  screenWidth: number;
  screenHeight: number;
  nearestCameraSpacePointZ: number;
  farthestCameraSpacePointZ: number;
  colorAccessor: (index: number) => string;
  labeledPoints: number[];
  labelAccessor: (index: number) => string;
  highlightedPoints: number[];
  highlightStroke: (index: number) => string;

  constructor(
      camera: THREE.PerspectiveCamera, cameraTarget: THREE.Vector3,
      screenWidth: number, screenHeight: number,
      nearestCameraSpacePointZ: number, farthestCameraSpacePointZ: number,
      colorAccessor: (index: number) => string, labeledPoints: number[],
      labelAccessor: (index: number) => string, highlightedPoints: number[],
      highlightStroke: (index: number) => string) {
    this.camera = camera;
    this.cameraTarget = cameraTarget;
    this.screenWidth = screenWidth;
    this.screenHeight = screenHeight;
    this.nearestCameraSpacePointZ = nearestCameraSpacePointZ;
    this.farthestCameraSpacePointZ = farthestCameraSpacePointZ;
    this.colorAccessor = colorAccessor;
    this.labeledPoints = labeledPoints;
    this.labelAccessor = labelAccessor;
    this.highlightedPoints = highlightedPoints;
    this.highlightStroke = highlightStroke;
  }
}
