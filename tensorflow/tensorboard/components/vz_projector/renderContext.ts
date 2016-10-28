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
 * LabelRenderParams describes the set of points that should have labels
 * rendered next to them.
 */
export class LabelRenderParams {
  pointIndices: Float32Array;
  scaleFactors: Float32Array;
  useSceneOpacityFlags: Int8Array;  // booleans
  defaultFontSize: number;
  fillColor: number;
  strokeColor: number;

  constructor(
      pointIndices: Float32Array, scaleFactors: Float32Array,
      useSceneOpacityFlags: Int8Array, defaultFontSize: number,
      fillColor: number, strokeColor: number) {
    this.pointIndices = pointIndices;
    this.scaleFactors = scaleFactors;
    this.useSceneOpacityFlags = useSceneOpacityFlags;
    this.defaultFontSize = defaultFontSize;
    this.fillColor = fillColor;
    this.strokeColor = strokeColor;
  }
}

/**
 * RenderContext contains all of the state required to color and render the data
 * set. ScatterPlot passes this to every attached visualizer as part of the
 * render callback.
 * TODO(nicholsonc): This should only contain the data that's changed between
 * each frame. Data like colors / scale factors / labels should be recomputed
 * only when they change.
 */
export class RenderContext {
  camera: THREE.Camera;
  cameraTarget: THREE.Vector3;
  screenWidth: number;
  screenHeight: number;
  nearestCameraSpacePointZ: number;
  farthestCameraSpacePointZ: number;
  pointColors: Float32Array;
  pointScaleFactors: Float32Array;
  labelAccessor: (index: number) => string;
  labels: LabelRenderParams;
  traceColors: {[trace: number]: Float32Array};

  constructor(
      camera: THREE.Camera, cameraTarget: THREE.Vector3, screenWidth: number,
      screenHeight: number, nearestCameraSpacePointZ: number,
      farthestCameraSpacePointZ: number, pointColors: Float32Array,
      pointScaleFactors: Float32Array, labelAccessor: (index: number) => string,
      labels: LabelRenderParams, traceColors: {[trace: number]: Float32Array}) {
    this.camera = camera;
    this.cameraTarget = cameraTarget;
    this.screenWidth = screenWidth;
    this.screenHeight = screenHeight;
    this.nearestCameraSpacePointZ = nearestCameraSpacePointZ;
    this.farthestCameraSpacePointZ = farthestCameraSpacePointZ;
    this.pointColors = pointColors;
    this.pointScaleFactors = pointScaleFactors;
    this.labelAccessor = labelAccessor;
    this.labels = labels;
    this.traceColors = traceColors;
  }
}
