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

export interface Point3D {
  /** Original x coordinate. */
  x: number;
  /** Original y coordinate. */
  y: number;
  /** Original z coordinate. */
  z: number;
}
;

/** The spacial data of points and lines that will be shown in the projector. */
export interface DataSet {
  points: DataPoint[];
  traces: DataTrace[];
}

/**
 * Points in 3D space that will be used in the projector. If the projector is
 * in 2D mode, the Z coordinate of the point will be 0.
 */
export interface DataPoint {
  projectedPoint: Point3D;
  /** index of the trace, used for highlighting on click */
  traceIndex?: number;
  /** index in the original data source */
  dataSourceIndex: number;
}

/** A single collection of points which make up a trace through space. */
export interface DataTrace {
  /** Indices into the DataPoints array in the Data object. */
  pointIndices: number[];
}

export type OnHoverListener = (index: number) => void;
export type OnSelectionListener = (indexes: number[]) => void;

/** Supported modes of interaction. */
export enum Mode {
  SELECT,
  SEARCH,
  HOVER
}

export interface Scatter {
  /** Sets the data for the scatter plot. */
  setDataSet(dataSet: DataSet, spriteImage?: HTMLImageElement): void;
  /** Called with each data point in order to get its color. */
  setColorAccessor(colorAccessor: ((index: number) => string)): void;
  /** Called with each data point in order to get its label. */
  setLabelAccessor(labelAccessor: ((index: number) => string)): void;
  /** Called with each data point in order to get its x coordinate. */
  setXAccessor(xAccessor: ((index: number) => number)): void;
  /** Called with each data point in order to get its y coordinate. */
  setYAccessor(yAccessor: ((index: number) => number)): void;
  /** Called with each data point in order to get its z coordinate. */
  setZAccessor(zAccessor: ((index: number) => number)): void;
  /** Sets the interaction mode (search, select or hover). */
  setMode(mode: Mode): void;
  /** Returns the interaction mode. */
  getMode(): Mode;
  /** Resets the zoom level to 1.*/
  resetZoom(): void;
  /**
   * Increases/decreases the zoom level.
   *
   * @param multiplier New zoom level = old zoom level * multiplier.
   */
  zoomStep(multiplier: number): void;
  /**
   * Highlights the provided points.
   *
   * @param pointIndexes List of point indexes to highlight. If null,
   *   un-highlights all the points.
   * @param stroke The stroke color used to highlight the point.
   * @param favorLabels Whether to favor plotting the labels of the
   *   highlighted point. Default is false for all points.
   */
  highlightPoints(
      pointIndexes: number[], highlightStroke?: (index: number) => string,
      favorLabels?: (index: number) => boolean): void;
  /** Whether to show labels or not. */
  showLabels(show: boolean): void;
  /** Toggle between day and night modes. */
  setDayNightMode(isNight: boolean): void;
  /** Show/hide tick labels. */
  showTickLabels(show: boolean): void;
  /** Whether to show axes or not. */
  showAxes(show: boolean): void;
  /** Sets the axis labels. */
  setAxisLabels(xLabel: string, yLabel: string): void;
  /**
   * Recreates the scene (demolishes all datastructures, etc.)
   */
  recreateScene(): void;
  /**
   * Redraws the data. Should be called anytime the accessor method
   * for x and y coordinates changes, which means a new projection
   * exists and the scatter plot should repaint the points.
   */
  update(): void;
  /**
   * Should be called to notify the scatter plot that the container
   * was resized and it should resize and redraw itself.
   */
  resize(): void;
  /** Registers a listener that will be called when selects some points. */
  onSelection(listener: OnSelectionListener): void;
  /**
   * Registers a listener that will be called when the user hovers over
   * a point.
   */
  onHover(listener: OnHoverListener): void;
  /**
   * Should emulate the same behavior as if the user clicked on the point.
   * This is used to trigger a click from an external event, such as
   * a search query.
   */
  clickOnPoint(pointIndex: number): void;
}
