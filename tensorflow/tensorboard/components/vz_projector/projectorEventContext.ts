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

import {DistanceFunction, Projection} from './data';
import {NearestEntry} from './knn';

export type HoverListener = (index: number) => void;
export type SelectionChangedListener =
    (selectedPointIndices: number[], neighborsOfFirstPoint: NearestEntry[]) =>
        void;
export type ProjectionChangedListener = (projection: Projection) => void;
export type DistanceMetricChangedListener =
    (distanceMetric: DistanceFunction) => void;
export interface ProjectorEventContext {
  /** Register a callback to be invoked when the mouse hovers over a point. */
  registerHoverListener(listener: HoverListener);
  /** Notify the hover system that a point is under the mouse. */
  notifyHoverOverPoint(pointIndex: number);
  /** Registers a callback to be invoked when the selection changes. */
  registerSelectionChangedListener(listener: SelectionChangedListener);
  /**
   * Notify the selection system that a client has changed the selected point
   * set.
   */
  notifySelectionChanged(newSelectedPointIndices: number[]);
  /** Registers a callback to be invoked when the projection changes. */
  registerProjectionChangedListener(listener: ProjectionChangedListener);
  /** Notify listeners that a reprojection occurred. */
  notifyProjectionChanged(projection: Projection);
  registerDistanceMetricChangedListener(listener:
                                            DistanceMetricChangedListener);
  notifyDistanceMetricChanged(distMetric: DistanceFunction);
}
