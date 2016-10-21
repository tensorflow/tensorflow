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
import {NearestEntry} from './knn';
import {LabelRenderParams} from './renderContext';

const LABEL_FONT_SIZE = 10;
const LABEL_SCALE_DEFAULT = 1.0;
const LABEL_SCALE_LARGE = 1.7;
const LABEL_FILL_COLOR = 0x000000;
const LABEL_STROKE_COLOR = 0xFFFFFF;

const POINT_COLOR_UNSELECTED = 0xE3E3E3;
const POINT_COLOR_NO_SELECTION = 0x7575D9;
const POINT_COLOR_SELECTED = 0xFA6666;
const POINT_COLOR_HOVER = 0x760B4F;

const POINT_SCALE_DEFAULT = 1.0;
const POINT_SCALE_SELECTED = 1.2;
const POINT_SCALE_NEIGHBOR = 1.2;
const POINT_SCALE_HOVER = 1.2;

const LABELS_3D_COLOR_UNSELECTED = 0xFFFFFF;
const LABELS_3D_COLOR_NO_SELECTION = 0xFFFFFF;

/**
 * Interprets projector events and assembes the arrays and commands necessary
 * to use the ScatterPlot to render the current projected data set.
 */
export class ProjectorScatterPlotAdapter {
  generateVisibleLabelRenderParams(
      ds: DataSet, selectedPointIndices: number[],
      neighborsOfFirstPoint: NearestEntry[],
      hoverPointIndex: number): LabelRenderParams {
    if (ds == null) {
      return null;
    }

    const n = selectedPointIndices.length + neighborsOfFirstPoint.length +
        ((hoverPointIndex != null) ? 1 : 0);

    const visibleLabels = new Uint32Array(n);
    const scale = new Float32Array(n);
    const opacityFlags = new Int8Array(n);

    scale.fill(LABEL_SCALE_DEFAULT);
    opacityFlags.fill(1);

    let dst = 0;

    if (hoverPointIndex != null) {
      visibleLabels[dst] = hoverPointIndex;
      scale[dst] = LABEL_SCALE_LARGE;
      opacityFlags[dst] = 0;
      ++dst;
    }

    // Selected points
    {
      const n = selectedPointIndices.length;
      for (let i = 0; i < n; ++i) {
        visibleLabels[dst] = selectedPointIndices[i];
        scale[dst] = LABEL_SCALE_LARGE;
        opacityFlags[dst] = (n === 1) ? 0 : 1;
        ++dst;
      }
    }

    // Neighbors
    {
      const n = neighborsOfFirstPoint.length;
      for (let i = 0; i < n; ++i) {
        visibleLabels[dst++] = neighborsOfFirstPoint[i].index;
      }
    }

    return new LabelRenderParams(
        visibleLabels, scale, opacityFlags, LABEL_FONT_SIZE, LABEL_FILL_COLOR,
        LABEL_STROKE_COLOR);
  }

  generatePointScaleFactorArray(
      ds: DataSet, selectedPointIndices: number[],
      neighborsOfFirstPoint: NearestEntry[],
      hoverPointIndex: number): Float32Array {
    if (ds == null) {
      return new Float32Array(0);
    }

    const scale = new Float32Array(ds.points.length);
    scale.fill(POINT_SCALE_DEFAULT);

    // Scale up all selected points.
    {
      const n = selectedPointIndices.length;
      for (let i = 0; i < n; ++i) {
        const p = selectedPointIndices[i];
        scale[p] = POINT_SCALE_SELECTED;
      }
    }

    // Scale up the neighbor points.
    {
      const n = neighborsOfFirstPoint.length;
      for (let i = 0; i < n; ++i) {
        const p = neighborsOfFirstPoint[i].index;
        scale[p] = POINT_SCALE_NEIGHBOR;
      }
    }

    // Scale up the hover point.
    if (hoverPointIndex != null) {
      scale[hoverPointIndex] = POINT_SCALE_HOVER;
    }

    return scale;
  }

  generatePointColorArray(
      ds: DataSet, legendPointColorer: (index: number) => string,
      selectedPointIndices: number[], neighborsOfFirstPoint: NearestEntry[],
      hoverPointIndex: number, label3dMode: boolean): Float32Array {
    if (ds == null) {
      return new Float32Array(0);
    }

    const colors = new Float32Array(ds.points.length * 3);

    let unselectedColor = POINT_COLOR_UNSELECTED;
    let noSelectionColor = POINT_COLOR_NO_SELECTION;

    if (label3dMode) {
      unselectedColor = LABELS_3D_COLOR_UNSELECTED;
      noSelectionColor = LABELS_3D_COLOR_NO_SELECTION;
    }

    // Give all points the unselected color.
    {
      const n = ds.points.length;
      let dst = 0;
      if (selectedPointIndices.length > 0) {
        const c = new THREE.Color(unselectedColor);
        for (let i = 0; i < n; ++i) {
          colors[dst++] = c.r;
          colors[dst++] = c.g;
          colors[dst++] = c.b;
        }
      } else {
        if (legendPointColorer != null) {
          for (let i = 0; i < n; ++i) {
            const c = new THREE.Color(legendPointColorer(i));
            colors[dst++] = c.r;
            colors[dst++] = c.g;
            colors[dst++] = c.b;
          }
        } else {
          const c = new THREE.Color(noSelectionColor);
          for (let i = 0; i < n; ++i) {
            colors[dst++] = c.r;
            colors[dst++] = c.g;
            colors[dst++] = c.b;
          }
        }
      }
    }

    // Color the selected points.
    {
      const n = selectedPointIndices.length;
      const c = new THREE.Color(POINT_COLOR_SELECTED);
      for (let i = 0; i < n; ++i) {
        let dst = selectedPointIndices[i] * 3;
        colors[dst++] = c.r;
        colors[dst++] = c.g;
        colors[dst++] = c.b;
      }
    }

    // Color the neighbors.
    {
      const n = neighborsOfFirstPoint.length;
      const c = new THREE.Color(POINT_COLOR_SELECTED);
      for (let i = 0; i < n; ++i) {
        let dst = neighborsOfFirstPoint[i].index * 3;
        colors[dst++] = c.r;
        colors[dst++] = c.g;
        colors[dst++] = c.b;
      }
    }

    // Color the hover point.
    if (hoverPointIndex != null) {
      const c = new THREE.Color(POINT_COLOR_HOVER);
      let dst = hoverPointIndex * 3;
      colors[dst++] = c.r;
      colors[dst++] = c.g;
      colors[dst++] = c.b;
    }

    return colors;
  }
}
