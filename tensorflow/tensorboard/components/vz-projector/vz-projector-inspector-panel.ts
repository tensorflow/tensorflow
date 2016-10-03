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

import {DistanceFunction} from './data';
import * as vector from './vector';
import {ProjectorInput} from './vz-projector-input';
import {Projector} from './vz-projector';
import * as knn from './knn';
import {MetadataResult} from './data-loader';

// tslint:disable-next-line:no-unused-variable
import {PolymerElement, PolymerHTMLElement} from './vz-projector-util';

/** Color scale for nearest neighbors. */
const NN_COLOR_SCALE =
    d3.scale.linear<string>()
        .domain([1, 0.7, 0.4])
        .range(['hsl(285, 80%, 40%)', 'hsl(0, 80%, 65%)', 'hsl(40, 70%, 60%)'])
        .clamp(true);

// tslint:disable-next-line
export let PolymerClass = PolymerElement({
  is: 'vz-projector-inspector-panel',
  properties: {
    selectedMetadataField: String,
    metadataFields: Array
  }
});

export class InspectorPanel extends PolymerClass {
  distFunc: DistanceFunction;
  numNN: number;

  private selectedMetadataField: string;
  private metadataFields: string[];
  private dom: d3.Selection<HTMLElement>;
  private projector: Projector;
  private selectedPointIndex: number;

  initialize(projector: Projector) {
    this.projector = projector;
    this.dom = d3.select(this);
    // Dynamically creating elements inside .nn-list.
    this.scopeSubtree(this, true);
    this.setupUI();
  }

  /** Updates the nearest neighbors list in the inspector. */
  updateInspectorPane(indices: number[],
      neighbors: knn.NearestEntry[]) {
    if (neighbors.length > 0 || indices.length === 0) {
      this.selectedPointIndex = indices[0];
      this.updateMetadata();
    } else {
      this.selectedPointIndex = null;
    }
    this.updateIsolateButton(indices.length);
    this.updateNeighborsList(neighbors);
  }

  metadataChanged(result: MetadataResult) {
    let labelIndex = -1;
    this.metadataFields = result.stats.map((stats, i) => {
      if (!stats.isNumeric && labelIndex === -1) {
        labelIndex = i;
      }
      return stats.name;
    });
    labelIndex = Math.max(0, labelIndex);
    // Make the default label the first non-numeric column.
    this.selectedMetadataField = result.stats[labelIndex].name;
  }

  private updateNeighborsList(neighbors: knn.NearestEntry[]) {
    let nnlist = this.dom.select('.nn-list');
    nnlist.html('');
    this.dom.select('.nn').style('display', neighbors.length ? null : 'none');

    if (neighbors.length === 0) {
      return;
    }

    let minDist = neighbors.length > 0 ? neighbors[0].dist : 0;
    let n = nnlist.selectAll('.neighbor')
                .data(neighbors)
                .enter()
                .append('div')
                .attr('class', 'neighbor')
                .append('a')
                .attr('class', 'neighbor-link');

    n.append('span')
        .attr('class', 'label')
        .style('color', d => dist2color(this.distFunc, d.dist, minDist))
        .text(d => {
          let point = this.projector.currentDataSet.points[d.index];
          return point.metadata[this.selectedMetadataField];
        });

    n.append('span').attr('class', 'value').text(d => d.dist.toFixed(2));

    let bar = n.append('div').attr('class', 'bar');

    bar.append('div')
        .attr('class', 'fill')
        .style('border-top-color', d => {
          return dist2color(this.distFunc, d.dist, minDist);
        })
        .style('width', d =>
            normalizeDist(this.distFunc, d.dist, minDist) * 100 + '%');

    bar.selectAll('.tick')
        .data(d3.range(1, 4))
        .enter()
        .append('div')
        .attr('class', 'tick')
        .style('left', d => d * 100 / 4 + '%');

    n.on('click', d => {
      this.projector.notifySelectionChanged([d.index]);
    });
  }

  private updateIsolateButton(numPoints: number) {
    let isolateButton = this.dom.select('.set-filter');
    let clearButton = this.dom.select('button.clear-selection');
    if (numPoints > 1) {
      isolateButton.text(`Isolate ${numPoints} points`).style('display', null);
      clearButton.style('display', null);
    } else {
      isolateButton.style('display', 'none');
      clearButton.style('display', 'none');
    }
  }

  /** Updates the displayed metadata for the selected point. */
  private updateMetadata() {
    let metadataContainerElement = this.dom.select('.metadata');
    metadataContainerElement.selectAll('*').remove();
    let point = null;
    if (this.projector.currentDataSet != null &&
        this.selectedPointIndex != null) {
      point = this.projector.currentDataSet.points[this.selectedPointIndex];
    }
    this.dom.select('.metadata-container')
        .style('display', point != null ? '' : 'none');

    if (point == null) {
      return;
    }

    for (let metadataKey in point.metadata) {
      if (!point.metadata.hasOwnProperty(metadataKey)) {
        continue;
      }
      let rowElement = document.createElement('div');
      rowElement.className = 'metadata-row';

      let keyElement = document.createElement('div');
      keyElement.className = 'metadata-key';
      keyElement.textContent = metadataKey;

      let valueElement = document.createElement('div');
      valueElement.className = 'metadata-value';
      valueElement.textContent = '' + point.metadata[metadataKey];

      rowElement.appendChild(keyElement);
      rowElement.appendChild(valueElement);

      metadataContainerElement.append(function() {
        return this.appendChild(rowElement);
      });
    }
  }

  private setupUI() {
    this.distFunc = vector.cosDist;
    let eucDist = this.dom.select('.distance a.euclidean');
    eucDist.on('click', () => {
      this.dom.selectAll('.distance a').classed('selected', false);
      eucDist.classed('selected', true);
      this.distFunc = vector.dist;
      let neighbors = this.projector.currentDataSet.findNeighbors(
          this.selectedPointIndex, this.distFunc, this.numNN);
      this.updateNeighborsList(neighbors);
    });

    let cosDist = this.dom.select('.distance a.cosine');
    cosDist.on('click', () => {
      this.dom.selectAll('.distance a').classed('selected', false);
      cosDist.classed('selected', true);
      this.distFunc = vector.cosDist;
      let neighbors = this.projector.currentDataSet.findNeighbors(
          this.selectedPointIndex, this.distFunc, this.numNN);
      this.updateNeighborsList(neighbors);
    });

    let searchBox = this.querySelector('#search-box') as ProjectorInput;

    // Called whenever the search text input changes.
    let updateInput = (value: string, inRegexMode: boolean) => {
      if (value == null || value.trim() === '') {
        searchBox.message = '';
        this.projector.notifySelectionChanged([]);
        return;
      }
      let indices = this.projector.currentDataSet.query(value, inRegexMode,
          this.selectedMetadataField);
      if (indices.length === 0) {
        searchBox.message = '0 matches.';
      } else {
        searchBox.message = `${indices.length} matches.`;
      }
      this.projector.notifySelectionChanged(indices);
    };
    searchBox.onInputChanged((value, inRegexMode) => {
      updateInput(value, inRegexMode);
    });

    // Nearest neighbors controls.
    let numNNInput = this.dom.select('.num-nn input');
    let updateNumNN = () => {
      this.numNN = +numNNInput.property('value');
      this.dom.select('.num-nn span').text(this.numNN);
    };
    numNNInput.on('input', updateNumNN);
    updateNumNN();

    // Filtering dataset.
    this.dom.select('.set-filter').on('click', () => {
      this.projector.filterDataset();
      this.dom.select('.reset-filter').style('display', null);
      this.updateIsolateButton(0);
    });

    this.dom.select('.reset-filter').on('click', () => {
      this.projector.resetFilterDataset();
      this.dom.select('.reset-filter').style('display', 'none');
    });

    this.dom.select('.clear-selection').on('click', () => {
      this.projector.clearSelection();
    });
  }
}

/**
 * Normalizes the distance so it can be visually encoded with color.
 * The normalization depends on the distance metric (cosine vs euclidean).
 */
function normalizeDist(distFunc: DistanceFunction,
    d: number, minDist: number): number {
  return distFunc === vector.dist ? minDist / d : 1 - d;
}

/** Normalizes and encodes the provided distance with color. */
function dist2color(distFunc: DistanceFunction,
    d: number, minDist: number): string {
  return NN_COLOR_SCALE(normalizeDist(distFunc, d, minDist));
}

document.registerElement(InspectorPanel.prototype.is, InspectorPanel);
