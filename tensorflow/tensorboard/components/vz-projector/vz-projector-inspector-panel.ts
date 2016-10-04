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

/** Limit the number of search results we show to the user. */
const LIMIT_RESULTS = 100;

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
  private searchBox: ProjectorInput;

  private resetFilterButton: d3.Selection<HTMLElement>;
  private setFilterButton: d3.Selection<HTMLElement>;
  private clearSelectionButton: d3.Selection<HTMLElement>;
  private limitMessage: d3.Selection<HTMLElement>;

  ready() {
    this.dom = d3.select(this);
    this.resetFilterButton = this.dom.select('.reset-filter');
    this.setFilterButton = this.dom.select('.set-filter');
    this.clearSelectionButton = this.dom.select('.clear-selection');
    this.limitMessage = this.dom.select('.limit-msg');
    this.searchBox = this.querySelector('#search-box') as ProjectorInput;
    // https://www.polymer-project.org/1.0/docs/devguide/styling#scope-subtree
    this.scopeSubtree(this, true);
  }

  initialize(projector: Projector) {
    this.projector = projector;
    this.setupUI();
  }

  /** Updates the nearest neighbors list in the inspector. */
  updateInspectorPane(indices: number[],
      neighbors: knn.NearestEntry[]) {
    if (neighbors.length > 0) {
      this.selectedPointIndex = indices[0];
    } else {
      this.selectedPointIndex = null;
    }
    this.updateMetadata();
    this.updateIsolateButton(indices.length);
    this.updateNeighborsList(neighbors);
    if (neighbors.length === 0) {
      this.updateSearchResults(indices);
    } else {
      this.updateSearchResults([]);
    }
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

  datasetChanged() {
    this.resetFilterButton.attr('disabled', true);
  }

  private updateSearchResults(indices: number[]) {
    let container = this.dom.select('.matches-list');
    container.style('display', indices.length ? null : 'none');
    let list = container.select('.list');
    list.html('');
    if (indices.length === 0) {
      return;
    }
    this.limitMessage.style(
        'display', indices.length <= LIMIT_RESULTS ? 'none' : null);
    indices = indices.slice(0, LIMIT_RESULTS);
    let rows = list.selectAll('.row')
      .data(indices)
      .enter()
      .append('div').attr('class', 'row');
    rows.append('a')
      .attr('class', 'label')
      .attr('title', index => this.getLabelFromIndex(index))
      .text(index => this.getLabelFromIndex(index));
    rows.on('mouseenter', index => {
      this.projector.notifyHoverOverPoint(index);
    });
    rows.on('mouseleave', () => {
      this.projector.notifyHoverOverPoint(null);
    });
    rows.on('click', index => {
      this.projector.notifySelectionChanged([index]);
    });
  }

  private getLabelFromIndex(pointIndex: number): string {
    let point = this.projector.currentDataSet.points[pointIndex];
    return point.metadata[this.selectedMetadataField].toString();
  }

  private updateNeighborsList(neighbors: knn.NearestEntry[]) {
    let nnlist = this.dom.select('.nn-list');
    nnlist.html('');
    this.dom.select('.nn').style('display', neighbors.length ? null : 'none');

    if (neighbors.length === 0) {
      return;
    }

    this.searchBox.message = '';
    let minDist = neighbors.length > 0 ? neighbors[0].dist : 0;
    let n = nnlist.selectAll('.neighbor')
      .data(neighbors)
      .enter()
      .append('div')
      .attr('class', 'neighbor')
      .append('a')
      .attr('class', 'neighbor-link')
      .attr('title', d => this.getLabelFromIndex(d.index));


    let labelValue = n.append('div').attr('class', 'label-and-value');
    labelValue.append('div')
        .attr('class', 'label')
        .style('color', d => dist2color(this.distFunc, d.dist, minDist))
        .text(d => this.getLabelFromIndex(d.index));

    labelValue.append('div')
      .attr('class', 'value')
      .text(d => d.dist.toFixed(3));

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
    n.on('mouseenter', d => {
      this.projector.notifyHoverOverPoint(d.index);
    });
    n.on('mouseleave', () => {
      this.projector.notifyHoverOverPoint(null);
    });
    n.on('click', d => {
      this.projector.notifySelectionChanged([d.index]);
    });
  }

  private updateIsolateButton(numPoints: number) {
    if (numPoints > 1) {
      this.setFilterButton.text(`Isolate ${numPoints} points`)
          .attr('disabled', null);
      this.clearSelectionButton.attr('disabled', null);
    } else {
      this.setFilterButton.attr('disabled', true);
      this.clearSelectionButton.attr('disabled', true);
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

    // Called whenever the search text input changes.
    let updateInput = (value: string, inRegexMode: boolean) => {
      if (value == null || value.trim() === '') {
        this.searchBox.message = '';
        this.projector.notifySelectionChanged([]);
        return;
      }
      let indices = this.projector.currentDataSet.query(value, inRegexMode,
          this.selectedMetadataField);
      if (indices.length === 0) {
        this.searchBox.message = '0 matches.';
      } else {
        this.searchBox.message = `${indices.length} matches.`;
      }
      this.projector.notifySelectionChanged(indices);
    };
    this.searchBox.onInputChanged((value, inRegexMode) => {
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
    this.setFilterButton.on('click', () => {
      this.projector.filterDataset();
      this.resetFilterButton.attr('disabled', null);
      this.updateIsolateButton(0);
    });

    this.resetFilterButton.on('click', () => {
      this.projector.resetFilterDataset();
      this.resetFilterButton.attr('disabled', true);
    });

    this.clearSelectionButton.on('click', () => {
      this.projector.clearSelection();
    });
    this.resetFilterButton.attr('disabled', true);
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
