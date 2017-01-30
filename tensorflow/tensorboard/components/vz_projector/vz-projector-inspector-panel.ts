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

import {DistanceFunction, SpriteAndMetadataInfo, State} from './data';
import * as knn from './knn';
import {ProjectorEventContext} from './projectorEventContext';
import * as adapter from './projectorScatterPlotAdapter';
import * as vector from './vector';
import {Projector} from './vz-projector';
import {ProjectorInput} from './vz-projector-input';
// tslint:disable-next-line:no-unused-variable
import {PolymerElement, PolymerHTMLElement} from './vz-projector-util';

/** Limit the number of search results we show to the user. */
const LIMIT_RESULTS = 100;

// tslint:disable-next-line
export let PolymerClass = PolymerElement({
  is: 'vz-projector-inspector-panel',
  properties: {selectedMetadataField: String, metadataFields: Array}
});

export class InspectorPanel extends PolymerClass {
  distFunc: DistanceFunction;
  numNN: number;

  private projectorEventContext: ProjectorEventContext;

  private selectedMetadataField: string;
  private metadataFields: string[];
  private dom: d3.Selection<HTMLElement>;
  private projector: Projector;
  private selectedPointIndices: number[];
  private neighborsOfFirstPoint: knn.NearestEntry[];
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

  initialize(
      projector: Projector, projectorEventContext: ProjectorEventContext) {
    this.projector = projector;
    this.projectorEventContext = projectorEventContext;
    this.setupUI(projector);
    projectorEventContext.registerSelectionChangedListener(
        (selection, neighbors) =>
            this.updateInspectorPane(selection, neighbors));
  }

  /** Updates the nearest neighbors list in the inspector. */
  private updateInspectorPane(
      indices: number[], neighbors: knn.NearestEntry[]) {
    this.neighborsOfFirstPoint = neighbors;
    this.selectedPointIndices = indices;

    this.updateFilterButtons(indices.length + neighbors.length);
    this.updateNeighborsList(neighbors);
    if (neighbors.length === 0) {
      this.updateSearchResults(indices);
    } else {
      this.updateSearchResults([]);
    }
  }

  private enableResetFilterButton(enabled: boolean) {
    this.resetFilterButton.attr('disabled', enabled ? null : true);
  }

  restoreUIFromBookmark(bookmark: State) {
    this.enableResetFilterButton(bookmark.filteredPoints != null);
  }

  metadataChanged(spriteAndMetadata: SpriteAndMetadataInfo) {
    let labelIndex = -1;
    this.metadataFields = spriteAndMetadata.stats.map((stats, i) => {
      if (!stats.isNumeric && labelIndex === -1) {
        labelIndex = i;
      }
      return stats.name;
    });
    labelIndex = Math.max(0, labelIndex);
    // Make the default label the first non-numeric column.
    this.selectedMetadataField = spriteAndMetadata.stats[labelIndex].name;
  }

  datasetChanged() {
    this.enableResetFilterButton(false);
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
    let rows = list.selectAll('.row').data(indices).enter().append('div').attr(
        'class', 'row');
    rows.append('a')
        .attr('class', 'label')
        .attr('title', index => this.getLabelFromIndex(index))
        .text(index => this.getLabelFromIndex(index));
    rows.on('mouseenter', index => {
      this.projectorEventContext.notifyHoverOverPoint(index);
    });
    rows.on('mouseleave', () => {
      this.projectorEventContext.notifyHoverOverPoint(null);
    });
    rows.on('click', index => {
      this.projectorEventContext.notifySelectionChanged([index]);
    });
  }

  private getLabelFromIndex(pointIndex: number): string {
    let point = this.projector.dataSet.points[pointIndex];
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
        .style('color', d => adapter.dist2color(this.distFunc, d.dist, minDist))
        .text(d => this.getLabelFromIndex(d.index));

    labelValue.append('div')
        .attr('class', 'value')
        .text(d => d.dist.toFixed(3));

    let bar = n.append('div').attr('class', 'bar');

    bar.append('div')
        .attr('class', 'fill')
        .style(
            'border-top-color',
            d => {
              return adapter.dist2color(this.distFunc, d.dist, minDist);
            })
        .style(
            'width',
            d => adapter.normalizeDist(this.distFunc, d.dist, minDist) * 100 +
                '%');

    bar.selectAll('.tick')
        .data(d3.range(1, 4))
        .enter()
        .append('div')
        .attr('class', 'tick')
        .style('left', d => d * 100 / 4 + '%');
    n.on('mouseenter', d => {
      this.projectorEventContext.notifyHoverOverPoint(d.index);
    });
    n.on('mouseleave', () => {
      this.projectorEventContext.notifyHoverOverPoint(null);
    });
    n.on('click', d => {
      this.projectorEventContext.notifySelectionChanged([d.index]);
    });
  }

  private updateFilterButtons(numPoints: number) {
    if (numPoints > 1) {
      this.setFilterButton.text(`Isolate ${numPoints} points`)
          .attr('disabled', null);
      this.clearSelectionButton.attr('disabled', null);
    } else {
      this.setFilterButton.attr('disabled', true);
      this.clearSelectionButton.attr('disabled', true);
    }
  }

  private setupUI(projector: Projector) {
    this.distFunc = vector.cosDist;
    let eucDist = this.dom.select('.distance a.euclidean');
    eucDist.on('click', () => {
      this.dom.selectAll('.distance a').classed('selected', false);
      eucDist.classed('selected', true);
      this.distFunc = vector.dist;
      this.projectorEventContext.notifyDistanceMetricChanged(this.distFunc);
      let neighbors = projector.dataSet.findNeighbors(
          this.selectedPointIndices[0], this.distFunc, this.numNN);
      this.updateNeighborsList(neighbors);
    });

    let cosDist = this.dom.select('.distance a.cosine');
    cosDist.on('click', () => {
      this.dom.selectAll('.distance a').classed('selected', false);
      cosDist.classed('selected', true);
      this.distFunc = vector.cosDist;
      this.projectorEventContext.notifyDistanceMetricChanged(this.distFunc);
      let neighbors = projector.dataSet.findNeighbors(
          this.selectedPointIndices[0], this.distFunc, this.numNN);
      this.updateNeighborsList(neighbors);
    });

    // Called whenever the search text input changes.
    let updateInput = (value: string, inRegexMode: boolean) => {
      if (value == null || value.trim() === '') {
        this.searchBox.message = '';
        this.projectorEventContext.notifySelectionChanged([]);
        return;
      }
      let indices = projector.dataSet.query(
          value, inRegexMode, this.selectedMetadataField);
      if (indices.length === 0) {
        this.searchBox.message = '0 matches.';
      } else {
        this.searchBox.message = `${indices.length} matches.`;
      }
      this.projectorEventContext.notifySelectionChanged(indices);
    };
    this.searchBox.registerInputChangedListener((value, inRegexMode) => {
      updateInput(value, inRegexMode);
    });

    // Nearest neighbors controls.
    let numNNInput = this.$$('#nn-slider') as HTMLInputElement;
    let updateNumNN = () => {
      this.numNN = +numNNInput.value;
      this.dom.select('.num-nn .nn-count').text(this.numNN);
      if (this.selectedPointIndices != null) {
        this.projectorEventContext.notifySelectionChanged(
            [this.selectedPointIndices[0]]);
      }
    };
    numNNInput.addEventListener('change', updateNumNN);
    updateNumNN();

    // Filtering dataset.
    this.setFilterButton.on('click', () => {
      const indices = this.selectedPointIndices.concat(
          this.neighborsOfFirstPoint.map(n => n.index));
      projector.filterDataset(indices);
      this.enableResetFilterButton(true);
      this.updateFilterButtons(0);
    });

    this.resetFilterButton.on('click', () => {
      projector.resetFilterDataset();
      this.enableResetFilterButton(false);
    });

    this.clearSelectionButton.on('click', () => {
      projector.adjustSelectionAndHover([]);
    });
    this.enableResetFilterButton(false);
  }
}

document.registerElement(InspectorPanel.prototype.is, InspectorPanel);
