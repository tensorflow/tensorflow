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
import * as util from './util';
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
  private projector: Projector;
  private selectedPointIndices: number[];
  private neighborsOfFirstPoint: knn.NearestEntry[];
  private searchBox: ProjectorInput;

  private resetFilterButton: HTMLButtonElement;
  private setFilterButton: HTMLButtonElement;
  private clearSelectionButton: HTMLButtonElement;
  private limitMessage: HTMLDivElement;

  ready() {
    this.resetFilterButton =
        this.querySelector('.reset-filter') as HTMLButtonElement;
    this.setFilterButton =
        this.querySelector('.set-filter') as HTMLButtonElement;
    this.clearSelectionButton =
        this.querySelector('.clear-selection') as HTMLButtonElement;
    this.limitMessage = this.querySelector('.limit-msg') as HTMLDivElement;
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
    this.resetFilterButton.disabled = !enabled;
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
    const container = this.querySelector('.matches-list') as HTMLDivElement;
    container.style.display = indices.length ? null : 'none';
    const list = container.querySelector('.list') as HTMLDivElement;
    list.innerHTML = '';
    if (indices.length === 0) {
      return;
    }

    this.limitMessage.style.display =
        indices.length <= LIMIT_RESULTS ? 'none' : null;
    indices = indices.slice(0, LIMIT_RESULTS);

    for (let i = 0; i < indices.length; i++) {
      const index = indices[i];

      const row = document.createElement('div');
      row.className = 'row';

      const label = this.getLabelFromIndex(index);
      const rowLink = document.createElement('a');
      rowLink.className = 'label';
      rowLink.title = label;
      rowLink.innerText = label;

      rowLink.onmouseenter = () => {
        this.projectorEventContext.notifyHoverOverPoint(index);
      };
      rowLink.onmouseleave = () => {
        this.projectorEventContext.notifyHoverOverPoint(null);
      };
      rowLink.onclick = () => {
        this.projectorEventContext.notifySelectionChanged([index]);
      };

      row.appendChild(rowLink);
      list.appendChild(row);
    }
  }

  private getLabelFromIndex(pointIndex: number): string {
    const point = this.projector.dataSet.points[pointIndex];
    return point.metadata[this.selectedMetadataField].toString();
  }

  private updateNeighborsList(neighbors: knn.NearestEntry[]) {
    const nnlist = this.querySelector('.nn-list') as HTMLDivElement;
    nnlist.innerHTML = '';

    (this.querySelector('.nn') as HTMLDivElement).style.display =
        neighbors.length ? null : 'none';

    if (neighbors.length === 0) {
      return;
    }

    this.searchBox.message = '';
    const minDist = neighbors.length > 0 ? neighbors[0].dist : 0;

    for (let i = 0; i < neighbors.length; i++) {
      const neighbor = neighbors[i];

      const neighborElement = document.createElement('div');
      neighborElement.className = 'neighbor';

      const neighborElementLink = document.createElement('a');
      neighborElementLink.className = 'neighbor-link';
      neighborElementLink.title = this.getLabelFromIndex(neighbor.index);

      const labelValueElement = document.createElement('div');
      labelValueElement.className = 'label-and-value';

      const labelElement = document.createElement('div');
      labelElement.className = 'label';
      labelElement.style.color =
          adapter.dist2color(this.distFunc, neighbor.dist, minDist);
      labelElement.innerText = this.getLabelFromIndex(neighbor.index);

      const valueElement = document.createElement('div');
      valueElement.className = 'value';
      valueElement.innerText = neighbor.dist.toFixed(3);

      labelValueElement.appendChild(labelElement);
      labelValueElement.appendChild(valueElement);

      const barElement = document.createElement('div');
      barElement.className = 'bar';

      const barFillElement = document.createElement('div');
      barFillElement.className = 'fill';
      barFillElement.style.borderTopColor =
          adapter.dist2color(this.distFunc, neighbor.dist, minDist);
      barFillElement.style.width =
          adapter.normalizeDist(this.distFunc, neighbor.dist, minDist) * 100 +
          '%';
      barElement.appendChild(barFillElement);

      for (let j = 1; j < 4; j++) {
        const tickElement = document.createElement('div');
        tickElement.className = 'tick';
        tickElement.style.left = j * 100 / 4 + '%';
        barElement.appendChild(tickElement);
      }

      neighborElementLink.appendChild(labelValueElement);
      neighborElementLink.appendChild(barElement);
      neighborElement.appendChild(neighborElementLink);
      nnlist.appendChild(neighborElement);

      neighborElementLink.onmouseenter = () => {
        this.projectorEventContext.notifyHoverOverPoint(neighbor.index);
      };
      neighborElementLink.onmouseleave = () => {
        this.projectorEventContext.notifyHoverOverPoint(null);
      };
      neighborElementLink.onclick = () => {
        this.projectorEventContext.notifySelectionChanged([neighbor.index]);
      };
    }
  }

  private updateFilterButtons(numPoints: number) {
    if (numPoints > 1) {
      this.setFilterButton.innerText = `Isolate ${numPoints} points`;
      this.setFilterButton.disabled = null;
      this.clearSelectionButton.disabled = null;
    } else {
      this.setFilterButton.disabled = true;
      this.clearSelectionButton.disabled = true;
    }
  }

  private setupUI(projector: Projector) {
    this.distFunc = vector.cosDist;
    const eucDist =
        this.querySelector('.distance a.euclidean') as HTMLLinkElement;
    eucDist.onclick = () => {
      const links = this.querySelectorAll('.distance a');
      for (let i = 0; i < links.length; i++) {
        util.classed(links[i] as HTMLElement, 'selected', false);
      }
      util.classed(eucDist as HTMLElement, 'selected', true);

      this.distFunc = vector.dist;
      this.projectorEventContext.notifyDistanceMetricChanged(this.distFunc);
      const neighbors = projector.dataSet.findNeighbors(
          this.selectedPointIndices[0], this.distFunc, this.numNN);
      this.updateNeighborsList(neighbors);
    };

    const cosDist = this.querySelector('.distance a.cosine') as HTMLLinkElement;
    cosDist.onclick = () => {
      const links = this.querySelectorAll('.distance a');
      for (let i = 0; i < links.length; i++) {
        util.classed(links[i] as HTMLElement, 'selected', false);
      }
      util.classed(cosDist, 'selected', true);

      this.distFunc = vector.cosDist;
      this.projectorEventContext.notifyDistanceMetricChanged(this.distFunc);
      const neighbors = projector.dataSet.findNeighbors(
          this.selectedPointIndices[0], this.distFunc, this.numNN);
      this.updateNeighborsList(neighbors);
    };

    // Called whenever the search text input changes.
    const updateInput = (value: string, inRegexMode: boolean) => {
      if (value == null || value.trim() === '') {
        this.searchBox.message = '';
        this.projectorEventContext.notifySelectionChanged([]);
        return;
      }
      const indices = projector.dataSet.query(
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
    const numNNInput = this.$$('#nn-slider') as HTMLInputElement;
    const updateNumNN = () => {
      this.numNN = +numNNInput.value;
      (this.querySelector('.num-nn .nn-count') as HTMLSpanElement).innerText =
          '' + this.numNN;
      if (this.selectedPointIndices != null) {
        this.projectorEventContext.notifySelectionChanged(
            [this.selectedPointIndices[0]]);
      }
    };
    numNNInput.addEventListener('change', updateNumNN);
    updateNumNN();

    // Filtering dataset.
    this.setFilterButton.onclick = () => {
      const indices = this.selectedPointIndices.concat(
          this.neighborsOfFirstPoint.map(n => n.index));
      projector.filterDataset(indices);
      this.enableResetFilterButton(true);
      this.updateFilterButtons(0);
    };

    this.resetFilterButton.onclick = () => {
      projector.resetFilterDataset();
      this.enableResetFilterButton(false);
    };

    this.clearSelectionButton.onclick = () => {
      projector.adjustSelectionAndHover([]);
    };
    this.enableResetFilterButton(false);
  }
}

document.registerElement(InspectorPanel.prototype.is, InspectorPanel);
