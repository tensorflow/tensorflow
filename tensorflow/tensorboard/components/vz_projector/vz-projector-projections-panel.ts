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

import * as data from './data';
import {DataSet, Projection, ProjectionType, SpriteAndMetadataInfo, State} from './data';
import * as util from './util';
import * as vector from './vector';
import {Vector} from './vector';
import {Projector} from './vz-projector';
import {ProjectorInput} from './vz-projector-input';
// tslint:disable-next-line:no-unused-variable
import {PolymerElement, PolymerHTMLElement} from './vz-projector-util';

const NUM_PCA_COMPONENTS = 10;

// tslint:disable-next-line
export let ProjectionsPanelPolymer = PolymerElement({
  is: 'vz-projector-projections-panel',
  properties: {
    pcaIs3d:
        {type: Boolean, value: true, observer: '_pcaDimensionToggleObserver'},
    tSNEis3d:
        {type: Boolean, value: true, observer: '_tsneDimensionToggleObserver'},
    // PCA projection.
    pcaComponents: Array,
    pcaX: {type: Number, value: 0, observer: 'showPCAIfEnabled'},
    pcaY: {type: Number, value: 1, observer: 'showPCAIfEnabled'},
    pcaZ: {type: Number, value: 2, observer: 'showPCAIfEnabled'},
    // Custom projection.
    customSelectedSearchByMetadataOption: {
      type: String,
      observer: '_customSelectedSearchByMetadataOptionChanged'
    },
  }
});

type InputControlName = 'xLeft'|'xRight'|'yUp'|'yDown';

type CentroidResult = {
  centroid?: Vector; numMatches?: number;
};

type Centroids = {
  [key: string]: Vector; xLeft: Vector; xRight: Vector; yUp: Vector;
  yDown: Vector;
};

/**
 * A polymer component which handles the projection tabs in the projector.
 */
export class ProjectionsPanel extends ProjectionsPanelPolymer {
  private projector: Projector;
  private pcaComponents:
      Array<{id: number, componentNumber: number, percVariance: string}>;
  private currentProjection: ProjectionType;
  private polymerChangesTriggerReprojection: boolean;
  private dataSet: DataSet;
  private originalDataSet: DataSet;
  private dim: number;

  /** T-SNE perplexity. Roughly how many neighbors each point influences. */
  private perplexity: number;
  /** T-SNE learning rate. */
  private learningRate: number;

  private searchByMetadataOptions: string[];

  /** Centroids for custom projections. */
  private centroidValues: any;
  private centroids: Centroids;
  /** The centroid across all points. */
  private allCentroid: number[];

  /** Polymer properties. */
  // TODO(nsthorat): Move these to a separate view controller.
  public tSNEis3d: boolean;
  public pcaIs3d: boolean;
  public pcaX: number;
  public pcaY: number;
  public pcaZ: number;
  public customSelectedSearchByMetadataOption: string;

  /** Polymer elements. */
  private runTsneButton: HTMLButtonElement;
  private stopTsneButton: HTMLButtonElement;
  private perplexitySlider: HTMLInputElement;
  private learningRateInput: HTMLInputElement;
  private zDropdown: HTMLElement;
  private iterationLabel: HTMLElement;

  private customProjectionXLeftInput: ProjectorInput;
  private customProjectionXRightInput: ProjectorInput;
  private customProjectionYUpInput: ProjectorInput;
  private customProjectionYDownInput: ProjectorInput;

  initialize(projector: Projector) {
    this.polymerChangesTriggerReprojection = true;
    this.projector = projector;

    // Set up TSNE projections.
    this.perplexity = 30;
    this.learningRate = 10;

    // Setup Custom projections.
    this.centroidValues = {xLeft: null, xRight: null, yUp: null, yDown: null};
    this.clearCentroids();

    this.setupUIControls();
  }

  ready() {
    this.zDropdown = this.querySelector('#z-dropdown') as HTMLElement;
    this.runTsneButton = this.querySelector('.run-tsne') as HTMLButtonElement;
    this.stopTsneButton = this.querySelector('.stop-tsne') as HTMLButtonElement;
    this.perplexitySlider =
        this.querySelector('#perplexity-slider') as HTMLInputElement;
    this.learningRateInput =
        this.querySelector('#learning-rate-slider') as HTMLInputElement;
    this.iterationLabel = this.querySelector('.run-tsne-iter') as HTMLElement;
  }

  disablePolymerChangesTriggerReprojection() {
    this.polymerChangesTriggerReprojection = false;
  }

  enablePolymerChangesTriggerReprojection() {
    this.polymerChangesTriggerReprojection = true;
  }

  private updateTSNEPerplexityFromSliderChange() {
    if (this.perplexitySlider) {
      this.perplexity = +this.perplexitySlider.value;
    }
    (this.querySelector('.tsne-perplexity span') as HTMLSpanElement).innerText =
        '' + this.perplexity;
  }

  private updateTSNELearningRateFromUIChange() {
    if (this.learningRateInput) {
      this.learningRate = Math.pow(10, +this.learningRateInput.value);
    }
    (this.querySelector('.tsne-learning-rate span') as HTMLSpanElement)
        .innerText = '' + this.learningRate;
  }

  private setupUIControls() {
    {
      const self = this;
      const inkTabs = this.querySelectorAll('.ink-tab');
      for (let i = 0; i < inkTabs.length; i++) {
        inkTabs[i].addEventListener('click', function() {
          let id = this.getAttribute('data-tab');
          self.showTab(id);
        });
      }
    }

    this.runTsneButton.addEventListener('click', () => this.runTSNE());
    this.stopTsneButton.addEventListener(
        'click', () => this.dataSet.stopTSNE());

    this.perplexitySlider.value = this.perplexity.toString();
    this.perplexitySlider.addEventListener(
        'change', () => this.updateTSNEPerplexityFromSliderChange());
    this.updateTSNEPerplexityFromSliderChange();

    this.learningRateInput.addEventListener(
        'change', () => this.updateTSNELearningRateFromUIChange());
    this.updateTSNELearningRateFromUIChange();

    this.setupCustomProjectionInputFields();
    // TODO: figure out why `--paper-input-container-input` css mixin didn't
    // work.
    const inputs =
        this.querySelectorAll('paper-dropdown-menu paper-input input');
    for (let i = 0; i < inputs.length; i++) {
      (inputs[i] as HTMLElement).style.fontSize = '14px';
    }
  }

  restoreUIFromBookmark(bookmark: State) {
    this.disablePolymerChangesTriggerReprojection();

    // PCA
    this.pcaX = bookmark.pcaComponentDimensions[0];
    this.pcaY = bookmark.pcaComponentDimensions[1];
    if (bookmark.pcaComponentDimensions.length === 3) {
      this.pcaZ = bookmark.pcaComponentDimensions[2];
    }
    this.pcaIs3d = (bookmark.pcaComponentDimensions.length === 3);

    // t-SNE
    if (this.perplexitySlider) {
      this.perplexitySlider.value = bookmark.tSNEPerplexity.toString();
    }
    if (this.learningRateInput) {
      this.learningRateInput.value = bookmark.tSNELearningRate.toString();
    }
    this.tSNEis3d = bookmark.tSNEis3d;

    // custom
    this.customSelectedSearchByMetadataOption =
        bookmark.customSelectedSearchByMetadataOption;
    if (this.customProjectionXLeftInput) {
      this.customProjectionXLeftInput.set(
          bookmark.customXLeftText, bookmark.customXLeftRegex);
    }
    if (this.customProjectionXRightInput) {
      this.customProjectionXRightInput.set(
          bookmark.customXRightText, bookmark.customXRightRegex);
    }
    if (this.customProjectionYUpInput) {
      this.customProjectionYUpInput.set(
          bookmark.customYUpText, bookmark.customYUpRegex);
    }
    if (this.customProjectionYDownInput) {
      this.customProjectionYDownInput.set(
          bookmark.customYDownText, bookmark.customYDownRegex);
    }
    this.computeAllCentroids();

    this.setZDropdownEnabled(this.pcaIs3d);
    this.updateTSNEPerplexityFromSliderChange();
    this.updateTSNELearningRateFromUIChange();
    if (this.iterationLabel) {
      this.iterationLabel.innerText = bookmark.tSNEIteration.toString();
    }
    if (bookmark.selectedProjection != null) {
      this.showTab(bookmark.selectedProjection);
    }
    this.enablePolymerChangesTriggerReprojection();
  }

  populateBookmarkFromUI(bookmark: State) {
    this.disablePolymerChangesTriggerReprojection();

    // PCA
    bookmark.pcaComponentDimensions = [this.pcaX, this.pcaY];
    if (this.pcaIs3d) {
      bookmark.pcaComponentDimensions.push(this.pcaZ);
    }

    // t-SNE
    if (this.perplexitySlider != null) {
      bookmark.tSNEPerplexity = +this.perplexitySlider.value;
    }
    if (this.learningRateInput != null) {
      bookmark.tSNELearningRate = +this.learningRateInput.value;
    }
    bookmark.tSNEis3d = this.tSNEis3d;

    // custom
    bookmark.customSelectedSearchByMetadataOption =
        this.customSelectedSearchByMetadataOption;
    if (this.customProjectionXLeftInput != null) {
      bookmark.customXLeftText = this.customProjectionXLeftInput.getValue();
      bookmark.customXLeftRegex =
          this.customProjectionXLeftInput.getInRegexMode();
    }
    if (this.customProjectionXRightInput != null) {
      bookmark.customXRightText = this.customProjectionXRightInput.getValue();
      bookmark.customXRightRegex =
          this.customProjectionXRightInput.getInRegexMode();
    }
    if (this.customProjectionYUpInput != null) {
      bookmark.customYUpText = this.customProjectionYUpInput.getValue();
      bookmark.customYUpRegex = this.customProjectionYUpInput.getInRegexMode();
    }
    if (this.customProjectionYDownInput != null) {
      bookmark.customYDownText = this.customProjectionYDownInput.getValue();
      bookmark.customYDownRegex =
          this.customProjectionYDownInput.getInRegexMode();
    }

    this.enablePolymerChangesTriggerReprojection();
  }

  // This method is marked as public as it is used as the view method that
  // abstracts DOM manipulation so we can stub it in a test.
  // TODO(nsthorat): Move this to its own class as the glue between this class
  // and the DOM.
  setZDropdownEnabled(enabled: boolean) {
    if (this.zDropdown) {
      if (enabled) {
        this.zDropdown.removeAttribute('disabled');
      } else {
        this.zDropdown.setAttribute('disabled', 'true');
      }
    }
  }

  dataSetUpdated(dataSet: DataSet, originalDataSet: DataSet, dim: number) {
    this.dataSet = dataSet;
    this.originalDataSet = originalDataSet;
    this.dim = dim;
    const pointCount = (dataSet == null) ? 0 : dataSet.points.length;
    const perplexity = Math.max(5, Math.ceil(Math.sqrt(pointCount) / 4));
    this.perplexitySlider.value = perplexity.toString();
    this.updateTSNEPerplexityFromSliderChange();
    this.clearCentroids();

    (this.querySelector('#tsne-sampling') as HTMLElement).style.display =
        pointCount > data.TSNE_SAMPLE_SIZE ? null : 'none';
    const wasSampled =
        (dataSet == null) ? false : (dataSet.dim[0] > data.PCA_SAMPLE_DIM ||
                                     dataSet.dim[1] > data.PCA_SAMPLE_DIM);
    (this.querySelector('#pca-sampling') as HTMLElement).style.display =
        wasSampled ? null : 'none';
    this.showTab('pca');
  }

  _pcaDimensionToggleObserver() {
    this.setZDropdownEnabled(this.pcaIs3d);
    this.beginProjection(this.currentProjection);
  }

  _tsneDimensionToggleObserver() {
    this.beginProjection(this.currentProjection);
  }

  metadataChanged(spriteAndMetadata: SpriteAndMetadataInfo) {
    // Project by options for custom projections.
    let searchByMetadataIndex = -1;
    this.searchByMetadataOptions = spriteAndMetadata.stats.map((stats, i) => {
      // Make the default label by the first non-numeric column.
      if (!stats.isNumeric && searchByMetadataIndex === -1) {
        searchByMetadataIndex = i;
      }
      return stats.name;
    });
    this.customSelectedSearchByMetadataOption =
        this.searchByMetadataOptions[Math.max(0, searchByMetadataIndex)];
  }

  public showTab(id: ProjectionType) {
    this.currentProjection = id;

    const tab =
        this.querySelector('.ink-tab[data-tab="' + id + '"]') as HTMLElement;
    const allTabs = this.querySelectorAll('.ink-tab');
    for (let i = 0; i < allTabs.length; i++) {
      util.classed(allTabs[i] as HTMLElement, 'active', false);
    }

    util.classed(tab, 'active', true);

    const allTabContent = this.querySelectorAll('.ink-panel-content');
    for (let i = 0; i < allTabContent.length; i++) {
      util.classed(allTabContent[i] as HTMLElement, 'active', false);
    }

    util.classed(
        this.querySelector('.ink-panel-content[data-panel="' + id + '"]') as
            HTMLElement,
        'active', true);

    // guard for unit tests, where polymer isn't attached and $ doesn't exist.
    if (this.$ != null) {
      const main = this.$['main'];
      // In order for the projections panel to animate its height, we need to
      // set it explicitly.
      requestAnimationFrame(() => {
        this.style.height = main.clientHeight + 'px';
      });
    }

    this.beginProjection(id);
  }

  private beginProjection(projection: ProjectionType) {
    if (this.polymerChangesTriggerReprojection === false) {
      return;
    }
    if (projection === 'pca') {
      if (this.dataSet != null) {
        this.dataSet.stopTSNE();
      }
      this.showPCA();
    } else if (projection === 'tsne') {
      this.showTSNE();
    } else if (projection === 'custom') {
      if (this.dataSet != null) {
        this.dataSet.stopTSNE();
      }
      this.computeAllCentroids();
      this.reprojectCustom();
    }
  }

  private showTSNE() {
    const dataSet = this.dataSet;
    if (dataSet == null) {
      return;
    }
    const accessors =
        data.getProjectionComponents('tsne', [0, 1, this.tSNEis3d ? 2 : null]);
    const dimensionality = this.tSNEis3d ? 3 : 2;
    const projection =
        new Projection('tsne', accessors, dimensionality, dataSet);
    this.projector.setProjection(projection);

    if (!this.dataSet.hasTSNERun) {
      this.runTSNE();
    } else {
      this.projector.notifyProjectionPositionsUpdated();
    }
  }

  private runTSNE() {
    this.runTsneButton.disabled = true;
    this.stopTsneButton.disabled = null;
    this.dataSet.projectTSNE(
        this.perplexity, this.learningRate, this.tSNEis3d ? 3 : 2,
        (iteration: number) => {
          if (iteration != null) {
            this.iterationLabel.innerText = '' + iteration;
            this.projector.notifyProjectionPositionsUpdated();
          } else {
            this.runTsneButton.disabled = null;
            this.stopTsneButton.disabled = true;
          }
        });
  }

  // tslint:disable-next-line:no-unused-variable
  private showPCAIfEnabled() {
    if (this.polymerChangesTriggerReprojection) {
      this.showPCA();
    }
  }

  private updateTotalVarianceMessage() {
    let variances = this.dataSet.fracVariancesExplained;
    let totalVariance = variances[this.pcaX] + variances[this.pcaY];
    let msg = 'Total variance described: ';
    if (this.pcaIs3d) {
      totalVariance += variances[this.pcaZ];
    }
    msg += (totalVariance * 100).toFixed(1) + '%.';
    (this.querySelector('#total-variance') as HTMLElement).innerHTML = msg;
  }

  private showPCA() {
    if (this.dataSet == null) {
      return;
    }
    this.dataSet.projectPCA().then(() => {
      // Polymer properties are 1-based.
      const accessors = data.getProjectionComponents(
          'pca', [this.pcaX, this.pcaY, this.pcaZ]);

      const dimensionality = this.pcaIs3d ? 3 : 2;
      const projection =
          new Projection('pca', accessors, dimensionality, this.dataSet);
      this.projector.setProjection(projection);
      let numComponents = Math.min(NUM_PCA_COMPONENTS, this.dataSet.dim[1]);
      this.updateTotalVarianceMessage();
      this.pcaComponents = util.range(numComponents).map(i => {
        let fracVariance = this.dataSet.fracVariancesExplained[i];
        return {
          id: i,
          componentNumber: i + 1,
          percVariance: (fracVariance * 100).toFixed(1)
        };
      });
    });
  }

  private reprojectCustom() {
    if (this.centroids == null || this.centroids.xLeft == null ||
        this.centroids.xRight == null || this.centroids.yUp == null ||
        this.centroids.yDown == null) {
      return;
    }
    const xDir = vector.sub(this.centroids.xRight, this.centroids.xLeft);
    this.dataSet.projectLinear(xDir, 'linear-x');

    const yDir = vector.sub(this.centroids.yUp, this.centroids.yDown);
    this.dataSet.projectLinear(yDir, 'linear-y');

    const accessors = data.getProjectionComponents('custom', ['x', 'y']);
    const projection = new Projection('custom', accessors, 2, this.dataSet);
    this.projector.setProjection(projection);
  }

  clearCentroids(): void {
    this.centroids = {xLeft: null, xRight: null, yUp: null, yDown: null};
    this.allCentroid = null;
  }

  _customSelectedSearchByMetadataOptionChanged(newVal: string, oldVal: string) {
    if (this.polymerChangesTriggerReprojection === false) {
      return;
    }
    if (this.currentProjection === 'custom') {
      this.computeAllCentroids();
      this.reprojectCustom();
    }
  }

  private setupCustomProjectionInputFields() {
    this.customProjectionXLeftInput =
        this.setupCustomProjectionInputField('xLeft');
    this.customProjectionXRightInput =
        this.setupCustomProjectionInputField('xRight');
    this.customProjectionYUpInput = this.setupCustomProjectionInputField('yUp');
    this.customProjectionYDownInput =
        this.setupCustomProjectionInputField('yDown');
  }

  private computeAllCentroids() {
    this.computeCentroid('xLeft');
    this.computeCentroid('xRight');
    this.computeCentroid('yUp');
    this.computeCentroid('yDown');
  }

  private computeCentroid(name: InputControlName) {
    const input = this.querySelector('#' + name) as ProjectorInput;
    if (input == null) {
      return;
    }
    const value = input.getValue();
    if (value == null) {
      return;
    }
    let inRegexMode = input.getInRegexMode();
    let result = this.getCentroid(value, inRegexMode);
    if (result.numMatches === 0) {
      input.message = '0 matches. Using a random vector.';
      result.centroid = vector.rn(this.dim);
    } else {
      input.message = `${result.numMatches} matches.`;
    }
    this.centroids[name] = result.centroid;
    this.centroidValues[name] = value;
  }

  private setupCustomProjectionInputField(name: InputControlName):
      ProjectorInput {
    let input = this.querySelector('#' + name) as ProjectorInput;
    input.registerInputChangedListener((input, inRegexMode) => {
      if (this.polymerChangesTriggerReprojection) {
        this.computeCentroid(name);
        this.reprojectCustom();
      }
    });
    return input;
  }

  private getCentroid(pattern: string, inRegexMode: boolean): CentroidResult {
    if (pattern == null || pattern === '') {
      return {numMatches: 0};
    }
    // Search by the original dataset since we often want to filter and project
    // only the nearest neighbors of A onto B-C where B and C are not nearest
    // neighbors of A.
    let accessor = (i: number) => this.originalDataSet.points[i].vector;
    let r = this.originalDataSet.query(
        pattern, inRegexMode, this.customSelectedSearchByMetadataOption);
    return {centroid: vector.centroid(r, accessor), numMatches: r.length};
  }

  getPcaSampledDimText() {
    return data.PCA_SAMPLE_DIM.toLocaleString();
  }

  getPcaSampleSizeText() {
    return data.PCA_SAMPLE_SIZE.toLocaleString();
  }

  getTsneSampleSizeText() {
    return data.TSNE_SAMPLE_SIZE.toLocaleString();
  }
}

document.registerElement(ProjectionsPanel.prototype.is, ProjectionsPanel);
