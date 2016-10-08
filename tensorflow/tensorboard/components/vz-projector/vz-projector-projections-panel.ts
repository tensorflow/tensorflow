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
import {DataSet, MetadataInfo, PCA_SAMPLE_DIM, Projection, SAMPLE_SIZE} from './data';
import * as vector from './vector';
import {Projector} from './vz-projector';
import {ProjectorInput} from './vz-projector-input';
// tslint:disable-next-line:no-unused-variable
import {PolymerElement, PolymerHTMLElement} from './vz-projector-util';

// tslint:disable-next-line
export let ProjectionsPanelPolymer = PolymerElement({
  is: 'vz-projector-projections-panel',
  properties: {
    // PCA projection.
    pcaComponents: {type: Array, value: d3.range(1, 11)},
    pcaX: {type: Number, value: 1, observer: 'showPCA'},
    pcaY: {type: Number, value: 2, observer: 'showPCA'},
    pcaZ: {type: Number, value: 3, observer: 'showPCA'},
    hasPcaZ: {type: Boolean, value: true},
    // Custom projection.
    selectedSearchByMetadataOption: {
      type: String,
      value: 'label',
      observer: '_searchByMetadataOptionChanged'
    },
  }
});

/**
 * A polymer component which handles the projection tabs in the projector.
 */
export class ProjectionsPanel extends ProjectionsPanelPolymer {
  selectedSearchByMetadataOption: string;

  private projector: Projector;

  // The working subset of the data source's original data set.
  private currentDataSet: DataSet;
  private dim: number;

  /** Number of dimensions for the scatter plot. */
  private dimension: number;
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
  private pcaX: number;
  private pcaY: number;
  private pcaZ: number;
  private hasPcaZ: boolean;

  /** Polymer elements. */
  private runTsneButton: d3.Selection<HTMLButtonElement>;
  private stopTsneButton: d3.Selection<HTMLButtonElement>;

  private dom: d3.Selection<any>;

  initialize(projector: Projector) {
    this.projector = projector;

    this.dimension = 3;

    // Set up TSNE projections.
    this.perplexity = 30;
    this.learningRate = 10;

    // Set up PCA projections.
    this.hasPcaZ = true;

    // Setup Custom projections.
    this.centroidValues = {xLeft: null, xRight: null, yUp: null, yDown: null};
    this.clearCentroids();

    this.setupUIControls();
  }

  ready() {
    this.dom = d3.select(this);

    this.searchByMetadataOptions = ['label'];
  }

  private setupUIControls() {
    // Tabs
    const self = this;
    this.dom.selectAll('.ink-tab').on('click', function() {
      let id = this.getAttribute('data-tab');
      self.showTab(id);
    });

    // Unknown why, but the polymer toggle button stops working
    // as soon as you do d3.select() on it.
    let tsneToggle = this.querySelector('#tsne-toggle') as HTMLInputElement;
    let zCheckbox = this.querySelector('#z-checkbox') as HTMLInputElement;

    // PCA controls.
    zCheckbox.addEventListener('change', () => {
      // Make sure tsne stays in the same dimension as PCA.
      this.dimension = this.hasPcaZ ? 3 : 2;
      tsneToggle.checked = this.hasPcaZ;
      this.showPCA();
    });

    // TSNE controls.
    tsneToggle.addEventListener('change', () => {
      // Make sure PCA stays in the same dimension as tsne.
      this.hasPcaZ = tsneToggle.checked;
      this.dimension = tsneToggle.checked ? 3 : 2;
      this.showTSNE();
    });

    this.runTsneButton = this.dom.select('.run-tsne');
    this.runTsneButton.on('click', () => this.runTSNE());
    this.stopTsneButton = this.dom.select('.stop-tsne');
    this.stopTsneButton.on('click', () => {
      this.projector.currentDataSet.stopTSNE();
    });

    let perplexitySlider = this.$$('#perplexity-slider') as HTMLInputElement;
    let updatePerplexity = () => {
      this.perplexity = +perplexitySlider.value;
      this.dom.select('.tsne-perplexity span').text(this.perplexity);
    };
    perplexitySlider.value = this.perplexity.toString();
    perplexitySlider.addEventListener('change', updatePerplexity);
    updatePerplexity();

    let learningRateInput =
        this.$$('#learning-rate-slider') as HTMLInputElement;
    let updateLearningRate = () => {
      this.learningRate = Math.pow(10, +learningRateInput.value);
      this.dom.select('.tsne-learning-rate span').text(this.learningRate);
    };
    learningRateInput.addEventListener('change', updateLearningRate);
    updateLearningRate();
  }

  dataSetUpdated(dataSet: DataSet, dim: number) {
    this.currentDataSet = dataSet;
    this.dim = dim;
    this.clearCentroids();

    this.setupAllInputsInCustomTab();

    this.dom.select('#tsne-sampling')
        .style('display', dataSet.points.length > SAMPLE_SIZE ? null : 'none');
    this.dom.select('#pca-sampling')
        .style('display', dataSet.dim[1] > PCA_SAMPLE_DIM ? null : 'none');
    this.showTab('pca');
  }

  metadataChanged(metadata: MetadataInfo) {
    // Project by options for custom projections.
    let searchByMetadataIndex = -1;
    if (metadata.stats.length > 1) {
      this.searchByMetadataOptions = metadata.stats.map((stats, i) => {
        // Make the default label by the first non-numeric column.
        if (!stats.isNumeric && searchByMetadataIndex === -1) {
          searchByMetadataIndex = i;
        }
        return stats.name;
      });
    } else {
      this.searchByMetadataOptions = ['label'];
    }
    this.selectedSearchByMetadataOption =
        this.searchByMetadataOptions[Math.max(0, searchByMetadataIndex)];
  }

  public showTab(id: Projection) {
    let tab = this.dom.select('.ink-tab[data-tab="' + id + '"]');
    let pane =
        d3.select((tab.node() as HTMLElement).parentNode.parentNode.parentNode);
    pane.selectAll('.ink-tab').classed('active', false);
    tab.classed('active', true);
    pane.selectAll('.ink-panel-content').classed('active', false);
    pane.select('.ink-panel-content[data-panel="' + id + '"]')
        .classed('active', true);
    if (id === 'pca') {
      this.currentDataSet.stopTSNE();
      this.showPCA();
    } else if (id === 'tsne') {
      this.showTSNE();
    } else if (id === 'custom') {
      this.currentDataSet.stopTSNE();
      this.showCustom();
    }
  }

  private showTSNE() {
    this.projector.setProjection(
        'tsne',
        // Accessors.
        i => this.currentDataSet.points[i].projections['tsne-0'],
        i => this.currentDataSet.points[i].projections['tsne-1'],
        this.dimension === 3 ?
            (i => this.currentDataSet.points[i].projections['tsne-2']) :
            null,
        // Axis labels.
        'tsne-0', 'tsne-1', true /** deferUpdate */);

    if (!this.currentDataSet.hasTSNERun) {
      this.runTSNE();
    } else {
      this.projector.notifyProjectionsUpdated();
    }
  }

  private runTSNE() {
    this.runTsneButton.attr('disabled', true);
    this.stopTsneButton.attr('disabled', null);
    this.currentDataSet.projectTSNE(
        this.perplexity, this.learningRate, this.dimension,
        (iteration: number) => {
          if (iteration != null) {
            this.dom.select('.run-tsne-iter').text(iteration);
            this.projector.notifyProjectionsUpdated();
          } else {
            this.runTsneButton.attr('disabled', null);
            this.stopTsneButton.attr('disabled', true);
          }
        });
  }

  private showPCA() {
    if (this.currentDataSet == null) {
      return;
    }
    this.currentDataSet.projectPCA().then(() => {
      // Polymer properties are 1-based.
      let x = this.pcaX - 1;
      let y = this.pcaY - 1;
      let z = this.pcaZ - 1;
      let hasZ = this.dimension === 3;

      this.projector.setProjection(
          'pca',
          // Accessors.
          i => this.currentDataSet.points[i].projections['pca-' + x],
          i => this.currentDataSet.points[i].projections['pca-' + y],
          hasZ ? (i => this.currentDataSet.points[i].projections['pca-' + z]) :
                 null,
          // Axis labels.
          'pca-' + x, 'pca-' + y);
    });
  }

  private showCustom() {
    if (this.centroids == null || this.centroids.xLeft == null ||
        this.centroids.xRight == null || this.centroids.yUp == null ||
        this.centroids.yDown == null) {
      return;
    }
    let xDir = vector.sub(this.centroids.xRight, this.centroids.xLeft);
    this.currentDataSet.projectLinear(xDir, 'linear-x');

    let yDir = vector.sub(this.centroids.yUp, this.centroids.yDown);
    this.currentDataSet.projectLinear(yDir, 'linear-y');

    let xLabel = this.centroidValues.xLeft + ' → ' + this.centroidValues.xRight;
    let yLabel = this.centroidValues.yUp + ' → ' + this.centroidValues.yDown;

    this.projector.setProjection(
        'custom',
        // Accessors.
        i => this.currentDataSet.points[i].projections['linear-x'],
        i => this.currentDataSet.points[i].projections['linear-y'],
        null,  // Z accessor.
        // Axis labels.
        xLabel, yLabel);
  }

  clearCentroids(): void {
    this.centroids = {xLeft: null, xRight: null, yUp: null, yDown: null};
    this.allCentroid = null;
  }

  _searchByMetadataOptionChanged(newVal: string, oldVal: string) {
    // Ignore the initial call to the observer so we don't try to create these
    // projections pre-emptively.
    if (oldVal) {
      this.setupAllInputsInCustomTab(true /** callListenersImmediately */);
    }
  }

  private setupAllInputsInCustomTab(callListenersImmediately = false) {
    this.setupInputUIInCustomTab('xLeft', callListenersImmediately);
    this.setupInputUIInCustomTab('xRight', callListenersImmediately);
    this.setupInputUIInCustomTab('yUp', callListenersImmediately);
    this.setupInputUIInCustomTab('yDown', callListenersImmediately);
  }

  private setupInputUIInCustomTab(
      name: string, callListenersImmediately = false) {
    let input = this.querySelector('#' + name) as ProjectorInput;

    let updateInput = (value: string, inRegexMode: boolean) => {
      if (value == null) {
        return;
      }
      let result = this.getCentroid(value, inRegexMode);
      if (result.numMatches === 0) {
        input.message = '0 matches. Using a random vector.';
        result.centroid = vector.rn(this.dim);
      } else {
        input.message = `${result.numMatches} matches.`;
      }
      this.centroids[name] = result.centroid;
      this.centroidValues[name] = value;
    };

    updateInput(input.getValue(), false);

    // Setup the input text.
    input.onInputChanged((input, inRegexMode) => {
      updateInput(input, inRegexMode);
      this.showCustom();
    }, callListenersImmediately);
  }

  private getCentroid(pattern: string, inRegexMode: boolean): CentroidResult {
    if (pattern == null || pattern === '') {
      return {numMatches: 0};
    }
    let accessor = (i: number) => this.currentDataSet.points[i].vector;
    let r = this.projector.currentDataSet.query(
        pattern, inRegexMode, this.selectedSearchByMetadataOption);
    return {centroid: vector.centroid(r, accessor), numMatches: r.length};
  }

  getPcaSampledDim() {
    return PCA_SAMPLE_DIM.toLocaleString();
  }

  getTsneSampleSize() {
    return SAMPLE_SIZE.toLocaleString();
  }
}

type CentroidResult = {
  centroid?: number[]; numMatches?: number;
};

type Centroids = {
  [key: string]: number[]; xLeft: number[]; xRight: number[]; yUp: number[];
  yDown: number[];
};

document.registerElement(ProjectionsPanel.prototype.is, ProjectionsPanel);
