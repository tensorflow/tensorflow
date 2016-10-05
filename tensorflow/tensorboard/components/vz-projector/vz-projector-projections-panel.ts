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
import {DataSet, PCA_SAMPLE_DIM, Projection, SAMPLE_SIZE} from './data';
import * as vector from './vector';
import {Projector} from './vz-projector';
import {ProjectorInput} from './vz-projector-input';
// tslint:disable-next-line:no-unused-variable
import {PolymerElement, PolymerHTMLElement} from './vz-projector-util';


// tslint:disable-next-line
export let ProjectionsPanelPolymer = PolymerElement({
  is: 'vz-projector-projections-panel',
  properties: {
    pcaComponents: {type: Array, value: d3.range(1, 11)},
    pcaX: {type: Number, value: 0, observer: 'showPCA'},
    pcaY: {type: Number, value: 1, observer: 'showPCA'},
    pcaZ: {type: Number, value: 2, observer: 'showPCA'},
    hasPcaZ: {type: Boolean, value: true},
  }
});

/**
 * A polymer component which handles the projection tabs in the projector.
 */
export class ProjectionsPanel extends ProjectionsPanelPolymer {
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
  }

  private setupUIControls() {
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

    let perplexityInput = this.dom.select('.tsne-perplexity input');
    let updatePerplexity = () => {
      this.perplexity = +perplexityInput.property('value');
      this.dom.select('.tsne-perplexity span').text(this.perplexity);
    };
    perplexityInput.property('value', this.perplexity)
        .on('input', updatePerplexity);
    updatePerplexity();

    let learningRateInput = this.dom.select('.tsne-learning-rate input');
    let updateLearningRate = () => {
      let val = +learningRateInput.property('value');
      this.learningRate = Math.pow(10, val);
      this.dom.select('.tsne-learning-rate span').text(this.learningRate);
    };
    learningRateInput.property('value', 1).on('input', updateLearningRate);
    updateLearningRate();
  }

  dataSetUpdated(dataSet: DataSet, dim: number) {
    this.currentDataSet = dataSet;
    this.dim = dim;
    this.clearCentroids();

    this.setupInputUIInCustomTab('xLeft');
    this.setupInputUIInCustomTab('xRight');
    this.setupInputUIInCustomTab('yUp');
    this.setupInputUIInCustomTab('yDown');
  }

  public showProjectionTab(projection: Projection) {
    if (projection === 'pca') {
      this.currentDataSet.stopTSNE();
      this.showPCA();
    } else if (projection === 'tsne') {
      this.showTSNE();
    } else if (projection === 'custom') {
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
        'tsne-0', 'tsne-1');

    if (!this.currentDataSet.hasTSNERun) {
      this.runTSNE();
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
      let x = this.pcaX;
      let y = this.pcaY;
      let z = this.pcaZ;
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
    if (this.centroids.xLeft == null || this.centroids.xRight == null ||
        this.centroids.yUp == null || this.centroids.yDown == null) {
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

  private setupInputUIInCustomTab(name: string) {
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

    updateInput('', false);

    // Setup the input text.
    input.onInputChanged((input, inRegexMode) => {
      updateInput(input, inRegexMode);
      this.showCustom();
    }, false /** callImmediately */);
  }

  private getCentroid(pattern: string, inRegexMode: boolean): CentroidResult {
    if (pattern == null || pattern === '') {
      return {numMatches: 0};
    }
    let accessor = (i: number) => this.currentDataSet.points[i].vector;
    // TODO(nsthorat): Don't use labelOption, create a new dropdown for this
    // component.
    let r = this.projector.currentDataSet.query(
        pattern, inRegexMode, this.projector.labelOption);
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
