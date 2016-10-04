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

import {ColorOption, DataSet, PCA_SAMPLE_DIM, Projection, SAMPLE_SIZE, State} from './data';
import {updateWarningMessage} from './async';
import {DataProvider, getDataProvider, MetadataResult} from './data-loader';
import * as knn from './knn';
import {Mode, ScatterPlot} from './scatterPlot';
import {ScatterPlotVisualizer3DLabels} from './scatterPlotVisualizer3DLabels';
import {ScatterPlotVisualizerCanvasLabels} from './scatterPlotVisualizerCanvasLabels';
import {ScatterPlotVisualizerSprites} from './scatterPlotVisualizerSprites';
import {ScatterPlotVisualizerTraces} from './scatterPlotVisualizerTraces';
import {SelectionChangedListener, SelectionContext} from './selectionContext';
import * as vector from './vector';
import {BookmarkPanel} from './vz-projector-bookmark-panel';
import {DataPanel} from './vz-projector-data-panel';
import {ProjectorInput} from './vz-projector-input';
// tslint:disable-next-line:no-unused-variable
import {PolymerElement, PolymerHTMLElement} from './vz-projector-util';
import {InspectorPanel} from './vz-projector-inspector-panel';


/** T-SNE perplexity. Roughly how many neighbors each point influences. */
let perplexity: number = 30;
/** T-SNE learning rate. */
let learningRate: number = 10;
/** Number of dimensions for the scatter plot. */
let dimension = 3;

/** Highlight stroke color for the nearest neighbors. */
const NN_HIGHLIGHT_COLOR = '#FA6666';
/** Color to denote a missing value. */
const MISSING_VALUE_COLOR = 'black';
/** Highlight stroke color for the selected point */
const POINT_HIGHLIGHT_COLOR = '#760B4F';

/**
 * The minimum number of dimensions the data should have to automatically
 * decide to normalize the data.
 */
const THRESHOLD_DIM_NORMALIZE = 50;

type Centroids = {
  [key: string]: number[]; xLeft: number[]; xRight: number[]; yUp: number[];
  yDown: number[];
};

export let ProjectorPolymer = PolymerElement({
  is: 'vz-projector',
  properties: {
    // Private.
    pcaComponents: {type: Array, value: d3.range(1, 11)},
    pcaX: {type: Number, value: 0, observer: 'showPCA'},
    pcaY: {type: Number, value: 1, observer: 'showPCA'},
    pcaZ: {type: Number, value: 2, observer: 'showPCA'},
    routePrefix: String,
    hasPcaZ: {type: Boolean, value: true},
    labelOption: {type: String, observer: '_labelOptionChanged'},
    colorOption: {type: Object, observer: '_colorOptionChanged'}
  }
});

export class Projector extends ProjectorPolymer implements SelectionContext {
  // The working subset of the data source's original data set.
  currentDataSet: DataSet;

  private selectionChangedListeners: SelectionChangedListener[];

  private dataSet: DataSet;
  private dom: d3.Selection<any>;
  private pcaX: number;
  private pcaY: number;
  private pcaZ: number;
  private hasPcaZ: boolean;
  private scatterPlot: ScatterPlot;
  private dim: number;
  private highlightedPoints: {index: number, color: string}[];
  // The index of all selected points.
  private selectedPoints: number[];
  private centroidValues: any;
  private centroids: Centroids;
  /** The centroid across all points. */
  private allCentroid: number[];
  private dataProvider: DataProvider;
  private dataPanel: DataPanel;
  private bookmarkPanel: BookmarkPanel;
  private colorOption: ColorOption;
  private labelOption: string;
  private routePrefix: string;
  private selectedProjection: Projection = 'pca';
  private normalizeData: boolean;
  private inspectorPanel: InspectorPanel;

  // t-SNE.
  private runTsneButton: d3.Selection<HTMLButtonElement>;
  private stopTsneButton: d3.Selection<HTMLButtonElement>;

  ready() {
    this.selectionChangedListeners = [];
    this.dataPanel = this.$['data-panel'] as DataPanel;
    this.inspectorPanel = this.$['inspector-panel'] as InspectorPanel;
    this.inspectorPanel.initialize(this);
    // Get the data loader and initialize the data panel with it.
    getDataProvider(this.routePrefix, dataProvider => {
      this.dataProvider = dataProvider;
      this.dataPanel.initialize(this, dataProvider);
    });

    this.bookmarkPanel = this.$['bookmark-panel'] as BookmarkPanel;
    this.bookmarkPanel.initialize(this);

    // And select a default dataset.
    this.hasPcaZ = true;
    this.highlightedPoints = [];
    this.selectedPoints = [];
    this.centroidValues = {xLeft: null, xRight: null, yUp: null, yDown: null};
    this.clearCentroids();
    this.dom = d3.select(this);
    // Sets up all the UI.
    this.setupUIControls();
  }

  _labelOptionChanged() {
    let labelAccessor = (i: number): string => {
      return this.currentDataSet.points[i].metadata[this.labelOption] as string;
    };
    this.scatterPlot.setLabelAccessor(labelAccessor);
  }

  _colorOptionChanged() {
    let colorMap = this.colorOption.map;
    if (colorMap == null) {
      this.scatterPlot.setColorAccessor(null);
      return;
    };
    let colors = (i: number) => {
      let value = this.currentDataSet.points[i].metadata[this.colorOption.name];
      if (value == null) {
        return MISSING_VALUE_COLOR;
      }
      return colorMap(value);
    };
    this.scatterPlot.setColorAccessor(colors);
  }

  clearCentroids(): void {
    this.centroids = {xLeft: null, xRight: null, yUp: null, yDown: null};
    this.allCentroid = null;
  }

  setNormalizeData(normalizeData: boolean) {
    this.normalizeData = normalizeData;
    this.setCurrentDataSet(this.dataSet.getSubset());
  }

  updateDataSet(ds: DataSet) {
    this.dataSet = ds;
    if (this.scatterPlot == null || this.dataSet == null) {
      // We are not ready yet.
      return;
    }
    this.normalizeData = this.dataSet.dim[1] >= THRESHOLD_DIM_NORMALIZE;
    this.dataPanel.setNormalizeData(this.normalizeData);
    this.setCurrentDataSet(this.dataSet.getSubset());
    this.dom.select('.reset-filter').style('display', 'none');
    this.clearCentroids();
    this.setupInputUIInCustomTab('xLeft');
    this.setupInputUIInCustomTab('xRight');
    this.setupInputUIInCustomTab('yUp');
    this.setupInputUIInCustomTab('yDown');

    // Set the container to a fixed height, otherwise in Colab the
    // height can grow indefinitely.
    let container = this.dom.select('#container');
    container.style('height', container.property('clientHeight') + 'px');
  }

  /**
   * Registers a listener to be called any time the selected point set changes.
   */
  registerSelectionChangedListener(listener: SelectionChangedListener) {
    this.selectionChangedListeners.push(listener);
  }

  filterDataset() {
    this.setCurrentDataSet(this.currentDataSet.getSubset(this.selectedPoints));
    this.clearSelection();
    this.scatterPlot.recreateScene();
  }

  resetFilterDataset() {
    this.setCurrentDataSet(this.dataSet.getSubset(null));
    this.selectedPoints = [];
  }

  /**
   * Used by clients to indicate that a selection has occurred.
   */
  notifySelectionChanged(newSelectedPointIndices: number[]) {
    this.selectedPoints = newSelectedPointIndices;
    let neighbors: knn.NearestEntry[] = [];

    if (newSelectedPointIndices.length === 1) {
      const firstSelectedIndex = newSelectedPointIndices[0];
      this.scatterPlot.clickOnPoint(firstSelectedIndex);
      neighbors = this.currentDataSet.findNeighbors(firstSelectedIndex,
          this.inspectorPanel.distFunc, this.inspectorPanel.numNN);
      this.selectedPoints =
          [newSelectedPointIndices[0]].concat(neighbors.map(n => n.index));
    }

    this.selectionChangedListeners.forEach(
        l => l(this.selectedPoints, neighbors));
  }

  mergeMetadata(result: MetadataResult): void {
    let numTensors = this.dataSet.points.length;
    if (result.metadata.length !== numTensors) {
      updateWarningMessage(
          `Number of tensors (${numTensors}) do not match` +
          ` the number of lines in metadata (${result.metadata.length}).`);
    }
    this.dataSet.mergeMetadata(result.metadata);
    this.setCurrentDataSet(this.dataSet.getSubset());
    this.dataSet.spriteImage = result.spriteImage;
    this.dataSet.metadata = result.datasetMetadata;
    this.inspectorPanel.metadataChanged(result);
  }

  clearSelection() {
    this.notifySelectionChanged([]);
    this.scatterPlot.setMode(Mode.HOVER);
  }

  private unsetCurrentDataSet() {
    this.currentDataSet.stopTSNE();
  }

  private setCurrentDataSet(ds: DataSet) {
    this.clearSelection();
    if (this.currentDataSet != null) {
      this.unsetCurrentDataSet();
    }
    this.currentDataSet = ds;
    if (this.normalizeData) {
      this.currentDataSet.normalize();
    }
    this.scatterPlot.setDataSet(this.currentDataSet, this.dataSet.spriteImage);
    this.dim = this.currentDataSet.dim[1];
    this.dom.select('span.numDataPoints').text(this.currentDataSet.dim[0]);
    this.dom.select('span.dim').text(this.currentDataSet.dim[1]);
    this.showTab('pca', true /* recreateScene */);
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

    // Setup the input text.
    input.onInputChanged((input, inRegexMode) => {
      updateInput(input, inRegexMode);
      this.showCustom();
    });
  }

  private setupUIControls() {
    let self = this;

    // Global tabs
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
      dimension = this.hasPcaZ ? 3 : 2;
      tsneToggle.checked = this.hasPcaZ;
      this.showPCA(() => {
        this.scatterPlot.recreateScene();
      });
    });

    // TSNE controls.
    tsneToggle.addEventListener('change', () => {
      // Make sure PCA stays in the same dimension as tsne.
      this.hasPcaZ = tsneToggle.checked;
      dimension = tsneToggle.checked ? 3 : 2;
      if (this.scatterPlot) {
        this.showTSNE();
        this.scatterPlot.recreateScene();
      }
    });

    this.runTsneButton = this.dom.select('.run-tsne');
    this.runTsneButton.on('click', () => this.runTSNE());
    this.stopTsneButton = this.dom.select('.stop-tsne');
    this.stopTsneButton.on('click', () => {
      this.currentDataSet.stopTSNE();
    });

    let perplexityInput = this.dom.select('.tsne-perplexity input');
    let updatePerplexity = () => {
      perplexity = +perplexityInput.property('value');
      this.dom.select('.tsne-perplexity span').text(perplexity);
    };
    perplexityInput.property('value', perplexity).on('input', updatePerplexity);
    updatePerplexity();

    let learningRateInput = this.dom.select('.tsne-learning-rate input');
    let updateLearningRate = () => {
      let val = +learningRateInput.property('value');
      learningRate = Math.pow(10, val);
      this.dom.select('.tsne-learning-rate span').text(learningRate);
    };
    learningRateInput.property('value', 1).on('input', updateLearningRate);
    updateLearningRate();

    // View controls
    this.querySelector('#reset-zoom').addEventListener('click', () => {
      this.scatterPlot.resetZoom();
    });
    this.querySelector('#zoom-in').addEventListener('click', () => {
      this.scatterPlot.zoomStep(2);
    });
    this.querySelector('#zoom-out').addEventListener('click', () => {
      this.scatterPlot.zoomStep(0.5);
    });

    let selectModeButton = this.querySelector('#selectMode');
    selectModeButton.addEventListener('click', (event) => {
      this.scatterPlot.setMode(
          (selectModeButton as any).active ? Mode.SELECT : Mode.HOVER);
    });
    let nightModeButton = this.querySelector('#nightDayMode');
    nightModeButton.addEventListener('click', () => {
      this.scatterPlot.setDayNightMode((nightModeButton as any).active);
    });

    let labels3DModeButton = this.querySelector('#labels3DMode');
    labels3DModeButton.addEventListener('click', () => {
      this.createVisualizers((labels3DModeButton as any).active);
      this.scatterPlot.recreateScene();
      this.scatterPlot.update();
    });

    // Resize
    window.addEventListener('resize', () => {
      this.scatterPlot.resize();
    });

    // Canvas
    {
      this.scatterPlot = new ScatterPlot(
          this.getScatterContainer(),
          i => '' + this.currentDataSet.points[i].metadata[this.labelOption],
          this);
      this.createVisualizers(false);
    }

    this.scatterPlot.onHover(hoveredIndex => {
      if (hoveredIndex == null) {
        this.highlightedPoints = [];
      } else {
        let point = this.currentDataSet.points[hoveredIndex];
        this.dom.select('#hoverInfo').text(point.metadata[this.labelOption]);
        this.highlightedPoints =
            [{index: hoveredIndex, color: POINT_HIGHLIGHT_COLOR}];
      }
      this.highlightSelectedPointsAndNeighborsInScatterPlot();
    });

    this.scatterPlot.onCameraMove(
        (cameraPosition: THREE.Vector3, cameraTarget: THREE.Vector3) =>
            this.bookmarkPanel.clearStateSelection());

    this.registerSelectionChangedListener(
        (selectedPointIndices: number[],
         neighborsOfFirstPoint: knn.NearestEntry[]) =>
            this.onSelectionChanged(
                selectedPointIndices, neighborsOfFirstPoint));
  }

  private getScatterContainer(): d3.Selection<any> {
    return this.dom.select('#scatter');
  }

  private createVisualizers(inLabels3DMode: boolean) {
    const scatterPlot = this.scatterPlot;
    const selectionContext = this;
    scatterPlot.removeAllVisualizers();

    if (inLabels3DMode) {
      scatterPlot.addVisualizer(
          new ScatterPlotVisualizer3DLabels(selectionContext));
    } else {
      scatterPlot.addVisualizer(
          new ScatterPlotVisualizerSprites(selectionContext));

      scatterPlot.addVisualizer(
          new ScatterPlotVisualizerTraces(selectionContext));

      scatterPlot.addVisualizer(
          new ScatterPlotVisualizerCanvasLabels(this.getScatterContainer()));
    }
  }

  private onSelectionChanged(
      selectedPointIndices: number[],
      neighborsOfFirstPoint: knn.NearestEntry[]) {
    this.dom.select('#hoverInfo')
        .text(`Selected ${selectedPointIndices.length} points`);
    if (neighborsOfFirstPoint && (neighborsOfFirstPoint.length > 0)) {
      this.showTab('inspector');
    }
    this.inspectorPanel.updateInspectorPane(selectedPointIndices,
        neighborsOfFirstPoint);
    this.highlightSelectedPointsAndNeighborsInScatterPlot();
  }

  private showPCA(callback?: () => void) {
    if (this.currentDataSet == null) {
      return;
    }
    this.selectedProjection = 'pca';
    this.currentDataSet.projectPCA().then(() => {
      this.scatterPlot.showTickLabels(false);
      let x = this.pcaX;
      let y = this.pcaY;
      let z = this.pcaZ;
      let hasZ = dimension === 3;
      this.scatterPlot.setPointAccessors(
          i => this.currentDataSet.points[i].projections['pca-' + x],
          i => this.currentDataSet.points[i].projections['pca-' + y],
          hasZ ? (i => this.currentDataSet.points[i].projections['pca-' + z]) :
                 null);
      this.scatterPlot.setAxisLabels('pca-' + x, 'pca-' + y);
      this.scatterPlot.update();
      if (callback) {
        callback();
      }
    });
  }

  private showTab(id: string, recreateScene = false) {
    let tab = this.dom.select('.ink-tab[data-tab="' + id + '"]');
    let pane =
        d3.select((tab.node() as HTMLElement).parentNode.parentNode.parentNode);
    pane.selectAll('.ink-tab').classed('active', false);
    tab.classed('active', true);
    pane.selectAll('.ink-panel-content').classed('active', false);
    pane.select('.ink-panel-content[data-panel="' + id + '"]')
        .classed('active', true);
    if (id === 'pca') {
      this.showPCA(() => {
        if (recreateScene) {
          this.scatterPlot.recreateScene();
        }
      });
    } else if (id === 'tsne') {
      this.showTSNE();
    } else if (id === 'custom') {
      this.showCustom();
    }
  }

  private showCustom() {
    this.selectedProjection = 'custom';
    this.scatterPlot.showTickLabels(true);
    if (this.centroids.xLeft == null || this.centroids.xRight == null ||
        this.centroids.yUp == null || this.centroids.yDown == null) {
      return;
    }
    let xDir = vector.sub(this.centroids.xRight, this.centroids.xLeft);
    this.currentDataSet.projectLinear(xDir, 'linear-x');

    let yDir = vector.sub(this.centroids.yUp, this.centroids.yDown);
    this.currentDataSet.projectLinear(yDir, 'linear-y');

    this.scatterPlot.setPointAccessors(
        i => this.currentDataSet.points[i].projections['linear-x'],
        i => this.currentDataSet.points[i].projections['linear-y'], null);

    let xLabel = this.centroidValues.xLeft + ' → ' + this.centroidValues.xRight;
    let yLabel = this.centroidValues.yUp + ' → ' + this.centroidValues.yDown;
    this.scatterPlot.setAxisLabels(xLabel, yLabel);
    this.scatterPlot.update();
    this.scatterPlot.recreateScene();
  }

  private showTSNE() {
    this.selectedProjection = 'tsne';
    this.scatterPlot.showTickLabels(false);
    this.scatterPlot.setPointAccessors(
        i => this.currentDataSet.points[i].projections['tsne-0'],
        i => this.currentDataSet.points[i].projections['tsne-1'],
        dimension === 3 ?
            (i => this.currentDataSet.points[i].projections['tsne-2']) :
            null);
    this.inspectorPanel.updateInspectorPane([], []);
    this.scatterPlot.setAxisLabels('tsne-0', 'tsne-1');
    if (!this.currentDataSet.hasTSNERun) {
      this.runTSNE();
    } else {
      this.scatterPlot.update();
    }
  }

  private runTSNE() {
    this.runTsneButton.attr('disabled', true);
    this.stopTsneButton.attr('disabled', null);
    this.currentDataSet.projectTSNE(
        perplexity, learningRate, dimension, (iteration: number) => {
          if (iteration != null) {
            this.dom.select('.run-tsne-iter').text(iteration);
            this.scatterPlot.update();
          } else {
            this.runTsneButton.attr('disabled', null);
            this.stopTsneButton.attr('disabled', true);
          }
        });
  }

  private highlightSelectedPointsAndNeighborsInScatterPlot() {
    const selectedAndHighlightedPoints =
        this.highlightedPoints.map(x => x.index).concat(this.selectedPoints);
    const stroke = (i: number) => {
      return i < this.highlightedPoints.length ?
          this.highlightedPoints[i].color :
          NN_HIGHLIGHT_COLOR;
    };
    const favor = (i: number) => {
      return i === 0 || (i < this.highlightedPoints.length ? false : true);
    };
    this.scatterPlot.highlightPoints(
        selectedAndHighlightedPoints, stroke, favor);
  }

  getPcaSampledDim() { return PCA_SAMPLE_DIM.toLocaleString(); }

  getTsneSampleSize() { return SAMPLE_SIZE.toLocaleString(); }

  private getCentroid(pattern: string, inRegexMode: boolean): CentroidResult {
    if (pattern == null || pattern === '') {
      return {numMatches: 0};
    }
    let accessor = (i: number) => this.currentDataSet.points[i].vector;
    let r = this.currentDataSet.query(pattern, inRegexMode, this.labelOption);
    return {
      centroid: vector.centroid(r, accessor),
      numMatches: r.length
    };
  }

  /**
   * Gets the current view of the embedding and saves it as a State object.
   */
  getCurrentState(): State {
    let state: State = {};

    // Save the individual datapoint projections.
    state.projections = [];
    for (let i = 0; i < this.currentDataSet.points.length; i++) {
      state.projections.push(this.currentDataSet.points[i].projections);
    }

    // Save the type of projection.
    state.selectedProjection = this.selectedProjection;

    // Save the selected points.
    state.selectedPoints = this.selectedPoints;

    // Save the camera position and target.
    state.cameraPosition = this.scatterPlot.getCameraPosition();
    state.cameraTarget = this.scatterPlot.getCameraTarget();

    return state;
  }

  /** Loads a State object into the world. */
  loadState(state: State) {
    // Load the individual datapoint projections.
    for (let i = 0; i < state.projections.length; i++) {
      this.currentDataSet.points[i].projections = state.projections[i];
    }

    // Select the type of projection.
    if (state.selectedProjection === 'pca') {
      this.showPCA();
    } else if (state.selectedProjection === 'tsne') {
      this.currentDataSet.hasTSNERun = true;
      this.showTSNE();
    } else if (state.selectedProjection === 'custom') {
      this.showCustom();
    }
    this.showTab(state.selectedProjection);

    // Load the selected points.
    this.selectedPoints = state.selectedPoints;
    this.scatterPlot.clickOnPoint(this.selectedPoints[0]);

    // Load the camera position and target.
    this.scatterPlot.setCameraPositionAndTarget(
        state.cameraPosition, state.cameraTarget);
  }
}

type CentroidResult = {
  centroid?: number[]; numMatches?: number;
};

document.registerElement(Projector.prototype.is, Projector);
