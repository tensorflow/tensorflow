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

import {ColorOption, DataProto, DataSet, MetadataInfo, PointAccessor, Projection, State} from './data';
import {DataProvider, getDataProvider, ServingMode, TensorInfo} from './data-loader';
import {HoverContext, HoverListener} from './hoverContext';
import * as knn from './knn';
import * as logging from './logging';
import {ProjectorScatterPlotAdapter} from './projectorScatterPlotAdapter';
import {Mode, ScatterPlot} from './scatterPlot';
import {ScatterPlotVisualizer3DLabels} from './scatterPlotVisualizer3DLabels';
import {ScatterPlotVisualizerCanvasLabels} from './scatterPlotVisualizerCanvasLabels';
import {ScatterPlotVisualizerSprites} from './scatterPlotVisualizerSprites';
import {ScatterPlotVisualizerTraces} from './scatterPlotVisualizerTraces';
import {SelectionChangedListener, SelectionContext} from './selectionContext';
import {BookmarkPanel} from './vz-projector-bookmark-panel';
import {DataPanel} from './vz-projector-data-panel';
import {InspectorPanel} from './vz-projector-inspector-panel';
import {MetadataCard} from './vz-projector-metadata-card';
import {ProjectionsPanel} from './vz-projector-projections-panel';
// tslint:disable-next-line:no-unused-variable
import {PolymerElement, PolymerHTMLElement} from './vz-projector-util';

/**
 * The minimum number of dimensions the data should have to automatically
 * decide to normalize the data.
 */
const THRESHOLD_DIM_NORMALIZE = 50;
const POINT_COLOR_MISSING = 'black';

export let ProjectorPolymer = PolymerElement({
  is: 'vz-projector',
  properties: {
    routePrefix: String,
    dataProto: {type: String, observer: '_dataProtoChanged'},
    servingMode: String
  }
});

export class Projector extends ProjectorPolymer implements SelectionContext,
                                                           HoverContext {
  // The working subset of the data source's original data set.
  currentDataSet: DataSet;
  servingMode: ServingMode;

  private selectionChangedListeners: SelectionChangedListener[];
  private hoverListeners: HoverListener[];

  private dataSet: DataSet;
  private dom: d3.Selection<any>;
  private projectorScatterPlotAdapter: ProjectorScatterPlotAdapter;
  private scatterPlot: ScatterPlot;
  private dim: number;

  private selectedPointIndices: number[];
  private neighborsOfFirstPoint: knn.NearestEntry[];
  private hoverPointIndex: number;

  private dataProvider: DataProvider;
  private inspectorPanel: InspectorPanel;

  private selectedColorOption: ColorOption;
  private selectedLabelOption: string;
  private routePrefix: string;
  private normalizeData: boolean;
  private selectedProjection: Projection;

  /** Polymer component panels */
  private dataPanel: DataPanel;
  private bookmarkPanel: BookmarkPanel;
  private projectionsPanel: ProjectionsPanel;
  private metadataCard: MetadataCard;

  private statusBar: d3.Selection<HTMLElement>;

  ready() {
    this.selectionChangedListeners = [];
    this.hoverListeners = [];
    this.selectedPointIndices = [];
    this.neighborsOfFirstPoint = [];
    this.dom = d3.select(this);
    logging.setDomContainer(this);
    this.dataPanel = this.$['data-panel'] as DataPanel;
    this.inspectorPanel = this.$['inspector-panel'] as InspectorPanel;
    this.inspectorPanel.initialize(this);
    this.projectionsPanel = this.$['projections-panel'] as ProjectionsPanel;
    this.projectionsPanel.initialize(this);
    this.metadataCard = this.$['metadata-card'] as MetadataCard;
    this.statusBar = this.dom.select('#status-bar');
    this.bookmarkPanel = this.$['bookmark-panel'] as BookmarkPanel;
    this.scopeSubtree(this.$$('#wrapper-notify-msg'), true);
    this.setupUIControls();
    this.initializeDataProvider();
  }

  setSelectedLabelOption(labelOption: string) {
    this.selectedLabelOption = labelOption;
    let labelAccessor = (i: number): string => {
      return this.currentDataSet.points[i]
          .metadata[this.selectedLabelOption] as string;
    };
    this.scatterPlot.setLabelAccessor(labelAccessor);
    this.metadataCard.setLabelOption(this.selectedLabelOption);
  }

  setSelectedColorOption(colorOption: ColorOption) {
    this.selectedColorOption = colorOption;
    this.updateScatterPlot();
  }

  setNormalizeData(normalizeData: boolean) {
    this.normalizeData = normalizeData;
    this.setCurrentDataSet(this.dataSet.getSubset());
  }

  updateDataSet(ds: DataSet, metadata: MetadataInfo) {
    this.dataSet = ds;
    if (this.scatterPlot == null || this.dataSet == null) {
      // We are not ready yet.
      return;
    }
    this.normalizeData = this.dataSet.dim[1] >= THRESHOLD_DIM_NORMALIZE;
    if (metadata != null) {
      ds.mergeMetadata(metadata);
    }
    this.dataPanel.setNormalizeData(this.normalizeData);
    this.setCurrentDataSet(this.dataSet.getSubset());
    this.inspectorPanel.datasetChanged();
    if (metadata != null) {
      this.inspectorPanel.metadataChanged(metadata);
      this.projectionsPanel.metadataChanged(metadata);
    }
    // Set the container to a fixed height, otherwise in Colab the
    // height can grow indefinitely.
    let container = this.dom.select('#container');
    container.style('height', container.property('clientHeight') + 'px');
  }

  setSelectedTensor(run: string, tensorInfo: TensorInfo) {
    this.bookmarkPanel.setSelectedTensor(run, tensorInfo);
  }

  /**
   * Registers a listener to be called any time the selected point set changes.
   */
  registerSelectionChangedListener(listener: SelectionChangedListener) {
    this.selectionChangedListeners.push(listener);
  }

  filterDataset() {
    let indices = this.selectedPointIndices.concat(
        this.neighborsOfFirstPoint.map(n => n.index));
    this.setCurrentDataSet(this.currentDataSet.getSubset(indices));
    this.clearSelectionAndHover();
    this.scatterPlot.recreateScene();
  }

  resetFilterDataset() {
    this.setCurrentDataSet(this.dataSet.getSubset(null));
    this.selectedPointIndices = [];
  }

  /**
   * Used by clients to indicate that a selection has occurred.
   */
  notifySelectionChanged(newSelectedPointIndices: number[]) {
    this.selectedPointIndices = newSelectedPointIndices;
    let neighbors: knn.NearestEntry[] = [];

    if (newSelectedPointIndices.length === 1) {
      neighbors = this.currentDataSet.findNeighbors(
          newSelectedPointIndices[0], this.inspectorPanel.distFunc,
          this.inspectorPanel.numNN);
      this.metadataCard.updateMetadata(
          this.dataSet.points[newSelectedPointIndices[0]].metadata);
    } else {
      this.metadataCard.updateMetadata(null);
    }

    this.selectionChangedListeners.forEach(
        l => l(this.selectedPointIndices, neighbors));
  }

  /**
   * Registers a listener to be called any time the mouse hovers over a point.
   */
  registerHoverListener(listener: HoverListener) {
    this.hoverListeners.push(listener);
  }

  /**
   * Used by clients to indicate that a hover is occurring.
   */
  notifyHoverOverPoint(pointIndex: number) {
    this.hoverListeners.forEach(l => l(pointIndex));
  }

  _dataProtoChanged(dataProtoString: string) {
    let dataProto = dataProtoString ?
        JSON.parse(dataProtoString) as DataProto : null;
    this.initializeDataProvider(dataProto);
  }

  private initializeDataProvider(dataProto?: DataProto) {
    getDataProvider(this.servingMode, dataProto, this.routePrefix,
        dataProvider => {
      this.dataProvider = dataProvider;
      this.dataPanel.initialize(this, dataProvider);
      this.bookmarkPanel.initialize(this, dataProvider);
    });
  }

  private getLegendPointColorer(colorOption: ColorOption):
      (index: number) => string {
    if ((colorOption == null) || (colorOption.map == null)) {
      return null;
    }
    const colorer = (i: number) => {
      let value =
          this.currentDataSet.points[i].metadata[this.selectedColorOption.name];
      if (value == null) {
        return POINT_COLOR_MISSING;
      }
      return colorOption.map(value);
    };
    return colorer;
  }

  private get3DLabelModeButton(): any {
    return this.querySelector('#labels3DMode');
  }

  private get3DLabelMode(): boolean {
    const label3DModeButton = this.get3DLabelModeButton();
    return (label3DModeButton as any).active;
  }

  clearSelectionAndHover() {
    this.notifySelectionChanged([]);
    this.notifyHoverOverPoint(null);
    this.scatterPlot.setMode(Mode.HOVER);
  }

  private unsetCurrentDataSet() {
    this.currentDataSet.stopTSNE();
  }

  private setCurrentDataSet(ds: DataSet) {
    this.clearSelectionAndHover();
    if (this.currentDataSet != null) {
      this.unsetCurrentDataSet();
    }
    this.currentDataSet = ds;
    if (this.normalizeData) {
      this.currentDataSet.normalize();
    }
    this.dim = this.currentDataSet.dim[1];
    this.dom.select('span.numDataPoints').text(this.currentDataSet.dim[0]);
    this.dom.select('span.dim').text(this.currentDataSet.dim[1]);

    this.projectionsPanel.dataSetUpdated(this.currentDataSet, this.dim);

    this.scatterPlot.setDataSet(this.currentDataSet, this.dataSet.spriteImage);
    this.updateScatterPlot();
  }

  private setupUIControls() {
    // View controls
    this.querySelector('#reset-zoom').addEventListener('click', () => {
      this.scatterPlot.resetZoom();
      this.scatterPlot.startOrbitAnimation();
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

    const labels3DModeButton = this.get3DLabelModeButton();
    labels3DModeButton.addEventListener('click', () => {
      this.createVisualizers((labels3DModeButton as any).active);
      this.scatterPlot.recreateScene();
      this.updateScatterPlot();
      this.scatterPlot.update();
    });

    window.addEventListener('resize', () => {
      let container = this.dom.select('#container');
      let parentHeight =
          (container.node().parentNode as HTMLElement).clientHeight;
      container.style('height', parentHeight + 'px');
      this.scatterPlot.resize();
    });

    this.projectorScatterPlotAdapter = new ProjectorScatterPlotAdapter();

    this.scatterPlot = new ScatterPlot(
        this.getScatterContainer(), i => '' +
            this.currentDataSet.points[i].metadata[this.selectedLabelOption],
        this, this);
    this.createVisualizers(false);

    this.scatterPlot.onCameraMove(
        (cameraPosition: THREE.Vector3, cameraTarget: THREE.Vector3) =>
            this.bookmarkPanel.clearStateSelection());

    this.registerHoverListener(
        (hoverIndex: number) => this.onHover(hoverIndex));

    this.registerSelectionChangedListener(
        (selectedPointIndices: number[],
         neighborsOfFirstPoint: knn.NearestEntry[]) =>
            this.onSelectionChanged(
                selectedPointIndices, neighborsOfFirstPoint));
  }

  private onHover(hoverIndex: number) {
    this.hoverPointIndex = hoverIndex;
    let hoverText = null;
    if (hoverIndex != null) {
      const point = this.currentDataSet.points[hoverIndex];
      if (point.metadata[this.selectedLabelOption]) {
        hoverText = point.metadata[this.selectedLabelOption].toString();
      }
    }
    this.updateScatterPlot();
    if (this.selectedPointIndices.length === 0) {
      this.statusBar.style('display', hoverText ? null : 'none');
      this.statusBar.text(hoverText);
    }
  }

  private updateScatterPlot() {
    const dataSet = this.currentDataSet;
    const selectedSet = this.selectedPointIndices;
    const hoverIndex = this.hoverPointIndex;
    const neighbors = this.neighborsOfFirstPoint;
    const pointColorer = this.getLegendPointColorer(this.selectedColorOption);

    const pointColors =
        this.projectorScatterPlotAdapter.generatePointColorArray(
            dataSet, pointColorer, selectedSet, neighbors, hoverIndex,
            this.get3DLabelMode());
    const pointScaleFactors =
        this.projectorScatterPlotAdapter.generatePointScaleFactorArray(
            dataSet, selectedSet, neighbors, hoverIndex);
    const labels =
        this.projectorScatterPlotAdapter.generateVisibleLabelRenderParams(
            dataSet, selectedSet, neighbors, hoverIndex);

    this.scatterPlot.setPointColors(pointColors);
    this.scatterPlot.setPointScaleFactors(pointScaleFactors);
    this.scatterPlot.setLabels(labels);
    this.scatterPlot.render();
  }

  private getScatterContainer(): d3.Selection<any> {
    return this.dom.select('#scatter');
  }

  private createVisualizers(inLabels3DMode: boolean) {
    const scatterPlot = this.scatterPlot;
    const selectionContext = this;
    scatterPlot.removeAllVisualizers();

    if (inLabels3DMode) {
      scatterPlot.addVisualizer(new ScatterPlotVisualizer3DLabels());
    } else {
      scatterPlot.addVisualizer(new ScatterPlotVisualizerSprites());
      scatterPlot.addVisualizer(
          new ScatterPlotVisualizerTraces(selectionContext));
      scatterPlot.addVisualizer(
          new ScatterPlotVisualizerCanvasLabels(this.getScatterContainer()));
    }
  }

  private onSelectionChanged(
      selectedPointIndices: number[],
      neighborsOfFirstPoint: knn.NearestEntry[]) {
    this.selectedPointIndices = selectedPointIndices;
    this.neighborsOfFirstPoint = neighborsOfFirstPoint;
    let totalNumPoints =
        this.selectedPointIndices.length + neighborsOfFirstPoint.length;
    this.statusBar.text(`Selected ${totalNumPoints} points`)
        .style('display', totalNumPoints > 0 ? null : 'none');
    this.inspectorPanel.updateInspectorPane(
        selectedPointIndices, neighborsOfFirstPoint);
    this.updateScatterPlot();
  }

  setProjection(
      projection: Projection, dimensionality: number,
      pointAccessors: [PointAccessor, PointAccessor, PointAccessor]) {
    this.selectedProjection = projection;
    this.scatterPlot.setDimensions(dimensionality);
    this.scatterPlot.showTickLabels(false);
    this.scatterPlot.setPointAccessors(pointAccessors);

    /* tsne needs to do an iteration for the points to look reasonable */
    if (projection !== 'tsne') {
      this.scatterPlot.update();
    }

    this.scatterPlot.recreateScene();
    this.scatterPlot.setCameraDefForNextCameraCreation(null);
  }

  notifyProjectionsUpdated() {
    this.scatterPlot.update();
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

    state.selectedProjection = this.selectedProjection;
    state.is3d = this.projectionsPanel.is3d;
    if (this.selectedProjection === 'pca') {
      state.componentDimensions =
          this.projectionsPanel.getPCAComponentUIValues();
    }
    state.selectedPoints = this.selectedPointIndices;
    state.cameraDef = this.scatterPlot.getCameraDef();

    // Save the color and label by options.
    state.selectedColorOptionName = this.dataPanel.selectedColorOptionName;
    state.selectedLabelOption = this.selectedLabelOption;

    return state;
  }

  /** Loads a State object into the world. */
  loadState(state: State) {
    for (let i = 0; i < state.projections.length; i++) {
      this.currentDataSet.points[i].projections = state.projections[i];
    }
    if (state.selectedProjection === 'tsne') {
      this.currentDataSet.hasTSNERun = true;
    }

    this.projectionsPanel.disablePolymerChangesTriggerReprojection();
    this.projectionsPanel.is3d = state.is3d;
    if (state.selectedProjection === 'pca') {
      this.projectionsPanel.setPCAComponentUIValues(state.componentDimensions);
    }
    this.projectionsPanel.showTab(state.selectedProjection);
    this.projectionsPanel.enablePolymerChangesTriggerReprojection();

    // Load the color and label by options.
    this.dataPanel.selectedColorOptionName = state.selectedColorOptionName;
    this.selectedLabelOption = state.selectedLabelOption;

    this.scatterPlot.setCameraDefForNextCameraCreation(state.cameraDef);

    {
      const accessors = this.currentDataSet.getPointAccessors(
          state.selectedProjection, state.componentDimensions);
      this.setProjection(
          state.selectedProjection, state.is3d ? 3 : 2, accessors);
    }

    this.notifySelectionChanged(state.selectedPoints);
  }
}

document.registerElement(Projector.prototype.is, Projector);
