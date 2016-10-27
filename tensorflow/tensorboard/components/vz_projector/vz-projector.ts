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

import {ColorOption, DataPoint, DataProto, DataSet, MetadataInfo, PointAccessor, PointMetadata, Projection, State, stateGetAccessorDimensions} from './data';
import {DataProvider, ServingMode, TensorInfo} from './data-provider';
import {DemoDataProvider} from './data-provider-demo';
import {ProtoDataProvider} from './data-provider-proto';
import {ServerDataProvider} from './data-provider-server';
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

const INDEX_METADATA_FIELD = '__index__';

export class Projector extends ProjectorPolymer implements SelectionContext,
                                                           HoverContext {
  // The working subset of the data source's original data set.
  dataSet: DataSet;
  servingMode: ServingMode;

  private selectionChangedListeners: SelectionChangedListener[];
  private hoverListeners: HoverListener[];

  private originalDataSet: DataSet;
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
      return this.dataSet.points[i]
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
    this.setCurrentDataSet(this.originalDataSet.getSubset());
  }

  updateDataSet(ds: DataSet, metadata?: MetadataInfo, metadataFile?: string) {
    this.originalDataSet = ds;
    if (this.scatterPlot == null || this.originalDataSet == null) {
      // We are not ready yet.
      return;
    }
    this.normalizeData = this.originalDataSet.dim[1] >= THRESHOLD_DIM_NORMALIZE;
    metadata = metadata || this.makeDefaultMetadata(ds.points);
    ds.mergeMetadata(metadata);
    this.dataPanel.setNormalizeData(this.normalizeData);
    this.setCurrentDataSet(this.originalDataSet.getSubset());
    this.inspectorPanel.datasetChanged();

    this.inspectorPanel.metadataChanged(metadata);
    this.projectionsPanel.metadataChanged(metadata);
    this.dataPanel.metadataChanged(metadata, metadataFile);
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
    let selectionSize = this.selectedPointIndices.length;
    this.setCurrentDataSet(this.dataSet.getSubset(indices));
    this.adjustSelectionAndHover(d3.range(selectionSize));
    this.scatterPlot.recreateScene();
  }

  resetFilterDataset() {
    let originalPointIndices = this.selectedPointIndices.map(localIndex => {
      return this.dataSet.points[localIndex].index;
    });
    this.setCurrentDataSet(this.originalDataSet.getSubset());
    this.adjustSelectionAndHover(originalPointIndices);
  }

  /**
   * Used by clients to indicate that a selection has occurred.
   */
  notifySelectionChanged(newSelectedPointIndices: number[]) {
    this.selectedPointIndices = newSelectedPointIndices;
    let neighbors: knn.NearestEntry[] = [];

    if (newSelectedPointIndices.length === 1) {
      neighbors = this.dataSet.findNeighbors(
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

  private makeDefaultMetadata(points: DataPoint[]): MetadataInfo {
    let pointsInfo: PointMetadata[] = [];
    points.forEach(p => {
      let pointInfo: PointMetadata = {};
      pointInfo[INDEX_METADATA_FIELD] = p.index;
      pointsInfo.push(pointInfo);
    });
    return {
      stats: [{
        name: INDEX_METADATA_FIELD,
        isNumeric: false,
        tooManyUniqueValues: true,
        min: 0,
        max: pointsInfo.length - 1
      }],
      pointsInfo: pointsInfo
    };
  }

  private initializeDataProvider(dataProto?: DataProto) {
    if (this.servingMode === 'demo') {
      this.dataProvider = new DemoDataProvider();
    } else if (this.servingMode === 'server') {
      if (!this.routePrefix) {
        throw 'route-prefix is a required parameter';
      }
      this.dataProvider = new ServerDataProvider(this.routePrefix);
    } else if (this.servingMode === 'proto' && dataProto != null) {
      this.dataProvider = new ProtoDataProvider(dataProto);
    }

    this.dataPanel.initialize(this, this.dataProvider);
    this.bookmarkPanel.initialize(this, this.dataProvider);
  }

  private getLegendPointColorer(colorOption: ColorOption):
      (index: number) => string {
    if ((colorOption == null) || (colorOption.map == null)) {
      return null;
    }
    const colorer = (i: number) => {
      let value =
          this.dataSet.points[i].metadata[this.selectedColorOption.name];
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

  adjustSelectionAndHover(selectedPointIndices: number[], hoverIndex?: number) {
    this.notifySelectionChanged(selectedPointIndices);
    this.notifyHoverOverPoint(hoverIndex);
    this.scatterPlot.setMode(Mode.HOVER);
  }

  private unsetCurrentDataSet() {
    this.dataSet.stopTSNE();
  }

  private setCurrentDataSet(ds: DataSet) {
    this.adjustSelectionAndHover([]);
    if (this.dataSet != null) {
      this.unsetCurrentDataSet();
    }
    this.dataSet = ds;
    if (this.normalizeData) {
      this.dataSet.normalize();
    }
    this.dim = this.dataSet.dim[1];
    this.dom.select('span.numDataPoints').text(this.dataSet.dim[0]);
    this.dom.select('span.dim').text(this.dataSet.dim[1]);

    this.projectionsPanel.dataSetUpdated(
        this.dataSet, this.originalDataSet, this.dim);

    this.scatterPlot.setDataSet(this.dataSet, this.originalDataSet.spriteImage);
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
            this.dataSet.points[i].metadata[this.selectedLabelOption],
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
      const point = this.dataSet.points[hoverIndex];
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
    const dataSet = this.dataSet;
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
    const traceColors =
        this.projectorScatterPlotAdapter.generateLineSegmentColorMap(
            dataSet, pointColorer);

    this.scatterPlot.setPointColors(pointColors);
    this.scatterPlot.setPointScaleFactors(pointScaleFactors);
    this.scatterPlot.setLabels(labels);
    this.scatterPlot.setTraceColors(traceColors);
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

    if (this.dataSet.hasMeaningfulVisualization(projection)) {
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
    const state = new State();

    // Save the individual datapoint projections.
    state.projections = [];
    for (let i = 0; i < this.dataSet.points.length; i++) {
      const point = this.dataSet.points[i];
      const projections: {[key: string]: number} = {};
      const keys = Object.keys(point.projections);
      for (let j = 0; j < keys.length; ++j) {
        projections[keys[j]] = point.projections[keys[j]];
      }
      state.projections.push(projections);
    }
    state.selectedProjection = this.selectedProjection;
    state.tSNEIteration = this.dataSet.tSNEIteration;
    state.selectedPoints = this.selectedPointIndices;
    state.cameraDef = this.scatterPlot.getCameraDef();
    state.selectedColorOptionName = this.dataPanel.selectedColorOptionName;
    state.selectedLabelOption = this.selectedLabelOption;
    this.projectionsPanel.populateBookmarkFromUI(state);
    return state;
  }

  /** Loads a State object into the world. */
  loadState(state: State) {
    for (let i = 0; i < state.projections.length; i++) {
      const point = this.dataSet.points[i];
      const projection = state.projections[i];
      const keys = Object.keys(projection);
      for (let j = 0; j < keys.length; ++j) {
        point.projections[keys[j]] = projection[keys[j]];
      }
    }
    this.dataSet.hasTSNERun = (state.selectedProjection === 'tsne');
    this.dataSet.tSNEIteration = state.tSNEIteration;
    this.projectionsPanel.restoreUIFromBookmark(state);
    this.dataPanel.selectedColorOptionName = state.selectedColorOptionName;
    this.selectedLabelOption = state.selectedLabelOption;
    this.scatterPlot.setCameraDefForNextCameraCreation(state.cameraDef);
    {
      const dimensions = stateGetAccessorDimensions(state);
      const accessors =
          this.dataSet.getPointAccessors(state.selectedProjection, dimensions);
      this.setProjection(
          state.selectedProjection, dimensions.length, accessors);
    }
    this.notifySelectionChanged(state.selectedPoints);
  }
}

document.registerElement(Projector.prototype.is, Projector);
