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

import {AnalyticsLogger} from './analyticsLogger';
import * as data from './data';
import {ColorOption, ColumnStats, DataPoint, DataProto, DataSet, DistanceFunction, PointMetadata, Projection, SpriteAndMetadataInfo, State, stateGetAccessorDimensions} from './data';
import {DataProvider, EmbeddingInfo, ServingMode} from './data-provider';
import {DemoDataProvider} from './data-provider-demo';
import {ProtoDataProvider} from './data-provider-proto';
import {ServerDataProvider} from './data-provider-server';
import * as knn from './knn';
import * as logging from './logging';
import {DistanceMetricChangedListener, HoverListener, ProjectionChangedListener, ProjectorEventContext, SelectionChangedListener} from './projectorEventContext';
import {ProjectorScatterPlotAdapter} from './projectorScatterPlotAdapter';
import {MouseMode} from './scatterPlot';
import * as util from './util';
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
    servingMode: String,
    projectorConfigJsonPath: String,
    pageViewLogging: Boolean,
    eventLogging: Boolean
  }
});

const INDEX_METADATA_FIELD = '__index__';

export class Projector extends ProjectorPolymer implements
    ProjectorEventContext {
  // The working subset of the data source's original data set.
  dataSet: DataSet;
  servingMode: ServingMode;
  // The path to the projector config JSON file for demo mode.
  projectorConfigJsonPath: string;

  private selectionChangedListeners: SelectionChangedListener[];
  private hoverListeners: HoverListener[];
  private projectionChangedListeners: ProjectionChangedListener[];
  private distanceMetricChangedListeners: DistanceMetricChangedListener[];

  private originalDataSet: DataSet;
  private dataSetBeforeFilter: DataSet;
  private dom: d3.Selection<any>;
  private projectorScatterPlotAdapter: ProjectorScatterPlotAdapter;
  private dim: number;

  private dataSetFilterIndices: number[];
  private selectedPointIndices: number[];
  private neighborsOfFirstPoint: knn.NearestEntry[];
  private hoverPointIndex: number;

  private dataProvider: DataProvider;
  private inspectorPanel: InspectorPanel;

  private selectedColorOption: ColorOption;
  private selectedLabelOption: string;
  private routePrefix: string;
  private normalizeData: boolean;
  private projection: Projection;

  /** Polymer component panels */
  private dataPanel: DataPanel;
  private bookmarkPanel: BookmarkPanel;
  private projectionsPanel: ProjectionsPanel;
  private metadataCard: MetadataCard;

  private statusBar: d3.Selection<HTMLElement>;
  private analyticsLogger: AnalyticsLogger;
  private eventLogging: boolean;
  private pageViewLogging: boolean;

  ready() {
    this.dom = d3.select(this);
    logging.setDomContainer(this);

    this.analyticsLogger =
        new AnalyticsLogger(this.pageViewLogging, this.eventLogging);
    this.analyticsLogger.logPageView('embeddings');

    if (!util.hasWebGLSupport()) {
      this.analyticsLogger.logWebGLDisabled();
      logging.setErrorMessage(
          'Your browser or device does not have WebGL enabled. Please enable ' +
          'hardware acceleration, or use a browser that supports WebGL.');
      return;
    }

    this.selectionChangedListeners = [];
    this.hoverListeners = [];
    this.projectionChangedListeners = [];
    this.distanceMetricChangedListeners = [];
    this.selectedPointIndices = [];
    this.neighborsOfFirstPoint = [];

    this.dataPanel = this.$['data-panel'] as DataPanel;
    this.inspectorPanel = this.$['inspector-panel'] as InspectorPanel;
    this.inspectorPanel.initialize(this, this as ProjectorEventContext);
    this.projectionsPanel = this.$['projections-panel'] as ProjectionsPanel;
    this.projectionsPanel.initialize(this);
    this.bookmarkPanel = this.$['bookmark-panel'] as BookmarkPanel;
    this.bookmarkPanel.initialize(this, this as ProjectorEventContext);
    this.metadataCard = this.$['metadata-card'] as MetadataCard;
    this.statusBar = this.dom.select('#status-bar');
    this.scopeSubtree(this.$$('#notification-dialog'), true);
    this.setupUIControls();
    this.initializeDataProvider();
  }

  setSelectedLabelOption(labelOption: string) {
    this.selectedLabelOption = labelOption;
    this.metadataCard.setLabelOption(this.selectedLabelOption);
    this.projectorScatterPlotAdapter.setLabelPointAccessor(labelOption);
    this.projectorScatterPlotAdapter.updateScatterPlotAttributes();
    this.projectorScatterPlotAdapter.render();
  }

  setSelectedColorOption(colorOption: ColorOption) {
    this.selectedColorOption = colorOption;
    this.projectorScatterPlotAdapter.setLegendPointColorer(
        this.getLegendPointColorer(colorOption));
    this.projectorScatterPlotAdapter.updateScatterPlotAttributes();
    this.projectorScatterPlotAdapter.render();
  }

  setNormalizeData(normalizeData: boolean) {
    this.normalizeData = normalizeData;
    this.setCurrentDataSet(this.originalDataSet.getSubset());
  }

  updateDataSet(
      ds: DataSet, spriteAndMetadata?: SpriteAndMetadataInfo,
      metadataFile?: string) {
    this.dataSetFilterIndices = null;
    this.originalDataSet = ds;
    if (ds != null) {
      this.normalizeData =
          this.originalDataSet.dim[1] >= THRESHOLD_DIM_NORMALIZE;
      spriteAndMetadata = spriteAndMetadata || {};
      if (spriteAndMetadata.pointsInfo == null) {
        let [pointsInfo, stats] = this.makeDefaultPointsInfoAndStats(ds.points);
        spriteAndMetadata.pointsInfo = pointsInfo;
        spriteAndMetadata.stats = stats;
      }
      ds.mergeMetadata(spriteAndMetadata);
    }
    if (this.projectorScatterPlotAdapter != null) {
      if (ds == null) {
        this.projectorScatterPlotAdapter.setLabelPointAccessor(null);
        this.setProjection(null);
      } else {
        this.projectorScatterPlotAdapter.updateScatterPlotPositions();
        this.projectorScatterPlotAdapter.updateScatterPlotAttributes();
        this.projectorScatterPlotAdapter.resize();
        this.projectorScatterPlotAdapter.render();
      }
    }
    if (ds != null) {
      this.dataPanel.setNormalizeData(this.normalizeData);
      this.setCurrentDataSet(ds.getSubset());
      this.projectorScatterPlotAdapter.setLabelPointAccessor(
          this.selectedLabelOption);
      this.inspectorPanel.datasetChanged();

      this.inspectorPanel.metadataChanged(spriteAndMetadata);
      this.projectionsPanel.metadataChanged(spriteAndMetadata);
      this.dataPanel.metadataChanged(spriteAndMetadata, metadataFile);
      // Set the container to a fixed height, otherwise in Colab the
      // height can grow indefinitely.
      let container = this.dom.select('#container');
      container.style('height', container.property('clientHeight') + 'px');
    } else {
      this.setCurrentDataSet(null);
    }
  }

  setSelectedTensor(run: string, tensorInfo: EmbeddingInfo) {
    this.bookmarkPanel.setSelectedTensor(run, tensorInfo, this.dataProvider);
  }

  /**
   * Registers a listener to be called any time the selected point set changes.
   */
  registerSelectionChangedListener(listener: SelectionChangedListener) {
    this.selectionChangedListeners.push(listener);
  }

  filterDataset(pointIndices: number[]) {
    const selectionSize = this.selectedPointIndices.length;
    if (this.dataSetBeforeFilter == null) {
      this.dataSetBeforeFilter = this.dataSet;
    }
    this.setCurrentDataSet(this.dataSet.getSubset(pointIndices));
    this.dataSetFilterIndices = pointIndices;
    this.projectorScatterPlotAdapter.updateScatterPlotPositions();
    this.projectorScatterPlotAdapter.updateScatterPlotAttributes();
    this.adjustSelectionAndHover(d3.range(selectionSize));
  }

  resetFilterDataset() {
    const originalPointIndices = this.selectedPointIndices.map(
        filteredIndex => this.dataSet.points[filteredIndex].index);
    this.setCurrentDataSet(this.dataSetBeforeFilter);
    if (this.projection != null) {
      this.projection.dataSet = this.dataSetBeforeFilter;
    }
    this.dataSetBeforeFilter = null;
    this.projectorScatterPlotAdapter.updateScatterPlotPositions();
    this.projectorScatterPlotAdapter.updateScatterPlotAttributes();
    this.dataSetFilterIndices = [];
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

  registerProjectionChangedListener(listener: ProjectionChangedListener) {
    this.projectionChangedListeners.push(listener);
  }

  notifyProjectionChanged(projection: Projection) {
    this.projectionChangedListeners.forEach(l => l(projection));
  }

  registerDistanceMetricChangedListener(l: DistanceMetricChangedListener) {
    this.distanceMetricChangedListeners.push(l);
  }

  notifyDistanceMetricChanged(distMetric: DistanceFunction) {
    this.distanceMetricChangedListeners.forEach(l => l(distMetric));
  }

  _dataProtoChanged(dataProtoString: string) {
    let dataProto =
        dataProtoString ? JSON.parse(dataProtoString) as DataProto : null;
    this.initializeDataProvider(dataProto);
  }

  private makeDefaultPointsInfoAndStats(points: DataPoint[]):
      [PointMetadata[], ColumnStats[]] {
    let pointsInfo: PointMetadata[] = [];
    points.forEach(p => {
      let pointInfo: PointMetadata = {};
      pointInfo[INDEX_METADATA_FIELD] = p.index;
      pointsInfo.push(pointInfo);
    });
    let stats: ColumnStats[] = [{
      name: INDEX_METADATA_FIELD,
      isNumeric: false,
      tooManyUniqueValues: true,
      min: 0,
      max: pointsInfo.length - 1
    }];
    return [pointsInfo, stats];
  }

  private initializeDataProvider(dataProto?: DataProto) {
    if (this.servingMode === 'demo') {
      let projectorConfigUrl: string;

      // Only in demo mode do we allow the config being passed via URL.
      let urlParams = util.getURLParams(window.location.search);
      if ('config' in urlParams) {
        projectorConfigUrl = urlParams['config'];
      } else {
        projectorConfigUrl = this.projectorConfigJsonPath;
      }
      this.dataProvider = new DemoDataProvider(projectorConfigUrl);
    } else if (this.servingMode === 'server') {
      if (!this.routePrefix) {
        throw 'route-prefix is a required parameter';
      }
      this.dataProvider = new ServerDataProvider(this.routePrefix);
    } else if (this.servingMode === 'proto' && dataProto != null) {
      this.dataProvider = new ProtoDataProvider(dataProto);
    }

    this.dataPanel.initialize(this, this.dataProvider);
  }

  private getLegendPointColorer(colorOption: ColorOption):
      (ds: DataSet, index: number) => string {
    if ((colorOption == null) || (colorOption.map == null)) {
      return null;
    }
    const colorer = (ds: DataSet, i: number) => {
      let value = ds.points[i].metadata[this.selectedColorOption.name];
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
    this.setMouseMode(MouseMode.CAMERA_AND_CLICK_SELECT);
  }

  private setMouseMode(mouseMode: MouseMode) {
    let selectModeButton = this.querySelector('#selectMode');
    (selectModeButton as any).active = (mouseMode === MouseMode.AREA_SELECT);
    this.projectorScatterPlotAdapter.scatterPlot.setMouseMode(mouseMode);
  }

  private setCurrentDataSet(ds: DataSet) {
    this.adjustSelectionAndHover([]);
    if (this.dataSet != null) {
      this.dataSet.stopTSNE();
    }
    if ((ds != null) && this.normalizeData) {
      ds.normalize();
    }
    this.dim = (ds == null) ? 0 : ds.dim[1];
    this.dom.select('span.numDataPoints').text((ds == null) ? '0' : ds.dim[0]);
    this.dom.select('span.dim').text((ds == null) ? '0' : ds.dim[1]);

    this.dataSet = ds;

    this.projectionsPanel.dataSetUpdated(
        this.dataSet, this.originalDataSet, this.dim);

    this.projectorScatterPlotAdapter.setDataSet(this.dataSet);
    this.projectorScatterPlotAdapter.scatterPlot
        .setCameraParametersForNextCameraCreation(null, true);
  }

  private setupUIControls() {
    // View controls
    this.querySelector('#reset-zoom').addEventListener('click', () => {
      this.projectorScatterPlotAdapter.scatterPlot.resetZoom();
      this.projectorScatterPlotAdapter.scatterPlot.startOrbitAnimation();
    });

    let selectModeButton = this.querySelector('#selectMode');
    selectModeButton.addEventListener('click', (event) => {
      this.setMouseMode(
          (selectModeButton as any).active ? MouseMode.AREA_SELECT :
                                             MouseMode.CAMERA_AND_CLICK_SELECT);
    });
    let nightModeButton = this.querySelector('#nightDayMode');
    nightModeButton.addEventListener('click', () => {
      this.projectorScatterPlotAdapter.scatterPlot.setDayNightMode(
          (nightModeButton as any).active);
    });

    const labels3DModeButton = this.get3DLabelModeButton();
    labels3DModeButton.addEventListener('click', () => {
      this.projectorScatterPlotAdapter.set3DLabelMode(this.get3DLabelMode());
    });

    window.addEventListener('resize', () => {
      let container = this.dom.select('#container');
      let parentHeight =
          (container.node().parentNode as HTMLElement).clientHeight;
      container.style('height', parentHeight + 'px');
      this.projectorScatterPlotAdapter.resize();
    });

    {
      this.projectorScatterPlotAdapter = new ProjectorScatterPlotAdapter(
          this.getScatterContainer(), this as ProjectorEventContext);
      this.projectorScatterPlotAdapter.setLabelPointAccessor(
          this.selectedLabelOption);
    }

    this.projectorScatterPlotAdapter.scatterPlot.onCameraMove(
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
    if (this.selectedPointIndices.length === 0) {
      this.statusBar.style('display', hoverText ? null : 'none');
      this.statusBar.text(hoverText);
    }
  }

  private getScatterContainer(): d3.Selection<any> {
    return this.dom.select('#scatter');
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
  }

  setProjection(projection: Projection) {
    this.projection = projection;
    if (projection != null) {
      this.analyticsLogger.logProjectionChanged(projection.projectionType);
    }
    this.notifyProjectionChanged(projection);
  }

  notifyProjectionPositionsUpdated() {
    this.projectorScatterPlotAdapter.notifyProjectionPositionsUpdated();
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
    state.selectedProjection = this.projection.projectionType;
    state.dataSetDimensions = this.dataSet.dim;
    state.tSNEIteration = this.dataSet.tSNEIteration;
    state.selectedPoints = this.selectedPointIndices;
    state.filteredPoints = this.dataSetFilterIndices;
    this.projectorScatterPlotAdapter.populateBookmarkFromUI(state);
    state.selectedColorOptionName = this.dataPanel.selectedColorOptionName;
    state.selectedLabelOption = this.selectedLabelOption;
    this.projectionsPanel.populateBookmarkFromUI(state);
    return state;
  }

  /** Loads a State object into the world. */
  loadState(state: State) {
    this.setProjection(null);
    {
      this.projectionsPanel.disablePolymerChangesTriggerReprojection();
      if (this.dataSetBeforeFilter != null) {
        this.resetFilterDataset();
      }
      if (state.filteredPoints != null) {
        this.filterDataset(state.filteredPoints);
      }
      this.projectionsPanel.enablePolymerChangesTriggerReprojection();
    }
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
    this.inspectorPanel.restoreUIFromBookmark(state);
    this.dataPanel.selectedColorOptionName = state.selectedColorOptionName;
    this.selectedLabelOption = state.selectedLabelOption;
    this.projectorScatterPlotAdapter.restoreUIFromBookmark(state);
    {
      const dimensions = stateGetAccessorDimensions(state);
      const components =
          data.getProjectionComponents(state.selectedProjection, dimensions);
      const projection = new Projection(
          state.selectedProjection, components, dimensions.length,
          this.dataSet);
      this.setProjection(projection);
    }
    this.notifySelectionChanged(state.selectedPoints);
  }
}

document.registerElement(Projector.prototype.is, Projector);
