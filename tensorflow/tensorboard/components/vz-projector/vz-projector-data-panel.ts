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

import {ColorOption} from './data';
import {CheckpointInfo, ColumnStats, DataProvider, parseRawMetadata, parseRawTensors} from './data-loader';
import {Projector} from './vz-projector';
import {ColorLegendRenderInfo, ColorLegendThreshold} from './vz-projector-legend';
// tslint:disable-next-line:no-unused-variable
import {PolymerElement, PolymerHTMLElement} from './vz-projector-util';

export let DataPanelPolymer = PolymerElement({
  is: 'vz-projector-data-panel',
  properties: {
    selectedTensor: {type: String, observer: '_selectedTensorChanged'},
    selectedRun: {type: String, observer: '_selectedRunChanged'},
    colorOption: {type: Object, notify: true, observer: '_colorOptionChanged'},
    labelOption: {type: String, notify: true},
    normalizeData: Boolean
  }
});

export class DataPanel extends DataPanelPolymer {
  labelOption: string;
  colorOption: ColorOption;

  private normalizeData: boolean;
  private labelOptions: string[];
  private colorOptions: ColorOption[];
  private dom: d3.Selection<any>;

  private selectedTensor: string;
  private selectedRun: string;
  private dataProvider: DataProvider;
  private tensorNames: {name: string, shape: number[]}[];
  private runNames: string[];
  private projector: Projector;
  private checkpointInfo: CheckpointInfo;
  private colorLegendRenderInfo: ColorLegendRenderInfo;

  ready() {
    this.dom = d3.select(this);
    this.normalizeData = true;
  }

  initialize(projector: Projector, dp: DataProvider) {
    this.projector = projector;
    this.dataProvider = dp;
    this.setupUploadButtons();

    // Tell the projector whenever the data normalization changes.
    // Unknown why, but the polymer checkbox button stops working as soon as
    // you do d3.select() on it.
    this.querySelector('#normalize-data-checkbox')
        .addEventListener('change', () => {
          this.projector.setNormalizeData(this.normalizeData);
        });

    // Get all the runs.
    this.dataProvider.getRuns(runs => {
      this.runNames = runs;
      // If there is only 1 run, choose that one by default.
      if (this.runNames.length === 1) {
        this.selectedRun = runs[0];
      }
    });
  }

  getSeparatorClass(isSeparator: boolean): string {
    return isSeparator ? 'separator' : null;
  }

  updateMetadataUI(columnStats: ColumnStats[], metadataFile: string) {
    this.dom.select('#metadata-file')
        .text(metadataFile)
        .attr('title', metadataFile);
    // Label by options.
    let labelIndex = -1;
    this.labelOptions = columnStats.length > 1 ? columnStats.map((stats, i) => {
      // Make the default label by the first non-numeric column.
      if (!stats.isNumeric && labelIndex === -1) {
        labelIndex = i;
      }
      return stats.name;
    }) :
                                                 ['label'];
    this.labelOption = this.labelOptions[Math.max(0, labelIndex)];

    // Color by options.
    let standardColorOption: ColorOption[] = [
      {name: 'No color map'},
      // TODO(smilkov): Implement this.
      // {name: 'Distance of neighbors',
      //    desc: 'How far is each point from its neighbors'}
    ];
    let metadataColorOption: ColorOption[] =
        columnStats
            .filter(stats => {
              return !stats.tooManyUniqueValues || stats.isNumeric;
            })
            .map(stats => {
              let map: (v: string|number) => string;
              let items: {label: string, count: number}[];
              let thresholds: ColorLegendThreshold[];
              if (!stats.tooManyUniqueValues) {
                let scale = d3.scale.category20();
                let range = scale.range();
                // Re-order the range.
                let newRange = range.map((color, i) => {
                  let index = (i * 2) % (range.length - 1);
                  if (index === 0) {
                    index = range.length - 1;
                  }
                  return range[index];
                });
                items = stats.uniqueEntries;
                scale.range(newRange).domain(items.map(x => x.label));
                map = scale;
              } else {
                thresholds = [
                  {color: '#ffffdd', value: stats.min},
                  {color: '#1f2d86', value: stats.max}
                ];
                map = d3.scale.linear<string>()
                          .domain(thresholds.map(t => t.value))
                          .range(thresholds.map(t => t.color));
              }
              let desc = stats.tooManyUniqueValues ?
                  'gradient' :
                  stats.uniqueEntries.length + ' colors';
              return {name: stats.name, desc, map, items, thresholds};
            });
    if (metadataColorOption.length > 0) {
      // Add a separator line between built-in color maps
      // and those based on metadata columns.
      standardColorOption.push({name: 'Metadata', isSeparator: true});
    }
    this.colorOptions = standardColorOption.concat(metadataColorOption);
    this.colorOption = this.colorOptions[0];
  }

  setNormalizeData(normalizeData: boolean) {
    this.normalizeData = normalizeData;
  }

  _selectedTensorChanged() {
    if (this.selectedTensor == null) {
      return;
    }
    this.dataProvider.getTensor(this.selectedRun, this.selectedTensor, ds => {
      let metadataFile =
          this.checkpointInfo.tensors[this.selectedTensor].metadataFile;
      if (metadataFile) {
        this.dataProvider.getMetadata(
            this.selectedRun, ds, this.selectedTensor, stats => {
              this.projector.updateDataSet(ds);
              this.updateMetadataUI(stats, metadataFile);
            });
      } else {
        this.projector.updateDataSet(ds);
      }
    });
  }

  _selectedRunChanged() {
    this.dataProvider.getCheckpointInfo(this.selectedRun, info => {
      this.checkpointInfo = info;
      let names =
          Object.keys(this.checkpointInfo.tensors)
              .filter(name => {
                let shape = this.checkpointInfo.tensors[name].shape;
                return shape.length === 2 && shape[0] > 1 && shape[1] > 1;
              })
              .sort((a, b) => {
                let sizeA = this.checkpointInfo.tensors[a].shape[0];
                let sizeB = this.checkpointInfo.tensors[b].shape[0];
                if (sizeA === sizeB) {
                  // If the same dimension, sort alphabetically by tensor
                  // name.
                  return a <= b ? -1 : 1;
                }
                // Sort by first tensor dimension.
                return sizeB - sizeA;
              });
      this.tensorNames = names.map(name => {
        return {name, shape: this.checkpointInfo.tensors[name].shape};
      });
      this.dom.select('#checkpoint-file')
          .text(this.checkpointInfo.checkpointFile)
          .attr('title', this.checkpointInfo.checkpointFile);
      this.dataProvider.getDefaultTensor(this.selectedRun, defaultTensor => {
        this.selectedTensor = defaultTensor;
      });
    });
  }

  _colorOptionChanged() {
    if (this.colorOption.map == null) {
      this.colorLegendRenderInfo = null;
    } else if (this.colorOption.items) {
      let items = this.colorOption.items.map(item => {
        return {
          color: this.colorOption.map(item.label),
          label: item.label,
          count: item.count
        };
      });
      this.colorLegendRenderInfo = {items, thresholds: null};
    } else {
      this.colorLegendRenderInfo = {
        items: null,
        thresholds: this.colorOption.thresholds
      };
    }
  }

  private tensorWasReadFromFile(rawContents: string, fileName: string) {
    parseRawTensors(rawContents, ds => {
      this.dom.select('#checkpoint-file')
          .text(fileName)
          .attr('title', fileName);
      this.projector.updateDataSet(ds);
    });
  }

  private metadataWasReadFromFile(rawContents: string, fileName: string) {
    parseRawMetadata(rawContents, this.projector.dataSet, stats => {
      this.projector.updateDataSet(this.projector.dataSet);
      this.updateMetadataUI(stats, fileName);
    });
  }

  private setupUploadButtons() {
    // Show and setup the upload button.
    let fileInput = this.dom.select('#file');
    fileInput.on('change', () => {
      let file: File = (d3.event as any).target.files[0];
      // Clear out the value of the file chooser. This ensures that if the user
      // selects the same file, we'll re-read it.
      (d3.event as any).target.value = '';
      let fileReader = new FileReader();
      fileReader.onload = evt => {
        let content: string = (evt.target as any).result;
        this.tensorWasReadFromFile(content, file.name);
      };
      fileReader.readAsText(file);
    });

    let uploadButton = this.dom.select('#upload');
    uploadButton.on(
        'click', () => { (fileInput.node() as HTMLInputElement).click(); });

    // Show and setup the upload metadata button.
    let fileMetadataInput = this.dom.select('#file-metadata');
    fileMetadataInput.on('change', () => {
      let file: File = (d3.event as any).target.files[0];
      // Clear out the value of the file chooser. This ensures that if the user
      // selects the same file, we'll re-read it.
      (d3.event as any).target.value = '';
      let fileReader = new FileReader();
      fileReader.onload = evt => {
        let contents: string = (evt.target as any).result;
        this.metadataWasReadFromFile(contents, file.name);
      };
      fileReader.readAsText(file);
    });

    let uploadMetadataButton = this.dom.select('#upload-metadata');
    uploadMetadataButton.on('click', () => {
      (fileMetadataInput.node() as HTMLInputElement).click();
    });
  }

  _getNumTensorsLabel(): string {
    return this.tensorNames.length === 1 ? '1 tensor' :
                                           this.tensorNames.length + ' tensors';
  }

  _getNumRunsLabel(): string {
    return this.runNames.length === 1 ? '1 run' :
                                        this.runNames.length + ' runs';
  }

  _hasChoices(choices: any[]): boolean {
    return choices.length > 1;
  }
}

document.registerElement(DataPanel.prototype.is, DataPanel);
