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

import {CheckpointInfo, ColumnStats, DataProvider, parseRawMetadata, parseRawTensors} from './data-loader';
import {Projector} from './vz-projector';
// tslint:disable-next-line:no-unused-variable
import {PolymerElement, PolymerHTMLElement} from './vz-projector-util';

export let DataPanelPolymer = PolymerElement({
  is: 'vz-projector-data-panel',
  properties: {
    selectedTensor: {type: String, observer: 'selectedTensorChanged'},
    colorOption: {type: Object, notify: true},
    labelOption: {type: String, notify: true}
  }
});

export interface ColorOption {
  name: string;
  desc?: string;
  map?: (value: string|number) => string;
  isSeparator?: boolean;
};

export class DataPanel extends DataPanelPolymer {
  labelOption: string;
  colorOption: ColorOption;

  private labelOptions: string[];
  private colorOptions: ColorOption[];
  private dom: d3.Selection<any>;

  private selectedTensor: string;
  private dataProvider: DataProvider;
  private tensorNames: {name: string, shape: number[]}[];
  private projector: Projector;
  private checkpointInfo: CheckpointInfo;

  ready() {
    this.dom = d3.select(this);
  }

  initialize(projector: Projector, dp: DataProvider, dataInfo: CheckpointInfo) {
    this.projector = projector;
    this.dataProvider = dp;
    this.checkpointInfo = dataInfo;
    this.setupUI();
    let defaultTensor = dp.getDefaultTensor();
    if (defaultTensor != null) {
      this.selectedTensor = defaultTensor;
    }
  }

  getSeparatorClass(isSeparator: boolean): string {
    return isSeparator ? 'separator' : null;
  }

  private setupUI() {
    this.setupUploadButtons();
    let names = Object.keys(this.checkpointInfo.tensors)
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
                scale.range(newRange).domain(stats.uniqueValues);
                map = scale;
              } else {
                map = d3.scale.linear<string>()
                          .domain([stats.min, stats.max])
                          .range(['white', 'black']);
              }
              let desc = stats.tooManyUniqueValues ?
                  'gradient' :
                  stats.uniqueValues.length + ' colors';
              return {name: stats.name, desc: desc, map: map};
            });
    if (metadataColorOption.length > 0) {
      // Add a separator line between built-in color maps
      // and those based on metadata columns.
      standardColorOption.push({name: 'Metadata', isSeparator: true});
    }
    this.colorOptions = standardColorOption.concat(metadataColorOption);
    this.colorOption = this.colorOptions[0];
  }

  // tslint:disable-next-line:no-unused-variable
  private selectedTensorChanged() {
    this.dataProvider.getTensor(this.selectedTensor, ds => {
      let metadataFile =
          this.checkpointInfo.tensors[this.selectedTensor].metadataFile;
      if (metadataFile) {
        this.dataProvider.getMetadata(ds, this.selectedTensor, stats => {
          this.projector.updateDataSource(ds);
          this.updateMetadataUI(stats, metadataFile);
        });
      } else {
        this.projector.updateDataSource(ds);
      }
    });
  }

  private tensorWasReadFromFile(rawContents: string, fileName: string) {
    parseRawTensors(rawContents, ds => {
      this.dom.select('#checkpoint-file')
          .text(fileName)
          .attr('title', fileName);
      this.projector.updateDataSource(ds);
    });
  }

  private metadataWasReadFromFile(rawContents: string, fileName: string) {
    parseRawMetadata(rawContents, this.projector.dataSource, stats => {
      this.projector.updateDataSource(this.projector.dataSource);
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

  // tslint:disable-next-line:no-unused-variable
  private getNumTensorsLabel(tensorNames: string[]) {
    return tensorNames.length === 1 ? '1 tensor' :
                                      tensorNames.length + ' tensors';
  }
}

document.registerElement(DataPanel.prototype.is, DataPanel);
