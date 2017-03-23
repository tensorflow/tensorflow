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

import {ColorOption, ColumnStats, SpriteAndMetadataInfo} from './data';
import {DataProvider, EmbeddingInfo, parseRawMetadata, parseRawTensors, ProjectorConfig} from './data-provider';
import * as util from './util';
import {Projector} from './vz-projector';
import {ColorLegendRenderInfo, ColorLegendThreshold} from './vz-projector-legend';
// tslint:disable-next-line:no-unused-variable
import {PolymerElement, PolymerHTMLElement} from './vz-projector-util';

export let DataPanelPolymer = PolymerElement({
  is: 'vz-projector-data-panel',
  properties: {
    selectedTensor: {type: String, observer: '_selectedTensorChanged'},
    selectedRun: {type: String, observer: '_selectedRunChanged'},
    selectedColorOptionName: {
      type: String,
      notify: true,
      observer: '_selectedColorOptionNameChanged'
    },
    selectedLabelOption:
        {type: String, notify: true, observer: '_selectedLabelOptionChanged'},
    normalizeData: Boolean,
    showForceCategoricalColorsCheckbox: Boolean
  }
});

export class DataPanel extends DataPanelPolymer {
  selectedLabelOption: string;
  selectedColorOptionName: string;
  showForceCategoricalColorsCheckbox: boolean;

  private normalizeData: boolean;
  private labelOptions: string[];
  private colorOptions: ColorOption[];
  forceCategoricalColoring: boolean = false;
  private dom: d3.Selection<any>;

  private selectedTensor: string;
  private selectedRun: string;
  private dataProvider: DataProvider;
  private tensorNames: {name: string, shape: number[]}[];
  private runNames: string[];
  private projector: Projector;
  private projectorConfig: ProjectorConfig;
  private colorLegendRenderInfo: ColorLegendRenderInfo;
  private spriteAndMetadata: SpriteAndMetadataInfo;
  private metadataFile: string;

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

    let forceCategoricalColoringCheckbox =
        this.querySelector('#force-categorical-checkbox');
    forceCategoricalColoringCheckbox.addEventListener('change', () => {
      this.setForceCategoricalColoring(
          (forceCategoricalColoringCheckbox as HTMLInputElement).checked);
    });

    // Get all the runs.
    this.dataProvider.retrieveRuns(runs => {
      this.runNames = runs;
      // Choose the first run by default.
      if (this.runNames.length > 0) {
        this.selectedRun = runs[0];
      }
    });
  }

  setForceCategoricalColoring(forceCategoricalColoring: boolean) {
    this.forceCategoricalColoring = forceCategoricalColoring;
    (this.querySelector('#force-categorical-checkbox') as HTMLInputElement)
        .checked = this.forceCategoricalColoring;

    this.updateMetadataUI(this.spriteAndMetadata.stats, this.metadataFile);

    // The selected color option name doesn't change when we switch to using
    // categorical coloring for stats with too many unique values, so we
    // manually call this polymer observer so that we update the UI.
    this._selectedColorOptionNameChanged();
  }

  getSeparatorClass(isSeparator: boolean): string {
    return isSeparator ? 'separator' : null;
  }

  metadataChanged(
      spriteAndMetadata: SpriteAndMetadataInfo, metadataFile: string) {
    this.spriteAndMetadata = spriteAndMetadata;
    this.metadataFile = metadataFile;

    this.updateMetadataUI(this.spriteAndMetadata.stats, this.metadataFile);
    this.selectedColorOptionName = this.colorOptions[0].name;
  }

  private addWordBreaks(longString: string): string {
    if (longString == null) {
      return '';
    }
    return longString.replace(/([\/=-_,])/g, '$1<wbr>');
  }

  private updateMetadataUI(columnStats: ColumnStats[], metadataFile: string) {
    this.dom.select('#metadata-file')
        .html(this.addWordBreaks(metadataFile))
        .attr('title', metadataFile);
    // Label by options.
    let labelIndex = -1;
    this.labelOptions = columnStats.map((stats, i) => {
      // Make the default label by the first non-numeric column.
      if (!stats.isNumeric && labelIndex === -1) {
        labelIndex = i;
      }
      return stats.name;
    });
    this.selectedLabelOption = this.labelOptions[Math.max(0, labelIndex)];

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
              let isCategorical =
                  this.forceCategoricalColoring || !stats.tooManyUniqueValues;
              if (isCategorical) {
                let scale = d3.scale.category20();
                let range = scale.range();
                // Re-order the range.
                let newRange = range.map((color, i) => {
                  let index = (i * 3) % range.length;
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
              let desc = !isCategorical ? 'gradient' :
                                          stats.uniqueEntries.length +
                      ((stats.uniqueEntries.length > 20) ? ' non-unique' : '') +
                      ' colors';
              return {
                name: stats.name,
                desc: desc,
                map: map,
                items: items,
                thresholds: thresholds,
                tooManyUniqueValues: stats.tooManyUniqueValues
              };
            });

    if (metadataColorOption.length > 0) {
      // Add a separator line between built-in color maps
      // and those based on metadata columns.
      standardColorOption.push({name: 'Metadata', isSeparator: true});
    }
    this.colorOptions = standardColorOption.concat(metadataColorOption);
  }

  setNormalizeData(normalizeData: boolean) {
    this.normalizeData = normalizeData;
  }

  _selectedTensorChanged() {
    this.projector.updateDataSet(null, null, null);
    if (this.selectedTensor == null) {
      return;
    }
    this.dataProvider.retrieveTensor(
        this.selectedRun, this.selectedTensor, ds => {
          let metadataFile =
              this.getEmbeddingInfoByName(this.selectedTensor).metadataPath;
          this.dataProvider.retrieveSpriteAndMetadata(
              this.selectedRun, this.selectedTensor, metadata => {
                this.projector.updateDataSet(ds, metadata, metadataFile);
              });
        });
    this.projector.setSelectedTensor(
        this.selectedRun, this.getEmbeddingInfoByName(this.selectedTensor));
  }

  _selectedRunChanged() {
    this.dataProvider.retrieveProjectorConfig(this.selectedRun, info => {
      this.projectorConfig = info;
      let names =
          this.projectorConfig.embeddings.map(e => e.tensorName)
              .filter(name => {
                let shape = this.getEmbeddingInfoByName(name).tensorShape;
                return shape.length === 2 && shape[0] > 1 && shape[1] > 1;
              })
              .sort((a, b) => {
                let embA = this.getEmbeddingInfoByName(a);
                let embB = this.getEmbeddingInfoByName(b);

                // Prefer tensors with metadata.
                if (util.xor(!!embA.metadataPath, !!embB.metadataPath)) {
                  return embA.metadataPath ? -1 : 1;
                }

                // Prefer non-generated tensors.
                let isGenA = util.tensorIsGenerated(a);
                let isGenB = util.tensorIsGenerated(b);
                if (util.xor(isGenA, isGenB)) {
                  return isGenB ? -1 : 1;
                }

                // Prefer bigger tensors.
                let sizeA = embA.tensorShape[0];
                let sizeB = embB.tensorShape[0];
                if (sizeA !== sizeB) {
                  return sizeB - sizeA;
                }

                // Sort alphabetically by tensor name.
                return a <= b ? -1 : 1;
              });
      this.tensorNames = names.map(name => {
        return {name, shape: this.getEmbeddingInfoByName(name).tensorShape};
      });
      let wordBreakablePath =
          this.addWordBreaks(this.projectorConfig.modelCheckpointPath);
      this.dom.select('#checkpoint-file')
          .html(wordBreakablePath)
          .attr('title', this.projectorConfig.modelCheckpointPath);
      // If in demo mode, let the order decide which tensor to load by default.
      let defaultTensor = this.projector.servingMode === 'demo' ?
          this.projectorConfig.embeddings[0].tensorName :
          names[0];
      if (this.selectedTensor === defaultTensor) {
        // Explicitly call the observer. Polymer won't call it if the previous
        // string matches the current string.
        this._selectedTensorChanged();
      } else {
        this.selectedTensor = defaultTensor;
      }
    });
  }

  _selectedLabelOptionChanged() {
    this.projector.setSelectedLabelOption(this.selectedLabelOption);
  }

  _selectedColorOptionNameChanged() {
    let colorOption: ColorOption;
    for (let i = 0; i < this.colorOptions.length; i++) {
      if (this.colorOptions[i].name === this.selectedColorOptionName) {
        colorOption = this.colorOptions[i];
        break;
      }
    }
    if (!colorOption) {
      return;
    }

    this.showForceCategoricalColorsCheckbox = !!colorOption.tooManyUniqueValues;

    if (colorOption.map == null) {
      this.colorLegendRenderInfo = null;
    } else if (colorOption.items) {
      let items = colorOption.items.map(item => {
        return {
          color: colorOption.map(item.label),
          label: item.label,
          count: item.count
        };
      });
      this.colorLegendRenderInfo = {items, thresholds: null};
    } else {
      this.colorLegendRenderInfo = {
        items: null,
        thresholds: colorOption.thresholds
      };
    }
    this.projector.setSelectedColorOption(colorOption);
  }

  private tensorWasReadFromFile(rawContents: ArrayBuffer, fileName: string) {
    parseRawTensors(rawContents, ds => {
      this.dom.select('#checkpoint-file')
          .text(fileName)
          .attr('title', fileName);
      this.projector.updateDataSet(ds);
    });
  }

  private metadataWasReadFromFile(rawContents: ArrayBuffer, fileName: string) {
    parseRawMetadata(rawContents, metadata => {
      this.projector.updateDataSet(this.projector.dataSet, metadata, fileName);
    });
  }

  private getEmbeddingInfoByName(tensorName: string): EmbeddingInfo {
    for (let i = 0; i < this.projectorConfig.embeddings.length; i++) {
      let e = this.projectorConfig.embeddings[i];
      if (e.tensorName === tensorName) {
        return e;
      }
    }
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
        let content: ArrayBuffer = (evt.target as any).result;
        this.tensorWasReadFromFile(content, file.name);
      };
      fileReader.readAsArrayBuffer(file);
    });

    let uploadButton = this.dom.select('#upload-tensors');
    uploadButton.on('click', () => {
      (fileInput.node() as HTMLInputElement).click();
    });

    // Show and setup the upload metadata button.
    let fileMetadataInput = this.dom.select('#file-metadata');
    fileMetadataInput.on('change', () => {
      let file: File = (d3.event as any).target.files[0];
      // Clear out the value of the file chooser. This ensures that if the user
      // selects the same file, we'll re-read it.
      (d3.event as any).target.value = '';
      let fileReader = new FileReader();
      fileReader.onload = evt => {
        let contents: ArrayBuffer = (evt.target as any).result;
        this.metadataWasReadFromFile(contents, file.name);
      };
      fileReader.readAsArrayBuffer(file);
    });

    let uploadMetadataButton = this.dom.select('#upload-metadata');
    uploadMetadataButton.on('click', () => {
      (fileMetadataInput.node() as HTMLInputElement).click();
    });

    if (this.projector.servingMode === 'demo') {
      (this.$$('#demo-data-buttons-container') as HTMLElement).style.display =
          'block';

      // Fill out the projector config.
      let projectorConfigTemplate =
          this.$$('#projector-config-template') as HTMLTextAreaElement;
      let projectorConfigTemplateJson: ProjectorConfig = {
        embeddings: [{
          tensorName: 'My tensor',
          tensorShape: [1000, 50],
          tensorPath: 'https://gist.github.com/.../tensors.tsv',
          metadataPath: 'https://gist.github.com/.../optional.metadata.tsv',
        }],
      };
      this.setProjectorConfigTemplateJson(
          projectorConfigTemplate, projectorConfigTemplateJson);

      // Set up optional field checkboxes.
      let spriteFieldCheckbox = this.$$('#config-sprite-checkbox');
      spriteFieldCheckbox.addEventListener('change', () => {
        if ((spriteFieldCheckbox as any).checked) {
          projectorConfigTemplateJson.embeddings[0].sprite = {
            imagePath: 'https://github.com/.../optional.sprite.png',
            singleImageDim: [32, 32]
          };
        } else {
          delete projectorConfigTemplateJson.embeddings[0].sprite;
        }
        this.setProjectorConfigTemplateJson(
            projectorConfigTemplate, projectorConfigTemplateJson);
      });
      let bookmarksFieldCheckbox = this.$$('#config-bookmarks-checkbox');
      bookmarksFieldCheckbox.addEventListener('change', () => {
        if ((bookmarksFieldCheckbox as any).checked) {
          projectorConfigTemplateJson.embeddings[0].bookmarksPath =
              'https://gist.github.com/.../bookmarks.txt';
        } else {
          delete projectorConfigTemplateJson.embeddings[0].bookmarksPath;
        }
        this.setProjectorConfigTemplateJson(
            projectorConfigTemplate, projectorConfigTemplateJson);
      });
      let metadataFieldCheckbox = this.$$('#config-metadata-checkbox');
      metadataFieldCheckbox.addEventListener('change', () => {
        if ((metadataFieldCheckbox as any).checked) {
          projectorConfigTemplateJson.embeddings[0].metadataPath =
              'https://gist.github.com/.../optional.metadata.tsv';
        } else {
          delete projectorConfigTemplateJson.embeddings[0].metadataPath;
        }
        this.setProjectorConfigTemplateJson(
            projectorConfigTemplate, projectorConfigTemplateJson);
      });

      // Update the link and the readonly shareable URL.
      let projectorConfigUrlInput = this.$$('#projector-config-url');
      let projectorConfigDemoUrlInput = this.$$('#projector-share-url');
      let projectorConfigDemoUrlLink = this.$$('#projector-share-url-link');
      projectorConfigUrlInput.addEventListener('input', () => {
        let projectorDemoUrl = location.protocol + '//' + location.host +
            location.pathname + '?config=' +
            (projectorConfigUrlInput as any).value;

        (projectorConfigDemoUrlInput as any).value = projectorDemoUrl;
        (projectorConfigDemoUrlLink as any).href = projectorDemoUrl;
      });
    }
  }

  private setProjectorConfigTemplateJson(
      projectorConfigTemplate: HTMLTextAreaElement, config: ProjectorConfig) {
    projectorConfigTemplate.value =
        JSON.stringify(config, null, /** replacer */ 2 /** white space */);
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
