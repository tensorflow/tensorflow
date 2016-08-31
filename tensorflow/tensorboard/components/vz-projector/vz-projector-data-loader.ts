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

import {runAsyncTask, updateMessage} from './async';
import {DataPoint, DataSet, DatasetMetadata, DataSource} from './data';
import {PolymerElement} from './vz-projector-util';


/** Prefix added to the http requests when asking the server for data. */
const DATA_URL = 'data';

type DemoDataset = {
  fpath: string; metadata_path?: string; metadata?: DatasetMetadata;
};

type Metadata = {
  [key: string]: (number|string);
};

/** List of compiled demo datasets for showing the capabilities of the tool. */
const DEMO_DATASETS: {[name: string]: DemoDataset} = {
  'wiki_5k': {
    fpath: 'wiki_5000_50d_tensors.ssv',
    metadata_path: 'wiki_5000_50d_labels.ssv'
  },
  'wiki_10k': {
    fpath: 'wiki_10000_100d_tensors.ssv',
    metadata_path: 'wiki_10000_100d_labels.ssv'
  },
  'wiki_40k': {
    fpath: 'wiki_40000_100d_tensors.ssv',
    metadata_path: 'wiki_40000_100d_labels.ssv'
  },
  'smartreply_5k': {
    fpath: 'smartreply_5000_256d_tensors.tsv',
    metadata_path: 'smartreply_5000_256d_labels.tsv'
  },
  'smartreply_full': {
    fpath: 'smartreply_full_256d_tensors.tsv',
    metadata_path: 'smartreply_full_256d_labels.tsv'
  },
  'mnist_10k': {
    fpath: 'mnist_10k_784d_tensors.tsv',
    metadata_path: 'mnist_10k_784d_labels.tsv',
    metadata: {
      image:
          {sprite_fpath: 'mnist_10k_sprite.png', single_image_dim: [28, 28]}
    },
  },
  'iris': {fpath: 'iris_tensors.tsv', metadata_path: 'iris_labels.tsv'}
};

/** Maximum number of colors supported in the color map. */
const NUM_COLORS_COLOR_MAP = 20;

interface ServerInfo {
  tensors: {[name: string]: [number, number]};
  tensors_file: string;
  checkpoint_file: string;
  checkpoint_dir: string;
  metadata_file: string;
}

let DataLoaderPolymer = PolymerElement({
  is: 'vz-projector-data-loader',
  properties: {
    dataSource: {
      type: Object,  // DataSource
      notify: true
    },
    selectedDemo: {type: String, value: 'wiki_5k', notify: true},
    selectedTensor: {type: String, notify: true},
    labelOption: {type: String, notify: true},
    colorOption: {type: Object, notify: true},
    // Private.
    tensorNames: Array
  }
});

export type ColorOption = {
  name: string; desc?: string; map?: (value: string | number) => string;
  isSeparator?: boolean;
};

class DataLoader extends DataLoaderPolymer {
  dataSource: DataSource;
  selectedDemo: string;
  labelOption: string;
  labelOptions: string[];
  colorOption: ColorOption;
  colorOptions: ColorOption[];
  selectedTensor: string;
  tensorNames: {name: string, shape: number[]}[];

  private dom: d3.Selection<any>;

  ready() {
    this.dom = d3.select(this);
    if (this.dataSource) {
      // There is data already.
      return;
    }
    // Check to see if there is a server.
    d3.json(`${DATA_URL}/info`, (err, serverInfo) => {
      if (err) {
        // No server was found, thus operate in standalone mode.
        this.setupStandaloneMode();
        return;
      }
      // Server was found, thus show the checkpoint dir and the tensors.
      this.setupServerMode(serverInfo);
    });
  }

  getSeparatorClass(isSeparator: boolean): string {
    return isSeparator ? 'separator' : null;
  }

  private setupServerMode(info: ServerInfo) {
    // Display the server-mode controls.
    this.dom.select('.server-controls').style('display', null);
    this.dom.select('#checkpoint-file')
        .text(info.checkpoint_file)
        .attr('title', info.checkpoint_file);
    this.dom.select('#metadata-file')
        .text(info.metadata_file)
        .attr('title', info.metadata_file);

    // Handle the list of checkpoint tensors.
    this.dom.on('selected-tensor-changed', () => {
      this.selectedTensorChanged(this.selectedTensor);
    });
    let names = Object.keys(info.tensors)
                    .filter(name => {
                      let shape = info.tensors[name];
                      return shape.length == 2 && shape[0] > 1 && shape[1] > 1;
                    })
                    .sort((a, b) => info.tensors[b][0] - info.tensors[a][0]);
    this.tensorNames =
        names.map(name => { return {name, shape: info.tensors[name]}; });
  }

  private updateMetadataUI(columnStats: ColumnStats[]) {
    // Label by options.
    let labelIndex = -1;
    this.labelOptions = columnStats.length > 1 ? columnStats.map((stats, i) => {
      // Make the default label by the first non-numeric column.
      if (!stats.isNumeric && labelIndex == -1) {
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
      //{name: 'Distance of neighbors',
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
                  if (index == 0) {
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

  private setupStandaloneMode() {
    // Display the standalone UI controls.
    this.dom.select('.standalone-controls').style('display', null);

    // Demo dataset dropdown
    let demoDatasetChanged = (demoDataSet: DemoDataset) => {
      if (demoDataSet == null) {
        return;
      }

      this.dom.selectAll('.file-name').style('display', 'none');
      let separator = demoDataSet.fpath.substr(-3) == 'tsv' ? '\t' : ' ';
      fetchDemoData(`${DATA_URL}/${demoDataSet.fpath}`, separator)
          .then(points => {

            let p1 = demoDataSet.metadata_path ?
                new Promise<ColumnStats[]>((resolve, reject) => {
                  updateMessage('Fetching metadata...');
                  d3.text(
                      `${DATA_URL}/${demoDataSet.metadata_path}`,
                      (err: Error, rawMetadata: string) => {
                        if (err) {
                          console.error(err);
                          reject(err);
                          return;
                        }
                        resolve(parseAndMergeMetadata(rawMetadata, points));
                      });
                }) :
                null;

            let p2 = demoDataSet.metadata && demoDataSet.metadata.image ?
                fetchImage(
                    `${DATA_URL}/${demoDataSet.metadata.image.sprite_fpath}`) :
                null;

            Promise.all([p1, p2]).then(values => {
              this.updateMetadataUI(values[0]);
              let dataSource = new DataSource();
              dataSource.originalDataSet = new DataSet(points);
              dataSource.spriteImage = values[1];
              dataSource.metadata = demoDataSet.metadata;
              this.dataSource = dataSource;
            });
          });
    };

    this.dom.on('selected-demo-changed', () => {
      demoDatasetChanged(DEMO_DATASETS[this.selectedDemo]);
    });
    demoDatasetChanged(DEMO_DATASETS[this.selectedDemo]);

    // Show and setup the upload button.
    let fileInput = this.dom.select('#file');
    fileInput.on('change', () => {
      let file: File = (<any>d3.event).target.files[0];
      this.dom.select('#file-name')
          .style('display', null)
          .text(file.name)
          .attr('title', file.name);
      // Clear out the value of the file chooser. This ensures that if the user
      // selects the same file, we'll re-read it.
      (<any>d3.event).target.value = '';
      // Clear the value of the datasets dropdown.
      this.selectedDemo = null;
      let fileReader = new FileReader();
      fileReader.onload = evt => {
        let str: string = (evt.target as any).result;
        parseTensors(str).then(data => {
          let dataSource = new DataSource();
          dataSource.originalDataSet = new DataSet(data);
          this.dataSource = dataSource;
        });
      };
      fileReader.readAsText(file);
    });

    let uploadButton = this.dom.select('#upload');
    uploadButton.on(
        'click', () => { (<HTMLInputElement>fileInput.node()).click(); });

    // Show and setup the upload metadata button.
    let fileMetadataInput = this.dom.select('#file-metadata');
    fileMetadataInput.on('change', () => {
      let file: File = (<any>d3.event).target.files[0];
      this.dom.select('#file-metadata-name')
          .style('display', null)
          .text(file.name)
          .attr('title', file.name);
      // Clear out the value of the file chooser. This ensures that if the user
      // selects the same file, we'll re-read it.
      (<any>d3.event).target.value = '';
      // Clear the value of the datasets dropdown.
      this.selectedDemo = null;
      let fileReader = new FileReader();
      fileReader.onload = evt => {
        let str: string = (evt.target as any).result;
        parseAndMergeMetadata(str, this.dataSource.originalDataSet.points)
            .then(columnStats => {
              this.updateMetadataUI(columnStats);
              // Must make a shallow copy, otherwise polymer will not
              // fire the 'data-changed' event, even if we explicitly
              // call this.fire().
              this.dataSource = this.dataSource.makeShallowCopy();
            });
      };
      fileReader.readAsText(file);
    });

    let uploadMetadataButton = this.dom.select('#upload-metadata');
    uploadMetadataButton.on('click', () => {
      (<HTMLInputElement>fileMetadataInput.node()).click();
    });
  }

  private selectedTensorChanged(name: string) {
    // Get the tensor.
    updateMessage('Fetching tensor values...');
    d3.text(`${DATA_URL}/tensor?name=${name}`, (err: Error, tsv: string) => {
      if (err) {
        console.error(err);
        return;
      }
      parseTensors(tsv).then(dataPoints => {
        updateMessage('Fetching metadata...');
        d3.text(`${DATA_URL}/metadata`, (err: Error, rawMetadata: string) => {
          if (err) {
            console.error(err);
            return;
          }
          parseAndMergeMetadata(rawMetadata, dataPoints).then(columnStats => {
            this.updateMetadataUI(columnStats);
            let dataSource = new DataSource();
            dataSource.originalDataSet = new DataSet(dataPoints);
            this.dataSource = dataSource;
          });
        });
      });
    });
  }

  private getNumTensorsLabel(tensorNames: string[]) {
    return tensorNames.length === 1 ? '1 tensor' :
                                      tensorNames.length + ' tensors';
  }
}

function fetchImage(url: string): Promise<HTMLImageElement> {
  return new Promise<HTMLImageElement>((resolve, reject) => {
    let image = new Image();
    image.onload = () => resolve(image);
    image.onerror = (err) => reject(err);
    image.src = url;
  });
}

/** Makes a network request for a delimited text file. */
function fetchDemoData(url: string, separator: string): Promise<DataPoint[]> {
  return new Promise<DataPoint[]>((resolve, reject) => {
    updateMessage('Fetching tensors...');
    d3.text(url, (error: Error, dataString: string) => {
      if (error) {
        console.error(error);
        updateMessage('Error loading data.');
        reject(error);
      } else {
        parseTensors(dataString, separator).then(data => resolve(data));
      }
    });
  });
}

/** Parses a tsv text file. */
function parseTensors(content: string, delim = '\t'): Promise<DataPoint[]> {
  let data: DataPoint[] = [];
  let numDim: number;
  return runAsyncTask('Parsing tensors...', () => {
    let lines = content.split('\n');
    lines.forEach(line => {
      line = line.trim();
      if (line == '') {
        return;
      }
      let row = line.split(delim);
      let dataPoint: DataPoint = {
        metadata: {},
        vector: null,
        dataSourceIndex: data.length,
        projections: null,
        projectedPoint: null
      };
      // If the first label is not a number, take it as the label.
      if (isNaN(row[0] as any) || numDim == row.length - 1) {
        dataPoint.metadata['label'] = row[0];
        dataPoint.vector = row.slice(1).map(Number);
      } else {
        dataPoint.vector = row.map(Number);
      }
      data.push(dataPoint);
      if (numDim == null) {
        numDim = dataPoint.vector.length;
      }
      if (numDim != dataPoint.vector.length) {
        updateMessage('Parsing failed. Vector dimensions do not match');
        throw Error('Parsing failed');
      }
      if (numDim <= 1) {
        updateMessage(
            'Parsing failed. Found a vector with only one dimension?');
        throw Error('Parsing failed');
      }
    });
    return data;
  });
}

/** Statistics for a metadata column. */
type ColumnStats = {
  name: string; isNumeric: boolean; tooManyUniqueValues: boolean;
  uniqueValues?: string[];
  min: number;
  max: number;
};

function parseAndMergeMetadata(
    content: string, data: DataPoint[]): Promise<ColumnStats[]> {
  return runAsyncTask('Parsing metadata...', () => {
    let lines = content.split('\n').filter(line => line.trim().length > 0);
    let hasHeader = (lines.length - 1 == data.length);

    // Dimension mismatch.
    if (lines.length != data.length && !hasHeader) {
      throw Error('Dimensions do not match');
    }

    // If the first row doesn't contain metadata keys, we assume that the values
    // are labels.
    let columnNames: string[] = ['label'];
    if (hasHeader) {
      columnNames = lines[0].split('\t');
      lines = lines.slice(1);
    }
    let columnStats: ColumnStats[] = columnNames.map(name => {
      return {
        name: name,
        isNumeric: true,
        tooManyUniqueValues: false,
        min: Number.POSITIVE_INFINITY,
        max: Number.NEGATIVE_INFINITY
      };
    });
    let setOfValues = columnNames.map(() => d3.set());
    lines.forEach((line: string, i: number) => {
      let rowValues = line.split('\t');
      data[i].metadata = {};
      columnNames.forEach((name: string, colIndex: number) => {
        let value = rowValues[colIndex];
        let set = setOfValues[colIndex];
        let stats = columnStats[colIndex];
        data[i].metadata[name] = value;

        // Update stats.
        if (!stats.tooManyUniqueValues) {
          set.add(value);
          if (set.size() > NUM_COLORS_COLOR_MAP) {
            stats.tooManyUniqueValues = true;
          }
        }
        if (isNaN(value as any)) {
          stats.isNumeric = false;
        } else {
          stats.min = Math.min(stats.min, +value);
          stats.max = Math.max(stats.max, +value);
        }
      });
    });
    columnStats.forEach((stats, colIndex) => {
      let set = setOfValues[colIndex];
      if (!stats.tooManyUniqueValues) {
        stats.uniqueValues = set.values();
      }
    });
    return columnStats;
  });
}

document.registerElement(DataLoader.prototype.is, DataLoader);
