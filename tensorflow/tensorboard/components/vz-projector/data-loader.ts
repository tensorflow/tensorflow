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

import {runAsyncTask, updateMessage, updateWarningMessage} from './async';
import {DataPoint, DataSet, DatasetMetadata} from './data';

/** Maximum number of colors supported in the color map. */
const NUM_COLORS_COLOR_MAP = 20;

/** Information associated with a tensor. */
export interface TensorInfo {
  /** Name of the tensor. */
  name: string;
  /** The shape of the tensor. */
  shape: [number, number];
  /** The path to the metadata file associated with the tensor. */
  metadataFile: string;
}

/** Information for the model checkpoint. */
export interface CheckpointInfo {
  tensors: {[name: string]: TensorInfo};
  checkpointFile: string;
}

/** Interface between the data storage and the UI. */
export interface DataProvider {
  /** Returns a list of run names that have embedding config files. */
  getRuns(callback: (runs: string[]) => void): void;

  /**
   * Returns info about the checkpoint: number of tensors, their shapes,
   * and their associated metadata files.
   */
  getCheckpointInfo(run: string, callback: (d: CheckpointInfo) => void): void;

  /** Fetches and returns the tensor with the specified name. */
  getTensor(run: string, tensorName: string, callback: (ds: DataSet) => void);

  /**
   * Fetches the metadata for the specified tensor and merges it with the
   * specified data source.
   */
  getMetadata(
      run: string, ds: DataSet, tensorName: string,
      callback: (stats: ColumnStats[]) => void): void;

  /**
   * Returns the name of the tensor that should be fetched by default.
   * Used in demo mode to load a tensor when the app starts. Returns null if no
   * default tensor exists.
   */
  getDefaultTensor(run: string, callback: (tensorName: string) => void): void;
}

/**
 * Data provider that loads data provided by a python server (usually backed
 * by a checkpoint file).
 */
class ServerDataProvider implements DataProvider {
  /** Prefix added to the http requests when asking the server for data. */
  static DEFAULT_ROUTE_PREFIX = 'data';
  private routePrefix: string;

  constructor(routePrefix: string) {
    this.routePrefix = routePrefix;
  }

  getRuns(callback: (runs: string[]) => void): void {
    d3.json(`${this.routePrefix}/runs`, (err, runs) => {
      callback(runs);
    });
  }

  getCheckpointInfo(run: string, callback: (d: CheckpointInfo) => void): void {
    d3.json(`${this.routePrefix}/info?run=${run}`, (err, checkpointInfo) => {
      callback(checkpointInfo);
    });
  }

  getTensor(run: string, tensorName: string, callback: (ds: DataSet) => void) {
    // Get the tensor.
    updateMessage('Fetching tensor values...');
    d3.text(
        `${this.routePrefix}/tensor?run=${run}&name=${tensorName}`,
        (err: Error, tsv: string) => {
          if (err) {
            console.error(err);
            return;
          }
          parseTensors(tsv).then(dataPoints => {
            callback(new DataSet(dataPoints));
          });
        });
  }

  getMetadata(
      run: string, ds: DataSet, tensorName: string,
      callback: (stats: ColumnStats[]) => void) {
    updateMessage('Fetching metadata...');
    d3.text(
        `${this.routePrefix}/metadata?run=${run}&name=${tensorName}`,
        (err: Error, rawMetadata: string) => {
          if (err) {
            console.error(err);
            return;
          }
          parseAndMergeMetadata(rawMetadata, ds.points).then(columnStats => {
            callback(columnStats);
          });
        });
  }

  getDefaultTensor(run: string, callback: (tensorName: string) => void) {
    this.getCheckpointInfo(run, checkpointInfo => {
      let tensorNames = Object.keys(checkpointInfo.tensors);
      // Return the first tensor as default if there is only 1 tensor.
      callback(tensorNames.length === 1 ? tensorNames[0] : null);
    });
  }
}

/**
 * Returns a data provider, depending on what is available. The detection of
 * a server backend is done by issuing an HTTP request at /data/info and seeing
 * if it returns 200 or 404.
 *
 * @param routePrefix The prefix to add to the url routes when asking for data
 *     from the backend. For example, when hosted inside tensorboard, the route
 *     is prefixed by the plugin name.
 * @param callback Called with the data provider.
 */
export function getDataProvider(
    routePrefix: string, callback: (dp: DataProvider) => void) {
  if (routePrefix == null) {
    routePrefix = ServerDataProvider.DEFAULT_ROUTE_PREFIX;
  }
  d3.json(`${routePrefix}/runs`, (err, runs) => {
    callback(
        err ? new DemoDataProvider() : new ServerDataProvider(routePrefix));
  });
}

export function parseRawTensors(
    content: string, callback: (ds: DataSet) => void) {
  parseTensors(content).then(data => {
    callback(new DataSet(data));
  });
}

export function parseRawMetadata(
    contents: string, ds: DataSet, callback: (stats: ColumnStats[]) => void) {
  parseAndMergeMetadata(contents, ds.points).then(stats => {
    callback(stats);
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
      if (line === '') {
        return;
      }
      let row = line.split(delim);
      let dataPoint: DataPoint = {
        metadata: {},
        vector: null,
        index: data.length,
        projections: null,
        projectedPoint: null
      };
      // If the first label is not a number, take it as the label.
      if (isNaN(row[0] as any) || numDim === row.length - 1) {
        dataPoint.metadata['label'] = row[0];
        dataPoint.vector = row.slice(1).map(Number);
      } else {
        dataPoint.vector = row.map(Number);
      }
      data.push(dataPoint);
      if (numDim == null) {
        numDim = dataPoint.vector.length;
      }
      if (numDim !== dataPoint.vector.length) {
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
export interface ColumnStats {
  name: string;
  isNumeric: boolean;
  tooManyUniqueValues: boolean;
  uniqueEntries?: {label: string, count: number}[];
  min: number;
  max: number;
}

function parseAndMergeMetadata(
    content: string, data: DataPoint[]): Promise<ColumnStats[]> {
  return runAsyncTask('Parsing metadata...', () => {
    let lines = content.split('\n').filter(line => line.trim().length > 0);
    let hasHeader = lines[0].indexOf('\t') >= 0;

    // If the first row doesn't contain metadata keys, we assume that the values
    // are labels.
    let columnNames = ['label'];
    if (hasHeader) {
      columnNames = lines[0].split('\t');
      lines = lines.slice(1);
    }

    // Dimension mismatch.
    if (lines.length !== data.length) {
      updateWarningMessage(
          `Number of tensors (${data.length}) do not match` +
          ` the number of lines in metadata (${lines.length}).`);
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
    let mapOfValues = columnNames.map(() => d3.map<number>());
    lines.forEach((line: string, i: number) => {
      let rowValues = line.split('\t');
      data[i].metadata = {};
      columnNames.forEach((name: string, colIndex: number) => {
        let value = rowValues[colIndex];
        let map = mapOfValues[colIndex];
        let stats = columnStats[colIndex];
        // Normalize missing values.
        value = (value === '' ? null : value);
        data[i].metadata[name] = value;

        // Skip missing values.
        if (value == null) {
          return;
        }

        // Update stats.
        if (!stats.tooManyUniqueValues) {
          if (map.has(value)) {
            map.set(value, map.get(value) + 1);
          } else {
            map.set(value, 1);
          }
          if (map.size() > NUM_COLORS_COLOR_MAP) {
            stats.tooManyUniqueValues = true;
          }
        }
        if (isNaN(value as any)) {
          stats.isNumeric = false;
        } else {
          // Parse the data as number.
          data[i].metadata[name] = +value;
          stats.min = Math.min(stats.min, +value);
          stats.max = Math.max(stats.max, +value);
        }
      });
    });
    columnStats.forEach((stats, colIndex) => {
      let map = mapOfValues[colIndex];
      if (!stats.tooManyUniqueValues) {
        stats.uniqueEntries = map.entries().map(e => {
          return {label: e.key, count: e.value};
        });
      }
    });
    return columnStats;
  });
}

function fetchImage(url: string): Promise<HTMLImageElement> {
  return new Promise<HTMLImageElement>((resolve, reject) => {
    let image = new Image();
    image.onload = () => resolve(image);
    image.onerror = (err) => reject(err);
    image.src = url;
  });
}

type DemoDataset = {
  fpath: string; metadata_path?: string; metadata?: DatasetMetadata;
  shape: [number, number];
};

/** Data provider that loads data from a demo folder. */
class DemoDataProvider implements DataProvider {
  /** List of demo datasets for showing the capabilities of the tool. */
  private static DEMO_DATASETS: {[name: string]: DemoDataset} = {
    'Glove Wiki 5K': {
      shape: [5000, 50],
      fpath: 'wiki_5000_50d_tensors.ssv',
      metadata_path: 'wiki_5000_50d_labels.ssv'
    },
    'Glove Wiki 10K': {
      shape: [10000, 100],
      fpath: 'wiki_10000_100d_tensors.ssv',
      metadata_path: 'wiki_10000_100d_labels.ssv'
    },
    'Glove Wiki 40K': {
      shape: [40000, 100],
      fpath: 'wiki_40000_100d_tensors.ssv',
      metadata_path: 'wiki_40000_100d_labels.ssv'
    },
    'SmartReply 5K': {
      shape: [5000, 256],
      fpath: 'smartreply_5000_256d_tensors.tsv',
      metadata_path: 'smartreply_5000_256d_labels.tsv'
    },
    'SmartReply All': {
      shape: [35860, 256],
      fpath: 'smartreply_full_256d_tensors.tsv',
      metadata_path: 'smartreply_full_256d_labels.tsv'
    },
    'Mnist with images 10K': {
      shape: [10000, 784],
      fpath: 'mnist_10k_784d_tensors.tsv',
      metadata_path: 'mnist_10k_784d_labels.tsv',
      metadata: {
        image:
            {sprite_fpath: 'mnist_10k_sprite.png', single_image_dim: [28, 28]}
      },
    },
    'Iris': {
      shape: [150, 4],
      fpath: 'iris_tensors.tsv',
      metadata_path: 'iris_labels.tsv'
    }
  };
  /** Name of the folder where the demo datasets are stored. */
  private static DEMO_FOLDER = 'data';

  getRuns(callback: (runs: string[]) => void): void {
    callback(['Demo']);
  }

  getCheckpointInfo(run: string, callback: (d: CheckpointInfo) => void): void {
    let tensorsInfo: {[name: string]: TensorInfo} = {};
    for (let name in DemoDataProvider.DEMO_DATASETS) {
      if (!DemoDataProvider.DEMO_DATASETS.hasOwnProperty(name)) {
        continue;
      }
      let demoInfo = DemoDataProvider.DEMO_DATASETS[name];
      tensorsInfo[name] = {
        name: name,
        shape: demoInfo.shape,
        metadataFile: demoInfo.metadata_path
      };
    }
    callback({
      tensors: tensorsInfo,
      checkpointFile: 'Demo datasets',
    });
  }

  getDefaultTensor(run: string, callback: (tensorName: string) => void) {
    callback('SmartReply 5K');
  }

  getTensor(run: string, tensorName: string, callback: (ds: DataSet) => void) {
    let demoDataSet = DemoDataProvider.DEMO_DATASETS[tensorName];
    let separator = demoDataSet.fpath.substr(-3) === 'tsv' ? '\t' : ' ';
    let url = `${DemoDataProvider.DEMO_FOLDER}/${demoDataSet.fpath}`;
    updateMessage('Fetching tensors...');
    d3.text(url, (error: Error, dataString: string) => {
      if (error) {
        console.error(error);
        updateMessage('Error loading data.');
        return;
      }
      parseTensors(dataString, separator).then(points => {
        callback(new DataSet(points));
      });
    });
  }

  getMetadata(
      run: string, ds: DataSet, tensorName: string,
      callback: (stats: ColumnStats[]) => void) {
    let demoDataSet = DemoDataProvider.DEMO_DATASETS[tensorName];
    let dataSetPromise: Promise<ColumnStats[]> = null;
    if (demoDataSet.metadata_path) {
      dataSetPromise = new Promise<ColumnStats[]>((resolve, reject) => {
        updateMessage('Fetching metadata...');
        d3.text(
            `${DemoDataProvider.DEMO_FOLDER}/${demoDataSet.metadata_path}`,
            (err: Error, rawMetadata: string) => {
              if (err) {
                console.error(err);
                reject(err);
                return;
              }
              resolve(parseAndMergeMetadata(rawMetadata, ds.points));
            });
      });
    }
    let spritesPromise: Promise<HTMLImageElement> = null;
    if (demoDataSet.metadata && demoDataSet.metadata.image) {
      let spriteFilePath = demoDataSet.metadata.image.sprite_fpath;
      spritesPromise =
          fetchImage(`${DemoDataProvider.DEMO_FOLDER}/${spriteFilePath}`);
    }

    // Fetch the metadata and the image in parallel.
    Promise.all([dataSetPromise, spritesPromise]).then(values => {
      ds.spriteImage = values[1];
      ds.metadata = demoDataSet.metadata;
      callback(values[0]);
    });
  }
}
