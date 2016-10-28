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

import {ColumnStats, DataPoint, DataSet, MetadataInfo, PointMetadata, State} from './data';
import * as logging from './logging';
import {runAsyncTask} from './util';

/** Maximum number of colors supported in the color map. */
const NUM_COLORS_COLOR_MAP = 20;

export const METADATA_MSG_ID = 'metadata';
export const TENSORS_MSG_ID = 'tensors';

/** Information associated with a tensor. */
export interface TensorInfo {
  /** Name of the tensor. */
  name: string;
  /** The shape of the tensor. */
  shape: [number, number];
  /** The path to the metadata file associated with the tensor. */
  metadataFile: string;
  /** The path to the bookmarks file associated with the tensor. */
  bookmarksFile: string;
}

/** Information for the model checkpoint. */
export interface CheckpointInfo {
  tensors: {[name: string]: TensorInfo};
  checkpointFile: string;
}

export type ServingMode = 'demo' | 'server' | 'proto';

/** Interface between the data storage and the UI. */
export interface DataProvider {
  /** Returns a list of run names that have embedding config files. */
  retrieveRuns(callback: (runs: string[]) => void): void;

  /**
   * Returns info about the checkpoint: number of tensors, their shapes,
   * and their associated metadata files.
   */
  retrieveCheckpointInfo(run: string, callback: (d: CheckpointInfo) => void): void;

  /** Fetches and returns the tensor with the specified name. */
  retrieveTensor(run: string, tensorName: string, callback: (ds: DataSet) => void);

  /**
   * Fetches the metadata for the specified tensor.
   */
  retrieveMetadata(run: string, tensorName: string,
      callback: (r: MetadataInfo) => void): void;

  /**
   * Returns the name of the tensor that should be fetched by default.
   * Used in demo mode to load a tensor when the app starts. Returns null if no
   * default tensor exists.
   */
  getDefaultTensor(run: string, callback: (tensorName: string) => void): void;

  getBookmarks(run: string, tensorName: string, callback: (r: State[]) => void):
      void;
}

export function parseRawTensors(
    content: string, callback: (ds: DataSet) => void) {
  parseTensors(content).then(data => {
    callback(new DataSet(data));
  });
}

export function parseRawMetadata(
    contents: string, callback: (r: MetadataInfo) => void) {
  parseMetadata(contents).then(result => callback(result));
}

/** Parses a tsv text file. */
export function parseTensors(
    content: string, delim = '\t'): Promise<DataPoint[]> {
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
        logging.setModalMessage(
            'Parsing failed. Vector dimensions do not match');
        throw Error('Parsing failed');
      }
      if (numDim <= 1) {
        logging.setModalMessage(
            'Parsing failed. Found a vector with only one dimension?');
        throw Error('Parsing failed');
      }
    });
    return data;
  }, TENSORS_MSG_ID).then(dataPoints => {
    logging.setModalMessage(null, TENSORS_MSG_ID);
    return dataPoints;
  });
}

export function analyzeMetadata(
    columnNames, pointsMetadata: PointMetadata[]): ColumnStats[] {
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
  pointsMetadata.forEach(metadata => {
    columnNames.forEach((name: string, colIndex: number) => {
      let stats = columnStats[colIndex];
      let map = mapOfValues[colIndex];
      let value = metadata[name];

      // Skip missing values.
      if (value == null) {
        return;
      }

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
        metadata[name] = +value;
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
}

export function parseMetadata(content: string): Promise<MetadataInfo> {
  return runAsyncTask('Parsing metadata...', () => {
    let lines = content.split('\n').filter(line => line.trim().length > 0);
    let hasHeader = lines[0].indexOf('\t') >= 0;
    let pointsMetadata: PointMetadata[] = [];
    // If the first row doesn't contain metadata keys, we assume that the values
    // are labels.
    let columnNames = ['label'];
    if (hasHeader) {
      columnNames = lines[0].split('\t');
      lines = lines.slice(1);
    }
    lines.forEach((line: string) => {
      let rowValues = line.split('\t');
      let metadata: PointMetadata = {};
      pointsMetadata.push(metadata);
      columnNames.forEach((name: string, colIndex: number) => {
        let value = rowValues[colIndex];
        // Normalize missing values.
        value = (value === '' ? null : value);
        metadata[name] = value;
      });
    });
    return {
      stats: analyzeMetadata(columnNames, pointsMetadata),
      pointsInfo: pointsMetadata
    } as MetadataInfo;
  }, METADATA_MSG_ID).then(metadata => {
    logging.setModalMessage(null, METADATA_MSG_ID);
    return metadata;
  });
}

export function fetchImage(url: string): Promise<HTMLImageElement> {
  return new Promise<HTMLImageElement>((resolve, reject) => {
    let image = new Image();
    image.onload = () => resolve(image);
    image.onerror = (err) => reject(err);
    image.src = url;
  });
}
