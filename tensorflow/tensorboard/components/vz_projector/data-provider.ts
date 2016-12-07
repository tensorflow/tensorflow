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

import {ColumnStats, DataPoint, DataSet, SpriteAndMetadataInfo, PointMetadata, State} from './data';
import * as logging from './logging';
import {runAsyncTask} from './util';

/** Maximum number of colors supported in the color map. */
const NUM_COLORS_COLOR_MAP = 50;
const MAX_SPRITE_IMAGE_SIZE_PX = 8192;

export const METADATA_MSG_ID = 'metadata';
export const TENSORS_MSG_ID = 'tensors';

/** Matches the json format of `projector_config.proto` */
export interface SpriteMetadata {
  imagePath: string;
  singleImageDim: [number, number];
}

/** Matches the json format of `projector_config.proto` */
export interface EmbeddingInfo {
  /** Name of the tensor. */
  tensorName: string;
  /** The shape of the tensor. */
  tensorShape: [number, number];
  /**
   * The path to the tensors TSV file. If empty, it is assumed that the tensor
   * is stored in the checkpoint file.
   */
  tensorPath?: string;
  /** The path to the metadata file associated with the tensor. */
  metadataPath?: string;
  /** The path to the bookmarks file associated with the tensor. */
  bookmarksPath?: string;
  sprite?: SpriteMetadata;
}

/**
 * Matches the json format of `projector_config.proto`
 * This should be kept in sync with the code in vz-projector-data-panel which
 * holds a template for users to build a projector config JSON object from the
 * projector UI.
 */
export interface ProjectorConfig {
  embeddings: EmbeddingInfo[];
  modelCheckpointPath?: string;
}

export type ServingMode = 'demo' | 'server' | 'proto';

/** Interface between the data storage and the UI. */
export interface DataProvider {
  /** Returns a list of run names that have embedding config files. */
  retrieveRuns(callback: (runs: string[]) => void): void;

  /**
   * Returns the projector configuration: number of tensors, their shapes,
   * and their associated metadata files.
   */
  retrieveProjectorConfig(run: string,
      callback: (d: ProjectorConfig) => void): void;

  /** Fetches and returns the tensor with the specified name. */
  retrieveTensor(run: string, tensorName: string,
      callback: (ds: DataSet) => void);

  /**
   * Fetches the metadata for the specified tensor.
   */
  retrieveSpriteAndMetadata(run: string, tensorName: string,
      callback: (r: SpriteAndMetadataInfo) => void): void;

  getBookmarks(run: string, tensorName: string, callback: (r: State[]) => void):
      void;
}

export function retrieveTensorAsBytes(
    dp: DataProvider, embedding: EmbeddingInfo, run: string, tensorName: string,
    tensorsPath: string, callback: (ds: DataSet) => void) {
  // Get the tensor.
  logging.setModalMessage('Fetching tensor values...', TENSORS_MSG_ID);
  let xhr = new XMLHttpRequest();
  xhr.open('GET', tensorsPath);
  xhr.responseType = 'arraybuffer';
  xhr.onprogress = (ev) => {
    if (ev.lengthComputable) {
      let percent = (ev.loaded * 100 / ev.total).toFixed(1);
      logging.setModalMessage(
          'Fetching tensor values: ' + percent + '%', TENSORS_MSG_ID);
    }
  };
  xhr.onload = () => {
    if (xhr.status !== 200) {
      let msg = String.fromCharCode.apply(null, new Uint8Array(xhr.response));
      logging.setErrorMessage(msg);
      return;
    }
    let data = new Float32Array(xhr.response);
    let dim = embedding.tensorShape[1];
    let N = data.length / dim;
    if (embedding.tensorShape[0] > N) {
      logging.setWarningMessage(
          `Showing the first ${N.toLocaleString()}` +
          ` of ${embedding.tensorShape[0].toLocaleString()} data points`);
    }
    parseTensorsFromFloat32Array(data, dim).then(dataPoints => {
      callback(new DataSet(dataPoints));
    });
  };
  xhr.send();
}

export function parseRawTensors(
    content: string, callback: (ds: DataSet) => void) {
  parseTensors(content).then(data => {
    callback(new DataSet(data));
  });
}

export function parseRawMetadata(
    contents: string, callback: (r: SpriteAndMetadataInfo) => void) {
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
      };
      // If the first label is not a number, take it as the label.
      if (isNaN(row[0] as any) || numDim === row.length - 1) {
        dataPoint.metadata['label'] = row[0];
        dataPoint.vector = new Float32Array(row.slice(1).map(Number));
      } else {
        dataPoint.vector = new Float32Array(row.map(Number));
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

/** Parses a tsv text file. */
export function parseTensorsFromFloat32Array(data: Float32Array,
    dim: number): Promise<DataPoint[]> {
  return runAsyncTask('Parsing tensors...', () => {
    let N = data.length / dim;
    let dataPoints: DataPoint[] = [];
    let offset = 0;
    for (let i = 0; i < N; ++i) {
      dataPoints.push({
        metadata: {},
        vector: data.subarray(offset, offset + dim),
        index: i,
        projections: null,
      });
      offset += dim;
    }
    return dataPoints;
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

export function parseMetadata(content: string): Promise<SpriteAndMetadataInfo> {
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
    } as SpriteAndMetadataInfo;
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
    image.crossOrigin = '';
    image.src = url;
  });
}

export function retrieveSpriteAndMetadataInfo(metadataPath: string,
    spriteImagePath: string, spriteMetadata: SpriteMetadata,
    callback: (r: SpriteAndMetadataInfo) => void) {
  let metadataPromise: Promise<SpriteAndMetadataInfo> = Promise.resolve({});
  if (metadataPath) {
    metadataPromise = new Promise<SpriteAndMetadataInfo>((resolve, reject) => {
      logging.setModalMessage('Fetching metadata...', METADATA_MSG_ID);
      d3.text(metadataPath, (err: any, rawMetadata: string) => {
        if (err) {
          logging.setErrorMessage(err.responseText);
          reject(err);
          return;
        }
        resolve(parseMetadata(rawMetadata));
      });
    });
  }
  let spriteMsgId = null;
  let spritesPromise: Promise<HTMLImageElement> = null;
  if (spriteImagePath) {
    spriteMsgId = logging.setModalMessage('Fetching sprite image...');
    spritesPromise = fetchImage(spriteImagePath);
  }

  // Fetch the metadata and the image in parallel.
  Promise.all([metadataPromise, spritesPromise]).then(values => {
    if (spriteMsgId) {
      logging.setModalMessage(null, spriteMsgId);
    }
    let [metadata, spriteImage] = values;

    if (spriteImage && (spriteImage.height > MAX_SPRITE_IMAGE_SIZE_PX ||
                        spriteImage.width > MAX_SPRITE_IMAGE_SIZE_PX)) {
      logging.setModalMessage(
          `Error: Sprite image of dimensions ${spriteImage.width}px x ` +
          `${spriteImage.height}px exceeds maximum dimensions ` +
          `${MAX_SPRITE_IMAGE_SIZE_PX}px x ${MAX_SPRITE_IMAGE_SIZE_PX}px`);
    } else {
      metadata.spriteImage = spriteImage;
      metadata.spriteMetadata = spriteMetadata;
      callback(metadata);
    }
  });
}
