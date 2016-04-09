/* Copyright 2015 Google Inc. All Rights Reserved.

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
module tf.graph.parser {

/**
 * Parses a native js value, which can be either a string, boolean or number.
 *
 * @param value The value to be parsed.
 */
function parseValue(value: string): string|number|boolean {
  if (value === "true") {
    return true;
  }
  if (value === "false") {
    return false;
  }
  let firstChar = value[0];
  if (firstChar === "\"") {
    return value.substring(1, value.length - 1);
  }
  let num = parseFloat(value);
  return isNaN(num) ? value : num;
}

/**
 * Fetches a text file and returns a promise of the result.
 */
export function fetchPbTxt(filepath: string): Promise<string> {
  return new Promise<string>(function(resolve, reject) {
    d3.text(filepath, function(error, text) {
      if (error) {
        reject(error);
        return;
      }
      resolve(text);
    });
  });
}

/**
 * Fetches the metadata file, parses it and returns a promise of the result.
 */
export function fetchAndParseMetadata(path: string, tracker: ProgressTracker) {
  return runTask("Reading metadata pbtxt", 40, () => {
    if (path == null) {
      return Promise.resolve(null);
    }
    return fetchPbTxt(path).then(text => new Blob([text]));
  }, tracker)
  .then((blob: Blob) => {
    return runTask("Parsing metadata.pbtxt", 60, () => {
      return blob != null ? parseStatsPbTxt(blob) : null;
    }, tracker);
  });
}

/**
 * Fetches the graph file, parses it and returns a promise of the result.
 */
export function fetchAndParseGraphData(path: string, pbTxtFile: Blob,
    tracker: ProgressTracker) {
  return runTask("Reading graph pbtxt", 40, () => {
    return pbTxtFile ?
      Promise.resolve(pbTxtFile) :
      fetchPbTxt(path).then(text => new Blob([text]));
  }, tracker)
  .then(blob => {
    return runTask("Parsing graph.pbtxt", 60, () => {
      return parseGraphPbTxt(blob);
    }, tracker);
  });
}

/**
 * Parse a file object in a streaming fashion line by line (or custom delim).
 * Can handle very large files.
 * @param input The file object
 * @param callback The callback called on each line
 * @param chunkSize The size of each read chunk. (optional)
 * @param delim The delimiter used to split a line. (optional)
 * @returns A promise for when it is finished.
 */
export function streamParse(file: Blob, callback: (string) => void,
    chunkSize: number = 1000000, delim: string = "\n"): Promise<boolean> {
  return new Promise<boolean>(function(resolve, reject) {
    let offset = 0;
    let fileSize = file.size - 1;
    let data = "";

    function readHandler(evt) {
      if (evt.target.error == null) {
        offset += evt.target.result.length;
        let str = evt.target.result;
        let parts = str.split(delim);
        let first = data + parts[0];
        if (parts.length === 1) {
          data = first;
          readChunk(offset, chunkSize);
          return;
        }
        data = parts[parts.length - 1];
        callback(first);
        for (let i = 1; i < parts.length - 1; i++) {
          callback(parts[i]);
        }
      } else {
        // read error
        reject(evt.target.error);
        return;
      }
      if (offset >= fileSize) {
        if (data) {
          callback(data);
        }
        resolve(true);
        return;
      }
      readChunk(offset, chunkSize);
    }

    function readChunk(offset: number, size: number) {
      var reader = new FileReader();
      var blob = file.slice(offset, offset + size);
      reader.onload = readHandler;
      reader.readAsText(blob);
    }

    readChunk(offset, chunkSize);
  });
}

/**
 * Since proto-txt doesn't explicitly say whether an attribute is repeated
 * (an array) or not, we keep a hard-coded list of attributes that are known
 * to be repeated. This list is used in parsing time to convert repeated
 * attributes into arrays even when the attribute only shows up once in the
 * object.
 */
const GRAPH_REPEATED_FIELDS: {[attrPath: string]: boolean} = {
  "node": true,
  "node.input": true,
  "node.attr": true,
  "node.attr.value.list.type": true,
  "node.attr.value.shape.dim": true,
  "node.attr.value.tensor.string_val": true,
  "node.attr.value.tensor.tensor_shape.dim": true,
  "node.attr.value.list.shape": true,
  "node.attr.value.list.shape.dim": true,
  "node.attr.value.list.s": true
};

const METADATA_REPEATED_FIELDS: {[attrPath: string]: boolean} = {
  "step_stats.dev_stats": true,
  "step_stats.dev_stats.node_stats": true,
  "step_stats.dev_stats.node_stats.output": true,
  "step_stats.dev_stats.node_stats.memory": true,
  "step_stats.dev_stats.node_stats.output.tensor_description.shape.dim": true
};

/**
 * Parses a blob of proto txt file into a raw Graph object.
 */
export function parseGraphPbTxt(input: Blob): Promise<TFNode[]> {
  return parsePbtxtFile(input, GRAPH_REPEATED_FIELDS).then(obj => obj["node"]);
}

/**
 * Parses a blob of proto txt file into a StepStats object.
 */
function parseStatsPbTxt(input: Blob): Promise<StepStats> {
  return parsePbtxtFile(input, METADATA_REPEATED_FIELDS)
    .then(obj => obj["step_stats"]);
}

/**
 * Parses a blob of proto txt file into javascript object.
 *
 * @param input The Blob or file object implementing slice.
 * @param repeatedFields Map (Set) of all the repeated fields, since you can't
 *   tell directly from the pbtxt if a field is repeated or not.
 * @returns The parsed object.
 */
function parsePbtxtFile(input: Blob,
    repeatedFields: {[attrPath: string]: boolean}): Promise<Object> {
  let output: { [name: string]: any; } = {};
  let stack = [];
  let path: string[] = [];
  let current: { [name: string]: any; } = output;

  function splitNameAndValueInAttribute(line: string) {
    let colonIndex = line.indexOf(":");
    let name = line.substring(0, colonIndex).trim();
    let value = parseValue(line.substring(colonIndex + 2).trim());
    return {
      name: name,
      value: value
    };
  }

  /**
   * Adds a value, given the attribute name and the host object. If the
   * attribute already exists, but is not an array, it will convert it to an
   * array of values.
   *
   * @param obj The host object that holds the attribute.
   * @param name The attribute name (key).
   * @param value The attribute value.
   * @param path A path that identifies the attribute. Used to check if
   *     an attribute is an array or not.
   */
  function addAttribute(obj: Object, name: string,
      value: Object|string|number|boolean, path: string[]): void {
    // We treat "node" specially since it is done so often.
    let existingValue = obj[name];
    if (existingValue == null) {
      obj[name] = path.join(".") in repeatedFields ? [value] : value;
    } else if (Array.isArray(existingValue)) {
      existingValue.push(value);
    } else {
      obj[name] = [existingValue, value];
    }
  }

  // Run through the file a line at a time.
  return streamParse(input, function(line: string) {
    if (!line) {
      return;
    }
    switch (line[line.length - 1]) {
      case "{": // create new object
        let name = line.substring(0, line.length - 2).trim();
        let newValue: { [name: string]: any; } = {};
        stack.push(current);
        path.push(name);
        addAttribute(current, name, newValue, path);
        current = newValue;
        break;
      case "}":
        current = stack.pop();
        path.pop();
        break;
      default:
        let x = splitNameAndValueInAttribute(line);
        addAttribute(current, x.name, x.value, path.concat(x.name));
        break;
    }
  }).then(function() {
    return output;
  });
}

} // Close module tf.graph.parser.
