/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

 Licensed under the Apache License, Version 2.0 (the 'License');
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an 'AS IS' BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 =============================================================================*/
const svgNS = 'http://www.w3.org/2000/svg';

// The values below are the global defaults for labels, colors, etc. They
// have to be defined here, in order to be referenced as fallback values for
// the parameters of createChartGroup() and as initial values for the internal
// fields in the Polymer element.
let labels = ['-Inf', '-', '0', '+', '+Inf', 'NaN'];
let colors = ['#22b0dc', '#1c1c1c', '#808080 ', '#ddd', '#ffdb00', '#c00'];
let heightWidthRatio = 0.2;
let heightToFontRatio = 1 / 6;

export function attachChartGroup(
    data: Array<number>, width: number, parentElement: Element,
    _labels = labels, _colors = colors, _heightWidthRatio = heightWidthRatio,
    _heightToFontRatio = heightToFontRatio) {
  // Calculate the total of all entries in the data array.
  let dataSum = data.reduce(function(prevResult, currValue) {
    return prevResult + currValue;
  });
  let height = width * _heightWidthRatio;

  // Render data.
  let outerGroup = document.createElementNS(svgNS, 'g');
  Polymer.dom(parentElement).appendChild(outerGroup);

  let currentOffset = 0 as number;
  let currentDataLength = data.length as number;

  for (let slice = 0; slice < currentDataLength; slice++) {
    let sliceGroup = document.createElementNS(svgNS, 'g');
    Polymer.dom(outerGroup).appendChild(sliceGroup);
    let percentage = data[slice] / dataSum;
    let sliceWidth = percentage * width;

    // Create rectangle.
    let rect = createRect(currentOffset, sliceWidth, height, _colors[slice]);

    // Add to new rectangle to SVG.
    Polymer.dom(sliceGroup).appendChild(rect);

    // Add text to rectangle.
    let text = createText(
        currentOffset, sliceWidth, height, _labels[slice], _colors[slice],
        sliceGroup, _heightToFontRatio);

    // Add tooltip to rectangle and text.
    let rectTitle = createTitle(_labels[slice], data[slice]);
    let textTitle = createTitle(_labels[slice], data[slice]);
    Polymer.dom(rect).appendChild(rectTitle);
    Polymer.dom(text).appendChild(textTitle);

    currentOffset += sliceWidth;
  }
  return outerGroup;
}

function createText(
    currentOffset: number, sliceWidth: number, height: number, label: string,
    color: string, group: Element, heightToFontRatio: number) {
  // Add text to rectangle.
  let text = document.createElementNS(svgNS, 'text');
  Polymer.dom(group).appendChild(text);

  text.innerHTML = label;

  // Set location.
  text.setAttribute('x', (currentOffset + sliceWidth / 2).toString());
  text.setAttribute('y', (height / 2).toString());
  // Set text properties.
  let textColor = getTextColor(color);
  text.setAttribute('fill', textColor);
  // Center text.
  text.setAttribute('text-anchor', 'middle');
  text.setAttribute('dominant-baseline', 'middle');
  // Font size.
  let fontSize = height * heightToFontRatio;
  text.setAttribute('font-size', fontSize.toString());
  let textWidth = text.getBoundingClientRect().width;
  let textHeight = text.getBoundingClientRect().height;
  // Hide text if text is wider than the slice.
  if (textWidth > sliceWidth || textHeight > height || fontSize < 7) {
    text.innerHTML = '';
  }

  return text;
}

function createTitle(label: string, numberOfEntries: number) {
  let title = document.createElementNS(svgNS, 'title');
  title.innerHTML = label + ': ' + numberOfEntries.toString();

  return title;
}

function createRect(
    currentOffset: number, sliceWidth: number, height: number, color: string) {
  let rect = document.createElementNS(svgNS, 'rect') as HTMLElement;

  // Set location.
  rect.setAttribute('x', currentOffset.toString());
  rect.setAttribute('y', '0');
  // Set dimensions.
  rect.setAttribute('width', sliceWidth.toString());
  rect.setAttribute('height', height.toString());
  // Set colour.
  rect.setAttribute('fill', color);

  return rect;
}

function getTextColor(hexTripletColor: string) {
  let color = hexTripletColor;
  if (color.substring(0, 1) !== '#') {  // Lookup hex from name.
    let convertedHex = colorToHex(color);
    if (convertedHex) {
      color = convertedHex;
    } else {
      // RGB string is currently not handled.
      console.log(
          'WARNING: Could not convert color to hex,' +
          'please specify color as name or hex string.');
      return 'black';
    }
  }

  color = color.substring(1);  // Remove #.
  if (color.length === 3) {    // If short hex format is used.
    color = color.split('').reduce(
        // Double every character.
        function(initial: string, current: string) {
          return initial + current + current;
        },
        ''  // Initial value.
        );
  }

  let colorInt = parseInt(color, 16);   // Convert to integer.
  let r = (colorInt & 0xFF0000) >> 16;  // Extract each color component.
  let g = (colorInt & 0x00FF00) >> 8;
  let b = colorInt & 0x0000FF;
  // Calculate human perceptible luminance.
  let alpha = 1 - (0.299 * r + 0.587 * g + 0.114 * b) / 255;

  if (alpha < 0.5) {
    return 'black';
  } else {
    return 'white';
  }
}

/**
 * Renders a pixel using the provided color string to an unattached canvas,
 * then reads and returns the rgb image data of the pixel.
 *
 * @param color The color string to be converted.
 *
 * @example
 * // Returns [0, 0, 255, 255]
 * colorToRGBA('blue')
 *
 * @returns Returns the rgba colors as an array, i.e.
 * [r, g, b, a]
 */
function colorToRgba(color: string) {
  let canvas = document.createElement('canvas');
  canvas.height = 1;
  canvas.width = 1;

  let ctx = canvas.getContext('2d');
  ctx.fillStyle = color;
  ctx.fillRect(0, 0, 1, 1);
  let retArray = [] as number[];
  let imageData = ctx.getImageData(0, 0, 1, 1).data;
  imageData.forEach(function(d: number) {  // Copy data into standard Array.
    retArray.push(d);
  });
  return retArray;
}

/**
 * Turns a number (0-255) into a 2-character hex number (00-ff).
 * @param num
 *
 * @returns The converted string.
 */
function numToHex(num: number) {
  return ('0' + num.toString(16)).slice(-2);
}

/**
 * Converts any color string to its hex representation.
 * @param color The color string to be converted.
 *
 * @example
 * // Returns '#0000ff'
 * colorToHex('blue')
 *
 * @returns The hex color string.
 */
function colorToHex(color: string) {
  let rgba = colorToRgba(color) as Array<number|string>;
  return '#' +
      rgba.slice(0, 3)  // Remove alpha channel.
          .map(function(value: number) { return numToHex(value); })
          .join('');
}
Polymer({
  is: 'vz-data-summary',
  properties: {
    colors: {
      type: Array,
      // Has to match number of elements in the data array.
      observer: '_onColorsChange'
    },
    data: {type: Array, value: [], observer: '_onDataChange'},
    heightWidthRatio:
        {type: Number, value: heightWidthRatio, observer: '_onRatioChange'},
    heightToFontRatio: {
      type: Number,
      value: heightToFontRatio,
      observer: '_onFontRatioChange'
    },
    labels: {type: Array, observer: '_onLabelChange'},
    _colors: {
      type: Array,
      value: colors,
    },
    _data: Array,
    _labels: {type: Array, value: labels},
    _dataSum: {type: Number},
    _drawRequested: {type: Boolean, value: false},
    _height: {type: String},
    _isReady: {type: Boolean, value: false},
    _width: {type: Number}
  },
  behaviors: [Polymer.IronResizableBehavior],
  listeners: {'iron-resize': '_onWidthChange'},
  ready: function() {
    this._isReady = true;
    this._updateDimensions();
    // Trigger rendering if draw was requested before element was ready.
    if (this._drawRequested) {
      this._renderData();
    }
  },
  attached: function() {
    if (this._isReady) {
      this._updateInternalVariables();
      this._renderData();
    }
  },
  /**
   * Observer for this.colors.
   * @private
   */
  _onColorsChange: function() {
    // Verify passed array is valid.
    if (this._isColorsValid()) {
      // Copy over the array to the internal field if it has the correct
      // length.
      this._updateInternalVariables();
      this._renderData();
    }
  },
  /**
   * Data change handler, if new data is valid, sets flag to indicate data
   * now valid, updates the data extent and renders the new data.
   */
  _onDataChange: function() {
    // Validate new data.
    if (!this._isDataValid(this.data)) {
      return;
    }

    this._updateInternalVariables();  // Update the internal variables.

    // Calculate the sum.
    this._dataSum =
        this.data.reduce(function(prevResult: number, currValue: number) {
          return prevResult + currValue;
        });

    this._renderData();
  },
  /**
   * Observer for this.labels. Validates that this.labels is an array of
   * 6 elements before copying its contents to an internal array.
   */
  _onLabelChange: function() {
    if (this._isLabelsValid()) {
      this._updateInternalVariables();
      this._renderData();
    }
  },
  _onFontRatioChange: function() {
    if ((typeof this.heightToFontRatio) === 'number') {
      this._renderData();
    }
  },
  /**
   * Observer for this.heightWidthRatio.
   */
  _onRatioChange: function() {
    if (this.heightWidthRatio) {
      this._renderData();
    }
  },
  /**
   * Observer for width change. Depends on iron-resizable-behavior.
   */
  _onWidthChange: function() {
    this._width = this.$.summary.parentNode.width;
    this._renderData();
  },
  _internalFieldsValid: function() {
    return !!(this._data && this._labels && this._colors);
  },
  /**
   * Renders data, if element is ready, the dimensions are set, and valid
   * data exists.
   * @private
   */
  _renderData: function() {
    if (!this._isReady || !this._internalFieldsValid()) {
      this._drawRequested = true;
      return;
    }

    // Ensure dimensions are up-to-date and valid before starting rendering.
    this._updateDimensions();
    if (!this._width || !this._height) {
      return;
    }

    // Find element to append heat map to and determine dimensions.
    let svgSelection = this.$.summary as SVGElement;

    // Clear the SVG.
    this.resetSVG();

    // Render data.
    attachChartGroup(
        this._data, this._width, svgSelection, this._labels, this._colors,
        this.heightWidthRatio, this.heightToFontRatio);
  },
  /**
   * Verifies whether input data is valid.
   * @param [internal] {bool} Whether to check the internal or external
   * field.
   * @private
   */
  _isDataValid: function(internal: boolean = false) {
    return this._validateArrayByName('data', internal, isNumeric);
  },
  _isColorsValid: function(internal: boolean = false) {
    return this._validateArrayByName('colors', internal, isString);
  },
  _isLabelsValid: function(internal: boolean = false) {
    return this._validateArrayByName('labels', internal, isString);
  },
  _validateArrayByName: function(
      name: string, internal: boolean, typeChecker: CheckingFunctionType) {
    if (internal) {
      name = '_' + name;
    }
    return _isValidArray(this[name], typeChecker);
  },
  _updateInternalVariables: function() {
    // If all new data is valid.
    let isDataValid = this._isDataValid();
    let isColorsValid = this._isColorsValid();
    let isLabelsValid = this._isLabelsValid();

    if (isDataValid && isColorsValid && isLabelsValid) {
      // Ensure they all have the same length.
      let length = this.data.length;
      if (length === this.labels.length && length === this.colors.length) {
        this._data = this.data.slice();
        this._colors = this.colors.slice();
        this._labels = this.labels.slice();
      }
    } else {  // Check whether any new fields have the correct length.
      let internalDataValid = this._isDataValid(true);
      let internalColorsValid = this._isColorsValid(true);
      let internalLabelsValid = this._isLabelsValid(true);
      length = (internalDataValid && this._data.length) ||
          (internalColorsValid && this._colors.length) ||
          (internalLabelsValid && this._labels.length);

      this._updateIfSameLength(length, 'data');
      this._updateIfSameLength(length, 'colors');
      this._updateIfSameLength(length, 'labels');
    }
  },
  /**
   * Updates dimensions based on the width of the parent element.
   * @private
   */
  _updateDimensions: function() {
    let svgSelection = this.$.summary as SVGElement;

    // Recalculate height and width, and ensure data can be rendered.
    this._width = svgSelection.parentElement.clientWidth;
    this._height = this._width * this.heightWidthRatio;
    // Update the attribute height and width.
    svgSelection.setAttribute('height', this._height);
    svgSelection.setAttribute('width', this._width);
  },
  /**
   * Called when either colors, data, or labels are called. Copies calues
   * into internal fields iff the length of all three arrays is equal.
   */
  _updateIfSameLength: function(length: number, name: string) {
    if (this[name] && this[name].length === length) {
      this['_' + name] = this[name].slice();
    }
  },
  /**
   * Resets the SVG. Used by _renderData() but is exposed to provide the
   * user more control of the SVG's contents.
   */
  resetSVG: function() {
    // Reset the SVG.
    (this.$.summary as SVGElement).innerHTML = '';
  }
});

/**
 * Helper method for _isDataValid, _isColorsValid, _isLabelsValid. Check
 * whether the passed data is a) an array, b) the provided function
 * returns true for every element of the array.
 */
interface CheckingFunctionType {
  (value: any, index: number, array: any[]): boolean;
}

function _isValidArray(
    newData: Object[], typeCheckingFunction: CheckingFunctionType) {
  return Array.isArray(newData) &&
      // Returns true is every element in the array is numeric.
      newData.every(typeCheckingFunction);
}

function isNumeric(n: any) {
  return !isNaN(parseFloat(n)) && isFinite(n);
}

function isString(s: any) {
  return (typeof s) === 'string';
}
