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

module TF {
  export const palettes = {
    googleStandard: [
      '#db4437',  // google red 500
      '#ff7043',  // deep orange 400
      '#f4b400',  // google yellow 500
      '#0f9d58',  // google green 500
      '#00796b',  // teal 700
      '#00acc1',  // cyan 600
      '#4285f4',  // google blue 500
      '#5c6bc0',  // indigo 400
      '#ab47bc'   // purple 400
    ],
    googleCool: [
      '#9e9d24',  // lime 800
      '#0f9d58',  // google green 500
      '#00796b',  // teal 700
      '#00acc1',  // cyan 600
      '#4285f4',  // google blue 500
      '#5c6bc0',  // indigo 400
      '#607d8b'   // blue gray 500
    ],
    googleWarm: [
      '#795548',  // brown 500
      '#ab47bc',  // purple 400
      '#f06292',  // pink 300
      '#c2185b',  // pink 700
      '#db4437',  // google red 500
      '#ff7043',  // deep orange 400
      '#f4b400'   // google yellow 700
    ],
    googleColorBlindAssist: [
      '#ff7043',  // orange
      '#00ACC1',  // dark cyan
      '#AB47BC',  // bright purple
      '#2A56C6',  // dark blue
      '#0b8043',  // green
      '#F7CB4D',  // yellow
      '#c0ca33',  // lime
      '#5e35b1',  // purple
      '#A52714',  // red
    ],
    // These palettes try to be better for color differentiation.
    // https://personal.sron.nl/~pault/
    colorBlindAssist1:
        ['#4477aa', '#44aaaa', '#aaaa44', '#aa7744', '#aa4455', '#aa4488'],
    colorBlindAssist2: [
      '#88ccee', '#44aa99', '#117733', '#999933', '#ddcc77', '#cc6677',
      '#882255', '#aa4499'
    ],
    colorBlindAssist3: [
      '#332288', '#6699cc', '#88ccee', '#44aa99', '#117733', '#999933',
      '#ddcc77', '#cc6677', '#aa4466', '#882255', '#661100', '#aa4499'
    ],
    // based on this palette: http://mkweb.bcgsc.ca/biovis2012/
    colorBlindAssist4: [
      '#FF6DB6', '#920000', '#924900', '#DBD100', '#24FF24', '#006DDB',
      '#490092'
    ],
    mldash: [
      '#E47EAD', '#F4640D', '#FAA300', '#F5E636', '#00A077', '#0077B8',
      '#00B7ED'
    ]
  };
}
