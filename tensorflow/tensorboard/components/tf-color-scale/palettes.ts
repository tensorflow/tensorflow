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
      '#c53929',  // google red 700
      '#ff7043',  // deep orange 400
      '#f7cb4d',  // google yellow 300
      '#0b8043',  // google green 700
      '#80deea',  // cyan 200
      '#4285f4',  // google blue 500
      '#5e35b1'   // deep purple 600
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
    ],
    // This rainbow palette attempts to keep a constant brightness across hues.
    constantValue: [
      '#f44336', '#ffa216', '#c2d22d', '#51b455', '#1ca091', '#505ec4',
      '#a633ba'
    ]
  };
}
