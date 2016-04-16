module TF.Histogram {

 /**
  * Re-bins histogram data into uniform-width bins. Assumes a uniform distribution of values in given bins.
  *
  * @param {HistogramBin[]} bins - The original histogram data,
  * @param {number} numberOfBins - The number of uniform-width bins to split the data into.
  * @return {HistogramBin[]} - Re-binned histogram data. Does not modify original data, returns a new array.
  */
  export function rebinHistogram(bins: TF.Backend.HistogramBin[], numberOfBins: number) {
    if (bins.length === 0) {
      return [];
    }

    var oldBinsXExtent = [
      d3.min(bins, function(old: any) { return old.x; }),
      d3.max(bins, function(old: any) { return old.x + old.dx; })
    ];

    var newDx: number = (oldBinsXExtent[1] - oldBinsXExtent[0]) / numberOfBins;

    var newBins: TF.Backend.HistogramBin[] =
        d3.range(oldBinsXExtent[0], oldBinsXExtent[1], newDx)
            .map(function(newX) {

              // Take the count of each existing bin, multiply it by the
              // proportion of overlap with the new bin, then sum and store as
              // the count for new bin. If no overlap, will add zero, if 100%
              // overlap, will include full count into new bin.
              var newY = d3.sum(bins.map(function(old) {
                var intersectDx = Math.min(old.x + old.dx, newX + newDx) -
                    Math.max(old.x, newX);
                return (intersectDx > 0) ? (intersectDx / old.dx) * old.y : 0;
              }));

              return {x: newX, dx: newDx, y: newY};
            });

    return newBins;
  }
}
