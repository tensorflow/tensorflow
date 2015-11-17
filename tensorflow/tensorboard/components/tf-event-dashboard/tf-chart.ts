/// <reference path="../../typings/tsd.d.ts" />
/// <reference path="../../bower_components/plottable/plottable.d.ts" />

module TF {
  type TFDatum = [number, number, number];
  type tooltipMap = {[run: string]: string};
  export type TooltipUpdater = (tooltipMap, xValue, closestRun) => void;

  let Y_TOOLTIP_FORMATTER_PRECISION = 4;
  let STEP_AXIS_FORMATTER_PRECISION = 4;
  let Y_AXIS_FORMATTER_PRECISION = 3;

  export class BaseChart {
    protected dataCoordinator: TF.DataCoordinator;
    protected tag: string;
    protected tooltipUpdater: TooltipUpdater;

    protected xAccessor: Plottable.Accessor<number | Date>;
    protected xScale: Plottable.QuantitativeScale<number | Date>;
    protected yScale: Plottable.QuantitativeScale<number>;
    protected gridlines: Plottable.Components.Gridlines;
    protected center: Plottable.Components.Group;
    protected xAxis: Plottable.Axes.Numeric | Plottable.Axes.Time;
    protected yAxis: Plottable.Axes.Numeric;
    protected xLabel: Plottable.Components.AxisLabel;
    protected yLabel: Plottable.Components.AxisLabel;
    protected outer: Plottable.Components.Table;
    protected colorScale: Plottable.Scales.Color;
    protected xTooltipFormatter: (d: number) => string;
    constructor(
        tag: string,
        dataCoordinator: TF.DataCoordinator,
        tooltipUpdater: TooltipUpdater,
        xType: string,
        colorScale: Plottable.Scales.Color
      ) {
      this.dataCoordinator = dataCoordinator;
      this.tag = tag;
      this.colorScale = colorScale;
      this.tooltipUpdater = tooltipUpdater;
      this.buildChart(xType);
    }

    public changeRuns(runs: string[]) {
      throw new Error("Abstract method not implemented");
    }

    protected addCrosshairs(plot: Plottable.XYPlot<number | Date, number>, yAccessor): Plottable.Components.Group {
      var pi = new Plottable.Interactions.Pointer();
      pi.attachTo(plot);
      let xGuideLine = new Plottable.Components.GuideLineLayer<void>("vertical");
      let yGuideLine = new Plottable.Components.GuideLineLayer<void>("horizontal");
      xGuideLine.addClass("crosshairs");
      yGuideLine.addClass("crosshairs");
      var group = new Plottable.Components.Group([plot, xGuideLine, yGuideLine]);
      let yfmt = multiscaleFormatter(Y_TOOLTIP_FORMATTER_PRECISION);

      pi.onPointerMove((p: Plottable.Point) => {
        let run2val: {[run: string]: string} = {};
        let x: number = this.xScale.invert(p.x).valueOf();
        let yMin: number = this.yScale.domain()[0];
        let yMax: number = this.yScale.domain()[1];
        let closestRun: string = null;
        let minYDistToRun: number = Infinity;
        let yValueForCrosshairs: number = p.y;
        plot.datasets().forEach((dataset) => {
          let run: string = dataset.metadata().run;
          let data: TFDatum[] = dataset.data();
          let xs: number[] = data.map((d, i) => this.xAccessor(d, i, dataset).valueOf());
          let idx: number = _.sortedIndex(xs, x);
          if (idx === 0 || idx === data.length) {
            // Only find a point when the cursor is inside the range of the data
            // if the cursor is to the left or right of all the data, dont attach.
            return;
          }
          let previous = data[idx - 1];
          let next = data[idx];
          let x0: number = this.xAccessor(previous, idx - 1, dataset).valueOf();
          let x1: number = this.xAccessor(next, idx, dataset).valueOf();
          let y0: number = yAccessor(previous, idx - 1, dataset).valueOf();
          let y1: number = yAccessor(next, idx, dataset).valueOf();
          let slope: number = (y1 - y0) / (x1 - x0);
          let y: number = y0 + slope * (x - x0);

          if (y < yMin || y > yMax || y !== y) {
            // don't find data that is off the top or bottom of the plot.
            // also don't find data if it is NaN
            return;
          }
          let dist = Math.abs(this.yScale.scale(y) - p.y);
          if (dist < minYDistToRun) {
            minYDistToRun = dist;
            closestRun = run;
            yValueForCrosshairs = this.yScale.scale(y);
          }
          // Note this tooltip will display linearly interpolated values
          // e.g. will display a y=0 value halfway between [y=-1, y=1], even
          // though there is not actually any 0 datapoint. This could be misleading
          run2val[run] = yfmt(y);
        });
        xGuideLine.pixelPosition(p.x);
        yGuideLine.pixelPosition(yValueForCrosshairs);
        this.tooltipUpdater(run2val, this.xTooltipFormatter(x), closestRun);

      });

      pi.onPointerExit(() => {
        this.tooltipUpdater(null, null, null);
        xGuideLine.pixelPosition(-1);
        yGuideLine.pixelPosition(-1);
      });

      return group;

    }

    protected buildChart(xType: string) {
      if (this.outer) {
        this.outer.destroy();
      }
      var xComponents = getXComponents(xType);
      this.xAccessor = xComponents.accessor;
      this.xScale = xComponents.scale;
      this.xAxis = xComponents.axis;
      this.xAxis.margin(0).tickLabelPadding(3);
      this.xTooltipFormatter = xComponents.tooltipFormatter;
      this.yScale = new Plottable.Scales.Linear();
      this.yAxis = new Plottable.Axes.Numeric(this.yScale, "left");
      let yFormatter = multiscaleFormatter(Y_AXIS_FORMATTER_PRECISION);
      this.yAxis.margin(0).tickLabelPadding(5).formatter(yFormatter);
      this.yAxis.usesTextWidthApproximation(true);

      var center = this.buildPlot(this.xAccessor, this.xScale, this.yScale);

      this.gridlines = new Plottable.Components.Gridlines(this.xScale, this.yScale);

      var dzl = new Plottable.DragZoomLayer(this.xScale, this.yScale);

      this.center = new Plottable.Components.Group([center, this.gridlines, dzl]);
      this.outer =  new Plottable.Components.Table([
                                                   [this.yAxis, this.center],
                                                   [null, this.xAxis]
                                                  ]);
    }

    protected buildPlot(xAccessor, xScale, yScale): Plottable.Component {
      throw new Error("Abstract method not implemented.");
    }

    public renderTo(target: d3.Selection<any>) {
      this.outer.renderTo(target);
    }

    public redraw() {
      this.outer.redraw();
    }

    protected destroy() {
      this.outer.destroy();
    }
  }

  export class LineChart extends BaseChart {
    private plot: Plottable.Plots.Line<number | Date>;
    protected buildPlot(xAccessor, xScale, yScale): Plottable.Component {
      var yAccessor = accessorize("2");
      var plot = new Plottable.Plots.Line<number | Date>();
      plot.x(xAccessor, xScale);
      plot.y(yAccessor, yScale);
      plot.attr("stroke", (d: any, i: number, m: any) => m.run, this.colorScale);
      this.plot = plot;
      var group = this.addCrosshairs(plot, yAccessor);
      return group;
    }

    public changeRuns(runs: string[]) {
      var datasets = this.dataCoordinator.getDatasets(this.tag, runs);
      this.plot.datasets(datasets);
    }

  }

  export class HistogramChart extends BaseChart {
    private plots: Plottable.XYPlot<number | Date, number>[];

    public changeRuns(runs: string[]) {
      var datasets = this.dataCoordinator.getDatasets(this.tag, runs);
      this.plots.forEach((p) => p.datasets(datasets));
    }

    protected buildPlot(xAccessor, xScale, yScale): Plottable.Component {
      var percents =  [0, 228, 1587, 3085, 5000, 6915, 8413, 9772, 10000];
      var opacities = _.range(percents.length - 1).map((i) => (percents[i + 1] - percents[i]) / 2500);
      var accessors = percents.map((p, i) => (datum) => datum[2][i][1]);
      var median = 4;
      var medianAccessor = accessors[median];

      var plots = _.range(accessors.length - 1).map((i) => {
        var p = new Plottable.Plots.Area<number | Date>();
        p.x(xAccessor, xScale);

        var y0 = i > median ? accessors[i] : accessors[i + 1];
        var y  = i > median ? accessors[i + 1] : accessors[i];
        p.y(y, yScale);
        p.y0(y0);
        p.attr("fill", (d: any, i: number, m: any) => m.run, this.colorScale);
        p.attr("stroke", (d: any, i: number, m: any) => m.run, this.colorScale);
        p.attr("stroke-weight", (d: any, i: number, m: any) => "0.5px");
        p.attr("stroke-opacity", () => opacities[i]);
        p.attr("fill-opacity", () => opacities[i]);
        return p;
      });

      var medianPlot = new Plottable.Plots.Line<number | Date>();
      medianPlot.x(xAccessor, xScale);
      medianPlot.y(medianAccessor, yScale);
      medianPlot.attr("stroke", (d: any, i: number, m: any) => m.run, this.colorScale);

      this.plots = plots;
      var group = this.addCrosshairs(medianPlot, medianAccessor);
      return new Plottable.Components.Group([new Plottable.Components.Group(plots), group]);
    }
  }

  /* Create a formatter function that will switch between exponential and
   * regular display depending on the scale of the number being formatted,
   * and show `digits` significant digits.
   */
  function multiscaleFormatter(digits: number): ((v: number) => string) {
    return (v: number) => {
      var absv = Math.abs(v);
      if (absv < 1E-15) {
        // Sometimes zero-like values get an annoying representation
        absv = 0;
      }
      var f: (x: number) => string;
      if (absv >= 1E4) {
        f = d3.format("." + digits + "e");
      } else if (absv > 0 && absv < 0.01) {
        f = d3.format("." + digits + "e");
      } else {
        f = d3.format("." + digits + "g");
      }
      return f(v);
    };
  }

  function accessorize(key: string): Plottable.Accessor<number> {
    return (d: any, index: number, dataset: Plottable.Dataset) => d[key];
  }

  interface XComponents {
    /* tslint:disable */
    scale: Plottable.Scales.Linear | Plottable.Scales.Time,
    axis: Plottable.Axes.Numeric | Plottable.Axes.Time,
    accessor: Plottable.Accessor<number | Date>,
    tooltipFormatter: (d: number) => string;
    /* tslint:enable */
  }

  function stepX(): XComponents {
    var scale = new Plottable.Scales.Linear();
    var axis = new Plottable.Axes.Numeric(scale, "bottom");
    var formatter = Plottable.Formatters.siSuffix(STEP_AXIS_FORMATTER_PRECISION);
    axis.formatter(formatter);
    return {
      scale: scale,
      axis: axis,
      accessor: accessorize("1"),
      tooltipFormatter: formatter,
    };
  }

  function wallX(): XComponents {
    var scale = new Plottable.Scales.Time();
    var formatter = Plottable.Formatters.time("%a %b %e, %H:%M:%S");
    return {
      scale: scale,
      axis: new Plottable.Axes.Time(scale, "bottom"),
      accessor: (d: any, index: number, dataset: Plottable.Dataset) => {
        return d[0] * 1000; // convert seconds to ms
      },
      tooltipFormatter: (d: number) => formatter(new Date(d)),
    };
  }

  function relativeX(): XComponents {
    var scale = new Plottable.Scales.Linear();
    var formatter = (n: number) => {
      var days = Math.floor(n / 24);
      n -= (days * 24);
      var hours = Math.floor(n);
      n -= hours;
      n *= 60;
      var minutes = Math.floor(n);
      n -= minutes;
      n *= 60;
      var seconds = Math.floor(n);
      return days + "d " + hours + "h " + minutes + "m " + seconds + "s";
    };
    return {
      scale: scale,
      axis: new Plottable.Axes.Numeric(scale, "bottom"),
      accessor: (d: any, index: number, dataset: Plottable.Dataset) => {
        var data = dataset && dataset.data();
        // I can't imagine how this function would be called when the data is empty
        // (after all, it iterates over the data), but lets guard just to be safe.
        var first = data.length > 0 ? data[0][0] : 0;
        return (d[0] - first) / (60 * 60); // convert seconds to hours
      },
      tooltipFormatter: formatter,
    };
  }

  function getXComponents(xType: string): XComponents {
    switch (xType) {
      case "step":
        return stepX();
      case "wall_time":
        return wallX();
      case "relative":
        return relativeX();
      default:
        throw new Error("invalid xType: " + xType);
    }
  }
}
