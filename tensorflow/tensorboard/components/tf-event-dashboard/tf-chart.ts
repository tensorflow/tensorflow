/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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
  export type DataFn = (run: string, tag: string) =>
      Promise<Array<Backend.Datum>>;

  let Y_TOOLTIP_FORMATTER_PRECISION = 4;
  let STEP_FORMATTER_PRECISION = 4;
  let Y_AXIS_FORMATTER_PRECISION = 3;
  let TOOLTIP_Y_PIXEL_OFFSET = 20;
  let TOOLTIP_CIRCLE_SIZE = 4;
  let NAN_SYMBOL_SIZE = 6;

  interface Point {
    x: number;  // pixel space
    y: number;  // pixel space
    datum: TF.Backend.ScalarDatum;
    dataset: Plottable.Dataset;
  }

  export class BaseChart {
    protected dataFn: DataFn;
    protected tag: string;
    private run2datasets: {[run: string]: Plottable.Dataset};
    protected runs: string[];

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
    protected tooltip: d3.Selection<any>;
    protected dzl: Plottable.DragZoomLayer;
    constructor(
        tag: string, dataFn: DataFn, xType: string,
        colorScale: Plottable.Scales.Color, tooltip: d3.Selection<any>) {
      this.dataFn = dataFn;
      this.run2datasets = {};
      this.tag = tag;
      this.colorScale = colorScale;
      this.tooltip = tooltip;
    }

    /**
     * Change the runs on the chart. The work of actually setting the dataset
     * on the plot is deferred to the subclass because it is impl-specific.
     * Changing runs automatically triggers a reload; this ensures that the
     * newly selected run will have data, and that all the runs will be current
     * (it would be weird if one run was ahead of the others, and the display
     * depended on the order in which runs were added)
     */
    public changeRuns(runs: string[]) {
      this.runs = runs;
      this.reload();
    }

    /**
     * Reload data for each run in view.
     */
    public reload() {
      this.runs.forEach((run) => {
        let dataset = this.getDataset(run);
        this.dataFn(this.tag, run).then((x) => dataset.data(x));
      });
    }

    protected getDataset(run: string) {
      if (this.run2datasets[run] === undefined) {
        this.run2datasets[run] =
            new Plottable.Dataset([], {run: run, tag: this.tag});
      }
      return this.run2datasets[run];
    }

    protected buildChart(xType: string) {
      if (this.outer) {
        this.outer.destroy();
      }
      let xComponents = getXComponents(xType);
      this.xAccessor = xComponents.accessor;
      this.xScale = xComponents.scale;
      this.xAxis = xComponents.axis;
      this.xAxis.margin(0).tickLabelPadding(3);
      this.yScale = new Plottable.Scales.Linear();
      this.yAxis = new Plottable.Axes.Numeric(this.yScale, 'left');
      let yFormatter = multiscaleFormatter(Y_AXIS_FORMATTER_PRECISION);
      this.yAxis.margin(0).tickLabelPadding(5).formatter(yFormatter);
      this.yAxis.usesTextWidthApproximation(true);

      this.dzl = new Plottable.DragZoomLayer(this.xScale, this.yScale);

      let center = this.buildPlot(this.xAccessor, this.xScale, this.yScale);

      this.gridlines =
          new Plottable.Components.Gridlines(this.xScale, this.yScale);

      this.center =
          new Plottable.Components.Group([this.gridlines, center, this.dzl]);
      this.outer =  new Plottable.Components.Table([
                                                   [this.yAxis, this.center],
                                                   [null, this.xAxis]
                                                  ]);
    }

    protected buildPlot(xAccessor, xScale, yScale): Plottable.Component {
      throw new Error('Abstract method not implemented.');
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
    private linePlot: Plottable.Plots.Line<number|Date>;
    private scatterPlot: Plottable.Plots.Scatter<number|Date, Number>;
    private nanDisplay: Plottable.Plots.Scatter<number|Date, Number>;
    private yAccessor: Plottable.Accessor<number>;
    private lastPointsDataset: Plottable.Dataset;
    private datasets: Plottable.Dataset[];
    private updateSpecialDatasets;
    private nanDataset: Plottable.Dataset;

    constructor(
        tag: string, dataFn: DataFn, xType: string,
        colorScale: Plottable.Scales.Color, tooltip: d3.Selection<any>) {
      super(tag, dataFn, xType, colorScale, tooltip);
      this.datasets = [];
      // lastPointDataset is a dataset that contains just the last point of
      // every dataset we're currently drawing.
      this.lastPointsDataset = new Plottable.Dataset();
      this.nanDataset = new Plottable.Dataset();
      // need to do a single bind, so we can deregister the callback from
      // old Plottable.Datasets. (Deregistration is done by identity checks.)
      this.updateSpecialDatasets = this._updateSpecialDatasets.bind(this);
      this.buildChart(xType);
    }
    protected buildPlot(xAccessor, xScale, yScale): Plottable.Component {
      this.yAccessor = (d: Backend.ScalarDatum) => d.scalar;
      let linePlot = new Plottable.Plots.Line<number|Date>();
      linePlot.x(xAccessor, xScale);
      linePlot.y(this.yAccessor, yScale);
      linePlot.attr(
          'stroke', (d: Backend.Datum, i: number, dataset: Plottable.Dataset) =>
                        this.colorScale.scale(dataset.metadata().run));
      this.linePlot = linePlot;
      let group = this.setupTooltips(linePlot);

      // The scatterPlot will display the last point for each dataset.
      // This way, if there is only one datum for the series, it is still
      // visible. We hide it when tooltips are active to keep things clean.
      let scatterPlot = new Plottable.Plots.Scatter<number|Date, number>();
      scatterPlot.x(xAccessor, xScale);
      scatterPlot.y(this.yAccessor, yScale);
      scatterPlot.attr('fill', (d: any) => this.colorScale.scale(d.run));
      scatterPlot.attr('opacity', 1);
      scatterPlot.size(TOOLTIP_CIRCLE_SIZE * 2);
      scatterPlot.datasets([this.lastPointsDataset]);
      this.scatterPlot = scatterPlot;

      let nanDisplay = new Plottable.Plots.Scatter<number|Date, number>();
      nanDisplay.x(xAccessor, xScale);
      nanDisplay.y((x) => x.displayY, yScale);
      nanDisplay.attr('fill', (d: any) => this.colorScale.scale(d.run));
      nanDisplay.attr('opacity', 1);
      nanDisplay.size(NAN_SYMBOL_SIZE * 2);
      nanDisplay.datasets([this.nanDataset]);
      nanDisplay.symbol(Plottable.SymbolFactories.triangleUp);
      this.nanDisplay = nanDisplay;
      return new Plottable.Components.Group([nanDisplay, scatterPlot, group]);
    }

    /** Constructs special datasets. Each special dataset contains exceptional
     * values from all of the regular datasetes, e.g. last points in series, or
     * NaN values. Those points will have a `run` and `relative` property added
     * (since usually those are context in the surrounding dataset).
     */
    private _updateSpecialDatasets() {
      let lastPointsData =
          this.datasets
              .map((d) => {
                let datum = null;
                // filter out NaNs to ensure last point is a clean one
                let nonNanData = d.data().filter((x) => !isNaN(x.scalar));
                if (nonNanData.length > 0) {
                  let idx = nonNanData.length - 1;
                  datum = nonNanData[idx];
                  datum.run = d.metadata().run;
                  datum.relative = relativeAccessor(datum, -1, d);
                }
                return datum;
              })
              .filter((x) => x != null);
      this.lastPointsDataset.data(lastPointsData);

      // Take a dataset, return an array of NaN data points
      // the NaN points will have a "displayY" property which is the
      // y-value of a nearby point that was not NaN (0 if all points are NaN)
      let datasetToNaNData = (d: Plottable.Dataset) => {
        let displayY = null;
        let data = d.data();
        let i = 0;
        while (i < data.length && displayY == null) {
          if (!isNaN(data[i].scalar)) {
            displayY = data[i].scalar;
          }
          i++;
        }
        if (displayY == null) {
          displayY = 0;
        }
        let nanData = [];
        for (i = 0; i < data.length; i++) {
          if (!isNaN(data[i].scalar)) {
            displayY = data[i].scalar;
          } else {
            data[i].run = d.metadata().run;
            data[i].displayY = displayY;
            data[i].relative = relativeAccessor(data[i], -1, d);
            nanData.push(data[i]);
          }
        }
        return nanData;
      };
      let nanData = _.flatten(this.datasets.map(datasetToNaNData));
      this.nanDataset.data(nanData);
    }

    private setupTooltips(plot: Plottable.XYPlot<number|Date, number>):
        Plottable.Components.Group {
      let pi = new Plottable.Interactions.Pointer();
      pi.attachTo(plot);
      // PointsComponent is a Plottable Component that will hold the little
      // circles we draw over the closest data points
      let pointsComponent = new Plottable.Component();
      let group = new Plottable.Components.Group([plot, pointsComponent]);

      let hideTooltips = () => {
        this.tooltip.style('opacity', 0);
        this.scatterPlot.attr('opacity', 1);
        pointsComponent.content().selectAll('.point').remove();
      };

      let enabled = true;
      let disableTooltips = () => {
        enabled = false;
        hideTooltips();
      };
      let enableTooltips = () => { enabled = true; };

      this.dzl.interactionStart(disableTooltips);
      this.dzl.interactionEnd(enableTooltips);

      pi.onPointerMove((p: Plottable.Point) => {
        if (!enabled) {
          return;
        }
        let target: Point = {
          x: p.x,
          y: p.y,
          datum: null,
          dataset: null,
        };

        let centerBBox: SVGRect =
            (<any>this.gridlines.content().node()).getBBox();
        let points = plot.datasets().map(
            (dataset) => this.findClosestPoint(target, dataset));
        let pointsToCircle = points.filter(
            (p) => p != null &&
                Plottable.Utils.DOM.intersectsBBox(p.x, p.y, centerBBox));
        let pts: any = pointsComponent.content().selectAll('.point').data(
            pointsToCircle, (p: Point) => p.dataset.metadata().run);
        if (points.length !== 0) {
          pts.enter().append('circle').classed('point', true);
          pts.attr('r', TOOLTIP_CIRCLE_SIZE)
              .attr('cx', (p) => p.x)
              .attr('cy', (p) => p.y)
              .style('stroke', 'none')
              .attr(
                  'fill',
                  (p) => this.colorScale.scale(p.dataset.metadata().run));
          pts.exit().remove();
          this.drawTooltips(points, target);
        } else {
          hideTooltips();
        }
      });

      pi.onPointerExit(hideTooltips);

      return group;
    }

    private drawTooltips(points: Point[], target: Point) {
      // Formatters for value, step, and wall_time
      this.scatterPlot.attr('opacity', 0);
      let valueFormatter = multiscaleFormatter(Y_TOOLTIP_FORMATTER_PRECISION);

      let dist = (p: Point) =>
          Math.pow(p.x - target.x, 2) + Math.pow(p.y - target.y, 2);
      let closestDist = _.min(points.map(dist));
      points = _.sortBy(points, (d) => d.dataset.metadata().run);

      let rows = this.tooltip.select('tbody')
                     .html('')
                     .selectAll('tr')
                     .data(points)
                     .enter()
                     .append('tr');
      // Grey out the point if any of the following are true:
      // - The cursor is outside of the x-extent of the dataset
      // - The point is rendered above or below the screen
      // - The point's y value is NaN
      rows.classed('distant', (d) => {
        let firstPoint = d.dataset.data()[0];
        let lastPoint = _.last(d.dataset.data());
        let firstX =
            this.xScale.scale(this.xAccessor(firstPoint, 0, d.dataset));
        let lastX = this.xScale.scale(this.xAccessor(lastPoint, 0, d.dataset));
        let s = d.datum.scalar;
        let yD = this.yScale.domain();
        return target.x < firstX || target.x > lastX || s < yD[0] ||
            s > yD[1] || isNaN(s);
      });
      rows.classed('closest', (p) => dist(p) === closestDist);
      // It is a bit hacky that we are manually applying the width to the swatch
      // and the nowrap property to the text here. The reason is as follows:
      // the style gets updated asynchronously by Polymer scopeSubtree observer.
      // Which means we would get incorrect sizing information since the text
      // would wrap by default. However, we need correct measurements so that
      // we can stop the text from falling off the edge of the screen.
      // therefore, we apply the size-critical styles directly.
      rows.style('white-space', 'nowrap');
      rows.append('td')
          .append('span')
          .classed('swatch', true)
          .style(
              'background-color',
              (d) => this.colorScale.scale(d.dataset.metadata().run));
      rows.append('td').text((d) => d.dataset.metadata().run);
      rows.append('td').text(
          (d) =>
              isNaN(d.datum.scalar) ? 'NaN' : valueFormatter(d.datum.scalar));
      rows.append('td').text((d) => stepFormatter(d.datum.step));
      rows.append('td').text((d) => timeFormatter(d.datum.wall_time));
      rows.append('td').text(
          (d) => relativeFormatter(relativeAccessor(d.datum, -1, d.dataset)));

      // compute left position
      let documentWidth = document.body.clientWidth;
      let node: any = this.tooltip.node();
      let parentRect = node.parentElement.getBoundingClientRect();
      let nodeRect = node.getBoundingClientRect();
      // prevent it from falling off the right side of the screen
      let left =
          Math.min(0, documentWidth - parentRect.left - nodeRect.width - 60);
      this.tooltip.style('left', left + 'px');
      // compute top position
      if (parentRect.bottom + nodeRect.height + TOOLTIP_Y_PIXEL_OFFSET <
          document.body.clientHeight) {
        this.tooltip.style('top', parentRect.bottom + TOOLTIP_Y_PIXEL_OFFSET);
      } else {
        this.tooltip.style('bottom', parentRect.top - TOOLTIP_Y_PIXEL_OFFSET);
      }

      this.tooltip.style('opacity', 1);
    }

    private findClosestPoint(target: Point, dataset: Plottable.Dataset): Point {
      let points: Point[] = dataset.data().map((d, i) => {
        let x = this.xAccessor(d, i, dataset);
        let y = this.yAccessor(d, i, dataset);
        return {
          x: this.xScale.scale(x),
          y: this.yScale.scale(y),
          datum: d,
          dataset: dataset,
        };
      });
      let idx: number = _.sortedIndex(points, target, (p: Point) => p.x);
      if (idx === points.length) {
        return points[points.length - 1];
      } else if (idx === 0) {
        return points[0];
      } else {
        let prev = points[idx - 1];
        let next = points[idx];
        let prevDist = Math.abs(prev.x - target.x);
        let nextDist = Math.abs(next.x - target.x);
        return prevDist < nextDist ? prev : next;
      }
    }

    public changeRuns(runs: string[]) {
      super.changeRuns(runs);
      runs.reverse();  // draw first run on top
      this.datasets.forEach((d) => d.offUpdate(this.updateSpecialDatasets));
      this.datasets = runs.map((r) => this.getDataset(r));
      this.datasets.forEach((d) => d.onUpdate(this.updateSpecialDatasets));
      this.linePlot.datasets(this.datasets);
    }
  }

  export class HistogramChart extends BaseChart {
    private plots: Plottable.XYPlot<number | Date, number>[];
    constructor(
        tag: string, dataFn: DataFn, xType: string,
        colorScale: Plottable.Scales.Color, tooltip: d3.Selection<any>) {
      super(tag, dataFn, xType, colorScale, tooltip);
      this.buildChart(xType);
    }

    public changeRuns(runs: string[]) {
      super.changeRuns(runs);
      let datasets = runs.map((r) => this.getDataset(r));
      this.plots.forEach((p) => p.datasets(datasets));
    }

    protected buildPlot(xAccessor, xScale, yScale): Plottable.Component {
      let percents = [0, 228, 1587, 3085, 5000, 6915, 8413, 9772, 10000];
      let opacities = _.range(percents.length - 1)
                          .map((i) => (percents[i + 1] - percents[i]) / 2500);
      let accessors = percents.map((p, i) => (datum) => datum[i][1]);
      let median = 4;
      let medianAccessor = accessors[median];

      let plots = _.range(accessors.length - 1).map((i) => {
        let p = new Plottable.Plots.Area<number|Date>();
        p.x(xAccessor, xScale);

        let y0 = i > median ? accessors[i] : accessors[i + 1];
        let y = i > median ? accessors[i + 1] : accessors[i];
        p.y(y, yScale);
        p.y0(y0);
        p.attr(
            'fill', (d: any, i: number, dataset: Plottable.Dataset) =>
                        this.colorScale.scale(dataset.metadata().run));
        p.attr(
            'stroke', (d: any, i: number, dataset: Plottable.Dataset) =>
                          this.colorScale.scale(dataset.metadata().run));
        p.attr('stroke-weight', (d: any, i: number, m: any) => '0.5px');
        p.attr('stroke-opacity', () => opacities[i]);
        p.attr('fill-opacity', () => opacities[i]);
        return p;
      });

      let medianPlot = new Plottable.Plots.Line<number|Date>();
      medianPlot.x(xAccessor, xScale);
      medianPlot.y(medianAccessor, yScale);
      medianPlot.attr(
          'stroke',
          (d: any, i: number, m: any) => this.colorScale.scale(m.run));

      this.plots = plots;
      return new Plottable.Components.Group(plots);
    }
  }

  /* Create a formatter function that will switch between exponential and
   * regular display depending on the scale of the number being formatted,
   * and show `digits` significant digits.
   */
  function multiscaleFormatter(digits: number): ((v: number) => string) {
    return (v: number) => {
      let absv = Math.abs(v);
      if (absv < 1E-15) {
        // Sometimes zero-like values get an annoying representation
        absv = 0;
      }
      let f: (x: number) => string;
      if (absv >= 1E4) {
        f = d3.format('.' + digits + 'e');
      } else if (absv > 0 && absv < 0.01) {
        f = d3.format('.' + digits + 'e');
      } else {
        f = d3.format('.' + digits + 'g');
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
    /* tslint:enable */
  }

  let stepFormatter = Plottable.Formatters.siSuffix(STEP_FORMATTER_PRECISION);
  function stepX(): XComponents {
    let scale = new Plottable.Scales.Linear();
    let axis = new Plottable.Axes.Numeric(scale, 'bottom');
    axis.formatter(stepFormatter);
    return {
      scale: scale,
      axis: axis,
      accessor: (d: Backend.Datum) => d.step,
    };
  }

  let timeFormatter = Plottable.Formatters.time('%a %b %e, %H:%M:%S');

  function wallX(): XComponents {
    let scale = new Plottable.Scales.Time();
    return {
      scale: scale,
      axis: new Plottable.Axes.Time(scale, 'bottom'),
      accessor: (d: Backend.Datum) => d.wall_time,
    };
  }
  let relativeAccessor =
      (d: any, index: number, dataset: Plottable.Dataset) => {
        // We may be rendering the final-point datum for scatterplot.
        // If so, we will have already provided the 'relative' property
        if (d.relative != null) {
          return d.relative;
        }
        let data = dataset.data();
        // I can't imagine how this function would be called when the data is
        // empty (after all, it iterates over the data), but lets guard just
        // to be safe.
        let first = data.length > 0 ? +data[0].wall_time : 0;
        return (+d.wall_time - first) / (60 * 60 * 1000);  // ms to hours
      };

  let relativeFormatter = (n: number) => {
    // we will always show 2 units of precision, e.g days and hours, or
    // minutes and seconds, but not hours and minutes and seconds
    let ret = '';
    let days = Math.floor(n / 24);
    n -= (days * 24);
    if (days) {
      ret += days + 'd ';
    }
    let hours = Math.floor(n);
    n -= hours;
    n *= 60;
    if (hours || days) {
      ret += hours + 'h ';
    }
    let minutes = Math.floor(n);
    n -= minutes;
    n *= 60;
    if (minutes || hours || days) {
      ret += minutes + 'm ';
    }
    let seconds = Math.floor(n);
    return ret + seconds + 's';
  };
  function relativeX(): XComponents {
    let scale = new Plottable.Scales.Linear();
    return {
      scale: scale,
      axis: new Plottable.Axes.Numeric(scale, 'bottom'),
      accessor: relativeAccessor,
    };
  }

  // a very literal definition of NaN: true for NaN for a non-number type
  // or null, etc. False for Infinity or -Infinity
  let isNaN = (x) => +x !== x;

  function getXComponents(xType: string): XComponents {
    switch (xType) {
      case 'step':
        return stepX();
      case 'wall_time':
        return wallX();
      case 'relative':
        return relativeX();
      default:
        throw new Error('invalid xType: ' + xType);
    }
  }
}
