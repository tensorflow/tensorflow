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
module TF {
  export type DataFn = (run: string, tag: string) =>
      Promise<Array<Backend.Datum>>;

  let Y_TOOLTIP_FORMATTER_PRECISION = 4;
  let STEP_AXIS_FORMATTER_PRECISION = 4;
  let Y_AXIS_FORMATTER_PRECISION = 3;
  let TOOLTIP_Y_PIXEL_OFFSET = 15;
  let TOOLTIP_X_PIXEL_OFFSET = 0;
  let TOOLTIP_CIRCLE_SIZE = 4;
  let TOOLTIP_CLOSEST_CIRCLE_SIZE = 6;

  interface Point {
    run: string;
    x: number;  // pixel space
    y: number;  // pixel space
    datum: TF.Backend.ScalarDatum;
  }

  type CrosshairResult = {[runName: string]: Point};

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
    protected xTooltipFormatter: (d: number) => string;
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
      this.buildChart(xType);
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
      this.xTooltipFormatter = xComponents.tooltipFormatter;
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
    private yAccessor: Plottable.Accessor<number>;
    private lastPointsDataset: Plottable.Dataset;
    private datasets: Plottable.Dataset[];
    private updateLastPointDataset;

    constructor(
        tag: string, dataFn: DataFn, xType: string,
        colorScale: Plottable.Scales.Color, tooltip: d3.Selection<any>) {
      this.datasets = [];
      // lastPointDataset is a dataset that contains just the last point of
      // every dataset we're currently drawing.
      this.lastPointsDataset = new Plottable.Dataset();
      // need to do a single bind, so we can deregister the callback from
      // old Plottable.Datasets. (Deregistration is done by identity checks.)
      this.updateLastPointDataset = this._updateLastPointDataset.bind(this);
      super(tag, dataFn, xType, colorScale, tooltip);
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
      return new Plottable.Components.Group([scatterPlot, group]);
    }

    /** Iterates over every dataset, takes the last point, and puts all these
     * points in the lastPointsDataset.
     */
    private _updateLastPointDataset() {
      let relativeAccessor = relativeX().accessor;
      let data = this.datasets
                     .map((d) => {
                       let datum = null;
                       if (d.data().length > 0) {
                         let idx = d.data().length - 1;
                         datum = d.data()[idx];
                         datum.run = d.metadata().run;
                         datum.relative = relativeAccessor(datum, idx, d);
                       }
                       return datum;
                     })
                     .filter((x) => x != null);
      this.lastPointsDataset.data(data);
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
          run: null,
          x: p.x,
          y: p.y,
          datum: null,
        };

        let centerBBox: SVGRect =
            (<any>this.gridlines.content().node()).getBBox();
        let points =
            plot.datasets()
                .map((dataset) => this.findClosestPoint(target, dataset))
                .filter((p) => {
                  // Only choose Points that are within window (if we zoomed)
                  return Plottable.Utils.DOM.intersectsBBox(
                      p.x, p.y, centerBBox);
                });
        points.reverse();  // if multiple points are equidistant, choose 1st run
        let closestPoint: Point = _.min(points, (p: Point) => dist(p, target));

        points.reverse();  // draw 1st run last, to get the right occlusions
        let pts: any = pointsComponent.content().selectAll('.point').data(
            points, (p: Point) => p.run);
        if (points.length !== 0) {
          pts.enter().append('circle').classed('point', true);
          pts.attr(
                 'r', (p) => p === closestPoint ? TOOLTIP_CLOSEST_CIRCLE_SIZE :
                                                  TOOLTIP_CIRCLE_SIZE)
              .attr('cx', (p) => p.x)
              .attr('cy', (p) => p.y)
              .style('stroke', 'none')
              .attr('fill', (p) => this.colorScale.scale(p.run));
          pts.exit().remove();
          this.drawTooltips(closestPoint);
        } else {
          hideTooltips();
        }
      });

      pi.onPointerExit(hideTooltips);

      return group;
    }

    private drawTooltips(closestPoint: Point) {
      // Formatters for value, step, and wall_time
      this.scatterPlot.attr('opacity', 0);
      let valueFormatter = multiscaleFormatter(Y_TOOLTIP_FORMATTER_PRECISION);
      let stepFormatter = stepX().tooltipFormatter;
      let wall_timeFormatter = wallX().tooltipFormatter;

      let datum = closestPoint.datum;
      this.tooltip.select('#headline')
          .text(closestPoint.run)
          .style('color', this.colorScale.scale(closestPoint.run));
      let step = stepFormatter(datum.step);
      let date = wall_timeFormatter(+datum.wall_time);
      let value = valueFormatter(datum.scalar);
      this.tooltip.select('#step').text(step);
      this.tooltip.select('#time').text(date);
      this.tooltip.select('#value').text(value);

      this.tooltip.style('top', closestPoint.y + TOOLTIP_Y_PIXEL_OFFSET + 'px')
          .style(
              'left', () => this.yAxis.width() + TOOLTIP_X_PIXEL_OFFSET +
                  closestPoint.x + 'px')
          .style('opacity', 1);
    }

    private findClosestPoint(target: Point, dataset: Plottable.Dataset): Point {
      let run: string = dataset.metadata().run;
      let points: Point[] = dataset.data().map((d, i) => {
        let x = this.xAccessor(d, i, dataset);
        let y = this.yAccessor(d, i, dataset);
        return {
          x: this.xScale.scale(x),
          y: this.yScale.scale(y),
          datum: d,
          run: run,
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
      this.datasets.forEach((d) => d.offUpdate(this.updateLastPointDataset));
      this.datasets = runs.map((r) => this.getDataset(r));
      this.datasets.forEach((d) => d.onUpdate(this.updateLastPointDataset));
      this.linePlot.datasets(this.datasets);
    }
  }

  export class HistogramChart extends BaseChart {
    private plots: Plottable.XYPlot<number | Date, number>[];

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
    tooltipFormatter: (d: number) => string;
    /* tslint:enable */
  }

  function stepX(): XComponents {
    let scale = new Plottable.Scales.Linear();
    let axis = new Plottable.Axes.Numeric(scale, 'bottom');
    let formatter =
        Plottable.Formatters.siSuffix(STEP_AXIS_FORMATTER_PRECISION);
    axis.formatter(formatter);
    return {
      scale: scale,
      axis: axis,
      accessor: (d: Backend.Datum) => d.step,
      tooltipFormatter: formatter,
    };
  }

  function wallX(): XComponents {
    let scale = new Plottable.Scales.Time();
    let formatter = Plottable.Formatters.time('%a %b %e, %H:%M:%S');
    return {
      scale: scale,
      axis: new Plottable.Axes.Time(scale, 'bottom'),
      accessor: (d: Backend.Datum) => d.wall_time,
      tooltipFormatter: (d: number) => formatter(new Date(d)),
    };
  }

  function relativeX(): XComponents {
    let scale = new Plottable.Scales.Linear();
    let formatter = (n: number) => {
      let days = Math.floor(n / 24);
      n -= (days * 24);
      let hours = Math.floor(n);
      n -= hours;
      n *= 60;
      let minutes = Math.floor(n);
      n -= minutes;
      n *= 60;
      let seconds = Math.floor(n);
      return days + 'd ' + hours + 'h ' + minutes + 'm ' + seconds + 's';
    };
    return {
      scale: scale,
      axis: new Plottable.Axes.Numeric(scale, 'bottom'),
      accessor: (d: any, index: number, dataset: Plottable.Dataset) => {
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
      },
      tooltipFormatter: formatter,
    };
  }

  function dist(p1: Point, p2: Point): number {
    return Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));
  }

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
