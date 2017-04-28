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
/* tslint:disable:no-namespace variable-name */
import * as d3 from 'd3';  // from //third_party/javascript/typings/d3_v4
import * as _ from 'lodash'
import * as Plottable from 'Plottable/plottable';  // from //third_party/javascript/plottable/v3
import {Dataset} from 'Plottable/plottable';

import * as ChartHelpers from '../vz_line_chart_d3v4/vz-chart-helpers'

export class DistributionChart {
  private run2datasets: {[run: string]: Plottable.Dataset};
  protected runs: string[];

  protected xAccessor: Plottable.IAccessor<number|Date>;
  protected xScale: Plottable.QuantitativeScale<number|Date>;
  protected yScale: Plottable.QuantitativeScale<number>;
  protected gridlines: Plottable.Components.Gridlines;
  protected center: Plottable.Components.Group;
  protected xAxis: Plottable.Axes.Numeric|Plottable.Axes.Time;
  protected yAxis: Plottable.Axes.Numeric;
  protected xLabel: Plottable.Components.AxisLabel;
  protected yLabel: Plottable.Components.AxisLabel;
  protected outer: Plottable.Components.Table;
  protected colorScale: Plottable.Scales.Color;
  private plots: Plottable.XYPlot<number|Date, number>[];

  private targetSVG: d3.Selection<any, any, any, any>;

  constructor(xType: string, colorScale: Plottable.Scales.Color) {
    this.run2datasets = {};
    this.colorScale = colorScale;
    this.buildChart(xType);
  }

  protected getDataset(run: string) {
    if (this.run2datasets[run] === undefined) {
      this.run2datasets[run] = new Plottable.Dataset([], {run: run});
    }
    return this.run2datasets[run];
  }

  protected buildChart(xType: string) {
    if (this.outer) {
      this.outer.destroy();
    }
    let xComponents = ChartHelpers.getXComponents(xType);
    this.xAccessor = xComponents.accessor;
    this.xScale = xComponents.scale;
    this.xAxis = xComponents.axis;
    this.xAxis.margin(0).tickLabelPadding(3);
    this.yScale = new Plottable.Scales.Linear();
    this.yAxis = new Plottable.Axes.Numeric(this.yScale, 'left');
    let yFormatter = ChartHelpers.multiscaleFormatter(
        ChartHelpers.Y_AXIS_FORMATTER_PRECISION);
    this.yAxis.margin(0).tickLabelPadding(5).formatter(yFormatter);
    this.yAxis.usesTextWidthApproximation(true);

    let center = this.buildPlot(this.xAccessor, this.xScale, this.yScale);

    this.gridlines =
        new Plottable.Components.Gridlines(this.xScale, this.yScale);

    this.center = new Plottable.Components.Group([this.gridlines, center]);
    this.outer = new Plottable.Components.Table(
        [[this.yAxis, this.center], [null, this.xAxis]]);
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
          'fill',
          (d: any, i: number, dataset: Plottable.Dataset) =>
              this.colorScale.scale(dataset.metadata().run));
      p.attr(
          'stroke',
          (d: any, i: number, dataset: Plottable.Dataset) =>
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
        'stroke', (d: any, i: number, m: any) => this.colorScale.scale(m.run));

    this.plots = plots;
    return new Plottable.Components.Group(plots);
  }

  public setVisibleSeries(runs: string[]) {
    this.runs = runs;
    let datasets = runs.map((r) => this.getDataset(r));
    this.plots.forEach((p) => p.datasets(datasets));
  }

  /**
   * Set the data of a series on the chart.
   */
  public setSeriesData(name: string, data: any) {
    this.getDataset(name).data(data);
  }

  public renderTo(targetSVG: d3.Selection<any, any, any, any>) {
    this.targetSVG = targetSVG;
    this.outer.renderTo(targetSVG);
  }

  public redraw() {
    this.outer.redraw();
  }

  protected destroy() {
    this.outer.destroy();
  }
}


Polymer({
  is: 'vz-distribution-chart',
  properties: {
    /**
     * Scale that maps series names to colors. The default colors are from
     * d3.d3.schemeCategory10. Use this property to replace the default
     * line colors with colors of your own choice.
     * @type {Plottable.Scales.Color}
     * @required
     */
    colorScale: {
      type: Object,
      value: function() {
        return new Plottable.Scales.Color().range(d3.schemeCategory10);
      }
    },
    /**
     * The way to display the X values. Allows:
     * - "step" - Linear scale using the  "step" property of the datum.
     * - "wall_time" - Temporal scale using the "wall_time" property of the
     * datum.
     * - "relative" - Temporal scale using the "relative" property of the
     * datum if it is present or calculating from "wall_time" if it isn't.
     */
    xType: {type: String, value: 'step'},
    _attached: Boolean,
    _chart: Object,
    _visibleSeriesCache: {
      type: Array,
      value: function() {
        return []
      }
    },
    _seriesDataCache: {
      type: Object,
      value: function() {
        return {}
      }
    },
    _makeChartAsyncCallbackId: {type: Number, value: null}
  },
  observers: [
    '_makeChart(xType, colorScale, _attached)',
    '_reloadFromCache(_chart)',
  ],
  setVisibleSeries: function(names) {
    this._visibleSeriesCache = names;
    if (this._chart) {
      this._chart.setVisibleSeries(names);
      this.redraw();
    }
  },
  setSeriesData: function(name, data) {
    this._seriesDataCache[name] = data;
    if (this._chart) {
      this._chart.setSeriesData(name, data);
    }
  },
  redraw: function() {
    this._chart.redraw();
  },
  ready: function() {
    this.scopeSubtree(this.$.chartdiv, true);
  },
  _makeChart: function(xType, colorScale, _attached) {
    if (this._makeChartAsyncCallbackId === null) {
      this.cancelAsync(this._makeChartAsyncCallbackId);
    }

    this._makeChartAsyncCallbackId = this.async(function() {
      this._makeChartAsyncCallbackId = null;
      if (!_attached) return;
      if (this._chart) this._chart.destroy();
      var chart = new DistributionChart(xType, colorScale);
      var svg = d3.select(this.$.chartdiv);
      chart.renderTo(svg);
      this._chart = chart;
    }, 350);
  },
  _reloadFromCache: function() {
    if (this._chart) {
      this._chart.setVisibleSeries(this._visibleSeriesCache);
      this._visibleSeriesCache.forEach(function(name) {
        this._chart.setSeriesData(name, this._seriesDataCache[name] || []);
      }.bind(this));
    }
  },
  attached: function() {
    this._attached = true;
  },
  detached: function() {
    this._attached = false;
  }
});
