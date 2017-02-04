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

module VZ {
  export class LineChart {
    private name2datasets: {[name: string]: Plottable.Dataset};
    private seriesNames: string[];

    private xAccessor: Plottable.Accessor<number|Date>;
    private xScale: Plottable.QuantitativeScale<number|Date>;
    private yScale: Plottable.QuantitativeScale<number>;
    private gridlines: Plottable.Components.Gridlines;
    private center: Plottable.Components.Group;
    private xAxis: Plottable.Axes.Numeric|Plottable.Axes.Time;
    private yAxis: Plottable.Axes.Numeric;
    private outer: Plottable.Components.Table;
    private colorScale: Plottable.Scales.Color;
    private tooltip: d3.Selection<any>;
    private dzl: Plottable.DragZoomLayer;

    private linePlot: Plottable.Plots.Line<number|Date>;
    private smoothLinePlot: Plottable.Plots.Line<number|Date>;
    private scatterPlot: Plottable.Plots.Scatter<number|Date, Number>;
    private nanDisplay: Plottable.Plots.Scatter<number|Date, Number>;
    private scalarAccessor: Plottable.Accessor<number>;
    private smoothedAccessor: Plottable.Accessor<number>;
    private lastPointsDataset: Plottable.Dataset;
    private datasets: Plottable.Dataset[];
    private onDatasetChanged: (dataset: Plottable.Dataset) => void;
    private nanDataset: Plottable.Dataset;
    private smoothingWeight: number;
    private smoothingEnabled: Boolean;
    private tooltipSortingMethod: string;
    private tooltipPosition: string;

    private targetSVG: d3.Selection<any>;

    constructor(
        xType: string, yScaleType: string, colorScale: Plottable.Scales.Color,
        tooltip: d3.Selection<any>) {
      this.seriesNames = [];
      this.name2datasets = {};
      this.colorScale = colorScale;
      this.tooltip = tooltip;
      this.datasets = [];
      // lastPointDataset is a dataset that contains just the last point of
      // every dataset we're currently drawing.
      this.lastPointsDataset = new Plottable.Dataset();
      this.nanDataset = new Plottable.Dataset();
      // need to do a single bind, so we can deregister the callback from
      // old Plottable.Datasets. (Deregistration is done by identity checks.)
      this.onDatasetChanged = this._onDatasetChanged.bind(this);
      this.buildChart(xType, yScaleType);
    }

    private buildChart(xType: string, yScaleType: string) {
      if (this.outer) {
        this.outer.destroy();
      }
      let xComponents = VZ.ChartHelpers.getXComponents(xType);
      this.xAccessor = xComponents.accessor;
      this.xScale = xComponents.scale;
      this.xAxis = xComponents.axis;
      this.xAxis.margin(0).tickLabelPadding(3);
      this.yScale = LineChart.getYScaleFromType(yScaleType);
      this.yAxis = new Plottable.Axes.Numeric(this.yScale, 'left');
      let yFormatter = VZ.ChartHelpers.multiscaleFormatter(
          VZ.ChartHelpers.Y_AXIS_FORMATTER_PRECISION);
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

    private buildPlot(xAccessor, xScale, yScale): Plottable.Component {
      this.scalarAccessor = (d: VZ.ChartHelpers.ScalarDatum) => d.scalar;
      this.smoothedAccessor = (d: VZ.ChartHelpers.ScalarDatum) => d.smoothed;
      let linePlot = new Plottable.Plots.Line<number|Date>();
      linePlot.x(xAccessor, xScale);
      linePlot.y(this.scalarAccessor, yScale);
      linePlot.attr(
          'stroke', (d: VZ.ChartHelpers.Datum, i: number,
                     dataset: Plottable.Dataset) =>
                        this.colorScale.scale(dataset.metadata().name));
      this.linePlot = linePlot;
      let group = this.setupTooltips(linePlot);

      let smoothLinePlot = new Plottable.Plots.Line<number|Date>();
      smoothLinePlot.x(xAccessor, xScale);
      smoothLinePlot.y(this.smoothedAccessor, yScale);
      smoothLinePlot.attr(
          'stroke', (d: VZ.ChartHelpers.Datum, i: number,
                     dataset: Plottable.Dataset) =>
                        this.colorScale.scale(dataset.metadata().name));
      this.smoothLinePlot = smoothLinePlot;

      // The scatterPlot will display the last point for each dataset.
      // This way, if there is only one datum for the series, it is still
      // visible. We hide it when tooltips are active to keep things clean.
      let scatterPlot = new Plottable.Plots.Scatter<number|Date, number>();
      scatterPlot.x(xAccessor, xScale);
      scatterPlot.y(this.scalarAccessor, yScale);
      scatterPlot.attr('fill', (d: any) => this.colorScale.scale(d.name));
      scatterPlot.attr('opacity', 1);
      scatterPlot.size(VZ.ChartHelpers.TOOLTIP_CIRCLE_SIZE * 2);
      scatterPlot.datasets([this.lastPointsDataset]);
      this.scatterPlot = scatterPlot;

      let nanDisplay = new Plottable.Plots.Scatter<number|Date, number>();
      nanDisplay.x(xAccessor, xScale);
      nanDisplay.y((x) => x.displayY, yScale);
      nanDisplay.attr('fill', (d: any) => this.colorScale.scale(d.name));
      nanDisplay.attr('opacity', 1);
      nanDisplay.size(VZ.ChartHelpers.NAN_SYMBOL_SIZE * 2);
      nanDisplay.datasets([this.nanDataset]);
      nanDisplay.symbol(Plottable.SymbolFactories.triangleUp);
      this.nanDisplay = nanDisplay;

      return new Plottable.Components.Group(
          [nanDisplay, scatterPlot, smoothLinePlot, group]);
    }

    /** Updates the chart when a dataset changes. Called every time the data of
     * a dataset changes to update the charts.
     */
    private _onDatasetChanged(dataset: Plottable.Dataset) {
      if (this.smoothingEnabled) {
        this.resmoothDataset(dataset);
      }
      this.updateSpecialDatasets();
    }

    private updateSpecialDatasets() {
      if (this.smoothingEnabled) {
        this.updateSpecialDatasetsWithAccessor(this.smoothedAccessor);
      } else {
        this.updateSpecialDatasetsWithAccessor(this.scalarAccessor);
      }
    }

    /** Constructs special datasets. Each special dataset contains exceptional
     * values from all of the regular datasets, e.g. last points in series, or
     * NaN values. Those points will have a `name` and `relative` property added
     * (since usually those are context in the surrounding dataset).
     * The accessor will point to the correct data to access.
     */
    private updateSpecialDatasetsWithAccessor(accessor:
                                                  Plottable.Accessor<number>) {
      let lastPointsData =
          this.datasets
              .map((d) => {
                let datum = null;
                // filter out NaNs to ensure last point is a clean one
                let nonNanData =
                    d.data().filter((x) => !isNaN(accessor(x, -1, d)));
                if (nonNanData.length > 0) {
                  let idx = nonNanData.length - 1;
                  datum = nonNanData[idx];
                  datum.name = d.metadata().name;
                  datum.relative =
                      VZ.ChartHelpers.relativeAccessor(datum, -1, d);
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
          if (!isNaN(accessor(data[i], -1, d))) {
            displayY = accessor(data[i], -1, d);
          }
          i++;
        }
        if (displayY == null) {
          displayY = 0;
        }
        let nanData = [];
        for (i = 0; i < data.length; i++) {
          if (!isNaN(accessor(data[i], -1, d))) {
            displayY = accessor(data[i], -1, d);
          } else {
            data[i].name = d.metadata().name;
            data[i].displayY = displayY;
            data[i].relative = VZ.ChartHelpers.relativeAccessor(data[i], -1, d);
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
        let target: VZ.ChartHelpers.Point = {
          x: p.x,
          y: p.y,
          datum: null,
          dataset: null,
        };


        let bbox: SVGRect = (<any>this.gridlines.content().node()).getBBox();

        // pts is the closets point to the tooltip for each dataset
        let pts = plot.datasets()
                      .map((dataset) => this.findClosestPoint(target, dataset))
                      .filter(x => x != null);
        let intersectsBBox = Plottable.Utils.DOM.intersectsBBox;
        // We draw tooltips for points that are NaN, or are currently visible
        let ptsForTooltips = pts.filter(
            (p) => intersectsBBox(p.x, p.y, bbox) || isNaN(p.datum.scalar));
        // Only draw little indicator circles for the non-NaN points
        let ptsToCircle = ptsForTooltips.filter((p) => !isNaN(p.datum.scalar));

        let ptsSelection: any =
            pointsComponent.content().selectAll('.point').data(
                ptsToCircle,
                (p: VZ.ChartHelpers.Point) => p.dataset.metadata().name);
        if (pts.length !== 0) {
          ptsSelection.enter().append('circle').classed('point', true);
          ptsSelection.attr('r', VZ.ChartHelpers.TOOLTIP_CIRCLE_SIZE)
              .attr('cx', (p) => p.x)
              .attr('cy', (p) => p.y)
              .style('stroke', 'none')
              .attr(
                  'fill',
                  (p) => this.colorScale.scale(p.dataset.metadata().name));
          ptsSelection.exit().remove();
          this.drawTooltips(ptsForTooltips, target);
        } else {
          hideTooltips();
        }
      });

      pi.onPointerExit(hideTooltips);

      return group;
    }

    private drawTooltips(
        points: VZ.ChartHelpers.Point[], target: VZ.ChartHelpers.Point) {
      // Formatters for value, step, and wall_time
      this.scatterPlot.attr('opacity', 0);
      let valueFormatter = VZ.ChartHelpers.multiscaleFormatter(
          VZ.ChartHelpers.Y_TOOLTIP_FORMATTER_PRECISION);

      let dist = (p: VZ.ChartHelpers.Point) =>
          Math.pow(p.x - target.x, 2) + Math.pow(p.y - target.y, 2);
      let closestDist = _.min(points.map(dist));

      let valueSortMethod = this.scalarAccessor;
      if (this.smoothingEnabled) {
        valueSortMethod = this.smoothedAccessor;
      }

      if (this.tooltipSortingMethod === 'ascending') {
        points =
            _.sortBy(points, (d) => valueSortMethod(d.datum, -1, d.dataset));
      } else if (this.tooltipSortingMethod === 'descending') {
        points =
            _.sortBy(points, (d) => valueSortMethod(d.datum, -1, d.dataset))
                .reverse();
      } else if (this.tooltipSortingMethod === 'nearest') {
        points = _.sortBy(points, dist);
      } else {
        // The 'default' sorting method maintains the order of names passed to
        // setVisibleSeries(). However we reverse that order when defining the
        // datasets. So we must call reverse again to restore the order.
        points = points.slice(0).reverse();
      }

      let rows = this.tooltip.select('tbody')
                     .html('')
                     .selectAll('tr')
                     .data(points)
                     .enter()
                     .append('tr');
      // Grey out the point if any of the following are true:
      // - The cursor is outside of the x-extent of the dataset
      // - The point's y value is NaN
      rows.classed('distant', (d) => {
        let firstPoint = d.dataset.data()[0];
        let lastPoint = _.last(d.dataset.data());
        let firstX =
            this.xScale.scale(this.xAccessor(firstPoint, 0, d.dataset));
        let lastX = this.xScale.scale(this.xAccessor(lastPoint, 0, d.dataset));
        let s = this.smoothingEnabled ? d.datum.smoothed : d.datum.scalar;
        return target.x < firstX || target.x > lastX || isNaN(s);
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
              (d) => this.colorScale.scale(d.dataset.metadata().name));
      rows.append('td').text((d) => d.dataset.metadata().name);
      if (this.smoothingEnabled) {
        rows.append('td').text(
            (d) => isNaN(d.datum.smoothed) ? 'NaN' :
                                             valueFormatter(d.datum.smoothed));
      }
      rows.append('td').text(
          (d) =>
              isNaN(d.datum.scalar) ? 'NaN' : valueFormatter(d.datum.scalar));
      rows.append('td').text(
          (d) => VZ.ChartHelpers.stepFormatter(d.datum.step));
      rows.append('td').text(
          (d) => VZ.ChartHelpers.timeFormatter(d.datum.wall_time));
      rows.append('td').text(
          (d) => VZ.ChartHelpers.relativeFormatter(
              VZ.ChartHelpers.relativeAccessor(d.datum, -1, d.dataset)));

      // compute left position
      let documentWidth = document.body.clientWidth;
      let node: any = this.tooltip.node();
      let parentRect = node.parentElement.getBoundingClientRect();
      let nodeRect = node.getBoundingClientRect();
      // prevent it from falling off the right side of the screen
      let left = documentWidth - parentRect.left - nodeRect.width - 60, top = 0;

      if (this.tooltipPosition === 'right') {
        left = Math.min(parentRect.width, left);
      } else {  // 'bottom'
        left = Math.min(0, left);
        top = parentRect.height + VZ.ChartHelpers.TOOLTIP_Y_PIXEL_OFFSET;
      }

      this.tooltip.style(
          'transform', 'translate(' + left + 'px,' + top + 'px)');
      this.tooltip.style('opacity', 1);
    }

    private findClosestPoint(
        target: VZ.ChartHelpers.Point,
        dataset: Plottable.Dataset): VZ.ChartHelpers.Point {
      let points: VZ.ChartHelpers.Point[] = dataset.data().map((d, i) => {
        let x = this.xAccessor(d, i, dataset);
        let y = this.smoothingEnabled ? this.smoothedAccessor(d, i, dataset) :
                                        this.scalarAccessor(d, i, dataset);
        return {
          x: this.xScale.scale(x),
          y: this.yScale.scale(y),
          datum: d,
          dataset: dataset,
        };
      });
      let idx: number =
          _.sortedIndex(points, target, (p: VZ.ChartHelpers.Point) => p.x);
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

    private resmoothDataset(dataset: Plottable.Dataset) {
      // When increasing the smoothing window, it smoothes a lot with the first
      // few points and then starts to gradually smooth slower, so using an
      // exponential function makes the slider more consistent. 1000^x has a
      // range of [1, 1000], so subtracting 1 and dividing by 999 results in a
      // range of [0, 1], which can be used as the percentage of the data, so
      // that the kernel size can be specified as a percentage instead of a
      // hardcoded number, what would be bad with multiple series.
      let factor = (Math.pow(1000, this.smoothingWeight) - 1) / 999;
      let data = dataset.data();
      let kernelRadius = Math.floor(data.length * factor / 2);

      data.forEach((d, i) => {
        let actualKernelRadius = Math.min(kernelRadius, i);
        let start = i - actualKernelRadius;
        let end = i + actualKernelRadius + 1;
        if (end >= data.length) {
          // In the beginning, it's OK for the smoothing window to be small,
          // but this is not desirable towards the end. Rather than shrinking
          // the window, or extrapolating data to fill the gap, we're simply
          // not going to display the smoothed line towards the end.
          d.smoothed = Infinity;
        } else if (!_.isFinite(d.scalar)) {
          // Only smooth finite numbers.
          d.smoothed = d.scalar;
        } else {
          d.smoothed = d3.mean(
              data.slice(start, end).filter((d) => _.isFinite(d.scalar)),
              (d) => d.scalar);
        }
      });
    }

    private getDataset(name: string) {
      if (this.name2datasets[name] === undefined) {
        this.name2datasets[name] = new Plottable.Dataset([], {name: name});
      }
      return this.name2datasets[name];
    }

    static getYScaleFromType(yScaleType: string):
        Plottable.QuantitativeScale<number> {
      if (yScaleType === 'log') {
        return new Plottable.Scales.ModifiedLog();
      } else if (yScaleType === 'linear') {
        return new Plottable.Scales.Linear();
      } else {
        throw new Error('Unrecognized yScale type ' + yScaleType);
      }
    }

    /**
     * Update the selected series on the chart.
     */
    public setVisibleSeries(names: string[]) {
      names = names.sort();
      this.seriesNames = names;

      names.reverse();  // draw first series on top
      this.datasets.forEach((d) => d.offUpdate(this.onDatasetChanged));
      this.datasets = names.map((r) => this.getDataset(r));
      this.datasets.forEach((d) => d.onUpdate(this.onDatasetChanged));
      this.linePlot.datasets(this.datasets);

      if (this.smoothingEnabled) {
        this.smoothLinePlot.datasets(this.datasets);
      }
      this.updateSpecialDatasets();
    }

    /**
     * Set the data of a series on the chart.
     */
    public setSeriesData(name: string, data: VZ.ChartHelpers.ScalarDatum[]) {
      this.getDataset(name).data(data);
    }

    public smoothingUpdate(weight: number) {
      this.smoothingWeight = weight;
      this.datasets.forEach((d) => this.resmoothDataset(d));

      if (!this.smoothingEnabled) {
        this.linePlot.addClass('ghost');
        this.scatterPlot.y(this.smoothedAccessor, this.yScale);
        this.smoothingEnabled = true;
        this.smoothLinePlot.datasets(this.datasets);
      }

      this.updateSpecialDatasetsWithAccessor(this.smoothedAccessor);
    }

    public smoothingDisable() {
      if (this.smoothingEnabled) {
        this.linePlot.removeClass('ghost');
        this.scatterPlot.y(this.scalarAccessor, this.yScale);
        this.smoothLinePlot.datasets([]);
        this.smoothingEnabled = false;
        this.updateSpecialDatasetsWithAccessor(this.scalarAccessor);
      }
    }

    public setTooltipSortingMethod(method: string) {
      this.tooltipSortingMethod = method;
    }

    public setTooltipPosition(position: string) {
      this.tooltipPosition = position;
    }

    public renderTo(targetSVG: d3.Selection<any>) {
      this.targetSVG = targetSVG;
      this.setViewBox();
      this.outer.renderTo(targetSVG);
    }

    /** There's an issue in Chrome where the svg overflow is a bit
     * "flickery". There is a border on the gridlines on the extreme edge of the
     * chart, which behaves inconsistently and causes the screendiffing tests to
     * flake. We can solve this by creating 1px effective margin for the svg by
     * setting the viewBox on the containing svg.
     */
    private setViewBox() {
      // There's an issue in Firefox where if we measure with the old viewbox
      // set, we get horrible results.
      this.targetSVG.attr('viewBox', null);

      let parent = this.targetSVG.node().parentNode as HTMLElement;
      let w = parent.clientWidth;
      let h = parent.clientHeight;
      this.targetSVG.attr({
        'height': h,
        'viewBox': `0 0 ${w + 1} ${h + 1}`,
      });
    }

    public redraw() {
      this.outer.redraw();
      this.setViewBox();
    }

    public destroy() { this.outer.destroy(); }
  }
}
