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

// Each color scale is initialized with a configurable number of base hues.
// There are also several palettes available.
// TF.palettes.googleStandard, TF.palettes.googleColorBlind,
// TF.palettes.googleCool, TF.palettes.googleWarm, TF.palettes.constantValue
// Each string is hashed to an integer,
// then mapped to one of the base hues above.
// If there is a collision, the color that is later in an alphabetical sort
// gets nudged a little darker or lighter to disambiguate.
// I would call it mostly stable, in that the same array of strings will
// always return the same colors, but the same individual string may
// shift a little depending on its peers.
//
// runs = ["train", "test", "test1", "test2"]
// ccs = new TF.ColorScale(12, "googleStandard");
// ccs.domain(runs);
// ccs.getColor("train");
// ccs.getColor("test1");

module TF {
  export class ColorScale {
    public numColors: number;
    public internalColorScale: d3.scale.Linear<string, string>;
    private buckets: string[][];

    /**
     * The palette you provide defines your spectrum. The colorscale will
     * always use the full spectrum you provide. When you define "numColors"
     * it resamples at regular intervals along the full extent of the spectrum.
     * Thus you get the maximum distance between hues for the "numColors"
     * given. This allows the programmer to tweak the algorithm depending on
     * how big your expected domain is. If you generally think you're going to
     * have a small number of elements in the domain, then a small numColors
     * will be serviceable. With large domains, a small numColors would produce
     * too many hash collisions, so you'd want to bump it up to the threshold
     * of human perception (probably around 14 or 18).
     *
     * @param {number} [numColors=12] - The number of base colors you want
     *                 in the palette. The more colors, the smaller the number
     *                 the more hash collisions you will have, but the more
     *                 differentiable the base colors will be.
     *
     * @param {string[]} [palette=TF.palettes.googleColorBlind] - The color
     *                 palette you want as an Array of hex strings. Note, the
     *                 length of the array in this palette is independent of the
     *                 param numColors above. The scale will interpolate to
     *                 create the proper "numColors" given in the first param.
     *
     */
    constructor(numColors = 12, palette: string[] = TF.palettes.googleColorBlind) {
      this.numColors = numColors;
      this.domain([]);

      if (palette.length < 2) {
        throw new Error("Not enough colors in palette. Must be more than one.");
      }

      var k = (this.numColors - 1) / (palette.length - 1);
      this.internalColorScale = d3.scale.linear<string>()
        .domain(d3.range(palette.length).map((i) => i * k))
        .range(palette);
    }

    private hash(s: string): number {
     function h(hash, str) {
       hash = (hash << 5) - hash + str.charCodeAt(0);
       return hash & hash;
     }
     return Math.abs(Array.prototype.reduce.call(s, h, 0)) % this.numColors;
    }

    /**
     * Set the domain of strings so we can calculate collisions preemptively.
     * Can be reset at any point.
     *
     * @param {string[]} strings - An array of strings to use as the domain
     *                             for your scale.
     */
    public domain(strings: string[]) {
      this.buckets = d3.range(this.numColors).map(() => []);
      var sortedUniqueKeys = d3.set(strings).values().sort(function(a, b) { return a.localeCompare(b); });
      sortedUniqueKeys.forEach((s) => this.addToDomain(s));
    }

    private getBucketForString(s: string) {
      var bucketIdx = this.hash(s);
      return this.buckets[bucketIdx];
    }

    private addToDomain(s: string) {
      var bucketIdx = this.hash(s);
      var bucket = this.buckets[bucketIdx];
      if (bucket.indexOf(s) === -1) {
        bucket.push(s);
      }
    }

    private nudge(color: string, amount: number): any {
      // If amount is zero, just give back same color
      if (amount === 0) {
        return color;

      // For first tick, nudge lighter...
      } else if (amount === 1) {
        return d3.hcl(color).brighter(0.6);

      // ..otherwise nudge darker. Darker will approach black, which is visible.
      } else {
        return d3.hcl(color).darker((amount - 1) / 2);
      }
    }

    /**
     * Use the color scale to transform an element in the domain into a color.
     * If there was a hash conflict, the color will be "nudged" darker or lighter so that it is
     * unique.
     * @param {string} The input string to map to a color.
     * @return {string} The color corresponding to that input string.
     * @throws Will error if input string is not in the scale's domain.
     */

    public getColor(s: string): string {
      var bucket = this.getBucketForString(s);
      var idx = bucket.indexOf(s);
      if (idx === -1) {
        throw new Error("String was not in the domain.");
      }
      var color = this.internalColorScale(this.hash(s));
      return this.nudge(color, idx).toString();
    }

  }
}
