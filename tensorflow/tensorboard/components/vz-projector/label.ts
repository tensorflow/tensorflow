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

export interface BoundingBox {
  loX: number;
  loY: number;
  hiX: number;
  hiY: number;
}

/**
 * Accelerates label placement by dividing the view into a uniform grid.
 * Labels only need to be tested for collision with other labels that overlap
 * the same grid cells. This is a fork of {@code amoeba.CollisionGrid}.
 */
export class CollisionGrid {
  private numHorizCells: number;
  private numVertCells: number;
  private grid: BoundingBox[][];
  private bound: BoundingBox;
  private cellWidth: number;
  private cellHeight: number;

  /**
   * Constructs a new Collision grid.
   *
   * @param bound The bound of the grid. Labels out of bounds will be rejected.
   * @param cellWidth Width of a cell in the grid.
   * @param cellHeight Height of a cell in the grid.
   */
  constructor(bound: BoundingBox, cellWidth: number, cellHeight: number) {
    /** The bound of the grid. Labels out of bounds will be rejected. */
    this.bound = bound;

    /** Width of a cell in the grid. */
    this.cellWidth = cellWidth;

    /** Height of a cell in the grid. */
    this.cellHeight = cellHeight;

    /** Number of grid cells along the x axis. */
    this.numHorizCells = Math.ceil(this.boundWidth(bound) / cellWidth);

    /** Number of grid cells along the y axis. */
    this.numVertCells = Math.ceil(this.boundHeight(bound) / cellHeight);

    /**
     * The 2d grid (stored as a 1d array.) Each cell consists of an array of
     * BoundingBoxes for objects that are in the cell.
     */
    this.grid = new Array(this.numHorizCells * this.numVertCells);
  }

  private boundWidth(bound: BoundingBox) { return bound.hiX - bound.loX; }

  private boundHeight(bound: BoundingBox) { return bound.hiY - bound.loY; }

  private boundsIntersect(a: BoundingBox, b: BoundingBox) {
    return !(a.loX > b.hiX || a.loY > b.hiY || a.hiX < b.loX || a.hiY < b.loY);
  }

  /**
   * Checks if a given bounding box has any conflicts in the grid and inserts it
   * if none are found.
   *
   * @param bound The bound to insert.
   * @param justTest If true, just test if it conflicts, without inserting.
   * @return True if the bound was successfully inserted; false if it
   *         could not be inserted due to a conflict.
   */
  insert(bound: BoundingBox, justTest = false): boolean {
    // Reject if the label is out of bounds.
    if (bound.loX < this.bound.loX || bound.hiX > this.bound.hiX ||
        bound.loY < this.bound.loY || bound.hiY > this.bound.hiY) {
      return false;
    }

    let minCellX = this.getCellX(bound.loX);
    let maxCellX = this.getCellX(bound.hiX);
    let minCellY = this.getCellY(bound.loY);
    let maxCellY = this.getCellY(bound.hiY);

    // Check all overlapped cells to verify that we can insert.
    let baseIdx = minCellY * this.numHorizCells + minCellX;
    let idx = baseIdx;
    for (let j = minCellY; j <= maxCellY; j++) {
      for (let i = minCellX; i <= maxCellX; i++) {
        let cell = this.grid[idx++];
        if (cell) {
          for (let k = 0; k < cell.length; k++) {
            if (this.boundsIntersect(bound, cell[k])) {
              return false;
            }
          }
        }
      }
      idx += this.numHorizCells - (maxCellX - minCellX + 1);
    }

    if (justTest) {
      return true;
    }

    // Insert into the overlapped cells.
    idx = baseIdx;
    for (let j = minCellY; j <= maxCellY; j++) {
      for (let i = minCellX; i <= maxCellX; i++) {
        if (!this.grid[idx]) {
          this.grid[idx] = [bound];
        } else {
          this.grid[idx].push(bound);
        }
        idx++;
      }
      idx += this.numHorizCells - (maxCellX - minCellX + 1);
    }
    return true;
  }

  /**
   * Returns the x index of the grid cell where the given x coordinate falls.
   *
   * @param x the coordinate, in world space.
   * @return the x index of the cell.
   */
  private getCellX(x: number) {
    return Math.floor((x - this.bound.loX) / this.cellWidth);
  };

  /**
   * Returns the y index of the grid cell where the given y coordinate falls.
   *
   * @param y the coordinate, in world space.
   * @return the y index of the cell.
   */
  private getCellY(y: number) {
    return Math.floor((y - this.bound.loY) / this.cellHeight);
  };
}