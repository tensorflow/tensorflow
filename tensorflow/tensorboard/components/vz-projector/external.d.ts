// TODO(smilkov): Split into weblas.d.ts and numeric.d.ts and write
// typings for numeric.
interface Tensor {
  new(size: [number, number], data: Float32Array);
  transfer(): Float32Array;
  delete(): void;
}

interface Weblas {
  sgemm(M: number, N: number, K: number, alpha: number,
      A: Float32Array, B: Float32Array, beta: number, C: Float32Array):
      Float32Array;
  pipeline: {
     Tensor: Tensor;
     sgemm(alpha: number, A: Tensor, B: Tensor, beta: number,
         C: Tensor): Tensor;
  };
  util: {
    transpose(M: number, N: number, data: Float32Array): Tensor;
  };

}

declare let numeric: any;
declare let weblas: Weblas;