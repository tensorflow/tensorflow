export type Mocked<T> = {
  // tslint:disable-next-line: ban-types
  [K in keyof T]: T[K] extends Function ? T[K] & jest.Mock : T[K];
};
