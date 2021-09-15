import { AbiCoder } from "web3-eth-abi";

declare class SpecificAbiCoder extends AbiCoder {
  public decodeParameter(type: "uint256", hex: string): string;
  public decodeParameter(type: any, hex: string): unknown;
}

declare module "web3-eth-abi" {
  const coder: SpecificAbiCoder;
  export = coder;
}
