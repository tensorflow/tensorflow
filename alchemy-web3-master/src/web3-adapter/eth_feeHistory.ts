import { formatters } from "web3-core-helpers";
import { toNumber } from "web3-utils";

export function patchEthFeeHistoryMethod(web3: any): void {
  web3.eth.customRPC({
    name: "getFeeHistory",
    call: "eth_feeHistory",
    params: 3,
    inputFormatter: [
      toNumber,
      formatters.inputBlockNumberFormatter,
      (value: any) => value,
    ],
  });
}
