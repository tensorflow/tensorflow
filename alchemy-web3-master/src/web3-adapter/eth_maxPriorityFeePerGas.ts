export function patchEthMaxPriorityFeePerGasMethod(web3: any): void {
  web3.eth.customRPC({
    name: "getMaxPriorityFeePerGas",
    call: "eth_maxPriorityFeePerGas",
    params: 0,
  });
}
