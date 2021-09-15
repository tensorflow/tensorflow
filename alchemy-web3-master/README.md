# Alchemy Web3

Web3 client extended with Alchemy and browser provider integration.

## Introduction

Alchemy Web3 provides website authors with a drop-in replacement for the
[web3.js](https://github.com/ethereum/web3.js) Ethereum API client. It produces
a client matching that of web3.js, but brings multiple advantages to make use of
[Alchemy API](https://alchemyapi.io):

- **Uses Alchemy or an injected provider as needed.** Most requests will be sent
  through Alchemy, but requests involving signing and sending transactions are
  sent via a browser provider like [Metamask](https://metamask.io/) or [Trust
  Wallet](https://trustwallet.com) if the user has it installed, or via a
  custom provider specified in options.

- **Easy access to Alchemy's higher level API.** The client exposes methods to
  call Alchemy's exclusive features.

- **Automatically retries on rate limited requests.** If Alchemy returns a 429 response (rate limited), automatically retry after a short delay. This
  behavior is configurable.

- **Robust WebSocket subscriptions** which don't miss events if the WebSocket
  needs to be reconnected.

Alchemy Web3 is designed to require minimal configuration so you can start using
it in your app right away.

## Installation

### With a package manager

With Yarn:

```
yarn add @alch/alchemy-web3
```

Or with NPM:

```
npm install @alch/alchemy-web3
```

### With a CDN in the browser

Alternatively, add the following script tag to your page:

```html
<script src="https://cdn.jsdelivr.net/npm/@alch/alchemy-web3@latest/dist/alchemyWeb3.min.js"></script>
```

When using this option, you can create Alchemy-Web3 instances using the global variable `AlchemyWeb3.createAlchemyWeb3`.

## Usage

### Basic Usage

You will need an Alchemy account to access the Alchemy API. If you don't
have one yet, [contact Alchemy](mailto:hello@alchemyapi.io) to request one.

Create the client by importing the function `createAlchemyWeb3` and then passing
it your Alchemy app's URL and optionally a configuration object.

```ts
import { createAlchemyWeb3 } from "@alch/alchemy-web3";

// Using HTTPS
const web3 = createAlchemyWeb3(
  "https://eth-mainnet.alchemyapi.io/v2/<api-key>",
);
```

or

```ts
// Using WebSockets
const web3 = createAlchemyWeb3(
  "wss://eth-mainnet.ws.alchemyapi.io/ws/<api-key>",
);
```

You can use any of the methods described in the [web3.js
API](https://web3js.readthedocs.io/en/1.0/) and they will send requests to
Alchemy:

```ts
// Many web3.js methods return promises.
web3.eth.getBlock("latest").then((block) => {
  /* … */
});

web3.eth
  .estimateGas({
    from: "0xge61df…",
    to: "0x087a5c…",
    data: "0xa9059c…",
    gasPrice: "0xa994f8…",
  })
  .then((gasAmount) => {
    /* … */
  });
```

### With a Browser Provider

If the user has a provider in their browser available at `window.ethereum`, then
any methods which involve user accounts or signing will automatically use it.
This provider might be injected by [Metamask](https://metamask.io/), [Trust
Wallet](https://trustwallet.com/dapp) or other browsers or browser extensions if
the user has them installed. For example, the following will use a provider from
the user's browser:

```ts
web3.eth.getAccounts().then((accounts) => {
  web3.eth.sendTransaction({
    from: accounts[0],
    to: "0x6A823E…",
    value: "1000000000000000000",
  });
});
```

#### Note on using Metamask

As just discussed, Metamask will automatically be used for accounts and signing
if it is installed. However, for this to work **you must first request
permission from the user to access their accounts in Metamask**. This is a
security restriction required by Metamask: details can be found
[here](https://medium.com/metamask/https-medium-com-metamask-breaking-change-injecting-web3-7722797916a8).

To enable the use of Metamask, you must call
[`ethereum.enable()`](<https://metamask.github.io/metamask-docs/API_Reference/Ethereum_Provider#ethereum.enable()>).
An example of doing so is as follows:

```ts
if (window.ethereum) {
  ethereum
    .enable()
    .then((accounts) => {
      // Metamask is ready to go!
    })
    .catch((reason) => {
      // Handle error. Likely the user rejected the login.
    });
} else {
  // The user doesn't have Metamask installed.
}
```

Note that doing so will display a Metamask dialog to the user if they have not
already seen it and accepted, so you may choose to wait to enable Metamask until
the user is about to perform an action which requires it. This is also why
Alchemy Web3 will not automatically enable Metamask on page load.

### With a custom provider

You may also choose to bring your own provider for writes rather than relying on
one being present in the browser environment. To do so, use the `writeProvider`
option when creating your client:

```ts
const web3 = createAlchemyWeb3(ALCHEMY_URL, { writeProvider: provider });
```

Your provider should expose at least one of `sendAsync()` or `send()`, as
specified in [EIP
1193](https://github.com/ethereum/EIPs/blob/master/EIPS/eip-1193.md).

You may swap out the custom provider at any time by calling the
`setWriteProvider()` method:

```ts
web3.setWriteProvider(provider);
```

You may also disable the write provider entirely by passing a value of `null`.

### Automatic Retries

If Alchemy Web3 encounters a rate limited response, it will automatically retry
the request after a short delay. This behavior can be configured by passing the
following options when creating your client. To disable retries, set
`maxRetries` to 0.

#### `maxRetries`

The number of times the client will attempt to resend a rate limited request before giving up. Default: 3.

#### `retryInterval`

The minimum time waited between consecutive retries, in milliseconds. Default: 1000.

#### `retryJitter`

A random amount of time is added to the retry delay to help avoid additional
rate errors caused by too many concurrent connections, chosen as a number of
milliseconds between 0 and this value. Default: 250.

### Sturdier WebSockets

Alchemy Web3 brings multiple improvements to ensure correct WebSocket behavior
in cases of temporary network failure or dropped connections. As with any
network connection, you should not assume that a WebSocket will remain open
forever without interruption, but correctly handling dropped connections and
reconnection by hand can be challenging to get right. Alchemy Web3 automatically
adds handling for these failures with no configuration necessary.

If you use your WebSocket URL when initializing, then when you create
subscriptions using `web3.eth.subscribe()`, Alchemy Web3 will bring the
following advantages over standard Web3 subscriptions:

- Unlike standard Web3, you will not permanently miss events which arrive while
  the backing WebSocket is temporarily down. Instead, you will receive these
  events as soon as the connection is reopened. Note that if the connection is
  down for more than 120 blocks (approximately 20 minutes), you may still miss
  some events that were not part of the most recent 120 blocks.

- Compared to standard Web3, lowered rate of failure when sending requests over
  the WebSocket while the connection is down. Alchemy Web3 will attempt to send
  the requests once the connection is reopened. Note that it is still possible,
  with a lower likelihood, for outgoing requests to be lost, so you should still
  have error handling as with any network request.

## Alchemy's Transfers API

The produced client also grants easy access to Alchemy's [transfer API](https://docs.alchemyapi.io/documentation/alchemy-api-reference/transfers-api).

### `web3.alchemy.getAssetTransfers({fromBlock, toBlock, fromAddress, toAddress, contractAddresses, excludeZeroValue, maxCount, category, pageKey})`

Returns an array of asset transfers based on the specified parameters.

**Parameters:**

An object with the following fields:

- `fromBlock`: Optional inclusive from hex string block (default latest)
- `toBlock`: Optional inclusive to hex string block (default latest)
- `fromAddress`: Optional from hex string address (default wildcard)
- `toAddress`: Optional to hex string address (default wildcard)
  NOTE: `fromAddress` is ANDed with `toAddress`
- `contractAddresses`: Optional array of hex string contract addresses for "token" transfers (default wildcard)
  NOTE: `contractAddresses` are ORed together
- `excludeZeroValue`: Optional boolean to exclude transfers with zero value (default true)
- `maxCount`: Optional number to restrict payload size (default and max of 1000)
- `category`: Optional array of categories (default all categories ["external", "internal", "token"])
- `pageKey`: Optional uuid pageKey to retrieve the next payload

**Returns:**

An object with the following fields:

- `pageKey`: Uuid for next page of results (undefined for the last page of results).
- `transfers`: An array of objects with the following fields sorted in ascending order by block number
  - `category`: "external", "internal" or "token" - label for the transfer
  - `blockNum`: The block where the transfer occurred (hex string).
  - `from`: From address of transfer (hex string).
  - `to`: To address of transfer (hex string). `null` if contract creation.
  - `value`: Converted asset transfer value as a number (raw value divided by contract decimal). `null` if erc721 transfer or contract decimal not available.
  - `erc721TokenId`: Raw erc721 token id (hex string). `null` if not an erc721 "token" transfer
  - `asset`: "ETH" or the token's symbol. `null` if not defined in the contract and not available from other sources.
  - `hash`: Transaction hash (hex string).
  - `rawContract`: Object of raw values:
    - `value`: Raw transfer value (hex string). `null` if erc721 transfer
    - `address`: Contract address (hex string). `null` if "external" or "internal"
    - `decimal`: Contract decimal (hex string). `null` if not defined in the contract and not available from other sources.

## Alchemy's Enhanced API

The produced client also grants easy access to Alchemy's [enhanced API](https://docs.alchemyapi.io/documentation/alchemy-web3/enhanced-web3-api).

### `web3.alchemy.getTokenAllowance({contract, owner, spender})`

Returns token balances for a specific address given a list of contracts.

**Parameters:**

An object with the following fields:

- `contract`: The address of the token contract.
- `owner`: The address of the token owner.
- `spender`: The address of the token spender.

**Returns:**

The allowance amount, as a string representing a base-10 number.

### `web3.alchemy.getTokenBalances(address, contractAddresses)`

Returns token balances for a specific address given a list of contracts.

**Parameters:**

1. `address`: The address for which token balances will be checked.
2. `contractAddresses`: An array of contract addresses.

**Returns:**

An object with the following fields:

- `address`: The address for which token balances were checked.
- `tokenBalances`: An array of token balance objects. Each object contains:
  - `contractAddress`: The address of the contract.
  - `tokenBalance`: The balance of the contract, as a string representing a
    base-10 number.
  - `error`: An error string. One of this or `tokenBalance` will be `null`.

### `web3.alchemy.getTokenMetadata(address)`

Returns metadata (name, symbol, decimals, logo) for a given token contract address.

**Parameters:**

`address`: The address of the token contract.

**Returns:**

An object with the following fields:

- `name`: The token's name. `null` if not defined in the contract and not available from other sources.
- `symbol`: The token's symbol. `null` if not defined in the contract and not available from other sources.
- `decimals`: The token's decimals. `null` if not defined in the contract and not available from other sources.
- `logo`: URL of the token's logo image. `null` if not available.

### `web3.eth.subscribe("alchemy_fullPendingTransactions")`

Subscribes to pending transactions, similar to the standard Web3 call
`web3.eth.subscribe("pendingTransactions")`, but differs in that it emits
full transaction information rather than just transaction hashes.

Note that the argument passed to this function is permitted to be either of
`"alchemy_fullPendingTransactions"` or `"alchemy_newFullPendingTransactions"`,
which have the same effect. The latter is the string used in raw `eth_subscribe`
JSON-RPC calls, while the former is consistent with the existing Web3.js
subscription APIs (for example, `web3.eth.subscribe("pendingTransactions")`
corresponds to the raw JSON-RPC call of type `newPendingTransactions`). While
this is unfortunately confusing, supporting both strings attempts to balance
consistency and convenience.

### `web3.eth.subscribe("alchemy_filteredFullPendingTransactions", options)`

Like an `alchemy_fullPendingTransactions` subscription, but also allows passing
an `options` argument containing an `address` field to filter the returned
transactions to those from or to the specified address. The options argument is
as described in [the documentation
here](https://docs.alchemy.com/alchemy/guides/using-websockets#2-alchemy_filterednewfullpendingtransactions).

Similar to the previous point, note that the argument passed to this function
may be either of `"alchemy_filteredFullPendingTransactions"` or
`"alchemy_filteredNewPendingTransactions"`.

<br/>

Copyright © 2019 Alchemy Insights Inc.

## EIP 1559

### `web3.eth.getFeeHistory(blockRange, startingBlock, percentiles[])`

Fetches the fee history for the given block range as per the [eth spec](https://github.com/ethereum/eth1.0-specs/blob/master/json-rpc/spec.json).

**Parameters**

- `blockRange`: The number of blocks for which to fetch historical fees. Can be an integer or a hex string.
- `startingBlock`: The block to start the search. The result will look backwards from here. Can be a hex string or a predefined block string e.g. "latest".
- `percentiles`: (Optional) An array of numbers that define which percentiles of reward values you want to see for each block.

**Returns**

An object with the following fields:

- `oldestBlock`: The oldest block in the range that the fee history is being returned for.
- `baseFeePerGas`: An array of base fees for each block in the range that was looked up. These are the same values that would be returned on a block for the `eth_getBlockByNumber` method.
- `gasUsedRatio`: An array of the ratio of gas used to gas limit for each block.
- `reward`: Only returned if a percentiles paramater was provided. Each block will have an array corresponding to the percentiles provided. Each element of the nested array will have the tip provided to miners for the percentile given. So if you provide [50, 90] as the percentiles then each block will have a 50th percentile reward and a 90th percentile reward.

**Example**

Method call
```
web3.eth.getFeeHistory(4, "latest", [25, 50, 75]).then(console.log);

```
Logged response
```
{
  oldestBlock: 12930639,
  reward: [
    [ '0x649534e00', '0x66720b300', '0x826299e00' ],
    [ '0x649534e00', '0x684ee1800', '0x7ea8ed400' ],
    [ '0x5ea8dd480', '0x60db88400', '0x684ee1800' ],
    [ '0x59682f000', '0x5d21dba00', '0x5d21dba00' ]
  ],
  baseFeePerGas: [ '0x0', '0x0', '0x0', '0x0', '0x0' ],
  gasUsedRatio: [ 0.9992898398856537, 0.9999566454373825, 0.9999516, 0.9999378 ]
}
```

### `web3.eth.getMaxPriorityFeePerGas()`

Returns a quick estimate for `maxPriorityFeePerGas` in EIP 1559 transactions. Rather than using `feeHistory` and making a calculation yourself you can just use this method to get a quick estimate. Note: this is a geth-only method, but Alchemy handles that for you behind the scenes.

**Parameters**

None!

**Returns**

A hex, which is the `maxPriorityFeePerGas` suggestion. You can plug this directly into your transaction field.

**Example**

Method call
```
web3.eth.getMaxPriorityFeePerGas().then(console.log);
```
Logged response
```
0x560de0700
```
