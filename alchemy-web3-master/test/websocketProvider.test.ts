import SturdyWebSocket from "sturdy-websocket";
import { JsonRpcResponse } from "../src/types";
import {
  JsonRpcSenders,
  makePayloadFactory,
  makeSenders,
} from "../src/util/jsonRpc";
import { promisify } from "../src/util/promises";
import { AlchemyWebSocketProvider } from "../src/web3-adapter/webSocketProvider";
import { Mocked } from "./testUtils";

let ws: Mocked<SturdyWebSocket>;
let sendPayload: jest.Mock;
let senders: JsonRpcSenders;
let wsProvider: AlchemyWebSocketProvider;

beforeEach(() => {
  ws = {
    close: jest.fn(),
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
  } as any;
  sendPayload = jest.fn();
  senders = makeSenders(sendPayload, makePayloadFactory());
  wsProvider = new AlchemyWebSocketProvider(ws as any, sendPayload, senders);
});

afterEach(() => {
  wsProvider.disconnect();
});

describe("AlchemyWebSocketProvider", () => {
  it("sends and receives payloads", async () => {
    let resolve: (result: JsonRpcResponse) => void = undefined!;
    const promise = new Promise<JsonRpcResponse>((r) => (resolve = r));
    sendPayload.mockReturnValue(promise);
    const result = promisify((callback) =>
      wsProvider.send(
        {
          jsonrpc: "2.0",
          id: 10,
          method: "eth_getBlockByNumber",
          params: ["latest", false],
        },
        callback,
      ),
    );
    expect(sendPayload).toHaveBeenCalledWith({
      jsonrpc: "2.0",
      id: 10,
      method: "eth_getBlockByNumber",
      params: ["latest", false],
    });
    const { id } = sendPayload.mock.calls[0][0];
    const expected: JsonRpcResponse = {
      id,
      jsonrpc: "2.0",
      result: "Some block",
    };
    resolve(expected);
    expect(await result).toEqual(expected);
  });
});
