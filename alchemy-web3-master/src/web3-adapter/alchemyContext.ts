import SturdyWebSocket from "sturdy-websocket";
import { w3cwebsocket } from "websocket";
import { FullConfig, Provider } from "../types";
import {
  JsonRpcSenders,
  makePayloadFactory,
  makeSenders,
} from "../util/jsonRpc";
import { VERSION } from "../version";
import { makeHttpSender } from "./alchemySendHttp";
import { makeWebSocketSender } from "./alchemySendWebSocket";
import { makeAlchemyHttpProvider } from "./httpProvider";
import { makePayloadSender } from "./sendPayload";
import { AlchemyWebSocketProvider } from "./webSocketProvider";

const NODE_MAX_WS_FRAME_SIZE = 100 * 1024 * 1024; // 100 MB

export interface AlchemyContext {
  provider: any;
  senders: JsonRpcSenders;
  setWriteProvider(provider: Provider | null | undefined): void;
}

export function makeAlchemyContext(
  url: string,
  config: FullConfig,
): AlchemyContext {
  const makePayload = makePayloadFactory();
  if (/^https?:\/\//.test(url)) {
    const alchemySend = makeHttpSender(url);
    const { sendPayload, setWriteProvider } = makePayloadSender(
      alchemySend,
      config,
    );
    const senders = makeSenders(sendPayload, makePayload);
    const provider = makeAlchemyHttpProvider(sendPayload);
    return { provider, senders, setWriteProvider };
  } else if (/^wss?:\/\//.test(url)) {
    const protocol = isAlchemyUrl(url) ? `alchemy-web3-${VERSION}` : undefined;
    const ws = new SturdyWebSocket(url, protocol, {
      wsConstructor: getWebSocketConstructor(),
    });
    const alchemySend = makeWebSocketSender(ws);
    const { sendPayload, setWriteProvider } = makePayloadSender(
      alchemySend,
      config,
    );
    const senders = makeSenders(sendPayload, makePayload);
    const provider = new AlchemyWebSocketProvider(ws, sendPayload, senders);
    return { provider, senders, setWriteProvider };
  } else {
    throw new Error(
      `Alchemy URL protocol must be one of http, https, ws, or wss. Recieved: ${url}`,
    );
  }
}

function getWebSocketConstructor(): any {
  return isNodeEnvironment()
    ? (url: string, protocols?: string | string[] | undefined) =>
        new w3cwebsocket(url, protocols, undefined, undefined, undefined, {
          maxReceivedMessageSize: NODE_MAX_WS_FRAME_SIZE,
          maxReceivedFrameSize: NODE_MAX_WS_FRAME_SIZE,
        })
    : WebSocket;
}

function isNodeEnvironment(): boolean {
  return (
    typeof process !== "undefined" &&
    process != null &&
    process.versions != null &&
    process.versions.node != null
  );
}

function isAlchemyUrl(url: string): boolean {
  return url.indexOf("alchemyapi.io") >= 0;
}
