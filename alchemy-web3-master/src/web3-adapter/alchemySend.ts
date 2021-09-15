import SturdyWebSocket from "sturdy-websocket";
import { SingleOrBatchRequest, SingleOrBatchResponse } from "../types";

export interface AlchemySender {
  alchemySend: AlchemySendFunction;
  ws?: SturdyWebSocket;
}

export type AlchemySendFunction = (
  request: SingleOrBatchRequest,
) => Promise<AlchemySendResult>;

export type AlchemySendResult =
  | JsonRpcSendResult
  | RateLimitSendResult
  | NetworkErrorSendResult;

export interface JsonRpcSendResult {
  type: "jsonrpc";
  response: SingleOrBatchResponse;
}

export interface RateLimitSendResult {
  type: "rateLimit";
}

export interface NetworkErrorSendResult {
  type: "networkError";
  status: number;
  message: string;
}
