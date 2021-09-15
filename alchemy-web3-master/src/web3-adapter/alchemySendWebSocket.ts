import SturdyWebSocket from "sturdy-websocket";
import {
  isResponse,
  JsonRpcId,
  JsonRpcRequest,
  SingleOrBatchRequest,
  SingleOrBatchResponse,
  WebSocketMessage,
} from "../types";
import { AlchemySendFunction, AlchemySendResult } from "./alchemySend";

interface RequestContext {
  request: SingleOrBatchRequest;
  resolve(response: AlchemySendResult): void;
}

export function makeWebSocketSender(ws: SturdyWebSocket): AlchemySendFunction {
  const contextsById = new Map<JsonRpcId, RequestContext>();
  ws.addEventListener("message", (message) => {
    const response: WebSocketMessage = JSON.parse(message.data);
    if (!isResponse(response)) {
      return;
    }
    const id = getIdFromResponse(response);
    if (id === undefined) {
      return;
    }
    const context = contextsById.get(id);
    if (!context) {
      return;
    }
    const { resolve } = context;
    contextsById.delete(id);
    if (
      !Array.isArray(response) &&
      response.error &&
      response.error.code === 429
    ) {
      resolve({ type: "rateLimit" });
    } else {
      resolve({ response, type: "jsonrpc" });
    }
  });
  ws.addEventListener("down", () => {
    [...contextsById].forEach(([id, { request, resolve }]) => {
      if (isWrite(request)) {
        // Writes cannot be resent because they will fail for a duplicate nonce.
        contextsById.delete(id);
        resolve({
          type: "networkError",
          status: 0,
          message: `WebSocket closed before receiving a response for write request with id: ${id}.`,
        });
      }
    });
  });
  ws.addEventListener("reopen", () => {
    for (const { request } of contextsById.values()) {
      ws.send(JSON.stringify(request));
    }
  });

  return (request) =>
    new Promise((resolve) => {
      const id = getIdFromRequest(request);
      if (id !== undefined) {
        const existingContext = contextsById.get(id);
        if (existingContext) {
          const message = `Another WebSocket request was made with the same id (${id}) before a response was received.`;
          console.error(message);
          existingContext.resolve({
            message,
            type: "networkError",
            status: 0,
          });
        }
        contextsById.set(id, { request, resolve });
      }
      ws.send(JSON.stringify(request));
    });
}

function getIdFromRequest(
  request: SingleOrBatchRequest,
): JsonRpcId | undefined {
  if (!Array.isArray(request)) {
    return request.id;
  }
  return getCanonicalIdFromList(request.map((p) => p.id));
}

function getIdFromResponse(
  response: SingleOrBatchResponse,
): JsonRpcId | undefined {
  if (!Array.isArray(response)) {
    return response.id;
  }
  return getCanonicalIdFromList(response.map((p) => p.id));
}

/**
 * Since the JSON-RPC spec allows responses to be returned in a different order
 * than sent, we need a mechanism for choosing a canonical id from a list that
 * doesn't depend on the order. This chooses the "minimum" id by an arbitrary
 * ordering: the smallest string if possible, otherwise the smallest number,
 * otherwise null.
 */
function getCanonicalIdFromList(
  ids: Array<JsonRpcId | undefined>,
): JsonRpcId | undefined {
  const stringIds: string[] = ids.filter((id) => typeof id === "string") as any;
  if (stringIds.length > 0) {
    return stringIds.reduce((bestId, id) => (bestId < id ? bestId : id));
  }
  const numberIds: number[] = ids.filter((id) => typeof id === "number") as any;
  if (numberIds.length > 0) {
    return Math.min(...numberIds);
  }
  return ids.indexOf(null) >= 0 ? null : undefined;
}

function isWrite(request: SingleOrBatchRequest): boolean {
  return Array.isArray(request)
    ? request.every(isSingleWrite)
    : isSingleWrite(request);
}

const WRITE_METHODS = ["eth_sendTransaction", "eth_sendRawTransaction"];

function isSingleWrite(request: JsonRpcRequest): boolean {
  return WRITE_METHODS.includes(request.method);
}
