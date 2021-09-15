import {
  JsonRpcId,
  JsonRpcRequest,
  JsonRpcResponse,
  SendFunction,
} from "../types";
import { SendPayloadFunction } from "../web3-adapter/sendPayload";

export type PayloadFactory = (method: string, params?: any[]) => JsonRpcRequest;

export interface JsonRpcSenders {
  send: SendFunction;
  sendBatch(parts: BatchPart[]): Promise<any[]>;
}

export interface BatchPart {
  method: string;
  params?: any;
}

export function makePayloadFactory(): PayloadFactory {
  let nextId = 0;
  return (method, params) => ({
    method,
    params,
    jsonrpc: "2.0",
    id: `alc-web3:${nextId++}`,
  });
}

export function makeSenders(
  sendPayload: SendPayloadFunction,
  makePayload: PayloadFactory,
): JsonRpcSenders {
  const send: SendFunction = async (method, params) => {
    const response = await sendPayload(makePayload(method, params));
    if (response.error) {
      throw new Error(response.error.message);
    }
    return response.result;
  };

  async function sendBatch(parts: BatchPart[]): Promise<any[]> {
    const payload = parts.map(({ method, params }) =>
      makePayload(method, params),
    );
    const response = await sendPayload(payload);
    if (!Array.isArray(response)) {
      const message = response.error
        ? response.error.message
        : "Batch request failed";
      throw new Error(message);
    }
    const errorResponse = response.find((r) => !!r.error);
    if (errorResponse) {
      throw new Error(errorResponse.error!.message);
    }
    // The ids are ascending numbers because that's what Payload Factories do.
    return response
      .sort((r1, r2) => (r1.id as number) - (r2.id as number))
      .map((r) => r.result);
  }

  return { send, sendBatch };
}

export function makeResponse<T>(id: JsonRpcId, result: T): JsonRpcResponse<T> {
  return { jsonrpc: "2.0", id, result };
}
