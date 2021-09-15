import {
  SingleOrBatchRequest,
  SingleOrBatchResponse,
  Web3Callback,
} from "../types";
import { callWhenDone } from "../util/promises";
import { SendPayloadFunction } from "./sendPayload";

/**
 * Returns a "provider" which can be passed to the Web3 constructor.
 */
export function makeAlchemyHttpProvider(sendPayload: SendPayloadFunction) {
  function send(
    payload: SingleOrBatchRequest,
    callback: Web3Callback<SingleOrBatchResponse>,
  ): void {
    callWhenDone(sendPayload(payload), callback);
  }
  return { send };
}
