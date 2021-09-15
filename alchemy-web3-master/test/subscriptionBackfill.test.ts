import {
  Backfiller,
  BlockHead,
  makeBackfiller,
} from "../src/subscriptions/subscriptionBackfill";
import { JsonRpcRequest } from "../src/types";
import { fromHex, toHex } from "../src/util/hex";
import { JsonRpcSenders } from "../src/util/jsonRpc";
import { Mocked } from "./testUtils";

let senders: Mocked<JsonRpcSenders>;
let getNewHeadsBackfill: Backfiller["getNewHeadsBackfill"];
const isCancelled = () => false;

beforeEach(() => {
  senders = { send: jest.fn(), sendBatch: jest.fn() };
  ({ getNewHeadsBackfill } = makeBackfiller(senders));
});

describe("getNewHeadsBackfill", () => {
  it("returns blocks from start block number if no previous events", async () => {
    const heads = [
      makeNewHeadsEvent(10, "a"),
      makeNewHeadsEvent(11, "b"),
      makeNewHeadsEvent(12, "c"),
    ];
    senders.send.mockResolvedValue(toHex(12));
    senders.sendBatch.mockResolvedValue(heads);
    const result = await getNewHeadsBackfill(isCancelled, [], 9);
    expect(result).toEqual(heads);
    expectGetBlockRangeCalled(10, 13);
  });

  it("returns blocks since the latest event seen", async () => {
    const previousHeads = [
      makeNewHeadsEvent(10, "a"),
      makeNewHeadsEvent(11, "b"),
    ];
    const expected = [makeNewHeadsEvent(12, "c"), makeNewHeadsEvent(13, "d")];
    senders.sendBatch.mockResolvedValue(expected);
    senders.send
      .mockResolvedValueOnce(toHex(13))
      .mockResolvedValueOnce(makeNewHeadsEvent(11, "b"));
    const result = await getNewHeadsBackfill(isCancelled, previousHeads, 9);
    expectGetBlockCalled(11);
    expectGetBlockRangeCalled(12, 14);
    expect(result).toEqual(expected);
  });

  it("returns blocks since last reorg", async () => {
    const previousHeads = [
      makeNewHeadsEvent(10, "a"),
      makeNewHeadsEvent(11, "b"),
      makeNewHeadsEvent(12, "c"),
    ];
    const newHeads = [makeNewHeadsEvent(13, "d"), makeNewHeadsEvent(14, "e")];
    senders.sendBatch.mockResolvedValue(newHeads);
    senders.send
      .mockResolvedValueOnce(toHex(14))
      .mockResolvedValueOnce(makeNewHeadsEvent(12, "c'"))
      .mockResolvedValueOnce(makeNewHeadsEvent(11, "b'"))
      .mockResolvedValueOnce(makeNewHeadsEvent(10, "a"));
    const result = await getNewHeadsBackfill(isCancelled, previousHeads, 9);
    const expected = [
      makeNewHeadsEvent(11, "b'"),
      makeNewHeadsEvent(12, "c'"),
      makeNewHeadsEvent(13, "d"),
      makeNewHeadsEvent(14, "e"),
    ];
    expect(result).toEqual(expected);
    expectGetBlockCalled(12);
    expectGetBlockCalled(11);
    expectGetBlockCalled(10);
    expectGetBlockRangeCalled(13, 15);
  });

  it("returns all blocks from start if reorg goes that far", async () => {
    const previousHeads = [
      makeNewHeadsEvent(10, "a"),
      makeNewHeadsEvent(11, "b"),
      makeNewHeadsEvent(12, "c"),
    ];
    const newHeads = [makeNewHeadsEvent(13, "d"), makeNewHeadsEvent(14, "e")];
    senders.sendBatch.mockResolvedValue(newHeads);
    senders.send
      .mockResolvedValueOnce(toHex(14))
      .mockResolvedValueOnce(makeNewHeadsEvent(12, "c'"))
      .mockResolvedValueOnce(makeNewHeadsEvent(11, "b'"))
      .mockResolvedValueOnce(makeNewHeadsEvent(10, "a'"));
    const result = await getNewHeadsBackfill(isCancelled, previousHeads, 9);
    const expected = [
      makeNewHeadsEvent(10, "a'"),
      makeNewHeadsEvent(11, "b'"),
      makeNewHeadsEvent(12, "c'"),
      makeNewHeadsEvent(13, "d"),
      makeNewHeadsEvent(14, "e"),
    ];
    expect(result).toEqual(expected);
    expectGetBlockCalled(12);
    expectGetBlockCalled(11);
    expectGetBlockCalled(10);
    expectGetBlockRangeCalled(13, 15);
  });
});

function expectGetBlockCalled(blockNumber: number): void {
  expect(
    senders.send.mock.calls.some((call) => fromHex(call[0]) === blockNumber),
  );
}

function expectGetBlockRangeCalled(
  startInclusive: number,
  endExclusive: number,
): void {
  expect(senders.sendBatch).toBeCalled();
  const requestedBlockNumbers = senders.sendBatch.mock.calls[0][0].map(
    (request: JsonRpcRequest) => fromHex(request.params![0]),
  );
  const expectedRange: number[] = [];
  for (let i = startInclusive; i < endExclusive; i++) {
    expectedRange.push(i);
  }
  expect(requestedBlockNumbers).toEqual(expectedRange);
}

function makeNewHeadsEvent(blockNumber: number, hash: string): BlockHead {
  return { hash, number: toHex(blockNumber) } as any;
}
