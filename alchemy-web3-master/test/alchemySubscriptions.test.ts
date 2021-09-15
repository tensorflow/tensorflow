import { AlchemyWeb3, createAlchemyWeb3 } from "../src/index";

jest.mock("../src/web3-adapter/alchemyContext", () => {
  return {
    makeAlchemyContext: () => {
      return {
        provider: {},
        senders: [],
        setWriteProvider: {},
      };
    },
  };
});

let consoleWarnMock: jest.Mock;
let web3: AlchemyWeb3;

beforeEach(() => {
  consoleWarnMock = jest.fn();
  console.warn = consoleWarnMock;

  web3 = createAlchemyWeb3("");
});

describe("AlchemyWeb3", () => {
  it("suppresses the missing subscription warning for alchemy_ subscriptions", async () => {
    web3.eth.subscribe("alchemy_fullPendingTransactions");
    expect(consoleWarnMock.mock.calls.length).toEqual(0);
  });

  it("accepts extra parameters to alchemy_ methods", async () => {
    const subCall = () => {
      web3.eth.subscribe("alchemy_filteredFullPendingTransactions", {
        address: "0x6B3595068778DD592e39A122f4f5a5cF09C90fE2",
      });
    };
    expect(subCall).not.toThrow();
  });

  it("includes an aliased method name for alchemy_ subscription InvalidNumberOfParams errors", async () => {
    const subCall = () => {
      web3.eth.subscribe("alchemy_fullPendingTransactions", {
        address: "0x6B3595068778DD592e39A122f4f5a5cF09C90fE2",
      });
    };
    expect(subCall).toThrow(
      'Invalid number of parameters for "alchemy_newFullPendingTransactions". Got 1 expected 0!',
    );
  });

  it.todo(
    "includes a non-aliased method name for alchemy_ subscription InvalidNumberOfParams errors",
  );
  it.todo(
    "returns an InvalidNumberOfParams error for alchemy_ subscriptions that require params",
  );
});
