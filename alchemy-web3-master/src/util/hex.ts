export function toHex(n: number): string {
  return `0x${n.toString(16)}`;
}

export function fromHex(hexString: string): number {
  return Number.parseInt(hexString, 16);
}

export function formatBlock(block: string | number): string {
  if (typeof block === "string") {
    return block;
  } else if (typeof block === "number" && Number.isInteger(block)) {
    return toHex(block);
  }
  return block.toString();
}
