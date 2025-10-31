#!/usr/bin/env python3
"""
ddl_all_analysis.py

Generates DDL outputs for ALL 65,536 16-bit inputs (4 iterations), writes:
  - ddl_all_outputs.csv      (input_hex,input_dec,output_hex,output_dec)
  - ddl_collisions.txt       (outputs that have multiple preimages + lists)

Prints a summary and top-10 largest-collision outputs.

Usage:
  python ddl_all_analysis.py
"""
from collections import defaultdict

# --- Inline DDL (standalone) ---
def rol(x: int, n: int, bits: int = 16) -> int:
    n %= bits
    return ((x << n) & ((1 << bits) - 1)) | (x >> (bits - n))

def ror(x: int, n: int, bits: int = 16) -> int:
    n %= bits
    return (x >> n) | ((x << (bits - n)) & ((1 << bits) - 1))

def hamming_weight(x: int) -> int:
    return bin(x & 0xFFFF).count("1")

def ddl_step(x: int, bits: int = 16):
    x &= (1 << bits) - 1
    HW = hamming_weight(x)
    if HW == 0:
        shift = 0
        new = x ^ ror(x, shift, bits=bits)
    elif HW & 1:  # odd
        shift = HW
        new = x ^ rol(x, shift, bits=bits)
    else:         # even
        shift = HW - 1
        new = x ^ ror(x, shift, bits=bits)
    return new & ((1 << bits) - 1), HW

def ddl(x: int, iterations: int = 4, bits: int = 16) -> int:
    mask = (1 << bits) - 1
    cur = x & mask
    for _ in range(iterations):
        cur, _ = ddl_step(cur, bits=bits)
    return cur

# --- Main full-space run + analysis ---
def main():
    mapping = {}
    preimage = defaultdict(list)

    # Compute for all 16-bit inputs
    for x in range(0x10000):
        y = ddl(x, iterations=4, bits=16)
        mapping[x] = y
        preimage[y].append(x)

    # Write CSV mapping (to current directory)
    with open("ddl_all_outputs.csv", "w") as f:
        f.write("input_hex,input_dec,output_hex,output_dec\n")
        for x in range(0x10000):
            y = mapping[x]
            f.write("0x{0:04X},{1},0x{2:04X},{3}\n".format(x, x, y, y))

    # Write collisions file (outputs with >1 preimage)
    multi = {y: xs for y, xs in preimage.items() if len(xs) > 1}
    sorted_multi = sorted(multi.items(), key=lambda kv: (-len(kv[1]), kv[0]))

    with open("ddl_collisions.txt", "w") as f:
        f.write("Outputs with multiple preimages (output_hex : count -> list of input_hex)\n\n")
        for y, xs in sorted_multi:
            f.write("0x{0:04X} : {1} -> ".format(y, len(xs)) +
                    ", ".join("0x{0:04X}".format(x) for x in xs) + "\n")

    # Summary
    total_inputs = 0x10000
    distinct_outputs = len(preimage)
    num_outputs_with_multiple_preimages = sum(1 for xs in preimage.values() if len(xs) > 1)
    max_preimage_size = max(len(xs) for xs in preimage.values())

    print("Summary")
    print("-------")
    print(f"Total inputs: {total_inputs}")
    print(f"Distinct outputs: {distinct_outputs}")
    print(f"Number of outputs with >1 preimage: {num_outputs_with_multiple_preimages}")
    print(f"Max preimage size for any output: {max_preimage_size}")

    print("\nTop outputs with largest preimage sizes (output_hex : preimage_size):")
    for y, xs in sorted_multi[0:]:
        print(f"  0x{y:04X} : {len(xs)}")

if __name__ == "__main__":
    main()