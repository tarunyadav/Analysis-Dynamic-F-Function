"""
ddl_impl.py

Implementation of the Dynamic Diffusion Layer (DDL) described in the uploaded paper.

Provides:
 - rol(x, n, bits=16)
 - ror(x, n, bits=16)
 - hamming_weight(x)
 - ddl_step(x, bits=16) -> (new_x, HW, description)
 - ddl(x, iterations=4, bits=16, verbose=False) -> (final_x, states)

Usage:
 - Import functions in other modules:
     from ddl_impl import ddl, ddl_step
 - Run as a script to see example runs:
     python ddl_impl.py
"""
from typing import Tuple, List

MASK16 = (1 << 16) - 1

def rol(x: int, n: int, bits: int = 16) -> int:
    """Rotate-left x by n over 'bits' bits."""
    n = n % bits
    return ((x << n) & ((1 << bits) - 1)) | (x >> (bits - n))

def ror(x: int, n: int, bits: int = 16) -> int:
    """Rotate-right x by n over 'bits' bits."""
    n = n % bits
    return (x >> n) | ((x << (bits - n)) & ((1 << bits) - 1))

def hamming_weight(x: int) -> int:
    """Return Hamming weight (number of 1 bits) of x."""
    return bin(x & MASK16).count("1")

def ddl_step(x: int, bits: int = 16) -> Tuple[int, int, str]:
    """
    Perform one iteration of the DDL transformation on a 'bits'-wide word x.

    Rules (as implemented to match the paper):
      - HW = HammingWeight(x)
      - if HW is odd:   new = x XOR rol(x, HW)
      - if HW is even:  new = x XOR ror(x, HW - 1)
      - if HW == 0:     treat shift = 0 (xor with ror(x,0) = x -> yields 0)
    
    Returns:
      (new_value, HW, textual_description)
    """
    x = x & ((1 << bits) - 1)
    HW = hamming_weight(x)
    if HW == 0:
        shift = 0
        new = x ^ ror(x, shift, bits=bits)
        desc = f"HW=0 (even), ror by {shift} -> xor"
    elif HW % 2 == 1:
        shift = HW
        new = x ^ rol(x, shift, bits=bits)
        desc = f"HW={HW} odd, rol by {shift} -> xor"
    else:
        shift = HW - 1
        new = x ^ ror(x, shift, bits=bits)
        desc = f"HW={HW} even, ror by {shift} -> xor"
    return new & ((1 << bits) - 1), HW, desc

def ddl(x: int, iterations: int = 4, bits: int = 16, verbose: bool = False) -> Tuple[int, List[Tuple[int,int,int,str,int]]]:
    """
    Apply the DDL operation for a number of iterations.

    Parameters:
      x           : input integer (will be masked to 'bits' bits)
      iterations  : number of DDL iterations to apply (default 4)
      bits        : word width (default 16)
      verbose     : if True, prints step-by-step progress

    Returns:
      final_value, states_list

    states_list is a list of tuples:
      (step_index, before_value, HW, description, after_value)
    """
    mask = (1 << bits) - 1
    cur = x & mask
    states = []
    if verbose:
        print(f"Initial: 0x{cur:0{bits//4}X} ({cur:0{bits}b})")
    for i in range(1, iterations + 1):
        after, HW, desc = ddl_step(cur, bits=bits)
        states.append((i, cur, HW, desc, after))
        if verbose:
            print(f" Step {i}: {desc}")
            print(f"   before = 0x{cur:0{bits//4}X} {cur:0{bits}b}")
            print(f"   after  = 0x{after:0{bits//4}X} {after:0{bits}b}\n")
        cur = after & mask
    return cur, states

def _example_run():
    """Print example traces for a few test vectors (used when script is executed)."""
    examples = [0x0001, 0x0003, 0xC003, 0xFFFF, 0x0000, 0x1234]
    print("DDL example runs (4 iterations, 16-bit):\n")
    for ex in examples:
        final, states = ddl(ex, iterations=4, bits=16, verbose=False)
        print(f"Input 0x{ex:04X}:")
        for (i, before, HW, desc, after) in states:
            print(f"  Step {i}: {desc}")
            print(f"    before = 0x{before:04X} {before:016b} (HW={HW})")
            print(f"    after  = 0x{after:04X} {after:016b}")
        print(f"  Final: 0x{final:04X} {final:016b}\n")

if __name__ == "__main__":
    # Demo CLI: run examples
    _example_run()