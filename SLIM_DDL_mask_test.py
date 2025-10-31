#!/usr/bin/env python3
"""
slim_mask_test_with_builtin_key_schedule.py

- Implements SLIM key_scheduling ported from provided C code (4-bit nibbles).
- Builds round-keys from an 80-bit master key.
- Runs the high-order integral mask test (vary bits in 'mask') using Sbox->DDL round function.

Usage examples:
  python slim_mask_test_with_builtin_key_schedule.py --master-key-hex 0123456789ABCDEF0123 --mask 0x001F
  python slim_mask_test_with_builtin_key_schedule.py --master-key-nibbles "0 1 2 3 4 5 6 7 8 9 a b c d e f 0 1 2 3" --mask 0x03E0 --rounds 32

Notes:
 - The script expects master key expressed as 20 nibbles (each nibble 0..F).
 - The implemented key schedule follows the C code you provided (nibble rotation, S-box, msb/lsb splitting).
"""

import argparse, time, re
import numpy as np
import secrets

# ---------- Parameters ----------
DEFAULT_TRIALS = 200_000
DEFAULT_BATCH = 20_000
DEFAULT_ROUNDS = 32
DEFAULT_SET = 32
SEED_TRIALS = 98765

# ---------- PRESENT-like SBOX (SLIM Sbox from C code) ----------
SLIM_SBOX = [12, 5, 6, 11, 9, 0, 10, 13, 3, 14, 15, 8, 4, 7, 1, 2]

# ---------- DDL (same as before) ----------
def ddl_step_scalar(x):
    x &= 0xFFFF
    hw = bin(x).count("1")
    if hw == 0:
        shift = 0
        return x ^ (((x >> shift) | ((x << (16 - shift)) & 0xFFFF)) & 0xFFFF)
    if hw & 1:
        shift = hw & 0xF
        return x ^ ((((x << shift) & 0xFFFF) | (x >> (16 - shift))) & 0xFFFF)
    else:
        shift = (hw - 1) & 0xF
        return x ^ (((x >> shift) | ((x << (16 - shift)) & 0xFFFF)) & 0xFFFF)

def ddl_scalar(x, iterations=4):
    cur = x & 0xFFFF
    for _ in range(iterations):
        cur = ddl_step_scalar(cur)
    return cur & 0xFFFF

# ---------- Precompute tables ----------
def precompute_tables():
    SBOX4 = np.array([12,5,6,11,9,0,10,13,3,14,15,8,4,7,1,2], dtype=np.uint16)
    SBOX_LAYER_TABLE = np.empty(1<<16, dtype=np.uint16)
    for v in range(1<<16):
        out = 0
        for i in range(4):
            nib = (v >> (4*i)) & 0xF
            out |= (int(SBOX4[nib]) & 0xF) << (4*i)
        SBOX_LAYER_TABLE[v] = out
    DDL_MAP = np.empty(1<<16, dtype=np.uint16)
    for v in range(1<<16):
        DDL_MAP[v] = ddl_scalar(v, iterations=4)
    return SBOX_LAYER_TABLE, DDL_MAP

# ---------- Key schedule (ported from provided C code) ----------
def circular_shift_4bit(value, moves):
    """Rotate a 4-bit value left by moves (mod 4)."""
    moves = moves & 3
    value &= 0xF
    return (( (value << moves) & 0xF) | ((value >> (4 - moves)) & 0xF)) & 0xF

def key_scheduling(master_nibbles, rounds):
    """
    master_nibbles: list/iterable of 20 nibble values (0..15), ordered as in C code:
      master_key[0] .. master_key[19] where master_key[0] is left-most msb nibble (per their comment).
    rounds: total number of round keys desired (>=5).
    Returns list/np.array of 16-bit round keys length 'rounds' (or at least rounds as generated).
    """
    if len(master_nibbles) != 20:
        raise ValueError("master_nibbles must be length 20")
    # copy into local array
    master_key = [int(x) & 0xF for x in master_nibbles]

    # initial 5 round keys: composed from reversed master_key nibbles (reversed_index = 19 moving down)
    reversed_index = 19
    round_key = [0] * max(rounds, 5)  # ensure length at least rounds; will fill up to rounds
    for m in range(5):
        value = 0
        for n in range(4):
            base = master_key[reversed_index]
            value += (base << (n * 4))  # base is nibble, placed at 4*n
            reversed_index -= 1
        round_key[m] = value & 0xFFFF

    # split master key into msb[0..9] and lsb[0..9]
    msb = [master_key[i] for i in range(10)]
    lsb = [master_key[i+10] for i in range(10)]

    intermediate_msb_value = [0]*10
    intermediate_lsb_value = [0]*10

    nibble_index = 9
    # register_state unused in C beyond increment; we keep similar control flow
    register_state = 0

    # produce round-keys from index 5 up to rounds-1 (if rounds>5)
    for p in range(0, max(0, rounds - 5)):
        output = 0
        nibble_output = 0
        # produce four nibbles per round-key (y=0..3)
        for y in range(4):
            if nibble_index == -1:
                # reset lsb, msb from intermediate arrays
                lsb = intermediate_lsb_value.copy()
                msb = intermediate_msb_value.copy()
                # clear intermediate arrays
                intermediate_lsb_value = [0]*10
                intermediate_msb_value = [0]*10
                nibble_index = 9
                register_state += 1

            # circular shifts
            shifted_lsb_nibble = circular_shift_4bit(lsb[nibble_index], 2)
            shifted_msb_nibble = circular_shift_4bit(msb[nibble_index], 3)

            # sbox lookup using XOR of shifted_lsb and msb[nibble_index]
            idx = (shifted_lsb_nibble ^ msb[nibble_index]) & 0xF
            sbox_lsb_msb = SLIM_SBOX[idx]
            intermediate_lsb_value[nibble_index] = sbox_lsb_msb

            nibble_output = (shifted_msb_nibble ^ sbox_lsb_msb) & 0xF
            intermediate_msb_value[nibble_index] = nibble_output

            output |= (nibble_output << (4 * y))
            nibble_index -= 1

        round_key_index = p + 5
        if round_key_index >= len(round_key):
            round_key.extend([0]*(round_key_index - len(round_key) + 1))
        round_key[round_key_index] = output & 0xFFFF
    # ensure length == rounds
    if len(round_key) < rounds:
        round_key += [0] * (rounds - len(round_key))
    return np.array(round_key[:rounds], dtype=np.uint16)

# ---------- Encrypt function (vectorized) ----------
def encrypt_blocks_vec(pts_uint32, round_keys, SBOX_LAYER_TABLE, DDL_MAP):
    L = (pts_uint32 >> 16).astype(np.uint16)
    R = (pts_uint32 & 0xFFFF).astype(np.uint16)
    for k in round_keys:
        idx = np.bitwise_xor(R, np.uint16(k))
        sbox_out = SBOX_LAYER_TABLE[idx]
        fval = DDL_MAP[sbox_out]
        L, R = R, np.bitwise_xor(L, fval)
    return L, R

# ---------- Helpers for mask test ----------
def make_mapped_vals(mask):
    bit_positions = [i for i in range(16) if (mask >> i) & 1]
    k = len(bit_positions)
    vals = np.arange(1 << k, dtype=np.uint32)
    mapped = np.zeros_like(vals, dtype=np.uint32)
    for j, val in enumerate(vals):
        mv = 0
        for bit_index, pos in enumerate(bit_positions):
            if (val >> bit_index) & 1:
                mv |= (1 << pos)
        mapped[j] = mv
    return mapped, bit_positions

def test_mask_with_roundkeys(mask, round_keys, trials, batch_size, SBOX_LAYER_TABLE, DDL_MAP,set_size):
    k = bin(mask).count("1")
    if k == 0:
        raise ValueError("Mask must be nonzero")
    if (set_size == None):
        set_size = 1 << k
    
    
    if trials % batch_size != 0:
        raise ValueError("trials must be divisible by batch_size")
    # rng = np.random.default_rng(SEED_TRIALS + (mask & 0xFFFF))
    rng = np.random.default_rng()
    mapped_vals, _ = make_mapped_vals(mask)
    batches = trials // batch_size
    zero_sub = 0
    zero_rand = 0
    for b in range(batches):
        outside_bits = 16 - k
        fixed_outside = rng.integers(0, 1 << outside_bits, size=(batch_size,), dtype=np.uint32)
        base_R = np.zeros(batch_size, dtype=np.uint32)
        out_pos = [i for i in range(16) if ((mask >> i) & 1) == 0]
        for idx_out, pos in enumerate(out_pos):
            bit_values = ((fixed_outside >> idx_out) & 1).astype(np.uint32)
            base_R |= (bit_values << pos)
        Lvals = rng.integers(0, 1<<16, size=(batch_size,), dtype=np.uint32)
        xor_right_sub = np.zeros(batch_size, dtype=np.uint16)
        xor_right_rand = np.zeros(batch_size, dtype=np.uint16)
        rand_Rs = rng.integers(0, 1<<16, size=(set_size, batch_size), dtype=np.uint16)
        for i in range(set_size):
            Rs = (base_R | mapped_vals[i]).astype(np.uint32)
            pts = (Lvals << 16) | Rs
            _, Rc = encrypt_blocks_vec(pts, round_keys, SBOX_LAYER_TABLE, DDL_MAP)
            xor_right_sub ^= Rc
            Rs_rand = rand_Rs[i].astype(np.uint32)
            pts_rand = (Lvals << 16) | Rs_rand
            _, Rc2 = encrypt_blocks_vec(pts_rand, round_keys, SBOX_LAYER_TABLE, DDL_MAP)
            xor_right_rand ^= Rc2
        zero_sub += int((xor_right_sub == 0).sum())
        zero_rand += int((xor_right_rand == 0).sum())
    return {"mask": mask, "mask_size": k, "trials": trials, "total data": trials*pow(2,k),
            "zero_sum_count_mask": zero_sub, "zero_sum_count_random": zero_rand,
            "prob_zero_sum_mask": zero_sub / trials, "prob_zero_sum_random": zero_rand / trials}

# ---------- CLI ----------
def parse_master_key_hex(s):
    s = s.strip()
    if len(s) != 20:
        raise ValueError("master-key-hex must be exactly 20 hex characters (20 nibbles = 80 bits)")
    nibbles = []
    for ch in s:
        nibbles.append(int(ch, 16))
    return nibbles

def parse_master_key_nibbles(s):
    parts = re.split(r'[\s,;]+', s.strip())
    if len(parts) != 20:
        raise ValueError("master-key-nibbles must contain 20 nibble values")
    return [int(p,0) & 0xF for p in parts]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-key-hex", help="20 hex chars (80-bit) representing 20 nibbles, e.g. 0123456789ABCDEF0123",default="random") #0123456789ABCDEF0123
    parser.add_argument("--master-key-nibbles", help="20 nibble values (0..F) separated by spaces or commas")
    parser.add_argument("--mask", required=True, help="16-bit mask (hex 0x... or decimal)")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--set-size", type=int, default=None)
    args = parser.parse_args()


    master_key_hex = args.master_key_hex
    if master_key_hex == "random":
        master_key_hex = secrets.token_bytes(10).hex().zfill(20)
    if master_key_hex:
        master_nibbles = parse_master_key_hex(master_key_hex)
    elif args.master_key_nibbles:
        master_nibbles = parse_master_key_nibbles(args.master_key_nibbles)
    else:
        raise ValueError("Provide either --master-key-hex or --master-key-nibbles")

    ms = args.mask.strip()
    if ms.lower().startswith("0x"):
        mask = int(ms, 16)
    else:
        mask = int(ms, 0)
    if mask <= 0 or mask >= (1<<16):
        raise ValueError("mask must be 16-bit nonzero")

    if args.trials % args.batch_size != 0:
        raise ValueError("trials must be divisible by batch-size")

    # build round keys
    round_keys = key_scheduling(master_nibbles, args.rounds)
    # print("Generated round-keys (first 8):", [f"0x{rk:04X}" for rk in round_keys[:8]], " total:", len(round_keys))

    # precompute SBOX/DDL tables
    SBOX_LAYER_TABLE, DDL_MAP = precompute_tables()

    t0 = time.time()
    res = test_mask_with_roundkeys(mask, round_keys, args.trials, args.batch_size, SBOX_LAYER_TABLE, DDL_MAP,args.set_size)
    t1 = time.time()
    print("Result:", res)
    print("Elapsed: %.2f s" % (t1 - t0))

if __name__ == "__main__":
    main()