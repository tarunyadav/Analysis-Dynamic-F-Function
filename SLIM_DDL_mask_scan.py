#!/usr/bin/env python3
"""
slim_mask_scan_builtin_key_schedule.py

Scan candidate 16-bit masks for high-order integral (zero-sum) bias in
SLIM+DDL Feistel using the SLIM key schedule (port of provided C code).

Usage examples:

# sliding windows width 5 (default), 200k trials
python slim_mask_scan_builtin_key_schedule.py --master-key-hex 0123456789ABCDEF0123 --trials 200000 --batch-size 20000

# test all masks weight <= 4 (be careful with runtime)
python slim_mask_scan_builtin_key_schedule.py --master-key-hex 0123456789ABCDEF0123 --tests weight_leq --max-weight 4 --trials 50000 --batch-size 5000

# test single mask only
python slim_mask_scan_builtin_key_schedule.py --master-key-hex 0123456789ABCDEF0123 --test-masks 0x001F --trials 200000 --batch-size 20000
"""

import argparse, time, re, itertools
import numpy as np
import secrets

# ---------- Defaults ----------
DEFAULT_TRIALS = pow(2,14)#200_000
DEFAULT_BATCH = pow(2,10) #20_000
DEFAULT_ROUNDS = 32
SEED_TRIALS = 98765

# SLIM S-box (from the C code)
SLIM_SBOX = [12, 5, 6, 11, 9, 0, 10, 13, 3, 14, 15, 8, 4, 7, 1, 2]

# ---------- DDL and S-box precompute ----------
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

def precompute_tables():
    SBOX4 = np.array(SLIM_SBOX, dtype=np.uint16)
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

# ---------- Key schedule port (from C code) ----------
def circular_shift_4bit(value, moves):
    moves = moves & 3
    value &= 0xF
    return (((value << moves) & 0xF) | ((value >> (4 - moves)) & 0xF)) & 0xF

def key_scheduling(master_nibbles, rounds):
    if len(master_nibbles) != 20:
        raise ValueError("master_nibbles must be length 20")
    master_key = [int(x) & 0xF for x in master_nibbles]

    reversed_index = 19
    round_key = [0] * max(rounds, 5)
    for m in range(5):
        value = 0
        for n in range(4):
            base = master_key[reversed_index]
            value += (base << (n * 4))
            reversed_index -= 1
        round_key[m] = value & 0xFFFF

    msb = [master_key[i] for i in range(10)]
    lsb = [master_key[i + 10] for i in range(10)]
    intermediate_msb_value = [0]*10
    intermediate_lsb_value = [0]*10
    nibble_index = 9
    register_state = 0

    for p in range(0, max(0, rounds - 5)):
        output = 0
        for y in range(4):
            if nibble_index == -1:
                lsb = intermediate_lsb_value.copy()
                msb = intermediate_msb_value.copy()
                intermediate_lsb_value = [0]*10
                intermediate_msb_value = [0]*10
                nibble_index = 9
                register_state += 1

            shifted_lsb_nibble = circular_shift_4bit(lsb[nibble_index], 2)
            shifted_msb_nibble = circular_shift_4bit(msb[nibble_index], 3)
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

    if len(round_key) < rounds:
        round_key += [0] * (rounds - len(round_key))
    return np.array(round_key[:rounds], dtype=np.uint16)

# ---------- Feistel encrypt (vectorized) ----------
def encrypt_blocks_vec(pts_uint32, round_keys, SBOX_LAYER_TABLE, DDL_MAP):
    L = (pts_uint32 >> 16).astype(np.uint16)
    R = (pts_uint32 & 0xFFFF).astype(np.uint16)
    for k in round_keys:
        idx = np.bitwise_xor(R, np.uint16(k))
        sbox_out = SBOX_LAYER_TABLE[idx]
        fval = DDL_MAP[sbox_out]
        L, R = R, np.bitwise_xor(L, fval)
    return L, R

# ---------- Mask utilities ----------
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

# ---------- Single-mask test ----------
def test_mask_with_roundkeys(mask, round_keys, trials, batch_size, SBOX_LAYER_TABLE, DDL_MAP):
    k = bin(mask).count("1")
    if k == 0: raise ValueError("Mask must be nonzero")
    set_size = 1 << k
    if trials % batch_size != 0: raise ValueError("trials must be divisible by batch_size")
    rng = np.random.default_rng(SEED_TRIALS + (mask & 0xFFFF))
    # rng = np.random.default_rng()
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
    return {"mask": mask, "k": k, "trials": trials,
            "zero_subspace": zero_sub, "zero_random": zero_rand,
            "prob_subspace": zero_sub / trials, "prob_random": zero_rand / trials}

# ---------- Mask generators ----------
def sliding_windows_masks(width):
    return [((1<<width)-1) << pos for pos in range(0, 16-width+1)]

def nibble_masks():
    return [0xF << (4*i) for i in range(4)]

def single_bit_masks():
    return [1<<i for i in range(16)]

def all_masks_weight_leq(W):
    masks = []
    bits = list(range(16))
    for w in range(1, W+1):
        for comb in itertools.combinations(bits, w):
            mask = 0
            for b in comb:
                mask |= (1<<b)
            masks.append(mask)
    return masks

# ---------- Scan driver ----------
def run_scan(master_nibbles, rounds, masks, trials, batch_size):
    # build round keys
    round_keys = key_scheduling(master_nibbles, rounds)
    SBOX_LAYER_TABLE, DDL_MAP = precompute_tables()
    print("Using %d round-keys" % len(round_keys))
    results = []
    for mi, mask in enumerate(masks, start=1):
        t0 = time.time()
        res = test_mask_with_roundkeys(mask, round_keys, trials, batch_size, SBOX_LAYER_TABLE, DDL_MAP)
        dt = time.time() - t0
        #print(f"[{mi}/{len(masks)}] mask=0x{mask:04X} mask_size={res['k']} prob_zero_sum={res['prob_subspace']:.6f} dt={dt:.2f}s")
        results.append(res)
    results_sorted = sorted(results, key=lambda r: r['prob_subspace'], reverse=True)
    return results_sorted

# ---------- CLI parsing helpers ----------
def parse_master_key_hex(s):
    s = s.strip()
    if len(s) != 20:
        raise ValueError("master-key-hex must be exactly 20 hex characters (20 nibbles = 80 bits)")
    return [int(ch, 16) for ch in s]

def parse_master_key_nibbles(s):
    parts = re.split(r'[\s,;]+', s.strip())
    if len(parts) != 20:
        raise ValueError("master-key-nibbles must contain 20 nibble values")
    return [int(p,0) & 0xF for p in parts]

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-key-hex", help="20 hex chars (80 bits) representing 20 nibbles, e.g. 0123456789ABCDEF0123 or random",default="random") #0123456789ABCDEF0123
    parser.add_argument("--master-key-nibbles", help="20 nibble values (0..F) separated by spaces or commas")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--tests", choices=("sliding","nibbles","single","weight_leq","all"), default="sliding",
                        help="which mask set to test")
    parser.add_argument("--width", type=int, default=2, help="window width for sliding (only for tests=sliding)")
    parser.add_argument("--max-weight", type=int, default=2, help="max weight for weight_leq test")
    parser.add_argument("--test-masks", nargs="+", help="explicit masks to test, hex or decimal (overrides --tests)")
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
    trials = args.trials
    if args.test_masks:
        masks = []
        for t in args.test_masks:
            if str(t).lower().startswith("0x"):
                masks.append(int(t,16))
            else:
                masks.append(int(t,0))
    else:
        if args.tests == "sliding":
            masks = sliding_windows_masks(args.width)
            trials = pow(2,16-args.width)
        elif args.tests == "nibbles":
            masks = nibble_masks()
        elif args.tests == "single":
            masks = single_bit_masks()
        elif args.tests == "weight_leq":
            masks = all_masks_weight_leq(args.max_weight)
        elif args.tests == "all":
            masks = list(range(1,1<<16))
        else:
            masks = sliding_windows_masks(args.width)
    
    print("Testing %d masks; trials=%d batch=%d rounds=%d" % (len(masks), trials, args.batch_size, args.rounds))
    results_sorted = run_scan(master_nibbles, args.rounds, masks, trials, args.batch_size)

    print("\nTop masks by prob_zero_sum:")
    for r in results_sorted[:]:
        print(f"mask=0x{r['mask']:04X} | mask_size={r['k']:2d} | Trials={trials} | Total Data={trials*pow(2,r['k'])} | prob_zero_sum={r['prob_subspace']:.6f} | count_zero_sum={r['zero_subspace']}")
    # optionally, save results to a CSV file
    # You can uncomment to write results.csv
    # import csv
    # with open("mask_scan_results.csv","w",newline='') as f:
    #     w = csv.writer(f)
    #     w.writerow(["mask_hex","popcount","prob_subspace","hits","trials"])
    #     for r in results_sorted:
    #         w.writerow([f"0x{r['mask']:04X}", r['k'], r['prob_subspace'], r['zero_subspace'], r['trials']])

if __name__ == "__main__":
    main()