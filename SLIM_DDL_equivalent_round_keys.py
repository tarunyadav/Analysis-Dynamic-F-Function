#!/usr/bin/env python3
"""
use_per_round_equiv_keys.py

For a chosen master and plaintext:
 - compute exact per-round equivalence sets E_r = { K' | DDL(S(R_r ^ K')) == DDL(S(R_r ^ K_r)) }
   where R_r is the real right side before round r and K_r is the original round key.
 - print sizes and a capped sample of each E_r.
 - sample a small number of full 32-round candidate key sequences by taking one K' from each E_r
   (so candidate_keys = [K'_0, K'_1, ..., K'_31] not derived from a master).
 - run full 32-round encryption with the candidate_keys and compare ciphertext to baseline.
 - print all steps in a verbose, auditable fashion.

Defaults:
 - master = 0123456789ABCDEF0123
 - plaintext = 0x00000000
 - samples = 3
"""

import numpy as np
import argparse, time, random
from collections import defaultdict

# ---------- Sbox / inverse (SLIM / PRESENT) ----------
SBOX4 = [0xC,0x5,0x6,0xB,0x9,0x0,0xA,0xD,0x3,0xE,0xF,0x8,0x4,0x7,0x1,0x2]
INV_SBOX4 = [0]*16
for i,v in enumerate(SBOX4):
    INV_SBOX4[v] = i

def sbox_layer_16_scalar(v):
    out = 0
    for i in range(4):
        nib = (v >> (4*i)) & 0xF
        out |= (SBOX4[nib] & 0xF) << (4*i)
    return out & 0xFFFF

def inv_sbox_layer_16_scalar(v):
    out = 0
    for i in range(4):
        nib = (v >> (4*i)) & 0xF
        out |= (INV_SBOX4[nib] & 0xF) << (4*i)
    return out & 0xFFFF

# ---------- DDL implementation (scalar) and precompute-----------
def rol16_scalar(x,n): return (((x << n) & 0xFFFF) | (x >> (16 - n))) & 0xFFFF
def ror16_scalar(x,n): return ((x >> n) | ((x << (16 - n)) & 0xFFFF)) & 0xFFFF

def ddl_step_scalar(x):
    x &= 0xFFFF
    hw = bin(int(x)).count("1")
    if hw == 0:
        return x ^ ror16_scalar(x, 0)
    if hw & 1:
        shift = hw & 0xF
        return x ^ rol16_scalar(x, shift)
    else:
        shift = (hw - 1) & 0xF
        return x ^ ror16_scalar(x, shift)

def ddl_scalar(x, iterations=4):
    cur = int(x) & 0xFFFF
    for _ in range(iterations):
        cur = ddl_step_scalar(cur)
    return cur & 0xFFFF

# Precompute DDL_MAP and SBOX_LAYER_TABLE for vectorized checks
SBOX_LAYER_TABLE = np.empty(1<<16, dtype=np.uint16)
for v in range(1<<16):
    out = 0
    for i in range(4):
        nib = (v >> (4*i)) & 0xF
        out |= (SBOX4[nib] & 0xF) << (4*i)
    SBOX_LAYER_TABLE[v] = out

DDL_MAP = np.empty(1<<16, dtype=np.uint16)
for v in range(1<<16):
    DDL_MAP[v] = ddl_scalar(v, iterations=4)

# ---------- SLIM key schedule & encrypt ----------
SLIM_SBOX = [12,5,6,11,9,0,10,13,3,14,15,8,4,7,1,2]
def circular_shift_4bit(value, moves):
    moves &= 3
    value &= 0xF
    return (((value << moves) & 0xF) | (value >> (4 - moves))) & 0xF

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
            value += (base << (4*n))
            reversed_index -= 1
        round_key[m] = value & 0xFFFF
    msb = [master_key[i] for i in range(10)]
    lsb = [master_key[i + 10] for i in range(10)]
    intermediate_msb_value = [0]*10
    intermediate_lsb_value = [0]*10
    nibble_index = 9
    for p in range(0, max(0, rounds - 5)):
        output = 0
        for y in range(4):
            if nibble_index == -1:
                lsb = intermediate_lsb_value.copy()
                msb = intermediate_msb_value.copy()
                intermediate_lsb_value = [0]*10
                intermediate_msb_value = [0]*10
                nibble_index = 9
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

def encrypt_with_round_keys(plaintext32, round_keys):
    pts = np.array([np.uint32(plaintext32)], dtype=np.uint32)
    L = (pts >> 16).astype(np.uint16)
    R = (pts & 0xFFFF).astype(np.uint16)
    for k in round_keys:
        idx = np.bitwise_xor(R, np.uint16(k))
        sbox_out = SBOX_LAYER_TABLE[idx]
        fval = DDL_MAP[sbox_out]
        L, R = R, np.bitwise_xor(L, fval)
    return (int(L[0]), int(R[0]))

# ---------- helpers: master nibble conversions ----------
def master_hex_to_nibbles(hex20):
    if len(hex20) != 20:
        raise ValueError("master must be 20 hex chars (20 nibbles)")
    return [int(ch,16) for ch in hex20.upper()]

def master_nibbles_from_roundkeys_direct(rklist):
    nibbles = [None]*20
    rev = 19
    for m in range(5):
        val = int(rklist[m]) & 0xFFFF
        for n in range(4):
            nibb = (val >> (4*n)) & 0xF
            nibbles[rev] = nibb
            rev -= 1
    return nibbles

def master_nibbles_to_hex(nibbles):
    return ''.join('{:X}'.format(n & 0xF) for n in nibbles)

# ---------- compute original states & round keys ----------
def simulate_baseline(master_hex, plaintext32, rounds=32):
    master_nibs = master_hex_to_nibbles(master_hex)
    rks = key_scheduling(master_nibs, rounds)
    # compute R_i sequence and full states
    L = (plaintext32 >> 16) & 0xFFFF
    R = plaintext32 & 0xFFFF
    states = [(int(L), int(R))]  # (L0,R0)
    for r in range(rounds):
        K = int(rks[r])
        idx = (R ^ K) & 0xFFFF
        sbox_out = int(SBOX_LAYER_TABLE[idx])
        fval = int(DDL_MAP[sbox_out])
        L, R = R, (L ^ fval) & 0xFFFF
        states.append((int(L), int(R)))
    return rks, states

# ---------- compute exact equivalence sets per round (vectorized) ----------
def compute_equiv_sets_vectorized(rks, states, rounds=32, cap_show=40):
    """
    For each round r compute E_r = { K' in 0..65535 | DDL(S(Rr ^ K')) == DDL(S(Rr ^ Kr)) }.
    Uses numpy vectorized operations for speed.
    Returns list of lists (equiv sets).
    """
    equiv_sets = []
    all_keys = np.arange(1<<16, dtype=np.uint16)
    for r in range(rounds):
        Rr = np.uint16(states[r][1])   # R before round r
        Kr = np.uint16(rks[r])
        # compute target y
        idx_target = np.bitwise_xor(Rr, Kr).astype(np.uint32)
        sbox_x = SBOX_LAYER_TABLE[idx_target]
        y_target = int(DDL_MAP[sbox_x])
        # compute y for all candidate K'
        # idxs = Rr ^ all_keys  (vectorized)
        idxs = np.bitwise_xor(np.uint16(Rr), all_keys).astype(np.uint32)
        sbox_out_all = SBOX_LAYER_TABLE[idxs]        # uint16 array
        y_all = DDL_MAP[sbox_out_all]                # uint16 array
        # find K' where y_all == y_target
        mask = (y_all.astype(np.uint32) == np.uint32(y_target))
        keys_eq = all_keys[mask].astype(np.uint16)
        keys_list = keys_eq.tolist()
        equiv_sets.append(keys_list)
        # printing summary
        print(f"Round {r:02d}: R=0x{int(Rr):04X} K=0x{int(Kr):04X} -> DDL_y=0x{y_target:04X}  equiv_count={len(keys_list)}")
        if len(keys_list) <= cap_show:
            print("  members:", ", ".join(f"0x{k:04X}" for k in keys_list))
        else:
            sample_show = keys_list[:cap_show]
            print("  sample:", ", ".join(f"0x{k:04X}" for k in sample_show) + ", ...")
    return equiv_sets

# ---------- main driver ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--master", default="0123456789ABCDEF0123", help="20-hex-nibble master key")
    p.add_argument("--plaintext", default="0x00000000", help="32-bit plaintext hex")
    p.add_argument("--samples", type=int, default=3, help="how many candidate per-round-key sequences to try")
    p.add_argument("--seed", type=int, default=2025, help="random seed")
    p.add_argument("--cap-show", type=int, default=100, help="how many entries to show per round's set")
    args = p.parse_args()

    master_hex = args.master
    pt = int(args.plaintext, 0) & 0xFFFFFFFF
    samples = args.samples
    seed = args.seed

    print("Configuration:")
    print(" master:", master_hex)
    print(" plaintext:", f"0x{pt:08X}")
    print(" samples to try:", samples)
    print(" cap-show per round:", args.cap_show)
    print("----\n")

    # baseline
    t0 = time.time()
    rks, states = simulate_baseline(master_hex, pt, rounds=32)
    t1 = time.time()
    print("Baseline simulation done (round keys derived from master). Time: {:.3f}s".format(t1-t0))
    print("Baseline first 8 round-keys:", ", ".join(f"0x{int(rks[i]):04X}" for i in range(8)))
    print("Baseline initial state (L0,R0):", f"(0x{states[0][0]:04X}, 0x{states[0][1]:04X})")
    print("----\n")

    # compute equivalence sets per round (vectorized)
    print("Computing exact per-round equivalence sets (vectorized) ...")
    t0 = time.time()
    equiv_sets = compute_equiv_sets_vectorized(rks, states, rounds=32, cap_show=args.cap_show)
    t1 = time.time()
    print("Done. time {:.3f}s\n".format(t1-t0))

    # confirm that original K is inside each set
    for r in range(32):
        Korig = int(rks[r])
        if Korig not in equiv_sets[r]:
            print(f"WARNING: original round key 0x{Korig:04X} NOT in equivalence set for round {r}")

    # sample full candidate sequences by picking one K' from each equiv set
    rng = random.Random(seed)
    candidate_sequences = []
    total_comb = 1
    for s in equiv_sets[:5]:  # quick check product for first 5
        total_comb *= max(1, len(s))
    print("Rough product of sizes for first 5 rounds:", total_comb)
    # We will sample 'samples' distinct sequences
    tries = 0
    seen = set()
    while len(candidate_sequences) < samples and tries < samples * 200:
        seq = []
        for r in range(32):
            lst = equiv_sets[r]
            if not lst:
                # fallback to original
                seq.append(int(rks[r]))
            else:
                seq.append(int(rng.choice(lst)))
        tup = tuple(seq)
        if tup in seen:
            tries += 1
            continue
        seen.add(tup)
        candidate_sequences.append(seq)
        tries += 1

    # compute baseline ciphertext
    Lb, Rb = encrypt_with_round_keys(pt, rks)
    ct_base = ((Lb << 16) | Rb) & 0xFFFFFFFF
    print("Baseline ciphertext:", f"0x{ct_base:08X}\n")

    # For each candidate sequence print what keys tried per round and verify ciphertext
    for si, seq in enumerate(candidate_sequences):
        print("=== Candidate sample", si+1, "===")
        print("Attempting per-round keys (K'_0 .. K'_31):")
        print(", ".join(f"0x{seq[r]:04X}" for r in range(32)))
        # show first 8 for readability as well
        print("first 8 round keys:", ", ".join(f"0x{seq[r]:04X}" for r in range(8)))
        # show which rounds changed from original
        diffs = [r for r in range(32) if seq[r] != int(rks[r])]
        print(f"Rounds changed vs original: {len(diffs)} (indices: {diffs[:40]})")
        # encrypt with this per-round key schedule
        Lc, Rc = encrypt_with_round_keys(pt, np.array(seq, dtype=np.uint16))
        ct = ((Lc << 16) | Rc) & 0xFFFFFFFF
        print("Candidate ciphertext:", f"0x{ct:08X}")
        if ct == ct_base:
            print("==> MATCH: candidate produces same ciphertext as baseline (full 32-round match).")
        else:
            print("==> NO MATCH: differs from baseline.")
            xor = ct_base ^ ct
            print(" XOR = 0x{0:08X}, Hamming = {1}".format(xor, bin(xor).count("1")))
        # print round-by-round states comparison for first 6 rounds to demonstrate identity
        print("\nRound-by-round states (baseline vs candidate) first 8 rounds:")
        # compute states for candidate to display
        # derive states
        L = (pt >> 16) & 0xFFFF
        R = pt & 0xFFFF
        states_base = [(int(L), int(R))]
        states_cand = [(int(L), int(R))]
        # simulate baseline and candidate to compare states (only up to 8)
        for r in range(32):
            # baseline
            Kb = int(rks[r])
            idx_b = (states_base[-1][1] ^ Kb) & 0xFFFF
            y_b = int(DDL_MAP[SBOX_LAYER_TABLE[idx_b]])
            Lb_next = states_base[-1][1]
            Rb_next = (states_base[-1][0] ^ y_b) & 0xFFFF
            states_base.append((int(Lb_next), int(Rb_next)))
            # candidate
            Kc = int(seq[r])
            idx_c = (states_cand[-1][1] ^ Kc) & 0xFFFF
            y_c = int(DDL_MAP[SBOX_LAYER_TABLE[idx_c]])
            Lc_next = states_cand[-1][1]
            Rc_next = (states_cand[-1][0] ^ y_c) & 0xFFFF
            states_cand.append((int(Lc_next), int(Rc_next)))
            same = (states_base[-1] == states_cand[-1])
            print(f" r{r}: Baseline L={states_base[-1][0]:04X} R={states_base[-1][1]:04X} Kb=0x{Kb:04X} -> y=0x{y_b:04X}",
                  f" | Candidate L={states_cand[-1][0]:04X} R={states_cand[-1][1]:04X} Kc=0x{Kc:04X} -> y=0x{y_c:04X} Same={same}")
        print("\n")  # end of sample

    print("All samples processed. Done.")

if __name__ == "__main__":
    main()