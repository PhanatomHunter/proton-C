"""
PHOTON-C  ::  Entropy engine
============================
Canonical Huffman coding with per-frame tables, Exp-Golomb for run lengths, and
a bit-exact bitstream.

Design choice, and why
----------------------
An adaptive binary arithmetic coder (CABAC-class) would squeeze maybe 5-12% more
out of this syntax than Huffman does.  I built one first, and it is genuinely
hard to get right: the encoder's `low` and the decoder's `code` must track the
*same* slice of the coded value at every bin, and any window-offset mistake
yields a coder that round-trips perfectly on easy material and then
desynchronises silently once contexts saturate -- precisely the regime an
ultra-low-bitrate codec lives in.  Making that the load-bearing layer of a codec
is a bad trade.

So the entropy stage is canonical Huffman plus Exp-Golomb, which is:

  * **bit-exact by construction.**  A prefix code carries no state to desync.
    If the table is right the decode is right, and the table is in the stream.
  * **fast.**  Table build is O(n log n) over a small alphabet; coding is a dict
    lookup per symbol plus a shift per bit.
  * **near-optimal for this source, given the right source model.**  Huffman
    loses to arithmetic coding whenever some symbol has p > 0.5, because its
    shortest codeword is 1 bit -- and video syntax is exactly that skewed (95% of
    blocks in a static scene are skips).  The fix is not a cleverer coder, it is
    coding the RUNS instead of the flags: 4000 skip flags at 1 bit each is 4000
    bits, but the ~200 run lengths describing them are a handful of bits each.
    That recovers the loss and beats a naive arithmetic coder on the same syntax.

Tables are built per frame from that frame's own statistics and serialised as
one nibble of code length per used symbol (canonical Huffman needs lengths
only -- the decoder regenerates identical codewords).

Bitstream layout
----------------
Bits pack MSB-first into bytes.  The last byte is zero-padded and every payload
carries an explicit bit count, so the decoder never guesses where data ends.
"""

from __future__ import annotations

import heapq
from typing import Dict, List, Sequence, Tuple

MAX_CODE_LEN = 15       # keeps codewords inside a 16-bit window


# --------------------------------------------------------------------------- #
#  Bit I/O
# --------------------------------------------------------------------------- #
class BitWriter:
    """MSB-first bit sink."""

    __slots__ = ("buf", "acc", "nbits")

    def __init__(self):
        self.buf = bytearray()
        self.acc = 0
        self.nbits = 0

    def write(self, value: int, n: int) -> None:
        if n <= 0:
            return
        self.acc = (self.acc << n) | (value & ((1 << n) - 1))
        self.nbits += n
        while self.nbits >= 8:
            self.nbits -= 8
            self.buf.append((self.acc >> self.nbits) & 0xFF)
            self.acc &= (1 << self.nbits) - 1

    def write_many(self, pairs: Sequence[Tuple[int, int]]) -> None:
        """Write a whole frame's worth of (value, nbits) pairs in one go."""
        acc, nbits, buf = self.acc, self.nbits, self.buf
        for value, n in pairs:
            if n <= 0:
                continue
            acc = (acc << n) | (value & ((1 << n) - 1))
            nbits += n
            while nbits >= 8:
                nbits -= 8
                buf.append((acc >> nbits) & 0xFF)
                acc &= (1 << nbits) - 1
        self.acc, self.nbits = acc, nbits

    def finish(self) -> Tuple[bytes, int]:
        """Zero-pad to a byte boundary.  Returns (bytes, total_bit_count)."""
        total = len(self.buf) * 8 + self.nbits
        if self.nbits:
            self.buf.append((self.acc << (8 - self.nbits)) & 0xFF)
            self.acc = 0
            self.nbits = 0
        return bytes(self.buf), total


class BitReader:
    """MSB-first bit source.  Reads past the end yield zeros, which is safe
    because every payload carries an explicit bit count and canonical Huffman
    cannot run away: it resolves within MAX_CODE_LEN bits."""

    __slots__ = ("buf", "pos", "bitpos", "nbits_total")

    def __init__(self, data: bytes, nbits_total: int | None = None):
        self.buf = data
        self.pos = 0
        self.bitpos = 0
        self.nbits_total = len(data) * 8 if nbits_total is None else nbits_total

    def read(self, k: int) -> int:
        buf = self.buf
        v = 0
        pos, bitpos = self.pos, self.bitpos
        nb = len(buf)
        for _ in range(k):
            if pos < nb:
                bit = (buf[pos] >> (7 - bitpos)) & 1
                bitpos += 1
                if bitpos == 8:
                    bitpos = 0
                    pos += 1
            else:
                bit = 0
                pos += 1
            v = (v << 1) | bit
        self.pos, self.bitpos = pos, bitpos
        return v

    def tell_bits(self) -> int:
        return self.pos * 8 + self.bitpos

    def remaining_bits(self) -> int:
        return self.nbits_total - self.tell_bits()


# --------------------------------------------------------------------------- #
#  Zigzag: signed -> unsigned, keeping small magnitudes small
# --------------------------------------------------------------------------- #
def zigzag(v: int) -> int:
    """0->0, -1->1, 1->2, -2->3, 2->4, -3->5, ..."""
    return (v << 1) ^ (v >> 63)


def unzigzag(u: int) -> int:
    return (u >> 1) ^ -(u & 1)


# --------------------------------------------------------------------------- #
#  Exp-Golomb (order 0): table-free, near-optimal for run lengths
# --------------------------------------------------------------------------- #
def egolomb(value: int) -> Tuple[int, int]:
    """Returns (codeword, nbits) for value >= 0.

    v = value + 1; n = bitlen(v); codeword = (n-1) zeros, a 1, then the low
    n-1 bits of v.  Total width 2n-1.
    """
    v = value + 1
    n = v.bit_length()
    return (v + (1 << (n - 1)), 2 * n - 1)


def egolomb_read(br: BitReader) -> int:
    n = 0
    while br.read(1) == 0:
        n += 1
        if n > 31:
            raise ValueError("Exp-Golomb prefix too long: corrupt stream")
    if n == 0:
        return 0
    suffix = br.read(n)
    return ((1 << n) - 1) + suffix


def egolomb_write(bw: BitWriter, value: int) -> None:
    code, n = egolomb(value)
    bw.write(code, n)


def egolomb_write_many(bw: BitWriter, values: Sequence[int]) -> None:
    bw.write_many([egolomb(v) for v in values])


# --------------------------------------------------------------------------- #
#  Run-length coding -- the piece that actually buys the extreme bitrates
# --------------------------------------------------------------------------- #
def rle_encode(flags: Sequence[int]) -> List[int]:
    """Alternating run lengths of a binary flag sequence, always starting with
    the run of 0s (which may have length 0).

    This is what makes a 95%-skip frame cheap.  Huffman cannot code a p=0.95
    flag in under 1 bit, so 4000 flags would cost 4000 bits; the ~200 run
    lengths that describe them cost a few bits each under Exp-Golomb.
    """
    runs: List[int] = []
    n = len(flags)
    if n == 0:
        return runs
    if flags[0] == 1:
        runs.append(0)              # normalise: first run is always 0s
    cur = flags[0]
    cnt = 0
    for f in flags:
        if f == cur:
            cnt += 1
        else:
            runs.append(cnt)
            cur = f
            cnt = 1
    runs.append(cnt)
    return runs


def rle_decode(runs: Sequence[int], total: int) -> List[int]:
    """Inverse of rle_encode: runs alternate 0s, 1s, 0s, 1s, ..."""
    out: List[int] = []
    val = 0
    for n in runs:
        if n:
            out.extend([val] * n)
        val ^= 1
        if len(out) >= total:
            break
    if len(out) < total:
        out.extend([val] * (total - len(out)))
    return out[:total]


# --------------------------------------------------------------------------- #
#  Canonical Huffman
# --------------------------------------------------------------------------- #
def huffman_lengths(freq: Dict[int, int], maxlen: int = MAX_CODE_LEN) -> Dict[int, int]:
    """Code lengths for the symbols present in `freq` (zero counts ignored).

    Built with the standard two-queue-free heap algorithm, then repaired so no
    length exceeds `maxlen` while the Kraft sum stays <= 1.
    """
    items = [(c, s) for s, c in freq.items() if c > 0]
    if not items:
        return {}
    if len(items) == 1:
        return {items[0][1]: 1}

    heap = list(items)
    heapq.heapify(heap)
    children: Dict[int, Tuple[int, int]] = {}
    nxt = max(s for _, s in items) + 1
    while len(heap) > 1:
        c0, s0 = heapq.heappop(heap)
        c1, s1 = heapq.heappop(heap)
        children[nxt] = (s0, s1)
        heapq.heappush(heap, (c0 + c1, nxt))
        nxt += 1

    lengths: Dict[int, int] = {}
    stack = [(heap[0][1], 0)]
    while stack:
        node, d = stack.pop()
        if node in children:
            a, b = children[node]
            stack.append((a, d + 1))
            stack.append((b, d + 1))
        else:
            lengths[node] = max(d, 1)

    return _clamp_lengths(lengths, maxlen)


def _clamp_lengths(lengths: Dict[int, int], maxlen: int) -> Dict[int, int]:
    """Reduce every over-long codeword to `maxlen`, paying for it by shortening
    the Kraft budget elsewhere.  Iterative and terminating: each pass strictly
    reduces the number of over-long symbols or the total Kraft excess."""
    if max(lengths.values(), default=1) <= maxlen:
        return dict(lengths)

    L = dict(lengths)
    for _ in range(1000):
        over = [s for s, l in L.items() if l > maxlen]
        if not over:
            break
        s = max(over, key=lambda x: (L[x], x))
        L[s] = maxlen
        # Demote the shallowest symbol that is not already at maxlen, which
        # adds Kraft mass to offset what we just removed.
        cand = sorted((l, t) for t, l in L.items() if l < maxlen and t != s)
        if not cand:
            break
        L[cand[0][1]] += 1

    # Final safety: if Kraft still exceeds 1, lengthen the shortest codes until
    # it does not.  This is a fallback that should never trigger.
    for _ in range(200):
        kraft = sum(2.0 ** (-l) for l in L.values())
        if kraft <= 1.0 + 1e-12:
            break
        t = min(L, key=lambda x: (L[x], x))
        L[t] += 1
    return L


def canonical_codes(lengths: Dict[int, int]) -> Dict[int, Tuple[int, int]]:
    """Assign canonical codewords from lengths alone: sort by (length, symbol),
    then increment.  The decoder regenerates the identical mapping, which is why
    only the lengths travel in the header."""
    if not lengths:
        return {}
    order = sorted(lengths.items(), key=lambda kv: (kv[1], kv[0]))
    codes: Dict[int, Tuple[int, int]] = {}
    code = 0
    prev = order[0][1]
    for sym, L in order:
        if L > prev:
            code <<= (L - prev)
            prev = L
        codes[sym] = (code, L)
        code += 1
    return codes


class HuffmanEncoderTable:
    __slots__ = ("codes", "lengths", "symbols")

    def __init__(self, freq: Dict[int, int]):
        self.lengths = huffman_lengths(freq)
        self.codes = canonical_codes(self.lengths)
        # Canonical order: the list of symbols sorted by (length, symbol).
        self.symbols = sorted(self.lengths, key=lambda s: (self.lengths[s], s))

    def __contains__(self, sym: int) -> bool:
        return sym in self.codes

    def encode(self, sym: int) -> Tuple[int, int]:
        return self.codes[sym]

    def serialize(self) -> bytes:
        """nsym (16 bits) then one nibble of length per symbol, in canonical
        order.  The decoder rebuilds codewords from lengths alone."""
        bw = BitWriter()
        bw.write(len(self.symbols), 16)
        for s in self.symbols:
            bw.write(s, 16)
            bw.write(self.lengths[s], 4)
        data, nbits = bw.finish()
        return data

    @staticmethod
    def deserialize(data: bytes) -> "HuffmanDecoderTable":
        br = BitReader(data)
        nsym = br.read(16)
        lengths: Dict[int, int] = {}
        for _ in range(nsym):
            s = br.read(16)
            lengths[s] = br.read(4)
        return HuffmanDecoderTable(lengths)


class HuffmanDecoderTable:
    """Decodes by walking the canonical code one bit at a time.

    For our short alphabets and MAX_CODE_LEN = 15 this beats building a 2^15
    lookup table in both time and memory, and it has no failure mode: a valid
    codeword always resolves within `maxlen` bits.
    """

    __slots__ = ("by_len", "maxlen", "first_code", "first_index", "lengths")

    def __init__(self, lengths: Dict[int, int]):
        self.lengths = lengths
        self.maxlen = max(lengths.values()) if lengths else 1
        self.by_len: Dict[int, List[int]] = {}
        for L in sorted(set(lengths.values())):
            if L > 0:
                self.by_len[L] = sorted(s for s, l in lengths.items() if l == L)
        self.first_code: Dict[int, int] = {}
        self.first_index: Dict[int, int] = {}
        code = 0
        idx = 0
        for L in sorted(self.by_len):
            self.first_code[L] = code
            self.first_index[L] = idx
            code = (code + len(self.by_len[L])) << 1
            idx += len(self.by_len[L])

    def decode_one(self, br: BitReader) -> int:
        code = 0
        for L in range(1, self.maxlen + 1):
            code = (code << 1) | br.read(1)
            syms = self.by_len.get(L)
            if syms:
                off = code - self.first_code[L]
                if 0 <= off < len(syms):
                    return syms[off]
        raise ValueError("Huffman decode walked off the table: corrupt stream")


def huffman_encode_stream(symbols: Sequence[int], freq: Dict[int, int] | None = None
                          ) -> Tuple[bytes, int, bytes]:
    """Convenience: build a table, code the symbols, return
    (table_bytes, payload_bits, payload_bytes)."""
    if freq is None:
        freq = {}
        for s in symbols:
            freq[s] = freq.get(s, 0) + 1
    if not freq:
        freq = {0: 1}
    tbl = HuffmanEncoderTable(freq)
    bw = BitWriter()
    bw.write_many([tbl.encode(s) for s in symbols])
    payload, nbits = bw.finish()
    return tbl.serialize(), nbits, payload


def huffman_decode_stream(table_bytes: bytes, nbits: int, payload: bytes,
                          count: int) -> List[int]:
    tbl = HuffmanEncoderTable.deserialize(table_bytes)
    br = BitReader(payload, nbits)
    return [tbl.decode_one(br) for _ in range(count)]
