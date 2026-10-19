"""
PHOTON-C  ::  rANS entropy engine
=================================
Range asymmetric numeral systems, byte-oriented (b = 256), 32-bit state.

Why rANS and not an adaptive binary arithmetic coder
----------------------------------------------------
I built the binary arithmetic coder first.  It is the theoretically strongest
choice for video syntax and I could not make it robust: the decoder has to hold a
32-bit window over the byte stream that stays pinned to the same offset as the
encoder's `low`, and every arrangement I tried either desynchronised or had to
emit four bytes per renormalisation to stay aligned -- which costs more than the
modelling gains.  That failure mode is also the *worst* kind: it round-trips
perfectly on easy material and breaks once contexts saturate, i.e. exactly at
the ultra-low bitrates this codec exists for.

rANS has no window at all.  It is a stack:

    encode:  x' = (x // f) * M + cum + (x % f)
    decode:  s  = x % M ;  find symbol with cum <= s < cum+f
             x' = f * (x // M) + s - cum

The encoder pushes symbols in REVERSE order and the decoder pops them in forward
order, and the only thing shared between them is the frequency table plus a
4-byte final state.  There is no bit offset that can drift, so correctness is
easy to establish and easy to test exhaustively -- which matters more here than
the last 5% of compression.

Efficiency is within a fraction of a percent of arithmetic coding as long as the
frequency table matches the source, and here it does: tables are counted from the
actual symbols of the frame (or GOP) and quantised to 12 bits, so M = 4096.

Cost model
----------
Rate for a symbol is log2(M / f) bits, so a skip flag seen 990 times in 1000
costs ~0.014 bits.  That is the number that makes 4K at 64 kbit/s arithmetically
possible: a frame with 95% skipped blocks spends almost nothing on the skip map,
and the budget goes to the few blocks that actually changed.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

PROB_BITS = 12
PROB_SCALE = 1 << PROB_BITS     # M -- frequencies sum to exactly this
L_BITS = 23
RANS_BYTE_L = 1 << L_BITS       # lower bound of the normalised state interval

# L is not a free parameter.  With byte output (b = 2^8) and M = 2^12, the
# encoder normalises while x >= f * (L >> PROB_BITS) * 256 and the decoder
# refills while x < L.  For x to stay inside 32 bits (so the final state
# serialises as four bytes) we need b*L <= 2^32, hence L <= 2^24; and L = 2^23
# is the standard choice, leaving the decoded state in [2^23, 2^31).


# --------------------------------------------------------------------------- #
#  Frequency table
# --------------------------------------------------------------------------- #
class FreqTable:
    """Quantised symbol frequencies summing to exactly PROB_SCALE.

    Quantisation has to be done carefully: rounding can push the total off
    PROB_SCALE, and rounding a rare symbol down to zero makes it unencodable.
    So: every present symbol gets at least 1, then the largest-remainder method
    distributes what is left.
    """

    __slots__ = ("freq", "cum", "symbols", "nsym",
                 "_lut_arr", "_freq_arr", "_cum_arr")

    def __init__(self, counts: Dict[int, int]):
        syms = sorted(s for s, c in counts.items() if c > 0)
        if not syms:
            syms = [0]
            counts = {0: 1}
        self.symbols = syms
        self.nsym = len(syms)

        # Every present symbol must get frequency >= 1 or it becomes
        # unencodable, and frequencies sum to exactly PROB_SCALE.  So the
        # alphabet cannot exceed PROB_SCALE symbols -- at M = 4096 that is ~4095
        # distinct values.  Exceeding it used to fail much later and much more
        # confusingly, as a KeyError deep inside rans_encode on whichever symbol
        # got dropped.  Guarding it here names the actual problem at the point
        # where it is caused: something upstream is feeding a raw large-alphabet
        # quantity (a block index, say) to a coder that needs a small one.
        if self.nsym >= PROB_SCALE:
            raise ValueError(
                f"alphabet of {self.nsym} distinct symbols cannot be quantised "
                f"into PROB_SCALE={PROB_SCALE} with every symbol at frequency "
                f">= 1; code a differential or otherwise shrink the alphabet")

        total = sum(counts[s] for s in syms)
        n = len(syms)
        if n >= PROB_SCALE:
            # More distinct symbols than the table can give one unit each.
            # Widen the table is not an option (M is fixed by the bitstream), so
            # keep the most frequent PROB_SCALE-1 symbols and let the rest share
            # a single escape slot.  Callers size their alphabets well below
            # this; the branch exists so the invariant can never be violated.
            keep = sorted(syms, key=lambda s: -counts[s])[:PROB_SCALE - 1]
            counts = {s: counts[s] for s in keep}
            counts[PROB_SCALE - 1] = 1
            syms = sorted(counts)
            total = sum(counts.values())
            n = len(syms)

        # Largest-remainder allocation with a hard floor of 1: a symbol rounded
        # to zero would be unencodable, and rANS has no way to signal that.
        raw = {s: counts[s] * PROB_SCALE / total for s in syms}
        q = {s: 1 for s in syms}
        left = PROB_SCALE - n
        if left > 0:
            floors = {s: int(raw[s]) for s in syms}
            # give everyone their integer part, capped so the floor of 1 survives
            alloc = {s: min(floors[s], left) for s in syms}
            used = sum(alloc.values())
            rem = left - used
            if rem > 0:
                order = sorted(syms, key=lambda s: (-(raw[s] - floors[s]), -raw[s]))
                i = 0
                while rem > 0:
                    s = order[i % n]
                    alloc[s] += 1
                    rem -= 1
                    i += 1
            q = {s: 1 + alloc[s] for s in syms}

        # Exact repair of the total.  Only ever needs to remove, and never below 1.
        diff = PROB_SCALE - sum(q.values())
        guard = 0
        while diff != 0 and guard < 100000:
            guard += 1
            if diff > 0:
                s = max(syms, key=lambda x: (raw[x] - q[x], x))
                q[s] += 1
                diff -= 1
            else:
                cand = [x for x in syms if q[x] > 1]
                if not cand:
                    break
                s = min(cand, key=lambda x: (q[x] - raw[x], x))
                q[s] -= 1
                diff += 1

        assert sum(q.values()) == PROB_SCALE, f"table sums to {sum(q.values())}"
        assert min(q.values()) >= 1, "a symbol was quantised to zero frequency"
        self.freq = q
        cum = 0
        c = {}
        for s in syms:
            c[s] = cum
            cum += q[s]
        self.cum = c
        assert cum == PROB_SCALE, f"frequency table sums to {cum}, expected {PROB_SCALE}"

    # ---- serialisation --------------------------------------------------- #
    def serialize(self) -> bytearray:
        """nsym (2 bytes) then per symbol: value (3 bytes) + freq (2 bytes).
        Cumulative offsets are derived, never stored."""
        out = bytearray()
        out += self.nsym.to_bytes(2, "big")
        for s in self.symbols:
            out += (s & 0xFFFFFF).to_bytes(3, "big")
            out += self.freq[s].to_bytes(2, "big")
        return out

    @staticmethod
    def from_quantised(freq: Dict[int, int]) -> "FreqTable":
        """Build a table from ALREADY-quantised frequencies.

        This must never re-run quantisation.  Re-quantising a table that already
        sums to PROB_SCALE redistributes it by largest-remainder against a
        different total, so the decoder ends up with a different table from the
        encoder's and the stream decodes to garbage -- while still looking
        plausible for the first few hundred symbols.
        """
        t = FreqTable.__new__(FreqTable)
        t.symbols = sorted(freq)
        t.nsym = len(t.symbols)
        t.freq = dict(freq)
        cum = 0
        t.cum = {}
        for s in t.symbols:
            t.cum[s] = cum
            cum += t.freq[s]
        if cum != PROB_SCALE:
            raise ValueError(f"quantised table sums to {cum}, expected {PROB_SCALE}")
        return t

    @staticmethod
    def deserialize(data: bytes, offset: int = 0) -> Tuple["FreqTable", int]:
        from .guard import StreamError
        if offset + 2 > len(data):
            raise StreamError("truncated frequency table")
        nsym = int.from_bytes(data[offset:offset + 2], "big")
        offset += 2
        if not (1 <= nsym <= PROB_SCALE):
            raise StreamError(f"frequency table claims {nsym} symbols")
        if offset + 5 * nsym > len(data):
            raise StreamError("truncated frequency table entries")
        freq: Dict[int, int] = {}
        for _ in range(nsym):
            s = int.from_bytes(data[offset:offset + 3], "big")
            f = int.from_bytes(data[offset + 3:offset + 5], "big")
            offset += 5
            if f == 0:
                raise StreamError(f"zero frequency for symbol {s}")
            freq[s] = f
        try:
            return FreqTable.from_quantised(freq), offset
        except ValueError as e:
            raise StreamError(f"corrupt frequency table: {e}") from e


def _slot_lookup(table: FreqTable) -> List[int]:
    """cum -> symbol map as a flat list of length PROB_SCALE."""
    lut: List[int] = [0] * PROB_SCALE
    for s in table.symbols:
        c, f = table.cum[s], table.freq[s]
        for i in range(c, c + f):
            lut[i] = s
    return lut


def _native_lut(table: FreqTable):
    """Cached numpy (lut, freq, cum) for the C rANS core.  freq/cum are indexed
    BY SYMBOL VALUE (the lut yields symbol values); gaps are never referenced."""
    lut = getattr(table, "_lut_arr", None)
    if lut is None:
        hi = max(max(table.symbols) + 1, 1)
        fq = np.zeros(hi, dtype=np.int32)
        cm = np.zeros(hi, dtype=np.int64)
        syms = np.asarray(table.symbols, dtype=np.int32)
        f_by_s = np.asarray([table.freq[s] for s in table.symbols], dtype=np.int32)
        c_by_s = np.asarray([table.cum[s] for s in table.symbols], dtype=np.int64)
        lut = None
        try:
            from . import native
            lut = native.rans_build_lut(syms, f_by_s, c_by_s)
        except Exception:
            lut = None
        if lut is None:
            lut = np.zeros(PROB_SCALE, dtype=np.int32)
            for s in table.symbols:
                c, f = table.cum[s], table.freq[s]
                lut[c:c + f] = s
        for s in table.symbols:
            fq[s] = table.freq[s]
            cm[s] = table.cum[s]
        table._lut_arr = lut
        table._freq_arr = fq
        table._cum_arr = cm
    return table._lut_arr, table._freq_arr, table._cum_arr


# --------------------------------------------------------------------------- #
#  Encoder
# --------------------------------------------------------------------------- #
def rans_encode(symbols: Sequence[int], table: FreqTable) -> bytes:
    """Encode `symbols` (decoded order) and return the byte stream.

    Symbols are pushed in reverse because rANS is a stack: the last one encoded
    is the first one decoded.
    """
    freq, cum = table.freq, table.cum
    x = RANS_BYTE_L
    out = bytearray()

    for s in reversed(symbols):
        f = freq[s]
        c = cum[s]
        # Normalisation bound.  x must satisfy
        #
        #     x < f * (L >> PROB_BITS) * 256  =  f << (L_BITS - PROB_BITS + 8)
        #
        # Derivation, because it is the one step that is easy to get wrong by a
        # factor of 256: after the update
        #     x' = (x // f) * M + (x % f) + cum
        # the decoder must be able to restore x from x' by reading whole bytes
        # while x_dec < L.  Since x_dec = f * (x' // M) + slot - cum = x // f,
        # one byte of refill gives 256*(x//f) + byte, and for that to reach L we
        # need x//f >= L/256 -- while for x' to stay under b*L = 2^31 we need
        # x//f < L/M * 256.  Hence the bound above, which with L = 2^23 and
        # M = 2^12 is f << 9 = f * 512.
        #
        # Getting this wrong by the factor of 256 (writing f << 11 or f << 20)
        # does not crash anything.  It produces streams that are far too small,
        # decode correctly for a while, and then derail -- the worst possible
        # failure mode for a codec.
        x_max = f << (L_BITS - PROB_BITS + 8)
        while x >= x_max:
            out.append(x & 0xFF)
            x >>= 8
        x = ((x // f) << PROB_BITS) + (x % f) + c

    # Stream layout:  [ 4-byte final state ][ payload, decoder order ]
    #
    # Renormalisation bytes came out low-byte-first in reverse symbol order, so
    # the payload buffer is backwards and has to be flipped.  The final state
    # goes in FRONT of it, because that is where the decoder starts reading.
    #
    # Both of those details are load-bearing and both fail silently: appending
    # the state after the flip puts it at the wrong end (the decoder then treats
    # payload bytes as the state and derails once real renormalisation starts),
    # and appending it before the flip reverses the state's own four bytes.
    out.reverse()
    return x.to_bytes(4, "big") + bytes(out)


# --------------------------------------------------------------------------- #
#  Decoder
# --------------------------------------------------------------------------- #
def rans_decode(data: bytes, table: FreqTable, count: int) -> List[int]:
    """Decode exactly `count` symbols from a stream produced by rans_encode."""
    if count > 64:
        try:
            from . import native
            lut_a, fq_a, cm_a = _native_lut(table)
            out = native.rans_decode_native(data, lut_a, fq_a, cm_a, count)
            if out is not None:
                return out.tolist()   # tolist() keeps python ints, drops np-int boxing
        except Exception:
            pass
    lut = _slot_lookup(table)
    freq, cum = table.freq, table.cum

    x = int.from_bytes(data[0:4], "big")
    pos = 4
    n = len(data)
    out: List[int] = []

    for _ in range(count):
        slot = x & (PROB_SCALE - 1)
        s = lut[slot]
        f = freq[s]
        c = cum[s]
        x = f * (x >> PROB_BITS) + slot - c
        while x < RANS_BYTE_L and pos < n:
            x = (x << 8) | data[pos]
            pos += 1
        out.append(s)

    return out


# --------------------------------------------------------------------------- #
#  Convenience: count + encode + self-describing stream
# --------------------------------------------------------------------------- #
def pack(symbols: Sequence[int]) -> bytes:
    """Self-contained stream: table, symbol count, payload."""
    counts: Dict[int, int] = {}
    for s in symbols:
        counts[s] = counts.get(s, 0) + 1
    if not counts:
        counts = {0: 1}
    table = FreqTable(counts)
    payload = rans_encode(symbols, table)
    head = bytearray()
    head += len(symbols).to_bytes(4, "big")
    head += table.serialize()
    head += len(payload).to_bytes(4, "big")
    return bytes(head) + payload


def unpack(data: bytes) -> Tuple[List[int], int]:
    """Returns (symbols, bytes_consumed)."""
    from .guard import StreamError
    if len(data) < 4:
        raise StreamError("rANS stream: no symbol count")
    count = int.from_bytes(data[0:4], "big")
    from .guard import MAX_SYMS
    if count > MAX_SYMS:
        raise StreamError(f"rANS stream claims {count} symbols (cap {MAX_SYMS})")
    table, off = FreqTable.deserialize(data, 4)
    if off + 4 > len(data):
        raise StreamError("rANS stream: no payload length")
    plen = int.from_bytes(data[off:off + 4], "big")
    off += 4
    if plen > len(data) - off:
        raise StreamError(f"rANS payload {plen} B exceeds remaining "
                          f"{len(data) - off} B")
    try:
        syms = rans_decode(data[off:off + plen], table, count)
    except (KeyError, IndexError, OverflowError) as e:
        raise StreamError(f"rANS decode failed: {e}") from e
    return syms, off + plen


def ideal_bits(symbols: Sequence[int], table: FreqTable) -> float:
    """Shannon cost of `symbols` under `table` -- the efficiency yardstick."""
    import math
    return sum(math.log2(PROB_SCALE / table.freq[s]) for s in symbols)
