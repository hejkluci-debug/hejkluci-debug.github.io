"""QArt engine: a STANDARD QR code whose hidden modules agree with the artwork
lying on top of it (technique of Russ Cox, "QArt Codes", 2012).

Idea
----
The URL is followed by "#" and a long run of digits (numeric segment). The
digits are free: browsers never send a URL fragment to the server and our page
ignores it. Every free bit changes some modules of the symbol; because QR
encoding (Reed-Solomon over GF(256), interleaving, masking) is linear over
GF(2), each module is  base XOR (sum of free bits that touch it).
We pick the free bits by Gaussian elimination so that the modules UNDER the
cat get exactly the colour the decoder will see there (white porcelain ->
light module, blue paint -> dark module). The decoder therefore reads a
nearly perfect code even where the cat covers it, and error correction stays
free for camera noise. Everything stays inside ISO/IEC 18004: segments,
terminator and pad codewords are normal; only the digits are chosen.

To keep every 3-digit group a valid numeric value (< 1000) the top bit of
each 10-bit group is fixed to 0, so 9 of 10 bits are free.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
from qrcodegen import QrCode, QrSegment

ECC = {"L": QrCode.Ecc.LOW, "M": QrCode.Ecc.MEDIUM, "Q": QrCode.Ecc.QUARTILE, "H": QrCode.Ecc.HIGH}
CACHE = Path(__file__).parent / ".qart_cache"; CACHE.mkdir(exist_ok=True)


def _append(bits, val, n):
    for i in range(n - 1, -1, -1):
        bits.append((val >> i) & 1)


def _url_segments(url):
    """Alphanumeric where possible. A lower-case scheme ("https://") is kept as a
    short byte segment because Android matches intent schemes case-sensitively:
    "HTTPS://..." opens "no app found" on some Android scanners."""
    if QrSegment.is_alphanumeric(url):
        return [QrSegment.make_alphanumeric(url)]
    for pre in ("https://", "http://"):
        if url.startswith(pre) and QrSegment.is_alphanumeric(url[len(pre):]):
            return [QrSegment.make_bytes(pre.encode()), QrSegment.make_alphanumeric(url[len(pre):])]
    return [QrSegment.make_bytes(url.encode())]


class QArtEncoder:
    def __init__(self, url: str, version: int, ecl: str):
        self.url, self.version, self.ecl = url, version, ecl
        self.ec = ECC[ecl]
        self.size = version * 4 + 17
        cap = QrCode._get_num_data_codewords(version, self.ec) * 8
        head = _url_segments(url) + [QrSegment.make_bytes(b"#")]
        used = sum(4 + s.get_mode().num_char_count_bits(version) + len(s._bitdata) for s in head)
        ncc = QrSegment.Mode.NUMERIC.num_char_count_bits(version)
        room = cap - used - 4 - ncc
        groups = room // 10
        if groups < 1:
            raise ValueError("URL does not fit")
        self.ndigits = groups * 3
        bits: list[int] = []
        for s in head:
            _append(bits, s.get_mode().get_mode_bits(), 4)
            _append(bits, s.get_num_chars(), s.get_mode().num_char_count_bits(version))
            bits += s._bitdata
        _append(bits, QrSegment.Mode.NUMERIC.get_mode_bits(), 4)
        _append(bits, self.ndigits, ncc)
        start = len(bits)
        bits += [0] * (groups * 10)
        self.free = [start + g * 10 + k for g in range(groups) for k in range(0, 10)]
        self.top = {start + g * 10 for g in range(groups)}          # b9: 512
        self.second = {start + g * 10 + 1: start + g * 10 for g in range(groups)}  # b8 -> its b9
        bits += [0] * min(4, cap - len(bits))          # terminator
        bits += [0] * (-len(bits) % 8)
        pad = 0xEC
        while len(bits) < cap:
            _append(bits, pad, 8); pad ^= 0xEC ^ 0x11
        self.base_bits = np.array(bits, np.uint8)
        self.nfree = len(self.free)
        self._deltas()
        self._function_mask()

    # data bits -> codeword bytes
    @staticmethod
    def _bytes(bits):
        return bytes(np.packbits(bits).tolist())

    def _matrix(self, bits, mask):
        q = QrCode(self.version, self.ec, self._bytes(bits), mask)
        n = self.size
        return np.array([[q.get_module(x, y) for x in range(n)] for y in range(n)], np.uint8)

    def _deltas(self):
        key = hashlib.sha1(f"{self.url}|{self.version}|{self.ecl}|v2".encode()).hexdigest()[:16]
        f = CACHE / f"{key}.npz"
        if f.exists():
            d = np.load(f); self.delta = d["delta"]; self.base = d["base"]; return
        base = self._matrix(self.base_bits, 0)
        delta = np.zeros((self.nfree, self.size, self.size), np.uint8)
        for i, b in enumerate(self.free):
            bits = self.base_bits.copy(); bits[b] ^= 1
            delta[i] = self._matrix(bits, 0) ^ base
        self.delta, self.base = delta, base
        np.savez_compressed(f, delta=delta, base=base)

    def _function_mask(self):
        q = QrCode.__new__(QrCode)
        q._version, q._size, q._errcorlvl = self.version, self.size, self.ec
        q._modules = [[False] * self.size for _ in range(self.size)]
        q._isfunction = [[False] * self.size for _ in range(self.size)]
        q._draw_function_patterns()
        self.isfunc = np.array(q._isfunction, bool)

    def base_for_mask(self, mask):
        return self._matrix(self.base_bits, mask)

    def own_modules(self):
        """(y, x) of the data module that each free bit is stored in."""
        if hasattr(self, "_own"):
            return self._own
        cw, _, is_data, _, _, _ = codeword_map(self)
        datamod = np.zeros_like(self.isfunc)
        valid = cw >= 0
        datamod[valid] = is_data[cw[valid]]
        own = []
        for i in range(self.nfree):
            ys, xs = np.where((self.delta[i] > 0) & datamod)
            assert len(ys) == 1, (i, len(ys))
            own.append((int(ys[0]), int(xs[0])))
        self._own = own
        return own

    def solve(self, mask: int, target: np.ndarray, priority: np.ndarray):
        """target: n x n of -1 (free) / 0 light / 1 dark; priority: higher first.
        Keeps every 3-digit group < 1000 by forbidding (b9, b8) = (1, 1).
        Returns (data_bits, matrix, report)."""
        base = self.base_for_mask(mask)
        n = self.size
        own = self.own_modules()
        idx = {pos: i for i, pos in enumerate(self.free)}
        # --- numeric validity: decide every group's top bit up front ---
        fixed = []                      # (priority, single-bit row, rhs)
        BIG = 1e9
        for top in sorted(self.top):
            i9, i8 = idx[top], idx[top + 1]
            (y9, x9), (y8, x8) = own[i9], own[i8]
            want9 = target[y9, x9] ^ base[y9, x9] if target[y9, x9] >= 0 else 0
            want8 = target[y8, x8] ^ base[y8, x8] if target[y8, x8] >= 0 else None
            if want9 == 1 and want8 == 1 and priority[y8, x8] > priority[y9, x9]:
                want9 = 0
            fixed.append((BIG, 1 << i9, int(want9)))
            if want9 == 1:
                fixed.append((BIG, 1 << i8, 0))
        cand = [(priority[y, x], y, x) for y in range(n) for x in range(n)
                if target[y, x] >= 0 and not self.isfunc[y, x]]
        cand.sort(reverse=True)
        packed = np.packbits(self.delta.reshape(self.nfree, -1).T, axis=1, bitorder="little")
        rows = [(r, b) for _, r, b in fixed]
        rows += [(int.from_bytes(packed[y * n + x].tobytes(), "little"), int(target[y, x]) ^ int(base[y, x]))
                 for _, y, x in cand]
        basis: dict[int, tuple[int, int]] = {}
        forced = 0
        for row, rhs in rows:
            while row:
                p = row.bit_length() - 1
                if p in basis:
                    br, bb = basis[p]; row ^= br; rhs ^= bb
                else:
                    basis[p] = (row, rhs); forced += 1; break
        xsol = 0
        for p in sorted(basis):
            r, b = basis[p]
            val = b ^ (bin((r & ~(1 << p)) & xsol).count("1") & 1)
            if val:
                xsol |= 1 << p
        bits = self.base_bits.copy()
        for i, pos in enumerate(self.free):
            if (xsol >> i) & 1:
                bits[pos] ^= 1
        # sanity: every group is a valid 3-digit number
        st = self.free[0]
        for g in range(self.ndigits // 3):
            v = int("".join(str(int(b)) for b in bits[st + 10 * g: st + 10 * g + 10]), 2)
            assert v < 1000, v
        mat = self._matrix(bits, mask)
        want = (target >= 0) & ~self.isfunc
        wrong = want & (mat != target)
        report = {"constraints": int(want.sum()), "independent": forced, "free_bits": self.nfree,
                  "mismatch": int(wrong.sum()), "mismatch_frac": float(wrong.sum() / max(1, want.sum()))}
        return bits, mat, report

    def digits(self, bits) -> str:
        """The numeric fragment actually encoded (for the URL shown on phones)."""
        start = self.free[0]
        out = []
        for g in range(self.ndigits // 3):
            v = 0
            for k in range(10):
                v = (v << 1) | int(bits[start + g * 10 + k])
            out.append(f"{v:03d}")
        return "".join(out)


# ---------------------------------------------------------------------------
# codeword / block geometry, used to plan *where* unavoidable errors go
# ---------------------------------------------------------------------------
def codeword_map(enc: "QArtEncoder"):
    """Returns (cw_of_module n x n, -1 for function modules) as index in the
    interleaved codeword sequence, plus arrays block_of_cw, is_data_cw and
    the per-block (data_len, ec_len)."""
    v, ec, n = enc.version, enc.ec, enc.size
    nblocks = QrCode._NUM_ERROR_CORRECTION_BLOCKS[ec.ordinal][v]
    eclen = QrCode._ECC_CODEWORDS_PER_BLOCK[ec.ordinal][v]
    raw = QrCode._get_num_raw_data_modules(v) // 8
    nshort = nblocks - raw % nblocks
    shortlen = raw // nblocks
    dlens = [shortlen - eclen + (0 if i < nshort else 1) for i in range(nblocks)]
    block_of, is_data, data_index = [], [], []
    for i in range(max(dlens)):
        for b in range(nblocks):
            if i < dlens[b]:
                block_of.append(b); is_data.append(True)
                data_index.append(sum(dlens[:b]) + i)       # position in the data stream
    for i in range(eclen):
        for b in range(nblocks):
            block_of.append(b); is_data.append(False); data_index.append(-1)
    cw = np.full((n, n), -1, np.int32)
    bit = 0
    right = n - 1
    while right >= 1:
        if right == 6:
            right = 5
        for vert in range(n):
            for j in range(2):
                x = right - j
                upward = ((right + 1) & 2) == 0
                y = (n - 1 - vert) if upward else vert
                if not enc.isfunc[y, x] and bit < raw * 8:
                    cw[y, x] = bit // 8
                    bit += 1
        right -= 2
    return cw, np.array(block_of), np.array(is_data), np.array(data_index), dlens, eclen


def solve_planned(enc: "QArtEncoder", mask: int, target, priority, ec_use=0.6):
    """Like enc.solve, but first decides per RS block which codewords to give
    up (at most ec_use * floor(ec_len/2) per block) so that the remaining
    constraints fit into that block's free bits. Errors then land only in the
    given-up codewords, leaving the rest of the error correction for the camera."""
    cw, block_of, is_data, data_index, dlens, eclen = codeword_map(enc)
    nb = len(dlens)
    t = eclen // 2
    budget = int(np.floor(ec_use * t))
    # free bits per block
    free_cw = np.array(enc.free) // 8                 # data-stream codeword of each free bit
    starts = np.cumsum([0] + dlens[:-1])
    free_per_block = [int(((free_cw >= starts[b]) & (free_cw < starts[b] + dlens[b])).sum()) for b in range(nb)]
    tgt = target.copy()
    want = (tgt >= 0) & ~enc.isfunc & (cw >= 0)
    given_up = []
    for b in range(nb):
        cws = np.where(block_of == b)[0]
        counts = {int(c): int((want & (cw == c)).sum()) for c in cws}
        need = sum(counts.values())
        cap = int(free_per_block[b] * 0.98)
        for c, k in sorted(counts.items(), key=lambda kv: -kv[1]):
            if need <= cap or len([g for g in given_up if block_of[g] == b]) >= budget:
                break
            tgt[cw == c] = -1; need -= k; given_up.append(c)
    bits, mat, rep = enc.solve(mask, tgt, priority)
    wrong_cw = set()
    errs = (target >= 0) & ~enc.isfunc & (cw >= 0) & (mat != target)
    for y, x in zip(*np.where(errs)):
        wrong_cw.add(int(cw[y, x]))
    per_block = [sum(1 for c in wrong_cw if block_of[c] == b) for b in range(nb)]
    rep.update({"blocks": nb, "t_per_block": t, "given_up": len(given_up), "wrong_codewords_per_block": per_block,
                "worst_block_use": max(per_block) / max(1, t), "free_per_block": free_per_block})
    return bits, mat, rep
