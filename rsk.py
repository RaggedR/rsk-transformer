"""
RSK (Robinson-Schensted-Knuth) correspondence: forward and inverse.

For permutations: σ ∈ S_n → (P, Q) where both P, Q are standard Young tableaux (SYT).
For words: w ∈ {1,...,k}^m → (P, Q) where P is a semistandard Young tableau (SSYT)
and Q is a standard Young tableau (SYT) with entries {1,...,m}.
For matrices: A ∈ ℕ^{a×b} → (P, Q) where both P, Q are SSYT (Knuth's full RSK).

Forward: σ/w → (P, Q) via Schensted row insertion (bumping algorithm).
Inverse: (P, Q) → σ/w via reverse bumping, processing Q entries in decreasing order.

Tableaux are represented as list-of-lists: tableau[i] is row i (0-indexed),
each row is a list of integers in increasing order (strict for SYT, weak for SSYT rows).
"""

from __future__ import annotations

import copy
from itertools import permutations
from math import factorial
from typing import Iterator


# Type alias: a tableau is a list of rows, each row a list of ints
Tableau = list[list[int]]


def schensted_insert(tableau: Tableau, value: int) -> tuple[Tableau, tuple[int, int]]:
    """
    Insert `value` into `tableau` using Schensted row insertion (bumping).

    Returns the modified tableau and the (row, col) position where the new cell
    was added — this is needed to build Q alongside P.

    The algorithm: scan row 0 for the leftmost entry > value. If found, bump it
    and recursively insert the bumped value into the next row. If no entry is
    larger (value is larger than everything in the row), append to the end.
    """
    tableau = copy.deepcopy(tableau)
    row_idx = 0

    while row_idx < len(tableau):
        row = tableau[row_idx]
        # Binary search for leftmost entry strictly greater than value
        lo, hi = 0, len(row)
        while lo < hi:
            mid = (lo + hi) // 2
            if row[mid] <= value:
                lo = mid + 1
            else:
                hi = mid

        if lo < len(row):
            # Bump: replace row[lo] with value, continue inserting bumped value
            bumped = row[lo]
            row[lo] = value
            value = bumped
            row_idx += 1
        else:
            # No bumping needed: append to end of this row
            row.append(value)
            return tableau, (row_idx, len(row) - 1)

    # Value was bumped out of all existing rows — start a new row
    tableau.append([value])
    return tableau, (row_idx, 0)


def rsk_forward(sigma: list[int]) -> tuple[Tableau, Tableau]:
    """
    Compute the RSK correspondence: σ → (P, Q).

    σ is a permutation given as a list [σ(1), σ(2), ..., σ(n)] using values 1..n.
    Returns insertion tableau P and recording tableau Q.
    """
    P: Tableau = []
    Q: Tableau = []

    for i, val in enumerate(sigma, start=1):
        P, (row, col) = schensted_insert(P, val)
        # Record where the new cell appeared, labelled by position i
        while len(Q) <= row:
            Q.append([])
        Q[row].append(i)

    return P, Q


def reverse_bump(tableau: Tableau, row_idx: int, col_idx: int) -> tuple[Tableau, int]:
    """
    Reverse bumping: remove the cell at (row_idx, col_idx) and reverse the
    insertion path back to row 0.

    Returns the modified tableau and the value that was originally inserted
    into row 0 (i.e., the σ(i) value).
    """
    tableau = copy.deepcopy(tableau)

    # Remove the cell at the given position
    value = tableau[row_idx].pop(col_idx)
    if not tableau[row_idx]:
        tableau.pop(row_idx)

    # Reverse bump upward through the rows
    current_row = row_idx - 1
    while current_row >= 0:
        row = tableau[current_row]
        # Find the rightmost entry strictly less than value
        # (this is the entry that bumped `value` during forward insertion)
        pos = len(row) - 1
        while pos >= 0 and row[pos] >= value:
            pos -= 1

        if pos < 0:
            raise ValueError(f"Reverse bump failed at row {current_row}")

        bumped_back = row[pos]
        row[pos] = value
        value = bumped_back
        current_row -= 1

    return tableau, value


def rsk_inverse(P: Tableau, Q: Tableau) -> list[int]:
    """
    Inverse RSK: (P, Q) → σ.

    Process recording tableau Q entries in decreasing order. For each entry i
    in Q, find its position (row, col), remove it, then reverse-bump P at that
    position to recover σ(i).
    """
    P = copy.deepcopy(P)
    Q = copy.deepcopy(Q)

    # Build a map from Q-entry to (row, col)
    n = sum(len(row) for row in Q)
    sigma = [0] * n

    # Process in decreasing order: n, n-1, ..., 1
    for i in range(n, 0, -1):
        # Find position of i in Q
        found = False
        for row_idx, row in enumerate(Q):
            for col_idx, val in enumerate(row):
                if val == i:
                    # Remove from Q
                    Q[row_idx].pop(col_idx)
                    if not Q[row_idx]:
                        Q.pop(row_idx)
                    # Reverse bump from P at this position
                    P, original_val = reverse_bump(P, row_idx, col_idx)
                    sigma[i - 1] = original_val
                    found = True
                    break
            if found:
                break

        if not found:
            raise ValueError(f"Entry {i} not found in Q")

    return sigma


def tableau_shape(T: Tableau) -> list[int]:
    """Return the shape (partition) of a tableau as a list of row lengths."""
    return [len(row) for row in T]


def is_standard_young_tableau(T: Tableau) -> bool:
    """Check if T is a valid standard Young tableau (SYT)."""
    if not T:
        return True

    shape = tableau_shape(T)

    # Shape must be a partition (weakly decreasing row lengths)
    for i in range(len(shape) - 1):
        if shape[i] < shape[i + 1]:
            return False

    # Rows must be strictly increasing
    for row in T:
        for i in range(len(row) - 1):
            if row[i] >= row[i + 1]:
                return False

    # Columns must be strictly increasing
    for col_idx in range(shape[0]):
        for row_idx in range(len(T) - 1):
            if col_idx < len(T[row_idx]) and col_idx < len(T[row_idx + 1]):
                if T[row_idx][col_idx] >= T[row_idx + 1][col_idx]:
                    return False

    return True


def is_semistandard_young_tableau(T: Tableau) -> bool:
    """Check if T is a valid semistandard Young tableau (SSYT).

    Rows are weakly increasing (≤), columns are strictly increasing (<).
    """
    if not T:
        return True

    shape = tableau_shape(T)

    # Shape must be a partition (weakly decreasing row lengths)
    for i in range(len(shape) - 1):
        if shape[i] < shape[i + 1]:
            return False

    # Rows must be weakly increasing
    for row in T:
        for i in range(len(row) - 1):
            if row[i] > row[i + 1]:
                return False

    # Columns must be strictly increasing
    for col_idx in range(shape[0]):
        for row_idx in range(len(T) - 1):
            if col_idx < len(T[row_idx]) and col_idx < len(T[row_idx + 1]):
                if T[row_idx][col_idx] >= T[row_idx + 1][col_idx]:
                    return False

    return True


def matrix_to_biword(A: list[list[int]]) -> tuple[list[int], list[int]]:
    """
    Convert a non-negative integer matrix A ∈ ℕ^{a×b} to a sorted two-line array (biword).

    For each (i,j) with A[i][j] > 0, create A[i][j] copies of the pair (i+1, j+1).
    Sort by (top ascending, then bottom ascending).

    Returns:
        (top_line, bottom_line): both list[int] of length |λ| = Σ A[i][j].
        top_line has values in {1,...,a}, bottom_line in {1,...,b}.
    """
    top_line = []
    bottom_line = []
    for i, row in enumerate(A):
        for j, count in enumerate(row):
            for _ in range(count):
                top_line.append(i + 1)
                bottom_line.append(j + 1)
    # Already sorted by construction: i increases, within each i, j increases
    return top_line, bottom_line


def rsk_forward_biword(
    top_line: list[int], bottom_line: list[int]
) -> tuple[Tableau, Tableau]:
    """
    Forward RSK from a two-line array (biword).

    Insert bottom_line values into P via Schensted insertion.
    Record top_line values in Q at the new cell positions.

    Returns (P, Q) where both are SSYT:
      P is SSYT over {1,...,max(bottom_line)}
      Q is SSYT over {1,...,max(top_line)}
    """
    P: Tableau = []
    Q: Tableau = []

    for top_val, bot_val in zip(top_line, bottom_line):
        P, (row, col) = schensted_insert(P, bot_val)
        while len(Q) <= row:
            Q.append([])
        Q[row].append(top_val)

    return P, Q


def rsk_inverse_biword(P: Tableau, Q: Tableau) -> tuple[list[int], list[int]]:
    """
    Inverse RSK for SSYT pair: (P, Q) → (top_line, bottom_line).

    Process Q entries from max value to min. Within each value group (which
    forms a horizontal strip), process cells in decreasing column order
    (rightmost first). For each cell, remove from Q and reverse-bump from P.

    Returns:
        (top_line, bottom_line): the two-line array, sorted by
        (top ascending, bottom ascending).
    """
    P = copy.deepcopy(P)
    Q = copy.deepcopy(Q)

    pairs = []  # collect (top_val, bottom_val) pairs

    # Find max value in Q
    max_val = max(val for row in Q for val in row)

    for v in range(max_val, 0, -1):
        # Collect all cells in Q with value v, as (row, col) pairs
        cells = []
        for row_idx, row in enumerate(Q):
            for col_idx, val in enumerate(row):
                if val == v:
                    cells.append((row_idx, col_idx))

        # Process in decreasing column order (rightmost first)
        # For cells in different rows, column order is the tiebreaker
        cells.sort(key=lambda rc: rc[1], reverse=True)

        for row_idx, col_idx in cells:
            # Remove from Q
            Q[row_idx].pop(col_idx)
            if not Q[row_idx]:
                Q.pop(row_idx)

            # Reverse bump from P
            P, bot_val = reverse_bump(P, row_idx, col_idx)
            pairs.append((v, bot_val))

    # Reverse to get the original order (we processed max→min)
    pairs.reverse()
    top_line = [p[0] for p in pairs]
    bottom_line = [p[1] for p in pairs]
    return top_line, bottom_line


def biword_to_matrix(
    top_line: list[int], bottom_line: list[int], a: int, b: int
) -> list[list[int]]:
    """
    Reconstruct matrix A ∈ ℕ^{a×b} from a two-line array.

    Counts occurrences of each (i,j) pair.
    """
    A = [[0] * b for _ in range(a)]
    for top, bot in zip(top_line, bottom_line):
        A[top - 1][bot - 1] += 1
    return A


def verify_matrix_bijection(
    a: int, b: int, total_n: int, num_samples: int = 1000, verbose: bool = False,
) -> bool:
    """
    Verify RSK round-trip on random non-negative integer matrices.

    For each sample:
    1. Generate random matrix A ∈ ℕ^{a×b} with entry sum = total_n
    2. matrix_to_biword → (top, bottom)
    3. rsk_forward_biword → (P, Q)
    4. Check P and Q are SSYT with matching shapes
    5. rsk_inverse_biword → (top', bottom')
    6. Check (top', bottom') == (top, bottom)
    """
    import random

    errors = 0
    rng = random.Random(42)

    for _ in range(num_samples):
        # Random matrix: place total_n balls into a*b bins
        flat = [0] * (a * b)
        for _ in range(total_n):
            flat[rng.randint(0, a * b - 1)] += 1
        A = [flat[i * b:(i + 1) * b] for i in range(a)]

        top, bottom = matrix_to_biword(A)
        if len(top) != total_n:
            if verbose:
                print(f"FAIL: biword length {len(top)} != total_n {total_n}")
            errors += 1
            continue

        P, Q = rsk_forward_biword(top, bottom)

        if not is_semistandard_young_tableau(P):
            if verbose:
                print(f"FAIL: P not SSYT for A={A}")
            errors += 1
            continue

        if not is_semistandard_young_tableau(Q):
            if verbose:
                print(f"FAIL: Q not SSYT for A={A}")
            errors += 1
            continue

        if tableau_shape(P) != tableau_shape(Q):
            if verbose:
                print(f"FAIL: shape mismatch for A={A}")
            errors += 1
            continue

        top_back, bottom_back = rsk_inverse_biword(P, Q)
        if top_back != top or bottom_back != bottom:
            if verbose:
                print(f"FAIL: round-trip for A={A}")
                print(f"  top={top} vs {top_back}")
                print(f"  bot={bottom} vs {bottom_back}")
            errors += 1

    success = errors == 0
    if verbose or not success:
        print(f"  a={a}, b={b}, N={total_n}: {num_samples - errors}/{num_samples} passed, {errors} errors")
    return success


def verify_word_bijection(
    m: int, k: int, num_samples: int = 1000, verbose: bool = False
) -> bool:
    """
    Verify RSK round-trip on random words w ∈ {1,...,k}^m.

    For each sampled word:
    1. Forward: w → (P, Q)
    2. Check P is SSYT, Q is SYT, shapes match
    3. Inverse: (P, Q) → w'
    4. Check w' == w
    """
    import random

    errors = 0
    rng = random.Random(42)

    for _ in range(num_samples):
        word = [rng.randint(1, k) for _ in range(m)]
        P, Q = rsk_forward(word)

        if not is_semistandard_young_tableau(P):
            if verbose:
                print(f"FAIL: P not SSYT for w={word}, P={P}")
            errors += 1
            continue

        if not is_standard_young_tableau(Q):
            if verbose:
                print(f"FAIL: Q not SYT for w={word}, Q={Q}")
            errors += 1
            continue

        if tableau_shape(P) != tableau_shape(Q):
            if verbose:
                print(f"FAIL: shape mismatch for w={word}")
            errors += 1
            continue

        word_recovered = rsk_inverse(P, Q)
        if word_recovered != word:
            if verbose:
                print(f"FAIL: round-trip for w={word}: got {word_recovered}")
            errors += 1

    success = errors == 0
    if verbose or not success:
        print(f"  m={m}, k={k}: {num_samples - errors}/{num_samples} passed, {errors} errors")
    return success


def verify_bijection(n: int, verbose: bool = False) -> bool:
    """
    Verify RSK is a bijection by checking round-trip on all σ ∈ S_n.

    For each permutation:
    1. Forward: σ → (P, Q)
    2. Check P and Q are valid SYTs of the same shape
    3. Inverse: (P, Q) → σ'
    4. Check σ' == σ
    """
    count = 0
    errors = 0

    for perm in permutations(range(1, n + 1)):
        sigma = list(perm)
        P, Q = rsk_forward(sigma)

        # Verify tableaux properties
        if not is_standard_young_tableau(P):
            if verbose:
                print(f"FAIL: P not valid SYT for σ={sigma}, P={P}")
            errors += 1
            continue

        if not is_standard_young_tableau(Q):
            if verbose:
                print(f"FAIL: Q not valid SYT for σ={sigma}, Q={Q}")
            errors += 1
            continue

        if tableau_shape(P) != tableau_shape(Q):
            if verbose:
                print(f"FAIL: shape mismatch for σ={sigma}")
            errors += 1
            continue

        # Verify round-trip
        sigma_recovered = rsk_inverse(P, Q)
        if sigma_recovered != sigma:
            if verbose:
                print(f"FAIL: round-trip for σ={sigma}: got {sigma_recovered}")
            errors += 1
            continue

        count += 1

    expected = factorial(n)
    success = count == expected and errors == 0

    if verbose or not success:
        print(f"n={n}: {count}/{expected} passed, {errors} errors")

    return success


def generate_dataset(n: int) -> Iterator[tuple[list[int], Tableau, Tableau]]:
    """
    Generate all (σ, P, Q) triples for S_n.

    Yields tuples of (sigma, P, Q) where sigma uses values 1..n.
    """
    for perm in permutations(range(1, n + 1)):
        sigma = list(perm)
        P, Q = rsk_forward(sigma)
        yield sigma, P, Q


def tableau_positions(T: Tableau) -> list[tuple[int, int, int]]:
    """
    Extract (value, row, col) triples from a tableau.
    Rows and columns are 0-indexed.
    """
    positions = []
    for row_idx, row in enumerate(T):
        for col_idx, val in enumerate(row):
            positions.append((val, row_idx, col_idx))
    return positions


# ═══════════════════════════════════════════════════════════════════════════════
# Hillman-Grassl correspondence: RPP ↔ arbitrary filling
# ═══════════════════════════════════════════════════════════════════════════════

# Type alias: a filling is a list of rows, each row a list of non-negative ints
Filling = list[list[int]]


def partition_conjugate(lam: list[int]) -> list[int]:
    """Transpose a partition: row lengths → column lengths (and vice versa)."""
    if not lam:
        return []
    return [sum(1 for part in lam if part > j) for j in range(lam[0])]


def hook_length(shape: list[int], r: int, c: int) -> int:
    """Hook length at cell (r, c) in partition shape: arm + leg + 1."""
    conj = partition_conjugate(shape)
    arm = shape[r] - c - 1
    leg = conj[c] - r - 1
    return arm + leg + 1


def hook_lengths(shape: list[int]) -> list[list[int]]:
    """All hook lengths for a partition shape."""
    conj = partition_conjugate(shape)
    return [
        [shape[r] - c - 1 + conj[c] - r - 1 + 1 for c in range(shape[r])]
        for r in range(len(shape))
    ]


def is_rpp(T: Filling) -> bool:
    """Check if T is a valid reverse plane partition (weakly increasing rows and columns)."""
    if not T:
        return True
    shape = [len(row) for row in T]
    for i in range(len(shape) - 1):
        if shape[i] < shape[i + 1]:
            return False
    for row in T:
        for i in range(len(row) - 1):
            if row[i] > row[i + 1]:
                return False
    for c in range(shape[0]):
        for r in range(len(T) - 1):
            if c < shape[r] and c < shape[r + 1]:
                if T[r][c] > T[r + 1][c]:
                    return False
    return True


def _transpose_filling(filling: Filling, shape: list[int]) -> list[list[int]]:
    """Transpose a filling (rows → columns). Column j has entries from all rows containing column j."""
    if not shape:
        return []
    cols = []
    for j in range(shape[0]):
        col = []
        for r in range(len(filling)):
            if j < len(filling[r]):
                col.append(filling[r][j])
            else:
                break
        cols.append(col)
    return cols


def hillman_grassl_forward(shape: list[int], filling: Filling) -> Filling:
    """
    Hillman-Grassl forward: filling (λ-array) → RPP.

    Maps an arbitrary non-negative integer filling of shape λ to a reverse
    plane partition (weakly increasing rows and columns) of the same shape.
    Weight preservation: Σ RPP[r][c] = Σ filling[r][c] × hook_length(r, c).

    Algorithm (Gansner/SageMath): extract hook multiplicities from the filling,
    then for each (r, s), trace a path from (r, λ[r]-1) leftward to column s,
    moving south when the cell below has the same old value, incrementing each
    cell along the path.
    """
    lam = list(shape)
    num_rows = len(lam)

    # Transpose filling to work column-wise
    Mt = _transpose_filling(filling, lam)

    # Extract hook multiplicities: for each column j, collect (r, j) pairs
    # for each unit of filling at (r, j), processed bottom-to-top within column
    hook_mults: list[tuple[int, int]] = []
    for j, col_j in enumerate(Mt):
        col_j_hook_mults: list[tuple[int, int]] = []
        for r, entry in enumerate(col_j):
            if entry != 0:
                col_j_hook_mults += [(r, j)] * entry
        hook_mults += list(reversed(col_j_hook_mults))

    # Initialize zero RPP
    res = [[0] * rowlen for rowlen in lam]

    # Process each hook multiplicity in reverse order
    for r, s in reversed(hook_mults):
        i = r
        j = lam[r] - 1  # start at rightmost cell in row r
        while True:
            old = res[i][j]
            res[i][j] += 1
            # Try to move south: cell below has same old value
            if i + 1 < num_rows and j < lam[i + 1] and old == res[i + 1][j]:
                i += 1
            else:
                if j == s:
                    break
                j -= 1

    return res


def hillman_grassl_inverse(shape: list[int], rpp: Filling) -> Filling:
    """
    Hillman-Grassl inverse: RPP → filling (λ-array).

    Maps a reverse plane partition to an arbitrary non-negative integer filling
    of the same shape. Inverse of hillman_grassl_forward.

    Algorithm: iteratively find leftmost nonzero column, trace gamma path
    (north when cell above has same value, east otherwise), decrement path cells,
    record filling entry at (final_row, start_column).
    """
    lam = list(shape)
    res = [[0] * rowlen for rowlen in lam]

    # Deep copy and transpose the RPP to work column-wise
    Mt = _transpose_filling([row[:] for row in rpp], lam)

    while True:
        # Find leftmost nonzero column
        s = None
        for j_scan, col_scan in enumerate(Mt):
            if any(entry != 0 for entry in col_scan):
                s = j_scan
                break
        if s is None:
            break

        col_j = Mt[s]
        i = len(col_j) - 1  # bottommost row in this column
        j = s

        # Trace gamma path: north when values equal above, east otherwise
        while True:
            old = col_j[i]
            col_j[i] -= 1
            if i > 0 and old == col_j[i - 1]:
                i -= 1  # move north
            else:
                j += 1  # move east
                if j == lam[i]:
                    break  # hit right boundary of row i
                col_j = Mt[j]

        res[i][s] += 1

    return res


# ═══════════════════════════════════════════════════════════════════════════════
# Burge local rule (from Robin's thesis §2.2.10)
# ═══════════════════════════════════════════════════════════════════════════════


def _random_horizontal_strip_extension(mu: list[int], rng) -> list[int]:
    """Generate a random partition α such that α/μ is a horizontal strip.

    A horizontal strip means at most one box added per column:
    α'_j ∈ {μ'_j, μ'_j + 1} for all j, and α' is weakly decreasing.
    """
    mu_conj = partition_conjugate(mu) if mu else []
    max_col = (len(mu_conj) + 3) if mu_conj else 3

    alpha_conj: list[int] = []
    for j in range(max_col):
        base = mu_conj[j] if j < len(mu_conj) else 0
        can_add = not alpha_conj or base + 1 <= alpha_conj[-1]
        if can_add and rng.random() < 0.5:
            alpha_conj.append(base + 1)
        else:
            alpha_conj.append(base)

    while alpha_conj and alpha_conj[-1] == 0:
        alpha_conj.pop()

    return partition_conjugate(alpha_conj) if alpha_conj else []


def burge_forward_rule(
    alpha: list[int], beta: list[int], m: int, mu: list[int],
) -> list[int]:
    """
    Burge forward local rule: 𝔘_{α,β}(m, μ) = λ.

    Given border partitions α, β (with μ ⊂ α, μ ⊂ β as horizontal strips)
    and non-negative integer m, compute output partition λ.
    Uses 1-indexed column arithmetic internally (thesis §2.2.10).
    """
    alpha_conj = partition_conjugate(alpha) if alpha else []
    beta_conj = partition_conjugate(beta) if beta else []
    mu_conj = partition_conjugate(mu) if mu else []

    max_cols = max(len(alpha_conj), len(beta_conj), len(mu_conj), 1)
    ac = alpha_conj + [0] * (max_cols - len(alpha_conj))
    bc = beta_conj + [0] * (max_cols - len(beta_conj))
    mc = mu_conj + [0] * (max_cols - len(mu_conj))

    # A, B are 1-indexed column sets
    A = {j + 1 for j in range(max_cols) if ac[j] > mc[j]}
    B = {j + 1 for j in range(max_cols) if bc[j] > mc[j]}

    AB = A | B
    AB_inter = sorted(A & B)  # increasing order

    # ε: each i ∈ A∩B → smallest > i not in A∪B∪used
    used: set[int] = set()
    C: set[int] = set()
    for i in AB_inter:
        eps = i + 1
        while eps in AB or eps in used:
            eps += 1
        C.add(eps)
        used.add(eps)

    # D: first m elements of complement of A∪B∪C
    excluded = AB | C
    D: set[int] = set()
    col = 1
    while len(D) < m:
        if col not in excluded:
            D.add(col)
        col += 1

    # λ = μ + one box at bottom of each column in A∪B∪C∪D
    add_cols = A | B | C | D
    max_col = max(max(add_cols) if add_cols else 0, len(mu_conj))
    lam_conj: list[int] = []
    for j_0 in range(max_col):
        val = mc[j_0] if j_0 < len(mc) else 0
        if (j_0 + 1) in add_cols:
            val += 1
        lam_conj.append(val)

    while lam_conj and lam_conj[-1] == 0:
        lam_conj.pop()

    return partition_conjugate(lam_conj) if lam_conj else []


def burge_inverse_rule(
    alpha: list[int], beta: list[int], lam: list[int],
) -> tuple[int, list[int]]:
    """
    Burge inverse local rule: 𝔇_{α,β}(λ) = (m, μ).

    Given border partitions α, β and output partition λ, recover the
    non-negative integer m and input partition μ.
    Uses 1-indexed column arithmetic internally (thesis §2.2.10).
    """
    alpha_conj = partition_conjugate(alpha) if alpha else []
    beta_conj = partition_conjugate(beta) if beta else []
    lam_conj = partition_conjugate(lam) if lam else []

    max_cols = max(len(alpha_conj), len(beta_conj), len(lam_conj), 1)
    ac = alpha_conj + [0] * (max_cols - len(alpha_conj))
    bc = beta_conj + [0] * (max_cols - len(beta_conj))
    lc = lam_conj + [0] * (max_cols - len(lam_conj))

    A_bar = {j + 1 for j in range(max_cols) if lc[j] > ac[j]}
    B_bar = {j + 1 for j in range(max_cols) if lc[j] > bc[j]}

    AB_bar = A_bar | B_bar
    AB_bar_inter = sorted(A_bar & B_bar, reverse=True)  # decreasing order

    # δ: each i ∈ Ā∩B̄ → largest < i not in Ā∪B̄∪used
    used: set[int] = set()
    C_bar: set[int] = set()
    m = 0
    for i in AB_bar_inter:
        delta = i - 1
        while delta > 0 and (delta in AB_bar or delta in used):
            delta -= 1
        if delta > 0:
            C_bar.add(delta)
            used.add(delta)
        else:
            m += 1

    # μ = λ minus one box from bottom of each column in Ā∪B̄∪C̄
    remove_cols = A_bar | B_bar | C_bar
    mu_conj: list[int] = []
    for j_0 in range(len(lc)):
        val = lc[j_0]
        if (j_0 + 1) in remove_cols:
            val -= 1
        mu_conj.append(val)

    while mu_conj and mu_conj[-1] == 0:
        mu_conj.pop()

    mu = partition_conjugate(mu_conj) if mu_conj else []
    return m, mu


# ═══════════════════════════════════════════════════════════════════════════════
# Sampling and verification for Hillman-Grassl / Burge
# ═══════════════════════════════════════════════════════════════════════════════


def sample_filling(shape: list[int], max_val: int, rng) -> Filling:
    """Sample a random non-negative integer filling of shape λ, entries in {0, ..., max_val}."""
    return [[rng.randint(0, max_val) for _ in range(row_len)] for row_len in shape]


def verify_hg_bijection(
    shape: list[int], max_val: int, num_samples: int = 1000, verbose: bool = False,
) -> bool:
    """
    Verify Hillman-Grassl round-trip on random fillings.

    Checks: (1) forward produces valid RPP, (2) weight preservation,
    (3) round-trip filling → RPP → filling.
    """
    import random

    rng = random.Random(42)
    errors = 0
    hooks = hook_lengths(shape)

    for _ in range(num_samples):
        filling = sample_filling(shape, max_val, rng)
        rpp = hillman_grassl_forward(shape, filling)

        if not is_rpp(rpp):
            if verbose:
                print(f"FAIL: not RPP for filling={filling}, rpp={rpp}")
            errors += 1
            continue

        rpp_weight = sum(v for row in rpp for v in row)
        fill_weight = sum(
            filling[r][c] * hooks[r][c]
            for r in range(len(shape))
            for c in range(shape[r])
        )
        if rpp_weight != fill_weight:
            if verbose:
                print(f"FAIL: weight {rpp_weight} != {fill_weight} for filling={filling}")
            errors += 1
            continue

        filling_back = hillman_grassl_inverse(shape, rpp)
        if filling_back != filling:
            if verbose:
                print(f"FAIL: round-trip for filling={filling}, got {filling_back}")
            errors += 1

    success = errors == 0
    if verbose or not success:
        print(f"  shape={shape}, max_val={max_val}: {num_samples - errors}/{num_samples}, {errors} errors")
    return success


def verify_burge_local_rule(
    num_samples: int = 500, verbose: bool = False,
) -> bool:
    """Verify Burge local rule round-trip: 𝔘 then 𝔇 recovers (m, μ)."""
    import random

    rng = random.Random(42)
    errors = 0

    for _ in range(num_samples):
        # Random μ
        num_parts = rng.randint(0, 5)
        mu = sorted([rng.randint(1, 8) for _ in range(num_parts)], reverse=True) if num_parts else []

        # Random horizontal strip extensions
        alpha = _random_horizontal_strip_extension(mu, rng)
        beta = _random_horizontal_strip_extension(mu, rng)
        m = rng.randint(0, 3)

        lam = burge_forward_rule(alpha, beta, m, mu)
        m_back, mu_back = burge_inverse_rule(alpha, beta, lam)

        if m_back != m or mu_back != mu:
            if verbose:
                print(f"FAIL: α={alpha}, β={beta}, m={m}, μ={mu}")
                print(f"  λ={lam}, got m={m_back}, μ={mu_back}")
            errors += 1

    success = errors == 0
    if verbose or not success:
        print(f"  Burge local rule: {num_samples - errors}/{num_samples}, {errors} errors")
    return success


if __name__ == "__main__":
    # Self-test: verify RSK round-trip for small n (permutations)
    print("Verifying RSK bijection (permutations)...")
    for n in range(1, 7):
        ok = verify_bijection(n, verbose=True)
        status = "OK" if ok else "FAILED"
        print(f"  S_{n} ({factorial(n):>4d} permutations): {status}")

    # Self-test: verify RSK round-trip for words
    print("\nVerifying RSK bijection (words)...")
    for m, k in [(5, 3), (8, 5), (10, 4), (15, 10)]:
        ok = verify_word_bijection(m, k, num_samples=1000, verbose=True)
        status = "OK" if ok else "FAILED"
        print(f"  {m}-letter words over {k}-letter alphabet: {status}")

    # Self-test: verify RSK round-trip for matrices
    print("\nVerifying RSK bijection (matrices)...")
    for a, b, total_n in [(2, 2, 5), (3, 3, 10), (4, 4, 15), (3, 5, 12)]:
        ok = verify_matrix_bijection(a, b, total_n, num_samples=1000, verbose=True)
        status = "OK" if ok else "FAILED"
        print(f"  {a}x{b} matrices, N={total_n}: {status}")

    # Example: permutation
    sigma = [4, 2, 7, 3, 6, 1, 5, 8]
    P, Q = rsk_forward(sigma)
    print(f"\nExample (permutation): σ = {sigma}")
    print(f"P = {P}")
    print(f"Q = {Q}")
    print(f"Shape: {tableau_shape(P)}")
    sigma_back = rsk_inverse(P, Q)
    print(f"Inverse: {sigma_back}")
    print(f"Round-trip OK: {sigma_back == sigma}")

    # Example: word with repeated values
    word = [3, 1, 4, 1, 5, 2, 2, 3]
    P, Q = rsk_forward(word)
    print(f"\nExample (word): w = {word}")
    print(f"P = {P}  (SSYT: {is_semistandard_young_tableau(P)})")
    print(f"Q = {Q}  (SYT: {is_standard_young_tableau(Q)})")
    print(f"Shape: {tableau_shape(P)}")
    word_back = rsk_inverse(P, Q)
    print(f"Inverse: {word_back}")
    print(f"Round-trip OK: {word_back == word}")

    # Example: matrix
    A = [[1, 0, 2], [0, 1, 0], [1, 1, 0]]
    top, bottom = matrix_to_biword(A)
    P, Q = rsk_forward_biword(top, bottom)
    print(f"\nExample (matrix): A = {A}")
    print(f"Biword: top={top}, bottom={bottom}")
    print(f"P = {P}  (SSYT: {is_semistandard_young_tableau(P)})")
    print(f"Q = {Q}  (SSYT: {is_semistandard_young_tableau(Q)})")
    print(f"Shape: {tableau_shape(P)}")
    top_back, bottom_back = rsk_inverse_biword(P, Q)
    print(f"Inverse: top={top_back}, bottom={bottom_back}")
    print(f"Round-trip OK: {top_back == top and bottom_back == bottom}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Hillman-Grassl self-tests
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Verifying Hillman-Grassl bijection...")
    for shape, max_val in [
        ([2, 1], 3), ([3, 2], 3), ([3, 2, 1], 3), ([4, 3, 2, 1], 2),
        ([4, 4], 3), ([5, 5, 5], 2), ([3, 3, 3, 3], 2),
    ]:
        ok = verify_hg_bijection(shape, max_val, num_samples=1000, verbose=True)
        status = "OK" if ok else "FAILED"
        print(f"    → {status}")

    # Hillman-Grassl worked example
    shape_ex = [3, 2]
    filling_ex = [[1, 0, 0], [0, 0]]
    rpp_ex = hillman_grassl_forward(shape_ex, filling_ex)
    hooks_ex = hook_lengths(shape_ex)
    print(f"\nExample (Hillman-Grassl): shape={shape_ex}, filling={filling_ex}")
    print(f"  RPP = {rpp_ex}  (valid: {is_rpp(rpp_ex)})")
    print(f"  hooks = {hooks_ex}")
    rpp_w = sum(v for row in rpp_ex for v in row)
    fill_w = sum(filling_ex[r][c] * hooks_ex[r][c] for r in range(len(shape_ex)) for c in range(shape_ex[r]))
    print(f"  RPP weight = {rpp_w}, filling hook-weight = {fill_w}")
    filling_back = hillman_grassl_inverse(shape_ex, rpp_ex)
    print(f"  Round-trip: {filling_back} == {filling_ex}: {filling_back == filling_ex}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Burge local rule self-tests
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Verifying Burge local rule...")
    ok = verify_burge_local_rule(num_samples=500, verbose=True)
    print(f"  → {'OK' if ok else 'FAILED'}")

    # Worked example from thesis: 𝔇_{(6,5,5,3),(6,6,5,2)}(7,6,5,3,1) = (1, (6,5,4,2))
    alpha_ex = [6, 5, 5, 3]
    beta_ex = [6, 6, 5, 2]
    lam_ex = [7, 6, 5, 3, 1]
    m_ex, mu_ex = burge_inverse_rule(alpha_ex, beta_ex, lam_ex)
    print(f"\nWorked example (thesis §2.2.10):")
    print(f"  𝔇_{{({alpha_ex}),({beta_ex})}}({lam_ex}) = ({m_ex}, {mu_ex})")
    print(f"  Expected: (1, [6, 5, 4, 2]) → {'OK' if m_ex == 1 and mu_ex == [6, 5, 4, 2] else 'FAILED'}")
    # Verify round-trip
    lam_back = burge_forward_rule(alpha_ex, beta_ex, m_ex, mu_ex)
    print(f"  Round-trip 𝔘: {lam_back} == {lam_ex}: {lam_back == lam_ex}")
