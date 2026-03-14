"""
RSK (Robinson-Schensted-Knuth) correspondence: forward and inverse.

The RSK correspondence is a bijection between permutations σ ∈ S_n and pairs
(P, Q) of standard Young tableaux of the same shape λ ⊢ n.

Forward: σ → (P, Q) via Schensted row insertion (bumping algorithm).
Inverse: (P, Q) → σ via reverse bumping, processing entries in decreasing order.

Tableaux are represented as list-of-lists: tableau[i] is row i (0-indexed),
each row is a list of integers in increasing order.
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


if __name__ == "__main__":
    # Self-test: verify RSK round-trip for small n
    print("Verifying RSK bijection...")
    for n in range(1, 7):
        ok = verify_bijection(n, verbose=True)
        status = "OK" if ok else "FAILED"
        print(f"  S_{n} ({factorial(n):>4d} permutations): {status}")

    # Example
    sigma = [3, 1, 4, 1, 5, 9, 2, 6]  # Not a permutation, but let's use a real one
    sigma = [4, 2, 7, 3, 6, 1, 5, 8]
    P, Q = rsk_forward(sigma)
    print(f"\nExample: σ = {sigma}")
    print(f"P = {P}")
    print(f"Q = {Q}")
    print(f"Shape: {tableau_shape(P)}")
    sigma_back = rsk_inverse(P, Q)
    print(f"Inverse: {sigma_back}")
    print(f"Round-trip OK: {sigma_back == sigma}")
