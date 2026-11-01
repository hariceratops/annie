"""Benchmark: n row-dot-products against the same x, from one flat buffer.

Question: the weights live in one flat 1D buffer (SciPy's optimisers want a
1D parameter vector). To compute result[i] = row_i . x for all n rows, is it
faster to slice the buffer into rows (and maybe np.stack them), or to view
it as an (n, m) matrix with reshape and do a single A @ x?

Answer: the 2D reshape view wins or ties in every case measured. Nothing
built from slices beats it.

Results (uv env: Python 3.12.3, NumPy 2.5.4, scipy-openblas 0.3.34, 4 cores,
WSL2, float64). Time per call in us, best of 7, ratio vs the prebuilt view:

    n x m          A@x   reshape/call   slice loop   row-view loop   stack(slices)   stack(rows)
    1000 x 10        7     8 (1.1x)     2369 (318x)   1849 (248x)    1081 (145x)    689  (92x)
    1000 x 100      28    28 (1.0x)     2402  (86x)   1849  (66x)    1381  (49x)    708  (25x)
    1000 x 1000     62    74 (1.2x)     3087  (50x)   2483  (40x)    7137 (115x)   4784  (77x)
    100 x 100k    3468  3095 (0.9x)     3017 (0.9x)   3240 (0.9x)   32210 (9.3x)  36936 (10.6x)
    10 x 1M       2723  3156 (1.2x)    10336 (3.8x)   2955 (1.1x)   26166 (9.6x)  32458 (11.9x)
    10000 x 10      58    58 (1.0x)    23655 (405x)  18251 (313x)   11442 (196x)   6590 (113x)

The 80 MB cases run each variant only once per repeat, so +-20% there is
noise (WSL2 plus multithreaded BLAS); e.g. the 3.8x slice loop at 10 x 1M
cannot be slicing cost, since only 10 views are made.

Reasoning:
- reshape on a C-contiguous buffer only builds a new array header over the
  same memory, so the (n, m) view costs ~1 us and no copy. Rebuilding it on
  every call is as fast as caching it.
- A @ x on that view is one BLAS gemv: a single pass over n*m contiguous
  floats, multithreaded by OpenBLAS.
- A Python loop of row @ x pays ~2 us of interpreter and dispatch overhead
  per row and runs each dot single-threaded. It only ties gemv when rows are
  so long (m >= 100k) that each dot is memory-bound and overhead vanishes.
- np.stack always allocates and copies all n*m floats, even when every row
  is already contiguous, because the rows must become adjacent. Then gemv
  reads the copy again. So stacking costs at least the copy on top of the
  same gemv: never a win. With a fast BLAS it is ~10x slower at large m.
- Stacking beats the loop only for small m, where one call replaces n calls.

An earlier run on the system NumPy 1.26.4 (reference BLAS, slower gemv)
showed the same ordering with smaller gaps, e.g. stacking ~2.2x at large m:
the slower the gemv, the smaller the copy's share of the total.

Takeaway for the MLP: keep parameters in one flat buffer, pass it to SciPy
as-is, and inside the loss use w.reshape(n, m) @ x. Never slice-and-stack.
"""

import timeit
from collections.abc import Callable

import numpy as np

rng = np.random.default_rng(0)


def bench(fn: Callable[[], object], number: int) -> float:
    fn()
    return min(timeit.repeat(fn, number=number, repeat=7)) / number * 1e6


def run(n: int, m: int) -> dict[str, float]:
    flat = rng.random(n * m)
    x = rng.random(m)
    A = flat.reshape(n, m)
    buf = np.empty(n)
    rows = [flat[i * m:(i + 1) * m] for i in range(n)]

    # Baseline: (n, m) view built outside the timed region, one gemv.
    def prebuilt_view_matvec():
        return A @ x

    # Same gemv writing into a preallocated result; saves one n-float alloc.
    def prebuilt_view_matvec_out():
        return np.matmul(A, x, out=buf)

    # View rebuilt per call: measures what reshape itself costs.
    def reshape_matvec():
        return flat.reshape(n, m) @ x

    # n slices + n separate dot calls.
    def slice_loop():
        return [flat[i * m:(i + 1) * m] @ x for i in range(n)]

    # n dot calls only; slicing moved outside timing.
    def slice_loop_prebuilt():
        return [r @ x for r in rows]

    # n slices, copy into a new (n, m) array, then gemv on the copy.
    def slice_stack_matvec():
        return np.stack([flat[i * m:(i + 1) * m] for i in range(n)]) @ x

    # Best case for stacking: copy + gemv, slicing outside timing.
    def prebuilt_slices_stack_matvec():
        return np.stack(rows) @ x

    cases = {
        "A @ x (view built once)": prebuilt_view_matvec,
        "A @ x out= (view built once)": prebuilt_view_matvec_out,
        "flat.reshape(n,m) @ x (per call)": reshape_matvec,
        "loop of slice @ x (per call slicing)": slice_loop,
        "loop of prebuilt row views @ x": slice_loop_prebuilt,
        "stack(per-call slices) @ x": slice_stack_matvec,
        "stack(prebuilt row views) @ x": prebuilt_slices_stack_matvec,
    }
    expected = A @ x
    for f in cases.values():
        np.testing.assert_allclose(f(), expected)
    # Aim for a similar total runtime per size: more calls for small arrays.
    number = min(max(1, int(2e6 / (n * m)) * 5), 20000)
    return {k: bench(f, number) for k, f in cases.items()}


def main() -> None:
    for n, m in [(1000, 10), (1000, 100), (1000, 1000), (100, 100_000), (10, 1_000_000), (10000, 10)]:
        print(f"\nn={n} m={m} ({n * m * 8 / 1e6:.2f} MB)  [us per call, best of 7]")
        res = run(n, m)
        base = res["A @ x (view built once)"]
        for k, v in res.items():
            print(f"  {k:<40}{v:>12.2f}  {v / base:>7.1f}x")


if __name__ == "__main__":
    main()
