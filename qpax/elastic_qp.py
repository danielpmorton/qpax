"""Top-level elastic-QP dispatcher.

Routes ``solve_qp_elastic`` and ``solve_qp_elastic_primal`` to either the
explicit (``backend="e"``) or implicit (``backend="i"``) backend.
"""

from typing import Any

import jax

from qpax.explicit import elastic_qp as _explicit
from qpax.implicit import elastic_qp as _implicit


def solve_qp_elastic(
    Q: jax.Array,
    q: jax.Array,
    G: jax.Array,
    h: jax.Array,
    penalty: float,
    *,
    backend: str = "e",
    **kwargs: Any,
):
    """Solve the elastic QP relaxation of an inequality-constrained QP.

    Args:
        Q: ``(n, n)`` PSD cost matrix.
        q: ``(n,)`` linear cost.
        G: ``(p, n)`` inequality constraint matrix.
        h: ``(p,)`` inequality constraint RHS.
        penalty: per-unit cost of slack on each inequality.
        backend: ``"e"`` for explicit PDIP or ``"i"`` for implicit PDIP.
        **kwargs: forwarded to the selected backend (e.g. ``solver_tol`` and
            ``max_iter``).
    """
    if backend == "e":
        return _explicit.solve_qp_elastic(Q, q, G, h, penalty, **kwargs)
    if backend == "i":
        return _implicit.solve_qp_elastic(Q, q, G, h, penalty, **kwargs)
    raise ValueError(f"unknown backend {backend!r}; expected 'e' or 'i'")


def solve_qp_elastic_primal(
    Q: jax.Array,
    q: jax.Array,
    G: jax.Array,
    h: jax.Array,
    penalty: float,
    *,
    backend: str = "e",
    **kwargs: Any,
) -> jax.Array:
    """Differentiable elastic-QP solve returning only the primal ``x``.

    Args:
        Q: ``(n, n)`` PSD cost matrix.
        q: ``(n,)`` linear cost.
        G: ``(p, n)`` inequality constraint matrix.
        h: ``(p,)`` inequality constraint RHS.
        penalty: per-unit cost of slack on each inequality.
        backend: ``"e"`` for explicit PDIP or ``"i"`` for implicit PDIP.
        **kwargs: forwarded to the selected backend.
    """
    if backend == "e":
        return _explicit.solve_qp_elastic_primal(Q, q, G, h, penalty, **kwargs)
    if backend == "i":
        return _implicit.solve_qp_elastic_primal(Q, q, G, h, penalty, **kwargs)
    raise ValueError(f"unknown backend {backend!r}; expected 'e' or 'i'")


__all__ = ["solve_qp_elastic", "solve_qp_elastic_primal"]
