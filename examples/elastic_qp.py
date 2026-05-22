"""Solve one infeasible inequality QP with qpax's elastic relaxation.

The problem is

    minimize_x  0.5 x^T Q x + q^T x + penalty * 1^T t
    subject to  G x <= h + t,  t >= 0

The constraints are deliberately infeasible (x_0 <= -1 and x_0 >= 1), so the
elastic relaxation activates and returns a slack vector t with non-zero
entries on the violated rows.
"""

import jax
import jax.numpy as jnp
import numpy as np

import qpax

USE_F64 = False
BACKEND = "i"  # "e" for explicit, "i" for implicit
VERBOSE = True

if USE_F64:
    jax.config.update("jax_enable_x64", True)

dtype = jnp.float64 if USE_F64 else jnp.float32


def generate_infeasible_qp(n, dtype):
    rng = np.random.default_rng(0)

    M = rng.normal(size=(n, n))
    Q = M.T @ M + 1e-2 * np.eye(n)
    q = rng.normal(size=n)

    # Two opposing constraints on x_0 make the problem infeasible.
    G = np.zeros((2, n))
    G[0, 0] = 1.0
    G[1, 0] = -1.0
    h = np.array([-1.0, -1.0])

    return (jnp.asarray(arr, dtype=dtype) for arr in (Q, q, G, h))


# Build one infeasible QP and solve its elastic relaxation.
Q, q, G, h = generate_infeasible_qp(n=3, dtype=dtype)
penalty = 10.0

x, t, _, _, _, _, converged, iters = qpax.solve_qp_elastic(
    Q, q, G, h, penalty, backend=BACKEND, verbose=VERBOSE
)

print("converged:", int(converged))
print("iterations:", int(iters))
print("x:", x)
print("slack t:", t)
