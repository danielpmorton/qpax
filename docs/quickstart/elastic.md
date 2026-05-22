# Solve an elastic QP

The elastic relaxation adds non-negative slacks `t` to every inequality and
penalizes them in the cost, so the solver still returns a sensible answer
when the original constraints are infeasible. Reach for this recipe when
you cannot guarantee `G x \le h` is satisfiable.

## The problem

The example builds a deliberately infeasible inequality QP (the constraints
require both $x_0 \le -1$ and $x_0 \ge 1$) and solves its elastic form:

$$
\begin{aligned}
\min_{x,\, t}\;& \tfrac{1}{2}\, x^{\mathsf T} Q x + q^{\mathsf T} x
                + \rho\, \mathbf{1}^{\mathsf T} t \\
\text{s.t.}\;& G x \le h + t \\
& t \ge 0.
\end{aligned}
$$

[`qpax.solve_qp_elastic`][qpax.solve_qp_elastic] returns
`(x, t, s1, s2, z1, z2, converged, iters)`: the primal `x`, the slack `t`,
the two inequality slack/dual pairs `(s1, z1)` and `(s2, z2)`, a
convergence flag, and the iteration count. Non-zero entries of `t` mark
the rows where the original constraint had to be relaxed.

## Code

```python
--8<-- "examples/elastic_qp.py"
```
