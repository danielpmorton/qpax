# Algorithm

`qpax` is a primal–dual interior-point (PDIP) solver. This section walks
through the algorithms behind the solve and its derivatives:

- **[Forward and Backward](forward_backward.md)** — the PDIP problem
  statement, the forward pass (*Solve QP* and *Relax QP*), and the backward
  pass (*Computing gradients*). Use the tabs on the page to switch between
  the explicit (`backend="e"`) and implicit (`backend="i"`) variants.
- **[Elastic](elastic.md)** — the elastic-QP relaxation: problem statement
  and KKT conditions, followed by the forward pass (*Solve QP* and
  *Relax QP*) and backward pass (*Computing gradients*) for the explicit
  backend. The implicit backend is pending.
