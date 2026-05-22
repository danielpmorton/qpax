# Forward and Backward

Every QP `qpax` solves has the form

$$
\underset{x}{\text{minimize}} \quad \tfrac{1}{2}\,x^{\top} Q x + q^{\top} x
\quad \text{s.t.} \quad
A x = b, \;\; G x \leq h.
$$

Introducing a slack $s \geq 0$, equality multiplier $y$, and inequality
multiplier $z \geq 0$, the KKT conditions are

$$
\begin{aligned}
Q x + q + A^{\top} y + G^{\top} z &= 0, \\
G x + s - h &= 0, \\
A x - b &= 0, \\
s_i z_i &= 0, \qquad s, z \geq 0.
\end{aligned}
$$

As in primal-dual interior-point methods, we replace the hard complementarity $s_i z_i = 0$ with a
smooth condition $s_i z_i = \mu$, then drive $\mu \to 0$. The specific details of how the forward and backward passes are conducted depend on the underlying explicit or implicit backend (`backend="e"` or `backend="i"`).

<!-- Each iteration:

1. **Form residuals.** Evaluate the KKT residual at the current iterate.
2. **Factor the KKT system.** Reduce the saddle-point system to a smaller
   positive-definite system in $x$ using the elimination from `cvxopt`, and
   factor it (Cholesky by default, QR on request).
3. **Compute a Newton step.** Solve the reduced KKT system for the step
   directions $(\Delta x, \Delta s, \Delta z, \Delta y)$.
4. **Choose a step size.** Pick $\alpha \in (0, 1]$ so that the slack/dual
   variables stay strictly positive.
5. **Update.** Take a step of length $\alpha$ along the Newton direction.

This skeleton is shared by both backends. The **explicit** backend
(`backend="e"`) is the predictor–corrector PDIP from `cvxgen`: steps in the
slack/dual variables are taken directly, shortened by a fraction-to-the-boundary
line search. The **implicit** backend (`backend="i"`) parameterises the
slack/dual pair by manifold coordinates $v = z - s$ and a scalar gap
$\kappa = (s^{\top} z) / m$, takes a Newton step against an intrinsically
positive-definite reduced KKT matrix, and retracts back to the positive cone
after each step. Use the tabs below to switch between backends — the selection
is shared across every algorithm on this page. -->

## Forward pass

The forward pass first drives the iterate to a KKT point (*Solve QP*) and
then refines it to a $\kappa$-relaxed KKT point (*Relax QP*) so that the
implicit-function gradients used in the backward pass are well conditioned.

### Solve QP

=== "Explicit"

    ```text
    function solve_qp_explicit(Q, q, A, b, G, h; tol, max_iter)

        (x, s, z, y) ← initialize(Q, q, A, b, G, h)

        for i = 1, ..., max_iter do

            # Residuals and convergence check
            r ← evaluate_residuals(Q, q, A, b, G, h, x, y, z, s)
            if ‖r‖∞ < tol then
                return x, y, z, s
            end if

            # Factor the reduced KKT system once per iteration
            K̃ ← precompute_kkt_factors(Q, A, G, s, z)

            # 1. Predictor: affine step with μ = 0
            (Δxa, Δsa, Δza, Δya) ← solve_kkt(K̃, r, μ = 0)
            αa ← linesearch(s, z, Δsa, Δza)

            # 2. Centering parameter
            μ      ← (sᵀ z) / m
            μ_aff  ← ((s + αa Δsa)ᵀ (z + αa Δza)) / m
            σ      ← (μ_aff / μ)^3

            # 3. Corrector with second-order correction in complementarity
            (Δx, Δs, Δz, Δy) ← solve_kkt(K̃, r, σμ − Δsa ⊙ Δza)

            # 4. Step
            α ← linesearch(s, z, Δs, Δz)
            (x, s, z, y) ← (x + αΔx, s + αΔs, z + αΔz, y + αΔy)

        end for

    end function
    ```

=== "Implicit"

    ```text
    function solve_qp_implicit(Q, q, A, b, G, h; σ, tol, max_iter)

        (x, s, z, y) ← initialize(Q, q, A, b, G, h)

        for i = 1, ..., max_iter do

            # Manifold coordinates
            v ← z - s
            κ ← (sᵀ z) / m

            # Residuals and convergence check
            r ← evaluate_residuals(Q, q, A, b, G, h, x, y, z, s)
            if ‖r‖∞ < tol then
                return x, y, z, s
            end if

            # Factor the reduced KKT system once per iteration
            K̃        ← precompute_kkt_factors(Q, A, G, v, κ)

            # Solve KKT
            κ_target ← σκ
            (Δx, Δy, Δz, Δs, Δv, Δκ) ← solve_kkt(K̃, r, κ_target)

            # Step
            α ← linesearch(s, z, Δs, Δz)

            x ← x + αΔx
            y ← y + αΔy
            v ← v + αΔv
            κ ← κ + αΔκ

            # Retract back to positive slack-dual variables
            (z, s) ← retraction_map(v, κ)

        end for

    end function
    ```

### Relax QP

Once Solve QP has converged, the iterate sits at a KKT point where the complementarity gap collapses to zero, causing the implicit-function gradients used in differentiation to become non-informative. *Relax QP* drives the iterate to a $\kappa$-relaxed KKT point instead, trading a controlled bias for usable sensitivities.

=== "Explicit"

    ```text
    function relax_qp_explicit(Q, q, A, b, G, h, x, s, z, y; κ_relax, tol, max_iter)

        for i = 1, ..., max_iter do
            r ← evaluate_residuals(Q, q, A, b, G, h, x, y, z, s)
            if ‖r‖∞ < tol then
                return x, y, z, s
            end if

            K̃ ← precompute_kkt_factors(Q, A, G, s, z)
            (Δx, Δs, Δz, Δy) ← solve_kkt(K̃, r, κ_relax)
            α ← linesearch(s, z, Δs, Δz)

            (x, s, z, y) ← (x + αΔx, s + αΔs, z + αΔz, y + αΔy)
        end for

    end function
    ```

=== "Implicit"

    ```text
    function relax_qp_implicit(Q, q, A, b, G, h, x, y, z, s, κ_relax; K̃, tol, max_iter)

        for i = 1, ..., max_iter do
            v ← z - s
            κ ← (sᵀ z) / m

            r ← evaluate_residuals(Q, q, A, b, G, h, x, y, z, s)
            if ‖r‖∞ < tol then
                return x, y, z, s
            end if

            if K̃ is not cached then
                K̃ ← precompute_kkt_factors(Q, A, G, v, κ)
            end if

            (Δx, Δy, Δz, Δs, Δv, Δκ) ← solve_kkt(K̃, r, κ_relax)
            α ← linesearch(s, z, Δs, Δz)

            x ← x + αΔx
            y ← y + αΔy
            v ← v + αΔv
            κ ← κ + αΔκ

            (z, s) ← retraction_map(v, κ)
        end for

    end function
    ```

## Backward pass

The backward pass applies the implicit-function theorem at the relaxed KKT point to compute gradients.
<!-- It reuses the KKT factorisation cached during the
forward pass, so differentiating a converged solve costs little more than a
single extra triangular solve. -->

### Computing gradients

=== "Explicit"

    ```text
    function compute_qp_gradients_explicit(Q, q, A, b, G, h, x, y, z, s, κ_relax, ∇xl; K̃)

        r ← (-∇xl, 0, 0, 0)
        (dx, dy, dz, ds) ← solve_kkt(K̃, r, κ_relax)

        ∇Ql ← (dx xᵀ + x dxᵀ) / 2
        ∇ql ← dx
        ∇Al ← dy xᵀ + y dxᵀ
        ∇bl ← -dy
        ∇Gl ← dz xᵀ + z dxᵀ
        ∇hl ← -dz

        return ∇Ql, ∇ql, ∇Al, ∇bl, ∇Gl, ∇hl

    end function
    ```

=== "Implicit"

    ```text
    function compute_qp_gradients_implicit(Q, q, A, b, G, h, x, y, z, s, κ_relax, ∇xl; K̃)

        r ← (-∇xl, 0, 0, 0, 0)
        (dx, dy, dz, ds, _, _) ← solve_kkt(K̃, r, κ_relax)

        ∇Ql ← (dx xᵀ + x dxᵀ) / 2
        ∇ql ← dx
        ∇Al ← dy xᵀ + y dxᵀ
        ∇bl ← -dy
        ∇Gl ← dz xᵀ + z dxᵀ
        ∇hl ← -dz

        return ∇Ql, ∇ql, ∇Al, ∇bl, ∇Gl, ∇hl

    end function
    ```
