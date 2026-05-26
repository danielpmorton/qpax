# Examples

Explore the accompanying [examples repository](https://github.com/qpax-solver/qpax-examples) for end-to-end applications and backend comparisons:

- **[Bilevel Optimization](https://github.com/qpax-solver/qpax-examples/blob/main/examples/bilevel_trajectory_optimization)** — An end-to-end bilevel trajectory optimization example in which the inner QP enforces safety and smoothness, while an outer L-BFGS loop minimizes travel time.
- **[Learning a Multiagent Safety Filter](https://github.com/qpax-solver/qpax-examples/blob/main/examples/learning_safety_filter)** — Learn a multi-agent control barrier function safety filter from expert demonstrations.
- **[Autotuning MPC](https://github.com/qpax-solver/qpax-examples/blob/main/examples/autotuning_mpc/autotune_mpc.ipynb)** — Learn MPC cost weights from demonstration trajectories.
- **[Explicit / Implicit comparison](https://github.com/qpax-solver/qpax-examples/blob/main/examples/backend_comparison)** — Compare `qpax`'s implicit and explicit backends on the same problem.

## Projects using qpax

The following open-source projects use [qpax](https://github.com/qpax-solver/qpax) in compelling real-world applications:

- **[cbfpy](https://github.com/StanfordASL/cbfpy)**: Control Barrier Functions in Python and JAX.
- **[oscbf](https://github.com/StanfordASL/oscbf)**: Safe, high-performance, task-consistent manipulator control.
- **[frax](https://github.com/StanfordASL/frax)**: Fast robot kinematics and dynamics in JAX.

These projects are good references for seeing how qpax can be used beyond the examples in this repository.

## Contributing
If you are using `qpax` in an interesting application and would like it featured here, please open an [issue](https://github.com/qpax-solver/qpax-examples/issues) or [pull request](https://github.com/qpax-solver/qpax-examples/pulls) in the examples repository.
