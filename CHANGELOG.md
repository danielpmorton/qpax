# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0/).

Releases prior to `0.1.2` are documented in the
[GitHub Releases](https://github.com/qpax-solver/qpax/releases) page.

## [0.1.3] - 2026-05-26

### Changed

- Default solver `backend` switched from `"i"` (implicit retraction-manifold
  PDIP) to `"e"` (explicit predictor-corrector PDIP). This affects `solve_qp`,
  `solve_qp_primal`, `solve_qp_elastic`, and `solve_qp_elastic_primal`. Pass
  `backend="i"` explicitly to restore the previous default. (Jon Arrizabalaga)

[0.1.3]: https://github.com/qpax-solver/qpax/releases/tag/v0.1.3

## [0.1.2] - 2026-05-26

### Added

- Elastic mode for the implicit `"i"` backend. (Jon Arrizabalaga)

### Fixed

- Batching bug. (John Zhang)

[0.1.2]: https://github.com/qpax-solver/qpax/releases/tag/v0.1.2

## [0.1.4] - 2026-07-24

### Added

- Implicit backend (`backend="i"`) for the elastic QP solvers
  `solve_qp_elastic` and `solve_qp_elastic_primal`, which previously raised
  `NotImplementedError`. (Daniel Morton, Jon Arrizabalaga)

### Changed

- The implicit elastic solver now folds the per-constraint blocks into an
  `n`-by-`n` primal Schur complement solved with Cholesky, replacing the dense
  `(n + 3p)` LU factorization â better single-precision conditioning and speed.

[0.1.4]: https://github.com/qpax-solver/qpax/releases/tag/v0.1.4