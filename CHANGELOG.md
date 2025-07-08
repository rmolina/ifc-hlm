## [v0.2.1] - 2025-05-06
### Changed
- Add basic README for PyPI compatibility.

## [v0.2.0] - 2025-05-06
### Added
- New model: Model252.
- Added type hints for get_baseflow() in Model254.
### Changed
- Changed how `ode_solver` is treated in the YAML config file.
  It was used to pass the `method` argument to the `solve_ivp` function.
  It is now mapped to pass several options: `method`,  `atol`,  `rtol`, and `max_step`.

## [v0.1.0] - 2025-02-06
### Added
- Initial release with Model254.
