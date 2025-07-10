## v0.4.0 - 2025-07-09

### Added
- Replace field factories with simple defaults in BmiFields base class
- Add __init_subclass__ validation to ensure required fields are defined
- Create dedicated Model252BmiFields and Model254BmiFields classes
- Simplify model class definitions by using dedicated BMI field classes
- Remove unused TypeVar import

### Fixed
- Update DOI to represent all versions

## v0.3.0 - 2025-07-08

### Breaking Changes
- Renamed `init()` classmethod to `from_array()` in ModelStates classes

### Added
- Generic type parameters for type-safe model inheritance hierarchy
- mypy static type checking with library stubs as dev dependencies

### Fixed
- Fixed unbound variables in `load_network_topology()`, `load_network_parameters()`, and `initialize_time()`
- Fixed missing exception handling in `read_csv()` method

### Development
- Added comprehensive type checking infrastructure
- Eliminated pylint type override warnings
- Improved IDE experience with better type safety
- Relaxed dependency version constraints

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
