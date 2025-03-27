# SPDE Project Guidelines

## Build/Test Commands
- Activate project: `using DrWatson; @quickactivate "SPDE"`
- Run all tests: `julia test/runtests.jl`
- Run single test: `julia -e 'using Pkg; Pkg.test("SPDE", test_args=["SPDE tests"])'`
- Generate solution: `SPDE.generate_solution(σ, 2^8, 2^16)`

## Code Style Guidelines
- **Imports**: Group imports with `using` statements, import specific functions with `import`
- **Formatting**: 4-space indentation, avoid trailing whitespace
- **Types**: Use type annotations in function signatures
- **Naming**: 
  - Functions: camelCase (e.g., `partial_integration`, `get_drift_diffusion`)
  - Variables: lowercase, snake_case, descriptive names
- **Performance**: Use `@inbounds` for array access, `Threads.@threads` for parallelism
- **Error Handling**: Use defensive programming with boundary checks
- **Comments**: Include mathematical description for complex algorithms
- **Dependencies**: Use DrWatson for project management and data tracking

## Documentation Guidelines
- Update mathematical theory in `theory.md` when implementing new estimators
- Ensure mathematical notation is consistent between code comments and theory
- Add theoretical insights and relevant paper references to documentation
- Include explanation of numerical methods and their relationship to theory

## Development Process
- Use DrWatson's `@quickactivate` in all scripts
- Define experiments with parameter dictionaries
- Store results with DrWatson's `safesave` to avoid data loss
- Reference theoretical results from `theory.md` when developing new methods