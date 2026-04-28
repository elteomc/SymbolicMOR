# Shared helpers for benchmark scripts (included via `include`).

using LinearAlgebra

"""In-place RHS for du/dt = A*u + H*kron(u,u) + c."""
function quadratic_state_rhs!(du, u, p, t)
    A, H, c = p
    du .= A * u .+ H * kron(u, u) .+ c
    return nothing
end
