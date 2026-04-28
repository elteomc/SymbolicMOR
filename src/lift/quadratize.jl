# src/lift/quadratize.jl
#
# Phase 1 - Symbolic Quadratization
#
# Goal: Given a polynomial ODE system x_dot = f(x), introduce auxiliary
# variables w so that the augmented system [x_dot; w_dot] = F(x, w) is at most
# quadratic (degree <= 2) in the combined state (x, w).
#
# Algorithm: Traverse the AST of each RHS expression. For every monomial
# of degree > 2, iteratively replace the highest-degree sub-product with
# a new auxiliary variable until all monomials have degree <= 2.
#
# Reference:
#   Bychkov and Pogudin (2021), "Optimal monomial quadratization for ODE systems"
#   arXiv:2103.08013

using Symbolics
using Symbolics: value, get_variables

# Public API

"""
    LiftedSystem

Container for the result of lifting a nonlinear ODE into quadratic form.

Fields:
  - original_vars : original state variables [x1, ..., xn]
  - lifted_vars   : full lifted state [x1, ..., xn, w1, ..., wm]
  - F             : RHS of the lifted system (quadratic in lifted_vars)
  - aux_eqs       : definitions of auxiliary variables (wi = g(x))
"""
struct LiftedSystem
  original_vars::Vector{Num}
  lifted_vars::Vector{Num}
  F::Vector{Num}
  aux_eqs::Vector{Equation}
end

"""
    lift_system(vars, rhs)

Lift a polynomial ODE system x_dot = rhs(x) into a quadratic form.

# Arguments
- vars::Vector{Num} : symbolic state variables, for example [x, y, z]
- rhs::Vector{Num}  : RHS expressions, one per variable

# Returns
A LiftedSystem containing the augmented variables and quadratic RHS.

# Example - Lorenz system
    @variables x y z
    sigma, rho, beta = 10.0, 28.0, 8/3

    rhs = [
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z,
    ]

    ls = lift_system([x, y, z], rhs)
    println("Lifted state dim: ", length(ls.lifted_vars))
    println("Max degree in lifted RHS: ", maximum(degree_of.(ls.F, Ref(ls.lifted_vars))))
"""
function lift_system(vars::Vector{Num}, rhs::Vector{Num})
  length(vars) == length(rhs) ||
    throw(ArgumentError("vars and rhs must have the same length"))

  lifted_vars = copy(vars)
  aux_eqs = Equation[]
  lifted_rhs = copy(rhs)
  aux_counter = Ref(0)

  # Iteratively quadratize each component of the RHS
  for i in eachindex(lifted_rhs)
    lifted_rhs[i], lifted_vars, aux_eqs =
      _quadratize_expr(lifted_rhs[i], lifted_vars, aux_eqs, aux_counter)
  end

  # Compute the time derivatives of aux vars (chain rule via their definitions)
  aux_rhs = _differentiate_aux_vars(aux_eqs, vars, lifted_rhs[1:length(vars)])

  # Append aux var derivatives (also need to be quadratized)
  for drhs in aux_rhs
    drhs_subst = _substitute_aux(drhs, aux_eqs)
    q, lifted_vars, aux_eqs =
      _quadratize_expr(drhs_subst, lifted_vars, aux_eqs, aux_counter)
    push!(lifted_rhs, q)
  end

  return LiftedSystem(vars, lifted_vars, lifted_rhs, aux_eqs)
end

"""
    quadratize(expr, vars) -> (new_expr, new_vars, aux_eqs)

Lower-level entry point: quadratize a single expression in vars.
"""
function quadratize(expr::Num, vars::Vector{Num})
  aux_eqs = Equation[]
  aux_counter = Ref(0)
  new_vars = copy(vars)
  new_expr, new_vars, aux_eqs =
    _quadratize_expr(expr, new_vars, aux_eqs, aux_counter)
  return new_expr, new_vars, aux_eqs
end

# Internal helpers

"""
Recursively reduce an expression to quadratic degree by introducing
auxiliary variables for monomials of degree > 2.
"""
function _quadratize_expr(expr::Num, vars::Vector{Num}, aux_eqs::Vector{Equation}, counter::Ref{Int})
  d = _degree(expr, vars)
  d <= 2 && return expr, vars, aux_eqs

  expanded = Symbolics.expand(expr)
  monomials = _split_sum(expanded)

  new_monomials = Num[]
  for mono in monomials
    q, vars, aux_eqs = _quadratize_monomial(mono, vars, aux_eqs, counter)
    push!(new_monomials, q)
  end

  new_expr = isempty(new_monomials) ? Num(0) : sum(new_monomials)
  return new_expr, vars, aux_eqs
end

"""
Quadratize a single monomial (product of vars) by repeatedly replacing
the two highest-degree factors with an auxiliary variable.
"""
function _quadratize_monomial(mono::Num, vars::Vector{Num}, aux_eqs::Vector{Equation}, counter::Ref{Int})
  _degree(mono, vars) <= 2 && return mono, vars, aux_eqs

  coeff, factors = _factorize_monomial(mono, vars)

  while length(factors) > 2
    f1, f2 = factors[1], factors[2]
    counter[] += 1
    w_sym = only(@variables $(Symbol("w$(counter[])")))
    push!(aux_eqs, w_sym ~ f1 * f2)
    push!(vars, w_sym)
    factors = [w_sym; factors[3:end]]
  end

  result = coeff * prod(factors)
  return result, vars, aux_eqs
end

"""
Compute time derivatives of auxiliary variables using the chain rule.

dwi/dt = sum_j (dwi/dxj) * xj_dot
"""
function _differentiate_aux_vars(aux_eqs::Vector{Equation}, orig_vars::Vector{Num}, orig_rhs::Vector{Num})
  aux_rhs = Num[]
  for eq in aux_eqs
    g = eq.rhs
    dg_dt = sum(Symbolics.derivative(g, xj) * xj_dot
                for (xj, xj_dot) in zip(orig_vars, orig_rhs))
    push!(aux_rhs, Symbolics.simplify(dg_dt))
  end
  return aux_rhs
end

# Utility functions

function _substitute_aux(expr::Num, aux_eqs::Vector{Equation})
  for eq in aux_eqs
    expr = Symbolics.substitute(expr, Dict(eq.rhs => eq.lhs))
  end

  return Symbolics.simplify(expr)
end

"""
Total polynomial degree of expr in vars. Returns a large value if not polynomial.
"""
function _degree(expr::Num, vars::Vector{Num})
  try
    expanded = Symbolics.expand(expr)
    monomials = _split_sum(expanded)

    return maximum(
      sum(Symbolics.degree(m, v) for v in vars; init=0)
      for m in monomials;
      init=0
    )
  catch
    return typemax(Int)
  end
end

"""
Split a sum expression into individual summand terms.
"""
function _split_sum(expr::Num)
  ex = Symbolics.unwrap(expr)
  if Symbolics.isadd(ex)
    return Num.(collect(keys(Symbolics.arguments(ex))))
  else
    return [expr]
  end
end

"""
Extract (coefficient, [factor1, factor2, ...]) from a monomial.
"""
function _factorize_monomial(mono::Num, vars::Vector{Num})
  factors = Num[]

  for v in vars
    d = Symbolics.degree(mono, v)

    for _ in 1:d
      push!(factors, v)
    end
  end

  if isempty(factors)
    return mono, Num[]
  end

  # coefficient = whatever is left after dividing out the variable part
  var_part = prod(v^Symbolics.degree(mono, v) for v in vars if Symbolics.degree(mono, v) > 0)
  coeff = Symbolics.simplify(mono / var_part)

  return coeff, factors
end

# Display

function Base.show(io::IO, ls::LiftedSystem)
  n_orig = length(ls.original_vars)
  n_lift = length(ls.lifted_vars)
  println(io, "LiftedSystem:")
  println(io, "  Original state dim : $n_orig")
  println(io, "  Lifted state dim   : $n_lift  ($(n_lift - n_orig) aux vars introduced)")
  println(io, "  Auxiliary variables:")
  for eq in ls.aux_eqs
    println(io, "    $(eq.lhs) = $(eq.rhs)")
  end
  println(io, "  Lifted RHS (should be <= quadratic):")
  for (v, f) in zip(ls.lifted_vars, ls.F)
    println(io, "    s_dot($(v)) = $f")
  end
end
