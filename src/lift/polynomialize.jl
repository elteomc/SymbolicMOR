# src/lift/polynomialize.jl
#
# Phase 1 - Polynomialization
#
# Goal: Rewrite a non-polynomial ODE RHS so that every term is polynomial in
# an extended set of state variables. Each non-polynomial atom (exp, sin/cos,
# 1/g, log, sqrt, tanh) is replaced by a fresh polynomial auxiliary whose
# time derivative is itself expressed (via the chain rule) in terms of the
# already-introduced atoms, recursively closing the system.
#
# After polynomialization, the resulting RHS is a polynomial in the extended
# state and can be handed to the quadratization step.
#
# References:
#   Kramer & Willcox (2019), "Nonlinear Model Reduction via Lift & Learn"
#   Bychkov & Pogudin (2021), "Optimal Monomial Quadratization for ODE Systems"
#     arXiv:2103.08013

using Symbolics

const SymbolicUtils = Symbolics.SymbolicUtils

# Public API

"""
    PolynomializationResult

Container for the polynomialized form of a non-polynomial ODE.

Fields:
  - original_vars   : original state variables [x1, ..., xn]
  - polynomial_vars : original vars plus polynomial auxiliaries [x1, ..., xn, p1, p2, ...]
  - polynomial_rhs  : RHS for each entry of polynomial_vars; polynomial in polynomial_vars
  - aux_defs        : aux definitions, e.g. p1 ~ exp(x), p2 ~ sin(y), ...
"""
struct PolynomializationResult
  original_vars::Vector{Num}
  polynomial_vars::Vector{Num}
  polynomial_rhs::Vector{Num}
  aux_defs::Vector{Equation}
end

"""
    polynomialize_system(vars, rhs; max_iterations = 50)

Rewrite a non-polynomial ODE x_dot = rhs(x) into an extended polynomial form
by introducing polynomial auxiliaries whose time derivatives are themselves
polynomial in the extended state. Returns a `PolynomializationResult`.

Supported non-polynomial atoms (with `g` polynomial in current vars):
  - `exp(g)`
  - `sin(g)` / `cos(g)`   (introduced together for closure)
  - `1/g`, `g^(-k)` for integer `k > 0`
  - `log(g)`, `sqrt(g)`, `tanh(g)`

`log` and `sqrt` introduce a `1/g` (or `1/p`) term in the derivative; the loop
detects that on a later iteration and adds the corresponding inverse aux.
"""
function polynomialize_system(vars::AbstractVector, rhs::AbstractVector; max_iterations::Int = 50)
  length(vars) == length(rhs) ||
    throw(ArgumentError("vars and rhs must have the same length"))

  poly_vars = Num.(collect(vars))
  poly_rhs  = Num.(collect(rhs))
  aux_defs  = Equation[]
  aux_map   = Dict{Tuple{Symbol,String}, Num}()

  for _ in 1:max_iterations
    poly_rhs = [_expanded(_replace_known_auxiliaries(f, aux_map)) for f in poly_rhs]
    if all(_is_polynomial_in(f, poly_vars) for f in poly_rhs)
      return PolynomializationResult(Num.(collect(vars)), poly_vars, poly_rhs, aux_defs)
    end

    atom = nothing
    for f in poly_rhs
      atom = _first_polynomializable_atom(f, aux_map, poly_vars)
      atom === nothing || break
    end

    atom === nothing && throw(ErrorException(
      "polynomialization failed: encountered an unsupported non-polynomial term. " *
      "Supported atoms are exp(g), sin(g), cos(g), 1/g, log(g), sqrt(g), tanh(g) " *
      "with g polynomial in the current variables."
    ))

    _add_polynomialization_auxiliary!(poly_vars, poly_rhs, aux_defs, aux_map, atom)
  end

  throw(ErrorException("polynomialization did not close after $max_iterations iterations"))
end

# Internals

_canonical_expr(x) = Symbolics.unwrap(Symbolics.value(x))
_expanded(x) = Num(Symbolics.expand(x))

_expression_key(expr) = string(_canonical_expr(Symbolics.expand(expr)))
_aux_key(kind::Symbol, arg) = (kind, _expression_key(arg))

# Extract a Julia number from a possibly-wrapped symbolic literal. SymbolicUtils
# 4.x represents numeric constants inside expanded expressions as a leaf
# BasicSymbolic with a `.val` field, not as a plain `Number`. Returns the
# underlying number if `y` is numeric (plain or wrapped), else `nothing`.
function _as_number(y)
  y isa Number && return y
  if !SymbolicUtils.iscall(y) && !SymbolicUtils.issym(y) && hasproperty(y, :val)
    v = y.val
    v isa Number && return v
  end
  return nothing
end

_is_number_like(y) = _as_number(y) !== nothing

"""
Strict polynomial check. True iff `expr` is built only from constants, the
listed `vars`, the operations +, -, *, division by a numeric constant, and
non-negative integer powers of polynomial subexpressions. Anything else
(transcendental functions, unknown symbols, symbolic divisions, fractional or
negative powers) returns false.
"""
function _is_polynomial_in(expr, vars)
  y = _canonical_expr(expr)
  var_set = Set{Any}(_canonical_expr(v) for v in vars)
  return _walk_is_polynomial(y, var_set)
end

function _walk_is_polynomial(y, var_set)
  y in var_set && return true
  _is_number_like(y) && return true
  SymbolicUtils.iscall(y) || return false  # symbolic atom not in our var set

  op = SymbolicUtils.operation(y)
  args = SymbolicUtils.arguments(y)

  if op === (+) || op === (*) || op === (-)
    return all(_walk_is_polynomial(a, var_set) for a in args)
  elseif op === (/)
    length(args) == 2 || return false
    n = _as_number(_canonical_expr(args[2]))
    n === nothing && return false
    return _walk_is_polynomial(args[1], var_set)
  elseif op === (^)
    length(args) == 2 || return false
    n = _as_number(_canonical_expr(args[2]))
    (n === nothing || !(n isa Integer) || n < 0) && return false
    return _walk_is_polynomial(args[1], var_set)
  end

  return false
end

# Reconstruct an expression from its operation head and rewritten args.
function _reconstruct(op, args)
  if op === (+); return +(args...)
  elseif op === (*); return *(args...)
  elseif op === (-); return length(args) == 1 ? -args[1] : -(args...)
  elseif op === (/); return /(args...)
  elseif op === (^); return args[1]^args[2]
  else; return op(args...)
  end
end

# Replace any subexpression matching a known atom (sin, exp, 1/g, ...) with
# its polynomial auxiliary. Walks bottom-up.
function _replace_known_auxiliaries(expr, aux_map)
  y = _canonical_expr(expr)
  SymbolicUtils.iscall(y) || return expr

  op = SymbolicUtils.operation(y)
  args = collect(SymbolicUtils.arguments(y))
  rewritten = [_replace_known_auxiliaries(a, aux_map) for a in args]

  if op === exp && length(rewritten) == 1
    k = _aux_key(:exp, rewritten[1])
    haskey(aux_map, k) && return aux_map[k]
  elseif op === sin && length(rewritten) == 1
    k = _aux_key(:sin, rewritten[1])
    haskey(aux_map, k) && return aux_map[k]
  elseif op === cos && length(rewritten) == 1
    k = _aux_key(:cos, rewritten[1])
    haskey(aux_map, k) && return aux_map[k]
  elseif op === log && length(rewritten) == 1
    k = _aux_key(:log, rewritten[1])
    haskey(aux_map, k) && return aux_map[k]
  elseif op === sqrt && length(rewritten) == 1
    k = _aux_key(:sqrt, rewritten[1])
    haskey(aux_map, k) && return aux_map[k]
  elseif op === tanh && length(rewritten) == 1
    k = _aux_key(:tanh, rewritten[1])
    haskey(aux_map, k) && return aux_map[k]
  elseif op === (/) && length(rewritten) == 2
    num, den = rewritten
    _is_number_like(_canonical_expr(den)) && return _reconstruct(op, rewritten)
    k = _aux_key(:inv, den)
    haskey(aux_map, k) && return num * aux_map[k]
  elseif op === (^) && length(rewritten) == 2
    base, power = rewritten
    pv = _as_number(_canonical_expr(power))
    if pv isa Integer && pv < 0
      k = _aux_key(:inv, base)
      haskey(aux_map, k) && return aux_map[k]^(-pv)
    end
  end

  return _reconstruct(op, rewritten)
end

# Walk inside-out and return the innermost non-polynomial atom whose argument
# is itself polynomial in `vars`. That is the right next aux to introduce.
function _first_polynomializable_atom(expr, aux_map, vars)
  y = _canonical_expr(expr)
  SymbolicUtils.iscall(y) || return nothing

  op = SymbolicUtils.operation(y)
  args = collect(SymbolicUtils.arguments(y))

  for a in args
    found = _first_polynomializable_atom(_replace_known_auxiliaries(a, aux_map), aux_map, vars)
    found === nothing || return found
  end

  rewritten = [_replace_known_auxiliaries(a, aux_map) for a in args]

  if op === exp && length(rewritten) == 1
    arg = rewritten[1]
    _is_polynomial_in(arg, vars) && return (:exp, arg)
  elseif (op === sin || op === cos) && length(rewritten) == 1
    arg = rewritten[1]
    _is_polynomial_in(arg, vars) && return (:sincos, arg)
  elseif op === log && length(rewritten) == 1
    arg = rewritten[1]
    _is_polynomial_in(arg, vars) && return (:log, arg)
  elseif op === sqrt && length(rewritten) == 1
    arg = rewritten[1]
    _is_polynomial_in(arg, vars) && return (:sqrt, arg)
  elseif op === tanh && length(rewritten) == 1
    arg = rewritten[1]
    _is_polynomial_in(arg, vars) && return (:tanh, arg)
  elseif op === (/) && length(rewritten) == 2
    den = rewritten[2]
    _is_number_like(_canonical_expr(den)) && return nothing
    _is_polynomial_in(den, vars) && return (:inv, den)
  elseif op === (^) && length(rewritten) == 2
    base, power = rewritten
    pv = _as_number(_canonical_expr(power))
    if pv isa Integer && pv < 0 && _is_polynomial_in(base, vars)
      return (:inv, base)
    end
  end

  return nothing
end

# Total time derivative of `expr` along the dynamics rhs.
function _total_derivative(expr, vars, rhs)
  out = Num(0)
  for (v, f) in zip(vars, rhs)
    out = out + Symbolics.derivative(expr, v) * f
  end
  return Symbolics.expand(out)
end

# Allocate a fresh polynomial auxiliary p_{n_existing+1}.
_fresh_poly_var(n_existing::Int) = Symbolics.variable(Symbol(:p, n_existing + 1))

# Add the appropriate aux to close the dynamics for `atom`. Mutates poly_vars,
# poly_rhs, aux_defs, and aux_map.
function _add_polynomialization_auxiliary!(poly_vars::Vector{Num}, poly_rhs::Vector{Num},
                                            aux_defs::Vector{Equation}, aux_map::Dict, atom)
  kind, arg = atom
  arg = _replace_known_auxiliaries(arg, aux_map)
  gdot = _total_derivative(arg, poly_vars, poly_rhs)

  if kind === :exp
    k = _aux_key(:exp, arg)
    haskey(aux_map, k) && return
    p = _fresh_poly_var(length(aux_defs))
    aux_map[k] = p
    push!(aux_defs, p ~ exp(arg))
    push!(poly_vars, p)
    push!(poly_rhs, _expanded(p * gdot))

  elseif kind === :sincos
    sk = _aux_key(:sin, arg)
    ck = _aux_key(:cos, arg)
    haskey(aux_map, sk) && haskey(aux_map, ck) && return
    s = _fresh_poly_var(length(aux_defs))
    c = _fresh_poly_var(length(aux_defs) + 1)
    aux_map[sk] = s
    aux_map[ck] = c
    push!(aux_defs, s ~ sin(arg))
    push!(aux_defs, c ~ cos(arg))
    push!(poly_vars, s)
    push!(poly_vars, c)
    push!(poly_rhs, _expanded(c * gdot))
    push!(poly_rhs, _expanded(-s * gdot))

  elseif kind === :inv
    k = _aux_key(:inv, arg)
    haskey(aux_map, k) && return
    p = _fresh_poly_var(length(aux_defs))
    aux_map[k] = p
    push!(aux_defs, p ~ 1 / arg)
    push!(poly_vars, p)
    push!(poly_rhs, _expanded(-p^2 * gdot))

  elseif kind === :log
    k = _aux_key(:log, arg)
    haskey(aux_map, k) && return
    p = _fresh_poly_var(length(aux_defs))
    aux_map[k] = p
    push!(aux_defs, p ~ log(arg))
    push!(poly_vars, p)
    # d/dt log(g) = (1/g) * gdot. The (1/g) becomes a fresh inv aux on a later iteration.
    push!(poly_rhs, _expanded((1 / arg) * gdot))

  elseif kind === :sqrt
    k = _aux_key(:sqrt, arg)
    haskey(aux_map, k) && return
    p = _fresh_poly_var(length(aux_defs))
    aux_map[k] = p
    push!(aux_defs, p ~ sqrt(arg))
    push!(poly_vars, p)
    # d/dt sqrt(g) = (1/(2*sqrt(g))) * gdot = (1/(2*p)) * gdot. The (1/p) is polynomialized later.
    push!(poly_rhs, _expanded((1 / (2 * p)) * gdot))

  elseif kind === :tanh
    k = _aux_key(:tanh, arg)
    haskey(aux_map, k) && return
    p = _fresh_poly_var(length(aux_defs))
    aux_map[k] = p
    push!(aux_defs, p ~ tanh(arg))
    push!(poly_vars, p)
    # d/dt tanh(g) = (1 - tanh(g)^2) * gdot, polynomial in p.
    push!(poly_rhs, _expanded((1 - p^2) * gdot))

  else
    error("unsupported polynomialization atom kind: $kind")
  end

  return nothing
end
