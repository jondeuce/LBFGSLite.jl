module LBFGSLite

using LinearAlgebra: dot, norm

export LBFGSParams, LBFGSWorkspace
export optimize, optimize!

@enum LBFGS_STATUS::Int begin
    # L-BFGS reaches convergence.
    LBFGS_CONVERGENCE = 0
    # L-BFGS satisfies stopping criteria.
    LBFGS_STOP
    # The iteration has been canceled by the monitor callback.
    LBFGS_CANCELLED
    # L-BFGS line search returned successfully.
    LBFGS_LINESEARCH_SUCCESS

    # Unknown error.
    LBFGSERR_UNKNOWNERROR = -1024
    # Invalid number of variables specified.
    LBFGSERR_INVALID_N
    # Invalid parameter LBFGSParams.mem_size specified.
    LBFGSERR_INVALID_MEMSIZE
    # Invalid parameter LBFGSParams.g_epsilon specified.
    LBFGSERR_INVALID_GEPSILON
    # Invalid parameter LBFGSParams.past specified.
    LBFGSERR_INVALID_TESTPERIOD
    # Invalid parameter LBFGSParams.delta specified.
    LBFGSERR_INVALID_DELTA
    # Invalid parameter LBFGSParams.min_step specified.
    LBFGSERR_INVALID_MINSTEP
    # Invalid parameter LBFGSParams.max_step specified.
    LBFGSERR_INVALID_MAXSTEP
    # Invalid parameter LBFGSParams.f_dec_coeff specified.
    LBFGSERR_INVALID_FDECCOEFF
    # Invalid parameter LBFGSParams.s_curv_coeff specified.
    LBFGSERR_INVALID_SCURVCOEFF
    # Invalid parameter LBFGSParams.max_linesearch specified.
    LBFGSERR_INVALID_MAXLINESEARCH
    # The function value became NaN or Inf.
    LBFGSERR_INVALID_FUNCVAL
    # The line-search step became smaller than LBFGSParams.min_step.
    LBFGSERR_MINIMUMSTEP
    # The line-search step became larger than LBFGSParams.max_step.
    LBFGSERR_MAXIMUMSTEP
    # Line search reaches the maximum, assumptions not satisfied or precision not achievable.
    LBFGSERR_MAXIMUMLINESEARCH
    # The algorithm routine reaches the maximum number of iterations.
    LBFGSERR_MAXIMUMITERATION
    # Relative search interval width is at least LBFGSParams.machine_prec.
    LBFGSERR_WIDTHTOOSMALL
    # A logic error (negative line-search step) occurred.
    LBFGSERR_INVALIDPARAMETERS
    # The current search direction increases the cost function value.
    LBFGSERR_INCREASEGRADIENT
    # Array dimensions do not match.
    LBFGSERR_DIMENSIONMISMATCH
end

"""
    struct LBFGSParams{T <: AbstractFloat}

Parameter struct for LBFGS.

### Arguments

  - `mem_size::Int = 8`:
    The number of corrections to approximate the inverse hessian matrix.
    The L-BFGS routine stores the computation results of previous m iterations to approximate the inverse hessian matrix of the current iteration.
    This parameter controls the size of the limited memories (corrections).
    Values less than 3 are not recommended.
    Large values will result in excessive computing time.
  - `g_epsilon::T = 1.0e-5`:
    Epsilon for grad convergence test.
    DO NOT USE IT in nonsmooth cases!
    Set it to 0.0 and use past-delta-based test for nonsmooth functions.
    This parameter determines the accuracy with which the solution is to be found.
    A minimization terminates when ||g(x)||_∞ / max(1, ||x||_∞) < g_epsilon, where ||.||_∞ is the infinity norm.
    This should be greater than 1.0e-6 in practice because L-BFGS does not directly reduce first-order residual.
    It still needs the function value which can be corrupted by machine_prec when ||g|| is small.
  - `past::Int = 3`:
    Distance for delta-based convergence test.
    This parameter determines the distance, in iterations, to compute the rate of decrease of the cost function.
    If the value of this parameter is zero, the library does not perform the delta-based convergence test.
  - `delta::T = 1.0e-6`:
    Delta for convergence test.
    This parameter determines the minimum rate of decrease of the cost function.
    The library stops iterations when the following condition is met: |f' - f| / max(1, |f|) < delta, where f' is the cost value of past iterations ago, and f is the cost value of the current iteration.
  - `max_iterations::Int = 0`:
    The maximum number of iterations.
    The `optimize` function terminates an minimization process with `st::LBFGSERR_MAXIMUMITERATION` status code when the iteration count exceedes this parameter.
    Setting this parameter to zero continues a minimization process until a convergence or error.
  - `max_linesearch::Int = 64`:
    The maximum number of trials for the line search.
    This parameter controls the number of function and gradients evaluations per iteration for the line search routine.
  - `min_step::T = 1.0e-20`:
    The minimum step of the line search routine.
    This value need not be modified unless the exponents are too large for the machine being used, or unless the problem is extremely badly scaled (in which case the exponents should be increased).
  - `max_step::T = 1.0e+20`:
    The maximum step of the line search.
    This value need not be modified unless the exponents are too large for the machine being used, or unless the problem is extremely badly scaled (in which case the exponents should be increased).
  - `f_dec_coeff::T = 1.0e-4`:
    A parameter to control the accuracy of the line search routine.
    This parameter should be greater than zero and smaller than 1.0.
  - `s_curv_coeff::T = 0.9`:
    A parameter to control the accuracy of the line search routine.
    If the function and gradient evaluations are inexpensive with respect to the cost of the iteration (which is sometimes the case when solving very large problems) it may be advantageous to set this parameter to a small value.
    A typical small value is 0.1.
    This parameter should be greater than the f_dec_coeff parameter and smaller than 1.0.
  - `cautious_factor::T = 1.0e-6`:
    A parameter to ensure the global convergence for nonconvex functions.
    The parameter performs the so called cautious update for L-BFGS, especially when the convergence is not sufficient.
    The parameter must be positive but might as well be less than 1.0e-3 in practice.
"""
Base.@kwdef struct LBFGSParams{T <: AbstractFloat}
    mem_size::Int = 8
    g_epsilon::T = T(1e-5)
    past::Int = 3
    delta::T = T(1e-6)
    max_iterations::Int = 0
    max_linesearch::Int = 64
    min_step::T = T(1e-20)
    max_step::T = T(1e+20)
    f_dec_coeff::T = T(1e-4)
    s_curv_coeff::T = T(0.9)
    cautious_factor::T = T(1e-6)
end

struct LBFGSWorkspace{T <: AbstractFloat}
    # Intermediate variables
    xp::Vector{T}
    g::Vector{T}
    gp::Vector{T}
    d::Vector{T}
    pf::Vector{T}

    # Limited memory
    lm_alpha::Vector{T}
    lm_s::Matrix{T}
    lm_y::Matrix{T}
    lm_ys::Vector{T}
end

function LBFGSWorkspace(x::AbstractVector{T}, params::LBFGSParams{T}) where {T <: AbstractFloat}
    m = params.mem_size
    n = length(x)

    # Intermediate variables
    xp = zeros(T, n)
    g  = zeros(T, n)
    gp = zeros(T, n)
    d  = zeros(T, n)
    pf = zeros(T, max(1, params.past))

    # Limited memory
    lm_alpha = zeros(T, m)
    lm_s = zeros(T, n, m)
    lm_y = zeros(T, n, m)
    lm_ys = zeros(T, m)

    return LBFGSWorkspace(xp, g, gp, d, pf, lm_alpha, lm_s, lm_y, lm_ys)
end

"""
    optimize(
        fg!::F
        x::AbstractVector{T},
        params::LBFGSParams{T} = LBFGSParams{T}(),
    ) = optimize!(fg!, copy(x), LBFGSWorkspace(x, params), params)

    optimize!(
        fg!::F
        x::AbstractVector{T},
        work::LBFGSWorkspace{T},
        params::LBFGSParams{T} = LBFGSParams{T}(),
    ) -> x, fx::T, st::LBFGS_STATUS

Minimize a function using L-BFGS.

Assumptions:

  - f(x) is either C2 or C0 but piecewise C2;
  - f(x) is lower bounded;
  - f(x) has bounded level sets;
  - g(x) is either the gradient or subgradient;
  - The gradient exists at the initial guess x0.

### Arguments

  - `fg!(x, g)::F`: In-place function that computes the objective fx = fg!(x, g) and stores the gradient in g.
  - `x::AbstractVector{T}`: Vector of decision variables. Used for the initial guess.
  - `work::LBFGSWorkspace{T}`: Struct with pre-allocated arrays.
  - `params::LBFGSParams{T} = LBFGSParams{T}()`: The parameters for L-BFGS optimization.

### Returns

  - `x::AbstractVector{T}`: Vector of decision variables.
  - `fx::T`: Final value of cost function.
  - `st::LBFGS_STATUS`: Status code.
"""
function optimize(fg!, x, params = LBFGSParams{eltype(x)}())
    return optimize!(fg!, copy(x), LBFGSWorkspace(x, params), params)
end

function optimize!(
    fg!::F,
    x::AbstractVector{T},
    work::LBFGSWorkspace{T},
    params::LBFGSParams{T} = LBFGSParams{T}(),
) where {F, T}
    # Check the input parameters for errors.
    if length(x) <= 0
        return x, -one(T), LBFGSERR_INVALID_N
    end

    if params.mem_size <= 0
        return x, -one(T), LBFGSERR_INVALID_MEMSIZE
    end

    if params.g_epsilon < 0
        return x, -one(T), LBFGSERR_INVALID_GEPSILON
    end

    if params.past < 0
        return x, -one(T), LBFGSERR_INVALID_TESTPERIOD
    end

    if params.delta < 0
        return x, -one(T), LBFGSERR_INVALID_DELTA
    end

    if params.min_step < 0
        return x, -one(T), LBFGSERR_INVALID_MINSTEP
    end

    if params.max_step < params.min_step
        return x, -one(T), LBFGSERR_INVALID_MAXSTEP
    end

    if !(params.f_dec_coeff > 0 && params.f_dec_coeff < 1)
        return x, -one(T), LBFGSERR_INVALID_FDECCOEFF
    end

    if !(params.s_curv_coeff < 1 && params.s_curv_coeff > params.f_dec_coeff)
        return x, -one(T), LBFGSERR_INVALID_SCURVCOEFF
    end

    if params.max_linesearch <= 0
        return x, -one(T), LBFGSERR_INVALID_MAXLINESEARCH
    end

    if size(work.xp) != size(x)
        return x, -one(T), LBFGSERR_DIMENSIONMISMATCH
    end

    if size(work.g) != size(x)
        return x, -one(T), LBFGSERR_DIMENSIONMISMATCH
    end

    if size(work.gp) != size(x)
        return x, -one(T), LBFGSERR_DIMENSIONMISMATCH
    end

    if size(work.d) != size(x)
        return x, -one(T), LBFGSERR_DIMENSIONMISMATCH
    end

    if length(work.pf) < params.past
        return x, -one(T), LBFGSERR_INVALID_TESTPERIOD
    end

    if length(work.lm_alpha) != params.mem_size
        return x, -one(T), LBFGSERR_INVALID_MEMSIZE
    end

    if size(work.lm_s)[end] != params.mem_size
        return x, -one(T), LBFGSERR_INVALID_MEMSIZE
    end

    if size(work.lm_s)[1:end-1] != size(x)
        return x, -one(T), LBFGSERR_DIMENSIONMISMATCH
    end

    if size(work.lm_y)[end] != params.mem_size
        return x, -one(T), LBFGSERR_INVALID_MEMSIZE
    end

    if size(work.lm_y)[1:end-1] != size(x)
        return x, -one(T), LBFGSERR_DIMENSIONMISMATCH
    end

    if length(work.lm_ys) != params.mem_size
        return x, -one(T), LBFGSERR_INVALID_MEMSIZE
    end

    return _optimize!(fg!, x, work, params)
end

function _optimize!(
    fg!::F,
    x::AbstractVector{T},
    work::LBFGSWorkspace{T},
    params::LBFGSParams{T},
) where {F, T}
    # Input parameters.
    m = params.mem_size
    ϵg = params.g_epsilon
    local st::LBFGS_STATUS

    # Intermediate variables.
    xp, g, gp, d, pf = work.xp, work.g, work.gp, work.d, work.pf

    # Limited memory.
    lm_alpha, lm_s, lm_y, lm_ys = work.lm_alpha, work.lm_s, work.lm_y, work.lm_ys

    fill!(lm_alpha, 0)
    fill!(lm_s, 0)
    fill!(lm_y, 0)
    fill!(lm_ys, 0)

    # Evaluate the function value and its gradient.
    fx = fg!(x, g)

    # Store the initial value of the cost function.
    @inbounds pf[1] = fx

    # Compute the direction. We assume the initial hessian matrix H_0 is the identity matrix.
    @. d = -g

    # Make sure that the initial variables are not a stationary point.
    gnorm_inf = norm(g, Inf)
    xnorm_inf = norm(x, Inf)

    if gnorm_inf <= ϵg * max(one(T), xnorm_inf)
        # The initial guess is already a stationary point.
        st = LBFGS_CONVERGENCE
    else
        # Compute the initial step:
        step = one(T) / norm(d)

        iter = 1
        last = 1
        bound = 0

        @inbounds while true
            # Store the current position and gradient vectors.
            copyto!(xp, x)
            copyto!(gp, g)

            # If the step bound can be provided dynamically, then apply it.
            step = step < params.max_step ? step : params.max_step / 2

            # Search for an optimal step.
            fx, step, st = line_search_lewisoverton!(fg!, x, fx, g, step, d, xp, gp, params)

            if Int(st) < 0
                # Revert to the previous point.
                copyto!(x, xp)
                copyto!(g, gp)
                break
            end

            # Convergence test.
            # The criterion is given by the following formula:
            #   ||g(x)||_∞ / max(1, ||x||_∞) < g_epsilon.
            gnorm_inf = norm(g, Inf)
            xnorm_inf = norm(x, Inf)
            if gnorm_inf < ϵg * max(one(T), xnorm_inf)
                # Convergence.
                st = LBFGS_CONVERGENCE
                break
            end

            # Test for stopping criterion.
            # The criterion is given by the following formula:
            #   |f(past_x) - f(x)| / max(1, |f(x)|) < delta.
            if params.past > 0
                # We don't test the stopping criterion while iter < past.
                if params.past <= iter
                    # The stopping criterion.
                    rate = abs(pf[mod1(iter + 1, params.past)] - fx) / max(one(T), abs(fx))
                    if rate < params.delta
                        st = LBFGS_STOP
                        break
                    end
                end

                # Store the current value of the cost function.
                pf[mod1(iter + 1, params.past)] = fx
            end

            if params.max_iterations != 0 && params.max_iterations <= iter
                # Maximum number of iterations.
                st = LBFGSERR_MAXIMUMITERATION
                break
            end

            # Count the iteration number.
            iter += 1

            # Update vectors s and y:
            #   s_{k+1} = x_{k+1} - x_{k} = \step * d_{k}.
            #   y_{k+1} = g_{k+1} - g_{k}.
            @views @. lm_s[:, last] = x - xp
            @views @. lm_y[:, last] = g - gp

            # Compute scalars ys and yy:
            #   ys = y^t \cdot s = 1 / \rho.
            #   yy = y^t \cdot y.
            # Notice that yy is used for scaling the hessian matrix H_0 (Cholesky factor).
            ys = @views dot(lm_y[:, last], lm_s[:, last])
            yy = @views norm(lm_y[:, last])^2
            lm_ys[last] = ys

            # Compute the negative of gradients.
            @. d = -g

            # Only cautious update is performed here as long as
            #   (y^t \cdot s) / ||s_{k+1}||^2 > \epsilon * ||g_{k}||^\alpha,
            # where \epsilon is the cautious factor and a proposed value for \alpha is 1.
            # This is not for enforcing the PD of the approxomated Hessian
            # since ys > 0 is already ensured by the weak Wolfe condition.
            # This is to ensure the global convergence as described in:
            # Dong-Hui Li and Masao Fukushima. On the global convergence of the BFGS method for nonconvex unconstrained optimization problems. SIAM Journal on Optimization, Vol 11, No 4, pp. 1054-1064, 2011.
            cau = @views norm(lm_s[:, last])^2 * norm(gp) * params.cautious_factor

            if ys > cau
                # Recursive formula to compute dir = -(H \cdot g).
                # This is described in page 779 of:
                # Jorge Nocedal. Updating Quasi-Newton Matrices with Limited Storage. Mathematics of Computation, Vol. 35, No. 151, pp. 773--782, 1980.
                bound += 1
                bound = min(m, bound)
                last = mod1(last + 1, m)

                j = last
                for _ in 1:bound
                    # if (--j == -1) j = m-1;
                    j = mod1(j - 1, m)
                    # \alpha_{j} = \rho_{j} s^{t}_{j} \cdot q_{k+1}.
                    alpha = @views dot(lm_s[:, j], d) / lm_ys[j]
                    # q_{i} = q_{i+1} - \alpha_{i} y_{i}.
                    @views @. d -= alpha * lm_y[:, j]
                    lm_alpha[j] = alpha
                end

                @. d *= ys / yy

                for _ in 1:bound
                    # \beta_{j} = \rho_{j} y^t_{j} \cdot \gamm_{i}.
                    beta = @views dot(lm_y[:, j], d) / lm_ys[j]
                    # \gamm_{i+1} = \gamm_{i} + (\alpha_{j} - \beta_{j}) s_{j}.
                    @views @. d += (lm_alpha[j] - beta) * lm_s[:, j]
                    # if (++j == m) j = 0
                    j = mod1(j + 1, m)
                end
            end

            # The search direction d is ready. We try step = 1 first.
            step = one(T)
        end
    end

    return x, fx, st
end

function line_search_lewisoverton!(
    fg!::F,
    x::AbstractVector{T},
    fx::T,
    g::AbstractVector{T},
    step::T,
    s::AbstractVector{T},
    xp::AbstractVector{T},
    gp::AbstractVector{T},
    params::LBFGSParams{T},
) where {F, T}
    count = 0
    brackt = false
    touched = false
    mu = zero(T)
    nu = params.max_step
    local st::LBFGS_STATUS

    if !(step > 0)
        st = LBFGSERR_INVALIDPARAMETERS
        return fx, step, st
    end

    # Compute the initial gradient in the search direction.
    dginit = dot(gp, s)

    # Make sure that s points to a descent direction.
    if dginit > 0
        st = LBFGSERR_INCREASEGRADIENT
        return fx, step, st
    end

    # The initial value of the cost function.
    finit = fx
    dgtest = params.f_dec_coeff * dginit
    dstest = params.s_curv_coeff * dginit

    @inbounds while true
        @. x = xp + step * s

        # Evaluate the function and gradient values.
        fx = fg!(x, g)
        count += 1

        # Test for errors.
        if !isfinite(fx)
            st = LBFGSERR_INVALID_FUNCVAL
            break
        end

        # Check the Armijo condition.
        if fx > finit + step * dgtest
            nu = step
            brackt = true
        else
            # Check the weak Wolfe condition.
            if dot(g, s) < dstest
                mu = step
            else
                st = LBFGS_LINESEARCH_SUCCESS
                break
            end
        end

        if count > params.max_linesearch
            # Maximum number of iteration.
            st = LBFGSERR_MAXIMUMLINESEARCH
            break
        end

        if brackt && (nu - mu) < eps(T) * nu
            # Relative interval width is at least machine_prec.
            st = LBFGSERR_WIDTHTOOSMALL
            break
        end

        if brackt
            step = (mu + nu) / 2
        else
            step *= 2
        end

        if step < params.min_step
            # The step is the minimum value.
            st = LBFGSERR_MINIMUMSTEP
            break
        end

        if step > params.max_step
            if touched
                # The step is the maximum value.
                st = LBFGSERR_MAXIMUMSTEP
                break
            else
                # The maximum value should be tried once.
                touched = true
                step = params.max_step
            end
        end
    end

    return fx, step, st
end

end # module LBFGSLite
