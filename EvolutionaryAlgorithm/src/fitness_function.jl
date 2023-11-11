export FitnessFunctionParameters, FitnessFunction

include("population.jl")

struct FitnessFunctionParameters
    use_penalty::Bool
    center_solution::Bool
    penalty_factor::Float64
end

function FitnessFunction(parameters::FitnessFunctionParameters)::Function
    return function (population::Vector{Individual})
        for individual in population
            if individual.has_changed
                if parameters.center_solution
                    individual.x = individual.x .- mean(individual.x)
                    individual.y = individual.y .- mean(individual.y)
                    individual.theta = individual.theta .- mean(individual.theta)
                end
                penalty::Float64 = Penalty(individual)
                individual.fitness = Evaluate(individual) + (parameters.use_penalty ? parameters.penalty_factor * penalty : 0)	
                individual.has_changed = false
                individual.is_feasible = penalty < 1e-6
            end
        end
        return population
    end
end

function Evaluate(individual::Individual)::Float64
    c::Float64 = 0.5*(√(2)-1)
    top::Float64 = maximum(individual.y)
    bottom::Float64 = minimum(individual.y)
    left::Float64 = minimum(individual.x)
    right::Float64 = maximum(individual.x)
    t::Float64 = 0
    for i in 1:individual.size
        if individual.x[i] < left + c
            t = individual.theta[i]
            left = min(left, minimum(individual.x[i] .+ 0.5 * [sin(t + π/4), sin(t - π/4), sin(t + 3π/4), sin(t - 3π/4)] * √(2)))
        end
        if individual.x[i] > right - c
            t = individual.theta[i]
            right = max(right, maximum(individual.x[i] .+ 0.5 * [sin(t + π/4), sin(t - π/4), sin(t + 3π/4), sin(t - 3π/4)] * √(2)))
        end
        if individual.y[i] < bottom + c
            t = individual.theta[i]
            bottom = min(bottom, minimum(individual.y[i] .+ 0.5 * [cos(t + π/4), cos(t - π/4), cos(t + 3π/4), cos(t - 3π/4)] * √(2)))
        end
        if individual.y[i] > top - c
            t = individual.theta[i]
            top = max(top, maximum(individual.y[i] .+ 0.5 * [cos(t + π/4), cos(t - π/4), cos(t + 3π/4), cos(t - 3π/4)] * √(2)))
        end
    end
    return max(top - bottom, right - left)
end

function Penalty(individual::Individual)::Float64
    penalty::Float64 = 0
    for i = 1:individual.size
        for j = 1:individual.size
            if i != j
                penalty += Intersection(individual, i, j)
            end
        end
    end
    return penalty
end

function Intersection(individual::Individual, i::Int64, j::Int64)::Float64
    λ::Float64 = .5*√(2)
    intersection::Float64 = 0
    if (individual.x[i] - individual.x[j])^2 + (individual.y[i] - individual.y[j])^2 < 2
        x::Float64 = (individual.x[i] - individual.x[j])
        y::Float64 = (individual.y[i] - individual.y[j])

        θ::Float64 = individual.theta[i]
        ω::Float64 = individual.theta[j]

        Px::Vector{Float64} = x .+ λ * [sin(θ + π/4), sin(θ - π/4), sin(θ + 3π/4), sin(θ - 3π/4)]
        Py::Vector{Float64} = y .+ λ * [cos(θ + π/4), cos(θ - π/4), cos(θ + 3π/4), cos(θ - 3π/4)]
        Pxy::Matrix{Float64} = R(ω) * [Px'; Py']

        
        Rx::Vector{Float64} = -x .+ λ * [sin(ω + π/4), sin(ω - π/4), sin(ω + 3π/4), sin(ω - 3π/4)]
        Ry::Vector{Float64} = -y .+ λ * [cos(ω + π/4), cos(ω - π/4), cos(ω + 3π/4), cos(ω - 3π/4)]
        Rxy::Matrix{Float64} = R(θ) * [Rx'; Ry']

        if Intersect(Pxy[1,:]) && Intersect(Pxy[2,:]) && Intersect(Rxy[1,:]) && Intersect(Rxy[2,:])
            intersection += IntersectionArea(Pxy) * IntersectionArea(Rxy)
        end
    end
    return intersection
end

function R(θ::Float64)::Matrix{Float64}
    [[cos(θ) -sin(θ)]; [sin(θ) cos(θ)]]
end

function Intersect(v::Vector{Float64})::Bool
    return ((minimum(v) < 0.5) && (maximum(v) > 0.5)) || ((minimum(v) < -0.5) && (maximum(v) > -0.5))
end

function IntersectionArea(u::Matrix{Float64})::Float64
    x::Vector{Float64} = u[1,:]
    y::Vector{Float64} = u[2,:]
    return (min(0.5, maximum(x)) - max(-0.5, minimum(x)))*(min(0.5, maximum(y)) - max(-0.5, minimum(y)))
end