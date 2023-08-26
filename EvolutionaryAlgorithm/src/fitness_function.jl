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
    intersection::Float64 = 0
    if (individual.x[i] - individual.x[j])^2 + (individual.y[i] - individual.y[j])^2 < 2
        x::Float64 = (individual.x[i] - individual.x[j])
        y::Float64 = (individual.y[i] - individual.y[j])
        t::Float64 = individual.theta[j] - individual.theta[i]
        projected_x:: Vector{Float64} = x .+ 0.5 * [sin(t + π/4), sin(t - π/4), sin(t + 3π/4), sin(t - 3π/4)] * √(2)
        projected_y:: Vector{Float64} = y .+ 0.5 * [cos(t + π/4), cos(t - π/4), cos(t + 3π/4), cos(t - 3π/4)] * √(2)
        projected_x_:: Vector{Float64} = -x .+ 0.5 * [sin(-t + π/4), sin(-t - π/4), sin(-t + 3π/4), sin(-t - 3π/4)] * √(2)
        projected_y_:: Vector{Float64} = -y .+ 0.5 * [cos(-t + π/4), cos(-t - π/4), cos(-t + 3π/4), cos(-t - 3π/4)] * √(2)

        if Intersect(projected_x) && Intersect(projected_y) && Intersect(projected_x_) && Intersect(projected_y_)
            intersection += IntersectionArea(projected_x, projected_y)*IntersectionArea(projected_x_, projected_y_)
        end
    end
    return intersection
end

function Intersect(v::Vector{Float64})::Bool
    return ((minimum(v) < 0.5) && (maximum(v) > 0.5)) || ((minimum(v) < -0.5) && (maximum(v) > -0.5))
end

function IntersectionArea(x::Vector{Float64}, y::Vector{Float64})::Float64
    return (min(0.5, maximum(x)) - max(-0.5, minimum(x)))*(min(0.5, maximum(y)) - max(-0.5, minimum(y)))
end