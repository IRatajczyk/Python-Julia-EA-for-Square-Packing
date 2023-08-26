export InitializeEmptyIndividual, Population, MarkChangedGenotype!, Individual, PopulationParameters

using Random

struct PopulationParameters
    problem_size::Int64
    initialize_feasible::Bool
    use_exp::Bool
    dispersion_factor::Float64
end

mutable struct Individual
    size::Int64
    x::Vector{Float64}
    y::Vector{Float64}
    theta::Vector{Float64}
    fitness::Float64
    has_changed::Bool
    is_feasible::Bool
end

function Population(parameters::PopulationParameters)::Function
    initializer::Function = IndividualInitializer(parameters)
    return function (population_size::Int64)
            population::Vector{Individual} = Vector{Individual}(undef, population_size)
            for i = 1:population_size
                population[i] = initializer()
            end
            return population
        end
end


function IndividualInitializer(parameters::PopulationParameters)::Function
    return function ()
        return Individual(
            parameters.problem_size, 
            (parameters.use_exp ? randexp(Float64, parameters.problem_size) : randn(Float64, parameters.problem_size)) * parameters.dispersion_factor,
            (parameters.use_exp ? randexp(Float64, parameters.problem_size) : randn(Float64, parameters.problem_size)) * parameters.dispersion_factor,
            rand(Float64, parameters.problem_size) * π * 2,
            Inf, 
            true,
            false)
        end
end

function InitializeEmptyIndividual(problem_size::Int64):: Individual
    return Individual(
        problem_size, 
        zeros(Float64, problem_size), 
        zeros(Float64, problem_size), 
        zeros(Float64, problem_size), 
        Inf, 
        true, 
        false
        )
end

function MarkChangedGenotype!(individual::Individual)
    individual.has_changed = true
    individual.is_feasible = false
    individual.fitness = Inf
end
