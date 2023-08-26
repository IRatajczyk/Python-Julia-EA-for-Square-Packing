export CrossoverParameters, Crossover

include("population.jl")

struct CrossoverParameters
    crossover_rate::Float64
    n_cuts::Int64
end

function Crossover(parameters::CrossoverParameters)
    return function (population::Vector{Individual})
        offspring::Vector{Individual} = Individual[]
        sort!(population, by = x -> rand(Float64))
        for i in 1:length(population):2
            if rand(Float64) < parameters.crossover_rate
                offspring =vcat(offspring, Cross(population[i], population[i+1], parameters.n_cuts))
            end
        end
        return offspring
    end
end

function Cross(parent1::Individual, parent2::Individual, n_cuts::Int)::Vector{Individual}
    child1::Individual = InitializeEmptyIndividual(parent1.size)
    child2::Individual = InitializeEmptyIndividual(parent2.size)
    cuts = sort(rand(1:parent1.size, n_cuts), by = x -> -x)
    for cut in cuts
        child1.x[1:cut] = parent1.x[1:cut]
        child1.y[1:cut] = parent1.y[1:cut]
        child1.theta[1:cut] = parent1.theta[1:cut]
        child2.x[1:cut] = parent2.x[1:cut]
        child2.y[1:cut] = parent2.y[1:cut]
        child2.theta[1:cut] = parent2.theta[1:cut]
    end
    return [child1, child2]
end
