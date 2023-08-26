export MutationParameters, Mutation

include("population.jl")

struct MutationParameters
    mutation_rate::Float64
    std_dev_spatial::Float64
    std_dev_angular::Float64
    transposition_rate::Float64
    allow_transpose::Bool
    fraction_transposed::Float64
end


function Mutation(parameters::MutationParameters)::Function
    return function (population::Vector{Individual})
                for individual in population
                    if rand(Float64) < parameters.mutation_rate
                        individual.x += randn(Float64, individual.size) * parameters.std_dev_spatial
                        individual.y += randn(Float64, individual.size) * parameters.std_dev_spatial
                        individual.theta += randn(Float64, individual.size) * parameters.std_dev_angular
                        MarkChangedGenotype!(individual)
                    end
                end
                if parameters.allow_transpose & (rand(Float64) < parameters.transposition_rate)
                    for individual in population
                        if rand(Float64) < parameters.fraction_transposed
                            x_temp::Vector{Float64} = individual.x
                            individual.x = individual.y
                            individual.y = x_temp
                            MarkChangedGenotype!(individual)
                        end
                    end
                end
                return population
            end
        end