export SelectionParameters, Selection

include("population.jl")

struct SelectionParameters
    allow_elitism::Bool
    population_size::Int64
    tournament_size::Int64
    tournament_selection_factor::Float64
    elite_count::Int64
    
end

function Selection(parameters::SelectionParameters)::Function
    return function (population::Vector{Individual}, elite::Vector{Individual})
        temporal_population::Vector{Individual} = Vector{Individual}(undef, 0)
        tournament_participants::Vector{Individual} = Individual[]
        entire_population::Vector{Individual} = copy([population; elite])
        entire_population = sort(entire_population, by = x -> x.fitness)
        elite::Vector{Individual} = entire_population[1:parameters.elite_count]
        while length(temporal_population) < parameters.population_size
            sort!(entire_population, by = x -> rand(Float64))
            tournament_participants = sort(copy(entire_population[1:parameters.tournament_size]), by = x -> x.fitness)
            p::Float64 = parameters.tournament_selection_factor
            for i = 1:parameters.tournament_size
                if rand(Float64) < p
                    push!(temporal_population, tournament_participants[i])
                    break
                end
                p *= parameters.tournament_selection_factor
            end
        end
        return copy(temporal_population), copy(elite)        
    end
end