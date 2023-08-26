module EvolutionaryAlgorithm

using Statistics

export ProceedEvolutionaryAlgorithm, History

include("population.jl")
include("fitness_function.jl")
include("mutation.jl")
include("crossover.jl")
include("selection.jl")


function ProceedEvolutionaryAlgorithm(
    problem_size::Int64,
    initialize_feasible::Bool,
    use_exp::Bool,
    dispersion_factor::Float64, allow_iteration_stop::Bool,
    max_iterations::Int64,
    allow_fitness_stop::Bool,
    best_fitness::Float64,
    allow_fitness_std_criterion::Bool,
    fitness_std_threshold::Float64,
    population_size::Int64,
    verbose::Bool, crossover_rate::Float64,
    n_cuts::Int64, mutation_rate::Float64,
    std_dev_spatial::Float64,
    std_dev_angular::Float64,
    transposition_rate::Float64,
    allow_transpose::Bool,
    fraction_transposed::Float64, use_penalty::Bool,
    center_solution::Bool,
    penalty_factor::Float64, allow_elitism::Bool,
    tournament_size::Int64,
    tournament_selection_factor::Float64,
    elite_count::Int64)::History

    parameters = EvolutionatyAlgorithmParameters(
        allow_iteration_stop,
        max_iterations,
        allow_fitness_stop,
        best_fitness,
        allow_fitness_std_criterion,
        fitness_std_threshold,
        population_size,
        verbose
    )


    population = Population(
        PopulationParameters(
            problem_size,
            initialize_feasible,
            use_exp,
            dispersion_factor
        )
    )
    mutate = Mutation(
        MutationParameters(
            mutation_rate,
            std_dev_spatial,
            std_dev_angular,
            transposition_rate,
            allow_transpose,
            fraction_transposed
        )
    )
    crossover = Crossover(
        CrossoverParameters(
            crossover_rate,
            n_cuts
        )
    )

    selection = Selection(
        SelectionParameters(
            allow_elitism,
            population_size,
            tournament_size,
            tournament_selection_factor,
            elite_count
        )
    )

    fitness_function = FitnessFunction(
        FitnessFunctionParameters(
            use_penalty,
            center_solution,
            penalty_factor
        )
    )

    operators = Operators(
        population,
        fitness_function,
        selection,
        crossover,
        mutate
    )

    return Evolve(parameters, operators)
end


struct EvolutionatyAlgorithmParameters
    allow_iteration_stop::Bool
    max_iterations::Int64
    allow_fitness_stop::Bool
    best_fitness::Float64
    allow_fitness_std_criterion::Bool
    fitness_std_threshold::Float64
    population_size::Int64
    verbose::Bool
end

struct Operators
    initialize_population::Function
    evaluate_fitness_function::Function
    select::Function
    cross::Function
    mutate::Function
end

struct History
    best_individual_history::Vector{Individual}
    best_fitness::Vector{Float64}
    mean_fitness::Vector{Float64}
    population_std::Vector{Float64}
    population::Vector{Individual}
    iterations::Int64
end

function Evolve(
    parameters::EvolutionatyAlgorithmParameters,
    operators::Operators
)::History

    stopCondition = StopCondition(parameters)

    best_fitness::Float64 = Inf
    population_std::Float64 = Inf
    best_individual_history::Vector{Individual} = Vector{Individual}(undef, parameters.max_iterations)
    mean_fitness_history::Vector{Float64} = zeros(Float64, parameters.max_iterations,)
    best_fintess_history::Vector{Float64} = zeros(Float64, parameters.max_iterations,)
    population_std_history::Vector{Float64} = zeros(Float64, parameters.max_iterations,)
    t::Int64 = 1

    population::Vector{Individual} = operators.initialize_population(parameters.population_size)
    elite::Vector{Individual} = Vector{Individual}(undef, 0)
    offspring::Vector{Individual} = Vector{Individual}(undef, 0)

    if parameters.verbose
        println("Setup complete")
    end

    while !stopCondition(t, best_fitness, population_std)
        population = operators.mutate(deepcopy(population))

        offspring = operators.cross(population)

        population = operators.evaluate_fitness_function([population; offspring])

        population, elite = operators.select(population, copy(elite))

        best_individual_history[t] = argmin(x -> x.fitness, [population; elite])
        best_fitness = minimum(x -> x.fitness, [population; elite])
        best_fintess_history[t] = best_fitness
        mean_fitness_history[t] = mean(map(x -> x.fitness, population))
        population_std = std(map(x -> x.fitness, population), mean=mean_fitness_history[t])
        population_std_history[t] = population_std

        if parameters.verbose
            println("Iteration: ", t)
            println("Best fitness: ", best_fitness)
            println("Mean fitness: ", mean_fitness_history[t])
            println("Population std: ", population_std)
        end

        t += 1
    end
    return History(
        best_individual_history,
        best_fintess_history,
        mean_fitness_history,
        population_std_history,
        population,
        t - 1
    )
end


function StopCondition(parameters::EvolutionatyAlgorithmParameters)::Function
    return function (iterations::Int64, best_fitness::Float64, population_std::Float64)
        ((iterations > parameters.max_iterations) && (parameters.allow_iteration_stop)) ||
            ((best_fitness <= parameters.best_fitness) && (parameters.allow_fitness_stop)) ||
            ((population_std <= parameters.fitness_std_threshold) && (parameters.allow_fitness_std_criterion))
    end

end

history::History = ProceedEvolutionaryAlgorithm(
    100,
    true,
    false,
    1.0,
    true,
    1000,
    true,
    0.0,
    false,
    0.0,
    100,
    true,
    0.8, 
    2,
    0.1,
    0.1,    
    0.1,
    0.1,
    true,
    0.5,
    true,
    true,
    200.0,
true,
8,
2.0,
15
)

end # module EvolutionaryAlgorithm
