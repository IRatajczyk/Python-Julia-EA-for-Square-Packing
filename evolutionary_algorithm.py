from typing import Any, Dict, List
import json
import time
from io import StringIO
import numpy as np
import matplotlib.pyplot as plt
from juliacall import Main as jl
from juliacall import Pkg as jlPkg


class Solution:
    """
    Solution of an evolutionary algorithm run in Python friendly format.
    """

    def __init__(self, solution):
        self.problem_size: int = solution.size
        self.x: np.ndarray = np.array(solution.x)
        self.y: np.ndarray = np.array(solution.y)
        self.theta: np.ndarray = np.array(solution.theta)
        self.fitness: float = solution.fitness
        self.is_feasible: bool = solution.is_feasible

    def __str__(self):
        return f"{'Feasible' if self.is_feasible else 'Non-feasible'} solution with fitness {self.fitness}"
    
    def __repr__(self):
        return self.__str__()	

    def visualize(self) -> None:
        """	
        Solution visualization.
        """
        plt.figure(figsize=(10, 10))
        p: float = np.pi/4
        l: float = np.sqrt(2)/2
        for x, y, t in zip(self.x, self.y, self.theta):
            plt.plot(
                [x+l*np.sin(t-p), x+l*np.sin(t+p), x+l*np.sin(t+3*p), x+l*np.sin(t-3*p), x+l*np.sin(t-p)],
                [y+l*np.cos(t-p), y+l*np.cos(t+p), y+l*np.cos(t+3*p), y+l*np.cos(t-3*p), y+l*np.cos(t-p)],
                color="black"
            )
        plt.axis('scaled')
        plt.show()


class EAHistory:
    """
    History of an evolutionary algorithm run.
    """

    def __init__(self, best_individual_history, best_fitness, mean_fitness, population_std):
        self.best_individual_history: List[Solution] = [Solution(solution) for solution in best_individual_history]
        self.best_fitness: np.ndarray = np.array(best_fitness)
        self.mean_fitness: np.ndarray = np.array(mean_fitness)
        self.population_std: np.ndarray = np.array(population_std)

    def visualize(self) -> None:
        """
        Visualizes the history of the algorithm.
        """
        plt.figure(figsize=(10, 10))
        plt.plot(self.best_fitness, label="Best fitness")
        plt.plot(self.mean_fitness, label="Mean fitness")
        plt.plot(self.population_std, label="Population std")
        plt.grid()
        plt.legend()
        plt.show()


class EvolutionaryAlgorithmWrapper:
    """
    Wrapper for the EvolutionaryAlgorithm Julia package.
    """

    def __init__(
            self,
            params: [str | Dict[str, Any]],
            julia_package_path: str = r"EvolutionaryAlgorithm"
    ):

        jlPkg.activate(julia_package_path)
        jl.seval("using EvolutionaryAlgorithm")

        self.params = self._load_json(
            params) if isinstance(params, str) else params
        self.history: [EAHistory | None] = None
        self.time_measured: float = -1
        self._check()

    def _check(self):
        assert self.params["problem_size"] > 0, f"problem_size must be positive, got {self.params['problem_size']}"
        assert self.params["population_size"] > 0, f"population_size must be positive, got {self.params['population_size']}"
        assert self.params["allow_iteration_stop"] or self.params["allow_fitness_stop"] or self.params["allow_fitness_std_criterion"], "At least one of allow_iteration_stop and allow_time_stop must be true"
        assert not self.params["allow_iteration_stop"] or self.params["max_iterations"] > 0, f"max_iterations must be positive, got {self.params['max_iterations']}"
        assert not self.params["allow_fitness_stop"] or self.params["best_fitness"] > 0, f"fitness_threshold must be positive, got {self.params['fitness_threshold']}"
        assert not self.params["allow_fitness_std_criterion"] or self.params["fitness_std_threshold"] > 0, f"fitness_std_threshold must be positive, got {self.params['fitness_std_threshold']}"
        assert 0 <= self.params["mutation_rate"] <= 1, f"mutation_rate must be in [0, 1], got {self.params['mutation_rate']}"
        assert 0 <= self.params["crossover_rate"] <= 1, f"crossover_rate must be in [0, 1], got {self.params['crossover_rate']}"
        assert not self.params["allow_transpose"] or (0 <= self.params["fraction_transposed"] <= 1), f"transpose_rate must be in [0, 1], got {self.params['fraction_transposed']}"
        assert not self.params["use_penalty"] or self.params["penalty_factor"] > 0, f"penalty_factor must be positive, got {self.params['penalty_factor']}"
        assert not self.params["allow_elitism"] or self.params["elite_count"] > 0, f"elite_count must be positive, got {self.params['elite_count']}"
        assert self.params["tournament_size"] > 0, f"tournament_size must be positive, got {self.params['tournament_size']}"
        assert self.params["tournament_selection_factor"] > 0, f"tournament_selection_factor must be positive, got {self.params['tournament_selection_factor']}"

    def _load_json(self, path: str) -> Dict[str, Any]:
        with open(path, "r") as file:
            return json.load(file)

    def warmup(self) -> None:
        """
        Runs the algorithm once to warmup the Julia runtime.
        """
        jl.ProceedEvolutionaryAlgorithm(
            self.params["problem_size"],
            self.params["initialize_feasible"],
            self.params["use_exponential_sapling"],
            self.params["dispersion_factor"],
            self.params["allow_iteration_stop"] or True,
            3,
            self.params["allow_fitness_stop"],
            self.params["best_fitness"],
            self.params["allow_fitness_std_criterion"],
            self.params["fitness_std_threshold"],
            self.params["population_size"],
            self.params["verbose"] and False,
            1.0,  # self.params["crossover_rate"],
            self.params["n_cuts"],
            1.0,  # self.params["mutation_rate"],
            self.params["std_dev_spatial"],
            self.params["std_dev_angular"],
            self.params["transposition_rate"],
            self.params["allow_transpose"] or True,
            1.0,  # self.params["fraction_transposed"],
            self.params["use_penalty"],
            self.params["center_solution"],
            self.params["penalty_factor"],
            self.params["allow_elitism"],
            self.params["tournament_size"],
            self.params["tournament_selection_factor"],
            self.params["elite_count"]
        )

    def run(self, io_stream: [StringIO | None] = None, measure_time: bool = False):
        """
        Runs the algorithm and stores the history.
        """
        verbose_flag: bool = io_stream is not None and self.params["verbose"]
        if verbose_flag:
            jl_io = jl.eval("IOBuffer()")
            jl.global_io = jl_io

        if measure_time:
            time_measured = time.process_time()

        self.history = jl.ProceedEvolutionaryAlgorithm(
            self.params["problem_size"],
            self.params["initialize_feasible"],
            self.params["use_exponential_sapling"],
            self.params["dispersion_factor"],
            self.params["allow_iteration_stop"],
            self.params["max_iterations"],
            self.params["allow_fitness_stop"],
            self.params["best_fitness"],
            self.params["allow_fitness_std_criterion"],
            self.params["fitness_std_threshold"],
            self.params["population_size"],
            verbose_flag,
            self.params["crossover_rate"],
            self.params["n_cuts"],
            self.params["mutation_rate"],
            self.params["std_dev_spatial"],
            self.params["std_dev_angular"],
            self.params["transposition_rate"],
            self.params["allow_transpose"],
            self.params["fraction_transposed"],
            self.params["use_penalty"],
            self.params["center_solution"],
            self.params["penalty_factor"],
            self.params["allow_elitism"],
            self.params["tournament_size"],
            self.params["tournament_selection_factor"],
            self.params["elite_count"]
        )
        if measure_time:
            self.time_measured = time.process_time() - time_measured
        return self.history

    def get_parsed_history(self) -> EAHistory:
        """
        Returns a dictionary with the following keys:
        - best_fitness: best fitness value for each iteration
        - mean_fitness: mean fitness value for each iteration
        - std_fitness: standard deviation of fitness values for each iteration
        """
        assert self.history is not None, "Run the algorithm first"
        return EAHistory(
            self.history.best_individual_history,
            self.history.best_fitness,
            self.history.mean_fitness,
            self.history.population_std,
        )

    def visualize_solution(self, solution=[Solution | None], index: int = -1) -> None:
        """
        Visualizes a solution. If no solution is passed then the best solution is visualized.
        """
        if solution is None:
            assert self.history is not None, "Run the algorithm first"
            solution: Solution = Solution(list(self.history.best_individual_history)[index])
        solution.visualize()

