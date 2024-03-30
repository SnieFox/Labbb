from config import G, N, c_values
import numpy as np
from selection.selection_method import SelectionMethod
from copy import copy


class RWS(SelectionMethod):
    def select(self, population):
        fitness_list = population.fitnesses
        fitness_sum = sum(fitness_list)

        if fitness_sum == 0:
            fitness_list = [0.0001 for _ in fitness_list]
            fitness_sum = 0.0001 * N

        probabilities = [fitness/fitness_sum for fitness in fitness_list]
        chosen = np.random.choice(population.chromosomes, size=N, p=probabilities)
        mating_pool = np.array([copy(chr) for chr in chosen])
        population.update_chromosomes(mating_pool)


class DisruptiveRWS(SelectionMethod):
    def select(self, population):
        f_avg = population.get_fitness_avg()
        f_scaled = [abs(fitness - f_avg) for fitness in population.fitnesses]
        fitness_sum = sum(f_scaled)

        if fitness_sum == 0:
            f_scaled = [0.0001 for _ in f_scaled]
            fitness_sum = 0.0001 * N

        probabilities = [fitness/fitness_sum for fitness in f_scaled]
        chosen = np.random.choice(population.chromosomes, size=N, p=probabilities)
        mating_pool = np.array([copy(chr) for chr in chosen])
        population.update_chromosomes(mating_pool)


class BlendedRWS(SelectionMethod):
    def __init__(self):
        self.i = 0

    def select(self, population):
        f_scaled = [fitness / (G + 1 - self.i) for fitness in population.fitnesses]
        fitness_sum = sum(f_scaled)

        if fitness_sum == 0:
            f_scaled = [0.0001 for _ in f_scaled]
            fitness_sum = 0.0001 * N

        probabilities = [fitness/fitness_sum for fitness in f_scaled]
        chosen = np.random.choice(population.chromosomes, size=N, p=probabilities)
        mating_pool = np.array([copy(chr) for chr in chosen])
        population.update_chromosomes(mating_pool)

        self.i += 1


class WindowRWS(SelectionMethod):
    def __init__(self, h=2):
        self.h = h
        self.f_h_worst = []

    def select(self, population):
        if len(self.f_h_worst) < self.h:
            self.f_h_worst.append(min(population.fitnesses))
        else:
            self.f_h_worst[0] = self.f_h_worst[1]
            self.f_h_worst[1] = min(population.fitnesses)
        f_worst = min(self.f_h_worst)

        f_scaled = [fitness - f_worst for fitness in population.fitnesses]
        fitness_sum = sum(f_scaled)

        if fitness_sum == 0:
            f_scaled = [0.0001 for _ in f_scaled]
            fitness_sum = 0.0001 * N

        probabilities = [fitness/fitness_sum for fitness in f_scaled]
        chosen = np.random.choice(population.chromosomes, size=N, p=probabilities)
        mating_pool = np.array([copy(chr) for chr in chosen])
        population.update_chromosomes(mating_pool)


class ExpRankingRWS(SelectionMethod):
    def __init__(self, c_variant=2):
        self.c_variant = c_variant
        self.c = self._get_c_value()

    def _get_c_value(self):
        for i in sorted(c_values.keys()):
            if N <= i:
                return c_values[i][self.c_variant-1]
        raise ValueError('N value not supported')

    def _get_rank_probabilities(self):
        return [((self.c - 1) / (self.c ** N - 1)) * self.c ** (N - rank) for rank in range(N)]

    def select(self, population):
        np.random.shuffle(population.chromosomes)
        #population.chromosomes = [x for _, x in sorted(zip(population.fitnesses, population.chromosomes), key=lambda pair: pair[0])]

        probabilities = self._get_rank_probabilities()

        probabilities = np.array(probabilities)
        probabilities = probabilities / np.sum(probabilities)

        chosen = np.random.choice(population.chromosomes, size=N, p=probabilities)
        population.update_chromosomes(np.array([copy(chr) for chr in chosen]))


class PowerScalingRWS(SelectionMethod):
    def __init__(self, k=1.005):
        self.k = k

    def select(self, population):
        fitness_list = [f ** self.k for f in population.fitnesses]
        fitness_sum = sum(fitness_list)

        if fitness_sum == 0:
            fitness_list = [0.0001 for _ in fitness_list]
            fitness_sum = 0.0001 * N

        probabilities = [f / fitness_sum for f in fitness_list]
        chosen = np.random.choice(population.chromosomes, size=N, p=probabilities)
        mating_pool = np.array([copy(chr) for chr in chosen])
        population.update_chromosomes(mating_pool)


class TruncationRWS(SelectionMethod):
    def __init__(self, c=1, sigma=2):
        self.c = c
        self.sigma = sigma

    def select(self, population):
        fitnesses = np.array(population.fitnesses)
        fitness_mean = np.mean(fitnesses)
        fitness_std = np.std(fitnesses)

        f_scaled = fitnesses - (fitness_mean - self.c * self.sigma)
        f_scaled = np.where(f_scaled > 0, f_scaled, 0)  # Ensure no negative fitness values
        fitness_sum = np.sum(f_scaled)

        if fitness_sum == 0:
            f_scaled = np.full(N, 0.0001)
            fitness_sum = 0.0001 * N

        probabilities = f_scaled / fitness_sum
        chosen = np.random.choice(population.chromosomes, size=N, p=probabilities)
        mating_pool = np.array([copy(chr) for chr in chosen])
        population.update_chromosomes(mating_pool)
