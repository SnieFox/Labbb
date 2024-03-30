import numpy as np
from config import DELTA, SIGMA, get_pop_seed
from model.chromosome import Chromosome
from model.population import Population
from model.encoding import Encoder
from math import sin, pi, exp, log


class FitnessFunc:
    def __init__(self, chr_length):
        self.chr_length = chr_length
        self.optimal = None

    def apply(self, genotype):
        raise NotImplementedError()

    def get_optimal(self):
        raise NotImplementedError()
    
    def get_phenotype(self, genotype):
        raise NotImplementedError()

    def generate_population_for_run(self, run_i):
        return Population(self, seed=get_pop_seed(run_i))
    
    def check_chromosome_success(self, chr: Chromosome):
        y_diff = abs(chr.fitness - self.get_optimal().fitness)
        x_diff = abs(self.get_phenotype(chr.genotype) - self.get_phenotype(self.get_optimal().genotype))
        return y_diff <= DELTA and x_diff <= SIGMA


class FconstALL(FitnessFunc):
    def apply(self, genotype):
        return 100

    def get_optimal(self):
        if not self.optimal:
            self.optimal = Chromosome(0, np.full(self.chr_length, b'0'), self)
        return self.optimal
    
    def get_phenotype(self, genotype):
        return 0
    
    def check_chromosome_success(self, ch):
        return True


class FH(FitnessFunc):
    def __init__(self, chr_length):
        super().__init__(chr_length)

    def apply(self, genotype):
        k = len([True for gene in genotype if gene == b'1'])
        return self.chr_length - k

    def get_optimal(self):
        if not self.optimal:
            self.optimal = Chromosome(0, np.full(self.chr_length, b'0'), self)
        return self.optimal

    def get_phenotype(self, genotype):
        return len([True for gene in genotype if gene == b'1'])


class Fx2(FitnessFunc):
    def __init__(self, encoder: Encoder):
        super().__init__(encoder.length)
        self.encoder = encoder
        self.is_caching = encoder.length <= 12
        self.cache_dict = {}
        if self.is_caching:
            for v in self.encoder.get_all_values():
                self.cache_dict[v.tobytes()] = self.encoder.decode(v)**2

    def apply(self, genotype):
        if self.is_caching:
            return self.cache_dict[genotype.tobytes()]
        return self.encoder.decode(genotype)**2

    def get_optimal(self):
        if not self.optimal:
            optimal_genotype = self.encoder.encode(self.encoder.upper_bound)
            self.optimal =  Chromosome(0, optimal_genotype, self)
        return self.optimal

    def get_phenotype(self, chromosome):
        return self.encoder.decode(chromosome)


class F5122subx2(FitnessFunc):
    def __init__(self, encoder: Encoder):
        super().__init__(encoder.length)
        self.encoder = encoder
        self.is_caching = encoder.length <= 12
        self.cache_dict = {}
        if self.is_caching:
            for v in self.encoder.get_all_values():
                self.cache_dict[v.tobytes()] = 5.12**2 - self.encoder.decode(v)**2

    def apply(self, genotype):
        if self.is_caching:
            return self.cache_dict[genotype.tobytes()]
        return 5.12**2 - self.encoder.decode(genotype)**2

    def get_optimal(self):
        if not self.optimal:
            optimal_genotype = self.encoder.encode(0)
            self.optimal =  Chromosome(0, optimal_genotype, self)
        return self.optimal

    def get_phenotype(self, genotype):
        return self.encoder.decode(genotype)
    
class Fexp(FitnessFunc):
    def __init__(self, c, encoder: Encoder):
        self.c = c
        super().__init__(encoder.length)
        self.encoder = encoder
        self.is_caching = encoder.length <= 12
        self.cache_dict = {}
        if self.is_caching:
            for v in self.encoder.get_all_values():
                self.cache_dict[v.tobytes()] = exp(self.c * self.encoder.decode(v))

    def apply(self, genotype):
        if self.is_caching:
            return self.cache_dict[genotype.tobytes()]
        return exp(self.c * self.encoder.decode(genotype))

    def get_optimal(self):
        if not self.optimal:
            optimal_genotype = self.encoder.encode(self.encoder.upper_bound)
            self.optimal =  Chromosome(0, optimal_genotype, self)
        return self.optimal

    def get_phenotype(self, chromosome):
        return self.encoder.decode(chromosome)


class Rastrigin(FitnessFunc):
    def __init__(self, encoder: Encoder):
        super().__init__(encoder.length)
        self.encoder = encoder
        self.is_caching = encoder.length <= 12
        self.cache_dict = {}
        self.a = 7
        if self.is_caching:
            for v in self.encoder.get_all_values():
                x = self.encoder.decode(v)
                self.cache_dict[v.tobytes()] = abs(10 * np.cos(2*np.pi*self.a) - self.a**2) + 10 * np.cos(2*np.pi*x)  -x**2

    def apply(self, genotype):
        if self.is_caching:
            return self.cache_dict[genotype.tobytes()]
        x = self.encoder.decode(genotype)
        return abs(10 * np.cos(2*np.pi*self.a) - self.a**2) + 10 * np.cos(2*np.pi*x)  -x**2

    def get_optimal(self):
        if not self.optimal:
            optimal_genotype = self.encoder.encode(0)
            self.optimal =  Chromosome(0, optimal_genotype, self)
        return self.optimal

    def get_phenotype(self, chromosome):
        return self.encoder.decode(chromosome)


class DebsTestFunction2(FitnessFunc):
    def __init__(self, encoder: Encoder):
        super().__init__(encoder.length)
        self.encoder = encoder

    def apply(self, genotype):
        x = self.encoder.decode(genotype)
        return exp(-2 * log(2) * ((x - 0.1) / 0.8) ** 2) * sin(5 * pi * x) ** 6

    def get_optimal(self):
        if not self.optimal:
            optimal_genotype = self.encoder.encode(0.1)
            self.optimal = Chromosome(0, optimal_genotype, self)
        return self.optimal

    def get_phenotype(self, chromosome):
        return self.encoder.decode(chromosome)


class DebsTestFunction4(FitnessFunc):
    def __init__(self, encoder: Encoder):
        super().__init__(encoder.length)
        self.encoder = encoder

    def apply(self, genotype):
        x = self.encoder.decode(genotype)
        return exp(-2 * log(2) * ((x - 0.08) / 0.854) ** 2) * sin(5 * pi * (x ** 0.75 - 0.05)) ** 6

    def get_optimal(self):
        if not self.optimal:
            optimal_genotype = self.encoder.encode(0.08)
            self.optimal = Chromosome(0, optimal_genotype, self)
        return self.optimal

    def get_phenotype(self, chromosome):
        return self.encoder.decode(chromosome)