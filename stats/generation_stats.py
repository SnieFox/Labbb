from config import N
from model.population import Population

# stats that are used for graphs
class GenerationStats:
    def __init__(self, population: Population, param_names: tuple[str]):
        self.population = population
        self.param_names = param_names

        self.f_avg = None
        self.f_std = None
        self.f_best = None
        self.num_of_best = None
        self.optimal_count = None
        self.growth_rate = None
        self.difference = None
        self.intensity = None
        self.reproduction_rate = None
        self.loss_of_diversity = None
        self.optimal_individual_lost = False

        self.NI_loose = None

    def calculate_non_successful_runs_stats(self):
        non_successful_runs = [run for run in self.population.runs if not run.is_successful]
        non_successful_count = len(non_successful_runs)

        if non_successful_count > 0:
            non_nis = [run.NI for run in non_successful_runs]
            self.nonSuc = non_successful_count / N
            self.nonMin_NI = min(non_nis)
            self.nonMax_NI = max(non_nis)
            self.nonAvg_NI = np.mean(non_nis)
            self.nonSigma_NI = np.std(non_nis)

            non_founds = [run.F_found for run in non_successful_runs]
            self.nonAvg_F_found = np.mean(non_founds)
            self.nonSigma_F_found = np.std(non_founds)
            self.nonMax_F_found = max(non_founds)

    def calculate_stats_before_selection(self, prev_gen_stats):
        self.ids_before_selection = set(self.population.get_ids())

        if self.param_names[0] != 'FconstALL':
            self.f_avg = self.population.get_fitness_avg()
            self.f_std = self.population.get_fitness_std()
            self.f_best = self.population.get_fitness_max()
            self.num_of_best = self.population.count_fitness_at_least(self.f_best)
            self.optimal_count = self.population.count_optimal_genotype()
            
            if not prev_gen_stats:
                self.growth_rate = 1
            else:
                num_of_prev_best = self.population.count_fitness_at_least(prev_gen_stats.f_best)
                self.growth_rate = num_of_prev_best / prev_gen_stats.num_of_best

    def calculate_stats_after_selection(self):
        ids_after_selection = set(self.population.get_ids())
        self.reproduction_rate = len(ids_after_selection) / N
        self.loss_of_diversity = len([True for id in self.ids_before_selection if id not in ids_after_selection]) / N
        self.optimal_individual_lost = self.check_optimal_individual_lost(ids_after_selection)
        self.ids_before_selection = None

        if self.param_names[0] != 'FconstALL':
            self.difference = self.population.get_fitness_avg() - self.f_avg

            # Оновлення метрики для втрати оптимальної особини
            if self.optimal_individual_lost:
                self.NI_loose = self.population.get_iteration_number()

            if self.f_std == 0:
                self.intensity = 1
            else:
                self.intensity = self.difference / self.f_std

    def check_optimal_individual_lost(self, ids_after_selection):
        optimal_individual_id = self.population.get_optimal_individual_id()
        if optimal_individual_id not in ids_after_selection:
            return True
        return False