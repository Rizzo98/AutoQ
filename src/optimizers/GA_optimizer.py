import pygad
import numpy as np
import sys
from src.optimizers import Optimizer

class GAOptimizer(Optimizer):
    def __init__(self, bitSpace, model, qmnAnalysis=None, regressor=None, epsilon=0.01, sol_per_pop=16) -> None:
        super().__init__(bitSpace, model, qmnAnalysis)
        self.regressor = regressor
        self.epsilon = epsilon
        self.model = model['model']
        self.layers = self.model.layers
        self.layer_names = [l.name for l in self.layers]
        self.num_genes = len(self.layers)
        self.ga_instance = pygad.GA(num_generations=1_000,
                       num_parents_mating=2,
                       fitness_func=self.fitness,
                       sol_per_pop=sol_per_pop,
                       num_genes=self.num_genes,
                       parent_selection_type="rank",
                       keep_parents=1,
                       crossover_type='single_point',
                       mutation_type='random',
                       mutation_probability=.05,
                       gene_type=int,
                       gene_space= [[0,1,2] for _ in range(self.num_genes)],
                       callback_generation=self.callback_gen,
                       initial_population = None,
                       stop_criteria=['saturate_30']
                    )

        self.__x, self.__cost = [],[]
        for _,v in self.layers.items():
            self.__cost.append(v['weight_perc'])
            self.__x+=[52,0,0,0,0]

    def fitness(self, solution, solution_idx):
        for i in range(0,len(self.__x),5):
            layer_name = self.layer_names[i//5]
            bit = self.bitSpace[solution[i//5]]
            self.__x[i] = self.qmnAnalysis.accuracies[layer_name][solution[i//5]]
            if bit==32:
                self.__x[i+1], self.__x[i+2], self.__x[i+3], self.__x[i+4] = 0,0,0,0
            else:
                Qmn = self.qmnAnalysis.stats[str(bit)][layer_name]
                self.__x[i+1] = Qmn['out_quantizer']['error']['clipped_vals']
                self.__x[i+2] = Qmn['out_quantizer']['error']['fraction_error']
                avg_weight_clipping_error, avg_weight_fraction_error,count = 0,0,0
                for k in Qmn:
                    if k!='class' and k!='out_quantizer':
                        if 'error' in Qmn[k]:
                            avg_weight_clipping_error+= Qmn[k]['error']['clipped_vals']
                            avg_weight_fraction_error+= Qmn[k]['error']['fraction_error']
                            count+=1
                self.__x[i+3] = avg_weight_clipping_error/count
                self.__x[i+4] = avg_weight_fraction_error/count

        y_pred = self.regressor.predict([self.__x])[0]
        c = np.sum([self.layers[self.layer_names[i]]['weight_perc']*self.bitSpace[solution[i]] for i in range(self.num_genes)])
        return y_pred**2-(self.epsilon*c)**2

    def callback_gen(self, ga_instance):
        if ga_instance.generations_completed%10==0:
            sys.stdout.write(f"Generation : {ga_instance.generations_completed}  ")
            sys.stdout.write(f"Fitness of the best solution : {ga_instance.best_solution()[1]}\n")

    def optimize(self) -> None:
        self.ga_instance.run()
        return self.ga_instance.population