#############
#Example implemented
##############

import os
from genetic_opt import *

#Create directories for saving files
savedir = '/Desktop/GeneticTest/'
saveloc = savedir + 'python_test/'

if not os.path.isdir(saveloc):
    os.mkdir(saveloc) 


#Define the function we want to optimize
def myfunc(x1, x2, *args):
    """Some multivariable function with two inputs we wish to optimize over
    Possibility for more than 2 inputs is included via *args
    """
    f = 3*(1-x1) **2.* np.exp(-(x1 **2) - (x2+1)**2) - \
    10*(x1/5 - x1**3 - x2**5)* np.exp(-x1**2-x2**2) - \
    1/3* np.exp(-(x1+1)**2 - x2**2)
    return f

#Plot the function we will optimize
go_params = {}
go_params['gridsize'] = 300
go_params['span'] = 1
go_params['nvar'] = 4
grid_opt = GridOptimization(go_params, myfunc)
grid_opt.plot_2d_function( saveloc + 'surf_plot')

# Run grid optimization
zmax, inds = grid_opt.grid_optimization()

print('Grid search optimization maximum: {0}'.format(zmax))
print('Grid search optimization maximum indices: {0}'.format(inds))


#Run genetic optimization
p = {}
p['nvar'] = 6 #Set number of variables
p['span'] = 1
p['f1'] = .20 #Fraction to randomly mutate
p['mutate_els'] = 1 #number of parameter elements to mutate within a parameter set
p['f2'] = .30 #Fraction to randomly combine
p['f3'] = .40 #Keep the top f3 fraction for the next iteration
p['epochs'] = 20 #Number of generations to cycle through
p['popsize'] = 20 #Size of the population
p['rseed'] = 1 #Seed rng for reproducibility

gen_opt = GeneticOptimization(p, myfunc)
fh, sv, svf, bi, bif = gen_opt.genetic_alg()

#Plot fitness history
gen_opt.scatterplot_fitness(fh, saveloc + 'fitness_scatterplot_test', zmax)

#Save hyperparameters to a txt file
p['fitness_history'] = fh
p['survivor'] = sv
p['survivor_fitness'] = svf
p['best_individual'] = bi
p['best_individual_fitness'] = bif
with open(saveloc + 'optimization_parameters.txt', 'w') as f: 
    for key, value in p.items(): 
        f.write('%s:%s\n' % (key, value))

