#############
#Genetic optimization algorithm 
##############
import pdb

import matplotlib.pyplot as plt
import numpy as np


class GridOptimization():
    """
    Parameters
    ----------
    func: function
        function to optimize
    params: dict
        Dictionary containing parameters for genetic optimization
    params['nvar']: int
        Number of variables to optimize over
    params['span']: float
        Span to optimize variables over
    params['gridsize']: int
        Size of grid to search over

    """

    def __init__(self, params, func):
        self.params = params
        self.f = func



    def grid_optimization(self):
        """
        Numerically optimize a function with _nvar_ variables through (inefficient) gridsearch
        return :zmax: float
            maximum function value found
        return :inds: array[float]
            indices for maximum function value found
        """
        span = self.params['span']
        gridsize = self.params['gridsize']
        nvar = self.params['nvar']

        
        ls = np.linspace(-span, span, gridsize)
        X = np.meshgrid(*(2*[ls]))
    
        z = self.f(*X)

        zmax = np.amax(z)

        linear_ind = np.argmax(z)
        ind_out = np.unravel_index(linear_ind, tuple([gridsize for i in range(0,nvar)]))
        inds = [ls[i] for i in ind_out]
        inds = np.flip(inds) #Meshgrid switches dimensions; must flip

        return zmax, inds


    def plot_2d_function(self, savepath):
        """
        Plot the function we plan to optimize
        savepath: string
            save path
        """

        fontsize_1 = 16
        fontsize_2 = 14
    
        fig, ax = plt.subplots()

        span = self.params['span']
        gridsize = self.params['gridsize']

        
        x = np.linspace(-span, span, gridsize)
        y = np.linspace(-span, span, gridsize)
        
        [X, Y] = np.meshgrid(x, y)
        plt.contour(X, Y, self.f(X, Y), levels=20)

        plt.title('Contour plot of 2D function to be optimized', fontsize=fontsize_1)
        plt.ylabel('Y', fontsize=fontsize_2)
        plt.xlabel('X', fontsize=fontsize_2)
        plt.tight_layout() 
        fig.savefig(savepath + ".png")
        plt.close()




class GeneticOptimization():
    """ 
    Parameters
    ----------
    func: function
        function to optimize
    params: dict
        Dictionary containing parameters for genetic optimization
    params['nvar']: int
        Number of variables to optimize over
    params['span']: float
        Span to optimize variables over
    params['f1']: float
        Fraction of population to randomly mutate
    params['mutate_els']: int
        number of parameter elements to mutate within a parameter set
    params['f2']: float
        Fraction of population to randomly combine
    params['f3']: float
        Keep the top f3 fraction for the next iteration
    params['epochs']: int
        Number of generations to cycle through
    params['popsize']: int
        Size of the population
    params['rseed']: int
        Seed for random number generator
    """

    def __init__(self, params, func):
        self.params = params
        self.f = func


    def genetic_alg(self):
        """ Optimize via a genetic algorithm, using mutation and combination
        to create new members in the population, and keeping only the 
        "fittest" individuals
        return :fit_hist: array containing the fitness history over each epoch
        return :survivor: the fittest survivor
        return :survivor_fitness: the fitness of the fittest survivor
        return :best_individual: the fittest individual over all epochs
        return :best_individual_fitness: the fitness of the fittest individual over all epochs
        """
        span = self.params['span'] #bound the optimization space of each variable over span
        nvar = self.params['nvar'] #number of variables to optimize the function over

        f1 = self.params['f1'] #Fraction to randomly mutate
        mutate_els = self.params['mutate_els'] #number of parameter elements to mutate within a parameter set
        f2 = self.params['f2'] #Fraction to randomly combine
        f3 = self.params['f3'] #Keep the top f3 fraction for the next iteration
        epochs = self.params['epochs'] #Number of generations to cycle through
        popsize = self.params['popsize'] #Size of the population

        #seed rng for reproducibility
        np.random.seed(self.params['rseed'])

        # Generate random population, each number bounded by span
        pop = (np.random.rand(popsize,nvar) - 0.5) * 2*span

        #initialize best individual and best individual fitness arbitrarily
        best_individual = pop[0, 0:nvar]
        best_individual_fitness = self.f(*np.transpose(best_individual))
    
        #Store fitness history here
        fit_hist = np.zeros((epochs, popsize))
        
        for j in range(0, epochs):
        
            print('Running epoch {0}'.format(j));
       
            if np.shape(pop)[0] < popsize:
                new_pop_members = (np.random.rand(popsize - np.shape(pop)[0],nvar) - 0.5) * 2*span
                pop = np.append(pop, new_pop_members, axis=0) 

            # Mutate random f1 proportion
            mutate_indices = np.random.choice(np.arange(0,popsize), size = int(np.ceil(popsize*f1)), replace = False)
            
            # For each element in pop, randomly mutate _mutate_els_ variables
            for i in range(0, len(mutate_indices)):
                # Randomly select indices to mutate
                mutate_el_indices = np.random.choice(np.arange(0,nvar), size = mutate_els, replace = False)
                # Mutate that index of individual mutate_indices(i) of the population
                pop[mutate_indices[i],mutate_el_indices] = (np.random.rand(1) - 0.5) * 2*span


            # Genetically combine random f2; produce 2 offspring per 2 parents
            # to avoid population loss            
            num_parents = int(np.ceil((popsize*f2)/2)*2) # make sure result is even
            combine_indices = np.random.choice(np.arange(0,popsize), size = num_parents, replace = False)


            # For each 2 elements in pop (the parents), randomly combine variables
            for i in range(0, num_parents, 2):
                # Randomly select indices to mutate
                combine_el_indices = np.random.choice(np.arange(0,nvar), size = int(np.ceil(nvar/2)), replace = False)
                
                parent1 = np.copy(pop[combine_indices[i], :])
                parent2 = np.copy(pop[combine_indices[i+1], :])
                
                #Combine parent1 and parent2 of the population to produce 2 offspring
                pop[combine_indices[i],combine_el_indices] = parent2[combine_el_indices]
                pop[combine_indices[i+1],combine_el_indices] = parent1[combine_el_indices]
        


            # Fitness test on random population

            fitness = self.f(*np.transpose(pop))
            arr = np.append(pop, np.transpose(np.array([fitness])), axis= 1)
            print('Initial parameters (first _nvar_ columns) and fitness values (last col): {0}'.format(arr))
            
            # Save fitness history
            fit_hist[j, :] = fitness

            # Keep top f3 in fitness
            arr = arr[arr[:,nvar].argsort()]
            arr = arr[int(np.ceil(len(arr)*(1-f3))):len(arr), :] #keep top f3 in fitness
            
            #Keep track of best individual over all epochs
            if arr[len(arr)-1, nvar] > best_individual_fitness:
                best_individual_fitness = arr[len(arr)-1, nvar]
                best_individual = pop[len(pop)-1, :]

            # Strip off fitness values and repeat
            pop = arr[:, 0:nvar]
        
        survivor = pop[len(pop)-1, :]
        survivor_fitness = arr[len(arr)-1, nvar]
        
        return fit_hist, survivor, survivor_fitness, best_individual, best_individual_fitness



    def scatterplot_fitness(self, hist, savepath, fmax):
        #%Scatter plot the fitness of each individual at each epoch
        #:hist: a 2D array of fitnesses with each epoch
        #:savepath: the path to save the figure
        #:fmax: function maximum determined by gridsearch
        
        fontsize_1 = 20;
        fontsize_2 = 16;
        fontsize_3 = 14;
        
    
        fig, ax = plt.subplots()
            
        ax.plot(np.arange(1,len(hist[0])+1), np.mean(hist, 0), '--',color='black', label='Population Mean')
        ax.plot(np.arange(1,len(hist[0])+ 1), fmax*np.ones(len(hist[0])), '--r', label = 'Grid Search Max')
        plt.legend()

        for x in hist:
            plt.scatter(np.arange(1,len(hist[0])+1), x, s=40, color=(0 ,.7, .7), alpha = 0.7)


        ax.grid(b=True)
        plt.xlim(0, len(hist[0])+1)

        
        plt.title('Fitnesses during genetic optimization', fontsize=fontsize_1)
        plt.ylabel('Fitness', fontsize=fontsize_2)
        plt.xlabel('Epoch', fontsize=fontsize_2)
        plt.tight_layout() 
        fig.savefig(savepath + ".png")
        plt.close()
                 
