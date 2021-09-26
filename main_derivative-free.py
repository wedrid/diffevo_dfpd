from Dataset import * 
import random
import numpy as np
from derivative_free_PD import * 
from regressione_lineare import * 
from datetime import datetime
import json
import pickle 
import time
import tqdm

#TODO check correcteness of should_exit and embedd within the code.

results_dictionary = []

il_dataset = "servo"
#alcune inits
#numVar = 12
#–constraint = 6
filename = f"{il_dataset}_datasaver-"+str(datetime.now())[0:16].replace(':','-').replace(' ', 'at')+"_Randomized.pickle"
local_searches_counter = 0
stuck_counter = 0
previous_fvalues = None
should_exit_history = None

def should_exit(pop_fvalues):
    '''
    :param list pop_fvalues: list containing all the fvalues for the current population. 
    returns None or a string with description of why it exited. If return is not None then the algorithm should terminate
    '''
    global results_dictionary
    global previous_fvalues
    global stuck_counter
    global local_searches_counter
    global should_exit_history

    flag = False
    differences_sum = 0
    for item in pop_fvalues: 
        for subitem in pop_fvalues:
            diff = abs(item - subitem)
            differences_sum += diff
    print(f"should exit: {differences_sum}")
    should_exit_history.append(differences_sum)
    print(f"stuck counter: {stuck_counter}")
    #print(f"should exit: {differences_sum}")
    if differences_sum <= 1e-4:
        return "Exited: because the sum of the differences were less than 1e-4"

    if previous_fvalues is None: 
        previous_fvalues = pop_fvalues.copy()
    else: 
        if previous_fvalues == pop_fvalues:
            stuck_counter += 1
            if stuck_counter >= 350: 
                return "Exited: population was stuck with the same fvalues for 100 epochs"
        else: 
            previous_fvalues = pop_fvalues.copy()
            stuck_counter = 0
    
    if local_searches_counter >= 20000: 
        return "The number local searches exceeded 20000 calls"
    #local_searches_counter += 1
    print(f"Local searches: {local_searches_counter}")
    return False

def local_search2(x0 = None, exact = False):
    global local_searches_counter
    local_searches_counter += 1
    pd = DFPenaltyDecomposition(func, tau_zero=1, x_0=x0, gamma=1.1, max_iterations=9999999999999999, save=False, l0_constraint=constraint)
    pd.startWithRandomizedStep()
    fval = pd.resultVal
    fargmin = pd.resultPoint
    #print(fargmin)
    return fval, fargmin

def differential_evolution(F = 1, CR = 1, P = 50, n = None, bound = 100, numero_epoche = 1, exact = False):
    assert (F > 0 and F < 2)
    assert (CR >= 0 and CR <= 1)
    assert P > 0 # P is the population size
    assert n is not None
    global local_searches_counter
    global stuck_counter
    global should_exit_history
    stuck_counter = 0
    local_searches_counter = 0
    should_exit_history = []
    datasaver = {}
    datasaver['type'] = "version1"
    datasaver['epochs_saver'] = []
    print(f"Bound: {bound} \n Constraint: {constraint}")
    population = ((np.random.rand(n, P) - 0.5)*2)*bound # default è 1, le componenti possono andare solo tra -1 e 1
    # for each element of the population do a local search
    population_fvalues = [None] * P
    start = time.time()
    print("Initialization ..")
    
    for i in tqdm.tqdm(range(0, P)):
        population_fvalues[i], _ = local_search2(x0 = population[:,i].reshape(n, 1), exact=exact)
        #print(f"Initialized {i} of {P-1}")

    print(population_fvalues)


    ordered_P = np.arange(P)
    
    for epoca in range(0, numero_epoche):
        epoch_blackbox = {'epoca': epoca}
        print(f"Epoca {epoca}")
        for i in range(0, P): 
            i_ = random.randint(0, n-1) #perchè i vettori iniziano da 0
            keys = generatek012(i, ordered_P)
            trial = (population[:,keys[0]] + F*(population[:, keys[1]] - population[:, keys[2]])).reshape((n,1))
            for j in range(0, n):
                if j != i_:
                    if random.uniform(0,1) < CR: 
                        trial[j] = population[j][i] # TODO check correctness
            
            trial_fvalue, _ = local_search2(x0 = trial, exact=exact)
            if trial_fvalue < population_fvalues[i]:
                # a = population[:,1]
                population[:,i] = trial.reshape(n,)
                population_fvalues[i] = trial_fvalue
                
            print(f"Epoca {epoca}, individuo {i}. Fvalues: \n {population_fvalues}")
            esci = should_exit(population_fvalues)
            if esci: 
                break
        epoch_blackbox['current_population'] = population.copy()
        epoch_blackbox['current_population_fvalues'] = population_fvalues.copy()
        datasaver['epochs_saver'].append(epoch_blackbox.copy())
        if esci: 
            break
        
    elapsed = time.time() - start
    datasaver['elapsed'] = elapsed
    datasaver['exit_reason'] = esci
    datasaver['should_exit_history'] = should_exit_history.copy()
    datasaver['final_population'] = population.copy()
    with open('version1' + filename, 'wb') as file:
        pickle.dump(datasaver, file)
        

    return population_fvalues, population



def generatek012(i, pvec):
    np.random.shuffle(pvec)
    temp = pvec[0:3]
    while i in temp:
        #print(temp)
        np.random.shuffle(pvec)
        temp = pvec[0:3]
    return temp

def prepare_linear_regression(dataset_name = "housing"):
    data = Dataset(name=dataset_name, directory="./datasets/")
    X, Y = data.get_dataset()
    Y = np.array([Y])
    Y = Y.transpose()
    print("Shape X " + str(X.shape)) 
    print("Shape Y " + str(Y.shape)) 
    fun = RegressioneLineare(X, Y)
    #numVar = fun.n
    constraint = int(fun.n/2)
    return (fun, fun.n, constraint)



def differential_evolution2(F = 0.5, CR = 1, P = 50, n = None, bound = 100, numero_epoche = 1, exact = False):
    assert (F > 0 and F < 2)
    assert (CR >= 0 and CR <= 1)
    assert P > 0 # P is the population size
    assert n is not None
    global local_searches_counter 
    global stuck_counter 
    global should_exit_history
    stuck_counter = 0
    local_searches_counter = 0
    should_exit_history = []
    datasaver = {}
    datasaver['type'] = "version2"
    datasaver['epochs_saver'] = []
    print(f"Bound: {bound} \n Constraint: {constraint}")
    population = ((np.random.rand(n, P) - 0.5)*2)*bound # default è 1, le componenti possono andare solo tra -1 e 1
    # for each element of the population do a local search
    population_fvalues = [None] * P
    start = time.time()
    print("Initialization ..")
    for i in range(0, P):
        population_fvalues[i], _  = local_search2(x0 = population[:,i].reshape(n, 1), exact=exact)
        print(f"Initialized {i} of {P-1}")

    print(population_fvalues)


    ordered_P = np.arange(P)
    for epoca in range(0, numero_epoche):
        epoch_blackbox = {'epoca': epoca}
        #print(f"Epoca {epoca}")
        for i in range(0, P):
            i_ = random.randint(0, n-1) #perchè i vettori iniziano da 0
            keys = generatek012(i, ordered_P)
            trial = (population[:,keys[0]] + F*(population[:, keys[1]] - population[:, keys[2]])).reshape((n,1))
            for j in range(0, n):
                if j != i_:
                    if random.uniform(0,1) < CR: 
                        trial[j] = population[j][i] # TODO check correctness
            
            trial_fvalue, trial_argmin = local_search2(x0 = trial, exact=exact)
            if trial_fvalue < population_fvalues[i]:
                # a = population[:,1]
                population[:,i] = trial_argmin.reshape(n,)
                population_fvalues[i] = trial_fvalue
                
            print(f"Epoca {epoca}, individuo {i}. Fvalues: \n {population_fvalues}")
            esci = should_exit(population_fvalues.copy())
            if esci: 
                break
        epoch_blackbox['current_population'] = population.copy()
        epoch_blackbox['current_population_fvalues'] = population_fvalues.copy()        
        datasaver['epochs_saver'].append(epoch_blackbox.copy())
        if esci: 
            break
    elapsed = time.time() - start
    datasaver['elapsed'] = elapsed
    datasaver['exit_reason'] = esci
    datasaver['should_exit_history'] = should_exit_history.copy()
    datasaver['final_population'] = population.copy()
    with open('version2' + filename, 'wb') as file:
        pickle.dump(datasaver, file)
    print(population)
    return population_fvalues, population


### main
# n = function.numberOfVariables()
func, numVar, constraint = prepare_linear_regression(il_dataset)
#func = QuadraticTestProblem(N=numVar, n = 4, s = 3, m = 10, old_problem = True)

fvalues2, pop2 = differential_evolution2(n = numVar, P=30, CR = 0.5, numero_epoche=1000000, exact=False, bound = 1000) #ATTENZIONE AL BOUND, se test "quadratico" limitare!
fvalues1, pop1 = differential_evolution(n = numVar, P=30, CR = 0.5, numero_epoche=100000000, exact=False, bound = 1000) #ATTENZIONE AL BOUND, se test "quadratico" limitare!

#print(pop1)

