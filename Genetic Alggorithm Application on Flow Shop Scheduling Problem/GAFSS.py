import numpy as np
import time as cl
from copy import deepcopy
import csv
import itertools
from math import factorial
from networkx.algorithms.flow.mincost import cost_of_flow



#Create an initial random population from given jobs with given size 
def createInitialPopulation(size,job):
    sol = range(job) #Create an initial sequence
    pop =[] #Create empty population
    while True: #Do this until desired size is reached
        if len(pop) == size:
            break
        sol = range(job) #Create initial sequence
        np.random.shuffle(sol) #Shuffle the sequence
        if sol in pop:
            continue
        pop.append(sol)#Add solution to the population
    return pop     

#Calculates the objective value given the solution and the time
def calculateObjective(solution,time):#Calculates the makespan of the given solution and time
    matrix = np.zeros((time.shape[1],time.shape[0])) #Create empty matrix
    for i in range(len(solution)):#For each element of the solution determine the ending time of the each machine process
        if i is 0:
            for m in range(time.shape[0]):
                if m is 0:
                    matrix[i,m]=time[m,solution[i]]                 
                else:
                    matrix[i,m]=matrix[i,m-1]+time[m,solution[i]]
        else:
            for m in range(time.shape[0]):
                if m is 0:
                    matrix[i,m]=matrix[i-1,m]+time[m,solution[i]]
                else:
                    matrix[i,m]=max(matrix[i,m-1]+time[m,solution[i]],matrix[i-1,m]+time[m,solution[i]])
    return matrix[len(solution)-1,time.shape[0]-1]      

#Determines the best solution in a given population (smallest makespan)
def determineBestSolution(population,time,objectives=None):
    if objectives is None:
        bestTime = 1000000 #initial time
        bestSol=[] #create an empty solution
        for p in population: #for all solutions in given population calulate the best solution and the best objective values
            tempObjective=calculateObjective(p, time) 
            if tempObjective<bestTime:
                bestTime=tempObjective
                bestSol=p
    else:
        index = np.argmin(objectives)
        bestSol=population[index]
        bestTime=objectives[index]
    return bestSol,bestTime

#Choose given number of good individuals in a given population
def chooseGoodIndividuals(population,time,goodIndividualSize):
    #Calculate costs of all elements in populations
    allcosts=[]
    for p in population:
        allcosts.append(calculateObjective(p, time))
    #Calculate probabilities
    inversAllCosts =[1/float(i) for i in allcosts]    
    norm = [float(j)/sum(inversAllCosts) for j in inversAllCosts]
    #Choose the good individuals
    order = np.random.choice(range(len(population)),size=goodIndividualSize,replace=False,p=norm)
    return [population[i] for i in order]

#Breed and create new children using one point crossover
def breeding(goodIndividuals,crossOverProb,time):
    newSolutions=[]
    for i in range(len(goodIndividuals)/2):
        #create random number
        ri = np.random.random()
        #Create new children if random number < prob. of crossover 
        if ri<crossOverProb:
            #make cross over
            c1,c2 = crossOver(goodIndividuals[2*i],goodIndividuals[2*i+1],time)
            newSolutions.append(c1)
            newSolutions.append(c2)
        #else take the same solutions
        else:
            newSolutions.append(goodIndividuals[2*i])
            newSolutions.append(goodIndividuals[2*i+1])
    return newSolutions        


def crossOver(solution1,solution2,time):
    r=np.random.randint(1,len(solution1))
    solution1_1= [solution1[i] for i in range(r)]
    solution1_2= [solution1[i] for i in range(r,len(solution1))]
    solution2_1= [solution2[i] for i in range(r)]
    solution2_2= [solution2[i] for i in range(r,len(solution2))]
    
    offspring1 = list(solution1_1)
    for s in range(len(solution2)):
        if solution2[s] in offspring1:
            continue
        else:
            offspring1.append(solution2[s])
    offspring2 = list(solution2_1)
    for s in range(len(solution1)):
        if solution1[s] in offspring2:
            continue
        else:
            offspring2.append(solution1[s])
    #print offspring1
    #print offspring2
    return offspring1,offspring2

#mutation
def mutation(solutions,prob,time):
    #for each element in solution space
    for s in solutions:
        #if prob>random number
        if np.random.random()<prob:
            #determine index 1
            index1 = np.random.randint(0,len(s))
            #determine index 2
            index2 = np.random.randint(0,len(s))
            #change the elements in given indexes
            temp = s[index1]
            s[index1]=s[index2]
            s[index2]=temp
    return solutions

#Change the bad members of population with new children
def substitutePopulation(pop,child,time):
    allcosts=[]
    #calculate cost of each element in population
    for p in pop:
        allcosts.append(calculateObjective(p, time))
    #determine probabilities of choosing the bad members
    norm = [float(j)/sum(allcosts) for j in allcosts]
    order = np.random.choice(range(len(pop)),size=len(child),replace=False,p=norm)
    #choose with probability
    pop = [i for j, i in enumerate(pop) if j not in order]
    for c in child:
        pop.append(c)
    return pop

#Binary tournament selection
def binaryTournamentSelection(pop,time):
    selectedSolutions=[] #Create empty list of solutions
    while True:
        sol1=None
        sol2=None
        #Choose a solution randomly
        index1=np.random.choice(range(len(pop)),size=2,replace=False)
        #Take the better solution
        if calculateObjective(pop[index1[0]], time)<calculateObjective(pop[index1[1]], time):
            sol1=pop[index1[0]]
        else:
            sol1=pop[index1[1]]
        #Choose another solution randomly
        index2=np.random.choice(range(len(pop)),size=2,replace=False)
        #Take the better solution
        if calculateObjective(pop[index2[0]], time)<calculateObjective(pop[index2[1]], time):
            sol2=pop[index2[0]]
        else:
            sol2=pop[index2[1]]
        if sol1 is not sol2:
            break
    selectedSolutions.append(sol1)
    selectedSolutions.append(sol2)
    return selectedSolutions

#Calculate average objective of the given population
def calculateAverageObjective(population,time):
    tt=0
    for p in population:
        tt=tt+calculateObjective(p, time)
    return tt/len(population)

def getObjectives(pop,time):
    objectiveValues=[]
    for p in pop:
        objectiveValues.append(calculateObjective(p, time))
    return objectiveValues


def geneticAlgorithm(time,populationSize,goodIndividualSize,crossOverProb,mutationProb,iter,optimal=None,binaryTournament=True,verbose=False,summary=True):
    population=createInitialPopulation(populationSize,time.shape[1])#Create initial population
    start = cl.clock()
    objectives = getObjectives(population,time)
    bestSolution,bestObjective = determineBestSolution(population,time,objectives)#Calculate best solution and objective for initial population
    gap =None
    if verbose is True:
        print "Iteration:",0,"Solution:",bestSolution,"Objective:",bestObjective
    #For a given number of iterations do the following
    for i in range(iter):
        #Get the good individuals
        if binaryTournament:
            goodIndividuals = binaryTournamentSelection(population,time)
        else:
            goodIndividuals = chooseGoodIndividuals(population,time,goodIndividualSize)

        #Create children
        children=breeding(goodIndividuals,crossOverProb,time)
        #Mutate the children
        children=mutation(children,mutationProb,time)
        #Update the population
        population=substitutePopulation(population,children,time)
        #
        recentBestSolution,recentBestObjective=determineBestSolution(children, time)#Calculate best solution and objective for new children
        #Update the best solution
        if recentBestObjective<bestObjective:
            bestSolution=recentBestSolution
            bestObjective=recentBestObjective
            if verbose is True:
                print "Iteration:",0,"Solution:",bestSolution,"Objective:",bestObjective     
    
    #Calculate average objective
    averageObjective=calculateAverageObjective(population,time)
    if optimal is not None:
        gap = 100*(bestObjective-optimal)/optimal #Calculate percent gap
        gap = round(gap,2)
    elapsed = cl.clock()-start 
    if summary is True:
        print" "
        print "Summary"
        print "Best solution is found at iteration",iter,"in",elapsed,"seconds"
        print "Best Solution is",bestSolution,"with objective value of",bestObjective
    return bestSolution,bestObjective,averageObjective,gap,elapsed

def parameterSelection(data,popSize,goodIndividualSize,crossOverProbs,mutationProbs,iterSize,verbose=False,summary=True):
    bestCost = 1000000000
    for iteration in iterSize:
        for p in popSize:
            for gI in goodIndividualSize:
                for co in crossOverProbs:
                    for mp in mutationProbs:
                        res,cost,averageCost,gap,elapsed=geneticAlgorithm(data, p, gI, co, mp, iteration,verbose=False,summary=False)
                        if cost<bestCost:
                            bestCost = cost
                            bestParameters = [p,gI,co,mp,iteration]
                            if verbose is True:
                                print ""
                                print "PopulationSize:",p,"GoodIndividualSize:",gI,"Cross-over Probs:",co,"Mutation Prob:",mp,"Iteration:",iteration
        
    if summary is True:
        print "Parameter Selection"
        print "PopulationSize:",bestParameters[0],"GoodIndividualSize:",bestParameters[1],"Cross-over Probs:",bestParameters[2],"Mutation Prob:",bestParameters[3],"Iteration:",bestParameters[4]               
    return bestParameters

def demo():
    #Instance 1 Data 
    job1=8 
    machine1=4
    
    temp='0 375 1  12 2 142 3 245 0 632 1 452 2 758 3 278 0 12 1 876 2 124 3 534 0 460 1 542 2 523 3 120 0 528 1 101 2 789 3 124 0 796 1 245 2 632 3 375 0 532 1 230 2 543 3 896 0 14 1 124 2 214 3 543'
    T2 = [int(x) for x in temp.split()]
    T2 = np.array(T2)
    T2 = np.reshape(T2,[8,8])
    time1 = []
    time1.append(T2[:,1])
    time1.append(T2[:,3])
    time1.append(T2[:,5])
    time1.append(T2[:,7])
    time1=np.array(time1)
    time1=np.reshape(time1,[4,8])
    
    bestParameters = parameterSelection(time1, [100,200], [2,4,6],[0.5,0.3,0.8],[1,0.5,0.1],[100,200], verbose=False, summary=True)
    geneticAlgorithm(time1, bestParameters[0], bestParameters[1], bestParameters[2], bestParameters[3], bestParameters[4], optimal=None, binaryTournament = False, verbose=False, summary=True)

demo()