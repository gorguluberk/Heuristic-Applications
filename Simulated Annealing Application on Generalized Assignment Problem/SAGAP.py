import numpy as np
import time as cl
from copy import deepcopy
import csv
import itertools

def determineFeasibleSolutions(solution,resource,capacity,k): #Generates feasible solutions in the neighborhood of a given solution
    allPossibleSolutions=[] #Create a list to hold all solutions in the neighborhood
    feasibleSolutions=[] #Create a list to hold feasible solutions in the neighborhood
    agents = resource.shape[0] #determines number of agent
    stop = False
    if k is 1: # controls if neighborhood structure is 1 element change 
        for index in range(len(solution)):
            for a in range(agents):
                if solution[index] != a: #not to create same solution again
                    tempSolution = list(solution) 
                    tempSolution[index]=a
                    allPossibleSolutions.append(tempSolution) #adds generated solution to list
    else: #makes the same operations for the move operator that changes more than 1 element
        l=0
        for index in itertools.combinations(range(len(solution)),k):
            l=l+len(index)
            for a in itertools.product(range(agents),repeat=k):
                stop = False
                tempSolution = list(solution)
                for q in range(k):
                    if solution[index[q]] != a[q]:
                        tempSolution[index[q]]=a[q]
                    else:
                        stop = True
                        break
                if stop is False:
                    allPossibleSolutions.append(tempSolution)                    
    for sol in allPossibleSolutions: #For each solution in the neighborhood checks the feasibility
        if checkFeasibility(sol, resource, capacity) is True:
            feasibleSolutions.append(sol)
    return feasibleSolutions #Returns the feasible solutions

def determineQuickFeasibleSolutions(solution,resource,capacity,k): #Generates feasible solutions in the neighborhood of a given solution
    allPossibleSolutions=[] #Create a list to hold all solutions in the neighborhood
    feasibleSolutions=[] #Create a list to hold feasible solutions in the neighborhood
    agents = resource.shape[0] #determines number of agent
    stop = False
    if k is 1: # controls if neighborhood structure is 1 element change 
        for index in range(len(solution)):
            for a in range(agents):
                if solution[index] != a: #not to create same solution again
                    tempSolution = list(solution) 
                    tempSolution[index]=a
                    allPossibleSolutions.append(tempSolution) #adds generated solution to list
    else: #makes the same operations for the move operator that changes more than 1 element
        l=0
        for index in itertools.combinations(range(len(solution)),k):
            l=l+len(index)
            for a in itertools.product(range(agents),repeat=k):
                stop = False
                tempSolution = list(solution)
                for q in range(k):
                    if solution[index[q]] != a[q]:
                        tempSolution[index[q]]=a[q]
                    else:
                        stop = True
                        break
                if stop is False:
                    allPossibleSolutions.append(tempSolution)                    
    for sol in allPossibleSolutions: #For each solution in the neighborhood checks the feasibility
        if quickFeasibilityCheck(sol, resource, capacity) is True:
            feasibleSolutions.append(sol)
    return feasibleSolutions #Returns the feasible solutions

def determineAllSolutions(solution,resource,capacity,k): #Returns all possible solutions in a neighborhood (basicly same as determineFeasible solution except it does not make feasibility check)
    allPossibleSolutions=[]
    agents = resource.shape[0]
    stop = False
    if k is 1:
        for index in range(len(solution)):
            for a in range(agents):
                if solution[index] != a:
                    tempSolution = list(solution)
                    tempSolution[index]=a
                    allPossibleSolutions.append(tempSolution) 
    else:
        l=0
        for index in itertools.combinations(range(len(solution)),k):
            l=l+len(index)
            for a in itertools.product(range(agents),repeat=k):
                stop = False
                tempSolution = list(solution)
                for q in range(k):
                    if solution[index[q]] != a[q]:
                        tempSolution[index[q]]=a[q]
                    else:
                        stop = True
                        break
                if stop is False:
                    allPossibleSolutions.append(tempSolution)    
    return allPossibleSolutions

def checkFeasibility(sol,resource,capacity): #Checks the feasibility
    caps = calculateRemainingCapacity(sol, resource, capacity) #Calculates the remaining capacities of the agents 
    decision=all(item >= 0 for item in caps) #Checks whether they are negative or not
    return decision #Returns true if it is feasible else false

def calculateRemainingCapacity(solution,resource,capacity): #Calculates the remaining capacities when a solution is selected
    remainingCapacity = list(capacity)
    for j in range(len(solution)):
        remainingCapacity[solution[j]]=remainingCapacity[solution[j]] - resource[solution[j],j] #Decreases the capacity for each work assigned to that agent
    return remainingCapacity
def quickFeasibilityCheck(solution,resource,capacity): #Checks the feasibility without calculating remaining capacity of all agents
    remainingCapacity = list(capacity)
    for j in range(len(solution)):
        remainingCapacity[solution[j]]=remainingCapacity[solution[j]] - resource[solution[j],j]
        if remainingCapacity[solution[j]]<0:
            return False
    return True
def calculateObjectiveCost(solution,cost):#Calculates the total cost for given solution and cost matrix
    objective=0
    for j in range(len(solution)):
        objective=objective + cost[solution[j],j]
    return objective

def createRandomSolution(m1,n1,resource,capacity): #Generates random solutions to be used 
    solutionSet=np.arange(n1) #create an initial set
    cap = list(capacity) 
    jobs = np.arange(n1)
    np.random.shuffle(jobs) #shuffle the indexes of the jobs
    fail=False
    for j in jobs: #for all jobs
        k=0
        while(True): #until the job is assigned
            k=k+1
            index = np.random.random(1) 
            if index < 0.8: # asssign the job to the most available worker
                temp = resource[:,j].tolist().index(min(resource[:,j]))
                if(cap[temp]>resource[temp,j]):
                    solutionSet[j]=temp
                    cap[temp]=cap[temp]-resource[temp,j]      
                    break
                temp =np.random.randint(0,5,1)[0]
                if(cap[temp]>resource[temp,j]):
                    solutionSet[j]=temp
                    cap[temp]=cap[temp]-resource[temp,j]      
                    break
            else: #randomize part, assign the job to a random worker
                temp =np.random.randint(0,5,1)[0]
                if(cap[temp]>resource[temp,j]):
                    solutionSet[j]=temp
                    cap[temp]=cap[temp]-resource[temp,j]      
                    break
            if k>1000:
                fail=True
                break
        if fail is True:
            break
    if fail is True:
        solutionSet=createRandomSolution(m1, n1, resource, capacity) #if it fails to generate try again 
    #print 'Random Solution is Generated' 
    return solutionSet 

def createCompletelyRandomFeasibleSolution(m1,n1,resource,capacity): #create a completely random solution
    while(True):#try until a feasible solution is generated
        solution = np.random.randint(0,m1,n1) #choose n1 random integer between 0 and m1
        feasibility = checkFeasibility(solution, resource, capacity) #check feasiblity of the solution
        if feasibility is True: #if the solution is feasible return the solution
            finalSolution=solution
            break
    return finalSolution

def chooseSolutions(allSolutions,consideredSolutions): #chooses the solutions that we have not considered yet
    for s in consideredSolutions:
        try:
            allSolutions.remove(s)
        except:
            continue
    return allSolutions
def correctSolution(solution): #Since the indexes start from 0 at python after solving it we increment each index
    for i in range(len(solution)):
        solution[i]=solution[i]+1
    return solution
def fastMove(solution,resource,capacity):# Applies move opeator greedily
    genSol = list(solution) #takes the initial solution
    agentNo = len(capacity)
    jobNo = len(solution)
    counter = 0
    while(True): #Loop continuous until a feasible solution is generated
        counter =counter+1
        j = np.random.randint(0,jobNo,1)[0] #choose random job
        a = np.random.randint(0,agentNo,1)[0] #choose random agent
        if genSol[j] == a: #prevents same solution from being generated
            continue
        genSol[j] = a #assign job to agent
        if quickFeasibilityCheck(genSol, resource, capacity) is True: #checks feasiblity greedily
            break
        if counter > 20: #if it has problem finding solution greedily searches all solution space
            ss=determineFeasibleSolutions(solution, resource, capacity,1) #determines all feasible solutions in the neighborhood
            if len(ss)<1:
                genSol = None
                break
            genSol = ss[np.random.randint(0,len(ss),1)[0]] #generate a new solution
            break
    return genSol
                                                         
def simulatedAnnealing(initialSolution,resource,capacity,cost,T,Lmax,minT,r,i_p,f_p,minim=True,fast=False,verbose=False,summary=True):
    start_time = cl.clock()
    solution = list(initialSolution) #make current solution initial solution
    currentCost=calculateObjectiveCost(solution, cost) #calculate the cost of the current solution
    initialCost = currentCost
    if verbose is True:
        print "Iteration:",0,"Solution:",initialSolution,"Cost:",initialCost
    bestSolution=list(solution) #make solution as the best solution
    bestCost=currentCost #make current cost best cost
    T=float(T)#change type of the temperature to float
    k=1 #neighborhood index
    tecrt=0 
    counter=0 #counts cycles 
    #testTemp=0 used for temperature estimation
    #tao=0 used for temperature estimation 
    while(tecrt<minT and T > 0.15):#stopping conditions tecrt<5 or T>0.15
        counter=counter+1 
        j=0
        for l in range(Lmax):#for Lmax number of iterations
            if fast is False: #determines all neighbors before selection
                neighborhood = determineQuickFeasibleSolutions(solution, resource, capacity,k) #determines the neighborhood structure
                if len(neighborhood)<1:
                    continue
                chooseIndex = np.random.randint(0,len(neighborhood),1)[0] #generates random integer
                solutionCandidate = list(neighborhood[chooseIndex]) #selects random solution from the nighborhood
            else:
                solutionCandidate = fastMove(solution,resource,capacity) #applies fast move operation
                if solutionCandidate is None:
                    continue
            objectiveCandidate = calculateObjectiveCost(solutionCandidate, cost)  #calculates the objective cost of the current candidate
            delta = objectiveCandidate-currentCost #calculating delta
            #testTemp=testTemp+abs(delta)
            #tao=tao+1
            if minim is True: #decides minimizing or maximizing
                if delta<=0: #if the candidate solution is better, update the solutions
                    solution = list(solutionCandidate) 
                    currentCost = calculateObjectiveCost(solution, cost) 
                    j = j+1
                else:#else start the probabilistic accpetance part
                    u = np.random.uniform(size=1)
                    if np.exp(-1*float(delta)/T)>u[0]:
                        solution = list(solutionCandidate)
                        currentCost = calculateObjectiveCost(solution, cost)
                        j = j+1
                if currentCost < bestCost:#stores the best solution so far
                    bestCost = currentCost
                    bestSolution = list(solution)
                    #print bestSolution,bestCost
            else:#Same process as above but for maximizing
                if delta>=0:
                    solution = list(solutionCandidate)
                    currentCost = calculateObjectiveCost(solution, cost) 
                    j = j+1
                else:
                    u = np.random.uniform(size=1)
                    if np.exp(1*float(delta)/T)>u[0]:
                        solution = list(solutionCandidate)
                        currentCost = calculateObjectiveCost(solution, cost)
                        j = j+1
                if currentCost > bestCost:
                    bestCost = currentCost
                    bestSolution = list(solution)
                    if verbose is True:
                        print "Iteration:",counter,"Solution:",bestSolution,"Cost:",bestCost
        if float(j)/Lmax>i_p: #if too many acceptance is made in one cycle decrease the temperature to half
            T=T/2
        else:
            T = T*r
        if float(j)/Lmax<=f_p: #if too few acceptance is made triggers the stopping variable
            tecrt=tecrt+1
        else:
            tecrt=0
    elapsed = cl.clock()-start_time 
    if summary is True:
        print" "
        print "Summary"
        print "Best solution is found at iteration",counter,"in",elapsed,"seconds"
        print "Initial Solution is",initialSolution,"with objective value of",initialCost
        print "Best Solution is",bestSolution,"with objective value of",bestCost
        if minim is True:
            print "Initial Solution is improved",float(initialCost-bestCost)/initialCost*100,"%"
        else:
            print "Initial Solution is improved",float(bestCost-initialCost)/initialCost*100,"%"
       
    return bestSolution,bestCost,counter,elapsed

def parameterSelection(facilityNo,demandNo,costs,demands,capacities,T_candidates,Lmax_candidates,minT_candidates,r_candidates, i_p_candidates,f_p_candidates,minim=True,verbose=False,summary=True):
    if verbose is True:
        print "Parameter Selection"
    initialSolution = createRandomSolution(demandNo, facilityNo, demands, capacities)
    bestParameters=[]
    if minim is True:
        bestCost = 10000000000000
    else:
        bestCost = -10000000000
    for T in T_candidates:
        for Lmax in Lmax_candidates:
            for minT in minT_candidates:
                for r in r_candidates:
                    for i_p in i_p_candidates:
                        for f_p in f_p_candidates:
                            res,cost,counter,elapsed = simulatedAnnealing(initialSolution, demands, capacities, costs, T, Lmax, minT, r, i_p, f_p,minim=True,verbose=False,summary=False)
                            if minim is True:
                                if cost<bestCost:
                                    bestCost = cost
                                    bestParameters = [T,Lmax,minT,r,i_p,f_p]
                                    if verbose is True:
                                        print " "
                                        print "T:",T,"Lmax:",Lmax,"minT",minT,"r:",r,"i_p:",i_p,"f_p:",f_p
                                        print "Iteration:",counter,"Solution",res,"Cost",bestCost
                            else:
                                if cost>bestCost:
                                    bestCost = cost
                                    bestParameters = [T,Lmax,minT,r,i_p,f_p]
                                    if verbose is True:
                                        print " "
                                        print "T:",T,"Lmax:",Lmax,"minT",minT,"r:",r,"i_p:",i_p,"f_p:",f_p
                                        print "Iteration:",counter,"Solution",res,"Cost",bestCost
    if summary is True:
        print "Parameter Selection"
        print "T:",bestParameters[0],"Lmax:",bestParameters[1],"minT:",bestParameters[2],"r:",bestParameters[3],"i_p:",bestParameters[4],"f_p:",bestParameters[5]
    return bestParameters

def demo():
    facilityNo=5
    demandNo=25
    costs =np.asarray(map(int,"18 19 19 17 24 25 24 25 25 23 20 21 25 17 25 21 25 19 23 19 20 15 25 23 17 25 17 18 16 18 15 23 20 19 22 23 18 17 16 16 24 16 23 23 24 19 17 15 17 17 17 25 15 23 21 20 24 17 21 22 22 15 18 23 17 22 20 24 19 18 15 15 18 19 19 21 16 25 23 18 21 18 16 21 21 15 21 24 23 24 23 20 25 24 18 19 23 22 22 16 24 16 24 19 16 25 23 25 17 21 21 22 17 25 19 21 23 19 17 24 19 15 20 15 20".split(" ")))
    costs=costs.reshape((facilityNo,demandNo))
    demands=np.asarray(map(int,"25 23 5 13 6 15 24 9 17 11 5 6 8 14 9 9 21 23 13 8 22 20 24 15 20 18 8 5 20 8 7 13 17 9 16 19 11 6 12 25 23 9 21 11 15 24 23 15 21 12 7 25 13 9 16 16 8 17 5 17 10 18 21 25 17 24 20 16 9 18 18 18 16 6 24 25 11 8 7 25 20 24 16 9 15 22 10 17 6 22 11 19 20 14 14 8 18 22 18 22 7 16 20 18 13 10 15 20 5 19 11 6 11 23 15 21 15 20 21 11 9 25 17 18 12".split(" ")))
    demands=demands.reshape((facilityNo,demandNo))
    capacities=np.asarray(map(int,"58 58 62 64 60".split()))

    initialSolution = createRandomSolution(facilityNo, demandNo, demands, capacities)
    parameters = parameterSelection(facilityNo, demandNo, costs, demands, capacities, [10,20,30], [10,20,30], [0.1,0.5,1], [0.1,0.5,0.75], [0.1,0.5,0.75], [0.1,0.5,0.75], minim=True, verbose=False, summary=True)
    simulatedAnnealing(initialSolution, demands, capacities, costs, parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5])

