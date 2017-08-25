import numpy as np
import time as cl
import csv
import itertools
import os

def calculateEuclideanDistance(x1,x2,y1,y2): #Computes the euclidean distance between the points
    return round(np.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)),2)

def createDistanceMatrix(x,y): #Creates a distance matrix which contains the distances between each node pairs
    distanceMatrix = list() #create empty list
    for i in range(len(x)):#for each two points calculate the euclidean distance and append into the matrix
        temp = list()
        for j in range(len(x)):
            temp.append(calculateEuclideanDistance(x[i],x[j],y[i],y[j]))
        distanceMatrix.append(temp)
    distanceMatrix=np.asarray(distanceMatrix)
    return distanceMatrix

def createRandomSolution(solutionSize,instanceSize): #Creates random initial solutions
    mylist = range(instanceSize) #creates a list containing the id's of the facilities
    solutionSet = np.random.choice(mylist,size=solutionSize,replace=False) #Randomly chooses solutionSize amount of number from the list 
    return solutionSet
 

def CalculateCostMatrix(distanceMatrix,demands): #Calculates the cost matrix using the current solution, demands and distance matrix
    costMatrix = np.zeros((np.shape(distanceMatrix))) #Make a copy of distance matrix
    for i in range(len(distanceMatrix)): #Calculates cost matrix for construction heuristic (no initial solution)
        for j in range(len(distanceMatrix)): 
            costMatrix[i,j] = distanceMatrix[i,j]*demands[j]
    return costMatrix

def UpdateCostMatrix(costMatrix,solutionSet): #Calculates the cost matrix using the current solution, demands and distance matrix
    costM = costMatrix
    for i in range(len(costM)):
        for j in range(len(costM)):
            costM[i,j]=min(costM[i,j],min(costM[solutionSet,j]))   
    return costM


def correctSolution(solutionSet):#Python uses idexing starting from 0. Each operation is done to the lists starting from 0. 
    #However, facility id's do not start from 0. This function corrects the facility id's after finding the solution. 
    solutionSet = [x+1 for x in solutionSet]
    return sorted(solutionSet)


def GreedyHeuristic(distanceMatrix,demands,facilityNumber):
    completeSet = range(0,len(distanceMatrix)) #Make a list containing all facility id's 
    solutionSet = list() #initialize the solution set
    remainingSet = range(0,len(distanceMatrix)) #initialize the remaining set containing the facility id's that are not in the solution set
    
    costMatrix = CalculateCostMatrix(distanceMatrix, demands) #Obtain the initial cost matrix
    totalCosts = calculateCost(costMatrix)
    solutionSet.append(totalCosts.index(min(totalCosts))) #Choose the facility with minimum cost and add to the solution Set
    remainingSet.remove(totalCosts.index(min(totalCosts))) #Remove the selected facility from the remaining set

    #Find facilities until desired number of facilities are added to the solution set 
    for k in range(facilityNumber-1):
        costMatrix = UpdateCostMatrix(costMatrix, solutionSet) #Update the cost matrix according to the solution set
        totalCosts=calculateCost(costMatrix)
        solutionSet.append(totalCosts.index(min([totalCosts[x] for x in remainingSet]))) #Update the solution set
        remainingSet.remove(totalCosts.index(min([totalCosts[x] for x in remainingSet]))) #Update the remaining set
    
    objectiveCost = calculateTotalCost(solutionSet, distanceMatrix, demands)

    return solutionSet,objectiveCost,remainingSet,costMatrix

def calculateCost(costMatrix): #Calculates cost of opening new facility for each facilities and also current cost produced by the solution space
    allCost = [] 
    for j in range(len(costMatrix)): #Cost of adding new facility to the solution set
        allCost.append(sum(costMatrix[j,:]))
    return allCost
def calculateTotalCost(solution,distanceMatrix,demand):#Calculates the current objective value using the given solution set
    cost = 0
    for i in range(len(distanceMatrix)):
        cost = cost+min(distanceMatrix[solution,i])*demand[i]
    return cost


def LocalSearch(initialSolution,x_coordinate,y_coordinate,demand,greedy=False,neighborRadius=1,verbose=False,summary=True): #greedy=False -> Best improvement, greedy=True -> First improvement
    start_time = cl.time()
    distanceMatrix = createDistanceMatrix(x_coordinate, y_coordinate)
    completeSet = range(0,len(distanceMatrix)) #Create list of facilities
    remainingSet = list(completeSet) 
    [remainingSet.remove(x) for x in initialSolution] #Initialize the remaining set
    tempCost=calculateTotalCost(initialSolution,distanceMatrix,demand) #Create a copy of the total cost
    initialCost=tempCost
    
    print "Iteration:",0,"Solution:",initialSolution,"Cost",initialCost
    
    newSolution = list(initialSolution) #create a copy of the initial solution
    iteration = 0 #initialize the iteration counter
    solutionCounter = 0 #initialize the solution counter (# of solution evaluated)
    outNode = None # initialize the outNode(id of node to be removed in each iteration)
    while outNode is not None or iteration is 0: #iterate until there is not a better solution (no change in the solution set)(outnode=None)
        stop = False #break the loop if first improve is used
        outNode = None #intialize the outnode at the beginning of the each iteration
        for k in itertools.combinations(newSolution,neighborRadius):#For all facilitiy k in the solution set, take it out and put all other possible facilities m in the remaining set in one by one. Find the best improvement
            temporarySolution = list(newSolution) 
            for p in range(neighborRadius):
                temporarySolution.remove(k[p]) #remove facility k from solution set
            for m in itertools.combinations(remainingSet,neighborRadius):
                solutionCounter = solutionCounter+1
                for p in range(neighborRadius):
                    temporarySolution.append(m[p])
                totalCost =calculateTotalCost(temporarySolution,distanceMatrix,demand) #calculate the total cost   
                if  tempCost > totalCost: #if total cost is smaller than the previous cost update the temp cost
                    tempCost = totalCost #update the temp cost
                    outNode = k #update the outnode 
                    inNode = m #update the in node
                    if greedy:
                        stop = True
                        break
                for p in range(neighborRadius):
                    temporarySolution.remove(m[p])
            if stop:
                break
            
        iteration = iteration + 1
        print "Iteration:",iteration,"Solution:",temporarySolution,"Cost",tempCost
        #after evaluating all alternatives update the solution set
        try:
            for p in range(neighborRadius):
                newSolution.remove(outNode[p])
                newSolution.append(inNode[p])
        except:
            continue
    elapsed_time =cl.time()-start_time
    
    if summary is True:
        print" "
        print "Summary"
        print "Best solution is found at iteration",iteration,"in",elapsed_time,"seconds"
        print "Initial Solution is",initialSolution,"with objective value of",initialCost
        print "Best Solution is",newSolution,"with objective value of",tempCost
        print "Initial Solution is improved",(initialCost-tempCost)/initialCost*100,"%"
    
    return iteration, newSolution, tempCost, elapsed_time, initialSolution, initialCost


def demo(greedy=False,neighborRadius=1):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    x = np.loadtxt(dir_path+'/x51.dat')[:,1]
    y = np.loadtxt(dir_path+'/y51.dat')[:,1]
    demands = np.loadtxt(dir_path+'/dem51.dat')[:,1]
    
    initialSolution = createRandomSolution(4, len(x))
    iteration, newSolution, tempCost, elapsed_time, initialSolution, initialCost = LocalSearch(initialSolution, x, y, demands, greedy, neighborRadius,False,True)
