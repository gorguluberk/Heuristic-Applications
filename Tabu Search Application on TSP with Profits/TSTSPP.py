import numpy as np
import time as cl

#Computes the euclidean distance between the points
def calculateEuclideanDistance(x1,x2,y1,y2): 
    return round(np.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)),2)

 #Creates a distance matrix which contains the distances between each node pairs
def createDistanceMatrix(x,y):
    distanceMatrix = list() 
    for i in range(len(x)):
        temp = list()
        for j in range(len(x)):
            temp.append(calculateEuclideanDistance(x[i],x[j],y[i],y[j]))
        distanceMatrix.append(temp)
    distanceMatrix=np.asarray(distanceMatrix)
    return distanceMatrix

#Calculates the distance traveled for given solution
def calculateTourLength(solution,distanceMatrix):
    tourLen=distanceMatrix[solution[len(solution)-1],0]
    for i in range(len(solution)-1):
        tourLen=tourLen+distanceMatrix[solution[i],solution[i+1]]
    return tourLen  

#Calculates the total profit of the visited nodes
def calculateProfit(solution,profit): 
    p = 0
    for s in solution:
        p=p+profit[s]
    return p

#Calculates the objective value of the given solution
def calculateObjective(solution,distanceMatrix,profit): 
    return calculateProfit(solution, profit)-calculateTourLength(solution, distanceMatrix)

#Creates random solution with at 'most nodeNumber' nodes
def createRandomIndividual(nodeNumber):
    sol=[0]
    r = np.random.randint(1,nodeNumber)
    temp = np.random.choice(range(1,nodeNumber),r-1,replace=False)
    sol=sol+list(temp)
    return sol

#Creates neighborhood
def move(solution,distanceMatrix,profits): 
    
    ####Insertion
    neighborhood = []
    tempTabuList = []
    for j in range(1,len(profits)):
        if j not in solution:
            for i in range(1,len(solution)):
                p1 = list(np.array(solution)[(range(i))])
                p2 = list(np.array(solution)[(range(i,len(solution)))])
                p1.append(j)
                p = p1 + p2
                neighborhood.append(p)
                tempTabuList.append([1000,i,j])
        else:
            continue
    
    ####Deletion
    for i in range(1,len(solution)):
        s = list(solution)
        tempTabuList.append([-1*s[i]])
        s.remove(s[i])
        neighborhood.append(s)
    
    ####Swap
    for i in range(1,len(solution)):
        for j in range(1,len(solution)):
            if i is not j:
                s = list(solution)
                temp = s[i]
                s[i]=s[j]
                s[j]=temp
                neighborhood.append(s)
                tempTabuList.append([s[i],s[j]])
    ####3-opt
    for i in range(1,len(solution)):
        for j in range(1,len(solution)):
            if i is not j:
                s = list(solution)
                t = s[i]
                s.remove(s[i])
                temp = [s[k] for k in range(j)]
                temp2 = [s[k] for k in range(j,len(s))]
                temp.append(t)
                s = temp+temp2
                neighborhood.append(s)
                tempTabuList.append([2000,t])
                
    ####2-opt
    for i in range(1,len(solution)-2):
        for j in range(i+2,len(solution)):
            s = list(solution)
            s1 = [s[k] for k in range(i)]
            s2 = [s[k] for k in range(i,j)]
            s3 = [s[k] for k in range(j,len(s))]
            s2.reverse()
            ss = s1 + s2 + s3
            neighborhood.append(ss)
            tempTabuList.append(['2-opt',i,j])
    return neighborhood,tempTabuList

#Calculates objectives of the all given neighborhood
def calculateAllObjectives(population,distanceMatrix,profits): 
    objectives=[]
    for p in population:
        objectives.append(calculateObjective(p, distanceMatrix, profits))
    return objectives

#Updates the tabu list in each iteration
def updateTabu(tabuList,tabuTenure,chosenTabu,initialTenure): 
    toRemoveList=[]
    if len(tabuList) != 0:
        for i in range(len(tabuTenure)):
            tabuTenure[i] = tabuTenure[i]-1
            if tabuTenure[i] == 0:
                toRemoveList.append(i)

    tabuTenure = [x for i,x in enumerate(tabuTenure) if i not in toRemoveList]
    tabuList = [x for i,x in enumerate(tabuList) if i not in toRemoveList]
    
    tabuList.append(chosenTabu)
    tabuTenure.append(initialTenure)
    return tabuList,tabuTenure

#Checks whether a move is in tabu or not
def checkTabu(tempTabuList,tabuList): 
    for i in tabuList:
        if set(tempTabuList) == set(i):
            return True
    return False

#Updates long term memory in each iteration
def updateLongTermMemory(p,memory,memoryNum): 
    if p in memory:
        memoryNum[memory.index(p)]=memoryNum[memory.index(p)]+1
    else:
        memory.append(p)
        memoryNum.append(1)
    return memory,memoryNum

#If a solution in long term memory penalizes its objective function accordingly
def addLongTermObjective(p,memory,memoryNum,treshold): 
    if p in memory:
        return min(treshold-memoryNum[memory.index(p)],0)*2
    else:
        return 0
    
#Updates the objectives of the neighborhood according to their occurance in long term memory
def updateLongTermMemoryObjective(population,memory,memoryNum,treshold,objectives): 
    tempObjectives = list(objectives)
    for i in range(len(population)):
        tempObjectives[i] = tempObjectives[i]+ addLongTermObjective(population[i], memory, memoryNum, treshold)
    return tempObjectives

 #Determines the best solution in a given neighborhood
def determineBestSolution(population,distanceMatrix,profits,tempTabuList,tabuList,bestObjective,memory,memoryNum,treshold,objectives=None):
    if objectives is None:
        bestTime = -10000 
        bestSol=[]
        for i in range(len(population)): 
            p = population[i]
            if checkTabu(tempTabuList[i],tabuList) is True:
                continue
            tempObjective =calculateObjective(p, distanceMatrix, profits) + addLongTermObjective(p,memory,memoryNum,treshold)
            if tempObjective>bestTime:
                bestTime=tempObjective
                bestSol=p
    else:
        while True:
            updatedObjective = updateLongTermMemoryObjective(population,memory,memoryNum,treshold,objectives)
            index = np.argmax(updatedObjective)
            if objectives[index]<=bestObjective:
                if checkTabu(tempTabuList[index],tabuList) is True:
                    population.remove(population[index])
                    objectives.remove(objectives[index])
                    tempTabuList.remove(tempTabuList[index])
                    continue
            bestSol=population[index]
            bestTime=objectives[index]
            chosenTabu = tempTabuList[index]
            break
    return bestSol,bestTime,chosenTabu

#Forgetting long term memory for once in given number of iteration
def forget(memoryNum,i,k):
    temp = list(memoryNum)
    if i%k == 0:
        for i in range(len(temp)):
            temp[i]= 0
    return temp

#Creates the remaining set of solutions (Nodes that are not in the given solution)
def createRemaining(solution,completeSet):
    remaining = []
    for i in completeSet:
        if i in solution:
            continue
        else:
            remaining.append(i)
    return remaining
            
#Greedy construction heuristic
def greedyInitialSolution(size,distanceMatrix,profit):
    completeSet = range(size)
    solution = [0]
    while True:
        notInSet = createRemaining(solution, completeSet)
        currentCost = calculateObjective(solution, distanceMatrix, profit)
        bestCost = -100000000
        for s in notInSet:
            temp = list(solution)
            temp.append(s)
            tempCost = calculateObjective(temp, distanceMatrix, profit)
            if tempCost > bestCost:
                bestCost = tempCost
                bestSolution = list(temp) 
        if bestCost > currentCost:
            solution = bestSolution
        else:
            break 
    return solution

#Prepares the result format before printing
def resultPreparation(solution,time,distanceMatrix,profit):
    obj = calculateObjective(solution, distanceMatrix, profit)
    newSolution = []
    for s in solution:
        newSolution.append(s+1)
    newSolution.remove(1)
    customerVisited=len(newSolution)
    newTime = round(time,2)
    
    print obj,customerVisited,newSolution,newTime

#Fixes the index issue
def modifiedResult(solution):
    newSolution = []
    for s in solution:
        newSolution.append(s+1)
    return newSolution

def validate(solution,distanceMatrix,profit):
    newSolution = [0]
    for s in solution:
        newSolution.append(s-1)
    return round(calculateObjective(newSolution, distanceMatrix, profit),2),len(newSolution)-1
    
#Tabu Search algorithm
def TabuSearch(solution,distanceMatrix,profits,it,tabuTime,treshold,k,verbose=False,summary=True):
    #Initialization
    start = cl.time()
    bestObjective = calculateObjective(solution, distanceMatrix, profits)
    initialObjective = bestObjective
    bestSolution = list(solution)
    tempSolution = list(solution)
    initialSolution = list(bestSolution)
    if verbose is True:
        print "Iteration:",0,"Solution:",bestSolution,"Objective:",bestObjective
    tabuList=[]
    tabuTenure=[]
    memory=[]
    memoryNum=[]
    bestElapsed=0
    shakeP = 0
    lastObj = 100000
    counter=0
    controller = 0
    y=0
    #Main
    while True:
        y=y+1
        #Create neighborhood
        n,tempTabuList = move(tempSolution,distanceMatrix,profits)
        obj = calculateAllObjectives(n, distanceMatrix, profits)
        objTemp = list(obj)
        n=[i for (j,i) in sorted(zip(obj,n),reverse=True)]
        obj=sorted(obj,reverse=True)
        tempTabuList=[i for (j,i) in sorted(zip(objTemp,tempTabuList),reverse=True)]
        counter=counter+1
        
        #If we can not improve solutions for certain number of iterations we apply shaking
        #Shaking
        if shakeP>100:
            tempN,ttl =  move(tempSolution,distanceMatrix,profits)
            tempSolutionIndex = np.random.choice(range(len(tempN)))
            tempSolution = tempN[tempSolutionIndex]
            tempObj = calculateObjective(tempSolution, distanceMatrix, profits)
            chosenTabu=ttl[tempSolutionIndex]
            controller = controller+1
            shakeP=0
        #Choosing best solution
        else:
            tempSolution,tempObj,chosenTabu = determineBestSolution(n, distanceMatrix, profits,tempTabuList,tabuList,bestObjective,memory,memoryNum,treshold,obj)
       
        #Update tabu list and long term memory
        tabuList,tabuTenure=updateTabu(tabuList,tabuTenure,chosenTabu,tabuTime)
        memory,memoryNum=updateLongTermMemory(tempSolution, memory, memoryNum)
        
        shakeP = shakeP+1
        lastObj = tempObj
        
        #Forget long term memory after certain number of iterations (in order to speed up)
        if counter == 50:
            memory=[]
            memoryNum=[]
            counter=0
        
        if controller == 2:
            break
        
        #Check the incumbent solution
        print tempObj
        if tempObj>bestObjective:
            controller = 0
            counter = 0
            shakeP = 0
            bestSolution=list(tempSolution)
            bestObjective=tempObj
            bestElapsed = cl.time()-start
            if verbose is True:
                print "Iteration:",0,"Solution:",bestSolution,"Objective:",bestObjective
            memory=[]
            memoryNum=[]
        fullElapsed = cl.time() -start
    if summary is True:
        print" "
        print "Summary"
        print "Best solution is found at iteration",y,"in",fullElapsed,"seconds"
        print "Initial Solution is",initialSolution,"with objective value of",initialObjective
        print "Best Solution is",bestSolution,"with objective value of",bestObjective

    return bestSolution,bestObjective,bestElapsed,fullElapsed

def demo():
    temp='37    49    52    20    40    21    17    31    52    51    42    31    5    12    36    52    27    17    13    57    62    42    16    8    7    27    30    43    58    58    37    38    46    61    62    63    32    45    59    5    10    21    5    30    39    32    25    25    48    56    30'
    H51x = [int(x) for x in temp.split()]
    temp='52    49    64    26    30    47    63    62    33    21    41    32    25    42    16    41    23    33    13    58    42    57    57    52    38    68    48    67    48    27    69    46    10    33    63    69    22    35    15    6    17    10    64    15    10    39    32    55    28    37    40'
    H51y = [int(x) for x in temp.split()]
    temp='0    27    31    26    17    18    32    29    20    18    27    19    24    30    21    23    34    19    29    18    30    22    16    20    13    26    31    19    14    24    28    10    33    22    20    13    25    33    28    16    15    22    18    25    31    26    17    28    12    19    14'
    H51p = [int(x) for x in temp.split()]
    
    distanceMatrix = createDistanceMatrix(H51x, H51y)
    initialSolution = greedyInitialSolution(10, distanceMatrix, H51p)
    TabuSearch(initialSolution, distanceMatrix, H51p, 100, 5, 10, 100)

