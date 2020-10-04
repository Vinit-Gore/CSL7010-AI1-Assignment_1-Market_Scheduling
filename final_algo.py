# %%
from dataclasses import dataclass, field
# Context stores all the input data
@dataclass(frozen=True)
class Context:
    K: int
    M: int
    T: int
    C: float
    N: int = field(init=False)
    def __post_init__(self):
        # set immutable value of N
        # workaround https://stackoverflow.com/questions/53756788/how-to-set-the-value-of-dataclass-field-in-post-init-when-frozen-true
        object.__setattr__(self, 'N', self.K * self.T * self.M)


# %% [markdown] id="JEmb8TMNTaLC"
# ## SymmetricMatrix
# A class to efficiently store symmetric matrices.
#
# `m[1,2]` = element at `row=1` and `col=2`.

# %% executionInfo={"elapsed": 963, "status": "ok", "timestamp": 1601112486132, "user": {"displayName": "Jaideep Singh Heer (M20CS056)", "photoUrl": "", "userId": "05136112523110687861"}, "user_tz": -330} id="_7A0V0b8Trji"
# Typings
from typing import Tuple
# Distance matrix data structure
# A symmetric square matrix of size NxN
class SymmetricMatrix:
    def __init__(self, size: int):
        # allocate space for lower half
        self.matrix = [ [None]*(i+1) for i in range(size) ]
        self.size = size
        # allocate space for cache
        self._cache_sum = None
    # get matrix element
    def __getitem__(self, pos: Tuple[int, int]):
        row, col = pos
        # ensure row >= col
        if(row < col):
            row, col = col, row
        return self.matrix[row][col]
    # set matrix element
    def __setitem__(self, pos: Tuple[int, int], newvalue):
        row, col = pos
        # ensure row >= col
        if(row < col):
            row, col = col, row
        self.matrix[row][col] = newvalue
        # invalidate sum cache
        self._cache_sum = None
    # define str function
    def __str__(self):
        s = f"SymmetricMatrix(size={self.size}) ["
        for i in range(self.size):
            s += "\n\t" + " ".join(map(str,self.matrix[i]))
        s += "\n]"
        return s
    # returns the sum of all values
    def sum(self) -> float:
        if(self._cache_sum == None):
            # calc. sum
            self._cache_sum = v = 0.0
            for i in self.matrix:
                v += sum(i)
                self._cache_sum -= i[-1]
            self._cache_sum = v*2
        return self._cache_sum
    # returns average of all values
    def average(self) -> float:
        return self.sum() / self.size**2


# %% [markdown] id="QQpF5DOj3Pc8"
# ## Schedule
# A class that represents a schedule as a table with `T` rows and `m` columns with each cell containing `k` elements.

# %%
from typing import List, Callable, Tuple
import random
class Schedule:
    # Represents a single cell
    class Cell:
        # a func. that takes the cell instance and two int, shop removed and shop added.
        # if both ints are None this means that everything changed and all cache needs to be invalidated.
        CelllUpdateListener = Callable[['Schedule.Cell', int, int], None]
        def __init__(self, context: Context, distances: SymmetricMatrix):
            self.context = context
            self.distances = distances
            # allocate shops
            self.shops = [None] * context.K
            # alocate listeners
            self._listeners = dict()
            # allocate cache
            self._invalidate_cache()
        # invalidate all cache
        def _invalidate_cache(self):
            self._cache_G = None
            # fire cahnge listeners to invalidate cache
            for listener in self._listeners.values():
                listener(self, None, None)
        # replace cell with contents of list
        def _fromList(self, lst: List[int]):
            self.shops = lst
            self._invalidate_cache()
        # getter for G value of just this cell
        @property
        def G(self) -> float:
            if(self._cache_G == None):
                # calc G
                self._cache_G = 0
                for i in range(len(self.shops)):
                    for j in range(i+1, len(self.shops)):
                        self._cache_G += 1 - self.distances[self.shops[i]-1, self.shops[j]-1]
            return self._cache_G
        # returns D between this cell and the other
        def calcD(self, other: 'Schedule.Cell') -> float:
            r = 0
            for s in self.shops:
                for l in other.shops:
                    r += self.distances[s-1,l-1]
            return r
        # returns the expectedG of a cell with k shops
        @staticmethod
        def expectedG(context: Context, avgDist: float) -> float:
            # nC2 * avgDist
            return (context.K*(context.K-1)/2) * avgDist
        # __getitem__ function overloads the [] operator
        # [x] will fetch the shop x in cell
        def __getitem__(self, pos):
            if(type(pos) == tuple):
                if(len(pos) == 1):
                    return self.shops[pos[0]]
            else:
                return self.shops[pos]
        def __setitem__(self, pos, newval):
            oldval = self.shops[pos]
            # replace shop with new val and invalidate cache
            self.shops[pos] = newval
            self._invalidate_cache()
            # fire cahnge listeners
            for listener in self._listeners.values():
                listener(self, oldval, newval)
        # to string
        def __str__(self):
            return " ".join([str(s) for s in self.shops])
        # register change listeners
        def _registerChangeListener(self, listener: CelllUpdateListener, register_id):
            self._listeners[register_id] = listener
        def _unregisterChangeListener(self, register_id):
            self._listeners.pop(register_id, None)
    
    # Represents a single timeslot
    class Timeslot:
        def __init__(self, context: Context, distances: SymmetricMatrix):
            self.context = context
            self.distances = distances
            # allocate cells
            self.cells = [ Schedule.Cell(context, distances) for i in range(context.M) ]
            # register change listeners
            for cell in self.cells:
                cell._registerChangeListener(self.onCellChange, self)
            # allocate cache
            self._cache_D = [None] * context.M
            self._invalidate_cache()
        # called when a cell contents change
        def onCellChange(self, cell: 'Schedule.Cell', removedShop: int, addedShop: int):
            if (removedShop == None or addedShop == None):
                # drastic change in cell, complete recalculation needed
                self._cache_D_valid = False
            # update D cache if D cache is valid
            if(self._cache_D_valid):
                updated_idx = -1
                total_delta = 0
                for i in range(len(self.cells)):
                    if(self.cells[i]==cell):
                        updated_idx = i
                        continue
                    toRem = sum(map(lambda s: self.distances[s-1,removedShop-1], self.cells[i].shops))
                    toAdd = sum(map(lambda s: self.distances[s-1,addedShop-1], self.cells[i].shops))
                    self._cache_D[i] += -toRem + toAdd
                    total_delta += -toRem + toAdd
                self._cache_D[updated_idx] += total_delta
            # invalidate G cache
            self._cache_G = None
        # invalidate all cache
        def _invalidate_cache(self):
            self._cache_D_valid = False
            self._cache_G = None
        # replace timeslot with contents of list
        def _fromList(self, lst: List[int]):
            # apply list to cells
            for i in range(len(self.cells)):
                self.cells[i]._fromList(lst[i*self.context.K : (i+1)*self.context.K])
            self._invalidate_cache()
        # getter for G value of this timeslot
        @property
        def G(self):
            if(self._cache_G == None):
                if(self._cache_D_valid == False):
                    self._buildDCache()
                self._cache_G = sum(map(lambda cell: cell.G, self.cells)) + (sum(self._cache_D) / 2)*self.context.C
            return self._cache_G
        # getter for D value of this timeslot
        @property
        def D(self):
            if(self._cache_D_valid == False):
                self._buildDCache()
            return sum(self._cache_D) / 2
        # returns the total D value of a cell in this timeslot
        def getCellD(self, cell_idx: int):
            # build cache if needed
            if(self._cache_D_valid == False):
                self._buildDCache()
            return self._cache_D[cell_idx]
        # returns the expectedG of a timeslot with m markets and k shops per cell
        @staticmethod
        def expectedG(context: Context, avgDist: float) -> float:
            return Schedule.Timeslot.expectedS(context, avgDist) + context.C*Schedule.Timeslot.expectedD(context, avgDist)
        # returns the expectedS of a timeslot with m markets and k shops per cell
        @staticmethod
        def expectedS(context: Context, avgDist: float) -> float:
            return context.M*Schedule.Cell.expectedG(context, avgDist)
        # returns the expectedD of a timeslot with m markets and k shops per cell
        @staticmethod
        def expectedD(context: Context, avgDist: float) -> float:
            m = context.M # select 2 out of m markets
            return (m*(m-1)/2)*Schedule.Timeslot.expectedCellD(context, avgDist)
        # returns the expectedD of a cell in a timeslot with m markets and k shops per cell
        @staticmethod
        def expectedCellD(context: Context, avgDist: float) -> float:
            return avgDist*(context.K**2)
        # builds D cache
        def _buildDCache(self):
            M = self.context.M
            self._cache_D = [0] * M
            for i in range(M):
                for j in range(i+1, M):
                    D = self.cells[i].calcD(self.cells[j])
                    self._cache_D[i] += D
                    self._cache_D[j] += D
            self._cache_D_valid = True
        # __getitem__ function overloads the [] operator
        # [y,z] will fetch y cell -> shop at position z in the cell
        # [x] will fetch the entire x cell
        def __getitem__(self, pos) -> 'Schedule.Cell':
            # magic
            if (type(pos) == tuple):
                if(len(pos) == 1):
                    return self.cells[pos[0]]
                return self.cells[pos[0]][pos[1:]]
            else:
                return self.cells[pos]
        def __setitem__(self, pos, newval: 'Schedule.Cell'):
            M = self.context.M
            oldval = self.cells[pos]
            self.cells[pos] = newval
            # update registry
            oldval._unregisterChangeListener(self)
            newval._registerChangeListener(self.onCellChange, self)
            # reset pos cache D
            self._cache_D[pos] = 0
            # update D cache if it is valid
            if(self._cache_D_valid):
                for i in range(M):
                    if(i==pos):
                        continue
                    # add new cell D value
                    delta = self.cells[i].calcD(newval)
                    self._cache_D[pos] += delta
                    # remove prev cell D value
                    delta -= self.cells[i].calcD(oldval)
                    self._cache_D[i] += delta
            # invalidate G
            self._cache_G = None
        # to string
        def __str__(self):
            return "\n".join(
                [
                    f"| {str(c)} |" for c in self.cells
                ]
            )

    def __init__(self, context: Context, distances: SymmetricMatrix):
        self.context = context
        self.distances = distances
        # allocate timeslots
        self.timeslots = [Schedule.Timeslot(context, distances) for i in range(context.T)]
    # returns the G value of the schedule
    @property
    def G(self) -> float:
        return sum(map(lambda t:t.G, self.timeslots))
    # returns the D value of the schedule
    @property
    def D(self) -> float:
        return sum(map(lambda t:t.D, self.timeslots))
    # returns the expected G
    @staticmethod
    def expectedG(context: Context, avgDist: float) -> float:
        return context.T * Schedule.Timeslot.expectedG(context, avgDist)
    # returns the expected D
    @staticmethod
    def expectedD(context: Context, avgDist: float) -> float:
        return context.T * Schedule.Timeslot.expectedD(context, avgDist)
    # replace timeslot with contents of list
    def _fromList(self, lst: List[int]):
        # apply list to cells
        sz = self.context.K * self.context.M
        for i in range(len(self.timeslots)):
            self.timeslots[i]._fromList(lst[i*sz : (i+1)*sz])
    # This randomises the schedule
    def randomize(self, seed = random.seed()):
        serial_sch = list(range(1,self.context.N+1))
        # shuffle schedule
        random.Random(seed).shuffle(serial_sch)
        # set schedule contents to contents of serial_sch
        self._fromList(serial_sch)
    # __getitem__ function overloads the [] operator
    # [x,y,z] will fetch x timeslot -> y market -> shop at position z in the cell
    # [x] will fetch the entire x timeslot
    def __getitem__(self, pos):
        # magic
        if (type(pos) == tuple):
            return self.timeslots[pos[0]][pos[1:]]
        else:
            return self.timeslots[pos]
    # to string
    def __str__(self):
        return "\n".join(
            [
                " | ".join(
                    str(self.timeslots[t].cells[m]) for t in range(self.context.T)
                ) for m in range(self.context.M)
            ]
        )

# %% [markdown] id="mB6LJff2gzIZ"
# # Utility functions
# ## input_context()
# Inputs data from the user and return a `Context` object.
# ## input_distances(context)
# Takes a `Context` object, then inputs the distance matrix from the user and returns the `SymmetricMatrix` object of the distance matrix.

# %% executionInfo={"elapsed": 8147, "status": "ok", "timestamp": 1601111600272, "user": {"displayName": "Jaideep Singh Heer (M20CS056)", "photoUrl": "", "userId": "05136112523110687861"}, "user_tz": -330} id="37hNO4nEhLk1"
def input_context() -> Context:
    K, M, T = [int(x) for x in [input(),input(),input()]]
    C = float(input())
    return Context(K,M,T,C)

def input_distances(context: Context) -> SymmetricMatrix:
    mat = SymmetricMatrix(size=context.N)
    for i in range(context.N):
        for idx, dist in enumerate(input().split()):
            mat[i,idx] = float(dist)
    return mat

from typing import Tuple
import random

class GA:
    def __init__(self, context: Context, distance: SymmetricMatrix):
        self.context = context
        self.distance = distance
        self.bestG = -1
        # create schedule object
        self.population = Schedule(context, distance)
        self.population._fromList(list(range(1,context.N+1)))
        # create loop detection structure
        self._resetHistory()
    
    def _resetHistory(self):
        self.mutation_history = dict()
        self.cross_history = ([], 2,)
    
    # takes two cells and performs one point crossover b/w them round the given pivot
    @staticmethod
    def _pivotCrossCells(context: Context, cell_a: Schedule.Cell, cell_b: Schedule.Cell, pivot: int):
        # return if not enough shops of invalid pivot
        if(context.K < 2 or pivot < 0 or pivot >= context.K):
            return
        # cross swap first half and 2nd half
        shops_a = cell_b.shops[pivot :] + cell_a.shops[: pivot]
        shops_b = cell_a.shops[pivot :] + cell_b.shops[: pivot]
        cell_a.shops = shops_a
        cell_b.shops = shops_b
        # invalidate cell cache since we directly modidied the shops
        cell_a._invalidate_cache()
        cell_b._invalidate_cache()
    
    # takes two cells and performs half point crossover b/w them
    @staticmethod
    def _halfPointCrossCells(context: Context, cell_a: Schedule.Cell, cell_b: Schedule.Cell):
        GA._pivotCrossCells(context, cell_a, cell_b, context.K//2)
    # takes two cells and performs one point random crossover b/w them
    @staticmethod
    def _randomPointCrossCells(context: Context, cell_a: Schedule.Cell, cell_b: Schedule.Cell):
        if(context.K < 2):
            return
        GA._pivotCrossCells(context, cell_a, cell_b, random.randint(1, context.K-1))
    
    # Accepts a timeslot and performs mutation in an attempt to improve its G value.
    # this returns False mutation is not possible
    def Mutation(self, timeslot: Schedule.Timeslot) -> bool:
        # return if not enough cells
        if(self.context.M < 2):
            return False
        # sort timeslot according to S values of cells
        timeslot.cells.sort(key=lambda cell: cell.G)
        cells = timeslot.cells
        # mutate b/w worst two cells
        a, b = cells[:2]
        # check history for loop detection with each element
        prevPair, betterOffset = self.mutation_history.get(timeslot, ([], 2,))
        if(a in prevPair and b in prevPair):
            # worst elements repeated, cross one with a better cell to introduce better genes
            if(len(cells) <= betterOffset):
                # ran out of better choices to cross, mutation impossible
                return False
            betterCell = cells[betterOffset]
            # cross a with the better offset cell
            GA._halfPointCrossCells(context, a, betterCell)
            betterOffset += 1
        else:
            # reset better offset
            betterOffset = 2
        # update history
        self.mutation_history[timeslot] = ([a,b], betterOffset,)
        GA._randomPointCrossCells(context, a, b)
        return True
    
    # performs crossover b/w two timeslots by swapping one pair of best and worst cells
    @staticmethod
    def _bestWorstCrossTimeslots(context: Context, slot_1: Schedule.Timeslot, slot_2: Schedule.Timeslot):
        max_min_idx_func = lambda iter: (iter.index(max(iter)), iter.index(min(iter)))
        best_idx_a, worst_idx_a = max_min_idx_func(slot_1._cache_D)
        best_idx_b, worst_idx_b = max_min_idx_func(slot_2._cache_D)
        # mutate best and worst cells
        # GA._randomPointCrossCells(context, slot_1[best_idx_a], slot_2[worst_idx_b])
        # GA._randomPointCrossCells(context, slot_2[best_idx_b], slot_1[worst_idx_a])
        # return
        # swap best with worst
        slot_1[best_idx_a], slot_2[worst_idx_b] = slot_2[worst_idx_b], slot_1[best_idx_a]
        slot_2[best_idx_b], slot_1[worst_idx_a] = slot_1[worst_idx_a], slot_2[best_idx_b]

    # performs crossover b/w two timeslots and returns the selected timeslots
    # it also returns a bool whith is false if crossover fails
    def Crossover(self) -> Tuple[Tuple[Schedule.Timeslot, ...], bool]: 
        # return if not enough timeslots
        if (self.context.T < 2):
            return (self.population[0],), False
        # select the worst 2 timeslots according to G value
        timeslots = sorted(self.population, key=lambda timeslot: timeslot.G)
        a, b = timeslots[:2]
        # check history for loop detection with each element
        prevPair, betterOffset = self.cross_history
        if(a in prevPair and b in prevPair):
            # worst elements repeated, cross one with a better timeslot to introduce better genes
            if(len(timeslots) <= betterOffset):
                # ran out of better choices to cross, mutation impossible
                return (a,b,), False
            betterSlot = timeslots[betterOffset]
            # cross a with the better offset timeslot
            GA._bestWorstCrossTimeslots(context, a, betterSlot)
            betterOffset += 1
        else:
            # reset better offset
            betterOffset = 2
        # update history
        self.cross_history = ([a,b], betterOffset,)
        # perform crossover
        GA._bestWorstCrossTimeslots(self.context, a, b)
        # return selected timeslots
        return (timeslots[0], timeslots[-1],), True
    
    # stores the best population
    def snapshot(self):
        if(self.bestG < self.population.G):
            self.bestG = self.population.G
            self.bestS = str(self.population)
        
    def Evolution(self) -> Tuple[str, float, int]:
    # create schedule object
        iterations = 0
        from timeit import default_timer as timer
        # run for 2 sec
        start = timer()
        while (timer() - start < 2):
            # starvation detection
            healthy = False
            # perform mutatation and crossover
            slots_crossed, healthy = self.Crossover()
            # snapshot G
            self.snapshot()
            if(not healthy):
                # mutate crossed timeslots, since crossover failed
                for slot in slots_crossed:
                    slot_health = self.Mutation(slot)
                    if(slot_health):    
                        # snapshot G
                        self.snapshot()
                    # healthy if any mutation is successfull
                    healthy = healthy or slot_health
            # snapshot G
            self.snapshot()
            iterations += 1
            # break loop on starvation
            if (not healthy):
                # randomize population
                self.population.randomize()
                # reset histories
                self._resetHistory()
        return (self.bestS, self.bestG, iterations,)

# Take input
context = input_context()
distance = input_distances(context)

# create object of GA
ga = GA(context, distance)

bestS, bestG, iterations = ga.Evolution()

print(bestS)
