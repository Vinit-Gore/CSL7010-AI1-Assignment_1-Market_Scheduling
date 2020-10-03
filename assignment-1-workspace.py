# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3.7.3 64-bit
#     language: python
#     name: python_defaultSpec_1601660090735
# ---

# %% [markdown] id="6GTphwF2xK9Z"
# # ASSIGNMENT 1: AUTOMATED MARKET OPENING SCHEDULER DURING COVID
#
#
# > Goal: The goal of this assignment is to take a complex new problem and formulate and solve it as search. Formulation as search is an integral skill of AI that will come in handy whenever you are faced with a new problem. Heuristic search will allow you to find optimal solutions. Local search may not find the optimal solution, but is usually able to find good solutions for really large problems.
#
#
# ## Scenario: Optimization of Market Opening Schedule.
# A city has **n** types of shops. 
#
# The government wants to create an opening schedule for the markets ensuring the safety of maximum people. Due to the current COVID situation the government wants the people to make minimum movement out of their houses. They have approached you to take your help in order to organize the opening of shops in a best possible schedule. You need to use the power of AI and write a generalized search algorithm to find the best possible schedule. 
#
# The city has **m** market places which can be opened parallely. In a market place during each time slot the government is planning to open **k** types of shops. And in a day there are a total of **T** time slots available. We can assume that `n = T.m.k`. 
#
# For example, in figure below `m = 2`, `T = 3` and `k = 4`
#
# ||||
# |:-|:-|:-|
# |Type: 1,2,3,4|Type: 5,6,7,8|Type: 9,10,11,12|
# |Type: 13,14,15,16|Type: 17,18,19,20|Type: 21,22,23,24|
#
#
# We first define the characteristics of a good schedule. For any good schedule people should make minimum movement and most of the people should feel no conflict about which market they should go for purchasing.
#
# That is: 
# - All types of shops opening in one time slot in the same market should sell related items (items generally bought together).
# - All types of shops opening in parallel markets should be as far away as possible to avoid people’s movement to all of the markets (selling items that are generally not bought together).
#
# To operationalize this intuition let us assume we are given a function representing the distance between two types/categories: **`d(t1, t2)`**, such that **d** is between 0 and 1. We can similarly define a similarity between two, **`s(t1, t2) = 1 - d(t1, t2)`**. 
#
# Now we can define the goodness of a schedule as follows:
# - `Sum(similarities of all pairs within a single time slot in the same market) + C * Sum(distances of all pairs within a single time slot in the parallel market)`.
#
# In our example, the goodness will be computed as,
#
# > ```
# > G(Schedule) = s(1,2) + s(1,3) + s(1,4) + s(2,3) + s(2,4) + s(3,4) + 
# >               s(5,6) + s(5,7) + s(5,8) + s(6,7) + s(6,8) + s(7,8) +
# >         ……. + s(13,14) + …. + s(21,22) + ….. + 
# >    + C x [d(1,13) + d(1,14)… d(2,13) + d(2,14) + … + d(5,17) + d(5,18) + …]
# > ```
#
# The constant **C** trades off the importance of semantic coherence of one market versus reducing conflict across parallel markets.
#
# **Your goal is to find a schedule with the maximum goodness.**
#

# %% [markdown] id="ma3vR73IzYW1"
# Input:
#
# Line 1: **k**: total types of shops opening in one time slot in one market
#
# Line 2: **m**: number of parallel markets
#
# Line 3: **T**: number of time slots
#
# Line 4: **C**: trade-off constant
#
# Starting on the fifth line we have a space separated list of distances between a type of shop and rest others. Note that `d(x,y) = d(y,x)`. Also, all `d(x,x) = 0`.
#

# %% [markdown] id="V2QotcGQ3orC"
# Important Instructions:
# - You may work in teams of maximum three or by yourself. If you are short of partners, our recommendation is that this assignment is quite straightforward and a partner should not be required.
# - You cannot use built-in libraries/implementations for search or scheduling algorithms.
# - Please do not search the Web for solutions to the problem. Your submission will be checked for plagiarism with the codes available on Web as well as the codes submitted by other teams. Any team found guilty will be awarded a suitable penalty as per IIT rules.
# - Your code will be automatically evaluated. You get a zero if your output is not automatically parsable.
# - You are allowed to use any of the two programming languages: C++, Python.
#

# %% [markdown] id="WrItzjyw9Uhr"
# ### Sample Input
# ```
# 2
# 2
# 1
# 1
# 0 0.4 0.8 1
# 0.4 0 0.6 0.7
# 0.8 0.6 0 0.3
# 1 0.7 0.3 0
# ```
# ### Output
# Your algorithm should return the max-goodness schedule.
#
# #### Output Format:
# Space separated list of shop ids (i.e, shop’s type ids), where time slots are separated by bars and parallel markets are separated by line.
# For the above problem the optimal solution is t1 and t2 in one market; and t3 and t4 in the other market. It will be represented as: 
# ```
# 1 2
# 3 4
# ```
# Other equivalent ways to represent this same solution:
# ```
# 4 3
# 2 1
# ```
# OR
# ```
# 2 1
# 3 4
# ```
# etc. All are valid and have the total goodness of 4.4 (Verify).
#
# Another sample input is provided along with the assignment representing similar easy problems.
#
# We recommend you to experiment with other problems as well.
#

# %% [markdown] id="Urbyt-N2VxBU"
# # Utility functions for colab notebook

# %% [markdown] id="nkr9sE4ZPPq6"
# ## Ray for parallelism
# Install and setup the Ray library.
# https://github.com/ray-project/tutorial

# %% executionInfo={"elapsed": 7806, "status": "ok", "timestamp": 1601111599799, "user": {"displayName": "Jaideep Singh Heer (M20CS056)", "photoUrl": "", "userId": "05136112523110687861"}, "user_tz": -330} id="2EjS5UJ7PEff" outputId="d27b62eb-b011-484d-9794-8cb40ba4b02d" tags=[]
# Install Ray for parallelism
print('NOTE: Intentionally crashing session to use the newly installed library.\n')

# !pip uninstall -y pyarrow
# !pip install ray[debug]==0.7.5
# !pip install bs4

try:
    import ray
except:
    # A hack to force the runtime to restart, needed to include the above dependencies.
    import os
    os._exit(0)


# %% [markdown] id="CfTtz2WuVoJG"
# ## I/O Helper Functions
#
# ### set_input(inp_str)
# This takes a string which will be used to read lines for the input function.
#
# ### input()
# This is a proxy to use the given string instead of asking the user for input.

# %% executionInfo={"elapsed": 7751, "status": "ok", "timestamp": 1601111599802, "user": {"displayName": "Jaideep Singh Heer (M20CS056)", "photoUrl": "", "userId": "05136112523110687861"}, "user_tz": -330} id="CHFZ6tJ5S0WP"
# =======================================================
#       Fake input function for colab notebook
# =======================================================
__input = None
__fake_input_ctr = -1
def set_input(inp_str):
    global __input
    __input = [x.strip() for x in inp_str.split('\n')]
    __input = [x for x in __input if x!=""]
    global __fake_input_ctr
    __fake_input_ctr = -1
def input() -> str:
    global __fake_input_ctr
    __fake_input_ctr += 1
    return __input[__fake_input_ctr]
# =======================================================


# %% [markdown] id="WT_hoHZyWUi-"
# ## Test cases
# These are test cases used to test the program.
#
# A test case is just an input string.
#
# `TestCases` is a list of test cases.

# %% executionInfo={"elapsed": 7735, "status": "ok", "timestamp": 1601111599805, "user": {"displayName": "Jaideep Singh Heer (M20CS056)", "photoUrl": "", "userId": "05136112523110687861"}, "user_tz": -330} id="7iWIuLVgWaxO"
TestCases = [
                  """
                  2
                  2
                  1
                  1
                  0 0.4 0.8 1
                  0.4 0 0.6 0.7
                  0.8 0.6 0 0.3
                  1 0.7 0.3 0
                  """,
                  """
                  2
                  2
                  3
                  1
                  0 0.0 0.8 0.9 0.2 0.1 1.0 0.8 0.2 0.3 0.8 0.8
                  0.0 0 0.8 1.0 0.2 0.1 0.8 0.9 0.2 0.2 0.8 1.0
                  0.8 0.8 0 0.1 0.8 0.9 0.2 0.2 0.7 0.9 0.0 0.2
                  0.9 1.0 0.1 0 0.7 0.9 0.0 0.2 0.9 0.9 0.1 0.1
                  0.2 0.2 0.8 0.7 0 0.1 0.8 1.0 0.3 0.2 0.9 0.7
                  0.1 0.1 0.9 0.9 0.1 0 0.8 0.9 0.1 0.2 0.8 0.9
                  1.0 0.8 0.2 0.0 0.8 0.8 0 0.0 0.9 0.8 0.1 0.0
                  0.8 0.9 0.2 0.2 1.0 0.9 0.0 0 0.8 0.9 0.2 0.0
                  0.2 0.2 0.7 0.9 0.3 0.1 0.9 0.8 0 0.2 0.8 0.8
                  0.3 0.2 0.9 0.9 0.2 0.2 0.8 0.9 0.2 0 0.8 0.9
                  0.8 0.8 0.0 0.1 0.9 0.8 0.1 0.2 0.8 0.8 0 0.3
                  0.8 1.0 0.2 0.1 0.7 0.9 0.0 0.0 0.8 0.9 0.3 0
                  """,
]


# %% executionInfo={"elapsed": 7724, "status": "ok", "timestamp": 1601111599806, "user": {"displayName": "Jaideep Singh Heer (M20CS056)", "photoUrl": "", "userId": "05136112523110687861"}, "user_tz": -330} id="op_eHI3Axqas"
# Generate test cases
def generate_test_case(K,M,T,C) -> str:
    N = K*M*T
    r = f"{K}\n{M}\n{T}\n{C}\n"
    import numpy as np
    dist = np.random.rand(N,N)
    dist = np.around(np.maximum(dist, dist.transpose()), decimals = 1)
    for i in dist:
        r += " ".join(map(str,i)) + "\n"
    return r

# generate test case for N=100
TestCases.append(generate_test_case(10,5,2,2.4))

# %% [markdown] id="lynaubXR1e_O"
# # Typedefs and Data Structures

# %% [markdown] id="jwpRphtbTBqf"
# ## Context
# This stores the variables `K`, `M`, `T`, `C` and `N` as derived from the input.
#
# This DS is immutable, i.e. once created, data inside it cannot be changed.

# %% executionInfo={"elapsed": 7715, "status": "ok", "timestamp": 1601111599807, "user": {"displayName": "Jaideep Singh Heer (M20CS056)", "photoUrl": "", "userId": "05136112523110687861"}, "user_tz": -330} id="oAm4yZ7Q7cQI"
# Typings
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
    def sum(self):
        if(self._cache_sum == None):
            # calc. sum
            self._cache_sum = v = 0
            for i in self.matrix:
                v += sum(i)
                self._cache_sum -= i[-1]
            self._cache_sum = v*2
        return self._cache_sum
    # returns average of all values
    def average(self):
        return self.sum() / self.size**2


# %% [markdown] id="QQpF5DOj3Pc8"
# ## Schedule
# A class that represents a schedule as a table with `T` rows and `m` columns with each cell containing `k` elements.

# %%
from typing import List, Callable
import random
class Schedule:
    # Represents a single cell
    class Cell:
        # a func. that takes the cell instance and two int, shop removed and shop added
        CelllUpdateListener = Callable[[Schedule.Cell, int, int], None]
        def __init__(self, context: Context, distances: SymmetricMatrix):
            self.context = context
            self.distances = distances
            # allocate shops
            self.shops = [None] * context.K
            # allocate cache
            self._invalidate_cache()
            # alocate listeners
            self._listeners = dict()
        # invalidate all cache
        def _invalidate_cache(self):
            self._cache_G = None
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
        def calcD(self, other: Schedule.Cell) -> float:
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
        def onCellChange(self, cell: Schedule.Cell, removedShop: int, addedShop: int):
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
            return Schedule.Timeslot.expectedCellD(context, avgDist)*context.M/2
        # returns the expectedD of a cell in a timeslot with m markets and k shops per cell
        # the total D value of a timeslot is half of (this value * M)
        @staticmethod
        def expectedCellD(context: Context, avgDist: float) -> float:
            return avgDist*(context.K**2)/2
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
        def __getitem__(self, pos):
            # magic
            if (type(pos) == tuple):
                if(len(pos) == 1):
                    return self.cells[pos[0]]
                return self.cells[pos[0]][pos[1:]]
            else:
                return self.cells[pos]
        def __setitem__(self, pos, newval: Schedule.Cell):
            M = self.context.M
            oldval = self.cells[pos]
            self.cells[pos] = newval
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
    def G(self) -> float:
        return sum(map(lambda t:t.G), self.timeslots)
    # returns the D value of the schedule
    def D(self) -> float:
        return sum(map(lambda t:t.D), self.timeslots)
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

# %% executionInfo={"elapsed": 8159, "status": "ok", "timestamp": 1601111600270, "user": {"displayName": "Jaideep Singh Heer (M20CS056)", "photoUrl": "", "userId": "05136112523110687861"}, "user_tz": -330} id="03S_eu4A4GLh"
# Imports
from typing import Tuple
import random
# Schedule indexed as [time, market, k]
# Schedule[time] gives the entire time slot which is a list of T cells with each cell being a list having k shops.
class Schedule_old:
    # This is used to locate different elements of the schedule
    from dataclasses import dataclass
    @dataclass
    class Position:
        timeslot_idx: int = None
        market_idx: int = None
        shop_idx: int = None
        # validates the position object w.r.t the given context
        def validate(self, context: Context, 
                                 validate_timeslot=True, 
                                 validate_market=True, 
                                 validate_shop=True) -> bool:
            if(validate_timeslot):
                if(self.timeslot_idx < 0 
                     or self.timeslot_idx >= context.T):
                    return False
            if(validate_market):
                if(self.market_idx < 0 
                     or self.market_idx >= context.M):
                    return False
            if(validate_shop):
                if(self.shop_idx < 0 
                     or self.shop_idx >= context.K):
                    return False
            return True
            
    
    # This generates a schedule full of None values.
    def __init__(self, context: Context, distance_matrix: SymmetricMatrix):
        # save context and distance matrix
        self.context = context
        self.distance = distance_matrix
        # allocate space for schedule
        self.matrix = [None]*context.T            # T rows
        for m in range(context.T):
            self.matrix[m] = [None]*context.M       # M columns
            for t in range(context.M):
                self.matrix[m][t] = [None]*context.K  # k shops per cell
                
    def custom_sch(self, seq: list):
        #input your custom schedule. Make sure all elements are integers and no element is repeated.
        #Entering cell-wise i.e. 1 2 | 5 6
        #                        3 4 | 7 8
        i = 0
        for t in range(self.context.T):
            for m in range(self.context.M):
                for k in range(self.context.K):
                    self.matrix[t][m][k] = seq[i]
                    i = i+1
        
    
    # This randomises the schedule
    def randomize(self, seed = random.seed()):
        serial_sch = list(range(1,self.context.N+1))
        # shuffle schedule
        # random.Random(seed).shuffle(serial_sch)
        # set schedule contents to contents of serial_sch
        self.custom_sch(serial_sch)
        
    # define str funtion to print schedule as final output
    def __str__(self):
        return "\n".join(
                [
                 " | ".join(
                         " ".join(
                                 map(str, self.matrix[t][m])
                         )
                         for t in range(self.context.T)
                 ) 
                 for m in range(self.context.M)
                ]
        )
    
    # __getitem__ function overloads the [] operator
    # [x,y,z] will fetch x timeslot -> y market -> shop at position z in the cell
    # [x] will fetch the entire x timeslot
    def __getitem__(self, pos):
        # magic
        if (type(pos) == tuple):
            l = len(pos)
            if l == 1:
                self.matrix[pos[0]]
            elif l == 2:
                return self.matrix[pos[0]][pos[1]]
            elif l == 3:
                return self.matrix[pos[0]][pos[1]][pos[2]]
        else:
            return self.matrix[pos]

    # ============= Problem Specific =============
    # calculates the G value of a given schedule element or the entire schedule
    def calc_G(self, position: Position = Position()) -> float:
        if (position.timeslot_idx != None 
             and position.market_idx != None):
            # calc cell G value
            return self.calc_cell_G(position)
        elif (position.timeslot_idx != None):
            # calc timeslot G value
            return self.calc_timeslot_G(position)
        else:
            # calc schedule G value
            G = 0
            pos = self.Position(0,None,None)
            while(pos.timeslot_idx < self.context.T):
                G += self.calc_timeslot_G(pos)
                pos.timeslot_idx += 1
            return G
    
    # calculates the D value between two cells
    # this does not multiply the values with C
    def calc_intercell_D(self, pos_1: Position, pos_2: Position) -> float:
        # validate cell positions
        if (not pos_1.validate(self.context, validate_shop=False)
                or not pos_2.validate(self.context, validate_shop=False)):
            raise Exception("Invalid position:", pos_1, pos_2)
        if (pos_1.timeslot_idx != pos_2.timeslot_idx):
            raise Exception("Cannot calculate G value between cells of different timeslots.", pos_1, pos_2)
        if(pos_1 == pos_2):
            return 0
        # calculate inter-cell G value
        D = 0
        shops_1 = self.matrix[pos_1.timeslot_idx][pos_1.market_idx]
        shops_2 = self.matrix[pos_2.timeslot_idx][pos_2.market_idx]
        k = self.context.K
        for i in range(k):
            for j in range(k):
                D += self.distance[shops_1[i]-1, shops_2[j]-1]
        return D
    
    # calculates the G value of a timeslot
    def calc_timeslot_G(self, position: Position) -> float:
        # check if position valid
        if(not position.validate(self.context, validate_market=False, validate_shop=False)):
            raise Exception("Invalid position:", position)
        # calculate S and D
        S, D = 0, 0
        # position iterators
        pos_1, pos_2 = self.Position(position.timeslot_idx,0,None), self.Position(position.timeslot_idx,0,None)
        # pos_1 runs through each cell along timeslot
        while pos_1.market_idx < self.context.M:
            # add current cell G
            S += self.calc_cell_G(pos_1)
            # add intra cell G w.r.t current cell
            pos_2.market_idx = pos_1.market_idx + 1
            while pos_2.market_idx < self.context.M:
                D += self.calc_intercell_D(pos_1, pos_2)
                pos_2.market_idx += 1
            pos_1.market_idx += 1
        # G = S + C*D
        return S + self.context.C * D

    # calculates the G value of a single cell
    def calc_cell_G(self, position: Position) -> float:
        # check if position valid
        if(not position.validate(self.context, validate_shop=False)):
            raise Exception("Invalid position:", position)
        shops = self.matrix[position.timeslot_idx][position.market_idx]
        S = 0
        for i in range(len(shops)):
            for j in range(i+1, len(shops)):
                S += 1 - self.distance[shops[i]-1, shops[j]-1]
        return S
    # ============================================


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


# %% [markdown] id="ofAx-KwY7S-7"
# # Testing

# %% [markdown] id="PLIwWsMc7ne4"
# ## TestCases

# %% executionInfo={"elapsed": 8132, "status": "ok", "timestamp": 1601111600273, "user": {"displayName": "Jaideep Singh Heer (M20CS056)", "photoUrl": "", "userId": "05136112523110687861"}, "user_tz": -330} id="H0nvL5k87lX4" outputId="8c8bff93-9d24-4bc9-aa99-f0fda5477174"
for t in range(len(TestCases)):
    print(f"\nTst Case {t}:-\n", TestCases[t])

# %% [markdown] id="p077XZYw7i7o"
# ## Context

# %% executionInfo={"elapsed": 8116, "status": "ok", "timestamp": 1601111600274, "user": {"displayName": "Jaideep Singh Heer (M20CS056)", "photoUrl": "", "userId": "05136112523110687861"}, "user_tz": -330} id="rEBmWnCE7U8U" outputId="082e2106-34b0-4833-8cda-6eb78170d2b8"
# ============== Testing ==============
print(Context(K=2,C=2,T=2,M=2))
# =====================================

# %% [markdown] id="qaCQegCr7gmr"
# ## SymmetricMatrix

# %% executionInfo={"elapsed": 8096, "status": "ok", "timestamp": 1601111600275, "user": {"displayName": "Jaideep Singh Heer (M20CS056)", "photoUrl": "", "userId": "05136112523110687861"}, "user_tz": -330} id="UZ1OAjwb7fh-" outputId="2f796666-8582-4ac0-dc58-a8995ec1d67f" tags=[]
# ============== Testing ==============
m = SymmetricMatrix(4)
m[1,2] = 0.12
m[0,3] = 1.2
print(m)
# =====================================

# %% [markdown] id="vx-nMs4i8Rco"
# ## Schedule

# %% executionInfo={"elapsed": 8068, "status": "ok", "timestamp": 1601111600276, "user": {"displayName": "Jaideep Singh Heer (M20CS056)", "photoUrl": "", "userId": "05136112523110687861"}, "user_tz": -330} id="-waDipGR8QPd" outputId="2fae03b4-1f56-4e03-c6e6-9f9101edc5cf" tags=[]
# ============== Testing ==============
set_input(TestCases[1])

context = input_context()
distance = input_distances(context)

s = Schedule(context, distance)
# print entire schedule
print("\nSchedule:-")
s.randomize()

print(f"Schedule:-")
print(s)
print(f"Schedule G-Value: {s.G}\n")

print(f"s[0][0] D value: {s[0].getCellD(0)}")
print(f"s[0][1] D value: {s[0].getCellD(1)}")
s[0][0][0] = 1
print()
print(s)
print(f"s[0][0] D value: {s[0].getCellD(0)}")
print(f"s[0][1] D value: {s[0].getCellD(1)}")
print(f"s[0].G: {s[0].G}")
print("correct")
s[0]._buildDCache()
s[0]._cache_G = None
print(f"s[0][0] D value: {s[0].getCellD(0)}")
print(f"s[0][1] D value: {s[0].getCellD(1)}")
print(f"s[0].G: {s[0].G}")

print("\n swap testing \n")
print(s)
print()
t = s[0,1]
s[0][1] = s[1][1]
s[1][1] = t
print(s)
# =====================================

# %% [markdown] id="FXuMr-Ik8mJZ"
# # Benchmarks

# %% executionInfo={"elapsed": 8044, "status": "ok", "timestamp": 1601111600277, "user": {"displayName": "Jaideep Singh Heer (M20CS056)", "photoUrl": "", "userId": "05136112523110687861"}, "user_tz": -330} id="ekIFN0Uq8oFK" outputId="557a4d3b-ed08-4c21-a68e-47f64d199c31"
# benchmark setup
set_input(TestCases[2])
context = input_context()
distance = input_distances(context)
s = Schedule(context, distance)
s.randomize()

print(s)

# %% executionInfo={"elapsed": 12703, "status": "ok", "timestamp": 1601111604958, "user": {"displayName": "Jaideep Singh Heer (M20CS056)", "photoUrl": "", "userId": "05136112523110687861"}, "user_tz": -330} id="G2rBxBh78qcn" outputId="cb34d390-1dbb-441c-f23f-f2d2793d7a64"
# %%timeit
s.calc_G()

# %% [markdown] id="kTYYQxe59vxd"
# # Algo testing utilities
# This code is used to test an algo's performance on different test cases in order to compare multiple different algos.
#

# %% [markdown] id="pMGRcB9GBff9"
# ## Test DS definitions
# These are used to manage the test perfomance data of an algo.

# %% executionInfo={"elapsed": 12690, "status": "ok", "timestamp": 1601111604959, "user": {"displayName": "Jaideep Singh Heer (M20CS056)", "photoUrl": "", "userId": "05136112523110687861"}, "user_tz": -330} id="Qyndj28v98RT"
from dataclasses import dataclass
from typing import List, Any

@dataclass
class TestResult:
    G_values: List[float] # G values for each exexution
    Times: List[float]      # Execution time for each execution
    Iterations: List[int] # No. of iteration returned by each execution
    def to_dict(self):
        return {
            'G_value': self.G_values,
            'Time': self.Times,
            'Iterations': self.Iterations,
        }

@dataclass
class TestCase:
    context: Context
    distance: SymmetricMatrix

@dataclass
class TestParams:
    test_case_repetition: int
    generated_test_case_count: int
    test_cases: List[TestCase]


# %% [markdown] id="YIfkoRtCBqYe"
# ## Algo definition
# This defines what an algo is supposed to be. An algo that must be tested has to be a function that takes a `Context` and an `SymmetricMatrix` as input, and returns a tuple of the output string and the best G value as output.

# %% executionInfo={"elapsed": 12683, "status": "ok", "timestamp": 1601111604961, "user": {"displayName": "Jaideep Singh Heer (M20CS056)", "photoUrl": "", "userId": "05136112523110687861"}, "user_tz": -330} id="ezd5phnbAlVS"
from typing import Callable, Tuple

# Define an algo
# Inputs: Context, 
#         Distance matrix
# Outputs: str of best schedule,
#          G value of best schedule, 
#          no. of iterations
Algo = Callable[[Context, SymmetricMatrix], Tuple[str, float, int]]


# %% [markdown] id="3-pX_ls6QDPh"
# ## Test case generation

# %% executionInfo={"elapsed": 12676, "status": "ok", "timestamp": 1601111604963, "user": {"displayName": "Jaideep Singh Heer (M20CS056)", "photoUrl": "", "userId": "05136112523110687861"}, "user_tz": -330} id="UGG_hbFJP2mw"
# Generate context
def genContext(N: int = None) -> Context:
    # returns a list of List[int,int,int] that are factors of N
    def factorize(N):
        r = []
        for i in range(N):
            for j in range(N):
                if i*j > N:
                    break
                for k in range(N):
                    if i*j*k > N:
                        break
                    elif (i*j*k == N):
                        r.append([i,j,k])
        return r
    import random as rn
    if (N == None):
        N = rn.randint(1,100)
    C = round(rn.random(), 1)
    f = factorize(N)
    while len(f) == 0:
        N = rn.randint(1,100)
        f = factorize(N)
    l = rn.choice(factorize(N))
    rn.shuffle(l)
    K, M, T = l
    return Context(K,M,T,C)

# Generate distance matrix
def genDistMatrix(context: Context) -> SymmetricMatrix:
    import random as rn
    ret = SymmetricMatrix(context.N)
    for i in range(context.N):
        ret[i, i] = 0
        for j in range(i+1, context.N):
            ret[i, j] = round(rn.random(), 1)
    return ret

# Generate test case
def genTestCase(context: Context = None) -> TestCase:
    if(context == None):
        context = genContext()
    return TestCase(context, genDistMatrix(context))



# %% [markdown] id="SYqgrw2bCV5V"
# ## Testing functions
# These functions test the algo and record its performance.

# %% executionInfo={"elapsed": 15307, "status": "ok", "timestamp": 1601111607613, "user": {"displayName": "Jaideep Singh Heer (M20CS056)", "photoUrl": "", "userId": "05136112523110687861"}, "user_tz": -330} id="_oypfCHbAk00" outputId="e1807261-9027-4afb-d633-0f326abfc4c4"
from typing import Tuple, List
from timeit import default_timer as timer
import multiprocessing as mp
import ray

# init ray
ray.init(ignore_reinit_error=True)

@ray.remote
def __bench_algo(algo, ctx, dst):
    # Benchmark algo
    start = timer()
    retVal = algo(ctx, dst)
    end = timer()
    # Benchmark end
    return (end-start, retVal,)

# Takes an algo by runnig it multiple times over a given test case
@ray.remote
def test_algo_for_set(algo: Algo, tparam: TestParams, tcase: TestCase) -> Tuple[TestCase, TestResult]:
    # ray
    res = []
    for i in range(tparam.test_case_repetition):
        res.append(
            __bench_algo.remote(algo, tcase.context, tcase.distance)
        )
    res = ray.get(res)
    # extract results
    times, retVals = zip(*res)
    S, G, Iter = zip(*retVals)
    return (tcase, TestResult(G, times, Iter),)

def testAlgo(algo: Algo, test_params: TestParams) -> List[Tuple[TestCase, TestResult]]:
    # prepare test cases
    par = []
    # add generated test cases
    for i in range(test_params.generated_test_case_count):
        par.append(
            test_algo_for_set.remote(
                algo, test_params, genTestCase()
            )
        )
    # add given test cases
    for e in test_params.test_cases:
        par.append(
            test_algo_for_set.remote(
                algo, test_params, e
            )
        )
    # ray multi process test cases
    return ray.get(par)



# %% [markdown] id="e9UwwXt9bSu1"
# ## Plotting functions

# %% executionInfo={"elapsed": 1758, "status": "ok", "timestamp": 1601114495058, "user": {"displayName": "Jaideep Singh Heer (M20CS056)", "photoUrl": "", "userId": "05136112523110687861"}, "user_tz": -330} id="oj_5VlRzbWot"
import pandas as pd

def expectedG(test_case: TestCase) -> float:
    ctx, dst = test_case.context, test_case.distance
    # calc sum of dist
    s = 0
    for r in dst.matrix:
        s += sum(r)
    s *= 2
    # calc average distance
    avgDist = s / (ctx.N ** 2)
    # create dummy schedule to calc. expected G
    dummyDst = SymmetricMatrix(ctx.N)
    for i in range(ctx.N):
        for j in range(i+1, ctx.N):
            dummyDst[i,j] = avgDist
    # return expected G
    s = Schedule(ctx, dummyDst)
    s.randomize()
    return s.calc_G()

# converts TestResult to a dataframe with normalised G values
def testresult_to_df(test_res: Tuple[TestCase, TestResult]) -> pd.DataFrame:
    tcase, tres = test_res
    # make df
    df = pd.DataFrame(tres.to_dict())
    # normalise G to expected G, i.e. estimated average G
    df['G_value'] *= 1/expectedG(tcase)
    df = df.rename(columns={'G_value': 'normalised_G'})
    # add test case values
    df['K'] = tcase.context.K
    df['M'] = tcase.context.M
    df['T'] = tcase.context.T
    df['N'] = tcase.context.N
    df['C'] = tcase.context.C
    return df

def test_out_to_df(test_out: List[Tuple[TestCase, TestResult]]) -> pd.DataFrame:
    df = pd.DataFrame()
    for e in test_out:
        df = df.append(testresult_to_df(e), sort=False, ignore_index=True)
    return df

# creates a boxplot of M*K vs normalised G
def mk_boxplot(data: pd.DataFrame):
    # group data by test case
    data.groupby(['K','M','T','C'])


# %% [markdown]
# ## Standard test paramaters
# Used for comparing different algos with the same test cases.

# %%
std_test_params = [
    # Test with 10 common generated test cases
    TestParams(
        test_case_repetition = 20, 
        generated_test_case_count = 0,
        test_cases = [
            genTestCase() for i in range(10)
        ]
    )
]


# %% [markdown] id="2hKq6IfZgJ7R"
# # Algorithms
# The main algos we create.

# %% [markdown]
# # Genetic Algorithm

# %%
from typing import Tuple
import random

class GA:
    def __init__(self, context: Context, distance: SymmetricMatrix):
        # create schedule object
        self.population = Schedule (context, distance)
        self.population.randomize()
    
    def Mutation(self, timeslot):
        #select market randomly
        mother_index = random.randint(0, self.population.context.M - 1)
        father_index = random.randint(0, self.population.context.M - 1)
        
        #in case of M<=2 mutation will result in a continuous loop. Will be ttaken care of by crossover
        if mother_index is father_index:
            return
        
        #random cell for mutation    
        motherCell = timeslot [mother_index]
        fatherCell = timeslot [father_index]    
        
        mother_min, father_min = 1, 1
        mother_k, father_k = 0,0  #just in case S is 1 for all shop pairs
        
        for i in range(len(motherCell)):
            for j in range(i+1, len(motherCell)):
                S_mother = 1 - self.population.distance[motherCell[i]-1, motherCell[j]-1]
                S_father = 1 - self.population.distance[fatherCell[i]-1, fatherCell[j]-1]
                
                if S_mother < mother_min:
                    mother_k = random.choice([i,j]) #select either element of lowest G pair
                    #mother_gene = timeslot[mother_index][mother_k]  
                    mother_min = S_mother
                    
                if S_father < father_min:
                    father_k = random.choice([i,j])
                    #father_gene = timeslot[father_index][father_k]
                    father_min = S_father
                    
        #swap shops
        temp = timeslot[mother_index][mother_k]
        timeslot[mother_index][mother_k] = timeslot[father_index][father_k]
        timeslot[father_index][father_k] = temp
        
        #print (mother_index, mother_k, mother_gene, end='\n')
        #print (father_index, father_k, father_gene, end='\n')
        
    def Evolution(self) -> Tuple[str, float, int]:
    # create schedule object
    
        #Pick a time schedule to mutate
        Time_n = random.randint(0,self.population.context.T-1)
        G_parent = self.population.calc_G(self.population.Position(timeslot_idx=Time_n))
    
        print("Parent: ",self.population[Time_n], G_parent)
        print('Parent: \n', self.population)
    
        #Offspring after mutation
        self.Mutation(self.population[Time_n])
        G_child = self.population.calc_G(s.Position(timeslot_idx=Time_n))
    
        print("Mutant: ",self.population[Time_n], G_child)
        print('Mutant: \n', self.population)
        
        return (str(self.population), G_child, 1)


# %% [markdown] id="B_GeFozK-9Z9"
# ## Random Search

# %% executionInfo={"elapsed": 15272, "status": "ok", "timestamp": 1601111607620, "user": {"displayName": "Jaideep Singh Heer (M20CS056)", "photoUrl": "", "userId": "05136112523110687861"}, "user_tz": -330} id="JhQJemIhRbC6"
from typing import Tuple
def randomSearch(context: Context, distance: SymmetricMatrix) -> Tuple[str, float, int]:
    # create schedule object
    s = Schedule(context, distance)
    # setup vars
    bestG = -1
    bestS = None
    iterations = 0
    from timeit import default_timer as timer
    # run for 2 sec
    start = timer()
    while (timer() - start < 2):
        s.randomize()
        G = s.calc_G()
        if(G > bestG):
            bestS = str(s)
            bestG = G
        iterations +=1
    return (bestS, bestG, iterations,)




# %% [markdown]
# ### Test on pre-defined test cases

# %% cellView="both" executionInfo={"elapsed": 238783, "status": "ok", "timestamp": 1601111831207, "user": {"displayName": "Jaideep Singh Heer (M20CS056)", "photoUrl": "", "userId": "05136112523110687861"}, "user_tz": -330} id="Zu6JOxnItEDE" outputId="f8fcdab9-8911-49bf-913d-ef1a4317e6d8"
# ============================
# Set input for colab notebook
# ============================
set_input(TestCases[1])
# ============================

# Take input
context = input_context()
distance = input_distances(context)

# run random search
bestS, bestG, iterations = randomSearch(context, distance)

print(bestS)
print("\nG:", bestG)
print("Iterations:", iterations)

# %%
# ============================
# Set input for colab notebook
# ============================
set_input(TestCases[1])
# ============================

# Take input
context = input_context()
distance = input_distances(context)

# create object of GA
ga = GA(context, distance)

bestS, bestG, iterations = ga.Evolution()

print("Schedule:\n",bestS)
print("\nG:", bestG)
print("Iterations:", iterations)

# %% [markdown]
# ### Test on generated test cases

# %% executionInfo={"elapsed": 236531, "status": "ok", "timestamp": 1601111828894, "user": {"displayName": "Jaideep Singh Heer (M20CS056)", "photoUrl": "", "userId": "05136112523110687861"}, "user_tz": -330} id="UsjtMPFzTAbw" outputId="b3245be1-0f90-47df-952c-eea21011c460"
# test algo
test_result = testAlgo(randomSearch, std_test_params[0])

# %% executionInfo={"elapsed": 902, "status": "ok", "timestamp": 1601114740560, "user": {"displayName": "Jaideep Singh Heer (M20CS056)", "photoUrl": "", "userId": "05136112523110687861"}, "user_tz": -330} id="eVtLa9nreCWU" outputId="a3699c89-9c6f-46ba-949f-4a42076c0ae3"
# show mean performance for different test cases
analyse = test_out_to_df(test_result).groupby(['K','M','T','C']).mean()
analyse
