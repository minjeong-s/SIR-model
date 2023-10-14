# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# - quantitative modeling of infectious disease dynamics
# - Dynamics are modeled using a standard SIR (Susceptible-Infected-Removed) model of disease spread.
# - to study the impact of suppression through social distancing on the spread of the infection: social distancing이 감염 전파에 미치는 영향
#

# +
import numpy as np
from numpy import exp
import matplotlib.pyplot as plt

from scipy.integrate import odeint # for solving differential equations
# -

# # The SIR Model
# - All individuals in the population are assumed to be in one of these four states.
# - The states are: susceptible (S), exposed (E), infected (I) and removed (R)
# - S → E → I → R.
# - E는 바이러스에 노출되었지만 아직 감염되지 않은 그룹
# - R: 감염이 되었다가 회복 or 사망
# - 회복된 사람은 면역을 얻는다고 가정
#
# - All individuals in the population are eventually infected, when the transmission rate is positive and i(0)>0   
#
# - 우리가 시뮬레이션을 통해 알고자 하는 것
# 1. the number of infections at a given time (which determines whether or not the health care system is overwhelmed)
# 2. how long the caseload can be deferred (hopefully until a vaccine arrives)
#
#
# - 미분방정식: 소문자는 각 상태에 해당하는 사람들의 비율(fraction)     
# (1)     
# s˙(t)= −β(t)s(t)i(t)    
# e˙(t)= β(t)s(t)i(t)−σe(t)       
# i˙(t)= σe(t)−γi(t) 
#
# β(t): transmission rate - the rate at which individuals bump into others and expose them to the virus (바이러스에 노출되어지는 비율, 즉 S에서 E가 되는 비율)   
# σ: infection rate - the rate at which those who are exposed become infected (E에서 I가 되는 비율, E노출되어지는 사람 중에서 I감염된 사람)   
# γ: recovery rate - the rate at which infected people recover or die (감염된 사람이 회복되거나 죽는 비율, I에서 R이 되는 비율)   
# the dot symbol represents the time derivative dy/dt
#
# - The system (1) can be written in vector form as    
# x˙=F(x,t), x:=(s,e,i) (2)    
# for suitable definition of F (see the code below).
#
#
# - Both σ and γ are thought of as fixed, biologically determined parameters.
# - Atkeson's note에서는 다음과 같이 파라미터를 정함
# - σ=1/5.2  to reflect an average incubation period of 5.2 days.
# - γ=1/18 to match an average illness duration of 18 days.
# - β(t):= R(t)*γ where R(t) is the effective reproduction number at time t.
#
# - caseload 담당건수, 업무량 (일정기간에 돌봐야 하는 사람들의 수)
# c = i + r   
# c is the cumulative caseload 누적 업무량: all those who have or have had the infection
#
#

# # Implementation

# +
# First set the population size to match the US
pop_size = 3.3e8

# set the parameter as described above
γ= 1/18
σ=1/5.2


# +
# construct a function that represents F in (2)
# 각 비율의 time derivative를 함수로 나타내기 

def F(x, t, R0=1.6):
    '''
    Time derivative of the state vector.
    x˙=F(x,t), x:=(s,e,i) 
        * x is the state vector (array_like) x:=(s,e,i) 
        * t is time (scalar)
        * R0 is the effective transmission rate, defaulting to a constant
        (기본값은 상수)
        * R0(basic reproduction number) 값은 
        감염자가 없는 인구집단에 처음으로 감염자가 발생하였을 때 
        첫 감염자가 평균적으로 감염시킬 수 있는 2차 감염자의 수를 나타낸 것이다. 
        예를 들어 R0 가 1보다 크다면, 
        최소 한 사람 이상이 추가적으로 감염될 수 있다는 뜻이며, 
        이 경우 감염병이 인구 집단 내에서 대확산 될 가능성이 발생한다.
    '''
    s, e, i = x
    
    # New exposure of susceptibles
    β=R0(t)*γ if callable(R0) else R0*γ
    ne = β*s*i
    
    # Time derivatives
    ds = - ne
    de = ne - σ*e
    di= σ*e - γ*i
    
    return ds, de, di
    
# note that R0 can be either constant or a given function of time
# 그래서 callable(R0) 해서 불러올 R0 함수가 있으면 R0(t)*r 이고, 
# 불러올 함수가 없으면 그냥 상수 R0에 대해서 R0*r
# callable(object): 전달받은 object인자가 호출가능한지 여부를 판단해서 T,F를 반환
'''
>>> sample = 1
>>> callable(sample)
False
>>> def funcSample():
...     print('sample')
...
>>> sample = funcSample
>>> callable(sample)
True
'''
# -

# initial conditions of s,e,i
i_0 = 1e-7
e_0 = 4*i_0
s_0 = 1 - i_0 - e_0

# the vector form of initial condition
x_0 = s_0, e_0, i_0


# +
# We solve for the time path numerically using odeint(미분방정식 푸는거)
# at a sequence of dates t_vec

def solve_path(R0, t_vec, x_init=x_0):
    '''
    Solve for i(t) and c(t) via numerical integration,
    given the time path for R0
    '''
    G = lambda x, t: F(x, t, R0)
    s_path, e_path, i_path = odeint(G, x_init, t_vec).transpose()
    
    c_path = 1 - s_path - e_path # cumulative cases
    
    return i_path, c_path


# -

# # Experiments

# +
# The time period we investigate will be 550 days, or around 18 months

t_length = 550
grid_size=1000
t_vec = np.linspace(0, t_length, grid_size) # np.linspace(start, stop, num=숫자 몇개 생성?)
# -

# ## Experiment 1: Constant R0 Case
# - R0가 함수가 아니라 상수인 경우

# +
# We calculate the time path of infected people under different assumptions for R0
# R0 상수값을 다르게 해가면서 infected에 대한 time path를 계산해보기
R0_vals=np.linspace(1.6, 3.0, 6)
labels=[f'$R0 = {r:.2f}$' for r in R0_vals]
i_paths, c_paths = [], []

for r in R0_vals:
    i_path, c_path = solve_path(r, t_vec)
    i_paths.append(i_path)
    c_paths.append(c_path)


# -

# plot the time paths
def plot_paths(paths, labels, times=t_vec):
    fig, ax = plt.subplots()
    
    for path, label in zip(paths, labels):
        ax.plot(times, path, label=label)
        
    ax.legend(loc='upper left')
    
    plt.show()


# current cases as a fraction of the population
# As expected, lower effective transmission rates defer the peak of infections.
# They also lead to a lower peak in current cases.
plot_paths(i_paths, labels)

# cumulative cases as a fraction of population
plot_paths(c_paths, labels)
