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

# # Replication of Andrew Atkeson
# ## Behavior and the Dynamics of Epidemics (2021)

# #### Cases
#
# - with extra mitigation, let mitigate = 1
# - without vaccines, let Lambda = 0
# - with extra mitigation and vaccines, let labmda = 0.004, mitigate = 1
# - with waning immunity and vaccines, lambda = 0.004, mitigate = 0, xi = 1/(1.5 * 365), betabarv = 5 * gamma 
# - without behavior and without vaccines, lambda = 0, kappa = 0, xi = 0
# - eliminate the seasonality, let seasonalsize = 0 
# - eliminate the pandemic fatigue, fatiguesize = 1

# +
import numpy as np
from scipy.stats import norm
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime
import scipy.io
from scipy.integrate import ode
import pandas as pd

from pptx import Presentation      
from pptx.util import Inches

# +
zeta = 1/30;
gamma = 0.4;
sigma = 0.425;
eta = 0.025;
nu = 0.2;
betabar = 3*gamma;
betabarv = 4.5*gamma;
kappa = 2.5000e+05;
Lambda = 0.004; # vaccination, without vaccines, let Lambda = 0

xi = 0*(1/(1.5*365)) # waning immunity R에서 S로 다시 돌아가는 비율, 면역력 지속을 18개월로 가정 

mitigate = 0; # set mitigate = 1 for extra mitigation(mitigation 기간 조정은 함수 식에서! (t>76)&(t<806) 기간 조정하기)

population = 330000000;
seasonalsize = 0.35;
seasonalposition = 20;
fatiguesize = 0.375;
fatiguemean = 285;
fatiguesig = 15;

tv = 289; # 변이 바이러스 도입 날짜
Evbar = 1/population;

E0 = 33/population; #
Ev0 = 0;
S0 = 1-E0;
I0 = 0;
Iv0 = 0;
R0 = 0;
H0 = 0;
D0 = 0;
t0 = 0;  
tfinal = 2*365; # 5*365
y0 = np.zeros([8]);
params = np.zeros([18,1]);
y0[0] = S0;
y0[1] = E0;
y0[2] = Ev0;
y0[3] = I0;
y0[4] = Iv0;
y0[5] = R0;
y0[6] = H0;
y0[7] = D0;
# -

params[0] = betabar;
params[1] = kappa;
params[2] = zeta;
params[3] = gamma;
params[4] = sigma;
params[5] = eta;
params[6] = nu;
params[7] = seasonalsize;
params[8] = fatiguesize;
params[9] = fatiguemean;
params[10] = fatiguesig;
params[11] = betabarv;
params[12] = tv;
params[13] = Evbar;
params[14] = seasonalposition;
params[15] = Lambda;
params[16] = xi;
params[17] = mitigate;


def bsirSEIRHD(t,y):

    S = y[0];
    E = y[1];
    Ev = y[2];
    I = y[3];   
    Iv = y[4];
    R = y[5];
    H = y[6];
    D = y[7];
    
    betabar= params[0];
    kappa = params[1];
    zeta = params[2];
    gamma = params[3];
    sigma = params[4];
    eta = params[5];
    nu = params[6];
    seasonalsize = params[7];
    fatiguesize = params[8];
    fatiguemean = params[9];
    fatiguesig = params[10];
    betabarv = params[11];
    tv = params[12];
    Evbar = params[13];
    seasonalposition = params[14];
    Lambda = params[15];
    xi = params[16];
    mitigate = params[17]
    

    psi = seasonalsize*(np.cos((t+seasonalposition)*2*np.pi/365)-1)/2 - mitigate*((t>76)&(t<806))*0.5;
    kappa = kappa*(1 -norm.cdf(t,fatiguemean,fatiguesig))+fatiguesize*kappa*norm.cdf(t,fatiguemean,fatiguesig);



    beta = betabar*np.exp(-kappa*nu*zeta*H+psi);
    betav = betabarv*np.exp(-kappa*nu*zeta*H+psi);

    x=np.zeros([8])
    x[0] = -beta*S*I - betav*S*Iv - (t>321)*Lambda*S + xi*R; # t>321은 백신 맞기 시작한 시기
    x[1] = beta*S*I - sigma*E;
    x[2] = betav*S*Iv - sigma*Ev +(t<(tv+2))*(t>tv)*Evbar;
    x[3] = sigma*E - gamma*I;
    x[4] = sigma*Ev - gamma*Iv;
    x[5] = (1-nu)*zeta*H + (1-eta)*gamma*(I+Iv) -(t<(tv+2))*(t>tv)*Evbar + (t>321)*Lambda*S - xi*R;
    x[6] = eta*gamma*(I+Iv) - zeta*H;
    x[7] = nu*zeta*H;

    yp = np.transpose(x);

    return yp

# # Key Part: calculate the ODE 

# ### use the "vode" 
# ##### Real-valued Variable-coefficient Ordinary Differential Equation solver, with fixed-leading-coefficient implementation. It provides implicit Adams method (for non-stiff problems) and a method based on backward differentiation formulas (BDF)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html

backend = 'vode'

# ### set ode function 

solver = ode(bsirSEIRHD).set_integrator(backend)

# ### set initial value

solver.set_initial_value(y0, t0)

# ### store the results of ingegrator in loop 

sol = []
time =[]
while solver.successful() and solver.t < tfinal:
    solver.integrate(tfinal, step=True)
    sol.append(solver.y)
    time.append(solver.t)
sol =np.array(sol)
time = np.array(time)

# ### define the variable by using the results of ode

S1 = sol[:,0]
E1 = sol[:,1]
Ev1 = sol[:,2];
I1 = sol[:,3];
Iv1 = sol[:,4];
R1 = sol[:,5];
H1 = sol[:,6];
D1 = sol[:,7]; # cumulative Deaths
Ddot1 = nu*zeta*(eta*gamma*(I1+Iv1) - zeta*H1) 
gD1 = Ddot1/(nu*zeta*H1)

### line 46, 47 in matlab 
cumulativedeaths = D1[tfinal-1]*population
cumulativeinfections = R1[tfinal-1]+D1[tfinal-1]
print(cumulativedeaths, cumulativeinfections)

# ### replicate Figure 1,2
# * use the mat file of US.mat  
# This file makes a matrix USdata with first column being number of day, second column is cumulative US deaths, third column is daily US deaths, fourth column is a seven day moving average of daily US deaths

# +
USDATA = pd.read_excel('usdata2.xlsx', header=None)
tdata = USDATA.iloc[:,0]-USDATA.iloc[0,0];
tdata=tdata.astype('float')

TIME=[]

for j in range(0,len(time)):
    TIME.append(datetime.datetime(2020,2,15)+datetime.timedelta(time[j]))

TIME_R=[]
for j in range(0,len(tdata)):
    TIME_R.append(datetime.datetime(2020,2,15)+datetime.timedelta(tdata[j]))


# +
## line 56, 57 in matlab
## what should I put in place of t1?? --> time?
## t1에 대응하는게 위에 코드에서 datetime.timedelta(time[j]) 인 것 같은데
# t1date에 대응하는 거는 TIME이고..
# tdatadate이 TIME_R이고 

psi_t = seasonalsize*(np.cos((time+seasonalposition)*2*np.pi/365)-1)/2
kappa_t = kappa*(1 - norm.cdf(time,fatiguemean,fatiguesig))+fatiguesize*kappa*norm.cdf(time,fatiguemean,fatiguesig)

# +
# Daily Deaths

fig = plt.figure(figsize=(9,6))

plt.plot(TIME, nu*zeta*H1*population, c='b')
plt.plot(TIME_R, USDATA.iloc[:,3], c='r')

plt.title('Daily Deaths US')
plt.xlabel('days')


# +
# The Fraction of new variant

fig = plt.figure(figsize=(9,6))

plt.plot(TIME, Iv1/(I1+Iv1))
plt.title('The Fraction of new variant in all currently infected US')
plt.xlabel('days')
# -

# The seasonal variation in transmission
fig = plt.figure(figsize=(9,6))
plt.plot(TIME, betabar*np.exp(psi_t)/gamma)
plt.title('The seasonal variation in transmission')
plt.xlabel('days')

# The time path of the semi-elasticity of behavior
fig = plt.figure(figsize=(9,6))
plt.plot(TIME, kappa_t)
plt.title('The time path of the semi-elasticity of behavior')
plt.xlabel('days')







# ## Analyze

# ### Baseline: both seasonality and fatigue

fig = plt.figure(figsize=(9,6))
plt.plot(TIME,D1*population)
plt.plot(TIME_R,USDATA[:,1])
plt.title('Cumulative Deaths US')
plt.xlabel('days')

fig = plt.figure(figsize=(9,6))
plt.plot(TIME,nu*zeta*H1*population)
plt.plot(TIME_R,USDATA[:,2])
plt.title('Daily Deaths US')
plt.xlabel('days')

len(TIME)

TIME

fig = plt.figure(figsize=(9,6))
plt.plot(TIME, nu*zeta*H1*population)
plt.plot(TIME_R,USDATA[:,2])

df

# ### Alternatives: seasonality, fatigue, new variant

# +
cases=2; ##
zeta = 1/30;
gamma = 0.4;
sigma = 0.425;
eta = 0.025;
nu = 0.2;
betabar = 3*gamma;
betabarv = 5*gamma;
kappa = 2.5000e+05;

population = 330000000;
seasonalsize = 0.35;
seasonalposition = 20;
fatiguesize = 0.375;
fatiguemean = 285;
fatiguesig = 15;
tv = 289; # 변이 바이러스 도입 날짜
Evbar = 1/population; ### 1 when new variant is introduced, 0 not introduced
E0 = 33/population; 
Ev0 = 0;
S0 = 1-E0;
I0 = 0;
Iv0 = 0;
R0 = 0;
H0 = 0;
D0 = 0;
t0 = 0;  
tfinal = 1*365;
y0 = np.zeros([8]);
params = np.zeros([16,1]);
y0[0] = S0;
y0[1] = E0;
y0[2] = Ev0;
y0[3] = I0;
y0[4] = Iv0;
y0[5] = R0;
y0[6] = H0;
y0[7] = D0;
# -

params[0] = betabar;
params[1] = kappa;
params[2] = zeta;
params[3] = cases;
params[4] = gamma;
params[5] = sigma;
params[6] = eta;
params[7] = nu;
params[8] = seasonalsize;
params[9] = fatiguesize;
params[10] = fatiguemean;
params[11] = fatiguesig;
params[12] = betabarv;
params[13] = tv;
params[14] = Evbar;
params[15] = seasonalposition;

backend = 'vode'
solver = ode(bsirSEIRHD).set_integrator(backend)
solver.set_initial_value(y0, t0)

sol = []
time =[]
while solver.successful() and solver.t < tfinal:
    solver.integrate(tfinal, step=True)
    sol.append(solver.y)
    time.append(solver.t)
sol =np.array(sol)
time = np.array(time)

# +
TIME=[]

for j in range(0,len(time)):
    TIME.append(datetime.datetime(2020,2,15)+datetime.timedelta(time[j]))

TIME_R=[]
for j in range(0,len(tdata)):
    TIME_R.append(datetime.datetime(2020,2,15)+datetime.timedelta(tdata[j]))
# -

S1 = sol[:,0]
E1 = sol[:,1]
Ev1 = sol[:,2];
I1 = sol[:,3];
Iv1 = sol[:,4];
R1 = sol[:,5];
H1 = sol[:,6];
D1 = sol[:,7]; # cumulative Deaths
Ddot1 = nu*zeta*(eta*gamma*(I1+Iv1) - zeta*H1) # daily ??
gD1 = Ddot1/(nu*zeta*H1)

# daily daeths in us
fig = plt.figure(figsize=(9,6))
plt.plot(TIME, nu*zeta*H1*population)
plt.plot(TIME_R,USDATA[:,2])

# ### With seasonality and fatigue, but only a short expected stay in H

# +
cases=1;
zeta = 1/10; ## 
gamma = 0.4;
sigma = 0.425;
eta = 0.025;
nu = 0.2;
betabar = 3*gamma;
betabarv = 5*gamma;
kappa = 2.5000e+05;

population = 330000000;
seasonalsize = 0.35;
seasonalposition = 20;
fatiguesize = 0.375;
fatiguemean = 285;
fatiguesig = 15;
tv = 289; # 변이 바이러스 도입 날짜
Evbar = 1/population;
E0 = 33/population; #
Ev0 = 0;
S0 = 1-E0;
I0 = 0;
Iv0 = 0;
R0 = 0;
H0 = 0;
D0 = 0;
t0 = 0;  
tfinal = 1*365;
y0 = np.zeros([8]);
params = np.zeros([16,1]);
y0[0] = S0;
y0[1] = E0;
y0[2] = Ev0;
y0[3] = I0;
y0[4] = Iv0;
y0[5] = R0;
y0[6] = H0;
y0[7] = D0;
# -

params[0] = betabar;
params[1] = kappa;
params[2] = zeta;
params[3] = cases;
params[4] = gamma;
params[5] = sigma;
params[6] = eta;
params[7] = nu;
params[8] = seasonalsize;
params[9] = fatiguesize;
params[10] = fatiguemean;
params[11] = fatiguesig;
params[12] = betabarv;
params[13] = tv;
params[14] = Evbar;
params[15] = seasonalposition;

backend = 'vode'
solver = ode(bsirSEIRHD).set_integrator(backend)
solver.set_initial_value(y0, t0)

sol = []
time =[]
while solver.successful() and solver.t < tfinal:
    solver.integrate(tfinal, step=True)
    sol.append(solver.y)
    time.append(solver.t)
sol =np.array(sol)
time = np.array(time)

# +
TIME=[]

for j in range(0,len(time)):
    TIME.append(datetime.datetime(2020,2,15)+datetime.timedelta(time[j]))

TIME_R=[]
for j in range(0,len(tdata)):
    TIME_R.append(datetime.datetime(2020,2,15)+datetime.timedelta(tdata[j]))
# -

S1 = sol[:,0]
E1 = sol[:,1]
Ev1 = sol[:,2];
I1 = sol[:,3];
Iv1 = sol[:,4];
R1 = sol[:,5];
H1 = sol[:,6];
D1 = sol[:,7]; # cumulative Deaths
Ddot1 = nu*zeta*(eta*gamma*(I1+Iv1) - zeta*H1) # daily ??
gD1 = Ddot1/(nu*zeta*H1)

# daily daeths in us
fig = plt.figure(figsize=(9,6))
plt.plot(TIME, nu*zeta*H1*population)
plt.plot(TIME_R,USDATA[:,2])

# ### Model Forecast for 2021 and 2022

# +
cases=1;
zeta = 1/30;
gamma = 0.4;
sigma = 0.425;
eta = 0.025;
nu = 0.2;
betabar = 3*gamma;
betabarv = 5*gamma;
kappa = 2.5000e+05;

population = 330000000;
seasonalsize = 0.35;
seasonalposition = 20;
fatiguesize = 0.375;
fatiguemean = 285;
fatiguesig = 15;
tv = 289; # 변이 바이러스 도입 날짜
Evbar = 1/population; ## new variant is introduced
E0 = 33/population; 
Ev0 = 0;
S0 = 1-E0;
I0 = 0;
Iv0 = 0;
R0 = 0;
H0 = 0;
D0 = 0;
t0 = 0;  
tfinal = 3*365;
y0 = np.zeros([8]);
params = np.zeros([16,1]);
y0[0] = S0;
y0[1] = E0;
y0[2] = Ev0;
y0[3] = I0;
y0[4] = Iv0;
y0[5] = R0;
y0[6] = H0;
y0[7] = D0;
# -

params[0] = betabar;
params[1] = kappa;
params[2] = zeta;
params[3] = cases;
params[4] = gamma;
params[5] = sigma;
params[6] = eta;
params[7] = nu;
params[8] = seasonalsize;
params[9] = fatiguesize;
params[10] = fatiguemean;
params[11] = fatiguesig;
params[12] = betabarv;
params[13] = tv;
params[14] = Evbar;
params[15] = seasonalposition;

backend = 'vode'
solver = ode(bsirSEIRHD).set_integrator(backend)
solver.set_initial_value(y0, t0)

sol = []
time =[]
while solver.successful() and solver.t < tfinal:
    solver.integrate(tfinal, step=True)
    sol.append(solver.y)
    time.append(solver.t)
sol =np.array(sol)
time = np.array(time)

# +
TIME=[]

for j in range(0,len(time)):
    TIME.append(datetime.datetime(2020,2,15)+datetime.timedelta(time[j]))

# -

S1 = sol[:,0]
E1 = sol[:,1]
Ev1 = sol[:,2];
I1 = sol[:,3];
Iv1 = sol[:,4];
R1 = sol[:,5];
H1 = sol[:,6];
D1 = sol[:,7]; # cumulative Deaths
Ddot1 = nu*zeta*(eta*gamma*(I1+Iv1) - zeta*H1) # daily ??
gD1 = Ddot1/(nu*zeta*H1)

fig = plt.figure(figsize=(9,6))
plt.plot(TIME,R1+H1+D1)
plt.title('Portion of the population recovered, hospitalized, or dead US')
plt.xlabel('days')

R1+H1+D1 # converges to 0.75

# +
# daily deaths
fig=plt.figure(figsize=(9,6))
plt.plot(TIME, nu*zeta*H1*population)
plt.plot(TIME_R,USDATA[:,2])

# new variant를 0로 주면 large third wave가 사라진다. 
# This large third wave of deaths in the fall and winter of 2021 is driven by the new, more contagious variant
