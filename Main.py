global opt_aV
import math
import matplotlib.pyplot as plt
import random
import numpy as np
from pyrsistent import l
from scipy.integrate import odeint
import tensorflow as tf
from tensorflow import keras
def model(x,t,temp):
    y1=-4000*x[0]**2*math.exp(-2500/temp)
    y2=4000*x[0]**2*math.exp(-2500/temp)-620000*x[1]*math.exp(-5000/temp)
    y3=1
    return y1,y2,y3

def data(steps=10):
    action=[]
    state1=[]
    state2=[]
    state3=[]
    x0=[0,0,0]
    for ep in range(100):
        x0[0]=random.random()
        x0[1]=0
        x0[2]=0
        for i in range(steps):
            temp=random.randint(298,398)
            action.append(temp)
            state1.append(x0[0])
            state2.append(x0[1])
            state3.append(x0[2])
            t = np.arange(i/steps,((i+1.01)/steps),1/100)
            y1= odeint(model,x0,t,args=(temp,))
            x0=y1[-1]
        state1.append(x0[0])
        state2.append(x0[1])
        state3.append(x0[2])
    print('1',max(state2))
    sMx=np.array([state1,state2,state3])
    sMx=sMx.transpose()
    return sMx

def sim_next(x,temp):
    t = np.arange(0,0.1,1/100)
    y1=odeint(model,x,t,args=(temp,))
    nx1=y1[-1,0]
    nx2=y1[-1,1]
    nx3=y1[-1,2]
    nx4 = 0; #store reward
    if nx3 >= 1: # end of time
        nx4 = nx2; # reward = nx2
    return [nx1,nx2,nx3,nx4]

m =1100 # number of samples
k = 11  # number of actions
N_max =10 # number of iterations for the reinforcment learning
sMx = data() # To be generated vis several episodes 
aV = list(range(298,408,10))
# for i in range(m):
#     for j in range(k):
NSCube = [[sim_next(sMx[i],aV[j]) for j in range(k)] for i in range(m)] # stores next_state_vect & the
#associated reward
opt_aV = np.zeros(m)
qV = -math.inf*np.ones(m)
for N in range(N_max):
    for i in range(m):
        for j in range(k):
            nsV = NSCube[i][j][0:3]
            nsV=np.tile(nsV, (11, 1))
            nq = NSCube[i][j][3] # the reward associated with next state
            if N>0:

                zin=np.insert(nsV,3, aV, axis=1)
                nq = nq + max(mdl.predict(zin)) 
            if nq > qV[i]:
                qV[i] = nq
                opt_aV[i] = aV[j]
    mdl = tf.keras.Sequential()
    mdl.add(tf.keras.layers.Dense(32,input_shape=(None,4)))
    mdl.add(tf.keras.layers.Dense(32))
    mdl.add(tf.keras.layers.Dense(1))
    mdl.compile(optimizer='Adam', loss='mse')
    opt_aV=np.array(opt_aV)

    xin=np.insert(sMx,3, opt_aV, axis=1)
    mdl.fit(xin, qV, batch_size=32, epochs=5)
    print('episodes',N)
# mdl.save('C:/Users/abdoe/Downloads')

mdl1= tf.keras.Sequential()
mdl1.add(tf.keras.layers.Dense(32))
mdl1.add(tf.keras.layers.Dense(32))
mdl1.add(tf.keras.layers.Dense(1))
mdl1.compile(optimizer='Adam', loss='mse')
mdl1.fit(sMx,opt_aV, batch_size=32, epochs=50)
x0f=[1,0,0]
action=[]
final=[]
for j in range(101):
    act=mdl1.predict(np.array([x0f]))
    action.append(act[0][0].tolist())
    t=np.arange(j/100,(j+1)/100,1/1000)
    y1=odeint(model,x0f,t,args=(act,))
    x0f=y1[-1]
    
   
print(y1[-1,1])
print(action)

plt.plot(action)
plt.show()