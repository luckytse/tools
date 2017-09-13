import PDE
import func
import maths
import sys
import numpy as np
import matplotlib.pyplot as plt

func.hello()
y=maths.Complex(3,-4)
print(y.r)

f=PDE.PDE1D(1,0,0)

N=1000
M=30
tBC=np.linspace(0., 1, M+1)[1:-1]
#tBC=np.concatenate([np.zeros(int(M/2)),np.ones(int(M/2-1))])
xLowBC=np.zeros(N)
xUpBC=np.linspace(0., 1, N)
solution1=f.euler_implicit(0.001,tBC,0.0,xLowBC,1.0,xUpBC)
solution2=f.euler_explicit(0.001,tBC,0.0,xLowBC,1.0,xUpBC)
solution3=f.euler_CrankNicolson(0.001,tBC,0.0,xLowBC,1.0,xUpBC)
solution4=f.euler_theta_method(0.001,tBC,0.0,xLowBC,1.0,xUpBC,0)
xGrid=np.linspace(0., 1, M+1)[1:-1]
plt.plot(xGrid,solution2)
plt.plot(xGrid,solution4)

plt.legend(['implicit', 'explicit','CN','theta'], loc='upper left')
plt.show()
print(solution1)