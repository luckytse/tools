
#make a parabolic equation
import numpy as np
from scipy import linalg

class PDE1D:
    #parabolic du/dt+a(t,x)d^2u/dx^2+b(t,x)du/dx+c(t,x)=0
    def __init__(self, a,b,c):
        self.a=a
        self.b=b
        self.c=c

    #this function needs modification, will be used to replace the repeated parts in other functions
    def pre_process(self,T,initialBC,xLower,xLowerBC,xUpper,xUpperBC):
        if self.b!=0 or self.c!=0:
            return 'please change the PDE to standard form'
        if xLowerBC.__len__()!=xUpperBC.__len__():
            return 'length of boundary condition should be equal'
        if initialBC.__len__()<=2:
            return 'please increase spatial discretization'

        M=initialBC.__len__()+1
        dx = (xUpper - xLower) / M

        N=xLowerBC.__len__()-1
        dt = T / N

        k=self.a*dt/dx**2

        return
    #initialBC: initial condition with M partitions, initialBC is with size M-1 by 1;
    # xLowerBC,xUpperBC: spatial boundary condition with N partition, with size N+1 by 1, this boundary condition include terminal time and start time;
    def euler_explicit(self,T,initialBC,xLower,xLowerBC,xUpper,xUpperBC):
        if self.b!=0 or self.c!=0:
            return 'please change the PDE to standard form'
        if xLowerBC.__len__()!=xUpperBC.__len__():
            return 'length of boundary condition should be equal'
        if initialBC.__len__()<=2:
            return 'please increase spatial discretization'

        M=initialBC.__len__()+1
        dx = (xUpper - xLower) / M

        N=xLowerBC.__len__()-1
        dt = T / N

        k=self.a*dt/dx**2
        if k>0.5:
            return 'please partition again in order for a*dt/dx^2 be less than 1/2'
        # interation matrix
        A=np.diag((1-2*k)*np.ones(M-1),0)+np.diag(k*np.ones(M-2),1)+np.diag(k*np.ones(M-2),-1)
        #boundary vector
        f=np.zeros(M-1)


        res=initialBC
        for i in range(1,N):
            f[0] = xLowerBC[-i]
            f[-1] = xUpperBC[-i]
            res=A.dot(res)+k*f

        return res

    #initialBC: initial condition with M partitions, initialBC is with size M-1 by 1;
    # xLowerBC,xUpperBC: spatial boundary condition with N partition, with size N+1 by 1, this boundary condition include terminal time and start time;
    def euler_implicit(self,T,initialBC,xLower,xLowerBC,xUpper,xUpperBC):
        if self.b!=0 or self.c!=0:
            return 'please change the PDE to standard form'
        if xLowerBC.__len__()!=xUpperBC.__len__():
            return 'length of boundary condition should be equal'
        if initialBC.__len__()<=2:
            return 'please increase spatial discretization'

        M=initialBC.__len__()+1
        dx = (xUpper - xLower) / M

        N=xLowerBC.__len__()-1
        dt = T / N

        k = self.a * dt / dx ** 2

        # interation matrix
        A=np.diag((1+2*k)*np.ones(M-1),0)-np.diag(k*np.ones(M-2),1)-np.diag(k*np.ones(M-2),-1)
        #boundary vector
        f=np.zeros(M-1)

        res=initialBC
        for i in range(1,N):
            f[0] = xLowerBC[-1-i]
            f[-1] = xUpperBC[-1-i]
            res=np.linalg.solve(A, res+k*f)
        return res

    #initialBC: initial condition with M partitions, initialBC is with size M-1 by 1;
    # xLowerBC,xUpperBC: spatial boundary condition with N partition, with size N+1 by 1, this boundary condition include terminal time and start time;
    def euler_CrankNicolson(self,T,initialBC,xLower,xLowerBC,xUpper,xUpperBC):
        if self.b!=0 or self.c!=0:
            return 'please change the PDE to standard form'
        if xLowerBC.__len__()!=xUpperBC.__len__():
            return 'length of boundary condition should be equal'
        if initialBC.__len__()<=2:
            return 'please increase spatial discretization'

        M=initialBC.__len__()+1
        dx = (xUpper - xLower) / M

        N=xLowerBC.__len__()-1
        dt = T / N

        k = self.a * dt / dx ** 2

        # interation matrix
        A=np.diag((1+k)*np.ones(M-1),0)-np.diag(k/2*np.ones(M-2),1)-np.diag(k/2*np.ones(M-2),-1)
        B = np.diag((1 - k) * np.ones(M - 1), 0) + np.diag(k / 2 * np.ones(M - 2), 1) + np.diag(k / 2 * np.ones(M - 2),
                                                                                                -1)        #boundary vector
        f=np.zeros(M-1)

        res=initialBC
        for i in range(1,N):
            f[0] = 1.0/2*(xLowerBC[-1-i]+xLowerBC[-i])
            f[-1] = 1.0/2*(xUpperBC[-1-i]+xUpperBC[-i])
            res=np.linalg.solve(A, B.dot(res)+k*f)
        return res

    #initialBC: initial condition with M partitions, initialBC is with size M-1 by 1;
    # xLowerBC,xUpperBC: spatial boundary condition with N partition, with size N+1 by 1, this boundary condition include terminal time and start time;
    def euler_theta_method(self,T,initialBC,xLower,xLowerBC,xUpper,xUpperBC,theta):
        if theta<0 or theta>1:
            return 'theta should be in range [0,1]'
        if self.b!=0 or self.c!=0:
            return 'please change the PDE to standard form'
        if xLowerBC.__len__()!=xUpperBC.__len__():
            return 'length of boundary condition should be equal'
        if initialBC.__len__()<=2:
            return 'please increase spatial discretization'

        M=initialBC.__len__()+1
        dx = (xUpper - xLower) / M

        N=xLowerBC.__len__()-1
        dt = T / N

        k = self.a * dt / dx ** 2

        # interation matrix
        A=np.diag((1+2*theta*k)*np.ones(M-1),0)-np.diag(theta*k*np.ones(M-2),1)-np.diag(theta*k*np.ones(M-2),-1)
        B = np.diag((1 - 2*(1-theta)*k) * np.ones(M - 1), 0) + np.diag((1-theta)*k* np.ones(M - 2), 1) + np.diag((1-theta)*k * np.ones(M - 2),
                                                                                                -1)        #boundary vector
        f=np.zeros(M-1)

        res=initialBC
        for i in range(1,N):
            f[0] = theta*xLowerBC[-1-i]+(1-theta)*xLowerBC[-i]
            f[-1] = theta*xUpperBC[-1-i]+(1-theta)*xUpperBC[-i]
            res=np.linalg.solve(A, B.dot(res)+k*f)
        return res