import numpy as np

f1_alpha=1000 #set alpha of f1

f3_epsilon=1e-6 #set epsilon of f3

f45_q=10**8

def f1(x): #ellipsoid function
    dim=x.shape[0] #dimension number
    result=0
    for i in range(dim):
        result+=f1_alpha**(i/(dim-1))*x[i]**2
    return result

def g1(x):
    dim=x.shape[0]
    result=np.zeros(dim)
    for i in range(dim):
        result[i]=2*(f1_alpha**(i/(dim-1)))*x[i]
    return result

def h1(x):
    dim=x.shape[0]
    result=np.zeros((dim,dim))
    for i in range(dim):
        result[i,i]=2*(f1_alpha**(i/(dim-1)))
    return result

f2 = lambda x: (1-x[0])**2+100*(x[1]-x[0]**2)**2; #Rosenbrok function

g2 = lambda x: np.array([-400*x[0]*(x[1]-x[0]**2)-2*(1-x[0]), 200*(x[1]-x[0]**2)])

h2 = lambda x: np.array([[2+1200*x[0]**2-400*x[1], -400*x[0]], [-400*x[0], 200]])

f3 = lambda x: np.log(f3_epsilon+f1(x)) #log ellipsoid function

def g3(x):
    dim=x.shape[0]
    result=np.zeros(dim)
    for i in range(dim):
        result[i]=(2*f1_alpha**(i/(dim-1))*x[i])/(f3_epsilon+f1(x))
    return result

def h3(x):
    dim=x.shape[0]
    result=np.zeros((dim,dim))
    f1_elli=f1(x)
    for i in range(dim):
        for j in range(dim):
            if(i==j):
                result[i,j]=((2*f1_alpha**(i/(dim-1)))/(f3_epsilon+f1_elli)
                           - (2*x[i]**2)/(f3_epsilon+f1_elli)**2)
            else:
                result[i,j]=((-4*f1_alpha**((i+j)/(dim-1))*x[i]*x[j])
                           / (f3_epsilon+f1_elli)**2)
    return result

funch = lambda x: (np.log(1+np.exp(-np.absolute(f45_q*x)))+np.maximum(f45_q*x,0))/f45_q
  
def f4(x):
    if(isinstance(x, int) or isinstance(x, float)):
        return (funch(x) + 100*funch(-x))
    else:
        dim=x.shape[0] #dimension number
        result=0
        for i in range(dim):
            result+=funch(x[i])+100*funch(-x[i])
        return result
'''
def f4_g(x):
    dim=x.shape[0] #dimension number
    result=np.zeros(dim)
    for i in range(dim):
        result[i]=(np.exp(f45_q*x[i]))/(1+np.exp(f45_q*x[i]))-100*(np.exp(-f45_q*x[i])/(1+np.exp(-f45_q*x[i])))
    return result
    
def f4_h(x):
    dim=x.shape[0] #dimension number
    result=np.zeros((dim,dim))
    for i in range(dim):
        for j in range(dim):
            if(i==j):
                result[i,j]=(101*f45_q*np.exp(-f45_q*x[i]))/(1+np.exp(-f45_q*x[i]))**2
            else:
                result[i,j]=0
    return result
'''
def h_d1(x):
    if x >= 0:
        return 1 / (1 + np.exp(-x * f45_q))
    return np.exp(x * f45_q) / (1 + np.exp(x * f45_q))

def h_d11(x):
    if x >= 0:
        return -(np.exp(-f45_q * x) / (1 + np.exp(-f45_q * x)))
    return -(1 / (1 + np.exp(f45_q * x)))

def h_d2(x):
    if x >= 0:
        return (f45_q * np.exp(-x * f45_q)) / (1 + np.exp(-x * f45_q))**2
    return (f45_q * np.exp(x * f45_q)) / (1 + np.exp(x * f45_q)) ** 2

def g4(x):
    if isinstance(x, int) or isinstance(x, float):
        return h_d1(x) - 100 * h_d1(-x)
    else:
        d = len(x)
        grad = np.zeros(d)
        for i in range(d):
            grad[i] = h_d1(x[i]) - 100 * h_d1(-x[i])
        return grad

def h4(x):
    if isinstance(x, int) or isinstance(x, float):
        return h_d2(x) + 100 * h_d2(-x)
    else:
        d = len(x)
        hessian = np.zeros((d, d))
        for i in range(d):
            hessian[i, i] = h_d2(x[i]) + 100 * h_d2(-x[i])
        return hessian

def f5(x):
    if(isinstance(x, int) or isinstance(x, float)):
        return funch(x)**2 + 100*funch(-x)**2
    else:
        dim=x.shape[0] #dimension number
        result=0;
        for i in range(dim):
            result+=funch(x[i])**2+100*funch(-x[i])**2
        return result

def g5(x):
    if isinstance(x, int) or isinstance(x, float):
        return 2 * h(x) * h_d1(x) - 100 * 2 * h(-x) * h_d1(-x)
    else:
        d = len(x)
        grad = np.zeros(d)
        for i in range(d):
            grad[i] = 2 * funch(x[i]) * h_d1(x[i]) - 100 * 2 * funch(-x[i]) * h_d1(-x[i])
        return grad

def h5(x):
    if isinstance(x, int) or isinstance(x, float):
        return 2 * np.exp(f45_q * x) * (np.exp(f45_q * x) + np.log(np.exp(f45_q * x) + 1)) / (
                    np.exp(f45_q * x) + 1) ** 2 + 200 * np.exp(-2 * f45_q * x) * (
                           np.exp(f45_q * x) * np.log(np.exp(-f45_q * x) + 1) + 1) / (np.exp(-f45_q * x) + 1) ** 2
    else:
        d = len(x)
        hessian = np.zeros((d, d))
        for i in range(d):
            hessian[i, i] = 2 * h_d1(x[i])**2 + 2*funch(x[i])*h_d2(x[i]) + 200*h_d11(x[i])**2
            # 2 * np.exp(f45_q * x[i]) * (np.exp(f45_q * x[i]) + np.log(np.exp(f45_q * x[i]) + 1)) / (
            #     np.exp(f45_q * x[i]) + 1) ** 2 + 200 * np.exp(-2 * f45_q * x[i]) * (
            #            np.exp(f45_q * x[i]) * np.log(np.exp(-f45_q * x[i]) + 1) + 1) / (np.exp(-f45_q * x[i]) + 1) ** 2
        return hessian
