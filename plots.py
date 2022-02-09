import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import MaxNLocator
import numpy as np

def cal_rate_convergence(xs, fun):
    x_ = xs[-1]
    x1 = xs[:-1]
    x2 = xs[1:]
    res = []
    for i in range(len(x1)):
        if fun(x1[i])-fun(x_) == 0:
            continue
        res.append((fun(x2[i])-fun(x_))/(fun(x1[i])-fun(x_)))
    return np.array(res)

def plot_convergence(xs, fun, label="", name=""):
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # _x = xs[-1]
    # xx = cal_convergence(xs)
    xx = cal_rate_convergence(xs, fun)
    it = np.arange(xx.shape[0]-1)
    plt.plot(it, xx[:-1], label=label)
    # plt.legend()
    plt.yscale("log")
    plt.title('convergence rate changes in '+name)
    plt.xlabel("iteration")
    plt.ylabel("convergence rate")
    plt.savefig(name+'_converge.pdf')
    #plt.show()
    plt.close('all')

def plot_deltas(dels, label="", name=""):
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    it = np.arange(dels.shape[0])
    plt.plot(it, dels, label=label)
    plt.title('Δk changes in '+name)
    plt.xlabel("iteration")
    plt.ylabel("Δk")
    plt.savefig(name+'_deltas.pdf')
    #plt.show()
    plt.close('all')

def init_x(step,start,end,dimension): #set the start&end coord, step size and dimension number pf input set
    xi=np.arange(start, end, step); x=xi #initialization of input xi for each dimension
    for i in range(dimension-1):
        x=np.vstack((np.around(x,decimals=9),np.around(xi,decimals=9))) #make x to d dimensions, from xi
    return x

def gen2d(step,start,end,dimension):
    x=init_x(step,start,end,dimension) #0.01,-3.5,3.5,2
    X1,X2 = np.meshgrid(x[0], x[1]) #generate all the data point
    dtsize=X1.shape[0] #data point number
    Y=np.zeros((dtsize,dtsize)) #initialize output results to 2D
    return X1,X2,Y

def plot2d(X1,X2,Y,x0,f,f_input,fcount,f_y,name):
    dtsize=X1.shape[0]
    for i in range(dtsize):
        for j in range(dtsize):
            X=np.vstack((np.around(X1[i,j],decimals=9),np.around(X2[i,j],decimals=9))) #choose every combination of 2D inputs
            Y[i,j]=f(X) #store the results
    fx1=np.zeros(fcount+1)
    fx2=np.zeros(fcount+1)
    for i in range(fcount+1):
        if(i<=0):
            fx1[i]=x0[0]
            fx2[i]=x0[1]
        else:
            fx1[i]=f_input[i-1][0]
            fx2[i]=f_input[i-1][1]
    #plot in 2D with color
    fig, ax = plt.subplots()
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    if(name=='f2'):
    	lv=[0,1,3,10,30,100,300,1000,3000,5000,8000,11000,15000,19000,25000]
    	Cset = plt.contourf(X1, X2, Y, levels=lv,norm=colors.PowerNorm(gamma=0.25),cmap='coolwarm_r')
    elif(name=='f3'):
    	Cset = plt.contourf(X1, X2, Y, levels=15,norm=colors.PowerNorm(gamma=2),cmap='coolwarm_r')
    elif(name=='f4'):
    	lv=[0,1,3,6,10,15,25,50,100,150,200,250,300,350,410,480,560,650,750]
    	Cset = plt.contourf(X1, X2, Y, levels=lv,norm=colors.PowerNorm(gamma=0.5),cmap='coolwarm_r')
    elif(name=='f5'):
    	lv=[0,1,3,6,10,15,25,50,100,200,300,500,700,1000,1400,1900,2500]
    	Cset = plt.contourf(X1, X2, Y, levels=lv,norm=colors.PowerNorm(gamma=0.4),cmap='coolwarm_r')
    else:
    	lv=[0,10,30,100,300,1000,2000,3300,4900,6800,8900,11300,13000]
    	Cset = plt.contourf(X1, X2, Y, levels=lv,norm=colors.PowerNorm(gamma=0.4),cmap='coolwarm_r')
    plt.plot(fx1, fx2,c="k")
    plt.colorbar(Cset)
    plt.title(name+' start at (%.1f,%.1f)' %(x0[0],x0[1]))
    plt.tight_layout()
    plt.savefig(name+'_2d.pdf')
    #plt.show()
    plt.close('all')