import numpy as np
import matplotlib.pyplot as plt

x=np.random.randn(10,1) #generating random x values
y=2*x+np.random.rand() #creating y as a fn of x
w=0.00 #initial attr set to 0
b=0.00

#cost function to determine the cost of the attributes
def cost_function(w,b,x,y):
    m=len(x)
    cost=0
    for i in range(m):
        y_pred=w+b*x[i]
        cost+=(y_pred-y[i])**2
    return cost

#cost gradient calculates the derivatives of attributes
def cost_gradient(w,b,x,y):
    m=len(x)
    dj_dw=0
    dj_db=0
    for i in range(m):
        dj_dw+=(w+b*x[i])-y[i]
        dj_db+=((w+b*x[i])-y[i])*x[i]
    dj_dw=dj_dw/m
    dj_db=dj_db/m
    return dj_dw,dj_db
cost=[]

#performing gradient descent 
def gradientdescent(w,b,x,y,iters,alpha):
    for i in range(iters):
        dj_dw,dj_db=cost_gradient(w,b,x,y)
        temp_w=w-(alpha*dj_dw)
        temp_b=b-(alpha*dj_db)
        w=temp_w
        b=temp_b
        print(cost_function(w,b,x,y),"w=",w,"b=",b)
        cost.append(cost_function(w,b,x,y))
    return w,b
iters=[x for x in range(500)]    
w,b=gradientdescent(w,b,x,y,500,0.01)
plt.scatter(x,y)
plt.plot(b*x+w,y)
plt.show()
plt.plot(iters,cost)
plt.xlabel("iters")
plt.ylabel("cost")
plt.show()
