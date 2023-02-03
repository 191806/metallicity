# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 14:36:11 2023

@author: lenovo
"""

import numpy as np,matplotlib.pyplot as plt
import pandas as pd
import glob
import os

error=(0.01**2+0.01**2)**0.5        #given magnitude error

data=pd.read_csv('dwarf_feh.csv')
u0=data.loc[:,['u']].values
g0=data.loc[:,['g']].values
i0=data.loc[:,['i']].values
u,g,i=u0.flatten(),g0.flatten(),i0.flatten()
xdata,ydata=g-i,u-g



#First step: refuse data beyond the applicability range
m=[]
n=[]
a,b=0.64,1.24   #a,b denote lower and upper limit of given (g-i), respectively
ind=np.where((xdata>a)&(xdata<b))
xdata=xdata[ind]
ydata=ydata[ind]  
for i in np.arange(0,len(xdata)):
    x=xdata[i]
    y=ydata[i]
    m.append(x)   # m is a empty list to restore (g-i) data
    n.append(y)   # n is a empty list to restore (u-g) data
np.savetxt("dwarf_g-i.csv",m)
np.savetxt("dwarf_u-g.csv",n)  
        


data=pd.read_csv('dwarf_feh.csv')
u0=data.loc[:,['u']].values
g0=data.loc[:,['g']].values
i0=data.loc[:,['i']].values
u,g,i=u0.flatten(),g0.flatten(),i0.flatten()
xdata,ydata=g-i,u-g      
m=[]
n=[]
a,b=0.38,0.64   
ind=np.where((xdata>a)&(xdata<b))
xdata=xdata[ind]
ydata=ydata[ind]
c=-1.5*np.ones(len(xdata))  # c is given [Fe/H] 
a00=1.04783976
a01=0.14091204
a02=0.02273432
a03=-0.0030035
a10=-0.85085737
a11=0.18129285
a12=-0.00705373
a20=2.57171456
a21=-0.0474126
a30=-0.91982667
need=a00+a01*c+a02*c**2+a03*c**3+a10*xdata+a11*xdata*c+a12*xdata*c**2\
             +a20*xdata**2+a21*xdata**2*c+a30*xdata**3  
ind=np.where(ydata>=need)   # choose data above [Fe/H]= -1.5 when 0.38<(g-i)<0.64
xdata=xdata[ind]
ydata=ydata[ind] 
for i in np.arange(0,len(xdata)):
    x=xdata[i]
    y=ydata[i]
    m.append(x)   # m is a empty list to restore (g-i) data
    n.append(y)   # n is a empty list to restore (u-g) data
np.savetxt("dwarf1_g-i.csv",m)
np.savetxt("dwarf1_u-g.csv",n)



data=pd.read_csv('dwarf_feh.csv')
u0=data.loc[:,['u']].values
g0=data.loc[:,['g']].values
i0=data.loc[:,['i']].values
u,g,i=u0.flatten(),g0.flatten(),i0.flatten()
xdata,ydata=g-i,u-g
m=[]
n=[]
a,b=0.26,0.38   
ind=np.where((xdata>a)&(xdata<b))
xdata=xdata[ind]
ydata=ydata[ind]
c=-1*np.ones(len(xdata))  # c is given [Fe/H]   
need=a00+a01*c+a02*c**2+a03*c**3+a10*xdata+a11*xdata*c+a12*xdata*c**2\
             +a20*xdata**2+a21*xdata**2*c+a30*xdata**3 
ind=np.where(ydata>=need)     # choose data that above [Fe/H]=-1 when 0.26<(g-i)<0.38
xdata=xdata[ind]
ydata=ydata[ind]  
for i in np.arange(0,len(xdata)):
    x=xdata[i]
    y=ydata[i]
    m.append(x)   # m is a empty list to restore (g-i) data
    n.append(y)   # n is a empty list to restore (u-g) data
np.savetxt("dwarf2_g-i.csv",m)
np.savetxt("dwarf2_u-g.csv",n)



csv_list=glob.glob('*g-i.csv')
for i in csv_list:
    fr=open(i,'r',encoding='utf-8').read()
    with open('g-i_use.csv','a',encoding='utf-8') as f:
         f.write(fr)         
csv_list=glob.glob('*u-g.csv')
for i in csv_list:
    fr=open(i,'r',encoding='utf-8').read()
    with open('u-g_use.csv','a',encoding='utf-8') as f:
         f.write(fr)
         
         
         
#Second step: predict [Fe/H] with derived polynomial
m=[]
xdata=np.loadtxt("g-i_use.csv",delimiter=',') 
ydata=np.loadtxt("u-g_use.csv",delimiter=',')
for i in np.arange(0,len(xdata)):
    x1=xdata[i]                        # x1 denotes (g-i) 
    y1=ydata[i]                        # y1 denotes (u-g) 
    if (x1>0.64):
        f1=np.linspace(-4,1,101)           # given [Fe/H]
        x10=x1+error*np.random.randn(101)       #given (g-i) with error
        y10=y1+error*np.random.randn(101)       #given (u-g) with error
        need=a00+a01*f1+a02*f1**2+a03*f1**3+a10*x10+a11*x10*f1+a12*x10*f1**2\
             +a20*x10**2+a21*x10**2*f1+a30*x10**3                       
        sigma=error
        likelihood=((2*np.pi)**0.5*sigma)**(-1)*(np.e)**(-((y10-need)**2)/(2*sigma**2))
        f=np.argmax(likelihood)
        m.append(f1[f])
    elif (0.38<=x1<=0.64):
        f1=np.linspace(-1.5,1,51)                    
        x10=x1+error*np.random.randn(51)       
        y10=y1+error*np.random.randn(51)       
        need=a00+a01*f1+a02*f1**2+a03*f1**3+a10*x10+a11*x10*f1+a12*x10*f1**2\
             +a20*x10**2+a21*x10**2*f1+a30*x10**3 
        sigma=error
        likelihood=((2*np.pi)**0.5*sigma)**(-1)*(np.e)**(-((y10-need)**2)/(2*sigma**2))
        f=np.argmax(likelihood)
        m.append(f1[f])
    else:
        f1=np.linspace(-1,1,41)                    
        x10=x1+error*np.random.randn(41)       
        y10=y1+error*np.random.randn(41)       
        need=a00+a01*f1+a02*f1**2+a03*f1**3+a10*x10+a11*x10*f1+a12*x10*f1**2\
             +a20*x10**2+a21*x10**2*f1+a30*x10**3 
        sigma=error
        likelihood=((2*np.pi)**0.5*sigma)**(-1)*(np.e)**(-((y10-need)**2)/(2*sigma**2))
        f=np.argmax(likelihood)
        m.append(f1[f])
np.savetxt("dwarf_feh_predicted.csv",m)

#Last step: delete intermediate files
os.remove("dwarf_u-g.csv")
os.remove("dwarf_g-i.csv")
os.remove("dwarf1_u-g.csv")
os.remove("dwarf1_g-i.csv")
os.remove("dwarf2_u-g.csv")
os.remove("dwarf2_g-i.csv")
os.remove("u-g_use.csv")
os.remove("g-i_use.csv")