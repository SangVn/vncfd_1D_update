#!/usr/bin/env python
# coding: utf-8
# Copyright (C) 2018  Nguyen Ngoc Sang, <https://github.com/SangVn> 

# Giải hệ phương trình Euler 1D sơ đồ Godunov
from numpy import zeros
import matplotlib.pyplot as plt
import pyximport; pyximport.install()
from godunov_flux import godunov_flux, riemann_exact_solution
from other_flux import rusanov_flux, hll_flux
from convert_variables import*
from runge_kutta import*
from muscl import*
from weno5 import*

#tính bước thời gian 
def time_step(P, CFL, dx):
    r, u, p = P[:, 0], P[:, 1], P[:, 2]
    c = (1.4*p/r)**0.5 #vận tốc âm thanh 
    u_max = max(abs(u) + c)        
    dt = CFL*dx/u_max
    return dt

def euler_solver(Ps, reconstr, x, time_target, CFL, flux=godunov_flux):
    #xác định phương pháp runge-kutta 
    rk = zeros((3,3))
    runge_kutta_order = 1
    if reconstr == godunov_reconstr:
        runge_kutta_order = 1
        runge_kutta_1(rk)
    elif reconstr == muscl_reconstr:
        runge_kutta_order = 2
        runge_kutta_2(rk)
    elif reconstr == weno5_reconstr:
        runge_kutta_order = 3
        runge_kutta_3(rk)
    else:
        print('Error: reconstruction!')
    
    nx = len(x)
    dx = x[1] - x[0] #xét lưới đều 
    Us = Ps.copy()
    P2U(Ps, Us)
    Fstar = Ps[:-1].copy()
    time = 0.0
    
    while(time < time_target):
        Un = Us.copy()
        #tìm dt
        dt = time_step(Ps, CFL, dx) 
        if(time+dt > time_target): dt = time + dt - time_target
        time += dt
        #print (time)
        for stage in range(runge_kutta_order):
            #bước 1: tái cấu trúc - reconstruction 
            P_left, P_right = reconstr(Ps)
            #bước 2: tìm nghiệm phân rã gián đoạn, tính hàm dòng
            for i in range(nx-1): Fstar[i] = flux(P_left[i], P_right[i])
    
            #bước 3: tích phân theo thời gian
            Us[1:-1] = rk[stage][0]*Un[1:-1] + rk[stage][1]*Us[1:-1] - rk[stage][2]*dt/dx*(Fstar[1:] - Fstar[:-1])
            
            U2P(Us, Ps) #tìm biến nguyên thủy
            
            #điều kiện biên: outflow
            Ps[0] = Ps[1]
            Ps[-1] = Ps[-2]        
    return  Ps.T


#hàm vẽ đồ thị so sánh 4 nghiệm: chính xác, godunov, muscl và weno5 
def plot_4P(x, Ps1, Ps2, Ps3, Ps4, img_name=None, legend=['P1', 'P2', 'P3', 'P4']):
    gm1 = 0.4 #gamma - 1
    e1 = Ps1[2]/(gm1*Ps1[0])
    e2 = Ps2[2]/(gm1*Ps2[0])
    e3 = Ps3[2]/(gm1*Ps3[0])
    e4 = Ps4[2]/(gm1*Ps4[0])
    
    f, axarr = plt.subplots(2, 2, figsize=(18,12))
    axarr[0, 0].plot(x, Ps1[0], x, Ps2[0], x, Ps3[0], x, Ps4[0])
    axarr[0, 0].set_title('density')
    axarr[0, 1].plot(x, Ps1[1], x, Ps2[1], x, Ps3[1], x, Ps4[1])
    axarr[0, 1].set_title('velocity')
    axarr[1, 0].plot(x, Ps1[2], x, Ps2[2], x, Ps3[2], x, Ps4[2])
    axarr[1, 0].set_title('pressure')
    axarr[1, 1].plot(x, e1, x, e2, x, e3, x, e4)
    axarr[1, 1].set_title('energy')
    f.subplots_adjust(hspace=0.2)
    #plt.legend(['exact solution', 'godunov reconstr', 'muscl reconstr', 'weno5 reconstr'])
    plt.legend(legend)
    if img_name is not None: plt.savefig(img_name+'.png')
    plt.show()
    
