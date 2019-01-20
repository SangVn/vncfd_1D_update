#!/usr/bin/env python
# coding: utf-8
# Copyright (C) 2018  Nguyen Ngoc Sang, <https://github.com/SangVn> 
from numpy import zeros

cdef double g      = 1.4
cdef double gp1    = g+1
cdef double gm1    = g-1
cdef double gp1d2  = (g+1)/2
cdef double gm1d2  = (g-1)/2
cdef double gp1d2g = (g+1)/(2*g)
cdef double gm1d2g = (g-1)/(2*g)
cdef double gm1dgp1 = (g-1)/(g+1)
cdef double gdgm1  = g/(g-1)
cdef double g2dgm1 = g*2/(g-1)
cdef double eps    = 1e-6     #epsilon - sai số 

#Giải bài toán phân rã gián đoạn trong trên từng bề mặt thể tích hữu hạn
def godunov_flux(Pl, Pr):
    cdef double r1 = Pl[0]
    cdef double u1 = Pl[1]
    cdef double p1 = Pl[2]
    cdef double r2 = Pr[0]
    cdef double u2 = Pr[1]
    cdef double p2 = Pr[2]
    
    #vận tốc âm thanh 
    cdef double c1 = (g*p1/r1)**0.5
    cdef double c2 = (g*p2/r2)**0.5

    #phương pháp lặp tìm P
    cdef double P0 = (p1*r2*c2 + p2*r1*c1 + (u1-u2)*r1*c1*r2*c2)/(r1*c1+r2*c2)
    if P0 < eps: P0 = eps
    cdef double z, alpha, phi
    
    cdef int iterations = 50 # max_iteration 
    while (True):
        P = P0 #áp suất P^{n-1}
        if P >= p1: a1 = (r1*(gp1d2*P + gm1d2*p1))**0.5
        else:
            pp = max(eps, P/p1)
            op = 1. - pp**gm1d2g
            if op>=eps: a1 = gm1d2g*r1*c1*(1. - pp)/op
            else: a1 = r1*c1
        if P >= p2: a2 = (r2*(gp1d2*P + gm1d2*p2))**0.5
        else:
            pp = max(eps, P/p2)
            op = 1. - pp**gm1d2g
            if op>=eps: a2 = gm1d2g*r2*c2*(1. - pp)/op
            else: a2 = r2*c2

        z = P/(p1+p2)
        alpha = gm1/(3*g)*(1. - z)/(z**gp1d2g)/(1. - (z**gm1d2g)) - 1.
        if alpha < 0.: alpha = 0.
        phi = (a2*p1 + a1*p2 + a1*a2*(u1 - u2))/(a1+a2)

        P0 = (alpha*P + phi)/(1. + alpha)#tính P^n
        iterations -= 1
        if (abs(P0 - P) < eps) or (not iterations): break
    #kết thúc vòng lặp!
    
    cdef double U = (a1*u1 + a2*u2 + p1 - p2)/(a1 + a2)
    #xét gián đoạn bên trái 
    if P > p1: #nếu là sóng xung kích 
        D1 = u1 - a1/r1
        R1 = r1*a1/(a1-r1*(u1-U))
    else: #nếu là sóng giãn
        D1 = u1 - c1
        c1star = c1 + gm1d2*(u1-U)
        D1star = U - c1star
        R1 = g*P/c1star**2
    #tương tự cho gián đoạn bên phải 
    if P > p2: #nếu là sóng xung kích 
        D2 = u2 + a2/r2
        R2 = r2*a2/(a2 + r2*(u2-U))
    else: #nếu là sóng giãn
        D2 = u2 + c2
        c2star = c2 - gm1d2*(u2-U)
        D2star = U + c2star
        R2 = g*P/c2star**2
        
    #xét cấu hình phân rã xác định nghiệm PStar = (Rstar, Ustar, Pstar)
    #tùy theo vị trí biên i+1/2 nằm trong vùng nào (xem bài 12)

    if D1>0 and D2>0:   #nằm bên trái sóng trái
        Rstar = r1
        Ustar = u1
        Pstar = p1
    elif D1<0 and D2<0: #nằm bên phải sóng phải
        Rstar = r2
        Ustar = u2
        Pstar = p2
    elif D1<0 and D2>0: #nằm giữa hai sóng 
        if U>=0: Rstar = R1 #nằm bên trái gián đoạn tiếp xúc
        else:    Rstar = R2 #nằm bên phải gián đoạn tiếp xúc
        Ustar = U
        Pstar = P
    elif D1<0 and D1star>0: #nằm trong sóng giãn trái
        Ustar = gm1dgp1*u1 + c1/gp1d2
        Pstar = p1*(Ustar/c1)**g2dgm1
        Rstar = g*p1/Ustar**2   
    elif D2>0 and D2star<0: #nằm trong sóng giãn phải 
        Ustar = gm1dgp1*u2 - c2/gp1d2
        Pstar = p2*(Ustar/c2)**g2dgm1
        Rstar = g*p2/Ustar**2
    else:
        print('Error: godunov_flux!')
        
    #vector biến gốc: PStar = [Rstar, Ustar, Pstar]
    FStar = [0, 0, 0] #vector hàm dòng
    FStar[0] = Rstar*Ustar
    FStar[1] = Rstar*Ustar**2 + Pstar
    FStar[2] = Rstar*Ustar**3/2. + Pstar*Ustar*gdgm1
    
    return FStar

#tìm nghiệm chính xác bài toán Riemann 
def riemann_exact_solution(Pl, Pr, x, xstar, time_target):
    cdef double r1 = Pl[0]
    cdef double u1 = Pl[1]
    cdef double p1 = Pl[2]
    cdef double r2 = Pr[0]
    cdef double u2 = Pr[1]
    cdef double p2 = Pr[2]
    
    #vận tốc âm thanh 
    cdef double c1 = (g*p1/r1)**0.5
    cdef double c2 = (g*p2/r2)**0.5

    #phương pháp lặp tìm P
    cdef double P0 = (p1*r2*c2 + p2*r1*c1 + (u1-u2)*r1*c1*r2*c2)/(r1*c1+r2*c2)
    if P0 < eps: P0 = eps
    cdef double z, alpha, phi
    
    cdef int iterations = 50 # max_iteration 
    while (True):
        P = P0 #áp suất P^{n-1}
        if P >= p1: a1 = (r1*(gp1d2*P + gm1d2*p1))**0.5
        else:
            pp = max(eps, P/p1)
            op = 1. - pp**gm1d2g
            if op>=eps: a1 = gm1d2g*r1*c1*(1. - pp)/op
            else: a1 = r1*c1
        if P >= p2: a2 = (r2*(gp1d2*P + gm1d2*p2))**0.5
        else:
            pp = max(eps, P/p2)
            op = 1. - pp**gm1d2g
            if op>=eps: a2 = gm1d2g*r2*c2*(1. - pp)/op
            else: a2 = r2*c2

        z = P/(p1+p2)
        alpha = gm1/(3*g)*(1. - z)/(z**gp1d2g)/(1. - (z**gm1d2g)) - 1.
        if alpha < 0.: alpha = 0.
        phi = (a2*p1 + a1*p2 + a1*a2*(u1 - u2))/(a1+a2)

        P0 = (alpha*P + phi)/(1. + alpha)#tính P^n
        iterations -= 1
        if (abs(P0 - P) < eps) or (not iterations): break
    #kết thúc vòng lặp!
    
    cdef double U = (a1*u1 + a2*u2 + p1 - p2)/(a1 + a2)

    #xét gián đoạn bên trái 
    if P > p1: #nếu là sóng xung kích 
        D1 = u1 - a1/r1
        D1star = D1
        R1 = r1*a1/(a1-r1*(u1-U))
    else: #nếu là sóng giãn
        D1 = u1 - c1
        c1star = c1 + gm1d2*(u1-U)
        D1star = U - c1star
        R1 = g*P/c1star**2
    #tương tự cho gián đoạn bên phải 
    if P > p2: #nếu là sóng xung kích 
        D2 = u2 + a2/r2
        D2star = D2
        R2 = r2*a2/(a2 + r2*(u2-U))
    else: #nếu là sóng giãn
        D2 = u2 + c2
        c2star = c2 - gm1d2*(u2-U)
        D2star = U + c2star
        R2 = g*P/c2star**2
        
    #Căn cứ vào vị trí của các gián đoạn tại thời điểm t và vị trí điểm x_i 
    w = (x - xstar)/time_target
    P_out = zeros((3, len(x)))
    for i, wi in enumerate(w):
        if wi<=D1:   #nằm bên trái sóng trái
            P_out[0, i] = r1
            P_out[1, i] = u1
            P_out[2, i] = p1
        elif wi>=D2: #nằm bên phải sóng phải
            P_out[0, i] = r2
            P_out[1, i] = u2
            P_out[2, i] = p2
        elif D1star<= wi <= U: #nằm giữa hai sóng 
            P_out[0, i] = R1
            P_out[1, i] = U
            P_out[2, i] = P
        elif U<= wi <= D2star: #nằm giữa hai sóng 
            P_out[0, i] = R2
            P_out[1, i] = U
            P_out[2, i] = P
        elif D1< wi < D1star: #nằm trong sóng giãn trái
            cstar = gm1dgp1*(u1 - wi) + c1/gp1d2
            P_out[1, i] = wi + cstar
            P_out[2, i] = p1*(cstar/c1)**g2dgm1           
            P_out[0, i] = g*P_out[2, i]/cstar**2
        elif D2star< wi < D2: #nằm trong sóng giãn phải 
            cstar = gm1dgp1*(wi - u2) + c2/gp1d2
            P_out[1, i] = wi - cstar
            P_out[2, i] = p2*(cstar/c2)**g2dgm1          
            P_out[0, i] = g*P_out[2, i]/cstar**2
        else:
            print('Error: riemann_exact_solution')

    return P_out
