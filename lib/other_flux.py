from numpy import array

g = 1.4
gm1 = 0.4
gdgm1 = 3.5

def P2U(P):
    r, u, p = P[0], P[1], P[2]
    U = array([r, r*u, r*u**2/2. + p/gm1])
    return U

def P2F(P):
    r, u, p = P[0], P[1], P[2]
    F = array([r*u, r*u**2 + p, r*u**3/2. + p*u*gdgm1])
    return F
    
def rusanov_flux(Pl, Pr):
    r1, u1, p1 = Pl[0], Pl[1], Pl[2]
    r2, u2, p2 = Pr[0], Pr[1], Pr[2]
    
    #enthalpy
    H1 = u1**2/2. + p1/r1*gdgm1
    H2 = u2**2/2. + p2/r2*gdgm1
    
    #Roe Averages
    rt = (r2/r1)**0.5
    r = rt*r1
    u = (u1 + rt*u2)/(1. + rt)
    H = (H1 + rt*H2)/(1. + rt)
    c = (gm1*(H - u**2/2.))**0.5
    smax = abs(u) + c
    
    #Rusanov flux
    F = 0.5*(P2F(Pl) + P2F(Pr)) - smax*(P2U(Pr) - P2U(Pl))
    return F

def hll_flux(Pl, Pr):
    r1, u1, p1 = Pl[0], Pl[1], Pl[2]
    r2, u2, p2 = Pr[0], Pr[1], Pr[2]
    
    #enthalpy
    H1 = u1**2/2. + p1/r1*gdgm1
    H2 = u2**2/2. + p2/r2*gdgm1
    
    #speed of sound
    c1 = (g*p1/r1)**0.5
    c2 = (g*p2/r2)**0.5
    
    #Roe Averages
    rt = (r2/r1)**0.5
    r = rt*r1
    u = (u1 + rt*u2)/(1. + rt)
    H = (H1 + rt*H2)/(1. + rt)
    c = (gm1*(H - u**2/2.))**0.5
    
    s1min = min(u1 - c1, u - c, 0.)
    s2max = max(u2 + c2, u + c, 0.)
    
    #HLL flux
    F = (s2max*P2F(Pl) - s1min*P2F(Pr) + s1min*s2max*(P2U(Pr) - P2U(Pl)))/(s2max - s1min)
    return F
    
