import numpy as np

######################################
# this file contains
#   ecc_ano(e,l): solve Kepler's equation
#   ecc_anohyp(e,l):solve Kepler's equation hyperbolic case
# orbital elements to cartesian phase space coordinates:
#   cartesian(GM, a, e, i, longnode, argperi, meananom) 
# cartesian phase space to orbital elements
#   keplerian(GM,x,y,z,xd,yd,zd)
# kepstep:   do a kepstep using f+g functions
#   solvex     needed by kepstep, universal coordinate diff kepler's eqn solver
#   C_prussing, S_prussing functions needed by solvex
######################################


# solve Kepler's equation, e<1 case
PREC_ecc_ano=1e-16  # precision
def ecc_ano(e,l):
    du=1.0;
    u0 = l + e*np.sin(l) + 0.5*e*e*np.sin(2.0*l); # first guess
    #also see M+D equation 2.55
    # supposed to be good to second order in e, from Brouwer+Clemence              
    counter=0;
    while (np.abs(du) > PREC_ecc_ano):
        l0 = u0 - e*np.sin(u0);  # Kepler's equation here!
        du = (l - l0)/(1.0 - e*np.cos(u0));
        u0 += du;  # this gives a better guess 
        counter = counter + 1
        if (counter > 10000): 
            break;  
        # equation 2.58 from M+D
        #print(du)
    
    return u0;
# kepler's equation is M = E - e sin E
# here l is M and we want to solve for E (eccentric anomali)

#to test:
#u0 = ecc_ano(e,l)
#print(l, u0 - e*np.sin(u0)) for checking accuracy, it works!


# hyperbolic case
def ecc_anohyp(e,l):
    du=1.0;
    u0 = np.log(2.0*l/e + 1.8); # Danby guess
    counter = 0;
    while(np.abs(du) > PREC_ecc_ano):
        fh = e*np.sinh(u0) - u0 - l;  # Kepler's equation hyperbolic here
        dfh = e*np.cosh(u0) - 1.0;
        du = -fh/dfh;
        u0 = u0 + du;
        counter = counter + 1;
        if (counter > 10000): 
            break;
    
    return u0;

# this things solves M = e sinh(E) - E
# to test:
# u0 = ecc_anohyp(e,l)
# to test:  print(l, e*np.sinh(u0) -u0)


# orbital elements to cartesian phase space coordinates
# parabolic case has not been correctly implemented
def cartesian(GM, a, e, i, longnode, argperi, meananom):
    # solve Kepler's equation, to get eccentric anomali
    if (e<1):
        E0 = ecc_ano(e,meananom);  
    else:
        E0 = ecc_anohyp(e,meananom);
        
        
    if (e<1.0):
        cosE = np.cos(E0);
        sinE = np.sin(E0);
    else: 
        cosE = np.cosh(E0);
        sinE = np.sinh(E0);
        
    a = np.abs(a);
    meanmotion = np.sqrt(GM/(a*a*a));
    foo = np.sqrt(np.abs(1.0 - e*e));
    
    # compute unrotated positions and velocities 
    rovera = (1.0 - e*cosE);
    if (e>1.0): 
        rovera = -1.0*rovera;
        
    x = a*(cosE - e);
    y = foo*a*sinE;
    z = 0.0;
    xd = -a*meanmotion * sinE/rovera;
    yd = foo*a*meanmotion * cosE/rovera;
    zd = 0.0;
    if (e>1.0): 
        x = -1.0*x;
        
    # rotate by argument of perihelion in orbit plane
    cosw = np.cos(argperi);
    sinw = np.sin(argperi);
    xp = x * cosw - y * sinw;
    yp = x * sinw + y * cosw;
    zp = z;
    xdp = xd * cosw - yd * sinw;
    ydp = xd * sinw + yd * cosw;
    zdp = zd;
    
    # rotate by inclination about x axis 
    cosi = np.cos(i);
    sini = np.sin(i);
    x = xp;
    y = yp * cosi - zp * sini;
    z = yp * sini + zp * cosi;
    xd = xdp;
    yd = ydp * cosi - zdp * sini;
    zd = ydp * sini + zdp * cosi;

    # rotate by longitude of node about z axis 
    cosnode = np.cos(longnode);
    sinnode = np.sin(longnode);
    state_x = x * cosnode - y * sinnode;
    state_y = x * sinnode + y * cosnode;
    state_z = z;
    state_xd = xd * cosnode - yd * sinnode;
    state_yd = xd * sinnode + yd * cosnode;
    state_zd = zd;
    return state_x, state_y, state_z, state_xd, state_yd, state_zd

  
# cartesian phase space to orbital elements
def keplerian(GM,x,y,z,xd,yd,zd):
    # find direction of angular momentum vector 
    rxv_x = y * zd - z * yd;
    rxv_y = z * xd - x * zd;
    rxv_z = x * yd - y * xd;
    hs = rxv_x*rxv_x + rxv_y*rxv_y + rxv_z*rxv_z;
    h = np.sqrt(hs);
    r = np.sqrt(x*x + y*y + z*z);
    vs = xd*xd + yd*yd + zd*zd;
    rdotv = x*xd + y*yd + z*zd;
    rdot = rdotv/r;

    orbel_i = np.arccos(rxv_z/h);  #inclination!
    if ((rxv_x !=0.0) or (rxv_y !=0.0)): 
        orbel_longnode = np.arctan2(rxv_x, -rxv_y);
    else:
        orbel_longnode = 0.0;

    orbel_a = 1.0/(2.0/r - vs/GM); # semi-major axis could be negative
    
    ecostrueanom = hs/(GM*r) - 1.0;
    esintrueanom = rdot * h/GM;
    # eccentricity
    orbel_e = np.sqrt(ecostrueanom*ecostrueanom + esintrueanom*esintrueanom);
    
    if ((esintrueanom!=0.0) or (ecostrueanom!=0.0)):
        trueanom = np.arctan2(esintrueanom, ecostrueanom);
    else:
        trueanom = 0.0
        
    cosnode = np.cos(orbel_longnode);
    sinnode = np.sin(orbel_longnode);
    
    # u is the argument of latitude 
    if (orbel_i == np.pi/2.0):  # this work around not yet tested
        u = 0.0
    else:
        rcosu = x*cosnode + y*sinnode;
        rsinu = (y*cosnode - x*sinnode)/np.cos(orbel_i);
        # this will give an error if i is pi/2 *******!!!!!!!!
        if ((rsinu!=0.0) or (rcosu!=0.0)): 
            u = np.arctan2(rsinu, rcosu);
        else:
            u = 0.0;

    orbel_argperi = u - trueanom;  # argument of pericenter
    
    # true anomaly to mean anomaly
    foo = np.sqrt(np.abs(1.0 - orbel_e)/(1.0 + orbel_e));
    if (orbel_e <1.0):
        eccanom = 2.0 * np.arctan(foo*np.tan(trueanom/2.0));
        orbel_meananom = eccanom - orbel_e * np.sin(eccanom);
    else:
        eccanom = 2.0 * np.arctanh(foo*np.tan(trueanom/2.0));
        orbel_meananom = orbel_e*np.sinh(eccanom) - eccanom;
  
    # adjust argperi to [-pi,pi]
    if (orbel_argperi > np.pi): 
        orbel_argperi = orbel_argperi - 2.0*np.pi;
    if (orbel_argperi < -np.pi): 
        orbel_argperi = orbel_argperi + 2.0*np.pi;

    return orbel_a,orbel_e,orbel_i,orbel_longnode,orbel_argperi,orbel_meananom
    
    
def C_prussing(y): # equation 2.40a Prussing + Conway
    if (np.fabs(y)<1e-4):
        return 1.0/2.0*(1.0 - y/12.0*(1.0 - y/30.0*(1.0 - y/56.0)));
    u = np.sqrt(np.fabs(y));
    if (y>0.0):
        return (1.0- np.cos(u))/ y;
    else:
        return (np.cosh(u)-1.0)/-y;

def S_prussing(y): # equation 2.40b Prussing +Conway
    if (np.fabs(y)<1e-4):
        return 1.0/6.0*(1.0 - y/20.0*(1.0 - y/42.0*(1.0 - y/72.0)));
    u = np.sqrt(np.fabs(y));
    u3 = u*u*u;
    if (y>0.0):
        return (u -  np.sin(u))/u3;
    else:
        return (np.sinh(u) - u)/u3;

N_LAG =5.0  # integer for recommeded Laguerre method

# universal solver for kepler's equation
def solvex(r0dotv0, alpha, M1, r0, dt):
    smu = np.sqrt(M1);
    foo = 1.0 - r0*alpha;
    sig0 = r0dotv0/smu;
    x = M1*dt*dt/r0; # initial guess could be improved

    u = 1.0;
    for i in range(7): # or while(fabs(u) > EPS){
        x2 = x*x;
        x3 = x2*x;
        alx2 = alpha*x2;
        Cp = C_prussing(alx2);
        Sp = S_prussing(alx2);
        F = sig0*x2*Cp + foo*x3*Sp + r0*x - smu*dt; # eqn 2.41 PC
        dF = sig0*x*(1.0 - alx2*Sp)  + foo*x2*Cp + r0; # eqn 2.42 PC
        ddF = sig0*(1.0-alx2*Cp) + foo*x*(1.0 - alx2*Sp);
        z = np.fabs((N_LAG - 1.0)*((N_LAG - 1.0)*dF*dF - N_LAG*F*ddF));
        z = np.sqrt(z);
        u = N_LAG*F/(dF + SIGN(dF)*z); // equation 2.43 PC
        x -= u;
        
    return x;

# given M1,x,y,z, vx,vy,vz, calculate new position and velocity at
# time t later,  using f and g functions and formulae from Prussing + Conway's book
#   here M1 is actually G(M_1+M_2)
# do a keplerian time step using f,g functions.
# is robust, covering hyperbolic and parabolic as well as elliptical orbits
# M1 = GM
#returns: xnew,ynew,znew,vxnew,vynew,vznew
K_EPS=1e-16

def kepstep(dt, M1, x,  y,  z, vx, vy, vz):
    r0 = np.sqrt(x*x + y*y + z*z + K_EPS); # current radius
    v2 = (vx*vx + vy*vy + vz*vz);  # current velocity
    r0dotv0 = (x*vx + y*vy + z*vz);
    alpha = (2.0/r0 - v2/M1);  # inverse of semi-major eqn 2.134 MD
    # here alpha=1/a and can be negative
    x_p = solvex(r0dotv0, alpha, M1, r0, dt); # solve universal kepler eqn
    smu = np.sqrt(M1);
    foo = 1.0 - r0*alpha;
    sig0 = r0dotv0/smu;
   
    x2 = x_p*x_p;
    x3 = x2*x_p;
    alx2 = alpha*x2;
    Cp = C_prussing(alx2);
    Sp = S_prussing(alx2);
    r = sig0*x_p*(1.0 - alx2*Sp)  + foo*x2*Cp + r0; # eqn 2.42  PC
    
    # f,g functions equation 2.38a  PC
    f_p= 1.0 - (x2/r0)*Cp;
    g_p= dt - (x3/smu)*Sp;
    # dfdt,dgdt function equation 2.38b PC
    dfdt = x_p*smu/(r*r0)*(alx2*Sp - 1.0);
    dgdt = 1.0 - (x2/r)*Cp;
    
    if (r0 > 0.0): # return something sensible, error catch if a particle is at Sun
        xnew = x*f_p + g_p*vx; # eqn 2.65 M+D
        ynew = y*f_p + g_p*vy;
        znew = z*f_p + g_p*vz;

        vxnew = dfdt*x + dgdt*vx; # eqn 2.70 M+D
        vynew = dfdt*y + dgdt*vy;
        vznew = dfdt*z + dgdt*vz;
        
    else: # do nothing
        xnew = x;
        ynew = y;
        znew = z;

        vxnew = vx;
        vynew = vy;
        vznew = vz;
        
    return xnew,ynew,znew,vxnew,vynew,vznew
   
