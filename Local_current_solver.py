import numpy as np
from scipy.integrate import odeint, simps
from scipy.optimize import least_squares, leastsq

Jcell = np.array([0.10128, 0.20117, 0.40180])   # --- Measured polarization curve: mean current density, A/cm2
Vcell = np.array([0.81869, 0.79964, 0.76343])   # --- Measured polarization curve: cel voltage, V
Rcell = np.array([0.04430, 0.04678, 0.04950])   # --- Measured cell ohmic resisitivty, Ohm*cm2

# --- Measured local current density for Jcell of 100, 200 and 400 mA/cm2:
jlocal_100 = np.array([0.11879, 0.12818, 0.10924, 0.10183, 0.10025, 0.10344, 0.09420, 0.09137, 0.08707, 0.08253])
jlocal_200 = np.array([0.21655, 0.23667, 0.20613, 0.20570, 0.19663, 0.20843, 0.19195, 0.18948, 0.18158, 0.18203])
jlocal_400 = np.array([0.41035, 0.45536, 0.40354, 0.41686, 0.39470, 0.42049, 0.38523, 0.38304, 0.36818, 0.37601])

nexp = 1                  # --- Working point in the arrays Jcell, Vcell, Rcell.
                          # --- nexp = 0 corresponds to Jcell = 101.28 mA/cm2, etc.
jlocal = jlocal_200       # --- Measured local current density corresponding to nexp
fname = 'shapes_200.dat'  # --- File name to store the results

F = 9.65e4                # --- Faraday constant
pcell, Tcell = 1.5, 80    # --- pressure (bar), temperature C
Db = 0.02                 # --- GDL oxygen diffusivity, cm^2/s
iast = 0.001              # --- ORR exchange curernt density, A/cm^3
b = 0.03                  # --- ORR Tafel slope, V

lt, lb, lam = 12e-4, 235e-4, 2.0    # --- CCL thickness, GDL thickness, oxygen flow stoi
cref = (pcell - 0.4734 / 2) * 0.21 / (10 * 8.314 * (273 + Tcell))   # --- reference (inlet) oxygen conc.
                                    # --- cref corresponds to 50% humidity of the inlet air stream
Jmean  = Jcell[nexp]
VcellJ = Vcell[nexp]
Rmem   = Rcell[nexp]
Voc    = 1.1649
J = Jmean / (iast * lt)
Rm = iast * lt * Rmem / b
Kp = np.exp((Voc - VcellJ) / b)

jlim = 4 * F * Db * cref / (lb * iast * lt)
zdist = np.linspace(0, 1, 101)    # --- A mesh for numerical calculation of integral.

def jcz_rhs(y, z, parms):   # --- rhs calculation.  y[0] = j(z), y[1] = c(z)
    denj = (1 + Rm * y[0]) * np.exp(Rm * y[0]) / Kp + 1 / jlim
    rsj = - y[0] / (lam * J * denj)
    rsc = - y[0] / (lam * J)
    return [rsj, rsc]


def jcz_solver(j0):   # --- j(z), c(z) solver. We solve both equations, though only j(z) is needed.
    pset = (j0)
    y0 = [j0, 1.]    #--- initial conditions
    sol = odeint(jcz_rhs, y0, zdist, args=(pset,))
    return sol

  
def shapes(fparms):
    j0 =  fparms[0] / (iast * lt)
    res = jcz_solver(j0)
    return res[:,0], res[:,1]    #--- returns dimensionless j(z), c(z); see normalization!


def resid(fitparms):
    ydata = np.array([1.0])
    jz, cz = shapes(fitparms)
    jtot = simps(jz, x=zdist, even='first') / J    # --- calculates integral using the Simpson's rule
    ymodel = np.array([jtot])
    return ymodel - ydata


def model(fitparms):
    jz, cz = shapes(fitparms)
    jtot = simps(jz, x=zdist, even='first') / J
    ymodel = np.array([jtot])
    return ymodel


def jacoby(fparms):   #--- Numerical Jacobian of the model function
    jacob = np.empty(1)
    ybase = model(fparms)
    dp = 0.001
    fpdp = np.copy(fparms)
    fpdp[0] = fparms[0] * (1 + dp)
    yp = model(fpdp)
    dy = yp - ybase
    jacob[0] = dy / (fparms[0] * dp)
    return jacob


fpinit = np.array([Jmean])
res, success = leastsq(resid, fpinit, Dfun=jacoby, args=())
print('j0 =', res, 'A/cm2')
jz, cz  = shapes(res)

zsegs = np.copy(zdist[5::10])             # --- segment locations in our experiment
jsegs = iast * lt * np.copy(jz[5::10])    # --- model currents in the segments
csegs = np.copy(cz[5::10])                # --- model oxygen conc. at the segments

# print('Voc =', b * np.log(Kp) + Vcell[nexp] )
print('Residual error of the total current =', Jmean - iast * lt * simps(jz, x=zdist, even='first'))
# print('c(1) =', cz[-1])
print('The results are in the file ' + fname, ': zsegs j_exprt j_model c_model' )
np.savetxt(fname, np.transpose([zsegs, jlocal, jsegs, csegs]), fmt="%f")
