import numpy as np

from lmfit import minimize, Parameters

import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['mathtext.fontset'] = 'dejavuserif'
rcParams['font.family'] = 'serif'
rcParams['xtick.labelsize'] = 16
rcParams['ytick.labelsize'] = 16
rcParams['axes.grid'] = True
rcParams['axes.titlesize'] = 24
rcParams['axes.labelsize'] = 20
rcParams['axes.titlepad'] = 15
rcParams['legend.frameon'] = True
rcParams['legend.facecolor'] = 'white'
rcParams['legend.fontsize'] = 18

vs_dat, pwrs_dat = np.loadtxt('pwrcurve.csv', delimiter=',', unpack=True,
                              skiprows=0)
sort_is=np.argsort(vs_dat)
vs_dat=vs_dat[sort_is]
pwrs_dat=pwrs_dat[sort_is]

def pwr_f(v,params):
    #a * v^3 + b * v^2 + c * v + d
    a=params['a'].value
    b=params['b'].value
    c=params['c'].value
    d=params['d'].value
    pwr = a*v**3. + b*v**2. + c*v + d
    return pwr
def resids_f(params, vs, pwrs_dat):
    pwrs_est=pwr_f(vs, params)
    resids = pwrs_est - pwrs_dat
    return resids 
params=Parameters()
params.add('a', vary=True, value=1.)
params.add('b', vary=True, value=1.)
params.add('c', vary=True, value=1.)
params.add('d', vary=True, value=0.)

fit = minimize(resids_f, params, args=(vs_dat, pwrs_dat))
with open('result.txt','w') as f:
    f.write('{}'.format(fit.params))
print(fit.params)

fig=plt.figure(figsize=(8,6))
ax=fig.add_subplot(111)
ax.plot(vs_dat, pwrs_dat, 'ob', label='data')
ax.plot(vs_dat, pwr_f(vs_dat, fit.params), '-r', label='fit')
ax.set_xlabel('velocity / [mi / hr]')
ax.set_ylabel('power / W')
ax.legend()
bbox_props=dict(boxstyle='square', fc='white') #Define props of txt box
ax.text(-0.2,300,
        '$P=a\cdot v^3+b\cdot v^2+c\cdot v+d$'
        '\n$a={0:0.4f}\pm{1:0.4f}$'
        '\n$b={2:0.2f}\pm{3:0.2f}$'
        '\n$c={4:0.1f}\pm{5:0.1f}$'
        '\n$d={6:0.0f}\pm{7:0.0f}$'\
        .format(fit.params['a'].value, fit.params['a'].stderr,
                fit.params['b'].value, fit.params['b'].stderr,
                fit.params['c'].value, fit.params['c'].stderr,
                fit.params['d'].value, fit.params['d'].stderr),
        fontsize=14, bbox=bbox_props, va='top')
plt.savefig('result.png', dpi=300)
plt.show()
