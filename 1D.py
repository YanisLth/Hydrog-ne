import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

hbar = 1
m = 1
a= 1 # unité atomique

Nr = 750
Rmax = 30
dr = Rmax / Nr
dt = 0.001
Nt = 2000

r = np.linspace(0.1, Rmax, Nr)
n = 2
l = 0  

V_eff = -1/r + l*(l+1)/(2*r**2)
V = np.diag(V_eff)

r0 = 5
sigma = 2

k=2*l+1
p= n-l-1
x=(2*r)/(n*a)

def Laguerre_associe(p, k, x):
    if p == 0:
        return 1
    if p == 1:
        return -x + k + 1
    
    Lm2 = 1
    Lm1 = -x + k + 1
    for m in range(2, p+1):
        L = ((2*m + k - 1 - x)*Lm1 - (m + k - 1)*Lm2) / m
        Lm2, Lm1 = Lm1, L
    return L

Rnl = (x**l) * np.exp(-x/2) * Laguerre_associe(p, k, x)
u = r * Rnl # paquet initial
u /= np.sqrt(np.trapz(np.abs(u)**2, r))

P = (np.eye(Nr, k=1) - 2*np.eye(Nr) + np.eye(Nr, k=-1)) / dr**2

H = -(1/2) * P + V

I = np.eye(Nr)
A = I - 1j * dt/2 * H
Z = I + 1j * dt/2 * H

plt.ion()
fig, ax = plt.subplots()
line, = ax.plot(r, np.abs(u)**2)
ax.set_xlabel("r")
ax.set_ylabel("|u(r)|^2")
ax.set_title("Propagation radiale avec l(l+1)/2r^2")
ax.grid()

for t in range(Nt):
    u = solve(A, Z @ u)
    u /= np.sqrt(np.trapz(np.abs(u)**2, r))

    if t % 10 == 0:
        line.set_ydata(np.abs(u)**2)
        plt.draw()
        plt.pause(0.01)

plt.ioff()
plt.show()
