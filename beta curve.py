#Funciona!!!

import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.scimath import arcsin

def tkm(beta, n):
    thetaim1 = np.pi/2 - beta
    thetaim2 = np.pi/2 - 2*beta
    thetaim3 = thetaim1
    sinThetatm1 = np.clip(np.sin(thetaim1)/n, -1, 1)
    sinThetatm2 = np.clip(np.sin(thetaim2)/n, -1, 1)
    sinThetatm3 = np.clip(np.sin(thetaim3)/n, -1, 1)
    thetatm1 = arcsin(sinThetatm1)
    thetatm2 = arcsin(sinThetatm2)
    thetatm3 = arcsin(sinThetatm3)
    tskm = -(np.sin(thetaim1 - thetatm1)/(np.sin(thetaim1 + thetatm1) + 1e-15)) * \
        (np.sin(thetaim2 - thetatm2)/(np.sin(thetaim2 + thetatm2) + 1e-15)) * \
        (np.sin(thetaim3 - thetatm3)/(np.sin(thetaim3 + thetatm3) + 1e-15))
    #tpkm = (np.tan(thetaim1 - thetatm1)/np.tan(thetaim1 + thetatm1)) * (np.tan(thetaim2 - thetatm2)/np.tan(thetaim2 + thetatm2)) * (np.tan(thetaim3 - thetatm3)/np.tan(thetaim3 + thetatm3))
    tpkm = (np.tan(thetaim1 - thetatm1) / (np.tan(thetaim1 + thetatm1) + 1e-15)) * \
       (np.tan(thetaim2 - thetatm2) / (np.tan(thetaim2 + thetatm2) + 1e-15)) * \
       (np.tan(thetaim3 - thetatm3) / (np.tan(thetaim3 + thetatm3) + 1e-15))

    return tskm, tpkm

def difference(tskm, tpkm, delta, phi, psi):
        s1in = np.cos(2*psi)
        s2in = np.sin(2*psi)*np.cos(delta)
        s3in = np.sin(2*psi)*np.sin(delta)
        
        s0 = 0.5*(np.abs(tpkm)**2 + np.abs(tskm)**2 - (np.abs(tpkm)**2 - np.abs(tskm)**2)*(np.cos(2*psi)*np.cos(2*phi) + np.cos(delta)*np.sin(2*psi)*np.sin(2*phi)))
        if s0 < 1e-15:
             s0 = s0 + 1e-15
        s1 = (1/(4*s0)) * (
            (np.abs(tskm)**2) * (2*np.cos(2*phi) + np.cos(2*psi)*(1 + np.cos(4*phi)) + np.cos(delta)*np.sin(2*psi)*np.sin(4*phi)) +
            (np.abs(tpkm)**2) * (-2 *np.cos(2*phi) + np.cos(2*psi)*(1 + np.cos(4*phi)) + np.cos(delta)*np.sin(2*psi)*np.sin(4*phi)) +
            4 * np.real(np.conjugate(tskm) * tpkm) * (np.cos(2*psi)*np.sin(2*phi)**2 - np.cos(delta)*np.sin(2*psi)*np.sin(4*phi)) +
            4 * np.imag(np.conjugate(tskm) * tpkm) * np.sin(delta)*np.sin(2*psi)*np.sin(2*phi)
            )
        s2 = (1/(4*s0))*(
            4*np.cos(2*phi)*np.sin(2*psi)*(np.real(np.conjugate(tskm)*tpkm)*np.cos(delta)*np.cos(2*phi) - np.imag(np.conjugate(tskm)*tpkm)*np.sin(delta)) + 
            2*np.sin(2*phi)*(np.abs(tskm)**2 - np.abs(tpkm)**2 + np.abs(tskm)**2*np.cos(2*psi)*np.cos(2*phi) + (np.abs(tskm)**2 + np.abs(tpkm)**2)*np.cos(delta)*np.sin(2*psi)*np.sin(2*phi)) +
            np.cos(2*psi)*np.sin(4*phi)*(np.abs(tpkm)**2 - 2*np.real(np.conjugate(tskm)*tpkm))
            )
        s3 = (1/s0)*((np.imag(np.conj(tskm)*tpkm)*(np.cos(delta)*np.cos(2*phi)*np.sin(2*psi) - np.cos(2*psi)*np.sin(2*phi))) +(np.real(np.conjugate(tskm)*tpkm)*np.sin(delta)*np.sin(2*psi)))
        dot = (s1in*s1 + s2in*s2 + s3in*s3)
        dot = np.clip(dot, -1, 1)
        d = np.acos(dot)
        return d, s1, s2, s3


'''def average_distance(beta, n, delta, psi):
    dsum = 0
    tskm, tpkm = tkm(beta, n)
    for i in range(0, 180):
        phi = np.deg2rad(i)
        d = difference(tskm, tpkm, delta, phi, psi)
        dsum += d
    davg = dsum / 180
    return davg''' 
#removido, não representa a integral corretamente, sem pesos
def avarage_distance_weighted(beta, n, delta, psi):
    phis = np.linspace(0, np.pi, 360)
    s1_list = []
    s2_list = []
    s3_list = []
    d_list = []
    tskm, tpkm = tkm(beta, n)

    for phi in phis:
        d, s1, s2, s3 = difference(tskm, tpkm, delta, phi, psi)
        d_list.append(d)
        s1_list.append(s1)
        s2_list.append(s2)
        s3_list.append(s3)
    s1_array = np.array(s1_list)
    s2_array = np.array(s2_list)
    s3_array = np.array(s3_list)
    d_array = np.array(d_list)

    ds1_dphi = np.gradient(s1_array, phis)
    ds2_dphi = np.gradient(s2_array, phis)
    ds3_dphi = np.gradient(s3_array, phis)

    weights = np.sqrt(ds1_dphi**2 + ds2_dphi**2 + ds3_dphi**2)
    if np.sum(weights) < 1e-15:
        avD = np.average(d_array)
    else:
        avD = np.sum(d_array * weights) / np.sum(weights)
    return avD

def beta_curve(n, delta, psi):
    betavalues = np.linspace(0, np.pi/4, 1000)
    distances = []
    lowest_beta = 0
    lowest_d = 100
    largest_d = 0
    for beta in betavalues:

        d = avarage_distance_weighted(beta, n, delta, psi)
        distances.append(d)
        if d > largest_d:
            largest_d = d
        if d < largest_d:
            if d < lowest_d:
                lowest_d = d
                lowest_beta = beta

    betavalues = np.rad2deg(betavalues)
    plt.plot(betavalues, distances)
    plt.show()
    lowest_beta  = round(np.rad2deg(lowest_beta), 3)
    lowest_d = lowest_d /np.pi
    lowest_d = round(lowest_d, 3)
    print(f"Lowest distance: {lowest_d}pi at beta = {lowest_beta} degrees")
    

def main():
    n = 0.033678 + 5.4208j #prata no nosso, 780 nm
    #n = 0.13883 + 4.4909j #ouro no nosso, 750 nm
    #n = 2.3669 + 8.4177j #aluminio no nosso, 750 nm
    #n = 0.1568 + 3.8060j #prata no artigo
    delta = 0
    psi = 0.7854
    beta_curve(n, delta, psi)

main()