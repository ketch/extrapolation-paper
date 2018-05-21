"""
Definitions of test problems.
"""
import fnbod
import numpy as np
from collections import namedtuple
import scipy
import fbrusselator

Problem = namedtuple("ProblemDefinition", ["name", "rhs", "rhs_reversed", "jacobian", "solout", "output_times", "y0", "atol","size"])

def kdv_setup(N=256):
    def kdv_init(t0):
        k = np.array(list(range(0,int(N/2))) + [0] + list(range(-int(N/2)+1,0)))
        E_ = np.exp(-1j * k**3 * t0)
        x = (2*np.pi/N)*np.arange(-int(N/2),int(N/2))
        A = 25
        B = 16
        u = 3*A**2/np.cosh(0.5*(A*(x+2.)))**2 + 3*B**2/np.cosh(0.5*(B*(x+1)))**2
        U_hat = E_*np.fft.fft(u)
        return U_hat

    def kdv_rhs(U_hat, t):
        # U_hat := exp(-i*k^3*t)*u_hat
        k = np.array(list(range(0,int(N/2))) + [0] + list(range(-int(N/2)+1,0)))
        E = np.exp(1j * k**3 * t)
        E_ = np.exp(-1j * k**3 * t)
        g = -0.5j * E_ * k
        return g*np.fft.fft(np.real(np.fft.ifft(E*U_hat))**2)

    def kdv_rhs_reversed(U_hat, t):
        return kdv_rhs(t, U_hat)

    def kdv_solout(U_hat):
        t = 0.003
        N = len(U_hat)
        k = np.array(list(range(0,int(N/2))) + [0] + list(range(-int(N/2)+1,0)))
        E = np.exp(1j * k**3 * t)
        return np.squeeze(np.real(np.fft.ifft(E*U_hat)))

    return Problem(name="kdv"+str(N),
                   rhs=kdv_rhs,
                   rhs_reversed=kdv_rhs_reversed,
                   jacobian=None,
                   solout=kdv_solout,
                   output_times=[0,0.003],
                   y0=kdv_init(0),
                   atol=None,
                   size=None)


def burgers_setup(N=64, epsilon=0.1):

    def burgers_init(t0):
        k = np.array(list(range(0,int(N/2))) + [0] + list(range(-int(N/2)+1,0)))
        E = np.exp(epsilon * k**2 * t0)
        x = (2*np.pi/N)*np.arange(-int(N/2),int(N/2))
        u = np.sin(x)**2 * (x<0.)
        U_hat = E*np.fft.fft(u)
        return U_hat

    def burgers_rhs(U_hat, t):
        # U_hat := exp(epsilon*k^2*t)*u_hat
        k = np.array(list(range(0,int(N/2))) + [0] + list(range(-int(N/2)+1,0)))
        E = np.exp(epsilon * k**2 * t)
        E_ = np.exp(-epsilon * k**2 * t)
        g = -0.5j * E * k
        return g*np.fft.fft(np.real(np.fft.ifft(E_*U_hat))**2)

    def burgers_rhs_reversed(U_hat, t):
        return burgers_rhs(t, U_hat)

    def burgers_solout(U_hat):
        t = 3.
        k = np.array(list(range(0,int(N/2))) + [0] + list(range(-int(N/2)+1,0)))
        E_ = np.exp(-epsilon * k**2 * t)
        return np.squeeze(np.real(np.fft.ifft(E_*U_hat)))

    return Problem(name="burgers"+str(N)+'_'+str(epsilon),
                   rhs=burgers_rhs,
                   rhs_reversed=burgers_rhs_reversed,
                   jacobian=None,
                   solout=burgers_solout,
                   output_times=[0.,3.],
                   y0=burgers_init(0.),
                   atol=None,
                   size=N)


def vdpol_rhs(y,t):
    epsilon=1e-6
    y2=1/epsilon*(((1-y[0]**2)*y[1])-y[0])
    return np.array([y[1],y2])

def vdpol_rhs_reversed(t,y):
    return vdpol_rhs(y,t)

def vdpol_jac(y,t):
    epsilon=1e-6
    J12=1/epsilon*(-2*y[0]*y[1]-1)
    J22=1/epsilon*(1-y[0]**2)
    return np.array([[0,1],[J12,J22]])


N = 20
alpha = 0.1

def five_pt_laplacian_sparse_periodic(m,a,b):
    """Construct a sparse matrix that applies the 5-point laplacian discretization
       with periodic BCs on all sides."""
    e=np.ones(m**2)
    e2=([1]*(m-1)+[0])*m
    e3=([0]+[1]*(m-1))*m
    h=(b-a)/(m-1)
    A=scipy.sparse.spdiags([-4*e,e2,e3,e,e],[0,-1,1,-m,m],m**2,m**2)
    # Top & bottom BCs:
    A_periodic_top = scipy.sparse.spdiags([e[0:m]],[2*m-m**2],m**2,m**2).transpose()
    A_periodic_bottom = scipy.sparse.spdiags(np.concatenate((np.zeros(m),e[0:m])),[2*m-m**2],m**2,m**2)
    A_periodic = A_periodic_top + A_periodic_bottom
    # Left & right BCs:
    for i in range(m):
        A_periodic[i*m,(i+1)*m-2] = 1.
        A_periodic[(i+1)*m-1,i*m+1] = 1.
    A = A + A_periodic
    A/=h**2
    A = A.tocsr()
    return A


A=five_pt_laplacian_sparse_periodic(N,0,1)


def FortBRUSS2Df(y,t):
    "Compiled Fortran brusselator 2D RHS function (faster than pure Python)."
    aux=fbrusselator.fnbruss(y,t,N)
    return aux

def brusselator_rhs_reversed(t,y):
    return FortBRUSS2Df(y,t)

def BRUSS2Dgradnonsparse(yn,tn):
    U=yn[0:N**2]
    V=yn[N**2:2*N**2]
    df1du = scipy.sparse.spdiags(2*U*V-4.4,0,N**2,N**2)+alpha*A
    df1dv = scipy.sparse.spdiags(U**2,0,N**2,N**2)
    df2du = scipy.sparse.spdiags(3.4-2*U*V,0,N**2,N**2)
    df2dv = scipy.sparse.spdiags(-U**2,0,N**2,N**2)+alpha*A
    left  = scipy.sparse.vstack([df1du,df2du])
    right = scipy.sparse.vstack([df1dv,df2dv])
    final = scipy.sparse.hstack([left, right]).todense()
    return final

def BRUSS2Dgrad(yn,tn):
    U=yn[0:N**2]
    V=yn[N**2:2*N**2]
    df1du = scipy.sparse.spdiags(2*U*V-4.4,0,N**2,N**2)+alpha*A
    df1dv = scipy.sparse.spdiags(U**2,0,N**2,N**2)
    df2du = scipy.sparse.spdiags(3.4-2*U*V,0,N**2,N**2)
    df2dv = scipy.sparse.spdiags(-U**2,0,N**2,N**2)+alpha*A
    left = scipy.sparse.vstack([df1du,df2du])
    right = scipy.sparse.vstack([df1dv,df2dv])
    final = scipy.sparse.hstack([left, right], format='csr')
    return final


def brusselator_setup(N=20, alpha=0.1):

    step=1/(N-1)
    x=np.multiply(step,list(range(N))*N)
    y=np.multiply(step,np.repeat(list(range(N)),N))

    y0 = np.zeros(2*N**2)
    y0[0:N**2] = np.multiply(22,np.multiply(y,np.power(1-y,3/2)))
    y0[N**2:2*N**2] = np.multiply(27,np.multiply(x,np.power(1-x,3/2)))

    return Problem(name="brusselator"+str(N),
                   rhs=FortBRUSS2Df,
                   rhs_reversed=brusselator_rhs_reversed,
                   jacobian=BRUSS2Dgradnonsparse,  # for scipy
                   solout=None,
                   output_times=[0,1.5,11.5],
                   y0=y0,
                   atol=None,
                   size=N)

def nbody_rhs(t,y):
    return fnbod.fnbod(t,y)

def nbody_rhs_reversed(t,y):
    return fnbod.fnbod(y,t)


brusselator = brusselator_setup()

kdv = kdv_setup()

burgers = burgers_setup()

nbody = Problem(name="nbody",
                rhs=nbody_rhs,
                rhs_reversed=nbody_rhs_reversed,
                jacobian=None,
                solout=None,
                output_times=[0, 0.08],
                y0=fnbod.init_fnbod(2400),
                atol=None,
                size=None)

vdpol = Problem(name="vdpol",
                rhs=vdpol_rhs,
                rhs_reversed=vdpol_rhs_reversed,
                jacobian=vdpol_jac,
                solout=None,
                output_times=np.arange(0,13,1.),
                y0=np.array([2.,0.]),
                atol=None,
                size=None)
