"""
From book: Solving Ordinary Differential Equations II,
IV.10 Numerical Experiment, Twelve Test Problems

Each ODE problem is defined with: problemName, right hand side function
(derivative function), jacobian matrix of RHS function, initial time (float),
initial value (np.array), times at which output is wanted, atolfact absolute
tolerance factor-> set to 1. as default (multiplies relative tolerance factor
to make absolute tolerance more stringent), atol absolute tolerance -> set to
None as default (required absolute tolerance for all relative tolerances
wanted).
"""
from scipy import integrate
import scipy
import numpy as np
from collections import namedtuple
import time
from parex import parex
import matplotlib.pyplot as plt
from compare_test import kdv_func, kdv_init
import fnbruss

TestProblemDefinition = namedtuple("TestProblemDefinition",
                                   ["problemName","rhs",
                                    "jacobian","initialTime", "y0",
                                    "denseOutput", "atolfact", "atol"])

# Vanderpol problem

# Observation: RHS function can't be nested in vdpol_problem():
# http://stackoverflow.com/questions/4677012/python-cant-pickle-type-x-attribute-lookup-failed

def vdpol_rhs(y,t):
    epsilon=1e-6
    y2=1/epsilon*(((1-y[0]**2)*y[1])-y[0])
    return np.array([y[1],y2])

def vdpol_jac(y,t):
    epsilon=1e-6
    J12=1/epsilon*(-2*y[0]*y[1]-1)
    J22=1/epsilon*(1-y[0]**2)
    return np.array([[0,1],[J12,J22]])

def vdpol_problem():
    return TestProblemDefinition("vdpol", vdpol_rhs, vdpol_jac, 0, np.array([2.,0.]),np.arange(0,13,1.),1.,None)

# ROBER problem

def rober_rhs(y,t):
    y1 = -0.04*y[0]+1e4*y[1]*y[2]
    y2 = 0.04*y[0]-1e4*y[1]*y[2]-3e7*y[1]**2
    y3 = 3e7*y[1]**2
    return np.array([y1,y2,y3])

def rober_jac(y,t):
    J11=-0.04
    J12=1e4*y[2]
    J13=1e4*y[1]
    J21=0.04
    J22=-1e4*y[2]-3e7*y[1]*2
    J23=-1e4*y[1]
    J31=0
    J32=3e7*y[1]*2
    J33=0
    return np.array([[J11,J12,J13],[J21,J22,J23],[J31,J32,J33]])

def rober_problem():
    base=13*[10.]
    base[0]=0
    denseOutput = np.power(base,range(0,13))
    return TestProblemDefinition("rober", rober_rhs, rober_jac, 0, np.array([1.,0,0]), denseOutput,1.e-6,None)

# OREGO problem

def orego_rhs(y,t):
    y1 = 77.27*(y[1]+y[0]*(1-8.375e-6*y[0]-y[1]))
    y2 = 1/77.27*(y[2]-(1+y[0])*y[1])
    y3 = 0.161*(y[0]-y[2])
    return np.array([y1,y2,y3])

def orego_jac(y,t):
    J11=77.27*(1-8.375e-6*y[0]*2-y[1])
    J12=77.27*(1-y[0])
    J13=0
    J21=1/77.27*(-y[1])
    J22=1/77.27*(-(1+y[0]))
    J23=1/77.27
    J31=0.161
    J32=0
    J33=-0.161
    return np.array([[J11,J12,J13],[J21,J22,J23],[J31,J32,J33]])

def orego_problem():
    denseOutput = np.arange(0,390,30.)
    return TestProblemDefinition("orego", orego_rhs, orego_jac, 0, np.array([1.,2.,3.]), denseOutput, 1.e-6,None)

# HIRES problem

def hires_rhs(y,t):
    y1 = -1.71*y[0]+0.43*y[1]+8.32*y[2]+0.0007
    y2 = 1.71*y[0]-8.75*y[1]
    y3 = -10.03*y[2]+0.43*y[3]+0.035*y[4]
    y4 = 8.32*y[1]+1.71*y[2]-1.12*y[3]
    y5 = -1.745*y[4]+0.43*y[5]+0.43*y[6]
    y6 = -280*y[5]*y[7]+0.69*y[3]+1.71*y[4]-0.43*y[5]+0.69*y[6]
    y7 = 280*y[5]*y[7]-1.81*y[6]
    y8 = -y7
    return np.array([y1,y2,y3,y4,y5,y6,y7,y8])

def hires_jac(y,t):
    return np.array([[-1.71,0.43,8.32,0,0,0,0,0],
                     [1.71,-8.75,0,0,0,0,0,0],
                     [0,0,-10.03,0.43,0.035,0,0,0],
                     [0,8.32,1.71,-1.12,0,0,0,0],
                     [0,0,0,0,-1.745,0.43,0.035,0],
                     [0,0,0,0.69,1.71,-0.43-280*y[7],0.69,-280*y[5]],
                     [0,0,0,0,0,280*y[7],-1.81,280*y[5]],
                     [0,0,0,0,0,-280*y[7],1.81,-280*y[5]]])

def hires_problem():
    denseOutput = np.array([0,321.8122,421.8122])
    return TestProblemDefinition("hires", hires_rhs, hires_jac, 0,
                                 np.array([1.,0,0,0,0,0,0,0.0057]),
                                 denseOutput, 1.e-4, None)

# E5 problem

def E5_jac(y,t):
    A=7.86e-10
    B=1.1e7
    C=1.13e3
    M=1e6
    return np.array([[-A-B*y[2],0,-B*y[0],0],
                     [A,-M*C*y[2],-M*C*y[1],0],
                     [A-B*y[2],-M*C*y[2],-M*C*y[1]-B*y[0],C],
                     [B*y[2],0,B*y[0],-C]])

def E5_rhs(y,t):
    A=7.86e-10
    B=1.1e7
    C=1.13e3
    M=1e6
    y1 = -A*y[0]-B*y[0]*y[2]
    y2 = A*y[0]-M*C*y[1]*y[2]
    y4 = B*y[0]*y[2]-C*y[3]
    y3 = y2-y4
    return np.array([y1,y2,y3,y4])

def E5_plot(ys, times):
    y1=[yt[0] for yt in ys]
    y2=[yt[1] for yt in ys]
    y3=[yt[2] for yt in ys]
    y4=[yt[3] for yt in ys]
    n=len(times)
    plt.plot(np.log10(times[1:n]),np.log10(y1[1:n]))
    plt.plot(np.log10(times[1:n]),np.log10(y2[1:n]))
    plt.plot(np.log10(times[1:n]),np.log10(y3[1:n]))
    plt.plot(np.log10(times[1:n]),np.log10(y4[1:n]))
    plt.show()

def E5_problem():
    base=8*[10.]
    base[0]=0
    exp = np.arange(-1,14,2)
    # OBS: the first exponent doesn't matter (base =0)
    exp[0]=1
    denseOutput = np.power(base,exp)
    return TestProblemDefinition("E5", E5_rhs, E5_jac, 0, np.array([1.76e-3,0,0,0]), denseOutput,1.,1.7e-24)


# BRUSS-2D problem (Brusselator)

A=0
N=10
step=0
x=0
y=0
alpha=0.1

def initializeBRUSS2DValues(Nval):
    global A,Aperm,N,step,x,y
    N=Nval
    A=five_pt_laplacian_sparse_periodic(N,0,1)
    step=1/(N-1)
    x=np.multiply(step,range(N)*N)
    y=np.multiply(step,np.repeat(range(N),N))


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

#Here we will use U to obtain the coordinates (x,y)
def BRUSS2DInhom(t):
   Nsq=N**2
   fout = np.zeros(Nsq)
   if t<1.1:
       return fout
   fout = np.add(np.power(x-0.3,2),np.power(y-0.6,2))<=0.01
   fout = 5*fout
   return fout

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

def FortBRUSS2Df(y,t):
    '''
    Compiled Fortran brusselator 2D RHS function (faster than python)
    '''
    aux=fnbruss.fnbruss(y,t,N)
    return aux

def BRUSS2DInitialValue(N):
    y0 = np.zeros(2*N**2)
    y0[0:N**2] = np.multiply(22,np.multiply(y,np.power(1-y,3/2)))
    y0[N**2:2*N**2] = np.multiply(27,np.multiply(x,np.power(1-x,3/2)))

    return y0

def BRUSS2DPlot(ys, times):
    X, Y = np.meshgrid(np.multiply(step,range(N)),np.multiply(step,range(N)))
    for i in range(len(ys)):
        z=ys[i]
        U=np.reshape(z[range(N**2)], (N,N))
        V=np.reshape(z[range(N**2,2*N**2)], (N,N))
        fig = plt.figure()
        fig.suptitle("time : " + str(times[i]))
        ax = fig.gca(projection='3d')
        ax.plot_wireframe(X, Y, U)
        ax.plot_wireframe(X, Y, V, color='r')
    plt.show()

def BRUSS2DProblem():
    initializeBRUSS2DValues(N)
    tf=11.5
    output_times = [0,1.5,tf]
    y0 = BRUSS2DInitialValue(N)
#     output_times = [0,0.5,1.,1.3,1.4,5.6,6.,6.1,6.2,10]
    return TestProblemDefinition("BRUSS2D_"+str(N), FortBRUSS2Df, BRUSS2Dgrad, 0, y0, output_times,1.,None)

# KDV problem

def KDVProblem():
    t0, tf = 0, 0.0003
    output_times = [t0,tf]
    y0 = kdv_init(t0)
    return TestProblemDefinition("kdv2", kdv_func, None, t0, y0, output_times,1.,None)


def get_all_tests():
    '''
    Get all the problem tests that you want to use to test
    (uncomment those that want to be used)
    '''
    tests = []
    tests.append(vdpol_problem())
#     tests.append(rober_problem())
#     tests.append(orego_problem())
#     tests.append(hires_problem())
#     tests.append(KDVProblem())
#     tests.append(E5_problem())
#     tests.append(BRUSS2DProblem())

    return tests

def storeTestsExactSolutions():
    '''
    Stores an exact solution (asking for a very stringent tolerance to a numerical method)
    '''
    for test in get_all_tests():
        output_times = test.output_times
        startTime = time.time()
        exactSolution, infodict = integrate.odeint(test.rhs,test.y0, output_times, Dfun=None, atol=1e-27, rtol=1e-13, mxstep=100000000, full_output=True)
        print("Store solution for " + test.problemName + "; solution: " + str(exactSolution))
        print("Time : " + str(time.time()-startTime) + " numb steps: " + str(infodict["nst"]))
        np.savetxt(getReferenceFile(test.problemName), exactSolution[1:len(exactSolution)])
        # Use a plot function to visualize results: like BRUSS2DPlot()


def getReferenceFile(problemName):
    '''
    Get the reference file name for a given problemName (keeps stored solutions tidy)
    '''
    return "reference_" + problemName + ".txt"


def comparisonTest():
    '''
    Mainly: loops over all the tolerances in tol to obtain a comparison plot of the behavior of all the
    algorithms in solverFunctions (in relation to their names, labels).

    It also iterates onto all possible configuration parameters to get different algorithm/parameters
    combinations to plot (to compare how different algorithms behave with different parameters configurations)
    '''
    dense=True
    tol = [1.e-12,1.e-10,1.e-8,1.e-7,1.e-5,1.e-3]
    resultDict={}
    solvers = ['semi-implicit euler', 'scipy']
    labels = ["Semi Eul", "Scipy int"]

    use_jacobian = False

    for test in get_all_tests():
        testProblemResult = []
        y_ref = np.loadtxt(getReferenceFile(test.problemName))
        output_times = test.output_times
        if(not dense):
            y_ref=y_ref[-1]
            output_times=[output_times[0], output_times[-1]]
        print(output_times)
        print(test.problemName)
        for i in range(len(tol)):
            print(tol[i])
            all_labels=[]
            if(test.atol is None):
                atol=test.atolfact*tol[i]
            else:
                atol = test.atol
            rtol=tol[i]
            print("rtol: " + str(rtol) + " atol:" + str(atol))

            k=0
            for solver in solvers:
                if solver is 'scipy':
                    if(use_jacobian):
                        jacobian = test.jacobian
                    else:
                        jacobian = None
                    startTime = time.time()
                    ys, infodict = integrate.odeint(test.rhs, test.y0, output_times, Dfun=jacobian, atol=atol, rtol=rtol, mxstep=100000000, full_output=True)
                    finalTime = time.time()

                    mean_order = 0
                    fe_seq = np.sum(infodict["nfe"])
                    mused = infodict["mused"]
                    print "1: adams (nonstiff), 2: bdf (stiff) -->" + str(mused)
                else:
                    if(use_jacobian):
                        jacobian = test.jacobian
                    else:
                        jacobian = None
                    startTime = time.time()
                    ys, infodict = parex.solve(test.rhs, output_times, test.y0,
                                               solver=solver, atol=atol,
                                               rtol=rtol, diagnostics=True,
                                               jac_fun=jacobian)
                    finalTime = time.time()

                    mean_order = infodict["k_avg"]
                    fe_seq = infodict["fe_seq"]

                fe_tot = np.sum(infodict["nfe"])
                nsteps = np.sum(infodict["nst"])
                je_tot = np.sum(infodict["nje"])
                ys=ys[1:len(ys)]

                relative_error = np.linalg.norm(ys-y_ref)/np.linalg.norm(y_ref)

                print(relative_error)

                testProblemResult.append([finalTime-startTime, relative_error, fe_tot, nsteps, mean_order, fe_seq, je_tot])
                print("Done: " + labels[k] + " time: " + str(finalTime-startTime) +  " rel error: " + str(relative_error) + " func eval: " + str(fe_tot) + " jac eval: " + str(je_tot) + " func eval seq: " + str(fe_seq)+ " num steps: " + str(nsteps) + " mean_order: " + str(mean_order))
                print("\n")

                all_labels.append(labels[k])
                k+=1

        resultDict[test.problemName] = testProblemResult

    return resultDict, all_labels


def plotResults(resultDict, labels):
    '''
    Plot all the results in resultDict. ResultDicts should contain for each test problem a
    list with all the results for that problem (each problem is plotted in separated windows).

    Each problem entry should contain a list with all the different algorithm/parameters combinations
    to be plotted. The labels parameter should contain the title to be shown as legend for each of this
    algorithm/parameters combination.

    Each algorithm/parameters combination is a list with all the indicator y-axis values (number
    of function evaluations, number of steps, mean order, time...) to be plotted (included
    the x-axis value: relative error).
    '''
    j=1
    for test in get_all_tests():
        testName = test.problemName
        resultTest = resultDict[testName]
        fig= plt.figure(j)
        fig.suptitle(testName)
        for i in range(0,len(resultTest)):
            res=resultTest[i]
            yfunceval=[resRow[2] for resRow in res]
            ytime=[resRow[0] for resRow in res]
            yfuncevalseq=[resRow[5] for resRow in res]
            yjaceval=[resRow[6] for resRow in res]
            x=[resRow[1] for resRow in res]
            
            plt.subplot(411)
            plt.loglog(x,ytime,"s-")
            plt.subplot(412)
            plt.loglog(x,yfunceval,"s-")
            plt.subplot(413)
            plt.loglog(x,yfuncevalseq,"s-")
            plt.subplot(414)
            plt.loglog(x,yjaceval,"s-")

        
        plt.subplot(411)
        plt.legend()
        plt.ylabel("time(log)")
        plt.subplot(412)
        plt.ylabel("fun.ev.")
        plt.subplot(413)
        plt.ylabel("fun.ev.seq")
        plt.subplot(414)
        plt.ylabel("jac.ev.")
        j+=1
#         fig.subplots_adjust(left=0.05, top = 0.96, bottom=0.06, right=0.99)
    plt.show()



if __name__ == "__main__":
    #If exact solution hasn't been yet calculated uncomment first line
#     storeTestsExactSolutions()
    resultDict, labels = comparisonTest()
    plotResults(resultDict, labels)
    print "done"
    
    
