from scipy import integrate
import numpy as np
import time
from parex import parex
import matplotlib.pyplot as plt
from problems import vdpol

solvers = ['semi-implicit euler', 'scipy']
tols = [1.e-3, 1.e-5, 1.e-7]
tols = [1.e-11]

for problem in [vdpol]:
    for tol in tols:
        for solver in solvers:
            start = time.time()
            if solver is 'scipy':
                y, diagnostics = integrate.odeint(problem.rhs, problem.y0, problem.output_times, Dfun=problem.jacobian, atol=tol, rtol=tol, mxstep=100000000, full_output=True)
                fe_seq = np.sum(diagnostics["nfe"])
                mean_order = None
            else:
                y, diagnostics = parex.solve(problem.rhs, problem.output_times,
                                             problem.y0, solver=solver, atol=tol,
                                             rtol=tol, diagnostics=True,
                                             jac_fun=problem.jacobian)
                mean_order = diagnostics["k_avg"]
                fe_seq = diagnostics["fe_seq"]

            end = time.time()
            wall_time = start-end

            fe_total = np.sum(diagnostics["nfe"])
            nsteps = np.sum(diagnostics["nst"])
            je_total = np.sum(diagnostics["nje"])

            plt.plot(problem.output_times,y)
            plt.title(str(tol)+' '+solver)
            plt.show()

            y = y.reshape(-1)

