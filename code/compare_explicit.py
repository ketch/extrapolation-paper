%matplotlib inline
from scipy import integrate
import numpy as np
import time
from parex import parex
import matplotlib.pyplot as plt
from problems import nbody, kdv, burgers

solvers = ['explicit midpoint','scipy DOPRI5', 'scipy DOP853']
tols = [1.e-3, 1.e-5, 1.e-7, 1.e-9, 1.e-11, 1.e-13]
problems = [nbody, kdv, burgers]

results = {}

for problem in problems:
    results[problem.name] = {}
    reference_file = "./reference_data/" + problem.name + ".txt"
    y_ref = np.loadtxt(reference_file)
    for solver in solvers:
        results[problem.name][solver] = {}
        for tol in tols:
            print(problem.name, solver, tol)
            results[problem.name][solver][tol] = {}
            start = time.time()
            if 'scipy' in solver:
                solver_name = solver.split()[-1].lower()
                if problem in [kdv, burgers]:
                    r = integrate.complex_ode(problem.rhs_reversed).set_integrator(solver_name, atol=tol, 
                                                                          rtol=tol, verbosity=10, 
                                                                          nsteps=1e6)
                else:
                    r = integrate.ode(problem.rhs_reversed).set_integrator(solver_name, atol=tol, rtol=tol, 
                                                                  verbosity=10, nsteps=1e7)
                t0 = problem.output_times[0]
                tf = problem.output_times[-1]
                r.set_initial_value(problem.y0, t0)
                r.integrate(r.t+(tf-t0))
                assert r.t == tf, "Integration failed. Try increasing the maximum allowed steps."
                y = r.y


                results[problem.name][solver][tol]['fe_seq'] = None
                results[problem.name][solver][tol]['mean_order'] = None
            else:
                y, diagnostics = parex.solve(problem.rhs, problem.output_times,
                                             problem.y0, solver=solver, atol=tol,
                                             rtol=tol, diagnostics=True,
                                             jac_fun=problem.jacobian)

                results[problem.name][solver][tol]['mean_order'] = diagnostics["k_avg"]
                results[problem.name][solver][tol]['fe_seq'] = diagnostics["fe_seq"]
                y = y[-1,:].squeeze()

            end = time.time()
            if problem.solout:
                y = problem.solout(y)
            results[problem.name][solver][tol]['wall_time'] = end-start

            results[problem.name][solver][tol]['fe_total'] = np.sum(diagnostics["nfe"])
            results[problem.name][solver][tol]['nsteps'] = np.sum(diagnostics["nst"])
            results[problem.name][solver][tol]['je_total'] = np.sum(diagnostics["nje"])

            #plt.plot(problem.output_times,y)
            #plt.title(str(tol)+' '+solver)

            y = y.reshape(-1)
            y_ref = y_ref.reshape(-1)

            results[problem.name][solver][tol]['relative_error'] = np.linalg.norm(y-y_ref)/np.linalg.norm(y_ref)


    for solver in solvers:
        wall_time = [results[problem.name][solver][tol]['wall_time'] for tol in tols]
        errors = [results[problem.name][solver][tol]['relative_error'] for tol in tols]
        plt.loglog(errors, wall_time, '-s')
    plt.legend(solvers);
    plt.savefig(problem.name+'.pdf')
