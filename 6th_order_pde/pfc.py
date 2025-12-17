import numpy as np


def MK(n):
    x = np.linspace(0, 1, n)
    h = x[1] - x[0]

    M = np.zeros((n, n), dtype=float)
    K = np.zeros((n, n), dtype=float)

    M[0, 0:2] = np.array([2, 1])
    M[-1, :] = M[0, ::-1]

    K[0, 0:2] = np.array([1, -1])
    K[-1, :] = K[0, ::-1]

    for i in range(1, n - 1):
        M[i, i - 1 : i + 2] = np.array([1, 4, 1])
        K[i, i - 1 : i + 2] = np.array([-1, 2, -1])

    return (h / 6) * M, (1 / h) * K


pi = np.pi
steps = [10001,20001,40001,80001]
time_steps = [0.0001,0.0001/2,0.0001/4,0.0001/8]
T = 0.01


for n,dt in zip(steps,time_steps):
    # phi_solution = []
    # sigma_solution = []
    # mu_solution = []



    # time loop

    mu_e = 0
    phi_e = 0
    sigma_e = 0

    # numeric solutions
    mu_solve = 0
    phi_solve = sigma_solve = 0

    M, K = MK(n)
    x = np.linspace(0, 1, n)
    t = 0
    phi_0 = np.exp(-t) * np.cos(pi * x)
    r = -0.025
    # assign grid
    grid = x

    while t <= T:
        #  define exact solutions
        phi_e = np.exp(-t) * np.cos(pi * x)
        mu_e = phi_e**3 + (r + 1 - (2 * pi**2) + pi**4) * phi_e
        sigma_e = -1 * pi**2 * phi_e
        grad_phi = pi * np.exp(-t) * np.sin(pi * x)
        # define rhs
        mu_f = M @ (-2 * (phi_0**3))
        z = np.zeros(n)
        f = phi_e * (
            3 * pi * phi_e**2
            + (r + 1) * pi**2
            - 2 * pi**4
            + pi**6
            - 1
            - 6 * (grad_phi) ** 2
        )
        F = M @ f + (1 / dt) * (M @ phi_0)

        RHS = np.hstack((mu_f, F, z))

        # Assembling 3*3 matrix, A

        A_11 = M
        A_12 = -(r + 1) * M - 3 * (phi_0) ** 2 * M + 2 * K
        A_13 = K
        A_21 = K
        A_22 = (1 / dt) * M
        A_23 = M * 0
        A_31 = M * 0
        A_32 = K
        A_33 = M

        A = np.block([[A_11, A_12, A_13], [A_21, A_22, A_23], [A_31, A_32, A_33]])

        direct_solve = np.linalg.solve(A, RHS)

        mu_solve = direct_solve[:n]
        phi_solve = direct_solve[n : 2 * n]
        sigma_solve = direct_solve[2 * n :]

        phi_0 = phi_solve
        e_phi = np.linalg.norm(phi_solve - phi_e, ord=np.inf)
        t += dt
    e_phi01 = np.linalg.norm(phi_solve - phi_e, ord=np.inf)
    # with open("errors.dat","a") as f:
    #     f.write(f"{e_phi01}\n")
    # mu_solution.append((mu_solve, mu_e))
    # phi_solution.append((phi_solve, phi_e))
    # sigma_solution.append((sigma_solve, sigma_e))

    # np.savez(
    #     "solutions_{n}.npz",
    #     mu_solution=mu_solution,
    #     phi_solution=phi_solution,
    #     sigma_solution=sigma_solution,
    # )
