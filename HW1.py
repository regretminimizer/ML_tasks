import json
import numpy as np
import matplotlib.pyplot as plt
with open("./config.json", "rt") as f:
    conf = json.load(fp=f)

print(conf)
g = 9.81
np.random.seed(conf["seed"])  

n_samples = conf["n_samples"]

v_normal = np.random.normal(
    loc=conf["v"]["normal"]["mean"],
    scale=conf["v"]["normal"]["std"],
    size=n_samples
    )
print(v_normal)
plt.figure(1)
plt.hist(v_normal, bins=50, label="V, m/s")
plt.savefig("V_n.png")

alpha_normal = np.random.normal(
    loc=conf["alpha"]["normal"]["mean"],
    scale=conf["alpha"]["normal"]["std"],
    size=n_samples
    )
print(alpha_normal)
plt.figure(2)
plt.hist(alpha_normal, bins=50, label="Alpha, grad")
plt.savefig("Alpha_n.png")

L_nn = (v_normal**2)*np.sin(2*np.deg2rad(alpha_normal))/g
print(L_nn)
plt.figure(3)
plt.hist(L_nn, bins=50, label="L, m")
plt.savefig("L_nn.png")

v_uniform = np.random.uniform(
    low=conf["v"]["uniform"]["min"],
    high=conf["v"]["uniform"]["max"],
    size=n_samples
    )
print(v_uniform)
plt.figure(4)
plt.hist(v_uniform, bins=50, label="V, m/s")
plt.savefig("V_u.png")

alpha_uniform = np.random.uniform(
    low=conf["alpha"]["uniform"]["min"],
    high=conf["alpha"]["uniform"]["max"],
    size=n_samples
    )
print(alpha_uniform)
plt.figure(5)
plt.hist(alpha_uniform, bins=50, label="Alpha, grad")
plt.savefig("Alpha_u.png")

L_uu = (v_uniform**2)*np.sin(2*np.deg2rad(alpha_uniform))/g
plt.figure(6)
plt.hist(L_uu, bins=50, label="L, m")
plt.savefig("L_uu.png")

L_nu = (v_normal**2)*np.sin(2*np.deg2rad(alpha_uniform))/g
plt.figure(7)
plt.hist(L_nu, bins=50, label="L, m")
plt.savefig("L_nu.png")

L_un = (v_uniform**2)*np.sin(2*np.deg2rad(alpha_normal))/g
print(L_un)
plt.figure(8)
plt.hist(L_un, bins=50, label="L, m")
plt.savefig("L_un.png")