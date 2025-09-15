# Ads that Stick: Near-Optimal Ad Optimization through Psychological Behavior Models

import math
import random

def loss_eval(t, delta):
  t_loss = 0
  for i in range(len(t)-1):
    for j in range(i+1, len(t)):
      t_loss += pow(delta, t[j]-t[i])
  return t_loss

def uniform_strat(T, n):
  l = []
  for i in range(n):
    l.append(T*i/(n-1))
  return l

def random_strat(n, T):
    samples = [random.uniform(0, T) for _ in range(n-2)]
    samples.append(0)
    samples.append(T)
    samples.sort()
    return samples

def edge_strat(n, T):
  l = []
  for i in range(math.floor(n/2)):
    l.append(0)
  if (n%2 != 0):
    l.append(T/2)
  for i in range(math.floor(n/2)):
    l.append(T)
  return l

def check_eqn(a, delta, n, T_a):
  n -= 1
  return 1/pow(a, 2)* pow((a*T_a), (n-2*a+2))/pow((1+a*T_a),(n-2*a))

def binary_search(n_rounds, T, delta, n, a):
  low = pow(delta, T)
  high = 1
  curr = 1
  mid = 0
  while (curr <= n_rounds):
    mid = (low+high)/2
    v = check_eqn(a, delta, n, mid)
    # print(mid, check_eqn(a, delta, n, mid), pow(delta, T))
    if v == pow(delta, T):
      return mid
    elif v < pow(delta, T):
      low = mid
      curr+=1
    else:
      high = mid
      curr+=1
  # return math.log(mid)/math.log(delta)
  # print("MID", mid)
  return mid

def check_feasibility(delta, n, T):
  a = 1
  while a<= n/2:
    if (check_eqn(a, delta, n, pow(delta, T)) <= pow(delta,T)) and (check_eqn(a, delta, n, 1) >= pow(delta,T)):
      return a
    a += 1
  return a

def opt_T_1(n_rounds, delta, n, T):
  if check_feasibility(delta, n, T) >= n/2:
    return (n/2, 0)
  else:
    return (check_feasibility(delta, n, T), binary_search(n_rounds, T, delta, n, check_feasibility(delta, n, T)))

def output_t_list(T_1, n, a, delta, T):
  l = []
  for i in range(a):
    l.append(1)
  for i in range(1, n-2*a+1):
    val = (1/a)*pow(a*T_1, i)/pow(a*T_1+1, i-1)
    # print(val, a)
    l.append(val)
  for i in range(a):
    l.append(pow(delta, T))
  for i in range(len(l)):
    # print((l[i]))
    l[i] = math.log(l[i])/math.log(delta)
  return l

def solver(n, T, delta, n_rounds):
  return output_t_list(opt_T_1(n_rounds, delta, n, T)[1], n, int(opt_T_1(n_rounds, delta, n, T)[0]), delta, T)

"""Can use the function solver to get the optimal times"""

# solver(10, 100, 0.53, 60)
# print(opt_T_1(60, 0.53, 10, 100))
# print(opt_T_1(50, 0.53, 10, 100))
print(binary_search(60, 100, 0.53, 10, 1))

"""Can use the function loss_eval to find the loss of strategies"""

import numpy as np
import matplotlib.pyplot as plt

# Assume your functions are already defined:
# solver(n, T, delta, n_rounds)
# uniform_strat(T, n)
# random_strat(n, T)
# edge_strat(n, T)
# loss_eval(t, delta)

n = 6       # number of arms, say
T = 60        # total time horizon
n_rounds = 100  # for solver
deltas = np.linspace(0.9, 0.9999, 1000)  # delta values from 0.01 to 1

# Precompute the strategies (if independent of delta)
uniform_strategy = uniform_strat(T, n)
random_strategy = random_strat(n, T)
edge_strategy = edge_strat(n, T)

# Now compute log(loss_eval) for each delta
solver_losses = []
uniform_losses = []
random_losses = []
edge_losses = []

for delta in deltas:
    # solver needs to be recomputed for each delta
    solver_strategy = solver(n, T, delta, n_rounds)
    solver_losses.append((loss_eval(solver_strategy, delta)))

    # Others do not depend on delta
    uniform_losses.append((loss_eval(uniform_strategy, delta)))
    t = 0
    for i in range(1000):
      random_strategy = random_strat(n, T)
      t+=(loss_eval(random_strategy, delta))
    random_losses.append(t/1000)
    edge_losses.append((loss_eval(edge_strategy, delta)))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(deltas, solver_losses, label="Our Strategy", linewidth=2)
plt.plot(deltas, uniform_losses, label="Uniform Strategy", linewidth=2)
plt.plot(deltas, random_losses, label="Random Strategy", linewidth=2)
plt.plot(deltas, edge_losses, label="Corner Strategy", linewidth=2)

plt.xlabel('\u03B4', fontsize=20)
plt.ylabel('Loss', fontsize=20)
plt.legend(fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
# plt.title('Loss vs \u03B4 for different strategies')
plt.legend(fontsize=20)
plt.grid(True)
plt.savefig("loss.png", dpi=300, bbox_inches='tight')
# plt.show()
plt.savefig("myImagePDF.pdf", format="pdf", bbox_inches="tight")

import numpy as np
import matplotlib.pyplot as plt

# Assume your functions are already defined:
# solver(n, T, delta, n_rounds)
# uniform_strat(T, n)
# random_strat(n, T)
# edge_strat(n, T)
# loss_eval(t, delta)

n = 15     # number of arms, say
T = 100        # total time horizon
n_rounds = 100  # for solver
deltas = np.linspace(0.9, 0.9999, 1000)  # delta values from 0.01 to 1

# Precompute the strategies (if independent of delta)
uniform_strategy = uniform_strat(T, n)
random_strategy = random_strat(n, T)
edge_strategy = edge_strat(n, T)

# Now compute log(loss_eval) for each delta
solver_losses = []
uniform_losses = []
random_losses = []
edge_losses = []

for delta in deltas:
    # solver needs to be recomputed for each delta
    solver_strategy = solver(n, T, delta, n_rounds)
    solver_losses.append(np.log(loss_eval(solver_strategy, delta)))

    # Others do not depend on delta
    uniform_losses.append(np.log(loss_eval(uniform_strategy, delta)))
    t = 0
    for i in range(1000):
      random_strategy = random_strat(n, T)
      t+=(np.log(loss_eval(random_strategy, delta)))
    random_losses.append(t/1000)
    edge_losses.append(np.log(loss_eval(edge_strategy, delta)))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(deltas, solver_losses, label="Our Strategy", linewidth=2)
plt.plot(deltas, uniform_losses, label="Uniform Strategy", linewidth=2)
plt.plot(deltas, random_losses, label="Random Strategy", linewidth=2)
plt.plot(deltas, edge_losses, label="Corner Strategy", linewidth=2)

plt.xlabel('\u03B4')
plt.ylabel('log(Loss)')
plt.title('log(Loss) vs \u03B4 for different strategies')
plt.legend()
plt.grid(True)
# plt.savefig("logloss.png", dpi=300, bbox_inches='tight')
plt.savefig("logloss.pdf", format="pdf", bbox_inches="tight")
plt.show()

solver(10, 100, 0.53, 60)

import numpy as np
import matplotlib.pyplot as plt

# Parameters
n = 6        # Number of elements in solver_strategy
T = 20
n_rounds = 100
deltas = np.linspace(0.01, 0.999, 1000)  # range of delta values

# Storage for strategy values
solver_values = [[] for _ in range(n)]  # one list per coordinate

# Compute solver strategy for each delta
for delta in deltas:
    solver_strategy = solver(n, T, delta, n_rounds)
    # print(delta, solver_strategy)
    for i in range(n):
        solver_values[i].append(solver_strategy[i])

# Now plot
plt.figure(figsize=(12, 7))
for i in range(n):
    plt.plot(deltas, solver_values[i], label=f"Ad No. {i+1}",linewidth=2)

plt.xlabel('\u03B4', fontsize=25)
plt.ylabel('Time', fontsize=25)
# plt.title('Advertisment Strategy as \u03B4 changes')
plt.legend(fontsize=25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.legend(fontsize=25)
plt.grid(True)
plt.savefig("AdvsDelta.pdf", format="pdf", bbox_inches="tight")
# plt.savefig("AdvsDelta.png", dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()

"""Find the value of B - gamma sigma (T)
delta = 0.9 - 0.99

"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
T = 120
n_rounds = 100
n_values = range(5, 70)  # n from 5 to 50
delta_values = [0.6, 0.7, 0.8, 0.9, 0.999]  # Chosen delta values

# Dictionary to store losses per delta
log_losses_per_delta = {delta: [] for delta in delta_values}

# Compute losses
for delta in delta_values:
    for n in n_values:
        strategy = solver(n, T, delta, n_rounds)
        loss = loss_eval(strategy, delta)
        log_losses_per_delta[delta].append(np.log(loss))

# Plotting
plt.figure(figsize=(10, 6))
for delta in delta_values:
    plt.plot(n_values, log_losses_per_delta[delta], label=f"delta = {delta}", linewidth=2)

plt.xlabel('Number of Ads (n)', fontsize=20)
plt.ylabel('log(Loss)', fontsize=20)
plt.legend(fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
# plt.title('log(Loss) vs Number of Arms for Different Delta Values')
plt.legend(fontsize=20)
plt.grid(True)
plt.savefig("logLoss.pdf", format="pdf", bbox_inches="tight")
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Parameters
T = 120
n_rounds = 100
n_values = range(5, 51)  # n from 5 to 50
delta_values = [0.6, 0.7, 0.8, 0.9, 0.99]  # Chosen delta values

# Dictionary to store losses per delta
log_losses_per_delta = {delta: [] for delta in delta_values}

# Compute losses
for delta in delta_values:
    for n in n_values:
        strategy = solver(n, T, delta, n_rounds)
        loss = loss_eval(strategy, delta)
        log_losses_per_delta[delta].append((loss))

# Plotting
plt.figure(figsize=(10, 6))
for delta in delta_values:
    plt.plot(n_values, log_losses_per_delta[delta], label=f"delta = {delta}", linewidth=2)

plt.xlabel('Number of Ads (n)', fontsize=20)
plt.ylabel('Loss',fontsize=20)
plt.legend(fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig("Loss.pdf", format="pdf", bbox_inches="tight")
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Parameters
T = 120
n_rounds = 100
n_values = range(2, 50)  # number of arms
delta = 0.9  # fixed delta
reward_values = [2, 2.5, 3, 4]  # different reward coefficients

# sigmoid = lambda x: (1 / (1 + np.exp(-(x/5))))
sigmoid = lambda x: 1-np.exp(-x/10)

# Dictionary to store loss vs n for each reward value
log_losses_per_reward = {reward: [] for reward in reward_values}

# Compute losses
for reward in reward_values:
    for n in n_values:
        strategy = solver(n, T, delta, n_rounds)
        loss = loss_eval(strategy, delta)
        # gain = sigmoid(n)
        gain = 0
        for i in range(n):
          gain += sigmoid(i)
        log_losses_per_reward[reward].append(reward*gain - loss)

# Plotting
plt.figure(figsize=(10, 6))
for reward in reward_values:
    plt.plot(n_values, log_losses_per_reward[reward], label=f"strength = {reward}", linewidth=2)

plt.xlabel('Number of Ads (n)', fontsize=20)
plt.ylabel('Reward', fontsize=20)
plt.ylim(0, 60)
# plt.title(f'Reward vs Number of Ads for Different Reward Values (delta = {delta})', fontsize=18)
plt.legend(fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig("Reward.pdf", format="pdf", bbox_inches="tight")
plt.show()

