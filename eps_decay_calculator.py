eps = 1
max_steps = 10000
eps_decay = 0.99975

for a in range(max_steps + 1):
    eps = eps*eps_decay
    if a % (max_steps/10) == 0 and a != 0:
        print(eps)