import matplotlib.pyplot as plt

ep_latency_list = {"TF1": [1, 2, 3], "TF2": [1, 2000, 300]}
plt.figure(figsize=(10, 6))
for tc, val in ep_latency_list.items():    
    plt.plot(val, label=tc)

plt.savefig("draft.png")
