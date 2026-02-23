import numpy as np
import matplotlib.pyplot as plt

def posterior_given_positive(prior, sens, spec):
    fpr = 1 - spec
    return (sens * prior) / (sens * prior + fpr * (1 - prior))

prior0 = 0.01
sens0 = 0.95
spec0 = 0.90

post0 = posterior_given_positive(prior0, sens0, spec0)
print("P(Disease | +) =", post0)

#1. Posterior vs prior (fixed sensitivity & specificity)
priors = np.linspace(0.0001, 0.20, 600)
post_vs_prior = posterior_given_positive(priors, sens0, spec0)

plt.figure()
plt.plot(priors, post_vs_prior)
plt.xlabel("Prior probability P(Disease)")
plt.ylabel("Posterior P(Disease | +)")
plt.title("Posterior vs Prior (fixed sensitivity=0.95, specificity=0.90)")
plt.grid(True)

plt.scatter([prior0], [post0])
plt.annotate(f"({prior0:.2f}, {post0:.3f})",
             xy=(prior0, post0),
             xytext=(prior0 + 0.02, post0 + 0.05),
             arrowprops=dict(arrowstyle="->"))
plt.show()

#2. Posterior vs sensitivity (fixed prior & specificity)
sensitivities = np.linspace(0.50, 0.999, 600)
post_vs_sens = posterior_given_positive(prior0, sensitivities, spec0)

plt.figure()
plt.plot(sensitivities, post_vs_sens)
plt.xlabel("Sensitivity P(+ | Disease)")
plt.ylabel("Posterior P(Disease | +)")
plt.title("Posterior vs Sensitivity (fixed prior=0.01, specificity=0.90)")
plt.grid(True)

plt.scatter([sens0], [post0])
plt.annotate(f"({sens0:.2f}, {post0:.3f})",
             xy=(sens0, post0),
             xytext=(sens0 - 0.25, post0 + 0.05),
             arrowprops=dict(arrowstyle="->"))
plt.show()

#3. Posterior vs specificity (fixed prior & sensitivity)
specificities = np.linspace(0.50, 0.999, 600)
post_vs_spec = posterior_given_positive(prior0, sens0, specificities)

plt.figure()
plt.plot(specificities, post_vs_spec)
plt.xlabel("Specificity P(- | No disease)")
plt.ylabel("Posterior P(Disease | +)")
plt.title("Posterior vs Specificity (fixed prior=0.01, sensitivity=0.95)")
plt.grid(True)

plt.scatter([spec0], [post0])
plt.annotate(f"({spec0:.2f}, {post0:.3f})",
             xy=(spec0, post0),
             xytext=(spec0 - 0.35, post0 + 0.05),
             arrowprops=dict(arrowstyle="->"))
plt.show()