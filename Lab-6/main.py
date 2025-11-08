import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Step 1. Generate training data
x = np.linspace(-3, 3, 50)
y = np.linspace(-3, 3, 50)
X, Y = np.meshgrid(x, y)
Z = np.cos(np.sin(Y)) * np.sin(X)

inputs = torch.tensor(np.vstack((X.flatten(), Y.flatten())).T, dtype=torch.float32)
targets = torch.tensor(Z.flatten().reshape(-1, 1), dtype=torch.float32)

# Plot target surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_title("Target function: cos(sin(y))*sin(x)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()

# Step 2. Gaussian membership function
class GaussianMF(nn.Module):
    def __init__(self, mean, sigma):
        super().__init__()
        self.mean = nn.Parameter(torch.tensor(mean, dtype=torch.float32))
        self.sigma = nn.Parameter(torch.tensor(sigma, dtype=torch.float32))

    def forward(self, x):
        return torch.exp(-0.5 * ((x - self.mean) / self.sigma) ** 2)

# Visualize membership functions
mfs = [GaussianMF(-2, 1), GaussianMF(0, 1), GaussianMF(2, 1)]
xx = torch.linspace(-3, 3, 200)
for mf in mfs:
    plt.plot(xx, mf(xx).detach(), label=f"μ(mean={mf.mean.item():.1f})")
plt.title("Gaussian membership functions")
plt.xlabel("x")
plt.ylabel("μ(x)")
plt.legend()
plt.grid(True)
plt.show()

# Step 2.5. ANFIS architecture diagram
fig, ax = plt.subplots(figsize=(11, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')
ax.set_title("ANFIS Architecture", fontsize=14, fontweight='bold')

# Layer titles
layer_titles = ["Inputs", "Membership Functions", "Rule Layer", "Normalization", "Output"]
for i, title in enumerate(layer_titles):
    ax.text(1.0 + 2*i, 7.6, title, ha='center', fontsize=10, fontweight='bold')

# Layer 1: Inputs
inputs_xy = [(1, 6), (1, 4)]
for label, (x, y) in zip(["X", "Y"], inputs_xy):
    ax.text(x, y, label, ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle="round,pad=0.4", fc='lightblue', ec='blue'))

# Layer 2: Membership functions (3 per input)
mf_x = [(3, 6.6), (3, 6.0), (3, 5.4)]
mf_y = [(3, 4.6), (3, 4.0), (3, 3.4)]
for label, (x, y) in zip(["μ1(x)", "μ2(x)", "μ3(x)"], mf_x):
    ax.text(x, y, label, ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc='lavender', ec='purple'))
for label, (x, y) in zip(["μ1(y)", "μ2(y)", "μ3(y)"], mf_y):
    ax.text(x, y, label, ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc='lavender', ec='purple'))

# Layer 3: Rule nodes (R1–R9, grid of 3x3)
rule_nodes = []
y_positions = [6.5, 5.5, 4.5]
x_rule = 5
for i, y1 in enumerate(y_positions):
    for j, y2 in enumerate(y_positions[::-1]):
        rule_nodes.append((x_rule, (y1 + y2) / 2))
for idx, (x, y) in enumerate(rule_nodes):
    ax.text(x, y, f"R{idx+1}", ha='center', va='center', fontsize=8,
            bbox=dict(boxstyle="circle,pad=0.3", fc='lightyellow', ec='orange'))

# Layer 4: Normalization
ax.text(7, 5, "Σw normalization", ha='center', va='center', fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", fc='lightgreen', ec='green'))

# Layer 5: Output
ax.text(9, 5, "f(x,y)", ha='center', va='center', fontsize=11,
        bbox=dict(boxstyle="round,pad=0.4", fc='lightcoral', ec='red'))

def connect(ax, start, end, style='-'):
    ax.plot([start[0], end[0]], [start[1], end[1]], style, color='gray', lw=1)

for sx, sy in inputs_xy:
    for ex, ey in (mf_x if sy == 6 else mf_y):
        connect(ax, (sx+0.4, sy), (ex-0.4, ey))

for i, (ex, ey) in enumerate(rule_nodes):
    ix = i // 3  # μx index
    iy = i % 3   # μy index
    connect(ax, (3.4, mf_x[ix][1]), (ex-0.5, ey))
    connect(ax, (3.4, mf_y[iy][1]), (ex-0.5, ey))

for (sx, sy) in rule_nodes:
    connect(ax, (sx+0.5, sy), (7-0.5, 5))

connect(ax, (7.4, 5), (8.6, 5), style='-')

plt.tight_layout()
plt.show()

# Step 3. ANFIS model definition
class ANFIS(nn.Module):
    def __init__(self):
        super().__init__()
        self.mf_x = nn.ModuleList([GaussianMF(m, 1.0) for m in [-2, 0, 2]])
        self.mf_y = nn.ModuleList([GaussianMF(m, 1.0) for m in [-2, 0, 2]])
        self.p = nn.Parameter(torch.randn(9, 1))
        self.q = nn.Parameter(torch.randn(9, 1))
        self.r = nn.Parameter(torch.randn(9, 1))

    def forward(self, x):
        x1, x2 = x[:, 0].unsqueeze(1), x[:, 1].unsqueeze(1)
        mu_x = torch.stack([mf(x1) for mf in self.mf_x], dim=2)
        mu_y = torch.stack([mf(x2) for mf in self.mf_y], dim=2)
        rules = [mu_x[:, :, i] * mu_y[:, :, j] for i in range(3) for j in range(3)]
        w = torch.cat(rules, dim=1)
        w_norm = w / (torch.sum(w, dim=1, keepdim=True) + 1e-6)
        f = self.p.T * x1 + self.q.T * x2 + self.r.T
        return torch.sum(w_norm * f, dim=1, keepdim=True)

# Step 4. Train model
model = ANFIS()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

EPOCHS = 200
losses = []

for epoch in range(EPOCHS):
    optimizer.zero_grad()
    output = model(inputs)
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d} | Loss = {loss.item():.6f}")

# Plot training loss (Screenshot 4)
plt.plot(losses)
plt.title("Training error (MSE)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# Step 5. Visualize results
with torch.no_grad():
    prediction = model(inputs).numpy().reshape(X.shape)

fig = plt.figure(figsize=(12,5))
ax1 = fig.add_subplot(1,2,1, projection='3d')
ax2 = fig.add_subplot(1,2,2, projection='3d')

ax1.plot_surface(X, Y, Z, cmap='viridis')
ax1.set_title("Target function: cos(sin(y))*sin(x)")

ax2.plot_surface(X, Y, prediction, cmap='plasma')
ax2.set_title("ANFIS model approximation")
plt.show()

# Step 6. Evaluate model
mse = np.mean((Z - prediction) ** 2)
print(f"\nMean Squared Error (MSE): {mse:.6f}")
