"""Machine-learning style NumPy script in single precision:
trains a small fully connected network (784-512-256-10) with mini-batch SGD on synthetic data."""
import numpy as np

rng = np.random.default_rng(0)
samples, features, classes, batch = 60000, 784, 10, 256

x_all = rng.standard_normal((samples, features), dtype=np.float32)
true_w = rng.standard_normal((features, classes), dtype=np.float32)
y_all = np.argmax(x_all @ true_w, axis=1)

sizes = [features, 512, 256, classes]
weights = [rng.standard_normal((i, o), dtype=np.float32) * np.float32(np.sqrt(2.0 / i))
           for i, o in zip(sizes[:-1], sizes[1:])]
biases = [np.zeros(o, dtype=np.float32) for o in sizes[1:]]
lr = np.float32(0.05)

for epoch in range(3):
    order = rng.permutation(samples)
    correct = 0
    for start in range(0, samples - batch + 1, batch):
        idx = order[start:start + batch]
        acts = [x_all[idx]]
        for w, b in zip(weights[:-1], biases[:-1]):
            acts.append(np.maximum(acts[-1] @ w + b, 0))
        logits = acts[-1] @ weights[-1] + biases[-1]
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        correct += int((probs.argmax(axis=1) == y_all[idx]).sum())

        grad = probs
        grad[np.arange(batch), y_all[idx]] -= 1
        grad /= batch
        for layer in range(len(weights) - 1, -1, -1):
            grad_w = acts[layer].T @ grad
            grad_b = grad.sum(axis=0)
            if layer:
                grad = (grad @ weights[layer].T) * (acts[layer] > 0)
            weights[layer] -= lr * grad_w
            biases[layer] -= lr * grad_b
    print(f"epoch {epoch}: training accuracy {correct / (samples // batch * batch):.3f}")
