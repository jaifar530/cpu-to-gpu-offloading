"""Classical machine learning with scikit-learn: scaling, PCA, ridge and logistic regression, k-means."""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

x, y = make_classification(n_samples=300000, n_features=120, n_informative=40, random_state=0)
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=30, svd_solver="full").fit(x)
z = pca.transform(x)

ridge = Ridge(alpha=1.0).fit(x, y)
logit = LogisticRegression(max_iter=200).fit(z, y)
km = KMeans(n_clusters=16, n_init=2, random_state=0).fit(z)

print(f"PCA variance kept {pca.explained_variance_ratio_.sum():.3f}, ridge R2 {ridge.score(x, y):.3f}, "
      f"logistic accuracy {logit.score(z, y):.3f}, k-means inertia {km.inertia_:.3e}")
