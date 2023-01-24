import time

import numpy as np
import csv

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from REM import REM
'''
#Ecoli
Data = np.genfromtxt('Data/ecoli.csv', delimiter = ",")
X = Data[:, :-1]
y = Data[:, -1]

n_samples, n_features = X.shape

bndwk = int(np.floor(np.min((30, np.log(n_samples)))))
Cluster = REM.REM(covariance_type = "full", criteria = "all", bandwidth = bndwk, tol = 1e-4)
t1 = time.perf_counter()
Cluster.fit(X)
t2 = time.perf_counter()
1
8
0.15
aic_idx = np.argmin(Cluster.aics_)
bic_idx = np.argmin(Cluster.bics_)
icl_idx = np.argmin(Cluster.icls_)

aic_y = Cluster.mixtures[aic_idx].predict(X)
bic_y = Cluster.mixtures[bic_idx].predict(X)
icl_y = Cluster.mixtures[icl_idx].predict(X)

aic_nc = len(np.unique(aic_y))
bic_nc = len(np.unique(bic_y))
icl_nc = len(np.unique(icl_y))


aic_ari = adjusted_rand_score(aic_y.astype(int), y)
aic_nmi = normalized_mutual_info_score(aic_y.astype(int), y)
bic_ari = adjusted_rand_score(bic_y.astype(int), y)
bic_nmi = normalized_mutual_info_score(bic_y.astype(int), y)
icl_ari = adjusted_rand_score(icl_y.astype(int), y)
icl_nmi = normalized_mutual_info_score(icl_y.astype(int), y)

with open('REM_real.csv', 'a') as f:
  w = csv.writer(f)
  w.writerow(['ecoli', 'aic', bndwk, aic_nc, aic_ari, aic_nmi, (Cluster.t4 - Cluster.t1) - (Cluster.t3 - Cluster.t2)])
  w.writerow(['ecoli', 'bic', bndwk, bic_nc, bic_ari, bic_nmi, (Cluster.t4 - Cluster.t1) - (Cluster.t3 - Cluster.t2)])
  w.writerow(['ecoli', 'icl', bndwk, icl_nc, icl_ari, icl_nmi, (Cluster.t4 - Cluster.t1) - (Cluster.t3 - Cluster.t2)])



#Iris
Data = np.genfromtxt('Data/iris.csv', delimiter = ",")
X = Data[:, :-1]
y = Data[:, -1]

n_samples, n_features = X.shape

bndwk = int(np.floor(np.min((30, np.sqrt(n_samples)))))
Cluster = REM.REM(covariance_type = "full", criteria = "all", bandwidth = bndwk, tol = 1e-5)

t1 = time.perf_counter()
Cluster.fit(X)
t2 = time.perf_counter()
1
2
1

aic_idx = np.argmin(Cluster.aics_)
bic_idx = np.argmin(Cluster.bics_)
icl_idx = np.argmin(Cluster.icls_)

aic_y = Cluster.mixtures[aic_idx].predict(X)
bic_y = Cluster.mixtures[bic_idx].predict(X)
icl_y = Cluster.mixtures[icl_idx].predict(X)

aic_nc = len(np.unique(aic_y))
bic_nc = len(np.unique(bic_y))
icl_nc = len(np.unique(icl_y))


aic_ari = adjusted_rand_score(aic_y.astype(int), y)
aic_nmi = normalized_mutual_info_score(aic_y.astype(int), y)
bic_ari = adjusted_rand_score(bic_y.astype(int), y)
bic_nmi = normalized_mutual_info_score(bic_y.astype(int), y)
icl_ari = adjusted_rand_score(icl_y.astype(int), y)
icl_nmi = normalized_mutual_info_score(icl_y.astype(int), y)

with open('REM_real.csv', 'a') as f:
  w = csv.writer(f)
  w.writerow(['iris', 'aic', aic_nc, aic_ari, aic_nmi,(Cluster.t4 - Cluster.t1) - (Cluster.t3 - Cluster.t2)])
  w.writerow(['iris', 'bic', bic_nc, bic_ari, bic_nmi,(Cluster.t4 - Cluster.t1) - (Cluster.t3 - Cluster.t2)])
  w.writerow(['iris', 'icl', icl_nc, icl_ari, icl_nmi,(Cluster.t4 - Cluster.t1) - (Cluster.t3 - Cluster.t2)])



#Wine
Data = np.genfromtxt('Data/wine.csv', delimiter = ",")
X = Data[:, :-1]
y = Data[:, -1]

n_samples, n_features = X.shape

bndwk = int(np.floor(np.min((30, np.sqrt(n_samples)))))
Cluster = REM.REM(covariance_type = "full", criteria = "all", bandwidth = bndwk, tol = 1e-5)
 

t1 = time.perf_counter()
Cluster.fit(X)
t2 = time.perf_counter()
2
7

aic_idx = np.argmin(Cluster.aics_)
bic_idx = np.argmin(Cluster.bics_)
icl_idx = np.argmin(Cluster.icls_)

aic_y = Cluster.mixtures[aic_idx].predict(X)
bic_y = Cluster.mixtures[bic_idx].predict(X)
icl_y = Cluster.mixtures[icl_idx].predict(X)

aic_nc = len(np.unique(aic_y))
bic_nc = len(np.unique(bic_y))
icl_nc = len(np.unique(icl_y))


aic_ari = adjusted_rand_score(aic_y.astype(int), y)
aic_nmi = normalized_mutual_info_score(aic_y.astype(int), y)
bic_ari = adjusted_rand_score(bic_y.astype(int), y)
bic_nmi = normalized_mutual_info_score(bic_y.astype(int), y)
icl_ari = adjusted_rand_score(icl_y.astype(int), y)
icl_nmi = normalized_mutual_info_score(icl_y.astype(int), y)

with open('REM_real.csv', 'a') as f:
  w = csv.writer(f)
  w.writerow(['wine', 'aic', bndwk, aic_nc, aic_ari, aic_nmi,(Cluster.t4 - Cluster.t1) - (Cluster.t3 - Cluster.t2)])
  w.writerow(['wine', 'bic', bndwk, bic_nc, bic_ari, bic_nmi,(Cluster.t4 - Cluster.t1) - (Cluster.t3 - Cluster.t2)])
  w.writerow(['wine', 'icl', bndwk, icl_nc, icl_ari, icl_nmi,(Cluster.t4 - Cluster.t1) - (Cluster.t3 - Cluster.t2)])



#Seeds
Data = np.genfromtxt('Data/seeds.csv', delimiter = ",")
X = Data[:, :-1]
y = Data[:, -1]

n_samples, n_features = X.shape

bndwk = int(np.floor(np.min((30, np.sqrt(n_samples)))))
Cluster = REM.REM(covariance_type = "full", criteria = "all", bandwidth = bndwk, tol = 1e-5)

t1 = time.perf_counter()
Cluster.fit(X)
t2 = time.perf_counter()
1
1.2
2

aic_idx = np.argmin(Cluster.aics_)
bic_idx = np.argmin(Cluster.bics_)
icl_idx = np.argmin(Cluster.icls_)

aic_y = Cluster.mixtures[aic_idx].predict(X)
bic_y = Cluster.mixtures[bic_idx].predict(X)
icl_y = Cluster.mixtures[icl_idx].predict(X)

aic_nc = len(np.unique(aic_y))
bic_nc = len(np.unique(bic_y))
icl_nc = len(np.unique(icl_y))


aic_ari = adjusted_rand_score(aic_y.astype(int), y)
aic_nmi = normalized_mutual_info_score(aic_y.astype(int), y)
bic_ari = adjusted_rand_score(bic_y.astype(int), y)
bic_nmi = normalized_mutual_info_score(bic_y.astype(int), y)
icl_ari = adjusted_rand_score(icl_y.astype(int), y)
icl_nmi = normalized_mutual_info_score(icl_y.astype(int), y)

with open('REM_real.csv', 'a') as f:
  w = csv.writer(f)
  w.writerow(['seeds', 'aic', aic_nc, aic_ari, aic_nmi,(Cluster.t4 - Cluster.t1) - (Cluster.t3 - Cluster.t2)])
  w.writerow(['seeds', 'bic', bic_nc, bic_ari, bic_nmi,(Cluster.t4 - Cluster.t1) - (Cluster.t3 - Cluster.t2)])
  w.writerow(['seeds', 'icl', icl_nc, icl_ari, icl_nmi,(Cluster.t4 - Cluster.t1) - (Cluster.t3 - Cluster.t2)])




#G2128
Data = np.genfromtxt('Data/G2.csv', delimiter = ",")
X = Data[:, :-1]
y = Data[:, -1]

n_samples, n_features = X.shape

bndwk = int(np.floor(np.min((30, np.sqrt(n_samples)))))
Cluster = REM.REM(covariance_type = "full", criteria = "all", bandwidth = bndwk, tol = 1e-5)

t1 = time.perf_counter()
Cluster.fit(X)
t2 = time.perf_counter()
1
0.0019
1200

aic_idx = np.argmin(Cluster.aics_)
bic_idx = np.argmin(Cluster.bics_)
icl_idx = np.argmin(Cluster.icls_)

aic_y = Cluster.mixtures[aic_idx].predict(X)
bic_y = Cluster.mixtures[bic_idx].predict(X)
icl_y = Cluster.mixtures[icl_idx].predict(X)

aic_nc = len(np.unique(aic_y))
bic_nc = len(np.unique(bic_y))
icl_nc = len(np.unique(icl_y))


aic_ari = adjusted_rand_score(aic_y.astype(int), y)
aic_nmi = normalized_mutual_info_score(aic_y.astype(int), y)
bic_ari = adjusted_rand_score(bic_y.astype(int), y)
bic_nmi = normalized_mutual_info_score(bic_y.astype(int), y)
icl_ari = adjusted_rand_score(icl_y.astype(int), y)
icl_nmi = normalized_mutual_info_score(icl_y.astype(int), y)

with open('REM_real.csv', 'a') as f:
  w = csv.writer(f)
  w.writerow(['G2', 'aic', aic_nc, aic_ari, aic_nmi,(Cluster.t4 - Cluster.t1) - (Cluster.t3 - Cluster.t2)])
  w.writerow(['G2', 'bic', bic_nc, bic_ari, bic_nmi,(Cluster.t4 - Cluster.t1) - (Cluster.t3 - Cluster.t2)])
  w.writerow(['G2', 'icl', icl_nc, icl_ari, icl_nmi,(Cluster.t4 - Cluster.t1) - (Cluster.t3 - Cluster.t2)])


#Satellite
Data = np.genfromtxt('Data/satellite.csv', delimiter = ",")
X = Data[:, :-1]
y = Data[:, -1]

n_samples, n_features = X.shape

bndwk = int(np.floor(np.min((30, np.log(n_samples)))))
Cluster = REM.REM(covariance_type = "full", criteria = "all", bandwidth = bndwk, tol = 1e-5)

t1 = time.perf_counter()
Cluster.fit(X)
t2 = time.perf_counter()
1
0.045
40
aic_idx = np.argmin(Cluster.aics_)
bic_idx = np.argmin(Cluster.bics_)
icl_idx = np.argmin(Cluster.icls_)

aic_y = Cluster.mixtures[aic_idx].predict(X)
bic_y = Cluster.mixtures[bic_idx].predict(X)
icl_y = Cluster.mixtures[icl_idx].predict(X)

aic_nc = len(np.unique(aic_y))
bic_nc = len(np.unique(bic_y))
icl_nc = len(np.unique(icl_y))


aic_ari = adjusted_rand_score(aic_y.astype(int), y)
aic_nmi = normalized_mutual_info_score(aic_y.astype(int), y)
bic_ari = adjusted_rand_score(bic_y.astype(int), y)
bic_nmi = normalized_mutual_info_score(bic_y.astype(int), y)
icl_ari = adjusted_rand_score(icl_y.astype(int), y)
icl_nmi = normalized_mutual_info_score(icl_y.astype(int), y)

with open('REM_real.csv', 'a') as f:
  w = csv.writer(f)
  w.writerow(['satellite', 'aic', aic_nc, aic_ari, aic_nmi,(Cluster.t4 - Cluster.t1) - (Cluster.t3 - Cluster.t2)])
  w.writerow(['satellite', 'bic', bic_nc, bic_ari, bic_nmi,(Cluster.t4 - Cluster.t1) - (Cluster.t3 - Cluster.t2)])
  w.writerow(['satellite', 'icl', icl_nc, icl_ari, icl_nmi,(Cluster.t4 - Cluster.t1) - (Cluster.t3 - Cluster.t2)])


Data = np.genfromtxt('Data/Raisin_Dataset.csv', delimiter = ",")
X = Data[1:, :-1]
y = Data[1:, -1]

n_samples, n_features = X.shape

bndwk = int(np.floor(np.min((30, np.log(n_samples)))))
Cluster = REM.REM(covariance_type = "full", criteria = "all", bandwidth = bndwk, tol = 1e-5)

Cluster.fit(X)

aic_idx = np.argmin(Cluster.aics_)
bic_idx = np.argmin(Cluster.bics_)
icl_idx = np.argmin(Cluster.icls_)

aic_y = Cluster.mixtures[aic_idx].predict(X)
bic_y = Cluster.mixtures[bic_idx].predict(X)
icl_y = Cluster.mixtures[icl_idx].predict(X)


aic_nc = len(np.unique(aic_y))
bic_nc = len(np.unique(bic_y))
icl_nc = len(np.unique(icl_y))

aic_ari = adjusted_rand_score(aic_y.astype(int), y)
aic_nmi = normalized_mutual_info_score(aic_y.astype(int), y)
bic_ari = adjusted_rand_score(bic_y.astype(int), y)
bic_nmi = normalized_mutual_info_score(bic_y.astype(int), y)
icl_ari = adjusted_rand_score(icl_y.astype(int), y)
icl_nmi = normalized_mutual_info_score(icl_y.astype(int), y)

with open('REM_real.csv', 'a') as f:
  w = csv.writer(f)
  w.writerow(['raisin', 'aic', aic_nc, aic_ari, aic_nmi,(Cluster.t4 - Cluster.t1) - (Cluster.t3 - Cluster.t2)])
  w.writerow(['raisin', 'bic', bic_nc, bic_ari, bic_nmi,(Cluster.t4 - Cluster.t1) - (Cluster.t3 - Cluster.t2)])
  w.writerow(['raisin', 'icl', icl_nc, icl_ari, icl_nmi,(Cluster.t4 - Cluster.t1) - (Cluster.t3 - Cluster.t2)])


# scale
Data = np.genfromtxt('Data/balance-scale.data', delimiter = ",")
X = Data[1:, 1:]
y = Data[1:, 0]

n_samples, n_features = X.shape

bndwk = int(np.floor(np.min((30, np.log(n_samples)))))
Cluster = REM.REM(covariance_type = "full", criteria = "all", bandwidth = bndwk, tol = 1e-5)

Cluster.fit(X)

aic_idx = np.argmin(Cluster.aics_)
bic_idx = np.argmin(Cluster.bics_)
icl_idx = np.argmin(Cluster.icls_)

aic_y = Cluster.mixtures[aic_idx].predict(X)
bic_y = Cluster.mixtures[bic_idx].predict(X)
icl_y = Cluster.mixtures[icl_idx].predict(X)


aic_nc = len(np.unique(aic_y))
bic_nc = len(np.unique(bic_y))
icl_nc = len(np.unique(icl_y))

aic_ari = adjusted_rand_score(aic_y.astype(int), y)
aic_nmi = normalized_mutual_info_score(aic_y.astype(int), y)
bic_ari = adjusted_rand_score(bic_y.astype(int), y)
bic_nmi = normalized_mutual_info_score(bic_y.astype(int), y)
icl_ari = adjusted_rand_score(icl_y.astype(int), y)
icl_nmi = normalized_mutual_info_score(icl_y.astype(int), y)

with open('REM_real.csv', 'a') as f:
  w = csv.writer(f)
  w.writerow(['raisin', 'aic', aic_nc, aic_ari, aic_nmi,(Cluster.t4 - Cluster.t1) - (Cluster.t3 - Cluster.t2)])
  w.writerow(['raisin', 'bic', bic_nc, bic_ari, bic_nmi,(Cluster.t4 - Cluster.t1) - (Cluster.t3 - Cluster.t2)])
  w.writerow(['raisin', 'icl', icl_nc, icl_ari, icl_nmi,(Cluster.t4 - Cluster.t1) - (Cluster.t3 - Cluster.t2)])


# credit card
Data = np.genfromtxt('Data/credit-card.csv', delimiter=",")
X = Data[1:, :-1]
y = Data[1:, -1]

n_samples, n_features = X.shape

bndwk = int(np.floor(np.min((30, np.log(n_samples)))))
Cluster = REM.REM(covariance_type="full", criteria="all", bandwidth=bndwk, tol=1e-5)

Cluster.fit(X)

aic_idx = np.argmin(Cluster.aics_)
bic_idx = np.argmin(Cluster.bics_)
icl_idx = np.argmin(Cluster.icls_)

aic_y = Cluster.mixtures[aic_idx].predict(X)
bic_y = Cluster.mixtures[bic_idx].predict(X)
icl_y = Cluster.mixtures[icl_idx].predict(X)

aic_nc = len(np.unique(aic_y))
bic_nc = len(np.unique(bic_y))
icl_nc = len(np.unique(icl_y))

aic_ari = adjusted_rand_score(aic_y.astype(int), y)
aic_nmi = normalized_mutual_info_score(aic_y.astype(int), y)
bic_ari = adjusted_rand_score(bic_y.astype(int), y)
bic_nmi = normalized_mutual_info_score(bic_y.astype(int), y)
icl_ari = adjusted_rand_score(icl_y.astype(int), y)
icl_nmi = normalized_mutual_info_score(icl_y.astype(int), y)

with open('REM_real.csv', 'a') as f:
  w = csv.writer(f)
  w.writerow(['credit card', 'aic', aic_nc, aic_ari, aic_nmi, (Cluster.t4 - Cluster.t1) - (Cluster.t3 - Cluster.t2)])
  w.writerow(['credit card', 'bic', bic_nc, bic_ari, bic_nmi, (Cluster.t4 - Cluster.t1) - (Cluster.t3 - Cluster.t2)])
  w.writerow(['credit card', 'icl', icl_nc, icl_ari, icl_nmi, (Cluster.t4 - Cluster.t1) - (Cluster.t3 - Cluster.t2)])

# CTG
Data = np.genfromtxt('Data/CTG.csv', delimiter=",")
X = Data[1:, :-1]
y = Data[1:, -1]

n_samples, n_features = X.shape

bndwk = int(np.floor(np.min((30, np.log(n_samples)))))
Cluster = REM.REM(covariance_type="full", criteria="all", bandwidth=bndwk, tol=1e-5)

Cluster.fit(X)

aic_idx = np.argmin(Cluster.aics_)
bic_idx = np.argmin(Cluster.bics_)
icl_idx = np.argmin(Cluster.icls_)

aic_y = Cluster.mixtures[aic_idx].predict(X)
bic_y = Cluster.mixtures[bic_idx].predict(X)
icl_y = Cluster.mixtures[icl_idx].predict(X)

aic_nc = len(np.unique(aic_y))
bic_nc = len(np.unique(bic_y))
icl_nc = len(np.unique(icl_y))

aic_ari = adjusted_rand_score(aic_y.astype(int), y)
aic_nmi = normalized_mutual_info_score(aic_y.astype(int), y)
bic_ari = adjusted_rand_score(bic_y.astype(int), y)
bic_nmi = normalized_mutual_info_score(bic_y.astype(int), y)
icl_ari = adjusted_rand_score(icl_y.astype(int), y)
icl_nmi = normalized_mutual_info_score(icl_y.astype(int), y)

with open('REM_real.csv', 'a') as f:
  w = csv.writer(f)
  w.writerow(['CTG', 'aic', aic_nc, aic_ari, aic_nmi, (Cluster.t4 - Cluster.t1) - (Cluster.t3 - Cluster.t2)])
  w.writerow(['CTG', 'bic', bic_nc, bic_ari, bic_nmi, (Cluster.t4 - Cluster.t1) - (Cluster.t3 - Cluster.t2)])
  w.writerow(['CTG', 'icl', icl_nc, icl_ari, icl_nmi, (Cluster.t4 - Cluster.t1) - (Cluster.t3 - Cluster.t2)])


# HCV
Data = np.genfromtxt('Data/hcvdat0.csv', delimiter=",")
X = Data[1:, 2:]
y = Data[1:, 1]

n_samples, n_features = X.shape

bndwk = int(np.floor(np.min((30, np.log(n_samples)))))
Cluster = REM.REM(covariance_type="full", criteria="all", bandwidth=bndwk, tol=1e-5)

Cluster.fit(X)

aic_idx = np.argmin(Cluster.aics_)
bic_idx = np.argmin(Cluster.bics_)
icl_idx = np.argmin(Cluster.icls_)

aic_y = Cluster.mixtures[aic_idx].predict(X)
bic_y = Cluster.mixtures[bic_idx].predict(X)
icl_y = Cluster.mixtures[icl_idx].predict(X)

aic_nc = len(np.unique(aic_y))
bic_nc = len(np.unique(bic_y))
icl_nc = len(np.unique(icl_y))

aic_ari = adjusted_rand_score(aic_y.astype(int), y)
aic_nmi = normalized_mutual_info_score(aic_y.astype(int), y)
bic_ari = adjusted_rand_score(bic_y.astype(int), y)
bic_nmi = normalized_mutual_info_score(bic_y.astype(int), y)
icl_ari = adjusted_rand_score(icl_y.astype(int), y)
icl_nmi = normalized_mutual_info_score(icl_y.astype(int), y)

with open('REM_real.csv', 'a') as f:
  w = csv.writer(f)
  w.writerow(['HCV', 'aic', aic_nc, aic_ari, aic_nmi, (Cluster.t4 - Cluster.t1) - (Cluster.t3 - Cluster.t2)])
  w.writerow(['HCV', 'bic', bic_nc, bic_ari, bic_nmi, (Cluster.t4 - Cluster.t1) - (Cluster.t3 - Cluster.t2)])
  w.writerow(['HCV', 'icl', icl_nc, icl_ari, icl_nmi, (Cluster.t4 - Cluster.t1) - (Cluster.t3 - Cluster.t2)])


# Hills
Data = np.genfromtxt('Data/Hill_Valley_with_noise_Training.data', delimiter=",")
X = Data[1:, :-1]
y = Data[1:, -1]

n_samples, n_features = X.shape

bndwk = int(np.floor(np.min((30, np.log(n_samples)))))
Cluster = REM.REM(covariance_type="full", criteria="all", bandwidth=bndwk, tol=1e-5)

Cluster.fit(X)

aic_idx = np.argmin(Cluster.aics_)
bic_idx = np.argmin(Cluster.bics_)
icl_idx = np.argmin(Cluster.icls_)

aic_y = Cluster.mixtures[aic_idx].predict(X)
bic_y = Cluster.mixtures[bic_idx].predict(X)
icl_y = Cluster.mixtures[icl_idx].predict(X)

aic_nc = len(np.unique(aic_y))
bic_nc = len(np.unique(bic_y))
icl_nc = len(np.unique(icl_y))

aic_ari = adjusted_rand_score(aic_y.astype(int), y)
aic_nmi = normalized_mutual_info_score(aic_y.astype(int), y)
bic_ari = adjusted_rand_score(bic_y.astype(int), y)
bic_nmi = normalized_mutual_info_score(bic_y.astype(int), y)
icl_ari = adjusted_rand_score(icl_y.astype(int), y)
icl_nmi = normalized_mutual_info_score(icl_y.astype(int), y)

with open('REM_real.csv', 'a') as f:
  w = csv.writer(f)
  w.writerow(['Hills', 'aic', aic_nc, aic_ari, aic_nmi, (Cluster.t4 - Cluster.t1) - (Cluster.t3 - Cluster.t2)])
  w.writerow(['Hills', 'bic', bic_nc, bic_ari, bic_nmi, (Cluster.t4 - Cluster.t1) - (Cluster.t3 - Cluster.t2)])
  w.writerow(['Hills', 'icl', icl_nc, icl_ari, icl_nmi, (Cluster.t4 - Cluster.t1) - (Cluster.t3 - Cluster.t2)])


# segment
Data = np.genfromtxt('Data/segment.dat', delimiter=" ")
X = Data[1:, :-1]
y = Data[1:, -1]

n_samples, n_features = X.shape

bndwk = int(np.floor(np.min((30, np.log(n_samples)))))
Cluster = REM.REM(covariance_type="full", criteria="all", bandwidth=bndwk, tol=1e-5)

Cluster.fit(X)

aic_idx = np.argmin(Cluster.aics_)
bic_idx = np.argmin(Cluster.bics_)
icl_idx = np.argmin(Cluster.icls_)

aic_y = Cluster.mixtures[aic_idx].predict(X)
bic_y = Cluster.mixtures[bic_idx].predict(X)
icl_y = Cluster.mixtures[icl_idx].predict(X)

aic_nc = len(np.unique(aic_y))
bic_nc = len(np.unique(bic_y))
icl_nc = len(np.unique(icl_y))

aic_ari = adjusted_rand_score(aic_y.astype(int), y)
aic_nmi = normalized_mutual_info_score(aic_y.astype(int), y)
bic_ari = adjusted_rand_score(bic_y.astype(int), y)
bic_nmi = normalized_mutual_info_score(bic_y.astype(int), y)
icl_ari = adjusted_rand_score(icl_y.astype(int), y)
icl_nmi = normalized_mutual_info_score(icl_y.astype(int), y)

with open('REM_real.csv', 'a') as f:
  w = csv.writer(f)
  w.writerow(['Segment', 'aic', aic_nc, aic_ari, aic_nmi, (Cluster.t4 - Cluster.t1) - (Cluster.t3 - Cluster.t2)])
  w.writerow(['Segment', 'bic', bic_nc, bic_ari, bic_nmi, (Cluster.t4 - Cluster.t1) - (Cluster.t3 - Cluster.t2)])
  w.writerow(['Segment', 'icl', icl_nc, icl_ari, icl_nmi, (Cluster.t4 - Cluster.t1) - (Cluster.t3 - Cluster.t2)])
'''

# transfusion
Data = np.genfromtxt('Data/transfusion.data', delimiter=",")
X = Data[1:, :-1]
y = Data[1:, -1]

n_samples, n_features = X.shape

bndwk = int(np.floor(np.min((30, np.log(n_samples)))))
Cluster = REM.REM(covariance_type="full", criteria="all", bandwidth=bndwk, tol=1e-5)

Cluster.fit(X)

aic_idx = np.argmin(Cluster.aics_)
bic_idx = np.argmin(Cluster.bics_)
icl_idx = np.argmin(Cluster.icls_)

aic_y = Cluster.mixtures[aic_idx].predict(X)
bic_y = Cluster.mixtures[bic_idx].predict(X)
icl_y = Cluster.mixtures[icl_idx].predict(X)

aic_nc = len(np.unique(aic_y))
bic_nc = len(np.unique(bic_y))
icl_nc = len(np.unique(icl_y))

aic_ari = adjusted_rand_score(aic_y.astype(int), y)
aic_nmi = normalized_mutual_info_score(aic_y.astype(int), y)
bic_ari = adjusted_rand_score(bic_y.astype(int), y)
bic_nmi = normalized_mutual_info_score(bic_y.astype(int), y)
icl_ari = adjusted_rand_score(icl_y.astype(int), y)
icl_nmi = normalized_mutual_info_score(icl_y.astype(int), y)

with open('REM_real.csv', 'a') as f:
  w = csv.writer(f)
  w.writerow(['Transfusion', 'aic', aic_nc, aic_ari, aic_nmi, (Cluster.t4 - Cluster.t1) - (Cluster.t3 - Cluster.t2)])
  w.writerow(['Transfusion', 'bic', bic_nc, bic_ari, bic_nmi, (Cluster.t4 - Cluster.t1) - (Cluster.t3 - Cluster.t2)])
  w.writerow(['Transfusion', 'icl', icl_nc, icl_ari, icl_nmi, (Cluster.t4 - Cluster.t1) - (Cluster.t3 - Cluster.t2)])