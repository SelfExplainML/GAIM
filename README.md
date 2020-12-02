# StatsGAIM
A Python wrapper of R:Stats::PPR for Generalized Additive Index Modeling

## Installation

- Python 3.7 or above
- matplotlib>=3.1.1
- numpy>=1.17.2
- pandas>=0.25.1
- rpy2>=3.3.5
- scikit-learn>=0.23.2
- csaps==1.0.2

```
pip install git+https://github.com/SelfExplainML/StatsGAIM.git
```

## Usage

### Regression Case

**Data Simulation**

```python
import numpy as np
from statsgaim.stats_ppr import PPRClassifier

# A quadratic SIM model
np.random.seed(2020)
n = int(1e4)
x = np.random.normal(0, 1, size=(n, 6))
beta = np.array([3, -2.5, 2, -1.5, 1.5, -1.0])/5
z = np.dot(x.reshape(-1,6),beta)
f = z**2
noise = np.random.randn(n)
y = f + noise
```

**Model Fitting**
```python
clf = PPRRegressor(nterms=1,optlevel=2)
clf.fit(x,y)
```

**Visualization**
```python
clf.visualize()
```
![regsim](https://github.com/SelfExplainML/GAIM/main/examples/reg_sim.png)

### Classification Case


**Data Simulation**

```python
import numpy as np
from statsgaim.stats_ppr import PPRClassifier

# A quadratic SIM model
np.random.seed(2020)
n = int(1e4)
x = np.random.normal(0, 0.3, size=(n, 6))
beta = np.array([3, -2.5, 2, -1.5, 1.5, -1.0])/5
z = np.dot(x.reshape(-1,6),beta)
f = z**2
noise = np.random.randn(n)

y = 1 / (1 + np.exp(-f)) + 0.05 * np.random.randn(n)
y = y - np.mean(y)
y[y <= 0] = 0
y[y > 0] = 1
```

**Model Fitting**
```python
clf = PPRClassifier(nterms=1,optlevel=2)
clf.fit(x,y)
```

**Visualization**
```python
clf.visualize()
```
![regsim](https://github.com/SelfExplainML/GAIM/main/examples/clf_sim.png)
