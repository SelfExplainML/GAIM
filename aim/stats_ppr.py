import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib import pyplot as plt
from abc import ABCMeta, abstractmethod

from sklearn.utils.extmath import softmax
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array, check_X_y
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, is_classifier, is_regressor

import rpy2
from rpy2 import robjects as ro
from rpy2.robjects import Formula
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri, pandas2ri

import os

numpy2ri.activate()
pandas2ri.activate()

try:
    stats = importr("stats")
except:
    utils = importr('utils')
    utils.install_packages('stats', repos='http://cran.us.r-project.org')
    stats = importr("stats")
    

class BasePPR(BaseEstimator, metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, nterms=5, optlevel=2):

        self.nterms = nterms
        self.optlevel = optlevel

    
    @property
    def importance_ratios_(self):
        """return the estimator importance ratios (the higher, the more important the feature)

        Returns
        -------
        dict of selected estimators
            the importance ratio of each fitted base learner.
        """
        
        if is_regressor(self):
            reg = self.reg
        elif is_classifier(self):
            reg = self.reg.reg
        
        mu = int(reg[-3][4])
        component_importance = reg[12][:mu]

        importance_ratios_ = {}
        total_importance = component_importance.sum()
        importance_ratios_ = component_importance/total_importance
        return importance_ratios_


    @property
    def projection_indices_(self):
        """return the projection indices

        Returns
        -------
        ndarray of shape (n_features, n_estimators)
            the projection indices
        """
        
        if is_regressor(self):
            reg = self.reg
        elif is_classifier(self):
            reg = self.reg.reg
            
        projection_indices_ = reg[11]
        mu = int(reg[-3][4])
        
        return projection_indices_[:,:mu]
    
    
    def decision_function(self, x):

        """output f(x) for given samples
        
        Parameters
        ---------
        x : array-like of shape (n_samples, d)
            containing the input dataset
        Returns
        -------
        np.array of shape (n_samples,)
            containing f(x) 
        """
        check_is_fitted(self, "reg")
        x = x.copy()
        
        if is_regressor(self):
            pred = stats.predict(self.reg,pd.DataFrame(x,
                                    columns=['x'+str(i+1) for i in range(x.shape[1])]))
        elif is_classifier(self):
            pred = stats.predict(self.reg.reg,pd.DataFrame(x,
                                    columns=['x'+str(i+1) for i in range(x.shape[1])]))
        
        return pred.flatten()
    
    def visualize(self,cols_per_row=3,
                  folder="./results/", name="global_plot", save_png=False, save_eps=False):
        if is_regressor(self):
            reg = self.reg
        elif is_classifier(self):
            reg = self.reg.reg
        
        n = int(reg[-3][3])
        mu = int(reg[-3][4])
        m = int(reg[-3][0])
        p = reg[3]
        q = reg[4]
        jf = int(q+6+m*(p+q))
        jt = jf+ m*n
        f = reg[-3][jf:(jf + mu*n)].reshape(mu, n)
        t = reg[-3][jt:(jt + mu*n)].reshape(mu, n)

        subfig_idx = 0
        max_ids = mu
        
        fig = plt.figure(figsize=(8 * cols_per_row, 4.6 * int(np.ceil(max_ids / cols_per_row))))
        outer = gridspec.GridSpec(int(np.ceil(max_ids / cols_per_row)), cols_per_row, wspace=0.15, hspace=0.25)

        projection_indices_ = reg[11]
        if projection_indices_.shape[1] > 0:
            xlim_min = - max(np.abs(projection_indices_.min() - 0.1), np.abs(projection_indices_.max() + 0.1))
            xlim_max = max(np.abs(projection_indices_.min() - 0.1), np.abs(projection_indices_.max() + 0.1))

        for indice in range(max_ids):

            beta = projection_indices_[:, indice]
            density, bins = np.histogram(np.dot(self.x, beta), bins=10, density=True)

            inner = outer[subfig_idx].subgridspec(2, 2, wspace=0.2, height_ratios=[6, 1], width_ratios=[3, 1])
            ax1_main = fig.add_subplot(inner[0, 0])
            xgrid = t[indice]
            ygrid = f[indice]
            ax1_main.plot(np.sort(xgrid), ygrid[np.argsort(xgrid)], color="red")
            ax1_main.set_xticklabels([])
            ax1_main.set_title("SIM " + str(indice + 1) + 
                               " (IR: " + str(np.round(100 * self.importance_ratios_[indice], 2)) + "%)",
                         fontsize=16)
            fig.add_subplot(ax1_main)

            ax1_density = fig.add_subplot(inner[1, 0])  
            xint = ((np.array(bins[1:]) + np.array(bins[:-1])) / 2).reshape([-1, 1]).reshape([-1])
            ax1_density.bar(xint, density, width=xint[1] - xint[0])
            ax1_main.get_shared_x_axes().join(ax1_main, ax1_density)
            ax1_density.set_yticklabels([])
            fig.add_subplot(ax1_density)

            ax2 = fig.add_subplot(inner[:, 1])
            if len(beta) <= 10:
                rects = ax2.barh(np.arange(len(beta)), [b for b in beta.ravel()][::-1])
                ax2.set_yticks(np.arange(len(beta)))
                ax2.set_yticklabels(["X" + str(idx + 1) for idx in range(len(beta.ravel()))][::-1])
                ax2.set_xlim(xlim_min, xlim_max)
                ax2.set_ylim(-1, len(beta))
                ax2.axvline(0, linestyle="dotted", color="black")
            fig.add_subplot(ax2)
            subfig_idx += 1
            
        if max_ids > 0:
            if save_png:
                if not os.path.exists(folder):
                    os.makedirs(folder)
                save_path = folder + name
                fig.savefig("%s.png" % save_path, bbox_inches="tight", dpi=100)
            else:
                plt.show()
            if save_eps:
                if not os.path.exists(folder):
                    os.makedirs(folder)
                save_path = folder + name
                fig.savefig("%s.eps" % save_path, bbox_inches="tight", dpi=100)
            else:
                plt.show()
                
class PPRRegressor(BasePPR,RegressorMixin):

    """
    This is a Python wrapper of the 'ppr' function in the 'stats'. We use GCV to determine the smoothness of
    the spline by default. 
    
    Parameters
    ---------
    nterms : number of ridge terms.
    optlevel : optimization level, can be 0 or 1 or 2. Set to be 2 (highlevel/backfitting) by default.
    """
    
    def __init__(self, nterms=5, optlevel=2):

        super(PPRRegressor, self).__init__(nterms=nterms, optlevel=optlevel)
    
    def _validate_input(self, x, y):

        """method to validate data
        
        Parameters
        ---------
        x : array-like of shape (n_samples, d)
            containing the input dataset
        y : array-like of shape (n_samples,)
            containing the output dataset
        """

        x, y = check_X_y(x, y, accept_sparse=["csr", "csc", "coo"],
                         multi_output=True, y_numeric=True)
        return x, y.ravel()
    
    
    def fit(self, x, y, sample_weight=None):

        """fit the smoothing spline
        Parameters
        ---------
        x : array-like of shape (n_samples, n_features)
            containing the input dataset
        y : array-like of shape (n_samples,)
            containing target values
        sample_weight : array-like of shape (n_samples,), optional
            containing sample weights
        Returns
        -------
        object 
            self : Estimator instance.
        """

        x, y = self._validate_input(x, y)

        n_samples = x.shape[0]
        if sample_weight is None:
            sample_weight = np.ones(n_samples)
            
        d = x.shape[1]
        self.columns = ['x'+str(i+1) for i in range(d)] + ['y']
        formula = self.columns[-1]+'~'+'+'.join(self.columns[:-1])

        kwargs = {"formula": Formula(formula),
                  "weights": sample_weight,
                   "nterms": self.nterms,
                   "optlevel": self.optlevel,
                   "sm.method": "gcvspline",
                   "data": pd.DataFrame(np.hstack([x, y.reshape(-1,1)]),
                                        columns=self.columns)}
        self.reg = stats.ppr(**kwargs)
        self.x = x.copy()
        
        return self
    

    def predict(self, x):

        """output f(x) for given samples
        
        Parameters
        ---------
        x : array-like of shape (n_samples, d)
            containing the input dataset
        Returns
        -------
        np.array of shape (n_samples,)
            containing f(x) 
        """
        
        return self.decision_function(x)
    

class PPRClassifier(BasePPR,ClassifierMixin):

    """
    This is a Python wrapper of the 'gppr' function in the 'gsg'. We use IRLS to fit the classification model.
    
    Parameters
    ---------
    nterms : number of ridge terms.
    optlevel : optimization level, can be 0 or 1 or 2. Set to be 2 (highlevel/backfitting) by default.
    tol: tolerance of IRLS algorithm. Set to be 1e-3 by default.
    maxit: maximum numbere of iterations for IRLS. Set to be 50 by default.
    bound_tol: tolerance to truncate the estimated probability to avoid numerical overflow. Set to be 1e-5 by default.
    """
    
    def __init__(self, nterms=5, optlevel=2, 
                 bound_tol=1e-5, tol=1e-3,
                 maxit=50):

        super(PPRClassifier, self).__init__(nterms=nterms, optlevel=optlevel)
        self.bound_tol = bound_tol
        self.tol = tol
        self.maxit = maxit
        
    def _linkfun(self, x):
        '''link function (logit, f(x)=log(x/(1-x)))
        Parameters
        ---------
        x: the probability P(y=1|X)
        '''
        return np.log(x/(1-x))
    
    def _linkinv(self, x):
        '''inversion of link function (sigmoid, f(x)=1/(1+exp(-x)))
        Parameters
        ---------
        x: mean function of the logistic model
        '''
        return np.exp(x)/(1+np.exp(x))
    
    def _varfun(self, x):
        '''variance function
        Parameters
        ---------
        x: the probability P(y=1|X)
        '''
        return x*(1-x)
    
    def _validate_input(self, x, y):
        
        """method to validate data
        
        Parameters
        ---------
        x : array-like of shape (n_samples, n_features)
            containing the input dataset
        y : array-like of shape (n_samples,)
            containing target values
        """

        x, y = check_X_y(x, y, accept_sparse=["csr", "csc", "coo"],
                         multi_output=True)
        if y.ndim == 2 and y.shape[1] == 1:
            y = column_or_1d(y, warn=False)

        self._label_binarizer = LabelBinarizer()
        self._label_binarizer.fit(y)
        self.classes_ = self._label_binarizer.classes_

        y = self._label_binarizer.transform(y) * 1.0
        return x, y.ravel()
    
    def fit(self, x, y, sample_weight=None):

        """fit the smoothing spline
        Parameters
        ---------
        x : array-like of shape (n_samples, n_features)
            containing the input dataset
        y : array-like of shape (n_samples,)
            containing target values
        sample_weight : array-like of shape (n_samples,), optional
            containing sample weights
        Returns
        -------
        object 
            self : Estimator instance.
        """

        x, y = self._validate_input(x, y)

        # initialize the expectation
        mu = np.mean(y)*np.ones(y.shape[0])
        self.deltamu = 1 #delta of the probability
        self.it = 0 #iterator

        while ((self.deltamu>=self.tol)&(self.it<self.maxit)):

            # calculate response and weight
            z = self._linkfun(mu)+(y-mu)/self._varfun(mu)
            cur_weights = self._varfun(mu)

            # fit ppr
            self.reg = PPRRegressor(nterms=self.nterms,
                          optlevel=self.optlevel)

            self.reg.fit(x,z,sample_weight=cur_weights)

            # mean
            eta = self.reg.predict(x)

            # probability
            mu_prev = mu.copy()
            mu = self._linkinv(eta)

            # deal with NaN that weights happen when mu in (0,1)
            mu[mu<self.bound_tol] = self.bound_tol
            mu[mu>(1-self.bound_tol)] = 1-self.bound_tol

            # calculate the difference
            self.deltamu = np.sum(np.abs(mu_prev-mu))/np.sum(np.abs(mu_prev))
            # counter + 1
            self.it += 1
        
        self.x = x.copy()
        
        return self
    
    def predict_proba(self, x):
        
        """output probability prediction for given samples
        
        Parameters
        ---------
        x : array-like of shape (n_samples, n_features)
            containing the input dataset
        Returns
        -------
        np.array of shape (n_samples, 2)
            containing probability prediction
        """

        pred = self.decision_function(x)
        pred_proba = softmax(np.vstack([-pred, pred]).T / 2, copy=False)
        return pred_proba
    

    def predict(self, x):
            
        """output binary prediction for given samples
        
        Parameters
        ---------
        x : array-like of shape (n_samples, n_features)
            containing the input dataset
        Returns
        -------
        np.array of shape (n_samples,)
            containing binary prediction
        """  

        pred_proba = self.predict_proba(x)[:, 1]
        return self._label_binarizer.inverse_transform(pred_proba)