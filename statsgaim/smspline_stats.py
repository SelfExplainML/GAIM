import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib import pyplot as plt
from abc import ABCMeta, abstractmethod

from sklearn.utils.extmath import softmax
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array, check_X_y
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

import rpy2
from rpy2 import robjects as ro
from rpy2.robjects import Formula
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri, pandas2ri

numpy2ri.activate()
pandas2ri.activate()

try:
    stats = importr("stats")
except:
    utils = importr('utils')
    utils.install_packages('stats', repos='http://cran.us.r-project.org')
    stats = importr("stats")

__all__ = ["SMSplineRegressor"]

    
class BaseSMSpline(BaseEstimator, metaclass=ABCMeta):


    @abstractmethod
    def __init__(self, knot_num=20, knot_dist="uniform", reg_gamma=0.1, xmin=-1, xmax=1):

        self.knot_num = knot_num
        self.knot_dist = knot_dist
        self.reg_gamma = reg_gamma
        self.xmin = xmin
        self.xmax = xmax

    def _estimate_density(self, x):
                
        """method to estimate the density of input data
        
        Parameters
        ---------
        x : array-like of shape (n_samples, n_features)
            containing the input dataset
        """

        self.density_, self.bins_ = np.histogram(x, bins=10, density=True)

    def _validate_hyperparameters(self):
        
        """method to validate model parameters
        """

        if not isinstance(self.knot_num, int):
            raise ValueError("knot_num must be an integer, got %s." % self.knot_num)
        
        if self.knot_num <= 0:
            raise ValueError("knot_num must be > 0, got" % self.knot_num)

        if self.knot_dist not in ["uniform", "quantile"]:
            raise ValueError("method must be an element of [uniform, quantile], got %s." % self.knot_dist)

        if (self.reg_gamma < 0) or (self.reg_gamma > 1):
            raise ValueError("reg_gamma must be >= 0 and <1, got %s." % self.reg_gamma)

        if self.xmin > self.xmax:
            raise ValueError("xmin must be <= xmax, got %s and %s." % (self.xmin, self.xmax))

    def diff(self, x, order=1):
             
        """method to calculate derivatives of the fitted adaptive spline to the input
        
        Parameters
        ---------
        x : array-like of shape (n_samples, 1)
            containing the input dataset
        order : int
            order of derivative
        """

        if "coefficients" in self.sm_.names:
            derivative = 0 if np.isnan(np.array(self.sm_[0][1])) else np.array(self.sm_[0][1])
        elif "spar" in self.sm_.names:
            derivative = np.array(stats.predict(self.sm_, x, deriv=order)[1]).ravel()
        
        return derivative

    def visualize(self):

        """draw the fitted shape function
        """

        check_is_fitted(self, "sm_")

        fig = plt.figure(figsize=(6, 4))
        inner = gridspec.GridSpec(2, 1, hspace=0.1, height_ratios=[6, 1])
        ax1_main = plt.Subplot(fig, inner[0]) 
        xgrid = np.linspace(self.xmin, self.xmax, 100).reshape([-1, 1])
        ygrid = self.decision_function(xgrid)
        ax1_main.plot(xgrid, ygrid)
        ax1_main.set_xticklabels([])
        ax1_main.set_title("Shape Function", fontsize=12)
        fig.add_subplot(ax1_main)
        
        ax1_density = plt.Subplot(fig, inner[1]) 
        xint = ((np.array(self.bins_[1:]) + np.array(self.bins_[:-1])) / 2).reshape([-1, 1]).reshape([-1])
        ax1_density.bar(xint, self.density_, width=xint[1] - xint[0])
        ax1_main.get_shared_x_axes().join(ax1_main, ax1_density)
        ax1_density.set_yticklabels([])
        ax1_density.autoscale()
        fig.add_subplot(ax1_density)
        plt.show()

    def decision_function(self, x):

        """output f(x) for given samples
        
        Parameters
        ---------
        x : array-like of shape (n_samples, 1)
            containing the input dataset
        Returns
        -------
        np.array of shape (n_samples,)
            containing f(x) 
        """

        check_is_fitted(self, "sm_")
        x = x.copy()
        x[x < self.xmin] = self.xmin
        x[x > self.xmax] = self.xmax
        if "coefficients" in self.sm_.names:
            pred = np.array(stats.predict_glm(self.sm_, ro.r("data.frame")(x=x))).ravel()
        elif "spar" in self.sm_.names:
            pred = np.array(stats.predict_smooth_spline(self.sm_, x, deriv=0)[1]).ravel()
        return pred


class SMSplineRegressor(BaseSMSpline, RegressorMixin):

    """Base class for Smoothing Spline regression.
    Details:
    1. This is an API for the well-known R package `stats`, and we call the function smoothing.spline through rpy2 interface.
    2. To handle input data with less than 4 unique samples, we replace smoothing.spline by glm. 
    3. During prediction, the data which is outside of the given `xmin` and `xmax` will be clipped to the boundary.
    
    Parameters
    ----------
    knot_num : int, optional. default=20
           the number of knots
    knot_dist : str, optional. default="uniform"
            the distribution of knots
      
        "uniform": uniformly over the domain
        "quantile": uniform quantiles of the given input data
    reg_gamma : float, optional. default=0.
            the roughness penalty strength of the spline algorithm, range from 0 to 1; 0. <=> GCV
    
    xmin : float, optional. default=-1
        the min boundary of the input
    
    xmax : float, optional. default=1
        the max boundary of the input
    """

    def __init__(self, knot_num=20, knot_dist="uniform", reg_gamma=0., xmin=-1, xmax=1):

        super(SMSplineRegressor, self).__init__(knot_num=knot_num,
                                  knot_dist=knot_dist,
                                  reg_gamma=reg_gamma,
                                  xmin=xmin,
                                  xmax=xmax)

    def _validate_input(self, x, y):

        """method to validate data
        
        Parameters
        ---------
        x : array-like of shape (n_samples, 1)
            containing the input dataset
        y : array-like of shape (n_samples,)
            containing the output dataset
        """

        x, y = check_X_y(x, y, accept_sparse=["csr", "csc", "coo"],
                         multi_output=True, y_numeric=True)
        return x, y.ravel()

    def get_loss(self, label, pred, sample_weight=None):
        
        """method to calculate the cross entropy loss
        
        Parameters
        ---------
        label : array-like of shape (n_samples,)
            containing the input dataset
        pred : array-like of shape (n_samples,)
            containing the output dataset
        sample_weight : array-like of shape (n_samples,), optional
            containing sample weights
        Returns
        -------
        float
            the cross entropy loss
        """

        loss = np.average((label - pred) ** 2, axis=0, weights=sample_weight)
        return loss

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

        self._validate_hyperparameters()
        x, y = self._validate_input(x, y)
        self._estimate_density(x)

        n_samples = x.shape[0]
        if sample_weight is None:
            sample_weight = np.ones(n_samples)
        else:
            sample_weight = np.round(sample_weight / np.sum(sample_weight) * n_samples, 4)

        unique_num = len(np.unique(x.round(decimals=6)))
        if unique_num >= 4:
            self.knot_num = min(unique_num, self.knot_num)
            if self.knot_dist == "uniform":
                knots = list(np.linspace(0, 1, self.knot_num + 2, dtype=np.float32))
                if self.reg_gamma > 0:
                    self.sm_ = stats.smooth_spline(x, y, all_knots=ro.FloatVector(knots),
                                         spar=self.reg_gamma, w=sample_weight, tol=1e-6 * (np.max(x) - np.min(x)))
                else:
                    self.sm_ = stats.smooth_spline(x, y, all_knots=ro.FloatVector(knots),
                                         w=sample_weight, tol=1e-6 * (np.max(x) - np.min(x)))
            elif self.knot_dist == "quantile":
                knots = np.percentile(x, list(np.linspace(0, 100, 50 + 2, dtype=np.float32)))
                knots = ((knots - knots.min()) / (knots.max() - knots.min())).tolist()
                if self.reg_gamma > 0:
                    self.sm_ = stats.smooth_spline(x, y, all_knots=ro.FloatVector(knots),
                                         spar=self.reg_gamma, w=sample_weight, tol=1e-6 * (np.max(x) - np.min(x)))
                else:
                    self.sm_ = stats.smooth_spline(x, y, all_knots=ro.FloatVector(knots),
                                         w=sample_weight, tol=1e-6 * (np.max(x) - np.min(x)))
        else:
            self.sm_ = stats.glm(Formula('y ~ x'), family="gaussian",
                    data=pd.DataFrame({"x":x.ravel(), "y":y.ravel()}),
                    weights=sample_weight)
        return self

    def predict(self, x):

        """output f(x) for given samples
        
        Parameters
        ---------
        x : array-like of shape (n_samples, 1)
            containing the input dataset
        Returns
        -------
        np.array of shape (n_samples,)
            containing f(x) 
        """

        pred = self.decision_function(x)
        return pred