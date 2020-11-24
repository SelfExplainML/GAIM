import numpy as np
from matplotlib import gridspec
from matplotlib import pyplot as plt
from abc import ABCMeta, abstractmethod

from sklearn.utils.extmath import softmax
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array, check_X_y
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

from csaps import CubicSmoothingSpline

__all__ = ["SMSplineRegressor"]

    
class BaseSMSpline(BaseEstimator, metaclass=ABCMeta):


    @abstractmethod
    def __init__(self, reg_gamma=0.95, xmin=-1, xmax=1):

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

        if (self.reg_gamma < 0) or (self.reg_gamma > 1):
            raise ValueError("reg_gamma must be >= 0 and <1, got %s." % self.reg_gamma)
        
        if self.xmin > self.xmax:
            raise ValueError("xmin must be <= xmax, got %s and %s." % (self.xmin, self.xmax))

    def diff(self, x, order=1):
             
        """method to calculate derivatives of the fitted spline to the input
        
        Parameters
        ---------
        x : array-like of shape (n_samples, 1)
            containing the input dataset
        order : int
            order of derivative
        """
        
        if 'csaps' in str(self.sm_.__class__):
            derivative = self.sm_(x,nu=order)
        else:
            if order == 1:
                scalars = self.sm_.coef_
            else:
                scalars = 0
            derivative = np.ones(x.shape[0])*scalars
        
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
        try:
            pred = self.sm_(x)
        except:
            pred = self.sm_.predict(x.reshape(-1,1))
        return pred.flatten()


class SMSplineRegressor(BaseSMSpline, RegressorMixin):

    """Base class for Cubic Smoothing Spline regression.
    Details:
    1. This is an API wrapper for the Python package `csaps`.
    2. To handle input data with less than 4 unique samples, we replace smoothing spline by glm. 
    3. During prediction, the data which is outside of the given `xmin` and `xmax` will be clipped to the boundary.
    
    Parameters
    ----------
    reg_gamma : float, optional. default=0.95
            the roughness penalty strength of the spline algorithm, range from 0 to 1 
    
    xmin : float, optional. default=-1
        the min boundary of the input
    
    xmax : float, optional. default=1
        the max boundary of the input
    """

    def __init__(self, reg_gamma=0.1, xmin=-1, xmax=1):

        super(SMSplineRegressor, self).__init__(
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

        unique_num = len(np.unique(x.round(decimals=6)))
        if unique_num >= 4:
            
            x_uni, idx_uni = np.unique(x,return_index=True)
            y_uni = y[idx_uni]
            x_uni_ord = np.sort(x_uni)
            y_uni_ord = y_uni[np.argsort(x_uni)]
            
            n_samples = x_uni.shape[0]
            if sample_weight is None:
                sample_weight = np.ones(n_samples)
            else:
                sample_weight = sample_weight[idx_uni][np.argsort(x_uni)]
                sample_weight = np.round(sample_weight / np.sum(sample_weight) * n_samples, 4)
            
            self.sm_ = CubicSmoothingSpline(xdata=x_uni_ord,
                           ydata=y_uni_ord,
                           weights=sample_weight,
                           smooth=self.reg_gamma)
        else:
            n_samples = x.shape[0]
            if sample_weight is None:
                sample_weight = np.ones(n_samples)
            else:
                sample_weight = np.round(sample_weight / np.sum(sample_weight) * n_samples, 4)
            
            self.sm_ = LinearRegression()
            self.sm_.fit(X=x,y=y,sample_weight=sample_weight)
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