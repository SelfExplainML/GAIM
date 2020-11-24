import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, RegressorMixin

from .smspline_stats import SMSplineRegressor as STATS_SMSplineRegressor
from .smspline_csaps import SMSplineRegressor as CSAPS_SMSplineRegressor
from .smspline_mgcv import SMSplineRegressor as MGCV_SMSplineRegressor
from .smspline_bigspline import SMSplineRegressor as BIG_SMSplineRegressor

class SIMRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, method="second", reg_lambda=0.1, reg_gamma=0., 
                 spline_method='csaps',
                 knot_num=10, degree=3, random_state=0):

        self.method = method
        self.spline_method = spline_method
        self.reg_lambda = reg_lambda # penalty of stein method
        self.reg_gamma = reg_gamma # penalty of regression spline
        self.knot_num = knot_num
        self.degree = degree
        self.random_state = random_state

    def first_stein(self, x, y, proj_mat=None):
        
        s1 = np.dot(self.inv_cov, (x - self.mu).T).T
        zbar = np.mean(y.reshape(-1, 1) * s1, axis=0)
        sigmat = np.dot(zbar.reshape([-1, 1]), zbar.reshape([-1, 1]).T)
        
        if proj_mat is not None:
            sigmat = np.dot(np.dot(proj_mat, sigmat), proj_mat)
            
        beta = np.linalg.svd(sigmat)[0][:, :1]
        beta[np.abs(beta) < self.reg_lambda * np.max(np.abs(beta))] = 0
        beta = beta/np.linalg.norm(beta)
        
        return beta

    def second_stein(self, x, y, proj_mat=None):

        n_samples, n_features = x.shape
        s1 = np.dot(self.inv_cov, (x - self.mu).T).T
        sigmat = np.tensordot(s1 * y.reshape([-1, 1]), s1, axes=([0], [0])) / n_samples
        sigmat -= np.mean(y) * self.inv_cov
        # ensure the semi-positive definite
        u,s,_ = np.linalg.svd(sigmat)
        # svd in python ensure the positive singular values
        sigmat = np.dot(u*s.reshape(1,-1),u.T)
        
        if proj_mat is not None:
            sigmat = np.dot(np.dot(proj_mat, sigmat), proj_mat)
        
        beta = np.linalg.svd(sigmat)[0][:, :1]
        beta[np.abs(beta) < self.reg_lambda * np.max(np.abs(beta))] = 0
        beta = beta/np.linalg.norm(beta)
        
        return beta
    
    def estimate_shape_function(self, x, y):
        
        if self.spline_method == 'csaps':
            self.shape_fit_ = CSAPS_SMSplineRegressor(reg_gamma=self.reg_gamma,
                                           xmin=self.xmin_, xmax=self.xmax_).fit(x, y)
        elif self.spline_method == 'stats':
            self.shape_fit_ = STATS_SMSplineRegressor(reg_gamma=self.reg_gamma,
                                                      knot_num=self.knot_num,
                                                      xmin=self.xmin_, xmax=self.xmax_, knot_dist='quantile').fit(x, y)
        elif self.spline_method == 'mgcv':
            self.shape_fit_ = MGCV_SMSplineRegressor(reg_gamma=self.reg_gamma,
                                                      knot_num=self.knot_num,
                                                      xmin=self.xmin_, xmax=self.xmax_, knot_dist='quantile').fit(x, y)
        elif self.spline_method == 'bigspline':
            self.shape_fit_ = BIG_SMSplineRegressor(reg_gamma=self.reg_gamma,
                                                      knot_num=self.knot_num,
                                                      xmin=self.xmin_, xmax=self.xmax_, knot_dist='quantile').fit(x, y)

    def visualize(self):

        """draw the fitted projection indices and ridge function
        """
        
        check_is_fitted(self, "beta_")
        check_is_fitted(self, "shape_fit_")

        xlim_min = - max(np.abs(self.beta_.min() - 0.1), np.abs(self.beta_.max() + 0.1))
        xlim_max = max(np.abs(self.beta_.min() - 0.1), np.abs(self.beta_.max() + 0.1))

        fig = plt.figure(figsize=(12, 4))
        outer = gridspec.GridSpec(1, 2, wspace=0.15)      
        inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0], wspace=0.1, hspace=0.1, height_ratios=[6, 1])
        ax1_main = plt.Subplot(fig, inner[0]) 
        xgrid = np.linspace(self.shape_fit_.xmin, self.shape_fit_.xmax, 100).reshape([-1, 1])
        ygrid = self.shape_fit_.decision_function(xgrid)
        ax1_main.plot(xgrid, ygrid)
        ax1_main.set_xticklabels([])
        ax1_main.set_title("Shape Function", fontsize=12)
        fig.add_subplot(ax1_main)
        
        ax1_density = plt.Subplot(fig, inner[1]) 
        xint = ((np.array(self.shape_fit_.bins_[1:]) + np.array(self.shape_fit_.bins_[:-1])) / 2).reshape([-1, 1]).reshape([-1])
        ax1_density.bar(xint, self.shape_fit_.density_, width=xint[1] - xint[0])
        ax1_main.get_shared_x_axes().join(ax1_main, ax1_density)
        ax1_density.set_yticklabels([])
        ax1_density.autoscale()
        fig.add_subplot(ax1_density)

        ax2 = plt.Subplot(fig, outer[1]) 
        if len(self.beta_) <= 10:
            rects = ax2.barh(np.arange(len(self.beta_)), [beta for beta in self.beta_.ravel()][::-1])
            ax2.set_yticks(np.arange(len(self.beta_)))
            ax2.set_yticklabels(["X" + str(idx + 1) for idx in range(len(self.beta_.ravel()))][::-1])
            ax2.set_xlim(xlim_min, xlim_max)
            ax2.set_ylim(-1, len(self.beta_))
            ax2.axvline(0, linestyle="dotted", color="black")
        else:
            active_beta = []
            active_beta_idx = []
            for idx, beta in enumerate(self.beta_.ravel()):
                if np.abs(beta) > 0:
                    active_beta.append(beta)
                    active_beta_idx.append(idx)
           
            rects = ax2.barh(np.arange(len(active_beta)), [beta for beta in active_beta][::-1])
            if len (active_beta) > 10:
                input_ticks = np.linspace(0.1 * len(active_beta), len(active_beta) * 0.9, 4).astype(int)
                input_labels = ["X" + str(active_beta_idx[idx] + 1) for idx in input_ticks][::-1] 
                ax2.set_yticks(input_ticks)
                ax2.set_yticklabels(input_labels)
            else:
                ax2.set_yticks(np.arange(len(active_beta)))
                ax2.set_yticklabels(["X" + str(idx + 1) for idx in active_beta_idx][::-1])
            ax2.set_xlim(xlim_min, xlim_max)
            ax2.set_ylim(-1, len(active_beta_idx))
            ax2.axvline(0, linestyle="dotted", color="black")
        ax2.set_title("Projection Indice", fontsize=12)
        fig.add_subplot(ax2)
        plt.show()

    def fit(self, x, y, proj_mat=None):
        
        self.mu = x.mean(0) 
        self.cov = np.cov(x.T)
        self.inv_cov = np.linalg.pinv(self.cov)
        
        if proj_mat is None:
            proj_mat = np.eye(x.shape[1])

        np.random.seed(self.random_state)
        if self.method == "first":
            self.beta_ = self.first_stein(x, y, proj_mat)
        elif self.method == "second":
            self.beta_ = self.second_stein(x, y, proj_mat)
        
        if len(self.beta_[np.abs(self.beta_) > 0]) > 0:
            if (self.beta_[np.abs(self.beta_) > 0][0] < 0):
                self.beta_ = - self.beta_
        
        xb = np.dot(x, self.beta_)
        self.xmin_ = np.min(xb)
        self.xmax_ = np.max(xb)
        self.estimate_shape_function(xb, y)
        return self

    def predict(self, x):

        check_is_fitted(self, "beta_")
        check_is_fitted(self, "shape_fit_")
        xb = np.dot(x, self.beta_)
        pred = self.shape_fit_.predict(xb)
        
        return pred