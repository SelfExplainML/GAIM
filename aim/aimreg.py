import numpy as np
import os 
from sklearn.utils.validation import check_is_fitted

from sklearn.utils import check_X_y, column_or_1d
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec

from .smspline_stats import SMSplineRegressor as STATS_SMSplineRegressor
from .smspline_csaps import SMSplineRegressor as CSAPS_SMSplineRegressor
from .smspline_mgcv import SMSplineRegressor as MGCV_SMSplineRegressor
from .smspline_bigspline import SMSplineRegressor as BIG_SMSplineRegressor

from .mysim import SIMRegressor

class AIMRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, nterms=5, opt_level='high',
                 # params of splines
                 reg_gamma=0.,
                 knot_num=10, knot_dist='quantile',
                 spline_method='stats',
                 # params of coefficients
                 beta_init_method='ols',
                 reg_lambda=0.,
                 beta_opt_method='gauss_newton',
                 # params of stage fitting
                 eps_stage=1e-4, stage_maxiter=10,
                 # params of backfitting
                 eps_backfit=1e-3, backfit_maxiter=10,
                 # param of elimination
                 elimination_ratio=0.05,
                 # misc params
                 random_state=0, verbose=0):

        # general params
        self.nterms = nterms       # number of ridge terms
        self.opt_level = opt_level # optimal level ('low': sequential, 'high': backfitting)
        
        # params of splines
        self.reg_gamma = reg_gamma # penalty of smoothness
        self.knot_num = knot_num   # number of knots (only applicable to 'stats','bigspline' and 'mgcv')
        self.knot_dist = knot_dist # distribution of knots (only applicable to 'stats','bigspline' and 'mgcv')
        self.spline_method = spline_method # spline methods
        
        # params of coefficients
        self.beta_init_method = beta_init_method # initial method for coefficients 
        self.beta_opt_method = beta_opt_method   # opitmization method to update coefficients 
        self.reg_lambda = reg_lambda # induce the sparsity
        
        # params of stage fitting (fit both spline and coefficients)
        self.eps_stage = eps_stage
        self.stage_maxiter = stage_maxiter
        
        # params of backfitting
        self.eps_backfit = eps_backfit
        self.backfit_maxiter = backfit_maxiter
        
        # param of elimination
        ##  remove the ridge terms whose important ratios are smaller than elimination_ratio after fitting
        self.elimination_ratio = elimination_ratio 
        
        # misc params
        self.random_state = random_state
        self.verbose = verbose
    
    
    def _validate_hyperparameters(self):
        """method to validate model parameters
        """

        if not isinstance(self.nterms, int):
            raise ValueError("nterms must be an integer, got %s." % self.nterms)
        elif self.nterms < 0:
            raise ValueError("nterms must be >= 0, got %s." % self.nterms)
        
        if self.opt_level not in ["low", "high"]:
            raise ValueError("opt_level must be an element of ['low', 'high'], got %s." % self.opt_level)
        
        if self.spline_method not in ['csaps','stats','bigspline','mgcv']:
            raise ValueError("opt_level must be an element of ['csaps','stats','bigspline','mgcv'], got %s." % self.spline_method)
        
        if not isinstance(self.reg_gamma, float):
            raise ValueError("reg_gamma must be a float number, got %s." % self.reg_gamma)
            
        if not isinstance(self.knot_num, int):
            raise ValueError("knot_num must be an integer, got %s." % self.knot_num)
        elif self.knot_num < 0:
            raise ValueError("knot_num must be >= 0, got %s." % self.knot_num)
            
        if self.knot_dist not in ["uniform", "quantile"]:
            raise ValueError("knot_dist must be an element of ['uniform', 'quantile'], got %s." % self.knot_dist)   
        
        if self.beta_init_method not in ['random', 'ols', 'marginal_regression', 'stein2nd']:
            raise ValueError("beta_init_method must be an element of ['random', 'ols', 'marginal_regression', 'stein2nd'], got %s." % self.beta_init_method)   
        
        if self.beta_opt_method not in ['exact_newton', 'gauss_newton']:
            raise ValueError("beta_opt_method must be an element of ['exact_newton', 'gauss_newton'], got %s." % self.beta_opt_method)  
        
        if not isinstance(self.reg_lambda, float):
            raise ValueError("reg_lambda must be a float number, got %s." % self.reg_lambda)
        if self.reg_lambda < 0:
            raise ValueError("reg_lambda must be >= 0, got" % self.reg_lambda)
        elif self.reg_lambda > 1:
            raise ValueError("reg_lambda must be <= 1, got" % self.reg_lambda)    
        
        if not isinstance(self.eps_stage, float):
            raise ValueError("eps_stage must be a float number, got %s." % self.eps_stage)
            
        if not isinstance(self.stage_maxiter, int):
            raise ValueError("stage_maxiter must be an integer, got %s." % self.stage_maxiter)
        elif self.stage_maxiter < 0:
            raise ValueError("stage_maxiter must be >= 0, got %s." % self.stage_maxiter)
        
        if not isinstance(self.eps_backfit, float):
            raise ValueError("eps_backfit must be a float number, got %s." % self.eps_backfit)
            
        if not isinstance(self.backfit_maxiter, int):
            raise ValueError("backfit_maxiter must be an integer, got %s." % self.backfit_maxiter)
        elif self.backfit_maxiter < 0:
            raise ValueError("backfit_maxiter must be >= 0, got %s." % self.backfit_maxiter)    
        
        if not isinstance(self.elimination_ratio, float):
                raise ValueError("elimination_ratio must be a float, got %s." % self.elimination_ratio)
        if self.elimination_ratio < 0:
            raise ValueError("elimination_ratio must be >= 0, got" % self.elimination_ratio)
        elif self.elimination_ratio > 1:
            raise ValueError("elimination_ratio must be <= 1, got" % self.elimination_ratio)
    
    @property
    def importance_ratios_(self):
        """return the estimator importance ratios (the higher, the more important the feature)

        Returns
        -------
        dict of selected estimators
            the importance ratio of each fitted base learner.
        """
        importance_ratios_ = {}
        if (self.component_importance_ is not None) and (len(self.component_importance_) > 0):
            total_importance = np.sum([item["importance"] for key, item in self.component_importance_.items()])
            importance_ratios_ = {key: {"type": item["type"],
                               "indice": item["indice"],
                               "ir": item["importance"] / total_importance} for key, item in self.component_importance_.items()}
        return importance_ratios_


    @property
    def projection_indices_(self):
        """return the projection indices

        Returns
        -------
        ndarray of shape (n_features, n_estimators)
            the projection indices
        """
        projection_indices = np.array([])
        if len(self.best_estimators_) > 0:
            projection_indices = np.array([est.beta_.flatten() 
                                for est in self.best_estimators_]).T
        return projection_indices
        
    @property
    def orthogonality_measure_(self):
        """return the orthogonality measure (the lower, the better)
        
        Returns
        -------
        float
            the orthogonality measure
        """
        ortho_measure = np.nan
        if len(self.best_estimators_) > 0:
            ortho_measure = np.linalg.norm(np.dot(self.projection_indices_.T,
                                      self.projection_indices_) - np.eye(self.projection_indices_.shape[1]))
            if self.projection_indices_.shape[1] > 1:
                ortho_measure /= self.projection_indices_.shape[1]
        return ortho_measure

    def second_stein(self, x, y):
        """perform the initialization via the second-order Stein's identity
        
        Parameters
        ---------
        x : array-like of shape (n_samples, n_features)
            containing the input dataset
        y : array-like of shape (n_samples,)
            containing target values
        
        Returns
        -------
        array-like of shape (n_features,)
            the weight vector
        """
            
        n_samples, n_features = x.shape
        mu = x.mean(0) 
        cov = np.cov(x.T)
        inv_cov = np.linalg.pinv(cov)
        s1 = np.dot(inv_cov, (x - mu).T).T
        sigmat = np.tensordot(s1 * y.reshape([-1, 1]), s1, axes=([0], [0])) / n_samples
        sigmat -= np.mean(y) * inv_cov
        # ensure the semi-positive definite
        u,s,_ = np.linalg.svd(sigmat)
        # svd in python ensure the positive singular values
        sigmat = np.dot(u*s.reshape(1,-1),u.T)
        beta = np.linalg.svd(sigmat)[0][:, :1]
    
        return beta
    
    def fit_stage(self, x, y, beta_init=None): 
        """perform alternative optimization for a single ridge term
        
        Parameters
        ---------
        x : array-like of shape (n_samples, n_features)
            containing the input dataset
        y : array-like of shape (n_samples,)
            containing target values
        beta_init: array-like of shape (n_features,)
            the initial vector for projection weights
        """
        
        ## initialization
        if beta_init is None:
            if self.beta_init_method == 'random':
                prev_beta = np.random.randn(x.shape[1],1)
            elif self.beta_init_method == 'ols':
                ls = LinearRegression()
                ls.fit(x, y)
                prev_beta = ls.coef_.reshape(-1,1)
            elif self.beta_init_method == 'marginal_regression':
                prev_beta = np.dot(x.T,y)
            elif self.beta_init_method == 'stein2nd':
                prev_beta = self.second_stein(x,y)
            prev_beta = prev_beta/np.linalg.norm(prev_beta)
        else:
            prev_beta = beta_init
        xb = np.dot(x,prev_beta)
        
        # fitting for each stage
        prev_loss = np.inf
        current_loss = -np.inf
        itr = 0
        
        while ((prev_loss<=current_loss) or (prev_loss>=(current_loss+self.eps_stage))) and (itr<self.stage_maxiter):

            ## update ridge function
            if self.spline_method == 'csaps':
                ridge_fun = CSAPS_SMSplineRegressor(
                                     reg_gamma=self.reg_gamma,
                                     xmin=xb.min(),
                                     xmax=xb.max())
            elif self.spline_method == 'stats':
                ridge_fun = STATS_SMSplineRegressor(knot_num=self.knot_num,
                                 knot_dist=self.knot_dist,
                                 reg_gamma=self.reg_gamma,
                                 xmin=xb.min(),
                                 xmax=xb.max())
            elif self.spline_method == 'bigspline':
                ridge_fun = BIG_SMSplineRegressor(knot_num=self.knot_num,
                                 knot_dist=self.knot_dist,
                                 reg_gamma=self.reg_gamma,
                                 xmin=xb.min(),
                                 xmax=xb.max())
            elif self.spline_method == 'mgcv':
                ridge_fun = MGCV_SMSplineRegressor(knot_num=self.knot_num,
                                 knot_dist=self.knot_dist,
                                 reg_gamma=self.reg_gamma,
                                 xmin=xb.min(),
                                 xmax=xb.max())
            ridge_fun.fit(xb,y.flatten())

            ## update beta
            residuals = y.flatten() - ridge_fun.predict(xb)
            
            if self.beta_opt_method == 'exact_newton':
                ### exact Newton Update

                # first and second order derivatives of f
                df = ridge_fun.diff(xb,order=1)
                ddf = ridge_fun.diff(xb,order=2)

                # calculate newton update step (delta)
                gd = np.mean((-2*residuals*df).reshape(-1,1)*x,axis=0)
                Hess = np.dot(x.T,(2*(df**2-residuals*ddf)).reshape(-1,1)*x)/x.shape[0]
                delta = -np.dot(np.linalg.pinv(Hess),gd).reshape(-1,1)

                # newton update
                beta = prev_beta+delta
                beta[np.abs(beta) < self.reg_lambda * np.max(np.abs(beta))] = 0
                beta = beta/np.linalg.norm(beta)
                
            elif self.beta_opt_method == 'gauss_newton':
                ### Gauss-Newton Update
                
                # follow https://en.wikipedia.org/wiki/Projection_pursuit_regression
                # follow ESL Chapter 11

                # first order derivatives of f
                df = ridge_fun.diff(xb,order=1)
                # b
                b = xb.flatten() + residuals.flatten()/(df.flatten()+1e-8)
                # weighted least squares
                lr = LinearRegression(fit_intercept=False)
                lr.fit(x,b,sample_weight=(df.flatten())**2)
                beta = lr.coef_.reshape(-1,1)
                beta[np.abs(beta) < self.reg_lambda * np.max(np.abs(beta))] = 0
                beta = beta/np.linalg.norm(beta)
            
            xb_tmp = np.dot(x,beta)
            tmp_loss = np.mean((y.flatten() - ridge_fun.predict(xb_tmp).flatten())**2)
            
            prev_loss = current_loss
            current_loss = tmp_loss
            prev_beta = beta
            xb = xb_tmp
            itr += 1
            
            if self.verbose:
                print('prev_loss:',prev_loss,'|current_loss:',current_loss,'|iter:',itr,
                      '|prev_loss > current_loss:',prev_loss > current_loss)
            
        return ridge_fun, beta
    
    def fit(self, x, y):
        """fit the AIM model
        
        Parameters
        ---------
        x : array-like of shape (n_samples, n_features)
            containing the input dataset
        y : array-like of shape (n_samples,)
            containing target values
        """
        
        self._validate_hyperparameters()
        
        r = y.copy()
        self.best_estimators_ = []
        
        np.random.seed(self.random_state)
        for i in range(self.nterms):
            if self.verbose:
                print('------------ nterm:', i + 1,'------------')
           
            ridge_fun, beta = self.fit_stage(x,r)
            est = SIMRegressor(degree=3, random_state=self.random_state)
            est.beta_ = beta
            est.shape_fit_ = ridge_fun
            
            self.best_estimators_.append(est)
            if len(self.best_estimators_)>1 and self.opt_level == 'high':
                self.back_fit_(x,y)
        
            xb = np.dot(x,beta)
            r = r - ridge_fun.predict(xb).reshape(-1, 1)
            
            if self.verbose:
                print('------------ MSE:', np.mean(r**2), '------------')
        
        if not self.elimination_ratio:
            component_importance = {}
            for indice, est in enumerate(self.best_estimators_):
                component_importance.update({"sim " + str(indice + 1): {"type": "sim", "indice": indice,
                                    "importance": np.std(est.predict(x))}})
            self.component_importance_ = dict(sorted(component_importance.items(), key=lambda item: item[1]["importance"])[::-1])
        else:
            importance_ratio = []
            for indice, est in enumerate(self.best_estimators_):
                importance_ratio.append(np.std(est.predict(x)))
            elimination_indicator = np.array(importance_ratio)/np.sum(importance_ratio)>self.elimination_ratio

            component_importance = {}
            best_estimators_eliminate_ = []
            i = 0
            for indice, est in enumerate(self.best_estimators_):
                if elimination_indicator[indice]:
                    best_estimators_eliminate_.append(est)
                    component_importance.update({"sim " + str(i + 1): {"type": "sim", "indice": i,
                                    "importance": np.std(est.predict(x))}})
                    i += 1
            self.component_importance_ = dict(sorted(component_importance.items(), key=lambda item: item[1]["importance"])[::-1])
            self.best_estimators_ = best_estimators_eliminate_

        return self

    def back_fit_(self, x, y):
        """perform the backfitting on the model
        
        Parameters
        ---------
        x : array-like of shape (n_samples, n_features)
            containing the input dataset
        y : array-like of shape (n_samples,)
            containing target values
        """
        
        ## backfitting
        prev_loss = np.inf
        current_loss = -np.inf
        itr = 0
        
        while ((prev_loss<=current_loss) or (prev_loss>=(current_loss+self.eps_backfit))) and (itr<self.backfit_maxiter):
            if self.verbose:
                print('----------------------------- backfitting:', 
                      itr+1,
                      '-----------------------------')
            
            for i in range(len(self.best_estimators_)):
                # residual calculation
                y_hat = self.predict(x).flatten()
                xb = np.dot(x,self.best_estimators_[i].beta_)
                y_hat_no_i = y_hat - self.best_estimators_[i].shape_fit_.predict(xb).flatten()
                r = y.flatten() - y_hat_no_i 
                # backfitting
                ridge_fun, beta = self.fit_stage(x, r, beta_init=self.best_estimators_[i].beta_.reshape(-1,1))
                # update
                self.best_estimators_[i].beta_ = beta.flatten()
                self.best_estimators_[i].shape_fit_ = ridge_fun

            prev_loss = current_loss 
            current_loss = np.mean((y.flatten()-self.predict(x).flatten())**2)
            itr += 1
            
            if self.verbose:
                print('backfitting prev_loss:',prev_loss, 
                      'backfitting current_loss:',current_loss)
        
        return self
    
    
    def predict(self, x):
        """make prediction
        
        Parameters
        ---------
        x : array-like of shape (n_samples, n_features)
            containing the input dataset
        """
        check_is_fitted(self, "best_estimators_")

        y_pred = 0
        for i, est in enumerate(self.best_estimators_):
            y_pred += est.predict(x).reshape([-1, 1])
        return y_pred
    
    def visualize(self, cols_per_row=3, 
                  show_top=None, 
                  show_indices = 20,
                  folder="./results/", name="global_plot", save_png=False, save_eps=False):

        """draw the global interpretation of the fitted model
        
        Parameters
        ---------
        cols_per_row : int, optional, default=3,
            the number of sim models visualized on each row
        show_top: None or int, default=None,
            optional, show top ridge components
        show_indices: int, default=20,
            only show first indices in high-dim cases
        folder : str, optional, defalut="./results/"
            the folder of the file to be saved
        name : str, optional, default="global_plot"
            the name of the file to be saved
        save_png : bool, optional, default=False
            whether to save the figure in png form
        save_eps : bool, optional, default=False
            whether to save the figure in eps form
        """
        check_is_fitted(self, "best_estimators_")
        
        subfig_idx = 0
        max_ids = len(self.best_estimators_)
        fig = plt.figure(figsize=(8 * cols_per_row, 4.6 * int(np.ceil(max_ids / cols_per_row))))
        outer = gridspec.GridSpec(int(np.ceil(max_ids / cols_per_row)), cols_per_row, wspace=0.15, hspace=0.25)
        
        xlim_min = - max(np.abs(self.projection_indices_.min() - 0.1), np.abs(self.projection_indices_.max() + 0.1))
        xlim_max = max(np.abs(self.projection_indices_.min() - 0.1), np.abs(self.projection_indices_.max() + 0.1))
        
        if show_top is None:
            estimators = [self.best_estimators_[item['indice']]  for _, item in self.component_importance_.items()]
        else:
            estimators = [self.best_estimators_[item['indice']]  for _, item in self.component_importance_.items()][:show_top]
            
        for indice, sim in enumerate(estimators):

            estimator_key = list(self.importance_ratios_)[indice]
            inner = outer[subfig_idx].subgridspec(2, 2, wspace=0.2, height_ratios=[6, 1], width_ratios=[3, 1])
            ax1_main = fig.add_subplot(inner[0, 0])
            xgrid = np.linspace(sim.shape_fit_.xmin, sim.shape_fit_.xmax, 100).reshape([-1, 1])
            ygrid = sim.shape_fit_.decision_function(xgrid)
            ax1_main.plot(xgrid, ygrid, color="red")
            ax1_main.set_xticklabels([])
            ax1_main.set_title("SIM " + str(self.importance_ratios_[estimator_key]["indice"] + 1) +
                         " (IR: " + str(np.round(100 * self.importance_ratios_[estimator_key]["ir"], 2)) + "%)",
                         fontsize=16)
            fig.add_subplot(ax1_main)

            ax1_density = fig.add_subplot(inner[1, 0])  
            xint = ((np.array(sim.shape_fit_.bins_[1:]) + np.array(sim.shape_fit_.bins_[:-1])) / 2).reshape([-1, 1]).reshape([-1])
            ax1_density.bar(xint, sim.shape_fit_.density_, width=xint[1] - xint[0])
            ax1_main.get_shared_x_axes().join(ax1_main, ax1_density)
            ax1_density.set_yticklabels([])
            fig.add_subplot(ax1_density)
            
            ax2 = fig.add_subplot(inner[:, 1])
            if len(sim.beta_) <= show_indices:
                rects = ax2.barh(np.arange(len(sim.beta_)), [beta for beta in sim.beta_.ravel()][::-1])
                ax2.set_yticks(np.arange(len(sim.beta_)))
                ax2.set_yticklabels(["X" + str(idx + 1) for idx in range(len(sim.beta_.ravel()))][::-1])
                ax2.set_xlim(xlim_min, xlim_max)
                ax2.set_ylim(-1, len(sim.beta_))
                ax2.axvline(0, linestyle="dotted", color="black")
            else:
                input_ticks = np.arange(show_indices)[::-1]
                rects = plt.barh(np.arange(show_indices), [beta for beta in sim.beta_.ravel()][:show_indices][::-1])
                ax2.set_yticks(input_ticks)
                ax2.set_yticklabels(["X" + str(idx + 1) for idx in input_ticks][::-1])
                ax2.set_xlim(xlim_min, xlim_max)
                ax2.set_ylim(-1, min(len(sim.beta_),show_indices))
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