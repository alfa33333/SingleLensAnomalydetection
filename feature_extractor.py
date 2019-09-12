'Automatic feature extractor'
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.datasets import make_moons, make_blobs
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from lensutils import read_data as read
from lensutils import SingleLens as slu

class outlierScore():
    def __init__(self, *args, **kwargs):
        self.outliers_fraction = 0.20
        self.random_state=42
        self.n_neighbors=20
        if kwargs:
            self.__dict__.update(kwargs)

    def main_detection(self,folder,solution,show=True,**kwargs):
        fitter = self.singlelens(folder,solution,**kwargs)
        self.y_pred, self.time_out, self.Y, self.t_vec, self.obs_vc,\
             self.err_vec, self.magl = self.outlier_detection(fitter,self.outliers_fraction,self.random_state,self.n_neighbors)
        names = ["Robust covariance","One-Class SVM","Isolation Forest","Local Outlier Factor"]
        #feat_len = np.linspace(0,len(self.Y),len(self.Y))
        self.print(names,self.Y[:,0],self.Y[:,1],self.y_pred,self.time_out,show=show, **kwargs)
        self.print(names,self.t_vec,self.obs_vc,self.y_pred,self.time_out,show=show, **kwargs)
        self.y_abs = self.extract_abs(self.y_pred)
        self.y_int = self.y_abs.astype(int)

    #def properties(self,attribute):
    #    return self.attribute


    @classmethod    
    def outlier_detection(cls,fitter,outliers_fraction,random_state,n_neighbors):
        Y, t_vec, obs_vc, err_vec, magl =  cls.chi2feature(fitter)

        # define outlier/anomaly detection methods to be compared
        anomaly_algorithms = [
            ("Robust covariance", EllipticEnvelope(contamination=outliers_fraction)),
            ("One-Class SVM", svm.OneClassSVM(nu=outliers_fraction, kernel="rbf",
                                            gamma=0.1)),
            ("Isolation Forest", IsolationForest(behaviour='new',
                                                contamination=outliers_fraction,
                                                random_state=random_state)),
            ("Local Outlier Factor", LocalOutlierFactor(
                n_neighbors=n_neighbors, contamination=outliers_fraction))]
        y_pred= {}
        for name, algorithm in anomaly_algorithms:
            t0 = time.time()
            algorithm.fit(Y)
            t1 = time.time()  


            # fit the data and tag outliers
            if name == "Local Outlier Factor":
                y_pred[name] = algorithm.fit_predict(Y)
            else:
                y_pred[name] = algorithm.fit(Y).predict(Y)
        
        time_out = t1-t0

        return y_pred, time_out , Y, t_vec, obs_vc, err_vec, magl

    @classmethod
    def singlelens(cls,folder,solution,**kwargs):
        if 't_range' in kwargs:
            data = cls.read(folder,kwargs.get('t_range'))
        else:
            data = cls.read(folder,t_range=None,max_uncertainty=1 )
        fitter = slu(data,solution)
        return fitter

    @staticmethod
    def read(folder,t_range, **kwargs):
        return read(folder,t_range,**kwargs)
    
    @staticmethod
    def chi2feature(fitter):
        t_vec = []
        obs_vc = []
        err_vec = []
        for data_key in fitter.data.keys():
            print(data_key)
            t, obs, err = fitter.data[data_key]
            coeffs, _ = fitter.linear_fit(data_key,fitter.magnification(t))
            obsl = (obs - coeffs[0])/coeffs[1]
            err1 = err/coeffs[1]
            t_vec = np.append(t_vec,t)
            obs_vc = np.append(obs_vc,obsl)
            err_vec = np.append(err_vec,err1)
        obs_vc = obs_vc[t_vec.argsort()]
        err_vec = err_vec[t_vec.argsort()]
        t_vec = np.sort(t_vec)
        magl = fitter.magnification(t_vec)
        chi2 = (magl-obs_vc)**2/err_vec**2
        manhatan = obs_vc-magl
        ##normalization
        chi_scaled = ( chi2 )/(np.max(chi2))
        t_scaled = (manhatan-np.mean(manhatan))/(np.max(manhatan)-np.min(manhatan))

        Y = np.hstack((t_scaled.reshape(len(t_scaled),1),chi_scaled.reshape(len(chi_scaled),1)))
        return Y, t_vec, obs_vc, err_vec, magl

    @staticmethod
    def extract_abs(y_pred,threshold = 1):
        name = [None] * len(y_pred.keys())
        for i,names in enumerate(y_pred.keys()):
            name[i] = names 
        test = np.c_[y_pred[name[0]], y_pred[name[1]],y_pred[name[2]],y_pred[name[3]]]
        y_abs = np.sum((test+1)//2,axis=1)<=threshold
        return y_abs
    

    @staticmethod
    def print(anomaly_algorithms,X,Y,y_pred,time=0,show=True,name_fig='figure',folder='./',**kwargs):
        ''' Prints the points in a 2D plot to a fixed shape.
            Inputs:
                anomaly_algorithms: The list of names "n". It must be a list.
                X, Y: The shape of "m" data points. 
                y_pred: the predicted data points. It is a dictionary.  
                time: time it took to calculate. Defauld is 0.
                show: if the plot is show or not. Default is True.
                name_fig: figure name. Defaul is 'figure'.
                folder: folder to save the figure. Default is './'. 
            Other parameters:
                **kwargs to pass to the savefig functions.

        '''
        matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
        #plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
        #                    hspace=.01)
        fig, axes = plt.subplots(1, len(anomaly_algorithms),figsize=(16, 4))
        if len(anomaly_algorithms)==1:
            axes = np.array([axes])
            fig.tight_layout()
        for i, ax in enumerate(axes):
            name = anomaly_algorithms[i]
            ax.set_title(name, size=18)
            colors = np.array(['#377eb8', '#ff7f00'])
            ax.scatter(X, Y, s=10, color=colors[(y_pred[name] + 1) // 2])

            ax.set_xlim(min(X), max(X))
            ax.set_ylim(min(Y), max(Y))
            ax.text(.99, .01, ('%.2fs' % (time)).lstrip('0'),
                        transform=plt.gca().transAxes, size=15,
                        horizontalalignment='right')
        fig.savefig(fname=folder+name_fig,**kwargs)
        if show:
            plt.show()
        plt.close(fig)

def outliers_plot(t_vec, obs_vc,magl,y_abs,save=True,**kwargs):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.set_title("Local Outlier Factor (LOF)")
    ax.scatter(t_vec, obs_vc, color='k', s=3., label='Data points')
    ax.scatter(t_vec[y_abs], obs_vc[y_abs], s=50,marker='o',edgecolors='r',facecolors='none' ,label='Outlier scores')
    ax.plot(t_vec,magl, label='Model')
    ax.axis('tight')
    ax.legend(loc='upper left')
    if save:
        fig.savefig("./figure2")
    plt.show()
    plt.close(fig)

class timeFeat(outlierScore):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.perct = 0.95

    def time_feature(self,folder, solution, **kwargs):
        super().main_detection(folder, solution, **kwargs)
        self.minlevel = self.light_level(self.magl, self.perct)
        self.y_test = self.significant_outliers(self.y_int,\
                                                self.magl, self.obs_vc, self.minlevel)
        self.biggest, self.average, self.score_track, self.unique, self.counts = self.time_score(self.y_test)
        self.frac, self.total = self.time_count(self.t_vec, self.magl, self.minlevel, self.score_track, self.biggest)

    def plot_score(self):
        plt.plot(self.t_vec,self.score_track)
        plt.show()

    @staticmethod
    def light_level(magl,perct):
        inlevel = np.max(magl)-(np.max(magl)-np.min(magl))*(perct)
        return inlevel
    
    @staticmethod
    def significant_outliers(y_int,magl,obs_vc,minlevel):
        y_test = np.copy(y_int)
        y_test[magl<minlevel] = 0
        y_test[obs_vc<1] =  0
        return y_test

    @staticmethod
    def time_score(y_test):
        score = 0
        score_track = np.zeros(len(y_test))
        for i,track in enumerate(y_test):
            if track == 1:
                score += 1
                score_track[i] = score
            elif track==0 and score>0:
                score -= 1
                score_track[i] = score
            else:
                score_track[i] = score
        unique, counts = np.unique(score_track, return_counts=True)
        weight_list = (1/counts)*np.sum(1/counts)
        weights = np.zeros_like(score_track)
        for i,val in enumerate(unique):
            weights[score_track==val] = weight_list[i]
        average = np.average(score_track,weights=weights)
        biggest = np.where(score_track>= unique[-1])[0]
        return biggest, average, score_track, unique, counts

    @staticmethod
    def time_count(t_vec,magl,minlevel,score_track,biggest):
        time_peak95 = np.max(t_vec[magl>minlevel])-np.min(t_vec[magl>minlevel])
        total = 0
        old = [-1,-1]
        for i in biggest:
            temp = np.where(score_track<=0)[0]
            maxval = temp[temp > i][0]
            minval = temp[temp < i][-1]
            if (old[0] == t_vec[maxval]) and (old[1] == t_vec[minval]) :
                continue
            time_outl = t_vec[maxval]-t_vec[minval]
            old = [t_vec[maxval], t_vec[minval]]
            frac = time_outl/time_peak95
            total += frac
            print('Fraction of the longest ourliers combination: {:4f}'.format(frac))
        print('total fraction longest on outliers: {:.4f}'.format(total))
        print('Average anomalytime on outliers:{:.4f}'.format(total/len(biggest)))
        return frac, total 

