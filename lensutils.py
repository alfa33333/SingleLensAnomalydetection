'''Utilities for single lens models
'''
import os
import numpy as np
from scipy import linalg

def read_data(data_directory, t_range=None, minimum_points=5, max_uncertainty=0.05):

    """Load and assemble data.

        We assume that all files in data_directory have first 3 columns (date,mag,err_mag).

        t_range (optional) is a tupple of (start_date,end_date).

        0.01 mag is added in quadrature to all uncertainties.

        Returns a dictionary of (source,source_data) pairs, where each source_data is a
        tupple of (date,flux,err_flux).
    """

    print('Reading data:')

    if t_range is None:
        t_range = (-1.e6, 1.e6)

    data = {}

    files = [fl_ind for fl_ind in os.listdir(data_directory) if fl_ind.split('.')[-1] == 'dat']
    print(files)

    for fl_ind in files:

        source = fl_ind[:8]
        if source[:3] == 'OGL':
            source = 'OGLE'

        source = source.replace('_', '-')
        datafile = data_directory+'/'+fl_ind
        if os.path.exists(datafile):
            dat = np.loadtxt(datafile, comments=['<', '#'], usecols=(0, 1, 2))
            time = dat[:, 0]
            if time[0] > 2440000:
                time -= 2450000
            points = np.where((t_range[0] < time) & (time < t_range[1]))[0]
            if '2018' in fl_ind and 'KMTS' in fl_ind:
                gpoints = np.where((time[points] < 8198.3) | (time[points] > 8198.8))[0]
                points = points[gpoints]
            time = time[points]
            if fl_ind[:3] == 'MOA':
                yaxis = dat[points, 1]
                dyaxis = dat[points, 2]
            else:
                yaxis = 10.0**(0.4*(25 - dat[points, 1]))
                dat[points, 2] = np.sqrt(dat[points, 2]**2 + (0.01)**2)
                dyaxis = 10.0**(0.4*(25 - dat[points, 1] + dat[points, 2])) - yaxis
            points = np.where(dyaxis/yaxis < max_uncertainty)[0]
            time = time[points]
            yaxis = yaxis[points]
            dyaxis = dyaxis[points]
            if len(time) >= minimum_points:
                data[source] = (time, yaxis, dyaxis)
    return data

class SingleLens():
    '''Utilities to generate the single lens model.'''
    def __init__(self, data, initial_parameters, reference_source=None):
        """Initialise the Single Lens.

		inputs:

			data:           A dictionary with each key being a data set name string and each
						value being a tuple of (date, flux, flux_err) each being numpy
						arrays.

			initial_parameters: A numpy array of starting guess values for u_0, t_0, t_E.
		"""

        self.data = data
        self.initial_parameters = initial_parameters
        self.par = initial_parameters

        if reference_source is None:

            self.reference_source = list(self.data.keys())[0]

            print('Using '+self.reference_source+' as reference.')

        else:

            if reference_source in self.data:

                self.reference_source = reference_source
                print('Using '+self.reference_source+' as reference.')

            else:

                self.reference_source = list(self.data.keys())[0]

                print('Warning:', reference_source, 'is not a valid data source.')
                print('Using', self.reference_source, 'as reference.')


        self.parameter_labels = [r"$u_0$", r"$t_0$", r"$t_E$"]
        self.ndim = 3
        self.bias = 25 # Basically never used unless renormalization.

        self.marginalise_linear_parameters = True
        self.fit_blended = True
        return

    def linear_fit(self, data_key, mag):
        ''' Finds the linear flux parameters
                A = F_a*A+F_b
        '''

        _, yaxis, yerr = self.data[data_key]

        cov = np.diag(yerr**2)

        cov_inv = linalg.inv(cov)

        if self.fit_blended:
            amp = np.vstack((np.ones_like(mag), mag))
            n_params = 2
        else:
            amp = (mag-1.0).reshape(1, len(mag))
            n_params = 1

        smatrix = np.dot(amp, np.dot(cov_inv, amp.T)).reshape(n_params, n_params)
        bflux = np.dot(amp, np.dot(cov_inv, yaxis).T)

        if self.marginalise_linear_parameters:

            try:
                mmatrix = linalg.inv(smatrix)
            except linalg.LinAlgError:
                return (0, 0), -np.inf

            gpar = np.dot(mmatrix, bflux)
            dvec = yaxis - np.dot(amp.T, gpar)
            chi2 = np.dot(dvec.T, np.dot(cov_inv, dvec))

            lnprob = np.log(2*np.pi) - 0.5*chi2 - 0.5*np.log(linalg.det(mmatrix))

            return gpar, lnprob

        else:

            try:
                aflux = linalg.solve(smatrix, bflux)
            except linalg.LinAlgError:
                return (0, 0), -np.inf
            dvec = yaxis - np.dot(amp.T, aflux)
            chi2 = np.dot(dvec.T, np.dot(cov_inv, dvec))
            lnprob = -np.log(np.sum(np.sqrt(2*np.pi*yerr**2))) - 0.5*chi2

            return aflux, lnprob

    def magnification(self, time, par=None):
        '''Returns the single lens Magnification from a given model'''

        if par is None:
            par = self.par

        u0_mod, t0_mod, te_mod = par[:3]

        tau = (time-t0_mod)/te_mod

        uvec = np.sqrt(u0_mod**2+tau**2)

        amp = (uvec**2 + 2.0)/(uvec*np.sqrt(uvec**2+4.0))

        return amp
    