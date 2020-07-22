# -----------------------------------------------------------------------------#
#                            glm_utils.py                                      #
#------------------------------------------------------------------------------#

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numbers

from itertools import combinations
from nistats import design_matrix
from nistats import hemodynamic_models
from nistats.reporting import plot_design_matrix

class Regressor:
    '''Implements representation of the single GLM regressor.
    
    Allows for conversion of regressor described as number of onsets and 
    optionally magnitude modulations into estimated BOLD timecourse through 
    make_first_level_design_matrix function from Nistats. Useful in situations
    where there are mutliple parametrically modulated regressors. Automatically
    handled both cases of unmodulated and modulated regressors.
    '''
    
    def __init__(self, name, frame_times, onset, *, duration=None, 
                 modulation=None):
        '''
        Args:
            name (str): Name of the regressor.
            frame_times (np.ndarray of shape (n_frames,)):
                The timing of acquisition of the scans in seconds.
            onset (array-like): 
                Specifies the start time of each event in seconds.
            duration (array-like, optional): 
                Duration of each event in seconds. Defaults duration is set to 0 
                (impulse function).
            modulation (array-like, optional): 
                Parametric modulation of event amplitude. Before convolution 
                regressor is demeaned. 
        '''
        if not isinstance(frame_times, np.ndarray) or frame_times.ndim != 1:
            msg = 'frame_times should be np.ndarray of shape (n_frames, )'
            raise TypeError(msg)

        self._name = name
        self._frame_times = frame_times
            
        n_events = len(onset)
        
        if duration is None:
            duration = np.zeros(n_events)
            
        if modulation is None or (len(modulation) > 1 
                                  and np.all(np.array(modulation) == modulation[0])):
            modulation = np.ones(n_events)
        elif len(modulation) > 1:
            modulation = np.array(modulation)
            modulation = modulation - np.mean(modulation)
        
        self._values, _ = hemodynamic_models.compute_regressor(
            exp_condition=np.vstack((onset, duration, modulation)),
            hrf_model='spm',
            frame_times=frame_times
        )
     
    @classmethod
    def from_values(cls, name, frame_times, values):
        '''Alternative constructor bypassing compute_regressor function.
        
        Args:
            name (str): Name of the regressor.
            frame_times (np.ndarray of shape (n_frames,)):
                The timing of acquisition of the scans in seconds.
            values (array-like): 
                Regressor values for each frame time.         
        '''
        if not isinstance(frame_times, np.ndarray) or frame_times.ndim != 1:
            msg = 'frame_times should be np.ndarray of shape (n_frames, )'
            raise TypeError(msg)
        if len(values) != len(frame_times):
            msg = 'length mismatch between values and frame_times ' + \
                 f'{len(values)} != {len(frame_times)}'
            raise ValueError(msg)
        
        obj = cls.__new__(cls)
        super(Regressor, obj).__init__()
        obj._name = name
        obj._frame_times = frame_times
        obj._values = np.array(values)
        
        return obj
        
    @property
    def name(self):
        return self._name
        
    @property
    def frame_times(self):
        return self._frame_times
        
    @property
    def values(self):
        return self._values
    
    @property
    def is_empty(self):
        return (self.values == 0).all()
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"
    
    def __len__(self):
        return len(self.frame_times)
    
    def plot(self, color='r') -> None:
        '''Plots BOLD timecourse for regressors:
        
        Args:
            color: Plot line color.
        '''
        fig, ax = plt.subplots(facecolor='w', figsize=(25, 3))

        ax.plot(self._frame_times, self.values, color)
        ax.set_xlim(0, np.max(self._frame_times))
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('est. BOLD')
        ax.grid()
    
    def corr(self, other):
        '''Calculate correlation between two regressors.'''
        if not isinstance(other, self.__class__):
            msg = f'{other} should be of {self.__class__} type but is {type(other)}'
            raise TypeError(msg)
        return np.corrcoef(self.values.T, other.values.T)[0, 1]
    
    def __add__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError(f'cannot add regressor and {type(other)}')
        if not (self.frame_times == other.frame_times).all():
            raise ValueError('frame_times for added regressors does not match')    
        
        result = self.from_values(
            name=f'{self.name}+{other.name}',
            frame_times=self._frame_times,
            values=self._values+other._values
        )         
        return result
    
    def __mul__(self, other):
        if not isinstance(other, numbers.Real):
            raise TypeError(f'cannot multiply regressor and {type(other)}')

        result = self.from_values(
            name=f'{other}*{self.name}',
            frame_times=self._frame_times,
            values=self._values*other
        )    
        return result
        
    def __rmul__(self, other):
        return self * other
    
    def __sub__(self, other):
        new_name = f'{self.name}-{other.name}'
        result = self + (-1) * other
        result._name = new_name
        return result
    
    def __truediv__(self, other):
        return self * (1 / other)
    
    
def my_make_first_level_design_matrix(regressors: list):
    '''Turn arbitrary number of regressors into first level design matrix.
    
    This function wraps make_first_level_design_matrix function from 
    nistats.design_matrix module to create design matrix from list of Regressor
    objects. Note that this design matrix lacks confounds regressors. If you
    want to include confounds, pass it to the FirstLevelModel.fit method.

    Args:
        regressors: list of Regressor objects

    Returns (2-tuple):
        Final GLM design matrix as DataFrame and dictionary with condition
        contrast vectors for all specifified regressors.
    '''
    if not isinstance(regressors, list) or not regressors:
        raise TypeError('regressors should be a non-empty list')
    if not all(isinstance(reg, Regressor) for reg in regressors):
        raise TypeError(f'regressors should be a list of {Regressor}')
    if not all([(r.frame_times == regressors[0].frame_times).all() 
                for r in regressors]):
        raise ValueError('frame_times for all regressors should be equal')
    frame_times = regressors[0].frame_times
    
    # Filter empty regressors (i.e. miss regressor for subjects with no misses)
    regressors = [r for r in regressors if r.is_empty == False]

    # Combine regressors into dataframe
    joined_regs_names = [r.name for r in regressors]
    joined_regs = pd.DataFrame(
        data=np.hstack([r.values for r in regressors]), 
        index=frame_times,
        columns=joined_regs_names
    )

    # Compute design matrix
    dm = design_matrix.make_first_level_design_matrix(
        frame_times=frame_times,
        add_regs=joined_regs,
        add_reg_names=joined_regs_names
    )

    # Create condition vectors for all regressors of interest
    conditions = {r.name: np.zeros(dm.shape[1]) for r in regressors}
    for condition_name in conditions:
        conditions[condition_name][list(dm.columns).index(condition_name)] = 1

    return (dm, conditions)      

def convolve(signal, t_r=2, oversampling=50, hrf_model='spm'):
    '''Convolve signal with hemodynamic response function.
    
    Performs signal convolution with requested hrf model. This function wraps around nistats 
    compute_regressor function usually used for creating task-based regressors. The trick is to 
    define neural regressor as a sequence of equally spaced (with the gap of 1TR) and modulated
    'task events'. Event amplitude modulation corresponds to neural signal amplitude at a given 
    timepoint.
    
    Args:
        signal (iterable):
            Neural signal.
        t_r (float):
            Repetition time in seconds.
        oversampling (int, optional):
            Convolution upsampling rate.
        hrf_model (str, optional):
            Hemodynamic response function type. See the documentation of compute regressor function 
            from nistats.hemodynamic_models for more details.
            
    Returns:
        Convolved neural signal in BOLD space.
    '''
    n_volumes = len(signal)
    frame_times = np.arange(0, n_volumes * t_r, t_r)
    onsets = np.zeros((3, n_volumes))
    for vol, amplitude in enumerate(signal):
        onsets[:, vol] = (vol * t_r, 0, amplitude)

    signal_bold = hemodynamic_models.compute_regressor(
        onsets,
        hrf_model=hrf_model,                              
        frame_times=frame_times,
        oversampling=oversampling,     
        fir_delays=None)[0].ravel()

    return signal_bold