import numpy as np
import pandas as pd

import unittest
import itertools
import operator

import sys
print(sys.path)

from nistats import design_matrix
from dn_utils.glm_utils import Regressor, my_make_first_level_design_matrix


class TestRegressor(unittest.TestCase):
    
    def test_correct_initialization(self):
        frame_times = np.arange(10) * 2
        # no events are specified
        Regressor('test', frame_times, [])
        Regressor('test', frame_times, np.array([]))
        # different event types
        Regressor('test', frame_times, [1, 2])
        Regressor('test', frame_times, np.array([1, 2]))
        # modulation specified
        Regressor('test', frame_times, [1, 2], modulation=[-1, 1])
        Regressor('test', frame_times, [1, 2], modulation=np.array([-1, 1]))
        Regressor('test', frame_times, np.array([1, 2]), 
                  modulation=[-1, 1])
        Regressor('test', frame_times, np.array([1, 2]), 
                  modulation=np.array([-1, 1]))
        # duration specified
        Regressor('test', frame_times, [1, 2], duration=[0.4, 0.6])
        Regressor('test', frame_times, [1, 2], duration=np.array([0.4, 0.6]))
        Regressor('test', frame_times, np.array([1, 2]), 
                  duration=[0.4, 0.6])
        Regressor('test', frame_times, np.array([1, 2]), 
                  duration=np.array([0.4, 0.6]))
        # both duration and modulation specified
        for onset, duration, modulation in itertools.product(([1, 2], 
                                                              np.array([1, 2])), 
                                                             repeat=3):
            Regressor('test', frame_times, onset, 
                      duration=duration, modulation=modulation)
            
    def test_incorrect_initialization_optional_arguments_as_positional(self):
        frame_times = np.arange(10) * 2
        # modulation and duration must be keyword arguments
        with self.assertRaises(TypeError):
            Regressor('test', frame_times, [0], [0])
            
    def test_incorrect_initialization_for_mandatory_arguments(self):
        frame_times = np.arange(10) * 2
        # no frame times and onset
        with self.assertRaises(TypeError):
            Regressor('test')
        # no onset
        with self.assertRaises(TypeError):
            Regressor('test', frame_times)
        # frame_times is not np.array    
        with self.assertRaises(TypeError):
            Regressor('test', [0, 2, 4, 6, 8, 10], [0])          
        # onset type is not array-like
        with self.assertRaises(TypeError):
            Regressor('test', frame_times, 0)   
            
    def test_incorrect_initialization_dimension_mismatch(self):
        frame_times = np.arange(10) * 2
        for v1, v2, v3 in itertools.permutations(([0, 2], 
                                                  np.array([0, 2]), 
                                                  (0, 2, 4))):
            with self.assertRaises(ValueError):
                Regressor('test', frame_times, v1, modulation=v2, duration=v3)   
                
    def test_incorrect_initialization_too_many_dimensions(self):
        # onset, modulation and duration should have single dimension, e.g. 
        # (3, ) and not (3, 1) or (1, 3)
        frame_times = np.arange(10) * 2
        a2v = np.array([1, 2, 3])[:, np.newaxis]
        a2h = np.array([1, 2, 3])[np.newaxis, :]
        for v1, v2, v3 in itertools.product((a2v, a2h), repeat=3):
            # this is single correct exception
            if v1.shape != (1, 3) and v2.shape != (1, 3) and v3.shape != (1, 3):
                with self.assertRaises(ValueError):
                    Regressor('test', frame_times, v1, 
                              modulation=v2, duration=v3)
                    
    def test_correct_values_no_duration_no_modulation(self):
        frame_times = np.arange(10) * 2
        reg = Regressor('test', frame_times, [0])
        values = np.array([[ 0.00000000e+00],
                           [ 1.63857515e-03],
                           [ 7.44648838e-03],
                           [ 7.75615867e-03],
                           [ 4.38271652e-03],
                           [ 1.56898270e-03],
                           [ 4.39318536e-05],
                           [-6.11118379e-04],
                           [-7.49595149e-04],
                           [-6.20977746e-04]])
        self.assertTrue(np.isclose(reg.values, values).all())
        self.assertTrue((reg.frame_times == frame_times).all())
        self.assertEqual(reg.name, 'test')
        self.assertEqual(len(reg), 10)
        
    def test_correct_values_with_duration(self):
        frame_times = np.arange(10) * 2
        reg = Regressor('test', frame_times, [0], duration=[.25])
        values = np.array([[ 0.        ],
                           [ 0.00947523],
                           [ 0.05035788],
                           [ 0.05529071],
                           [ 0.03214266],
                           [ 0.01189756],
                           [ 0.00075046],
                           [-0.00412143],
                           [-0.00525569],
                           [-0.00442661]])
        self.assertTrue(np.isclose(reg.values, values).all())
        self.assertTrue((reg.frame_times == frame_times).all())
        self.assertEqual(reg.name, 'test')
        self.assertEqual(len(reg), 10)     
        
    def test_correct_values_with_modulation(self):
        frame_times = np.arange(10) * 2
        reg = Regressor('test', frame_times, [0], modulation=[-1])
        values = np.array([[ 0.00000000e+00],
                           [-1.63857515e-03],
                           [-7.44648838e-03],
                           [-7.75615867e-03],
                           [-4.38271652e-03],
                           [-1.56898270e-03],
                           [-4.39318536e-05],
                           [ 6.11118379e-04],
                           [ 7.49595149e-04],
                           [ 6.20977746e-04]])
        self.assertTrue(np.isclose(reg.values, values).all())
        self.assertTrue((reg.frame_times == frame_times).all())
        self.assertEqual(reg.name, 'test')
        self.assertEqual(len(reg), 10)     
        
    def test_correct_values_with_duration_and_modulation(self):
        frame_times = np.arange(10) * 2
        reg = Regressor('test', frame_times, [0], 
                        modulation=[-1], duration=[.25])
        values = np.array([[ 0.        ],
                           [-0.00947523],
                           [-0.05035788],
                           [-0.05529071],
                           [-0.03214266],
                           [-0.01189756],
                           [-0.00075046],
                           [ 0.00412143],
                           [ 0.00525569],
                           [ 0.00442661]])
        self.assertTrue(np.isclose(reg.values, values).all())
        self.assertTrue((reg.frame_times == frame_times).all())
        self.assertEqual(reg.name, 'test')
        self.assertEqual(len(reg), 10)
        
    def test_from_values_constructor(self):
        frame_times = np.arange(10) * 2
        np.random.seed(0)
        values = np.random.random(10)
        reg = Regressor.from_values('test', frame_times, values)
        self.assertTrue((frame_times == reg.frame_times).all())
        self.assertTrue((values[:, np.newaxis] == reg.values).all())
        self.assertEqual('test', reg.name)
        
    def test_is_empty(self):
        frame_times = np.arange(10) * 2
        self.assertTrue(Regressor('test', frame_times, []).is_empty)
        # onset begins after last frame time
        self.assertTrue(Regressor('test', frame_times, [20]).is_empty)
        self.assertFalse(Regressor('test', frame_times, [0]).is_empty)
        self.assertTrue(
            Regressor.from_values('test', frame_times, np.zeros(10)).is_empty)
        self.assertTrue(
            Regressor.from_values('test', frame_times, 
                                  [0 for _ in range(10)]).is_empty)
        self.assertFalse(
            Regressor.from_values('test', frame_times, 
                                  [0.0001] + [0 for _ in range(9)]).is_empty)
        
    def test_adding_regressors(self):
        frame_times = np.arange(10) * 2
        np.random.seed(0)
        values1 = np.random.random(10)
        values2 = np.random.random(10)
        reg = (Regressor.from_values('test1', frame_times, values1) +
               Regressor.from_values('test2', frame_times, values2)) 
        self.assertTrue((frame_times == reg.frame_times).all())
        self.assertTrue((reg.values == (values1 + values2)[:, np.newaxis]).all())
        self.assertEqual(reg.name, 'test1+test2')
        
    def test_adding_incorrect_types(self):
        frame_times = np.arange(10) * 2
        reg = Regressor.from_values('test', frame_times, np.arange(10)-4.5)
        with self.assertRaises(TypeError):
            reg + np.random.random(10)
        with self.assertRaises(TypeError):
            reg + [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]   
        class SomeClass:
            values = np.random.random(10)
        with self.assertRaises(TypeError):
            reg + SomeClass()
        
    def test_adding_frame_times_does_not_match(self):
        frame_times1 = np.arange(10) * 2
        frame_times2 = np.arange(10) * 2.5
        np.random.seed(0)
        reg1 = Regressor.from_values('test', frame_times1, np.random.random(10))
        reg2 = Regressor.from_values('test', frame_times2, np.random.random(10))
        with self.assertRaises(ValueError):
            reg1 + reg2
        
    def test_subtracting_regressors(self):
        frame_times = np.arange(10) * 2
        np.random.seed(0)
        values1 = np.random.random(10)
        values2 = np.random.random(10)
        reg = (Regressor.from_values('test1', frame_times, values1) -
               Regressor.from_values('test2', frame_times, values2)) 
        self.assertTrue((frame_times == reg.frame_times).all())
        self.assertTrue((reg.values == (values1 - values2)[:, np.newaxis]).all())
        self.assertEqual(reg.name, 'test1-test2')
        
    def test_multiplication(self):
        frame_times = np.arange(10) * 2
        np.random.seed(0)
        values = np.random.random(10)
        for scalar in (-2, -1, -0.99, 0, .99, 1, 2):
            # right handed __mul__ is also tested here
            reg = scalar * Regressor.from_values('test1', frame_times, values)
            self.assertTrue((frame_times == reg.frame_times).all())
            self.assertTrue((reg.values == (values * scalar)[:, np.newaxis]).all())
            self.assertEqual(reg.name, f'{scalar}*test1')
            
    def test_multiplication_incorrect_types(self):
        frame_times = np.arange(10) * 2
        reg = Regressor.from_values('test', frame_times, np.random.random(10))
        with self.assertRaises(TypeError):
            reg * np.random.random(10)
        with self.assertRaises(TypeError):
            reg * '3'
        with self.assertRaises(TypeError):
            reg * reg
            
    def test_simple_linear_combination(self):
        frame_times = np.arange(10) * 2
        np.random.seed(0)
        values1 = np.random.random(10)
        values2 = np.random.random(10)
        for a, b in itertools.product((-2, -1, -0.99, 0, .99, 1, 2), repeat=2):
            for fn in [operator.add, operator.sub]:
                reg = fn(a*Regressor.from_values('test1', frame_times, values1),
                         b*Regressor.from_values('test2', frame_times, values2)) 
                self.assertTrue((frame_times == reg.frame_times).all())
                true_values = fn(a * values1, b * values2)
                self.assertTrue((reg.values ==  true_values[:, np.newaxis]).all())
                sign = '+' if fn == operator.add else '-'
                self.assertEqual(reg.name, f'{a}*test1{sign}{b}*test2')
        
    def test_division(self):
        frame_times = np.arange(10) * 2
        np.random.seed(0)
        values = np.random.random(10)
        for scalar in (-2, -1, -0.5, .5, 1, 2):
            reg = Regressor.from_values('test1', frame_times, values) / scalar
            self.assertTrue((frame_times == reg.frame_times).all())
            self.assertTrue((reg.values == (values / scalar)[:, np.newaxis]).all())
            self.assertEqual(reg.name, f'{1/scalar}*test1')
            
    def test_corr_method(self):
        frame_times = np.arange(10) * 2
        np.random.seed(0)
        values1 = np.random.random(10)
        values2 = np.random.random(10)
        reg1 = Regressor.from_values('test1', frame_times, values1)
        reg2 = Regressor.from_values('test2', frame_times, values2)
        self.assertEqual(reg1.corr(reg2), np.corrcoef(values1, values2)[0, 1])
        self.assertEqual(reg2.corr(reg1), np.corrcoef(values1, values2)[0, 1])
        
    def test_regressor_is_deameaned(self):
        frame_times = np.arange(100) * 2
        # modulation should be demaned when
        # modulator has more than one event and event values are not equal
        for modulation in ([1, 2], [-1, -2], [-1, 2], [-2, 1]):
            reg = Regressor('test', frame_times, 
                            onset=[0, 100], modulation=modulation)
            self.assertTrue(np.mean(reg.values) < 10**(-15))
            
    def test_regressor_is_not_demeaned(self):
        frame_times = np.arange(100) * 2
        reg1 = Regressor('test', frame_times, onset=[0, 100], 
                         modulation=[1, 1])
        reg2 = Regressor('test', frame_times, onset=[0, 100], 
                         modulation=[-.99, -.99])
        reg3 = Regressor('test', frame_times, onset=[100], 
                         modulation=[50])
        self.assertTrue(np.mean(reg1.values) > 10**(-15))
        self.assertTrue(np.mean(reg2.values) > 10**(-15))
        self.assertTrue(np.mean(reg3.values) > 10**(-15))

    def test_alternative_constructor_from_values(self):
        frame_times = np.arange(100) * 2
        reg1 = Regressor('test', frame_times, onset=[0])
        reg2 = Regressor.from_values(
            'test_from_values',
            frame_times=np.arange(100)*2, 
            values=np.zeros(100))
        self.assertEqual(reg1.values.shape, reg2.values.shape)


class TestMyMakeFirstLevelDesignMatrix(unittest.TestCase):
    
    def test_correct_initialization(self):
        frame_times = np.arange(10) * 2
        reg1 = Regressor('test1', frame_times, onset=[0])
        reg2 = Regressor('test2', frame_times, onset=[2])
        my_make_first_level_design_matrix([reg1])
        my_make_first_level_design_matrix([reg1, reg2])
        
    def test_incorrect_initialization(self):
        class DummyRegressor:
            frame_times = np.arange(100)
            values = np.arange(10)
            name = 'dummy'
        with self.assertRaises(TypeError):
            my_make_first_level_design_matrix([])
        with self.assertRaises(TypeError):
            my_make_first_level_design_matrix([[1, 2, 3], [-1, 1, 0]])
        with self.assertRaises(TypeError):
            my_make_first_level_design_matrix([DummyRegressor()])
            
    def test_frame_times_not_matching_error(self):
        frame_times1 = np.arange(10) * 2
        frame_times2 = np.arange(10) * 3
        reg1 = Regressor('test1', frame_times1, onset=[0])
        reg2 = Regressor('test2', frame_times2, onset=[2])
        with self.assertRaises(ValueError):
            my_make_first_level_design_matrix([reg1, reg2])
            
    def test_names_collision_error(self):
        frame_times = np.arange(10) * 2
        reg1 = Regressor('test', frame_times, onset=[0])
        reg2 = Regressor('test', frame_times, onset=[2])
        with self.assertRaises(ValueError):
            my_make_first_level_design_matrix([reg1, reg2])
            
    def test_empty_regressors_are_removed(self):
        frame_times = np.arange(10) * 2
        reg_empty1 = Regressor('empty1', frame_times, onset=[])
        reg_empty2 = Regressor('empty2', frame_times, onset=[])
        reg3 = Regressor('test3', frame_times, onset=[0])
        regressors = [reg_empty1, reg_empty2, reg3]
        dm, conditions = my_make_first_level_design_matrix(regressors)
        self.assertTrue('empty1' not in dm.columns)
        self.assertTrue('empty1' not in conditions)
        self.assertTrue('empty2' not in dm.columns)
        self.assertTrue('empty2' not in conditions)
        self.assertTrue('test3' in dm.columns)
        self.assertTrue('test3' in conditions)
        
    def test_all_regressors_are_empty_error(self):
        frame_times = np.arange(10) * 2
        reg_empty1 = Regressor('empty1', frame_times, onset=[])
        with self.assertRaises(ValueError):
            my_make_first_level_design_matrix([reg_empty1])
            
    def test_simple_design_matrix(self):
        reg = Regressor(name='test', frame_times=np.arange(10) * 2, onset=[0])
        dm, conditions = my_make_first_level_design_matrix([reg])
        # create true design matrix
        events = pd.DataFrame(columns=['onset', 'duration', 'trial_type'], 
                              data=[[0, 0, 'test']])
        dm_true = design_matrix.make_first_level_design_matrix(
            frame_times=np.arange(10) * 2,
            events=events,
            hrf_model='spm')
        self.assertTrue(dm_true.equals(dm))
        self.assertEqual(len(conditions), 1)
        self.assertTrue((conditions['test'] ==  np.array([1, 0])).all())
        
    def test_complex_design_matrix_with_duration(self):
        frame_times = np.arange(100) * 2
        reg1 = Regressor('test1', frame_times, onset=[0, 100])
        reg2 = Regressor('test2', frame_times, onset=[50, 150], duration=[.1, .1])
        dm, conditions = my_make_first_level_design_matrix([reg1, reg2])
        # true design matrix
        events = pd.DataFrame(
            columns=['onset', 'duration', 'trial_type'], 
            data=[[0, 0, 'test1'], [100, 0, 'test1'], 
                  [50, 0.1, 'test2'], [150, 0.1, 'test2']])
        dm_true = design_matrix.make_first_level_design_matrix(
            frame_times=np.arange(100) * 2,
            events=events,
            hrf_model='spm')
        self.assertTrue(dm_true.equals(dm))
        self.assertEqual({'test1', 'test2'}, set(conditions.keys()))
        self.assertEqual(conditions['test1'][0], 1)
        self.assertEqual(np.sum(conditions['test1']), 1)
        self.assertEqual(conditions['test2'][1], 1)
        self.assertEqual(np.sum(conditions['test2']), 1)
        
    def test_design_matrix_with_duration_and_modulation(self):
        frame_times = np.arange(10) * 2
        reg = Regressor('test', frame_times, onset=[0, 10], 
                        duration=[.2, .3], modulation=[2, 4])
        dm, conditions = my_make_first_level_design_matrix([reg])
        # true design matrix (used demeaned modulation)
        events = pd.DataFrame(
            columns=['onset', 'duration', 'trial_type', 'modulation'], 
            data=[[0, .2, 'test', -1], [10, .3, 'test', 1]]) 
        dm_true = design_matrix.make_first_level_design_matrix(
            frame_times=np.arange(10) * 2,
            events=events,
            hrf_model='spm')
        self.assertTrue(dm_true.equals(dm))
        self.assertEqual(len(conditions), 1)
        self.assertTrue((conditions['test'] ==  np.array([1, 0])).all())
        
        
if __name__ == '__main__':
    unittest.main()