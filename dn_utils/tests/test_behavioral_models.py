import unittest
import sys
import os
import numpy as np
import itertools

from dn_utils.behavioral_models import (load_behavioral_data, estimate_wbci_pd, 
    estimate_modulation)

class TestBehavioralModels(unittest.TestCase):

    def setUp(self):
        beh, meta = load_behavioral_data('../data/main_fmri_study/sourcedata/behavioral',
                verbose=False)
        self.beh = beh
        self.meta = meta

    # Reward condition
    def test_estimate_wbci_pd_ap_00_am_00_rew(self):
        wbci = estimate_wbci_pd(self.beh, self.meta, 0, condition=0, alpha_plus=0, alpha_minus=0)
        self.assertEqual(np.all(np.ones((110, 2)) * .5 == wbci), True)

    def test_estimate_wbci_pd_ap_10_am_05_rew(self):
        wbci = estimate_wbci_pd(self.beh, self.meta, 0, condition=0, alpha_plus=1, alpha_minus=0.5) 
        self.assertEqual(np.all(wbci[0] == [0.5, 0.5]), True)
        self.assertEqual(np.all(wbci[1] == [0.25, 0.75]), True)
        self.assertEqual(np.all(wbci[2] == [0.125, 0.875]), True)
        self.assertEqual(np.all(wbci[3] == [1, 0]), True)
    
    def test_estimate_wbci_pd_ap_05_am_1_rew(self):
        wbci = estimate_wbci_pd(self.beh, self.meta, 0, condition=0, alpha_plus=0.5, alpha_minus=1)
        self.assertEqual(np.all(wbci[0] == [0.5, 0.5]), True)
        self.assertEqual(np.all(wbci[1] == [0, 1]), True)
        self.assertEqual(np.all(wbci[2] == [0, 1]), True)
        self.assertEqual(np.all(wbci[3] == [0.5, 0.5]), True)

    def test_estimate_wbci_pd_ap_10_am_10_rew(self):
        wbci = estimate_wbci_pd(self.beh, self.meta, 0, condition=0, alpha_plus=1, alpha_minus=1)
        side_wbci = self.beh[0, 0, :, self.meta['dim4'].index('side_bci')]
        self.assertEqual(np.all(wbci[1:,1] == (side_wbci[:-1] == 1)), True)
    
    # Punishment condition
    def test_estimate_wbci_pd_ap_00_am_00_pun(self):
        wbci = estimate_wbci_pd(self.beh, self.meta, 0, condition=1, alpha_plus=0, alpha_minus=0)
        self.assertEqual(np.all(np.ones((110, 2)) * .5 == wbci), True)
    
    def test_estimate_wbci_pd_ap_10_am_05_pun(self):
        wbci = estimate_wbci_pd(self.beh, self.meta, 0, condition=1, alpha_plus=1, alpha_minus=0.5) 
        self.assertEqual(np.all(wbci[0] == [0.5, 0.5]), True)
        self.assertEqual(np.all(wbci[1] == [1, 0]), True)
        self.assertEqual(np.all(wbci[2] == [1, 0]), True)
        self.assertEqual(np.all(wbci[3] == [1, 0]), True)
    
    def test_estimate_wbci_pd_ap_05_am_1_pun(self):
        wbci = estimate_wbci_pd(self.beh, self.meta, 0, condition=1, alpha_plus=0.5, alpha_minus=1)
        self.assertEqual(np.all(wbci[0] == [0.5, 0.5]), True)
        self.assertEqual(np.all(wbci[1] == [0.75, 0.25]), True)
        self.assertEqual(np.all(wbci[2] == [1, 0]), True)
        self.assertEqual(np.all(wbci[3] == [1, 0]), True)
     
    def test_estimate_wbci_pd_ap_10_am_10_pun(self):
        wbci = estimate_wbci_pd(self.beh, self.meta, 0, condition=1, alpha_plus=1, alpha_minus=1)
        side_wbci = self.beh[0, 1, :, self.meta['dim4'].index('side_bci')]
        self.assertEqual(np.all(wbci[1:,1] == (side_wbci[:-1] == 1)), True)
        self.assertEqual(np.all(wbci[3] == [1, 0]), True)

    def test_estimate_modulation_perr(self):
        for sub in range(len(self.meta['dim1'])):
            for con in range(len(self.meta['dim2'])):
                for alpha_plus, alpha_minus in itertools.product([.01, .5, .99], repeat=2): 
 
                    won_bool = self.beh[sub, con, :, self.meta['dim4'].index('won_bool')]

                    wbci = estimate_wbci_pd(self.beh, self.meta, sub, con, alpha_plus, alpha_minus)
                    _, _, perr = estimate_modulation(self.beh, self.meta, sub, con, wbci)
                    # Discard zero prediction errors because they can be present in case of 
                    # loosing (miss) or winning (perfect prediction)
                    idx = np.nonzero(perr)

                    self.assertEqual(np.all(won_bool[idx] == (perr[idx] >= 0)), True)

if __name__ == '__main__':
    unittest.main()