import numpy as np
import unittest
from execute_heuristics import ExecuteHeuristic

class TestSolveTime(unittest.TestCase):
    def setUp(self):
        self.eh = ExecuteHeuristic("dummy", "", "", "")

    def test_solve_time_basic(self):
        times = np.array([0.0, 1.0, 2.0, 3.5])
        objs = np.array([10.0, 5.0, 2.0, 1.0])
        # threshold exactly on one of the recorded objectives
        thr = 2.0
        self.assertEqual(self.eh._solve_time(times, objs, thr, 10.0), 2.0)
        # threshold below all objectives: return limit
        thr = 0.5
        self.assertEqual(self.eh._solve_time(times, objs, thr, 10.0), 10.0)
        # threshold above first objective: return first time
        thr = 15.0
        self.assertEqual(self.eh._solve_time(times, objs, thr, 10.0), 0.0)

    def test_solve_time_with_duplicates(self):
        times = np.array([0, 1, 1, 2])
        objs = np.array([5, 3, 1, 0])
        thr = 1
        # should pick the first occurrence where objs <= thr, which is at time 1
        self.assertEqual(self.eh._solve_time(times, objs, thr, 5), 1)

if __name__ == '__main__':
    unittest.main()
