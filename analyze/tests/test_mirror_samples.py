"""Check that sample expansion preserves the previous causal experiment."""
import unittest

import numpy as np

from analyze.analyze_mirror_prior import virtual_node_forecast
from analyze.analyze_mirror_range import correction_operator, make_splits, observed_nodes
from analyze.analyze_mirror_samples import original_conditions, quartiles


class MirrorSampleTests(unittest.TestCase):
    def test_same_144_forecast_conditions_per_trajectory(self):
        query = original_conditions()
        self.assertEqual(query.shape, (144, 3))
        np.testing.assert_array_equal(np.unique(query[:, 0]), [1])
        np.testing.assert_array_equal(np.unique(query[:, 1]), np.arange(4, 40))
        np.testing.assert_array_equal(np.unique(query[:, 2]), [1, 2, 4, 8])
        self.assertEqual(int((query[:, 1]+query[:, 2]).max()), 47)

    def test_operator_matches_previous_virtual_node_implementation(self):
        rng = np.random.default_rng(7)
        logs = rng.normal(size=(4, 48))
        prior = rng.normal(size=48)
        steps = np.arange(1, 49)
        query = original_conditions()
        for degree in (1, 2, 3):
            for mode, weight in (("local", 0.), ("geometry", .1), ("mirror", .1)):
                operator = correction_operator(query, steps, 26.4, degree=degree,
                                               window=5, ridge=.01, weight=weight, mode=mode)
                actual = prior[query[:, 1]+query[:, 2]]+(logs-prior) @ operator.T
                expected = virtual_node_forecast(logs, prior, steps, np.arange(4, 40), 26.4,
                    degree=degree, window=5, ridge=.01, mirror_weight=weight,
                    virtual_target="repeat_anchor" if mode == "geometry" else "mirror")
                np.testing.assert_allclose(actual, expected.reshape(4, -1), atol=1e-11)

    def test_operators_cannot_read_future_or_unobserved_history(self):
        steps = np.arange(1, 49)
        query = np.array([(stride, anchor, 4) for stride in (1, 2, 4, 8) for anchor in (4, 20, 29, 39)])
        for mode, weight in (("local", 0.), ("geometry", 1.), ("mirror", 1.)):
            operator = correction_operator(query, steps, 26.4, degree=3, ridge=.01,
                                           weight=weight, mode=mode)
            for row, (stride, anchor, _) in zip(operator, query):
                forbidden = np.setdiff1d(np.arange(48), observed_nodes(anchor, stride))
                np.testing.assert_array_equal(row[forbidden], np.zeros(len(forbidden)))

    def test_each_partition_tests_all_500_exactly_once(self):
        splits = make_splits()
        for protocol in ("random_1", "random_2", "prompt_order"):
            seen = []
            for split in (s for s in splits if s["protocol"] == protocol):
                self.assertEqual([len(split[k]) for k in ("train", "validation", "test")], [300, 100, 100])
                train, val, test = (set(split[k]) for k in ("train", "validation", "test"))
                self.assertFalse(train & val or train & test or val & test)
                self.assertEqual(train | val | test, set(range(500)))
                seen.extend(split["test"])
            np.testing.assert_array_equal(np.sort(seen), np.arange(500))

    def test_diagnostic_quartiles_cover_all_samples_even_with_ties(self):
        groups = quartiles(np.ones(500))
        np.testing.assert_array_equal(np.bincount(groups), [125]*4)


if __name__ == "__main__":
    unittest.main()
