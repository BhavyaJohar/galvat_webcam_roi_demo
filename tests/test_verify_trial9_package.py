import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("verify_trial9", ROOT / "tools/verify_trial9_package.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Profiles(unittest.TestCase):
    def setUp(self):
        self.profiles = [json.loads((ROOT / "config/trial9" / p).read_text())
                         for p in module.PROFILES.values()]

    def test_real_profiles(self):
        for p in self.profiles:
            self.assertEqual(len(module.validate_profile(p)), 45)
            self.assertEqual(p["modelToCanonicalIds"][20], 44)

    def test_profile_differences_preserved(self):
        nano, small = self.profiles
        def value(p, name):
            return next(c["threshold"] for c in p["classes"] if c["name"] == name)
        self.assertEqual(value(nano, "wallet"), .05)
        self.assertEqual(value(small, "wallet"), .525)
        self.assertEqual(value(nano, "measuring_spoon"), .25)
        self.assertEqual(value(small, "measuring_spoon"), .125)

    def test_bad_thresholds(self):
        for value in [True, -1, 2, float("nan"), "0.5"]:
            p = copy.deepcopy(self.profiles[0])
            p["classes"][0]["threshold"] = value
            with self.assertRaises(ValueError):
                module.validate_profile(p)

    def test_bad_mapping_or_nms(self):
        for field, value in [("modelToCanonicalIds", list(range(45))),
                             ("applyAdditionalNms", True), ("referenceIou", .5)]:
            p = copy.deepcopy(self.profiles[0]); p[field] = value
            with self.assertRaises(ValueError):
                module.validate_profile(p)

    def test_bad_cardinality_and_ids(self):
        for boolean in [False, True]:
            p = copy.deepcopy(self.profiles[0])
            if boolean: p["classes"][1]["canonicalId"] = 0
            else: p["classes"].pop()
            with self.assertRaises(ValueError):
                module.validate_profile(p)

    def test_candidate_map_is_name_only_and_not_authenticated(self):
        candidate = json.loads((ROOT / "config/comparison/candidate_43_class_map.json").read_text())
        report = module.compare_candidate_map(candidate, self.profiles[0])
        self.assertEqual(report["shared_class_count"], 43)
        self.assertEqual(report["trial9_only_classes"], ["exit_sign", "measuring_spoon"])
        self.assertFalse(report["model_identity_verified"])
        pan = next(r for r in report["shared_mapping_by_name"] if r["name"] == "pan")
        self.assertEqual((pan["candidate_id"], pan["trial9_model_id"], pan["galvat_canonical_id"]), (20,22,14))

    def test_duplicate_candidate_name_rejected(self):
        candidate = {"identity":"unverified", "classes":[{"id":0,"name":"pan"},{"id":1,"name":"pan"}]}
        with self.assertRaises(ValueError):
            module.compare_candidate_map(candidate, self.profiles[0])


if __name__ == "__main__":
    unittest.main()
