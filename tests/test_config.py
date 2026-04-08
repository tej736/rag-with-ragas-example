import unittest

from app.config import ProviderConfig, normalize_provider, validate_distributions


class TestConfig(unittest.TestCase):
    def test_provider_config_validates(self):
        cfg = ProviderConfig(provider="openai", model="gpt-4o")
        cfg.validate()

    def test_provider_config_rejects_invalid_provider(self):
        with self.assertRaises(ValueError):
            ProviderConfig(provider="invalid", model="x").validate()

    def test_normalize_provider(self):
        self.assertEqual(normalize_provider(" HuggingFace "), "huggingface")

    def test_validate_distributions_sum(self):
        validate_distributions({"simple": 0.5, "reasoning": 0.25, "multi_context": 0.25})
        with self.assertRaises(ValueError):
            validate_distributions({"simple": 0.6, "reasoning": 0.25, "multi_context": 0.25})


if __name__ == "__main__":
    unittest.main()
