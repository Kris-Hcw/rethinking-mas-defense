"""Run the core tests supported by the public checkout alone."""

import sys
import unittest


PUBLIC_TEST_MODULES = (
    "tests.test_confidence",
    "tests.test_data",
    "tests.test_embeddings",
    "tests.test_llm_client",
    "tests.test_mas",
    "tests.test_runtime_paths",
    "tests.test_topology",
)


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromNames(PUBLIC_TEST_MODULES)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.wasSuccessful():
        print(f"PUBLIC_CHECKOUT_TESTS_OK ({result.testsRun}/{result.testsRun})")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
