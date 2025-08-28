import unittest


class NamespacePackagesTest(unittest.TestCase):

    def test_both_importable(self):
        import nspkg
        import nspkg.subpkg1
        import nspkg.subpkg1.subpkgmod
        import nspkg.subpkg2.subpkgmod

        self.assertEqual("nspkg.subpkg1", nspkg.subpkg1.expected_name)
        self.assertEqual(
            "nspkg.subpkg1.subpkgmod", nspkg.subpkg1.subpkgmod.expected_name
        )

        self.assertEqual("nspkg.subpkg2", nspkg.subpkg2.expected_name)
        self.assertEqual(
            "nspkg.subpkg2.subpkgmod", nspkg.subpkg2.subpkgmod.expected_name
        )


    def test_legacy_packages_importable(self):
        import legacy_main
        import legacy_main.subpkg1
        import legacy_main.subpkg1.subpkgmod

        import legacy_side
        import legacy_side.subpkg1
        import legacy_side.subpkg1.subpkgmod

        self.assertEqual("legacy_main.subpkg1", legacy_main.subpkg1.expected_name)
        self.assertEqual(
            "legacy_main.subpkg1.subpkgmod", legacy_main.subpkg1.subpkgmod.expected_name
        )
        self.assertEqual("legacy_side.subpkg1", legacy_side.subpkg1.expected_name)
        self.assertEqual(
            "legacy_side.subpkg1.subpkgmod", legacy_side.subpkg1.subpkgmod.expected_name
        )


if __name__ == "__main__":
    unittest.main()
