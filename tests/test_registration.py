"""Tests for jSACSMA plugin registration with symfluence."""

import pytest


class TestEntryPoint:
    """Test that the entry point is properly configured."""

    def test_entry_point_discoverable(self):
        """The 'sacsma' entry point should be discoverable."""
        from importlib.metadata import entry_points

        eps = entry_points(group='symfluence.plugins')
        names = [ep.name for ep in eps]
        assert 'sacsma' in names, f"'sacsma' not found in plugins: {names}"

    def test_entry_point_loads(self):
        """The entry point should load the register function."""
        from importlib.metadata import entry_points

        eps = entry_points(group='symfluence.plugins')
        sacsma_ep = next(ep for ep in eps if ep.name == 'sacsma')
        register_fn = sacsma_ep.load()
        assert callable(register_fn)

    def test_register_function_exists(self):
        """jsacsma.register should be importable."""
        from jsacsma import register
        assert callable(register)


class TestRegistration:
    """Test that registration populates the registry."""

    def test_register_populates_config_adapter(self):
        """After register(), SACSMA config adapter should be in registry."""
        from jsacsma import register
        register()

        from symfluence.core.registries import R
        assert 'SACSMA' in R.config_adapters

    def test_register_populates_runner(self):
        """After register(), SACSMA runner should be in registry."""
        from symfluence.core.registries import R
        assert 'SACSMA' in R.runners

    def test_register_populates_preprocessor(self):
        """After register(), SACSMA preprocessor should be in registry."""
        from symfluence.core.registries import R
        assert 'SACSMA' in R.preprocessors

    def test_register_populates_postprocessor(self):
        """After register(), SACSMA postprocessor should be in registry."""
        from symfluence.core.registries import R
        assert 'SACSMA' in R.postprocessors


class TestLazyImports:
    """Test that lazy imports work correctly."""

    def test_import_simulate(self):
        from jsacsma import simulate
        assert callable(simulate)

    def test_import_parameters(self):
        from jsacsma import DEFAULT_PARAMS, PARAM_BOUNDS
        assert len(PARAM_BOUNDS) == 26
        assert len(DEFAULT_PARAMS) == 26

    def test_import_sacsma_state(self):
        from jsacsma import SacSmaState
        assert SacSmaState is not None

    def test_import_snow17_from_jsacsma(self):
        """Snow-17 types should be accessible via jsacsma lazy imports."""
        from jsacsma import Snow17Params, Snow17State
        assert Snow17Params is not None
        assert Snow17State is not None
