"""
PluginSystem — Extensible plugin architecture for enterprise customization.

Allows:
  - enterprise rules
  - custom classifiers
  - graph providers
  - policy extensions
  - optimization middleware
  - governance plugins

Plugin types:
  - ClassifierPlugin: adds custom classification signals
  - OptimizerPlugin: middleware applied before/after optimization
  - PolicyPlugin: custom governance rules
  - GraphPlugin: alternative dependency discovery
  - TelemetryPlugin: custom telemetry backends
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from llm_prompt_optimizer.utils.helpers import get_logger

logger = get_logger(__name__)


# ── Base plugin interfaces ────────────────────────────────────────────────────

class BasePlugin(ABC):
    """All plugins must extend this base."""

    name: str = "unnamed_plugin"
    version: str = "0.0.1"
    plugin_type: str = "generic"

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None: ...

    def teardown(self) -> None:
        """Optional cleanup hook."""
        pass


class ClassifierPlugin(BasePlugin):
    """Adds custom signals to the prompt classifier."""
    plugin_type = "classifier"

    @abstractmethod
    def score(self, text: str) -> Dict[str, float]:
        """Return {category_name: score} additions."""
        ...


class OptimizerPlugin(BasePlugin):
    """Middleware applied to the optimization pipeline."""
    plugin_type = "optimizer"

    def pre_optimize(self, prompt_text: str, context: Dict[str, Any]) -> str:
        """Called before optimization. May transform the prompt."""
        return prompt_text

    def post_optimize(self, optimized_text: str, context: Dict[str, Any]) -> str:
        """Called after optimization. May post-process the result."""
        return optimized_text


class PolicyPlugin(BasePlugin):
    """Custom governance policy rules."""
    plugin_type = "policy"

    @abstractmethod
    def evaluate(self, context: Dict[str, Any]) -> List[str]:
        """Return list of violation messages, or empty list if clean."""
        ...


class GraphPlugin(BasePlugin):
    """Alternative dependency graph provider."""
    plugin_type = "graph"

    @abstractmethod
    def available(self) -> bool: ...

    @abstractmethod
    def discover(self, intent_lock: Any, max_depth: int) -> List[Any]: ...


class TelemetryPlugin(BasePlugin):
    """Custom telemetry backend."""
    plugin_type = "telemetry"

    @abstractmethod
    def handle_event(self, event: Any) -> None: ...


# ── Registry ──────────────────────────────────────────────────────────────────

@dataclass
class PluginRegistration:
    plugin: BasePlugin
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)


class PluginRegistry:
    """Central registry for all loaded plugins."""

    def __init__(self) -> None:
        self._plugins: List[PluginRegistration] = []

    def register(
        self, plugin: BasePlugin, config: Optional[Dict[str, Any]] = None
    ) -> None:
        config = config or {}
        try:
            plugin.initialize(config)
            self._plugins.append(PluginRegistration(plugin=plugin, config=config))
            logger.info(
                f"Plugin registered: {plugin.name} [{plugin.plugin_type}] v{plugin.version}"
            )
        except Exception as e:
            logger.error(f"Plugin '{plugin.name}' failed to initialize: {e}")

    def get_by_type(self, plugin_type: str) -> List[BasePlugin]:
        return [
            r.plugin
            for r in self._plugins
            if r.plugin.plugin_type == plugin_type and r.enabled
        ]

    def get_classifiers(self) -> List[ClassifierPlugin]:
        return [p for p in self.get_by_type("classifier") if isinstance(p, ClassifierPlugin)]

    def get_optimizers(self) -> List[OptimizerPlugin]:
        return [p for p in self.get_by_type("optimizer") if isinstance(p, OptimizerPlugin)]

    def get_policies(self) -> List[PolicyPlugin]:
        return [p for p in self.get_by_type("policy") if isinstance(p, PolicyPlugin)]

    def get_graph_providers(self) -> List[GraphPlugin]:
        return [p for p in self.get_by_type("graph") if isinstance(p, GraphPlugin)]

    def get_telemetry(self) -> List[TelemetryPlugin]:
        return [p for p in self.get_by_type("telemetry") if isinstance(p, TelemetryPlugin)]

    def disable(self, name: str) -> None:
        for reg in self._plugins:
            if reg.plugin.name == name:
                reg.enabled = False
                logger.info(f"Plugin disabled: {name}")

    def list_plugins(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": r.plugin.name,
                "type": r.plugin.plugin_type,
                "version": r.plugin.version,
                "enabled": r.enabled,
            }
            for r in self._plugins
        ]


# ── Loader ────────────────────────────────────────────────────────────────────

class PluginLoader:
    """Dynamically loads plugins from directories or module paths."""

    def __init__(self, registry: PluginRegistry) -> None:
        self.registry = registry

    def load_from_directory(self, directory: str, config: Optional[Dict[str, Any]] = None) -> int:
        """
        Scan a directory for plugin modules.
        Any module exposing a `register_plugin(registry)` function will be loaded.
        Returns number of plugins loaded.
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            logger.warning(f"Plugin directory not found: {directory}")
            return 0

        loaded = 0
        for py_file in dir_path.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            try:
                spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)  # type: ignore
                    if hasattr(module, "register_plugin"):
                        module.register_plugin(self.registry)
                        loaded += 1
                        logger.info(f"Loaded plugin from: {py_file.name}")
            except Exception as e:
                logger.error(f"Failed to load plugin from {py_file.name}: {e}")

        return loaded

    def load_from_module(self, module_path: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """Load a plugin from a dotted module path, e.g. 'mycompany.plugins.custom_policy'."""
        try:
            module = importlib.import_module(module_path)
            if hasattr(module, "register_plugin"):
                module.register_plugin(self.registry)
                return True
        except ImportError as e:
            logger.error(f"Cannot import plugin module '{module_path}': {e}")
        return False


# ── System façade ─────────────────────────────────────────────────────────────

class PluginSystem:
    """
    Top-level plugin system façade.
    Manages the registry, loader, and plugin lifecycle.
    """

    def __init__(self) -> None:
        self.registry = PluginRegistry()
        self.loader = PluginLoader(self.registry)

    def load_directory(self, directory: str) -> int:
        return self.loader.load_from_directory(directory)

    def load_module(self, module_path: str) -> bool:
        return self.loader.load_from_module(module_path)

    def register(self, plugin: BasePlugin, config: Optional[Dict[str, Any]] = None) -> None:
        self.registry.register(plugin, config)

    def apply_optimizer_plugins_pre(
        self, text: str, context: Dict[str, Any]
    ) -> str:
        for plugin in self.registry.get_optimizers():
            text = plugin.pre_optimize(text, context)
        return text

    def apply_optimizer_plugins_post(
        self, text: str, context: Dict[str, Any]
    ) -> str:
        for plugin in self.registry.get_optimizers():
            text = plugin.post_optimize(text, context)
        return text

    def evaluate_policy_plugins(self, context: Dict[str, Any]) -> List[str]:
        violations: List[str] = []
        for plugin in self.registry.get_policies():
            violations.extend(plugin.evaluate(context))
        return violations

    def get_best_graph_provider(self) -> Optional[GraphPlugin]:
        providers = [p for p in self.registry.get_graph_providers() if p.available()]
        return providers[0] if providers else None

    def list_plugins(self) -> List[Dict[str, Any]]:
        return self.registry.list_plugins()

    def teardown_all(self) -> None:
        for reg in self.registry._plugins:
            try:
                reg.plugin.teardown()
            except Exception:
                pass
