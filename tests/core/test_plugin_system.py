"""Tests for the plugin system."""

import pytest
from typing import Dict, Any, List

from llm_prompt_optimizer.plugins.system import (
    PluginSystem, BasePlugin, OptimizerPlugin, PolicyPlugin, ClassifierPlugin
)


class EchoOptimizerPlugin(OptimizerPlugin):
    name = "echo_optimizer"
    version = "0.1.0"

    def initialize(self, config: Dict[str, Any]) -> None:
        self.prefix = config.get("prefix", "[TEST]")

    def pre_optimize(self, text: str, context: Dict) -> str:
        return f"{self.prefix} {text}"

    def post_optimize(self, text: str, context: Dict) -> str:
        return text + "\n<!-- optimized -->"


class StrictPolicyPlugin(PolicyPlugin):
    name = "strict_policy"
    version = "0.1.0"

    def initialize(self, config: Dict[str, Any]) -> None:
        self.forbidden_words = config.get("forbidden_words", ["redesign"])

    def evaluate(self, context: Dict) -> List[str]:
        violations = []
        # Check for forbidden words in intent
        intent_lock = context.get("intent_lock")
        if intent_lock:
            for word in self.forbidden_words:
                if word in intent_lock.intent_summary.lower():
                    violations.append(f"Forbidden word in intent: {word}")
        return violations


class DummyClassifierPlugin(ClassifierPlugin):
    name = "dummy_classifier"
    version = "0.1.0"

    def initialize(self, config: Dict[str, Any]) -> None:
        pass

    def score(self, text: str) -> Dict[str, float]:
        if "migration" in text.lower():
            return {"migration": 0.9}
        return {}


def test_register_plugin():
    system = PluginSystem()
    plugin = EchoOptimizerPlugin()
    system.register(plugin, {"prefix": "[X]"})
    plugins = system.list_plugins()
    assert any(p["name"] == "echo_optimizer" for p in plugins)


def test_pre_optimize_plugin_applied():
    system = PluginSystem()
    plugin = EchoOptimizerPlugin()
    system.register(plugin, {"prefix": "[PRE]"})
    result = system.apply_optimizer_plugins_pre("hello world", {})
    assert result.startswith("[PRE]")


def test_post_optimize_plugin_applied():
    system = PluginSystem()
    plugin = EchoOptimizerPlugin()
    system.register(plugin, {})
    result = system.apply_optimizer_plugins_post("hello world", {})
    assert "<!-- optimized -->" in result


def test_policy_plugin_violations():
    system = PluginSystem()
    plugin = StrictPolicyPlugin()
    system.register(plugin, {"forbidden_words": ["redesign"]})

    from llm_prompt_optimizer.core.intent_guard.guard import IntentGuard
    from llm_prompt_optimizer.models.prompt import RawPrompt
    guard = IntentGuard()
    lock = guard.extract_and_lock(RawPrompt(text="redesign the entire architecture"))
    violations = system.evaluate_policy_plugins({"intent_lock": lock})
    assert len(violations) > 0


def test_policy_plugin_clean():
    system = PluginSystem()
    plugin = StrictPolicyPlugin()
    system.register(plugin, {"forbidden_words": ["redesign"]})

    from llm_prompt_optimizer.core.intent_guard.guard import IntentGuard
    from llm_prompt_optimizer.models.prompt import RawPrompt
    guard = IntentGuard()
    lock = guard.extract_and_lock(RawPrompt(text="debug EMA mismatch"))
    violations = system.evaluate_policy_plugins({"intent_lock": lock})
    assert len(violations) == 0


def test_disable_plugin():
    system = PluginSystem()
    plugin = EchoOptimizerPlugin()
    system.register(plugin, {})
    system.registry.disable("echo_optimizer")
    result = system.apply_optimizer_plugins_pre("hello", {})
    assert result == "hello"  # plugin disabled, no prefix


def test_teardown_all():
    system = PluginSystem()
    system.register(EchoOptimizerPlugin(), {})
    system.teardown_all()  # should not raise


def test_multiple_plugins_chained():
    system = PluginSystem()

    class Plugin1(OptimizerPlugin):
        name = "p1"
        version = "0.1.0"
        def initialize(self, c): pass
        def pre_optimize(self, text, ctx): return "P1:" + text

    class Plugin2(OptimizerPlugin):
        name = "p2"
        version = "0.1.0"
        def initialize(self, c): pass
        def pre_optimize(self, text, ctx): return text + ":P2"

    system.register(Plugin1())
    system.register(Plugin2())
    result = system.apply_optimizer_plugins_pre("X", {})
    assert "P1:" in result
    assert ":P2" in result
