"""
Example: Writing a Custom Plugin

Shows how to build enterprise governance and custom classifier plugins.
"""

from typing import Dict, Any, List
from llm_prompt_optimizer import Optimizer, OptimizerConfig
from llm_prompt_optimizer.plugins.system import PolicyPlugin, ClassifierPlugin, PluginSystem


class EnterpriseSecurityPolicy(PolicyPlugin):
    """
    Enterprise policy: block prompts that mention production database credentials.
    """
    name = "enterprise_security_policy"
    version = "1.0.0"
    plugin_type = "policy"

    SENSITIVE_PATTERNS = ["prod_db", "production_password", "secret_key", "api_secret"]

    def initialize(self, config: Dict[str, Any]) -> None:
        self.extra_patterns = config.get("extra_patterns", [])
        self.all_patterns = self.SENSITIVE_PATTERNS + self.extra_patterns

    def evaluate(self, context: Dict[str, Any]) -> List[str]:
        violations = []
        intent_lock = context.get("intent_lock")
        if intent_lock:
            text = intent_lock.intent_summary.lower()
            for pattern in self.all_patterns:
                if pattern in text:
                    violations.append(
                        f"Enterprise policy violation: sensitive pattern '{pattern}' detected in intent."
                    )
        return violations


class FinancialDomainClassifier(ClassifierPlugin):
    """
    Custom classifier that boosts financial domain categories.
    """
    name = "financial_domain_classifier"
    version = "1.0.0"
    plugin_type = "classifier"

    FINANCIAL_SIGNALS = [
        "portfolio", "trade", "ema", "signal", "backtest", "alpha",
        "strategy", "pnl", "drawdown", "sharpe", "volatility",
    ]

    def initialize(self, config: Dict[str, Any]) -> None:
        self.boost = config.get("boost", 0.3)

    def score(self, text: str) -> Dict[str, float]:
        text_lower = text.lower()
        hit_count = sum(1 for s in self.FINANCIAL_SIGNALS if s in text_lower)
        if hit_count > 0:
            return {"ml_ai": min(1.0, hit_count * self.boost)}
        return {}


def demo():
    """Demo of custom plugin integration."""
    # Build plugin system with enterprise plugins
    plugins = PluginSystem()
    plugins.register(
        EnterpriseSecurityPolicy(),
        config={"extra_patterns": ["internal_api_key"]},
    )
    plugins.register(
        FinancialDomainClassifier(),
        config={"boost": 0.25},
    )

    cfg = OptimizerConfig()
    cfg.policy.enable_audit_log = False
    optimizer = Optimizer(config=cfg, plugin_system=plugins)

    # Safe prompt
    result = optimizer.optimize(
        "Debug condition mismatch in the provided /path/repository/file No code changes."
    )
    print(f"Safe prompt optimized: success={result.success}")
    print(f"Policy violations: {result.policy_violations}")
    print(f"Plugins loaded: {optimizer.list_plugins()}")


if __name__ == "__main__":
    demo()
