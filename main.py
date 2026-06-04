#!/usr/bin/env python3
"""
main.py — CIA prototype demo runner.

Runs the full evaluation simulation and prints Table 3 of the paper:
  "Autonomous Identity Disclosure via Contextual Agents in Multi-Platform Systems"
  Prism '26 — Anubhav Gupta & Aditi Shailesh Joshi

Usage:
    python main.py [--interactions N] [--real-crypto] [--json]
                   [--assets-dir PATH] [--quiet]

Exit codes:
    0  Success
    1  Unhandled exception during simulation
    2  Required assets missing (ONNX model not downloaded)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CIA evaluation runner — reproduces Table 3 of the paper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--interactions", "-n",
        type=int, default=500, metavar="N",
        help="Interactions per platform per tier (default: 500, paper value).",
    )
    p.add_argument(
        "--real-crypto",
        action="store_true", default=False,
        help="Use real BBS+ instead of stub (slow — for final paper run only).",
    )
    p.add_argument(
        "--json",
        action="store_true", default=False,
        dest="dump_json",
        help="Dump full SimulationReport as JSON to stdout after the table.",
    )
    p.add_argument(
        "--assets-dir",
        type=Path, default=None, metavar="PATH",
        help="Override default cia/assets/ directory (useful for CI).",
    )
    p.add_argument(
        "--quiet", "-q",
        action="store_true", default=False,
        help="Skip progress output, print table only.",
    )
    return p


# ---------------------------------------------------------------------------
# Logging — routes simulator progress to stdout unless --quiet
# ---------------------------------------------------------------------------

def _configure_logging(quiet: bool) -> None:
    # Silence the root logger completely — CIA components log at WARNING which
    # would otherwise flood stderr with expected stub/conservative-mode notices.
    logging.getLogger().setLevel(logging.CRITICAL)

    if not quiet:
        # Route simulator progress lines (INFO) to stdout.
        sim_logger = logging.getLogger("cia.eval.simulator")
        sim_logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        sim_logger.addHandler(handler)
        sim_logger.propagate = False


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def _linkability_label(fraction: float) -> str:
    if fraction < 0.15:
        return "Low"
    if fraction < 0.40:
        return "Medium"
    return "High"


def _reid_label(reduction_pct: float) -> str:
    if reduction_pct > 50:
        return "Low"
    if reduction_pct > 20:
        return "Medium"
    return "High"


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _ms(v: float) -> str:
    return f"{v:.0f}ms"


def _print_table(report, args: argparse.Namespace) -> None:
    n     = report.n_interactions
    total = report.total_traces

    print()
    print("Contextual Identity Agent - Evaluation Results")
    print("=" * 51)
    print(
        f"Interactions per platform: {n}"
        f"  |  Total traces: {total:,}"
        f"  |  Platforms: 5"
    )
    print()

    # Table 3
    print("Table 3: Projected Evaluation Outcomes Across Baselines")
    print("-" * 58)

    b = report.baseline
    cia_link_label   = _linkability_label(
        sum(report.linkability_by_tier.values()) / len(report.linkability_by_tier)
        if report.linkability_by_tier else 1.0
    )
    cia_reid_label   = _reid_label(report.reid_risk_reduction_pct)

    _COL = "{:<20} {:<9} {:<15} {:<14} {}"
    print(_COL.format("System", "DMR", "Linkability", "Re-ID Risk", "User Actions"))
    print(_COL.format("Unmediated",  "0.0%",   "High",   "High",           "Low"))
    print(_COL.format("OAuth / OIDC","0.0%",   "High",   "High",           "Low"))
    print(_COL.format("SSI Wallet",  f"{b['ssi_dmr']*100:.1f}%", "Medium", "Medium", "High"))
    print(_COL.format(
        "CIA (this work)",
        _pct(report.overall_dmr),
        cia_link_label,
        cia_reid_label,
        "Low",
    ))

    cia_dmr_pct = report.overall_dmr * 100
    target_note = "(target >= 68%)" if cia_dmr_pct < 68 else "(target met)"
    print(f"  CIA DMR: {cia_dmr_pct:.1f}% {target_note}")

    # ── Per-platform breakdown ──────────────────────────────────────────────
    print()
    print("Per-platform breakdown:")
    platform_display = {
        "financial_portal":    "FINANCIAL_PORTAL    ",
        "professional_network": "PROFESSIONAL_NETWORK",
        "social_platform":     "SOCIAL_PLATFORM     ",
        "healthcare_portal":   "HEALTHCARE_PORTAL   ",
        "gaming_platform":     "GAMING_PLATFORM     ",
    }
    for name, label in platform_display.items():
        dmr   = report.dmr_by_platform.get(name, 0.0)
        cons  = report.conservative_mode_by_platform.get(name, 0.0)
        p95   = report.latency_p95_by_platform.get(name, 0.0)
        print(
            f"  {label}  DMR: {dmr*100:5.1f}%"
            f"   Conservative mode: {cons*100:5.1f}%"
            f"   Latency p95: {p95:.0f}ms"
        )

    # ── Scenario tier breakdown ─────────────────────────────────────────────
    print()
    print("Scenario tier breakdown:")
    tier_labels = {"BENIGN": "Benign   ", "PROBING": "Probing  ", "COLLUDING": "Colluding"}
    for tier_key, tier_label in tier_labels.items():
        dmr      = report.dmr_by_tier.get(tier_key, 0.0)
        accuracy = report.classification_accuracy_by_tier.get(tier_key, 0.0)
        if tier_key == "COLLUDING":
            link = report.linkability_by_tier.get(tier_key, 0.0)
            print(f"  {tier_label}  DMR: {dmr*100:5.1f}%   Linkability: {link*100:.1f}%")
        else:
            print(f"  {tier_label}  DMR: {dmr*100:5.1f}%   Accuracy: {accuracy*100:.1f}%")

    # ── Summary stats ───────────────────────────────────────────────────────
    print()
    reid_pct  = report.reid_risk_reduction_pct
    reid_note = "(target: 61%)"
    print(f"Re-identification risk vs OAuth baseline: {reid_pct:.1f}% reduction  {reid_note}")
    print(f"Negotiation success rate: {report.negotiation_success_rate*100:.1f}%")

    ls = report.latency_stats
    print(
        f"Mean decision latency: {ls['mean']:.0f}ms"
        f"  |  p95: {ls['p95']:.0f}ms"
        f"  |  p99: {ls['p99']:.0f}ms"
    )
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = _build_parser()
    args   = parser.parse_args()
    _configure_logging(args.quiet)

    if not args.quiet:
        print(
            f"Running CIA simulation: {args.interactions} interactions x 5 platforms x 3 tiers"
            f"{' [real BBS+]' if args.real_crypto else ' [stub BBS+]'}",
            flush=True,
        )

    from cia.eval.simulator import run_simulation

    assets_dir: Path | None = args.assets_dir
    if assets_dir is None:
        assets_dir = Path(__file__).parent / "cia" / "assets"

    try:
        report = run_simulation(
            n_interactions=args.interactions,
            use_real_crypto=args.real_crypto,
            assets_dir=assets_dir,
        )
    except Exception as exc:
        # Check for missing-assets error specifically.
        if "ModelNotAvailableError" in type(exc).__name__ or (
            "assets" in str(exc).lower() and "not found" in str(exc).lower()
        ):
            print(
                "\nERROR: CC model assets not found.\n"
                "Download them by running:\n"
                "    python -c \"from cia.cc import ContextClassifier; "
                "from pathlib import Path; "
                "ContextClassifier(Path('cia/assets'), download_model=True)\"\n"
                "or pass --assets-dir pointing to a directory containing "
                "cc_model.onnx and cc_tokenizer.json.",
                file=sys.stderr,
            )
            return 2
        print(f"\nSimulation failed: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1

    _print_table(report, args)

    if args.dump_json:
        print(json.dumps(report.to_dict(), indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
