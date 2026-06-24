# test genai without opening the gui
# loops through patients and prints the ollama report
# ollama must be running: ollama pull llama3
#
# python test_compare.py

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from db import init_db, list_patients, list_visits, visit_to_result
from genai_compare import GenAIError, generate_comparison_report, has_significant_change


def main():
    init_db()
    patients = list_patients()
    if not patients:
        print("No patients found. Run: python seed_data.py")
        return

    for p in patients:
        visits = list_visits(p.id)
        if len(visits) < 2:
            print(f"{p.name}: not enough visits ({len(visits)})")
            continue

        baseline = visit_to_result(visits[0])
        current = visit_to_result(visits[-1])
        changed = has_significant_change(baseline, current)
        print(f"\n{'='*60}")
        print(f"{p.name} — change detected: {'YES' if changed else 'NO'}")
        print(f"  Baseline: {baseline.visit_date} {baseline.class_label} {baseline.confidence*100:.0f}%")
        print(f"  Current:  {current.visit_date} {current.class_label} {current.confidence*100:.0f}%")

        try:
            # force=True so i always see a full report when i am testing prompts
            report = generate_comparison_report(p.name, baseline, current, force=True)
            print(f"\n{report}")
        except GenAIError as exc:
            print(f"\n[GenAI error] {exc}")


if __name__ == "__main__":
    main()
