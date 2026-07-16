#!/usr/bin/env python3
import csv
import json
import math
import re
import sys
from pathlib import Path


def clean_status(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("TERMINATION_REASON_"):
        s = s[len("TERMINATION_REASON_") :]
    return s


def load_solver_json(path: Path):
    text = path.read_text(errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        replacements = {"inf": "Infinity", "+inf": "Infinity", "-inf": "-Infinity", "nan": "NaN"}
        normalized = re.sub(
            r"(?<=:)([+-]?inf|nan)(?=\s*[,}])",
            lambda match: replacements[match.group(1).lower()],
            text,
            flags=re.IGNORECASE,
        )
        return json.loads(normalized)


def parse_cupdlpx_txt(path: Path):
    text = path.read_text(errors="replace")

    def grab(label, default=""):
        m = re.search(rf"{re.escape(label)}\s*:\s*([^\n]+)", text)
        return m.group(1).strip() if m else default

    objective = grab("Primal Objective Value") or grab("Primal Objective", "nan")
    return {
        "status": clean_status(grab("Termination Reason", "ERROR")),
        "time_sec": grab("Runtime (sec)", "nan"),
        "iterations": grab("Iterations Count", "0"),
        "objective": objective,
        "rel_gap": grab("Relative Objective Gap", "nan"),
        "rel_primal": grab("Relative Primal Residual", "nan"),
        "rel_dual": grab("Relative Dual Residual", "nan"),
    }


def parse_cupdlpc_json(path: Path):
    data = load_solver_json(path)
    code = data.get("terminationCode", data.get("termination_code", "ERROR"))
    return {
        "status": clean_status(str(code)),
        "time_sec": str(data.get("dSolvingTime", data.get("solving_time", math.nan))),
        "iterations": str(data.get("nIter", data.get("iteration_count", 0))),
        "objective": str(data.get("dPrimalObj", data.get("primal_objective", math.nan))),
        "rel_gap": str(data.get("dRelDualityGap", data.get("relative_objective_gap", math.nan))),
        "rel_primal": str(data.get("dRelPrimalFeas", math.nan)),
        "rel_dual": str(data.get("dRelDualFeas", math.nan)),
    }


def parse_cupdlp_jl_json(path: Path):
    data = json.loads(path.read_text(errors="replace"))
    stats = data.get("solution_stats", {})
    convergence = stats.get("convergence_information", [])
    candidate = convergence[0] if isinstance(convergence, list) and convergence else {}
    return {
        "status": clean_status(str(data.get("termination_string", data.get("termination_reason", "ERROR")))),
        "time_sec": str(data.get("solve_time_sec", stats.get("cumulative_time_sec", math.nan))),
        "iterations": str(data.get("iteration_count", 0)),
        "objective": str(candidate.get("primal_objective", math.nan)),
        "rel_gap": str(candidate.get("relative_optimality_gap", math.nan)),
        "rel_primal": str(candidate.get("relative_l2_primal_residual", math.nan)),
        "rel_dual": str(candidate.get("relative_l2_dual_residual", math.nan)),
    }


def parse_hprlp_c_text(path: Path):
    text = path.read_text(errors="replace")

    def grab(label, default=""):
        m = re.search(rf"^{re.escape(label)}\s*:\s*([^\n]+)", text, re.MULTILINE)
        return m.group(1).strip() if m else default

    def strip_seconds(value: str) -> str:
        return value.replace("seconds", "").strip()

    status = grab("Status", "ERROR")
    residual = grab("Residual", "nan")
    return {
        "status": clean_status(status),
        "time_sec": strip_seconds(grab("Time", "nan")),
        "iterations": grab("Iterations", "0"),
        "objective": grab("Primal Objective", "nan"),
        "rel_gap": residual,
        "rel_primal": residual,
        "rel_dual": residual,
    }


def parse_hprlp_jl_csv(path: Path):
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        row = next(csv.DictReader(handle))
    residual = row.get("rel_primal", row.get("rel_residual", math.nan))
    return {
        "status": clean_status(row.get("status", "ERROR")),
        "time_sec": row.get("time_sec", "nan"),
        "iterations": row.get("iterations", "0"),
        "objective": row.get("objective", "nan"),
        "rel_gap": row.get("rel_gap", "nan"),
        "rel_primal": residual,
        "rel_dual": row.get("rel_dual", residual),
    }


def resolve_solver_output(kind: str, path: Path) -> Path:
    if path.exists() or kind != "cupdlpx_c":
        return path
    # cuPDLPx uses the text before the first dot as the output stem for names
    # such as thk.48. Each instance has its own output directory, so the
    # unique summary in that directory is unambiguous.
    candidates = sorted(path.parent.glob("*_summary.txt"))
    return candidates[0] if len(candidates) == 1 else path


def main():
    kind, solver, dataset, split, tolerance, time_limit, instance, path = sys.argv[1:9]
    p = resolve_solver_output(kind, Path(path))
    if not p.exists():
        print(f"{solver},{kind},{dataset},{split},{instance},{tolerance},{time_limit},ERROR,nan,0,nan,nan,nan,nan,missing output {path}")
        return
    try:
        if kind == "cupdlpx_c":
            row = parse_cupdlpx_txt(p)
        elif kind == "cupdlpc":
            row = parse_cupdlpc_json(p)
        elif kind == "cupdlp_jl":
            row = parse_cupdlp_jl_json(p)
        elif kind == "hprlp_c":
            row = parse_hprlp_c_text(p)
        elif kind == "hprlp_jl":
            row = parse_hprlp_jl_csv(p)
        else:
            raise ValueError(f"unknown kind {kind}")
        print(",".join([
            solver,
            kind,
            dataset,
            split,
            instance,
            tolerance,
            time_limit,
            row["status"],
            row["time_sec"],
            row["iterations"],
            row["objective"],
            row["rel_gap"],
            row["rel_primal"],
            row["rel_dual"],
            "",
        ]))
    except Exception as exc:
        msg = str(exc).replace(",", ";").replace("\n", " ")
        print(f"{solver},{kind},{dataset},{split},{instance},{tolerance},{time_limit},ERROR,nan,0,nan,nan,nan,nan,{msg}")


if __name__ == "__main__":
    main()
