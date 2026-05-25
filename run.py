from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def safe_model_name(model: str) -> str:
    return model.replace("/", "_").replace(":", "_").replace(" ", "_")


def run_command(args: list[str]) -> None:
    print("$", " ".join(args))
    subprocess.run(args, cwd=ROOT, check=True)


def append_optional_flag(cmd: list[str], flag: str, value) -> None:
    if value is None:
        return
    cmd.extend([flag, str(value)])


def require_api_credentials() -> None:
    api_key = os.getenv("TARBENCH_API_KEY")
    base_url = os.getenv("TARBENCH_BASE_URL")
    if not api_key or not base_url:
        raise SystemExit(
            "Missing API settings.\n"
            "Set both TARBENCH_API_KEY and TARBENCH_BASE_URL before running online experiments."
        )


def prepare_artifacts() -> None:
    run_command([PYTHON, "experiments/build_datasets.py"])
    run_command([PYTHON, "experiments/icl/build_icl_library.py"])
    run_command([PYTHON, "experiments/rag/build_knowledge_base.py"])
    run_command([PYTHON, "experiments/token_compression/build_token_compression_eval_dataset.py"])


def icl_output_path(model: str, shot_count: int) -> Path:
    return ROOT / "experiments" / "icl" / "outputs" / f"icl_{safe_model_name(model)}_{shot_count}shot.json"


def rag_output_path(model: str, output_tag: str | None, without_rag: bool) -> Path:
    suffix = "_without_rag" if without_rag else ""
    tag = f"_{output_tag}" if output_tag else ""
    return ROOT / "experiments" / "rag" / "outputs" / f"rag_{safe_model_name(model)}{tag}{suffix}.json"


def token_output_path(model: str, output_tag: str | None) -> Path:
    tag = f"_{output_tag}" if output_tag else ""
    return (
        ROOT
        / "experiments"
        / "token_compression"
        / "outputs"
        / f"token_compression_{safe_model_name(model)}{tag}.json"
    )


def handle_prepare(_: argparse.Namespace) -> None:
    prepare_artifacts()


def handle_icl(args: argparse.Namespace) -> None:
    if not args.dry_run:
        require_api_credentials()
    cmd = [
        PYTHON,
        "experiments/icl/run_icl_benchmark.py",
        "--shot-count",
        str(args.shot_count),
    ]
    append_optional_flag(cmd, "--model", args.model)
    append_optional_flag(cmd, "--sample-count", args.sample_count)
    append_optional_flag(cmd, "--batch-size", args.batch_size)
    if args.resume:
        cmd.append("--resume")
    if args.dry_run:
        cmd.append("--dry-run")
    run_command(cmd)

    if args.dry_run or args.no_eval:
        return
    output_file = icl_output_path(args.model or "gpt-5.4", args.shot_count)
    run_command([PYTHON, "experiments/icl/evaluate_icl.py", str(output_file), "--write-enriched"])


def handle_rag(args: argparse.Namespace) -> None:
    if args.rebuild_kb:
        run_command([PYTHON, "experiments/rag/build_knowledge_base.py"])
    if not args.dry_run:
        require_api_credentials()
    cmd = [PYTHON, "experiments/rag/run_rag_benchmark.py"]
    append_optional_flag(cmd, "--model", args.model)
    append_optional_flag(cmd, "--sample-count", args.sample_count)
    append_optional_flag(cmd, "--benign-count", args.benign_count)
    append_optional_flag(cmd, "--abnormal-count", args.abnormal_count)
    append_optional_flag(cmd, "--output-tag", args.output_tag)
    if args.without_rag:
        cmd.append("--without-rag")
    if args.resume:
        cmd.append("--resume")
    if args.dry_run:
        cmd.append("--dry-run")
    run_command(cmd)

    if args.dry_run or args.no_eval:
        return
    output_file = rag_output_path(args.model or "gpt-5.4", args.output_tag, args.without_rag)
    run_command([PYTHON, "experiments/rag/evaluate_rag.py", str(output_file), "--write-enriched"])


def handle_token_compression(args: argparse.Namespace) -> None:
    if not args.dry_run:
        require_api_credentials()
    cmd = [PYTHON, "experiments/token_compression/run_token_compression_benchmark.py"]
    append_optional_flag(cmd, "--model", args.model)
    append_optional_flag(cmd, "--sample-count", args.sample_count)
    append_optional_flag(cmd, "--batch-size", args.batch_size)
    append_optional_flag(cmd, "--output-tag", args.output_tag)
    if args.resume:
        cmd.append("--resume")
    if args.dry_run:
        cmd.append("--dry-run")
    run_command(cmd)

    if args.dry_run or args.no_eval:
        return
    output_file = token_output_path(args.model or "gpt-5.4", args.output_tag)
    run_command(
        [
            PYTHON,
            "experiments/token_compression/evaluate_token_compression.py",
            str(output_file),
            "--write-enriched",
        ]
    )


def handle_full(args: argparse.Namespace) -> None:
    prepare_artifacts()
    icl_args = argparse.Namespace(
        model=args.model,
        shot_count=args.shot_count,
        sample_count=args.sample_count,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        resume=args.resume,
        no_eval=args.no_eval,
    )
    rag_args = argparse.Namespace(
        model=args.model,
        sample_count=args.sample_count,
        benign_count=None,
        abnormal_count=None,
        output_tag=args.output_tag,
        without_rag=False,
        rebuild_kb=False,
        dry_run=args.dry_run,
        resume=args.resume,
        no_eval=args.no_eval,
    )
    token_args = argparse.Namespace(
        model=args.model,
        sample_count=args.sample_count,
        batch_size=args.batch_size,
        output_tag=args.output_tag,
        dry_run=args.dry_run,
        resume=args.resume,
        no_eval=args.no_eval,
    )
    handle_icl(icl_args)
    handle_rag(rag_args)
    handle_token_compression(token_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="One-click runner for TARBench.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Rebuild datasets, ICL library, and RAG knowledge base.")
    prepare_parser.set_defaults(func=handle_prepare)

    icl_parser = subparsers.add_parser("icl", help="Run ICL benchmark and evaluation.")
    icl_parser.add_argument("--model", default="gpt-5.4")
    icl_parser.add_argument("--shot-count", type=int, default=5, choices=[0, 1, 3, 5])
    icl_parser.add_argument("--sample-count", type=int)
    icl_parser.add_argument("--batch-size", type=int)
    icl_parser.add_argument("--dry-run", action="store_true")
    icl_parser.add_argument("--resume", action="store_true")
    icl_parser.add_argument("--no-eval", action="store_true")
    icl_parser.set_defaults(func=handle_icl)

    rag_parser = subparsers.add_parser("rag", help="Run RAG benchmark and evaluation.")
    rag_parser.add_argument("--model", default="gpt-5.4")
    rag_parser.add_argument("--sample-count", type=int)
    rag_parser.add_argument("--benign-count", type=int)
    rag_parser.add_argument("--abnormal-count", type=int)
    rag_parser.add_argument("--output-tag")
    rag_parser.add_argument("--without-rag", action="store_true")
    rag_parser.add_argument("--rebuild-kb", action="store_true")
    rag_parser.add_argument("--dry-run", action="store_true")
    rag_parser.add_argument("--resume", action="store_true")
    rag_parser.add_argument("--no-eval", action="store_true")
    rag_parser.set_defaults(func=handle_rag)

    token_parser = subparsers.add_parser(
        "token-compression",
        help="Run token-compression benchmark and evaluation.",
    )
    token_parser.add_argument("--model", default="gpt-5.4")
    token_parser.add_argument("--sample-count", type=int)
    token_parser.add_argument("--batch-size", type=int)
    token_parser.add_argument("--output-tag")
    token_parser.add_argument("--dry-run", action="store_true")
    token_parser.add_argument("--resume", action="store_true")
    token_parser.add_argument("--no-eval", action="store_true")
    token_parser.set_defaults(func=handle_token_compression)

    full_parser = subparsers.add_parser("full", help="Prepare artifacts and run ICL, RAG, and token-compression.")
    full_parser.add_argument("--model", default="gpt-5.4")
    full_parser.add_argument("--shot-count", type=int, default=5, choices=[0, 1, 3, 5])
    full_parser.add_argument("--sample-count", type=int)
    full_parser.add_argument("--batch-size", type=int)
    full_parser.add_argument("--output-tag")
    full_parser.add_argument("--dry-run", action="store_true")
    full_parser.add_argument("--resume", action="store_true")
    full_parser.add_argument("--no-eval", action="store_true")
    full_parser.set_defaults(func=handle_full)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
