"""One researched request, two scripts, and independently renderable packages."""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
import sys

from movie_pipeline.core import PipelineError, save_json

from .models import RecapSpec, validate_script_scope

KINDS = ("season", "series")


def kinds_for(spec):
    """Before Season 2 both recaps would cover only Season 1: make one video, not two copies."""
    return ("season",) if spec.upcoming_season == 2 else KINDS


def _read(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PipelineError(f"Missing or invalid shared artifact: {Path(path).name}. Run prepare first.") from exc
    if not isinstance(value, dict):
        raise PipelineError(f"Shared artifact {Path(path).name} must contain an object.")
    return value


def load_spec(shared, *, create=False):
    """Later jobs always use the saved selection, even when their env differs."""
    shared = Path(shared)
    path = shared / "spec.json"
    if path.exists():
        return RecapSpec.from_dict(_read(path))
    if not create:
        raise PipelineError("Missing shared spec.json. Run prepare first.")
    spec = RecapSpec.from_env()
    save_json(path, spec.to_dict())
    return spec


def _asset_paths(value, shared, *, portable):
    """Persist paths relative to the shared artifact; reject escaped/symlink paths."""
    shared = Path(shared).resolve()
    if isinstance(value, list):
        return [_asset_paths(item, shared, portable=portable) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for field, item in value.items():
        if field == "path":
            if not isinstance(item, str) or not item.strip():
                raise PipelineError("Every asset path must contain text.")
            path = Path(item)
            if not portable and path.is_absolute():
                raise PipelineError("Shared assets must use portable relative paths.")
            resolved = path.resolve() if path.is_absolute() else (shared / path).resolve()
            try:
                relative = resolved.relative_to(shared)
            except ValueError as exc:
                raise PipelineError("Asset paths must remain inside the shared artifact directory.") from exc
            if not resolved.is_file():
                raise PipelineError("A shared asset file is missing. Download the complete shared artifact.")
            result[field] = relative.as_posix() if portable else str(resolved)
        else:
            result[field] = _asset_paths(item, shared, portable=portable)
    return result


def research_stage(shared):
    from .research import research_show

    shared = Path(shared)
    spec = load_spec(shared, create=True)
    bundle = research_show(spec, shared / "research-cache")
    save_json(shared / "research.json", bundle)
    return bundle


def write_stage(shared):
    from .gemini import shared_client
    from .writing import RecapWriter, apply_packaging, build_ledger, write_recap

    shared = Path(shared)
    spec = load_spec(shared)
    bundle = _read(shared / "research.json")
    cache = shared / "writing-cache"
    cache.mkdir(parents=True, exist_ok=True)
    writer = RecapWriter(cache)
    try:
        ledger = build_ledger(bundle, cache, writer)
        save_json(shared / "story-ledger.json", ledger)
        scripts = {}
        for kind in kinds_for(spec):
            script = write_recap(kind, spec, bundle, ledger, cache, writer)
            script = apply_packaging(script, shared_client())
            draft = shared / f"{kind}-draft.json"
            save_json(draft, script)
            validate_script_scope(script, spec, bundle)
            save_json(shared / f"{kind}-script.json", script)
            draft.unlink(missing_ok=True)
            scripts[kind] = script
        return scripts
    finally:
        # A failed unload must not mask a useful writing/scope failure. Rendering
        # also unloads the configured model before loading speech on its runner.
        try:
            writer.unload()
        except Exception:
            print("Writing model unload could not be confirmed.", flush=True)


def assets_stage(shared):
    from .assets import collect_assets

    shared = Path(shared)
    spec = load_spec(shared)
    bundle = _read(shared / "research.json")
    for kind in kinds_for(spec):
        validate_script_scope(_read(shared / f"{kind}-script.json"), spec, bundle)
    assets = collect_assets(bundle, spec, (shared / "asset-cache").resolve())
    portable = _asset_paths(assets, shared, portable=True)
    save_json(shared / "assets.json", portable)
    return portable


def _freeze_voice(spec):
    # Existing audio helpers read these process variables. Restore the saved
    # request before importing the renderer, so another job's env cannot swap it.
    os.environ["NARRATOR"] = spec.narrator
    os.environ["TTS_MODEL"] = spec.tts_model
    os.environ["TRY_4K"] = "true" if spec.try_4k else "false"


def render_stage(kind, shared, work, output, *, recovery=None):
    if kind not in KINDS:
        raise PipelineError("Choose one render kind: season or series.")
    shared = Path(shared)
    spec = load_spec(shared)
    if kind not in kinds_for(spec):
        raise PipelineError("Before Season 2 the series recap would repeat the Season 1 recap; "
                            "only the season package is produced.")
    bundle = _read(shared / "research.json")
    script = _read(shared / f"{kind}-script.json")
    validate_script_scope(script, spec, bundle)
    assets = _asset_paths(_read(shared / "assets.json"), shared, portable=False)
    _freeze_voice(spec)
    from .rendering import render_recap
    from .writing import RecapWriter

    try:
        RecapWriter(shared / "writing-cache").unload()
    except Exception:
        print("Writing model unload could not be confirmed before speech.", flush=True)
    return render_recap(script, spec, bundle, assets, Path(output) / kind,
                        Path(work) / kind, recovery=recovery)


def run_stage(stage, shared, work, output, *, kind=None, recovery=None):
    if stage == "prepare":
        research_stage(shared)
        scripts = write_stage(shared)
        assets_stage(shared)
        return scripts
    if stage == "research":
        return research_stage(shared)
    if stage == "write":
        return write_stage(shared)
    if stage == "assets":
        return assets_stage(shared)
    if stage == "render":
        return render_stage(kind, shared, work, output, recovery=recovery)
    raise PipelineError("Unknown recap stage.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare one show and render two scoped recaps.")
    parser.add_argument("--stage", choices=("prepare", "research", "write", "assets", "render"), required=True)
    parser.add_argument("--kind", choices=KINDS)
    parser.add_argument("--shared", type=Path, default=Path("work/shared"))
    parser.add_argument("--work", type=Path, default=Path("work/render"))
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--recovery", type=Path)
    args = parser.parse_args(argv)
    if args.stage == "render" and not args.kind:
        parser.error("--kind is required for render.")
    if args.stage != "render" and args.kind:
        parser.error("--kind applies only to render; prepare always writes both recaps.")
    try:
        run_stage(args.stage, args.shared, args.work, args.output,
                  kind=args.kind, recovery=args.recovery)
    except PipelineError as exc:
        message = re.sub(r"https?://\S+", "[source URL]", str(exc))
        for name in ("TELEGRAM_BOT_TOKEN", "TMDB_READ_TOKEN", "TMDB_API_KEY", "YOUTUBE_COOKIES",
                     "GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3", "GOOGLE_API_KEY"):
            secret = os.getenv(name, "")
            if secret:
                message = message.replace(secret, "[redacted]")
        print(f"Recap {args.stage} failed: {message}", file=sys.stderr, flush=True)
        return 1
    except Exception:
        # HTTP/model exceptions and their URLs can embed credentials. Do not
        # dump request headers, environment inputs or tracebacks into Actions.
        print(f"Recap {args.stage} failed. Check the shared artifacts and configured source/model services.",
              file=sys.stderr, flush=True)
        return 1
    print(f"Recap {args.stage} complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
