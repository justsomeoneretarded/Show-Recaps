"""Real-model script validation, optionally reusing a prior audited test ledger.

This developer-only runner does not change production's research/build path.
A resumed checkpoint comes from the same repository's earlier validation artifact;
all episode IDs, source ownership and exact evidence are rechecked against today's
scoped research before any narrative can be written.
"""
from __future__ import annotations
from dataclasses import replace
import json
import os
from pathlib import Path
from movie_pipeline.core import PipelineError, save_json
from recap_pipeline.cli import load_spec
from recap_pipeline.models import validate_script_scope
from recap_pipeline.writing import RecapWriter, build_ledger, write_recap, _episodes, _valid_event


def verify_checkpoint(ledger, bundle):
    episodes = {e["id"]:e for e in _episodes(bundle)}
    sources = {s["id"]:s for s in bundle.get("sources",[]) if isinstance(s,dict) and s.get("id")}
    original = ledger.get("events")
    if not isinstance(original,list) or not original:
        raise PipelineError("The audited validation checkpoint has no events.")
    accepted = [_valid_event(event,episodes,sources) for event in original]
    if any(event is None for event in accepted) or len({event["id"] for event in accepted}) != len(accepted):
        raise PipelineError("The audited checkpoint no longer matches the scoped episode evidence.")
    for before,after in zip(original,accepted):
        if bool(before.get("ending_state")) != bool(after.get("ending_state")):
            raise PipelineError("The checkpoint's ending evidence no longer matches.")
    return {**ledger,"events":accepted}


def main():
    root=Path("work/shared")
    bundle=json.loads((root/"research.json").read_text())
    # Keep the live integration check concise. Production Auto defaults are unchanged.
    spec=replace(load_spec(root),season_minutes="3",series_minutes="3")
    save_json(root/"spec.json",spec.to_dict())
    writer=RecapWriter(root/"writing-cache")
    try:
        checkpoint=root/"story-ledger.json"
        if os.getenv("RECAP_RESUME_LEDGER")=="true" and checkpoint.is_file():
            ledger=verify_checkpoint(json.loads(checkpoint.read_text()),bundle)
            print("Reusing scoped, evidence-revalidated audit checkpoint:",len(ledger["events"]),"events.",flush=True)
        else:
            ledger=build_ledger(bundle,root/"writing-cache",writer)
        save_json(checkpoint,ledger)
        for kind in ("season","series"):
            script=write_recap(kind,spec,bundle,ledger,root/"writing-cache",writer)
            draft=root/(kind+"-draft.json")
            save_json(draft,script)
            validate_script_scope(script,spec,bundle)
            save_json(root/(kind+"-script.json"),script)
            draft.unlink(missing_ok=True)
            print("RECAP_SCRIPT",kind,json.dumps(script,ensure_ascii=False),flush=True)
    finally:
        writer.unload()


if __name__=="__main__":
    main()
