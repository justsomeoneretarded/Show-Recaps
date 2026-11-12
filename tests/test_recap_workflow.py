"""The production workflow keeps scope and per-run controls consistent."""
from pathlib import Path
import yaml


def test_recap_workflow_two_outputs_share_one_frozen_prepare():
    data=yaml.load(Path(".github/workflows/generate-recaps.yml").read_text(),Loader=yaml.BaseLoader)
    inputs=data["on"]["workflow_dispatch"]["inputs"]
    assert len(inputs)<=10
    assert inputs["tts_model"]["default"]=="original"
    assert inputs["narrator"]["default"]=="Warguy"
    assert inputs["try_4k"]["default"]=="true"
    assert inputs["season_minutes"]["options"]==["Auto",*map(str,range(1,21))]
    assert inputs["voice_speed"]["default"]=="1.0" and "1.2" in inputs["voice_speed"]["options"]
    assert "GEMINI_API_KEY" in data["jobs"]["render"]["env"]
    assert inputs["series_minutes"]["options"]==inputs["season_minutes"]["options"]
    assert data["jobs"]["render"]["needs"]=="prepare"
    kinds=data["jobs"]["render"]["strategy"]["matrix"]["kind"]
    assert kinds.startswith("${{ fromJSON(inputs.upcoming_season == 2 &&")
    assert """'["season"]' || '["season", "series"]'""" in kinds
    assert data["jobs"]["render"]["strategy"]["fail-fast"]=="false"
    for job in data["jobs"].values():
        for step in job.get("steps",[]):
            assert "${{ inputs." not in step.get("run","")
    render=data["jobs"]["render"]["steps"]
    upload=next(s for s in render if s.get("name")=="Upload completed package")
    assert "if" not in upload
    assert upload["with"]["retention-days"]=="2"
    assert upload["with"]["path"]=="output/${{ matrix.kind }}/"
    shared=next(s for s in data["jobs"]["prepare"]["steps"] if s.get("uses","").startswith("actions/upload-artifact") and "if" not in s)
    assert shared["with"]["retention-days"]=="1"
    assert data["jobs"]["notify"]["needs"]==["prepare","render"]
    assert "if" not in data["jobs"]["notify"]


def test_recap_shared_artifact_does_not_include_cookies_or_models():
    data=yaml.load(Path(".github/workflows/generate-recaps.yml").read_text(),Loader=yaml.BaseLoader)
    step=next(s for s in data["jobs"]["prepare"]["steps"] if s.get("name")=="Package shared research and assets")
    assert step["run"]=="tar -czf recap-shared.tgz work/shared"
    assert "models" not in step["run"] and "cookies" not in step["run"]
    guards=data["jobs"]["prepare"]["steps"]
    assert any(s.get("if")=="always()" and 'youtube-cookies.txt' in s.get("run","") for s in guards)
