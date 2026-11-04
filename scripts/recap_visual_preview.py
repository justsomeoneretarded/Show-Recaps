"""Inspect actual scoped asset discovery and thumbnail composition without producing a show video."""
from __future__ import annotations
import base64
import json
from pathlib import Path
from PIL import Image
from recap_pipeline.cli import load_spec, _asset_paths
from recap_pipeline.thumbnails import build_thumbnails


def main():
    root=Path("work/shared")
    spec=load_spec(root)
    bundle=json.loads((root/"research.json").read_text())
    registry=_asset_paths(json.loads((root/'assets.json').read_text()),root,portable=False)
    season=spec.upcoming_season-1
    script={"kind":"season","show_title":bundle["show"]["title"],"upcoming_season":spec.upcoming_season,
            "seasons":[season],"title":f"{bundle['show']['title']}: Season {season} Recap",
            "thumbnail_concepts":[{"headline":f"SEASON {season} RECAP","concept":"A clear authentic character-focused recap image."}]}
    output=Path("recap-visual-validation")
    output.mkdir(exist_ok=True)
    images=build_thumbnails(script,bundle,registry,output)
    assert len(images)==3
    sheet=Image.new("RGB",(960,540*3),(8,11,18))
    for index,path in enumerate(images):
        with Image.open(path) as image:
            assert image.size==(1280,720)
            sheet.paste(image.convert("RGB").resize((960,540)),(0,index*540))
    contact=output/"contact.jpg"
    sheet.save(contact,quality=80)
    report={"show":bundle["show"]["title"],"seasons":sorted(registry["seasons"]),
            "source_count":len(registry.get("sources",[])),"warnings":registry.get("warnings",[])}
    (output/"asset-report.json").write_text(json.dumps(report,indent=2))
    print("RECAP_ASSET_REPORT",json.dumps(report),flush=True)
    print("RECAP_IMAGE_BEGIN",flush=True)
    print(base64.b64encode(contact.read_bytes()).decode(),flush=True)
    print("RECAP_IMAGE_END",flush=True)


if __name__=="__main__":
    main()
