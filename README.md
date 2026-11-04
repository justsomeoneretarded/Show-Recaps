# Show Recaps

One workflow makes two source-grounded recap packages before the next season of a show:

- **Previous season:** the season immediately before the selected incoming season.
- **The story so far:** every earlier season, connected into a broader series recap.

For incoming Season 4, these cover Season 3 and Seasons 1–3. Neither includes Season 4 plot or later footage. Historical boundaries are supported too: an incoming-season selection describes the cutoff, not a claim about today's release schedule.

## Run it

Open **Actions → Generate Show Recaps → Run workflow**, using branch **main**.

Supply the full show title, original premiere year, and incoming season number. Add an optional creative brief. Choose Warguy (default), Kit Laser or Female; original Chatterbox is the default speech model. Each recap has its own Auto/minutes control and the shared Try 4K checkbox defaults on.

The workflow researches once, writes two scripts from a shared story ledger, collects prior-season assets, then renders the two packages independently. Runtime targets allow ±25% and remain guidance; useful narration is not padded or cut through words to meet a stopwatch.

After success, download:

- **show-recap-season**
- **show-recap-series**

Each contains final-video.mp4, subtitles.srt, description.txt with measured chapters, recommended-title.txt, three thumbnail JPEGs, source credits, and production/source-coverage reports. Upload subtitles separately in YouTube Studio and paste the description's timestamps to expose chapters.

## Research and accuracy

TVMaze resolves the show and episodes. Verified Wikipedia season tables/episode plot sections provide detailed story enrichment; TMDB can fill supported gaps when a token is configured. Short promotional overviews are identified as such. Sources are scoped by production, season and episode, and plot claims retain evidence links.

Recap writing is separate from the countdown writer: it explains actions, motivations and consequences, then the supported final state. It does not use recommendation ranks, provider pitches or compulsory review bridges. A source-support editing pass rejects invented events and citations. Missing episodes and source limitations are reported honestly.

**Required ending information is a quality safeguard.** A teaser summary is not enough to invent how a finale ended. If the pipeline cannot establish a supported ending or meaningful story for the selected scope, it stops with a research/writing diagnostic instead of delivering an intro-only or fabricated recap. Available summaries do not certify exhaustive plot coverage; no automatic system guarantees flawless factual interpretation.

Footage discovery rejects incoming/wrong-season material, reactions, concept videos and full episodes. It uses genuine clips where season identity can be established, then verified prior-episode stills. If usable imagery is unavailable, an explicitly reported neutral title plate provides an illustrative fallback. The current pipeline cannot obtain arbitrary full episodes from streaming services, and clip availability determines how closely visuals can illustrate every event.

## Voice and sound

Original Chatterbox uses the existing permitted reference recordings, one model and one selected reference throughout. It generates bounded complete thoughts, verifies spoken words, and measures pacing before rebuilding subtitle timing. Processing and lossless intermediate audio reuse the tested production foundation. CPU generation can take time.

Narration sits over a quiet, consistent licensed instrumental bed. Optional native scene audio is retained only where a scoped, coherent excerpt can be selected. Copyright/source credits accompany the output; edit effects do not guarantee protection from claims.

## Optional secrets

No paid inference API is required. Configure these under this repository's **Settings → Secrets and variables → Actions** when desired:

- **YOUTUBE_COOKIES:** optional downloader authentication. These are private and excluded from packages.
- **TMDB_READ_TOKEN:** optional supplementary metadata/episode lookup.
- **TELEGRAM_BOT_TOKEN** and **TELEGRAM_CHAT_ID:** optional completion reminder after both packages were uploaded. Channel pinning also needs the bot's appropriate channel rights.

Secrets in the original movie repository are not automatically available here. Do not paste secrets into workflow inputs or commit them.

## Storage and reliability

Completed packages expire after 48 hours. The shared intermediate asset artifact expires after one day. Temporary runner files disappear with their jobs; model and Python caches remain separate. Each matrix render has its own scratch directory and bounded disk checks. Failed jobs do not upload partial videos as completed packages.

Original Chatterbox, the script model and dependencies run on GitHub CPU runners. Account quotas, runner timeouts, download restrictions and source outages still apply. The source and render reports expose missing coverage and illustrative visual fallbacks.

## Development

    python -m pip install -e '.[test]'
    pytest -q tests/test_recap_*.py

Install FFmpeg for the media integration tests. Production installs the voice extra automatically; unit tests use mocked model calls, and FFmpeg fixture renders are tests rather than generated show videos.

The Recap Validation workflow checks scoped research against a real show, runs source/model/media regressions, and can generate real script samples. It does not publish a video to YouTube. The inherited foundation tests and utility workflows remain available; use **Generate Show Recaps** for this project's production.

This project is adapted from the owner's [Top10-Movies](https://github.com/justsomeoneretarded/Top10-Movies) foundation at commit c6f82a621b9de57a2c180a74cd9e37e17f2a6a92. Existing voice, font and music assets retain their source credits. The original movie project is separate.
