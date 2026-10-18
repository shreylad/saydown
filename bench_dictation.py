"""Synthetic English speech QA: offline local Whisper + formatting, isolated from user data.

Run with bundled Python + site-packages: bench_dictation.py --output /path/to/report-directory
Uses macOS Samantha TTS and a cached model; never records the microphone or calls a speech API.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import statistics
import subprocess
import tempfile
import time
from unittest.mock import patch
import wave

CASES = [
    ('ordinal_agenda', 'Thank you for reaching out. First our agenda is doing apple tree, second agenda is doing green tree, third agenda is cleaning the grass and fourth agenda is making a barn house.', 'numbered', 4),
    ('gerund_agenda', 'Here is our agenda for the day. Cooking at home, working on the app, freely running on the lawn and working on the computer.', 'bullets', 4),
    ('intent_agenda', 'This is the agenda of the day I am going to cook, I am going to drive my car, I am going to clean the grasses and I am going to make the pesticides to clean the bad insects from the garden.', 'bullets', 4),
    ('things_to_do', 'Here are the things I need to do: cook dinner, drive my car and clean the garden.', 'bullets', 3),
    ('what_to_do', "Here's what I need to do: buy apples, cook dinner and review the report.", 'bullets', 3),
    ('three_tasks', 'I have three things to do: cook dinner, drive my car and clean the garden.', 'bullets', 3),
    ('imperative_and', 'My to do list: cook dinner and drive my car and clean the garden.', 'bullets', 3),
    ('shopping', 'Shopping list: apples, milk and bread.', 'bullets', 3),
    ('noun_compound', 'Shopping list: eggs, salt and pepper.', 'bullets', 2),
    ('task_compound', 'My tasks: I will cook fish and chips, I will drive and I will clean.', 'bullets', 3),
    ('doctor', 'My tasks: call Doctor Lee, review the report and clean the garden.', 'bullets', 3),
    ('repeated_sentences', 'This is our agenda. I will cook. I will drive. I will clean. Thanks for your help.', 'bullets', 3),
    ('prose_activities', 'I enjoy cooking at home, working on the app, and running outside.', 'prose', 0),
    ('prose_agenda', 'We discussed the agenda yesterday, worked on the app and went home.', 'prose', 0),
    ('prose_description', 'The agenda is new, exciting, and useful.', 'prose', 0),
    ('prose_intent', 'I am going to cook, I am going to drive my car and I am going to clean the garden.', 'prose', 0),
    ('single_task', 'Here is our agenda. Working on the app.', 'prose', 0),
    ('ordinary_ordinals', 'This is my first time here and I am having second thoughts.', 'prose', 0),
]


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    snapshots = Path.home() / '.cache/huggingface/hub/models--mlx-community--whisper-large-v3-turbo/snapshots'
    model = next((p for p in sorted(snapshots.glob('*')) if (p / 'config.json').exists()), None)
    if model is None:
        raise SystemExit('A cached local Whisper model is required; this benchmark never downloads one.')
    import mlx.core as mx
    import mlx_whisper
    import numpy as np
    with tempfile.TemporaryDirectory() as isolated:
        spec = importlib.util.spec_from_file_location('speech_qa', Path(__file__).with_name('dictate.py'))
        app = importlib.util.module_from_spec(spec)
        with patch.object(Path, 'home', return_value=Path(isolated)):
            spec.loader.exec_module(app)
        app.SETTINGS.write_text(json.dumps(app.DEFAULTS | {'language': 'en'}))
        text_repo = app.TEXT_MODEL
        text_snapshots = Path.home() / '.cache/huggingface/hub' / ('models--' + text_repo.replace('/', '--')) / 'snapshots'
        text_model = next((p for p in sorted(text_snapshots.glob('*')) if (p / 'config.json').exists()), None)
        if text_model:
            app.paths[text_repo] = str(text_model)
        mlx_whisper.transcribe(np.zeros(16000, np.float32), path_or_hf_repo=str(model), language='en')
        results = []
        for name, spoken, expected_style, expected_count in CASES:
            aiff, wav = output / (name + '.aiff'), output / (name + '.wav')
            if not wav.exists():
                subprocess.run(['/usr/bin/say', '-v', 'Samantha', '-o', str(aiff), spoken], check=True)
                subprocess.run(['/usr/bin/afconvert', '-f', 'WAVE', '-d', 'LEI16@16000', '-c', '1', str(aiff), str(wav)], check=True)
                aiff.unlink()
            with wave.open(str(wav)) as file:
                audio = np.frombuffer(file.readframes(file.getnframes()), dtype='<i2').astype(np.float32) / 32768
            mx.synchronize()
            start = time.perf_counter()
            raw = mlx_whisper.transcribe(audio, path_or_hf_repo=str(model), language='en')['text'].strip()
            mx.synchronize()
            transcription_s = time.perf_counter() - start
            start = time.perf_counter()
            formatted = app.polish(raw)
            formatting_s = time.perf_counter() - start
            bullets = len(re.findall(r'^- ', formatted, re.M))
            numbered = len(re.findall(r'^\d+\. ', formatted, re.M))
            style = 'bullets' if bullets else 'numbered' if numbered else 'prose'
            count = bullets or numbered
            passed = style == expected_style and count == expected_count
            result = dict(name=name, spoken=spoken, raw=raw, formatted=formatted,
                          expected_style=expected_style, expected_count=expected_count,
                          style=style, count=count, passed=passed, audio_s=len(audio)/16000,
                          transcription_s=transcription_s, formatting_s=formatting_s)
            results.append(result)
            print(f'{name}: {"PASS" if passed else "FAIL"} {style}/{count} ({transcription_s:.3f}s speech)', flush=True)
        summary = {'synthetic': True, 'voice': 'Samantha', 'passed': sum(r['passed'] for r in results),
                   'total': len(results), 'median_transcription_s': statistics.median(r['transcription_s'] for r in results),
                   'median_formatting_ms': statistics.median(r['formatting_s'] * 1000 for r in results), 'results': results}
        (output / 'results.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps({k:v for k,v in summary.items() if k != 'results'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    run(parser.parse_args().output)
