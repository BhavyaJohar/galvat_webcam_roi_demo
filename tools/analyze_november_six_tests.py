"""Reproduce the six returned device comparisons; raw CSVs remain unchanged."""
import argparse
import json
from pathlib import Path
from analyze_headset_tracking_acceptance import analyze


def summarize(report):
    sessions = [s for s in report['sessions'] if not s.get('excluded_from_postwarm_comparison')]
    duration = sum(s['postwarm_span_seconds'] for s in sessions)
    result = {'seconds': duration, 'sessions': len(sessions), 'rows': report['valid_rows']}
    if not duration:
        return result
    result['fresh_results_s'] = sum(s['detector_completions_per_second'] * s['postwarm_span_seconds'] for s in sessions) / duration
    for field in ['processed_camera_fps', 'callback_fps', 'detector_submission_frame_percent']:
        if field in sessions[0]:
            result[field] = sum(s[field] * s['postwarm_span_seconds'] for s in sessions) / duration
    result['metrics'] = {}
    for key in sessions[0]['metrics']:
        total = sum(s['postwarm_rows'] for s in sessions)
        result['metrics'][key] = {'mean': sum(s['metrics'][key]['mean'] * s['postwarm_rows'] for s in sessions) / total}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=Path('domains/galvat/trial_9/latest_test_results'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    payload = {'source_commit': '1cd44868eaef49647fef45360e21499c92ad6a0c',
               'warmup_note': 'User reports 60 seconds warmup BEFORE starting detection. Primary analysis keeps all logged within-session intervals; extra 60-second exclusion is a sensitivity analysis, not the main result.',
               'method': 'Within-session counter deltas divided by elapsed deltas; combined rates weighted by session duration. No gaps across sessions in denominator. Per-file SHA256 and per-session nearest-rank p95 retained.',
               'files': [], 'summary': {}}
    for p in sorted(args.folder.glob('*.csv')):
        full = analyze(p, 0)
        sensitivity = analyze(p, 60)
        key = p.stem.split('_')[-1] + ('_timing' if p.name.startswith('timing') else '_tracking')
        payload['files'].append({'primary': full, 'extra_60s_excluded': sensitivity})
        payload['summary'][key] = {'primary': summarize(full), 'extra_60s_excluded': summarize(sensitivity)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(payload['summary'], indent=2))


if __name__ == '__main__':
    main()
