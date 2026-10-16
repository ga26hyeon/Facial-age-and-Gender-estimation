"""Feedback evaluation by exact model version; age metric is distance to bin."""
import json
from collections import defaultdict
from pathlib import Path

def evaluate(records):
    groups = defaultdict(list)
    for r in records:
        if r.get('feedback'):
            groups[r['prediction']['model_version']].append(r)
    result = {}
    for version, rows in groups.items():
        distance, correct, gender = [], [], []
        for r in rows:
            p, f = r['prediction'], r['feedback']
            lo, hi = p['age_range']
            distance.append(max(lo-f['age'], f['age']-hi, 0))
            correct.append(lo <= f['age'] <= hi)
            if f['gender'] in ('male','female'):
                gender.append(p['gender'] == f['gender'])
        result[version] = {'feedback_count': len(rows), 'age_bin_hit_rate': sum(correct)/len(rows),
                           'mean_distance_to_age_bin': sum(distance)/len(rows),
                           'gender_evaluated_count': len(gender),
                           'gender_accuracy': sum(gender)/len(gender) if gender else None,
                           'retained_image_count': sum(bool(r['image_path']) for r in rows)}
    return result

if __name__ == '__main__':
    import sys
    path = Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).parent/'data'/'records.json'
    print(json.dumps(evaluate(json.loads(path.read_text(encoding='utf-8'))['records']), indent=2))
