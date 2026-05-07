import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('real_data/Klenke14/json/Chapter1.json', encoding='utf-8') as f:
    orig = json.load(f)
with open('Klenke14_revised/json/Chapter1.json', encoding='utf-8') as f:
    rev = json.load(f)

orig_map = {u['label']: u for u in orig}
rev_map = {u['label']: u for u in rev}

print(f'Original: {len(orig)} units')
print(f'Revised:  {len(rev)} units')
print(f'Added:    {set(rev_map) - set(orig_map)}')
print(f'Removed:  {set(orig_map) - set(rev_map)}')

diffs = 0
for label in sorted(set(orig_map) & set(rev_map)):
    o = orig_map[label]
    r = rev_map[label]
    changes = []
    if o.get('env') != r.get('env'):
        changes.append('env: %s -> %s' % (o.get('env'), r.get('env')))
    if o.get('content','')[:50] != r.get('content','')[:50]:
        changes.append('content changed')
    if bool(o.get('proof','')) != bool(r.get('proof','')):
        changes.append('proof: %s -> %s' % ('has' if o.get('proof') else 'none', 'has' if r.get('proof') else 'none'))
    if o.get('number_components') != r.get('number_components'):
        changes.append('num: %s -> %s' % (o.get('number_components'), r.get('number_components')))
    if changes:
        diffs += 1
        if diffs <= 15:
            print('  %s: %s' % (label, ' | '.join(changes)))

print('Total units with differences: %d' % diffs)
