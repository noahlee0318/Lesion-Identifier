export function calibrationProgress(images, profileId, today) {
  const dates = [...new Set(images.filter(r => {
    if (!profileId || r.profile_id !== profileId || r.kind !== 'calibration' ||
        !Number.isFinite(r.px_per_mm) || r.px_per_mm <= 0 ||
        typeof r.date !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(r.date) || r.date > today) return false;
    const timestamp = Date.parse(r.date + 'T00:00:00Z');
    return Number.isFinite(timestamp) && new Date(timestamp).toISOString().slice(0, 10) === r.date;
  }).map(r => r.date))].sort();
  return {dates, completed: dates.length, required: 3, complete: dates.length >= 3};
}

export function readiness(images, pose = 'frontal') {
  const eligible = images.filter(r => r.kind === 'session' && r.pose === pose && r.reviewed && Number.isFinite(r.px_per_mm) && r.px_per_mm > 0);
  const days = [...new Set(eligible.map(r => r.date))].sort();
  if (!images.length) return {stage:'Add your first photo', eligible:0, days:0, ready:false};
  if (!eligible.length) return {stage:'Calibrate scale and review your labels', eligible:0, days:0, ready:false};
  const end = Date.parse(days.at(-1) + 'T00:00:00Z') / 86400000;
  const groups = {train:[], val:[], test:[]};
  for (const r of eligible) {
    const d = Date.parse(r.date + 'T00:00:00Z') / 86400000;
    const key = d <= end-34 ? 'train' : d >= end-30 && d <= end-17 ? 'val' : d >= end-13 ? 'test' : null;
    if (key) groups[key].push(r);
  }
  const ready = Object.values(groups).every(g => g.length) && ['train','val'].every(k => groups[k].some(r=>r.labels.length));
  return {stage:ready ? 'Ready for an experimental local training run' : 'Collect more reviewed days before training', eligible:eligible.length, days:days.length, ready,
    counts:Object.fromEntries(Object.entries(groups).map(([k,v])=>[k,v.length]))};
}
