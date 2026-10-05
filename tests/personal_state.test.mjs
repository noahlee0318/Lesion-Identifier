import test from 'node:test';
import assert from 'node:assert/strict';
import {readiness, calibrationProgress} from '../cloudflare/public/personal-state.mjs';
const rows=Array.from({length:40},(_,i)=>({date:new Date(Date.UTC(2026,0,i+1)).toISOString().slice(0,10),pose:'frontal',kind:'session',reviewed:true,px_per_mm:10,labels:[{x:1,y:1,radius:1}]}));
test('daily sessions unlock only after three distinct calibrated dates in this profile',()=>{
  const calibration=rows.slice(0,3).map(r=>({...r,profile_id:'one',kind:'calibration'}));
  const gate=r=>calibrationProgress(r,'one','2026-01-03');
  assert.equal(gate([]).unlocked,false);
  assert.equal(gate(calibration.slice(0,2)).unlocked,false);
  assert.equal(gate([calibration[0],calibration[0],calibration[1]]).completed,2);
  assert.equal(gate(calibration).unlocked,true);
  for(const change of [{profile_id:'two'},{kind:'session'},{px_per_mm:null},{px_per_mm:0},{px_per_mm:Infinity},{date:'2026-01-04'},{date:'2026-02-30'}])
    assert.equal(gate([...calibration.slice(0,2),{...calibration[2],...change}]).unlocked,false);
  assert.equal(calibrationProgress(calibration,'two','2026-01-03').unlocked,false);
});
test('readiness requires labels and three separate calendar blocks',()=>{
  assert.equal(readiness([]).ready,false);
  assert.equal(readiness(rows.slice(0,34)).ready,false);
  assert.equal(readiness(rows.slice(0,35)).ready,true);
  assert.deepEqual(readiness(rows).counts,{train:6,val:14,test:14});
});
test('profiles supply their own rows; unreviewed, practice and other views cannot qualify',()=>{
  for(const change of [{reviewed:false},{kind:'calibration'},{pose:'left60'},{px_per_mm:null},{labels:[]}])
    assert.equal(readiness(rows.map(r=>({...r,...change}))).ready,false);
});
