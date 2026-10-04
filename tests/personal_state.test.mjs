import test from 'node:test';
import assert from 'node:assert/strict';
import {readiness} from '../cloudflare/public/personal-state.mjs';
const rows=Array.from({length:40},(_,i)=>({date:new Date(Date.UTC(2026,0,i+1)).toISOString().slice(0,10),pose:'frontal',kind:'session',reviewed:true,px_per_mm:10,labels:[{x:1,y:1,radius:1}]}));
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
