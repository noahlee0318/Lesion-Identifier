"""Synthetic-only tests. No face photos, downloads, or claims of acne accuracy."""
import base64
import copy
import hashlib
import io
import json
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from src import personal_ml as ml


def fixture():
    records=[]
    for i in range(40):
        pixels=np.full((32,32,3),30+i,dtype=np.uint8)
        spots=[]
        for j in range(i % 3):
            x,y=7+j*17,8+(i%5)*3
            pixels[y-2:y+3,x-2:x+3]=[230,40,40]
            spots.append({'x':x,'y':y,'radius':2.5})
        output=io.BytesIO();Image.fromarray(pixels).save(output,format='PNG');raw=output.getvalue()
        records.append({'id':str(i),'profile_id':'synthetic-person','date':(date(2026,1,1)+timedelta(days=i)).isoformat(),
                        'kind':'session','pose':'frontal','width':32,'height':32,'px_per_mm':2,
                        'reviewed':True,'labels':spots,
                        'sha256':hashlib.sha256(raw).hexdigest(),'bytes':base64.b64encode(raw).decode()})
    return {'schema':1,'profile_id':'synthetic-person','images':records}


class PersonalMLTests(unittest.TestCase):
    def test_split_and_exclusion(self):
        data=fixture();groups=ml.split_days(ml.eligible(data,'frontal'))
        self.assertEqual({k:len(v) for k,v in groups.items()},{'train':6,'val':14,'test':14,'gap':6})
        self.assertEqual(groups['train'][-1]['date'],'2026-01-06')
        self.assertEqual(groups['val'][0]['date'],'2026-01-10')
        with self.assertRaises(ValueError):ml.split_days(data['images'][:34])
        data['images'][0]['kind']='calibration';data['images'][1]['reviewed']=False
        self.assertEqual(len(ml.eligible(data,'frontal')),38)

    def test_mixed_profiles_duplicate_bytes_and_bad_coordinates_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'data.json'
            for mode in ('profile','duplicate','coordinate'):
                data=fixture()
                if mode=='profile':data['images'][0]['profile_id']='another-person'
                if mode=='duplicate':
                    data['images'][1]['bytes']=data['images'][0]['bytes'];data['images'][1]['sha256']=data['images'][0]['sha256']
                if mode=='coordinate':data['images'][1]['labels'][0]['x']=99
                path.write_text(json.dumps(data))
                with self.subTest(mode=mode),self.assertRaises(ValueError):ml.load_dataset(path)

    def test_maximum_matching_and_undefined_metrics(self):
        self.assertEqual(ml.counts([[0,0],[2,0]],[[1,0],[-1,0]],1,2),(2,0,0))
        result=ml.metrics([(0,0,0)])
        self.assertIsNone(result['precision']);self.assertIsNone(result['recall'])

    def test_real_train_evaluate_predict_and_frozen_provenance(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp,patch.object(ml,'SIZE',32):
            path=Path(temp)/'data.json';path.write_text(json.dumps(fixture()))
            data=ml.load_dataset(path);run=Path(temp)/'run'
            card=ml.train(data,run,epochs=2,device='cpu')
            self.assertTrue((run/'weights.pt').exists())
            self.assertEqual(card['status'],'trained_not_evaluated')
            self.assertLess(card['losses'][-1],card['losses'][0])
            with self.assertRaises(FileExistsError):ml.train(data,run,epochs=1,device='cpu')
            changed=copy.deepcopy(data);changed['images'][-2]['labels']=[]
            with self.assertRaises(ValueError):ml.evaluate(changed,run)
            report=ml.evaluate(data,run)
            self.assertEqual(report['test_images'],14)
            self.assertEqual(report['status'],'insufficient_evidence')
            self.assertEqual(len(report['per_image']),14)
            with self.assertRaises(FileExistsError):ml.evaluate(data,run)
            self.assertTrue(ml.predict(data,run,'39')['experimental'])
            changed['profile_id']='other'
            with self.assertRaises(ValueError):ml.predict(changed,run,'39')
            (run/'weights.pt').write_bytes(b'changed')
            with self.assertRaises(ValueError):ml.load_model(run)


if __name__=='__main__':unittest.main()
