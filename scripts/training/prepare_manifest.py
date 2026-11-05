#!/usr/bin/env python3
import argparse,json
from collections import Counter
from pathlib import Path
from duplexjev.training.data import StreamingEventDataset

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--manifest",required=True); ap.add_argument("--out",default="manifest_report.json"); a=ap.parse_args(); report={}
 for split in ("train","valid","test"):
  try: ds=StreamingEventDataset(a.manifest,split)
  except ValueError: report[split]={"events":0}; continue
  c=Counter((r["query"],str(r["label"])) for r in ds.rows)
  report[split]={"events":len(ds),"recordings":len(set(r["recording_id"] for r in ds.rows)),"counts":{f"{k[0]}:{k[1]}":v for k,v in c.items()}}
 Path(a.out).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
