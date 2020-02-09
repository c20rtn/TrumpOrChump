import pandas as pd
import ujson as json

tt_df = pd.read_json("Datasets/trumptweets.json")
records = map(json.loads, open('Datasets/realdonaldtrump.ndjson', encoding="utf8"))
rdt_df = pd.DataFrame.from_records(records)

print (tt_df)
print (rdt_df)
