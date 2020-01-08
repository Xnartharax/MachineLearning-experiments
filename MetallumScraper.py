import metallum
import json
names = []

for i in range(131*5):
    search = metallum.band_search("", strict=False, page_start=i*200)
    names += [band.name for band in search]
    fp = open("./metalNames.json", "w+")
    json_string = json.dumps(names)
    fp.write(json_string)
    fp.close()
    print(len(names))
