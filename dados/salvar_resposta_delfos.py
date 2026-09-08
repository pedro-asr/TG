import requests as req
import json

headers = {
    "API-Key": "UfKVCjs9h9QvsMhsSTNg2kmPDc5qycmP",
    "Content-Type": "application/json"
}

data = {"device_type_id": 2}

url = 'https://api.delfos.im/solar-data/data-studio/taxonomy/signal'
r = req.post(url, data=json.dumps(data), headers=headers)

resposta = r.json()

with open("resposta_delfos.txt", "w", encoding="utf-8") as f:
    json.dump(resposta, f, ensure_ascii=False, indent=2)

print("Arquivo salvo: resposta_delfos.txt")
print(json.dumps(resposta, ensure_ascii=False, indent=2))
