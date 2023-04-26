from http.client import responses

import requests
import json


def submit(results, url="http://kamino.disi.unitn.it:3001/results/"):
    res = json.dumps(results)
    # print(res)
    response = requests.post(url, res)
    try:
        result = json.loads(response.text)
        print(f"accuracy is {result['results']}")
    except json.JSONDecodeError:
        print(f"ERROR: {response.text}")



mydata = dict()
mydata['groupname'] = "i cani da caccia"

res = dict()
res["624c48ba52a3e52f36b037747c8ee85f4fc6c9ab.jpg"] = ["f73fd7684d9eb8de5385d32159dd3b6e2f7630f8.jpg", "9b3c995139bf1718f35ebc0ffd0cda61dac37eba.jpg", "d2960d384062952931795d01c77a2b13671dd2ab.jpg", "4e6cab2667403ea0d7ee2429201668c52ae7390a.jpg", "097e497e1787269b4b67d48529be4d935c58740c.jpg"]
res["f77a7f91ccbd7aaa08bfdc50252e3b074a62b2c6.jpg"] = ["ff7ba3c0051b2ce673da5a156d8c7cca930cdfd2.jpg", "d2960d384062952931795d01c77a2b13671dd2ab.jpg", "d2960d384062952931795d01c77a2b13671dd2ab.jpg", "4e6cab2667403ea0d7ee2429201668c52ae7390a.jpg", "097e497e1787269b4b67d48529be4d935c58740c.jpg"]
res["7b50557f6d3f6eb61ec0fe68675e9979734aa418.jpg"] = ["f73fd7684d9eb8de5385d32159dd3b6e2f7630f8.jpg", "d2960d384062952931795d01c77a2b13671dd2ab.jpg", "d2960d384062952931795d01c77a2b13671dd2ab.jpg", "4e6cab2667403ea0d7ee2429201668c52ae7390a.jpg", "097e497e1787269b4b67d48529be4d935c58740c.jpg"]
# res["wrong"] = ["w1", "w2"]

mydata["images"] = res
submit(mydata)
