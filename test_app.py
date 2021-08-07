from fastapi.testclient import TestClient
from main import app

# test to check the correct functioning of the /ping route
def test_ping():
    with TestClient(app) as client:
        response = client.get("/ping")
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"ping": "pong"}


# test to check if Iris Virginica is classified correctly
def test_cred_scoring():
    # defining a sample payload for the testcase
    payload = {
  "p1": "A11",
  "p2": 6,
  "p3": "A34",
  "p4": "A43",
  "p5": 1169,
  "p6": "A65",
  "p7": "A75",
  "p8": 4,
  "p9": "A93",
  "p10": "A101",
  "p11": 4,
  "p12": "A121",
  "p13": 67,
  "p14": "A143",
  "p15": "A152",
  "p16": 2,
  "p17": "A173",
  "p18": 1,
  "p19": "A192",
  "p20": "A201"
}
    with TestClient(app) as client:
        response = client.post("/cred_scoring", json=payload)
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"loan": "Bad"}
       
