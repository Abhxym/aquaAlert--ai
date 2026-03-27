import requests

api_key = "zdb1iehwyUDBmyH2IZGLbUyPdabU8fHE5N85SY6b"

def test_earth_api():
    print("Testing NASA Earth API (Landsat)...")
    url = f"https://api.nasa.gov/planetary/earth/assets?lon=77.2090&lat=28.6139&date=2021-01-01&dim=0.15&api_key={api_key}"
    response = requests.get(url)
    print("Earth API Status:", response.status_code)
    try:
        print("Data:", response.json())
    except Exception as e:
        print("Error parsing json:", e)

def test_power_api():
    print("\nTesting NASA POWER API...")
    url = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters=PRECTOTCORR&community=RE&longitude=77.2090&latitude=28.6139&start=20210101&end=20210131&format=JSON"
    response = requests.get(url)
    print("POWER API Status:", response.status_code)
    try:
        keys = list(response.json().keys())
        print("Keys:", keys)
    except Exception as e:
        print("Error parsing json:", e)

if __name__ == "__main__":
    test_earth_api()
    test_power_api()
