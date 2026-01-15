import requests
import json
import time
import re
from bs4 import BeautifulSoup

# config

BASE_URL = "https://universe.leagueoflegends.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Academic RAG Scraper)"
}

# raw lists, extracted manually (sad for me)

CHAMPIONS_RAW = [
    "aatrox","ahri","akali","akshan","alistar","ambessa","amumu","anivia","annie",
    "aphelios","ashe","aurelionsol","aurora","azir","bard","belveth","blitzcrank",
    "brand","braum","briar","caitlyn","camille","cassiopeia","chogath","corki",
    "darius","diana","drmundo","draven","ekko","elise","evelynn","ezreal",
    "fiddlesticks","fiora","fizz","galio","gangplank","garen","gnar","gragas",
    "graves","gwen","hecarim","heimerdinger","hwei","illaoi","irelia","ivern",
    "janna","jarvaniv","jax","jayce","jhin","jinx","kaisa","kalista",
    "karma","karthus","kassadin","katarina","kayle","kayn","kennen","khazix",
    "kindred","kled","kogmaw","ksante","leblanc","lee sin","leona","lillia","lissandra",
    "lucian","lulu","lux","malphite","malzahar","maokai","masteryi","milio",
    "missfortune","mordekaiser","morgana","naafiri","nami","nasus","nautilus",
    "neeko","nidalee","nilah","nocturne","nunu","olaf","orianna","ornn",
    "pantheon","poppy","pyke","qiyana","quinn","rakan","rammus","reksai","rell",
    "renataglasc","renekton","rengar","riven","rumble","ryze","samira","sejuani",
    "senna","seraphine","sett","shaco","shen","shyvana","singed","sion","sivir",
    "skarner","smolder","sona","soraka","swain","sylas","syndra","tahmkench",
    "taliyah","talon","taric","teemo","thresh","tristana","trundle","tryndamere",
    "twistedfate","twitch","udyr","urgot","varus","vayne","veigar","velkoz", "vex",
    "vi","viego","viktor","vladimir","volibear","warwick","monkeyking",
    "xayah","xerath","xinzhao","yasuo","yone","yorick","yunara","yuumi","zaahen",
    "zac","zed","zeri","ziggs","zilean","zoe","zyra"
]

REGIONS_RAW = [
    "mount-targon", "void", "noxus", "demacia", "shurima", "bandle-city", 
    "bilgewater", "ionia", "ixtal", "piltover", "shadow-isles", "freljord", "zaun"   
]


# utils

def fetch(url):
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    r.encoding = "utf-8"
    return r.text


def clean(text):
    return " ".join(text.split())



# lore extraction

def scrape_champion_lore(champion_raw):
    slug = champion_raw
    url = f"{BASE_URL}/en_US/story/champion/{slug}"
    soup = BeautifulSoup(fetch(url), "html.parser")

    meta = soup.find("meta", attrs={"name": "description"})
    if not meta:
        print(soup.prettify()) # DEBUGGING LINE
        return None

    text = clean(meta.get("content", ""))

    return {
        "type": "champion",
        "name": champion_raw,
        "slug": slug,
        "url": url,
        "text": text
    }


def scrape_region_lore(region_raw):
    slug = region_raw
    url = f"{BASE_URL}/en_US/region/{slug}"
    soup = BeautifulSoup(fetch(url), "html.parser")

    meta = soup.find("meta", attrs={"name": "description"})
    if not meta:
        return None

    text = clean(meta.get("content", ""))

    return {
        "type": "region",
        "name": region_raw,
        "slug": slug,
        "url": url,
        "text": text
    }



# main scraping pipeline

def main():
    dataset_champs = []
    dataset_regions = []
    print("Scraping champion lore...")
    for champ in CHAMPIONS_RAW:
        try:
            entry = scrape_champion_lore(champ)
            if entry and entry["text"]:
                dataset_champs.append(entry)
                print(f"{champ} lore found")
            else:
                print(f"{champ} (no lore found)")
            time.sleep(0.5)
        except Exception as e:
            print(f"{champ} failed:", e)

    print("\nScraping region lore...")
    for region in REGIONS_RAW:
        try:
            entry = scrape_region_lore(region)
            if entry and entry["text"]:
                dataset_regions.append(entry)
                print(f"{region}")
            else:
                print(f"{region} (no lore found)")
            time.sleep(0.5)
        except Exception as e:
            print(f"{region} failed:", e)

    with open("lol_universe_champion_lore.json", "w", encoding="utf-8") as f:
        json.dump(dataset_champs, f, ensure_ascii=False, indent=2)
    
    with open("lol_universe_region_lore.json", "w", encoding="utf-8") as f:
        json.dump(dataset_regions, f, ensure_ascii=False, indent=2)

    print("\nDone!")
    print(f"=== Saved {len(dataset_champs)} entries to lol_universe_region_lore.json ===")
    print(f"=== Saved {len(dataset_regions)} entries to lol_universe_region_lore.json ===")

if __name__ == "__main__":
    main()
