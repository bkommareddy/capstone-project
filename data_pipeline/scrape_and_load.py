import csv
import sqlite3
import time
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com/"
HOME_URL = BASE_URL + "index.html"

GBP_TO_INR = 105.50

MIN_BOOKS = 60
MIN_CATEGORIES = 3

RATING_WORD_TO_INT = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5,
}

REQUEST_DELAY_SECONDS = 0.3

def get_soup(url: str) -> BeautifulSoup:
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    time.sleep(REQUEST_DELAY_SECONDS)
    return BeautifulSoup(resp.text, "html.parser")

def discover_categories(home_url: str = HOME_URL) -> list[tuple[str, str]]:

    soup = get_soup(home_url)
    nav = soup.select_one("div.side_categories ul.nav-list ul")
    categories = []
    for a in nav.select("li > a"):
        name = a.get_text(strip=True)
        url = urljoin(home_url, a["href"])
        categories.append((name, url))
    return categories

def parse_listing_page(soup: BeautifulSoup, category_name: str) -> list[dict]:
    rows = []
    for article in soup.select("article.product_pod"):
        title = article.h3.a["title"].strip()

        price_text = article.select_one("p.price_color").get_text(strip=True)

        rating_classes = article.select_one("p.star-rating")["class"]

        rating_word = next(c for c in rating_classes if c != "star-rating")

        availability_text = article.select_one("p.instock.availability").get_text(strip=True)

        rows.append(
            {
                "title": title,
                "price_text": price_text,
                "rating_word": rating_word,
                "availability_text": availability_text,
                "category": category_name,
            }
        )
    return rows

def scrape_category(name: str, url: str) -> list[dict]:
    rows = []
    next_url = url
    while next_url:
        soup = get_soup(next_url)
        rows.extend(parse_listing_page(soup, name))
        next_link = soup.select_one("li.next > a")
        next_url = urljoin(next_url, next_link["href"]) if next_link else None
    return rows

def scrape_books() -> list[dict]:

    categories = discover_categories()
    all_rows: list[dict] = []
    used_categories = 0

    for name, url in categories:
        cat_rows = scrape_category(name, url)
        all_rows.extend(cat_rows)
        used_categories += 1
        if len(all_rows) >= MIN_BOOKS and used_categories >= MIN_CATEGORIES:
            break

    return all_rows

def clean_rows(raw_rows: list[dict]) -> list[dict]:

    cleaned = []
    dropped = 0

    for row in raw_rows:
        try:
            price_gbp = float(row["price_text"].replace("£", "").replace("Â", "").strip())
            rating = RATING_WORD_TO_INT[row["rating_word"]]
            in_stock = "in stock" in row["availability_text"].lower()

            cleaned.append(
                {
                    "title": row["title"],
                    "price_gbp": round(price_gbp, 2),
                    "price_inr": round(price_gbp * GBP_TO_INR, 2),
                    "rating": rating,
                    "in_stock": in_stock,
                    "category": row["category"],
                }
            )
        except (ValueError, KeyError):
            dropped += 1

    print(f"Cleaning complete: {len(cleaned)} rows kept, {dropped} rows dropped (unparseable).")
    return cleaned

def build_database(clean_data: list[dict], db_path: str = "zepto_books.db") -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.executescript(
        """
        DROP TABLE IF EXISTS books;
        DROP TABLE IF EXISTS categories;

        CREATE TABLE categories (
            category_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT UNIQUE NOT NULL
        );

        CREATE TABLE books (
            book_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            title       TEXT NOT NULL,
            price_gbp   REAL NOT NULL,
            price_inr   REAL NOT NULL,
            rating      INTEGER NOT NULL,
            in_stock    INTEGER NOT NULL,
            category_id INTEGER NOT NULL REFERENCES categories(category_id)
        );
        """
    )

    category_names = sorted({row["category"] for row in clean_data})
    cur.executemany(
        "INSERT INTO categories (category_name) VALUES (?)",
        [(name,) for name in category_names],
    )
    conn.commit()

    cat_id_lookup = dict(cur.execute("SELECT category_name, category_id FROM categories").fetchall())

    cur.executemany(
        """
        INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (
                row["title"],
                row["price_gbp"],
                row["price_inr"],
                row["rating"],
                int(row["in_stock"]),
                cat_id_lookup[row["category"]],
            )
            for row in clean_data
        ],
    )
    conn.commit()
    conn.close()
    print(f"Loaded {len(clean_data)} books across {len(category_names)} categories into {db_path}")

def write_csv(rows: list[dict], path: str) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def main() -> None:
    print("Scraping books.toscrape.com ...")
    raw_rows = scrape_books()
    write_csv(raw_rows, "books_raw.csv")
    print(f"Scraped {len(raw_rows)} raw rows -> books_raw.csv")

    clean_data = clean_rows(raw_rows)
    write_csv(clean_data, "books_clean.csv")
    print(f"Wrote {len(clean_data)} cleaned rows -> books_clean.csv")

    build_database(clean_data)

if __name__ == "__main__":
    main()
