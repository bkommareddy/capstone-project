import sqlite3
import sys

import pandas as pd

DB_PATH = sys.argv[1] if len(sys.argv) > 1 else "zepto_books.db"

def run_and_print(cur, label: str, sql: str, params: tuple = ()):
    print(f"\n--- {label} ---")
    print(sql.strip())
    rows = cur.execute(sql, params).fetchall()
    cols = [d[0] for d in cur.description]
    print(cols)
    for r in rows[:15]:
        print(r)
    if len(rows) > 15:
        print(f"... ({len(rows)} rows total)")
    return rows

def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    run_and_print(
        cur,
        "Q1: SELECT/WHERE — in-stock books rated 4 or 5",
        """
        SELECT title, rating, price_gbp
        FROM books
        WHERE in_stock = 1 AND rating >= 4
        """,
    )

    run_and_print(
        cur,
        "Q2: ORDER BY/LIMIT — 10 most expensive books",
        """
        SELECT title, price_gbp
        FROM books
        ORDER BY price_gbp DESC
        LIMIT 10
        """,
    )

    run_and_print(
        cur,
        "Q3: DISTINCT — distinct rating values present",
        """
        SELECT DISTINCT rating
        FROM books
        ORDER BY rating
        """,
    )

    run_and_print(
        cur,
        "Q4: BETWEEN — books priced between £20 and £30",
        """
        SELECT title, price_gbp
        FROM books
        WHERE price_gbp BETWEEN 20 AND 30
        ORDER BY price_gbp
        """,
    )

    join_sql = """
        SELECT c.category_name, b.title, b.rating, b.price_gbp
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        ORDER BY c.category_name, b.rating DESC, b.price_gbp DESC
        LIMIT 10
    """
    run_and_print(cur, "Q5: JOIN — sample of books with their category name", join_sql)

    print("\n=== pd.read_sql demonstration ===")
    df_in_stock = pd.read_sql(
        "SELECT title, rating, price_gbp FROM books WHERE in_stock = 1 AND rating >= 4",
        conn,
    )
    print("\nQ1 via pd.read_sql:")
    print(df_in_stock.head())

    df_join_sql = pd.read_sql(join_sql, conn)
    print("\nQ5 (JOIN) via pd.read_sql:")
    print(df_join_sql)

    print("\n=== pd.merge equivalence check ===")
    books_df = pd.read_sql("SELECT * FROM books", conn)
    categories_df = pd.read_sql("SELECT * FROM categories", conn)

    merged = books_df.merge(categories_df, on="category_id", how="inner")
    df_join_merge = (
        merged[["category_name", "title", "rating", "price_gbp"]]
        .sort_values(["category_name", "rating", "price_gbp"], ascending=[True, False, False])
        .head(10)
        .reset_index(drop=True)
    )
    print("\nQ5 (JOIN) via pd.merge:")
    print(df_join_merge)

    are_equal = df_join_sql.reset_index(drop=True).equals(df_join_merge)
    print(f"\npd.read_sql JOIN result matches pd.merge JOIN result: {are_equal}")

    conn.close()

if __name__ == "__main__":
    main()
