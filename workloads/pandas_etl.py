"""Data-pipeline style pandas script: joins, group-bys, sorting and rolling windows.
Expected to make almost no math-library calls."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
rows = 8_000_000

orders = pd.DataFrame({
    "customer": rng.integers(0, 200_000, rows),
    "product": rng.integers(0, 5_000, rows),
    "quantity": rng.integers(1, 10, rows),
    "price": rng.gamma(2.0, 20.0, rows),
    "day": rng.integers(0, 365, rows),
})
customers = pd.DataFrame({"customer": np.arange(200_000), "region": rng.integers(0, 50, 200_000)})

orders["revenue"] = orders["quantity"] * orders["price"]
joined = orders.merge(customers, on="customer")
by_region = joined.groupby(["region", "day"], as_index=False)["revenue"].sum()
by_region["rolling"] = by_region.groupby("region")["revenue"].transform(lambda s: s.rolling(7, min_periods=1).mean())
top_products = joined.groupby("product")["revenue"].agg(["sum", "mean", "count"]).sort_values("sum", ascending=False)
ranked = joined.sort_values(["customer", "revenue"], ascending=[True, False]).drop_duplicates("customer")

print(f"{len(joined):,} joined rows, {len(by_region):,} region-days, best product {top_products.index[0]}, "
      f"{len(ranked):,} customers")
